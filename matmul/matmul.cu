#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <chrono>

void generate_matrix(float* A, int rows, int cols) {
    for(int i=0; i<cols*rows; i++) {
        A[i] = (float)rand();
    }
}

bool nearly_equal(float a, float b, float rtol = 1e-5f, float atol = 1e-8f) {
    return fabsf(a - b) <= atol + rtol * fmaxf(fabsf(a), fabsf(b));
}

namespace transpose {
    void cpu(float* A, float* out, int rows, int cols) {
        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                out[j * rows + i] = A[i * cols + j];
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double use = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "transpose::cpu use: " << use << "ms" << std::endl;
    }

    __global__ void _cu(float* A, float* out, int rows, int cols) {
        extern __shared__ float tile[];
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        if (x < cols && y < rows) {
            tile[tid] = A[y * cols + x];
        }
        __syncthreads();
        int o_dim_x = blockDim.y;
        int o_dim_y = blockDim.x;
        int o_tid_x = tid % o_dim_x;
        int o_tid_y = tid / o_dim_x;
        int o_x = blockIdx.y * blockDim.y + o_tid_x;
        int o_y = blockIdx.x * blockDim.x + o_tid_y;
        if (o_x < rows && o_y < cols) {
            out[o_y * rows + o_x] = tile[o_tid_x * o_dim_y + o_tid_y];
        }
    }

    void gpu(float* A, float* out, int rows, int cols) {
        float* d_input, *d_output;
        cudaMalloc(&d_input, sizeof(float) * rows * cols);
        cudaMalloc(&d_output, sizeof(float) * rows * cols);
        cudaMemcpy(d_input, A, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
        dim3 block(32, 16, 1);
        dim3 grid((cols+block.x-1)/block.x, (rows+block.y-1)/block.y, 1);
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
        _cu<<<grid, block, sizeof(float) * block.x * block.y>>> (d_input, d_output, rows, cols);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float use;
        cudaEventElapsedTime(&use, start, end);
        std::cout << "transpose::gpu use: " << use << "ms" << std::endl;
        cudaMemcpy(out, d_output, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
        cudaFree(d_input);
        cudaFree(d_output);
    }

    void check() {
        int rows = 1024, cols = 2048;
        float* A = new float[rows * cols];
        float* out_cpu = new float[rows * cols];
        float* out_gpu = new float[rows * cols];
        generate_matrix(A, rows, cols);
        cpu(A, out_cpu, rows, cols);
        gpu(A, out_gpu, rows, cols);
        for(int i=0; i<cols; i++){
            for(int j=0; j<rows; j++){
                if (out_cpu[i*rows+j] != out_gpu[i*rows+j] || out_cpu[i*rows+j] != A[j*cols+i]) {
                    std::cout << "transpose wrong" << std::endl;
                    goto clean;
                }
            }
        }
        std::cout << "transpose right" << std::endl;
        
    clean:
        delete[] A;
        delete[] out_cpu;
        delete[] out_gpu;
    }
};

namespace matmul {
    void cpu_v1(float* A, float* B, float* C, int A_rows, int inner, int B_cols) {
        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0; i<A_rows; i++) {
            for(int j=0; j<B_cols; j++) {
                C[i * B_cols + j] = 0.0f;
                for(int k=0; k<inner; k++) {
                    C[i * B_cols + j] += A[i * inner + k] * B[k * B_cols + j];
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double use = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "matmul::cpu_v1 use: " << use << "ms" << std::endl;
    }

    void cpu_v2_trans(float* A, float* BT, float* C, int A_rows, int BT_rows, int cols) {
        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0; i<A_rows; i++) {
            for(int j=0; j<BT_rows; j++) {
                C[i*BT_rows + j] = 0.0f;
                for(int k=0; k<cols; k++) {
                    C[i*BT_rows + j] += A[i*cols + k] * BT[j*cols+k];
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double use = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "matmul::cpu_v1 use: " << use << "ms" << std::endl;
    }
    void cpu_v2(float* A, float* B, float* C, int A_rows, int inner, int B_cols) {
        float* BT = new float[B_cols * inner];
        transpose::cpu(B, BT, inner, B_cols);
        cpu_v2_trans(A, BT, C, A_rows, B_cols, inner);
        delete[] BT;
    }

    __global__ void _cu_withtrans(float* A, float* BT, float* C, int A_rows, int BT_rows, int cols) {
        extern __shared__ float buf[];
        int step = min(blockDim.x, blockDim.y);
        float* tile_A = buf;    // step * blockDim.y
        float* tile_BT = tile_A + blockDim.y * step;    // step * blockDim.x
        int col_id = blockDim.x * blockIdx.x + threadIdx.x;
        int row_id = blockDim.y * blockIdx.y + threadIdx.y;
        float ret = 0.0f;
        for(int i=0; i<(cols + step - 1)/step; i++){
            if (threadIdx.x < step && i*step + threadIdx.x < cols && row_id < A_rows) {
                tile_A[threadIdx.y * step + threadIdx.x] = A[row_id * cols + i*step + threadIdx.x];
            } else if (threadIdx.x < step) {
                tile_A[threadIdx.y * step + threadIdx.x] = 0.0f;
            }
            if (threadIdx.y < step && i*step + threadIdx.y < cols && col_id < BT_rows) {
                tile_BT[threadIdx.x * step + threadIdx.y] = BT[col_id * cols + i*step + threadIdx.y];
            } else if (threadIdx.y < step) {
                tile_BT[threadIdx.x * step + threadIdx.y] = 0.0f;
            }
            __syncthreads();
            if (row_id < A_rows && col_id < BT_rows) {
                for(int k=i*step; k<cols && k<i*step+step; k++) {
                    ret += tile_A[threadIdx.y * step + k-i*step] * tile_BT[threadIdx.x * step + k-i*step];
                }
            }
            __syncthreads();
        }
        if (row_id < A_rows && col_id < BT_rows) {
            C[row_id * BT_rows + col_id] = ret;
        }
    }

    void gpu_withtrans(float* A, float* BT, float* C, int A_rows, int BT_rows, int cols) {
        float* d_A, *d_BT, *d_C;
        cudaMalloc(&d_A, sizeof(float)*A_rows*cols);
        cudaMalloc(&d_BT, sizeof(float)*BT_rows*cols);
        cudaMalloc(&d_C, sizeof(float)*A_rows*BT_rows);
        cudaMemcpy(d_A, A, sizeof(float) *A_rows * cols, cudaMemcpyHostToDevice);
        cudaMemcpy(d_BT, BT, sizeof(float) *BT_rows * cols, cudaMemcpyHostToDevice);
        dim3 block(32, 16, 1);
        dim3 grid((BT_rows+block.x-1)/block.x, (A_rows+block.y-1)/block.y, 1);
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
        _cu_withtrans<<<grid, block, 
            sizeof(float) * min(block.x, block.y) * (block.x+block.y)>>>(d_A, d_BT, d_C, A_rows, BT_rows, cols);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float use;
        cudaEventElapsedTime(&use, start, end);
        std::cout << "gpu_withtrans use: " << use << "ms" << std::endl;
        cudaMemcpy(C, d_C, sizeof(float) * A_rows * BT_rows, cudaMemcpyDeviceToHost);
        cudaFree(d_A);
        cudaFree(d_BT);
        cudaFree(d_C);
    }

    void gpu(float* A, float* B, float* C, int A_rows, int inner, int B_cols) {
        float* BT = new float[B_cols * inner];
        transpose::gpu(B, BT, inner, B_cols);
        gpu_withtrans(A, BT, C, A_rows, B_cols, inner);
        delete[] BT;
    }

    bool check_equal(float* A, float* B, int rows, int cols) {
        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                if (!nearly_equal(A[i * cols + j], B[i *cols + j])) {
                    return false;
                }
            }
        }
        return true;
    }
};


int main() {
    transpose::check();
    int A_rows = 1024, A_cols =2048 ;
    int B_rows = A_cols, B_cols = 512;
    float* A = new float[A_rows * A_cols];
    float* B = new float[B_rows * B_cols];
    float* C_base = new float[A_rows * B_cols];
    float* C_cpu_v2 = new float[A_rows * B_cols];
    generate_matrix(A, A_rows, A_cols);
    generate_matrix(B, B_rows, B_cols);

    matmul::cpu_v1(A, B, C_base, A_rows, A_cols, B_cols);
    matmul::cpu_v2(A, B, C_cpu_v2, A_rows, A_cols, B_cols);
    std::cout << "cpu_v2: " 
        << (matmul::check_equal(C_base, C_cpu_v2, A_rows, B_cols) ? "right" : "wrong") 
        << std::endl;
    
    float* C_gpu = new float[A_rows * B_cols];
    matmul::gpu(A, B, C_gpu, A_rows, A_cols, B_cols);
    std::cout << "matmul::gpu: " 
        << (matmul::check_equal(C_base, C_gpu, A_rows, B_cols) ? "right" : "wrong") 
        << std::endl;

}
