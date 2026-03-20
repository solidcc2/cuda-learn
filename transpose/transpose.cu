#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

#define ROWS 1024
#define COLS 2048
#define BLOCK_X 32
#define BLOCK_Y 16

void generate_input(float* input, int rows, int cols) {
    int len = rows * cols;
    for (int i = 0; i < len; i++) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void transpose_cpu_ref(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

void transpose_cpu(const float* input, float* output, int rows, int cols) {
    for(int i=0; i<rows; i+=BLOCK_X) {
        for(int j=0; j<cols; j+=BLOCK_Y) {
            for(int x=i; x<i+BLOCK_X && x < rows; x++) {
                for(int y=j; y<j+BLOCK_Y && y < cols; y++) {
                    output[y * rows + x] = input[x * cols + y];
                }
            }
        }
    }
}

bool check_result(const float* expected, const float* actual, int rows, int cols) {
    int len = rows * cols;
    for (int i = 0; i < len; i++) {
        if (std::fabs(expected[i] - actual[i]) > 1e-5f) {
            std::cout << "Mismatch at " << i
                      << ", expected: " << expected[i]
                      << ", actual: " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

__global__ void _cu_transpose_gpu_v1(const float* input, float* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

void transpose_gpu_v1(const float* input, float* output, int rows, int cols) {
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, sizeof(float) * ROWS * COLS);
    cudaMalloc(&d_output, sizeof(float) * ROWS * COLS);
    cudaMemcpy(d_input, input, sizeof(float) * ROWS * COLS, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float) * ROWS * COLS);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((COLS + BLOCK_X - 1) / BLOCK_X, (ROWS + BLOCK_Y - 1) / BLOCK_Y);
    _cu_transpose_gpu_v1<<<grid, block>>>(d_input, d_output, ROWS, COLS);

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, sizeof(float) * ROWS * COLS, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

// shared memory tile
__global__ void _cu_transpose_gpu_v2(const float* input, float* output, int rows, int cols) {
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    extern __shared__ float tile[];
    if (x < cols && y < rows) {
        tile[threadIdx.y * blockDim.x + threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();
    int o_dim_x = blockDim.y;
    int o_dim_y = blockDim.x;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int o_tid_x = tid % o_dim_x;
    int o_tid_y = tid / o_dim_x;
    int o_x = blockIdx.y * blockDim.y + o_tid_x;
    int o_y = blockIdx.x * blockDim.x + o_tid_y;
    if (o_x < rows && o_y < cols) {
        output[o_y * rows + o_x] = tile[o_tid_x * o_dim_y + o_tid_y];
    }
}

// shared memory tile
void transpose_gpu_v2(const float* input, float* output, int rows, int cols) {
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, sizeof(float) * ROWS * COLS);
    cudaMalloc(&d_output, sizeof(float) * ROWS * COLS);
    cudaMemcpy(d_input, input, sizeof(float) * ROWS * COLS, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float) * ROWS * COLS);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((COLS + BLOCK_X - 1) / BLOCK_X, (ROWS + BLOCK_Y - 1) / BLOCK_Y);
    _cu_transpose_gpu_v2<<<grid, block, sizeof(float) * BLOCK_X * BLOCK_Y>>>(d_input, d_output, ROWS, COLS);

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, sizeof(float) * ROWS * COLS, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

// bank conflict
__global__ void _cu_transpose_gpu_v3(const float* input, float* output, int rows, int cols) {
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    extern __shared__ float tile[];
    if (x < cols && y < rows) {
        tile[threadIdx.y * (blockDim.x+1) + threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();
    int o_dim_x = blockDim.y;
    int o_dim_y = blockDim.x;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int o_tid_x = tid % o_dim_x;
    int o_tid_y = tid / o_dim_x;
    int o_x = blockIdx.y * blockDim.y + o_tid_x;
    int o_y = blockIdx.x * blockDim.x + o_tid_y;
    if (o_x < rows && o_y < cols) {
        // int pos = o_tid_x * o_dim_y + o_tid_y;
        // int tile_x = pos % blockDim.x;
        // int tile_y = pos / blockDim.x;
        // output[o_y * rows + o_x] = tile[tile_y * (blockDim.x+1) + tile_x];
        output[o_y * rows + o_x] = tile[o_tid_x * (o_dim_y + 1) + o_tid_y];
    }
}

// bank conflict
void transpose_gpu_v3(const float* input, float* output, int rows, int cols) {
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, sizeof(float) * ROWS * COLS);
    cudaMalloc(&d_output, sizeof(float) * ROWS * COLS);
    cudaMemcpy(d_input, input, sizeof(float) * ROWS * COLS, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float) * ROWS * COLS);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((COLS + BLOCK_X - 1) / BLOCK_X, (ROWS + BLOCK_Y - 1) / BLOCK_Y);
    _cu_transpose_gpu_v3<<<grid, block, sizeof(float) * (BLOCK_X+1) * BLOCK_Y>>>(d_input, d_output, ROWS, COLS);

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, sizeof(float) * ROWS * COLS, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    float* input = new float[ROWS * COLS];
    float* output_cpu_ref = new float[ROWS * COLS];
    float* output_cpu = new float[ROWS * COLS]();
    float* output_gpu = new float[ROWS * COLS]();

    generate_input(input, ROWS, COLS);
    transpose_cpu_ref(input, output_cpu_ref, ROWS, COLS);

    transpose_cpu(input, output_cpu, ROWS, COLS);
    if (check_result(output_cpu_ref, output_cpu, COLS, ROWS)) {
        std::cout << "CPU results match!" << std::endl;
    } else {
        std::cout << "CPU results do not match!" << std::endl;
    }

    memset(output_gpu, 0, sizeof(float) *ROWS *COLS);
    transpose_gpu_v1(input, output_gpu, ROWS, COLS);
    std::cout << "transpose_gpu_v1: " 
        << (check_result(output_cpu_ref, output_gpu, COLS, ROWS) ? "right" : "wrong")
        << std::endl;

    memset(output_gpu, 0, sizeof(float) *ROWS *COLS);
    transpose_gpu_v2(input, output_gpu, ROWS, COLS);
    std::cout << "transpose_gpu_v2: " 
        << (check_result(output_cpu_ref, output_gpu, COLS, ROWS) ? "right" : "wrong")
        << std::endl;
    
    memset(output_gpu, 0, sizeof(float) *ROWS *COLS);
    transpose_gpu_v3(input, output_gpu, ROWS, COLS);
    std::cout << "transpose_gpu_v3: " 
        << (check_result(output_cpu_ref, output_gpu, COLS, ROWS) ? "right" : "wrong")
        << std::endl;

    delete[] input;
    delete[] output_cpu_ref;
    delete[] output_cpu;
    delete[] output_gpu;
    return 0;
}
