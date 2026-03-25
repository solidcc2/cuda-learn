#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cstdlib>

#define SAMPLE_SIZE (1 << 24)

void generate_input(float* input, int n){
    for(int i=0; i<n; i++) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

float sum_cpu(float* input, int n) {
    float s = 0.0;
    for(int i=0; i < n; i++) {
        s += input[i];
    }
    return s;
}

__global__ void _cu_sum_gpu(float* d_buf1, float* d_buf2, int n) {
    extern __shared__ float local_buf[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.0f;
    if (tid*2 < n) s = d_buf1[tid*2];
    if (tid*2+1 < n) s += d_buf1[tid*2+1];
    local_buf[threadIdx.x] = s;
    __syncthreads();
    for (int bound = blockDim.x / 2; bound >= warpSize; bound /= 2) {
        if (threadIdx.x < bound) {
            local_buf[threadIdx.x] = local_buf[threadIdx.x] + local_buf[threadIdx.x + bound];
        }
        __syncthreads();
    }
    float var = local_buf[threadIdx.x];
    if (threadIdx.x < warpSize) {
        var += __shfl_down_sync(0xFFFFFFFF, var, 16);
        var += __shfl_down_sync(0xFFFFFFFF, var, 8);
        var += __shfl_down_sync(0xFFFFFFFF, var, 4);
        var += __shfl_down_sync(0xFFFFFFFF, var, 2);
        var += __shfl_down_sync(0xFFFFFFFF, var, 1);
    }
    if (threadIdx.x == 0) {
        d_buf2[blockIdx.x] = var;
    }
}

float sum_gpu(float* input, int n, int block_size) {
    float* d_buf1, *d_buf2;
    cudaMalloc(&d_buf1, sizeof(float) * n);
    cudaMalloc(&d_buf2, sizeof(float) * n);
    cudaMemcpy(d_buf1, input, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemset(d_buf2, 0, sizeof(float) * n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int dat_len = n;
    for(; dat_len >=block_size*2; dat_len = (dat_len + block_size * 2 - 1)/(block_size * 2)) {
        _cu_sum_gpu<<<(dat_len + block_size * 2 - 1) / (block_size * 2), block_size, block_size * sizeof(float)>>>(d_buf1, d_buf2, dat_len);
        float* tmp = d_buf1;
        d_buf1 = d_buf2;
        d_buf2 = tmp;
    }
    _cu_sum_gpu<<<(dat_len + block_size * 2 - 1) / (block_size * 2), block_size, block_size * sizeof(float)>>>(d_buf1, d_buf2, dat_len);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapse; 
    cudaEventElapsedTime(&elapse, start, stop);
    std::cout << "use: " << elapse << std::endl;
    float ret;
    cudaMemcpy(&ret, &d_buf2[0], sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_buf1);
    cudaFree(d_buf2);
    return ret;
}

bool check_result(float target, float gpu_result) {
    float abs_diff =  fabs(target - gpu_result);
    std::cout << "CPU result: " << target << ", GPU result: " << gpu_result << ", abs diff: " << abs_diff << std::endl;
    return abs_diff / fabs(target) < 1e-4;
}

int main() {
    float *input = (float*)malloc(SAMPLE_SIZE * sizeof(float));
    generate_input(input, SAMPLE_SIZE);
    float target = sum_cpu(input, SAMPLE_SIZE);
    int block_size;
    float gpu_result;
    block_size = 128;
    gpu_result = sum_gpu(input, SAMPLE_SIZE, block_size);
    std::cout << block_size << ": " << (check_result(target, gpu_result) ? "equal": "wrong") << std::endl;
    block_size = 256;
    gpu_result = sum_gpu(input, SAMPLE_SIZE, block_size);
    std::cout << block_size << ": " << (check_result(target, gpu_result) ? "equal": "wrong") << std::endl;
    block_size = 512;
    gpu_result = sum_gpu(input, SAMPLE_SIZE, block_size);
    std::cout << block_size << ": " << (check_result(target, gpu_result) ? "equal": "wrong") << std::endl;
    block_size = 1024;
    gpu_result = sum_gpu(input, SAMPLE_SIZE, block_size);
    std::cout << block_size << ": " << (check_result(target, gpu_result) ? "equal": "wrong") << std::endl;
    free(input);
}
