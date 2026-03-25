/*
给定长度为 N 的 float 数组 x，在 GPU 上计算总和 sum(x)。

基础版要求

写 reduction_v1：每个 block 用 shared memory 做树形归约，输出每个 block 的部分和。

再启动第二个 kernel（或循环调用）把部分和继续归约到 1 个值。

与 CPU 结果对比，误差 abs(gpu-cpu) < 1e-3。

测试 N = 1<<24。

进阶优化点（按顺序做）

减少分支发散（sequential addressing）。

加载两倍数据（每线程处理 2 个元素）。

使用 warp-level 优化（最后 32 个线程不再 __syncthreads()）。

使用 __shfl_down_sync 做 warp 内归约。

你会学到

grid/block 索引设计

shared memory 使用

__syncthreads() 的正确时机

内存带宽与 occupancy 的关系

CUDA 性能分析基本方法
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>

#define N (1 << 24)  // 16M elements
#define BLOCK_SIZE 256

void generate_input(float *x, int n) {
    for(int i=0; i<n; i++) {
        x[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void reduction_cpu(float *x, int n, float* result) {
    *result = 0.0f;
    for(int i=0; i<n; i++) {
        *result += x[i];
    }
}

__global__ void reduction_gpu(float *d_x, int n, float* d_result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float local_sum;
    if (threadIdx.x == 0) local_sum = 0.0f;
    __syncthreads();
    if (tid * 1 < n) {
        atomicAdd(&local_sum, d_x[tid]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(d_result, local_sum);
    }
}

__global__ void reduction_gpu_v2(float *d_x, int n, float* d_result){
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    __shared__ float thread_vals[BLOCK_SIZE];
    thread_vals[threadIdx.x] = 0.0f;
    float sum = 0.0f;
    if (i < n) sum = d_x[i];
    if (i+blockDim.x < n) sum += d_x[i+blockDim.x];
    thread_vals[threadIdx.x] = sum;
    __syncthreads();

    for (int bound = BLOCK_SIZE/2; bound > 0; bound/=2) {
        if (threadIdx.x < bound) {
            float sum = 0.0f;
            sum = thread_vals[threadIdx.x];
            sum += thread_vals[threadIdx.x + bound];
            thread_vals[threadIdx.x] = sum;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(d_result, thread_vals[0]);
    }
    __syncthreads();
}

__global__ void reduction_gpu_v3(float *d_x, int n, float* d_result_buffer){
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    volatile __shared__ float thread_vals[BLOCK_SIZE];
    thread_vals[threadIdx.x] = 0.0f;
    float sum = 0.0f;
    if (i < n) sum = d_x[i];
    if (i+blockDim.x < n) sum += d_x[i+blockDim.x];
    thread_vals[threadIdx.x] = sum;
    __syncthreads();

    for (int bound = BLOCK_SIZE/2; bound > 0; bound/=2) {
        if (threadIdx.x < bound) {
            float sum = thread_vals[threadIdx.x];
            sum += thread_vals[threadIdx.x + bound];
            thread_vals[threadIdx.x] = sum;
        }
        if (bound > 32) {
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        d_result_buffer[blockIdx.x] = thread_vals[0];
    }
}

__global__ void reduction_gpu_v4(float *d_x, int n, float* d_result_buffer){
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    __shared__ float thread_vals[BLOCK_SIZE];
    thread_vals[threadIdx.x] = 0.0f;
    float sum = 0.0f;
    if (i < n) sum = d_x[i];
    if (i+blockDim.x < n) sum += d_x[i+blockDim.x];
    thread_vals[threadIdx.x] = sum;
    __syncthreads();
    int bound = BLOCK_SIZE/2;
    for (; bound > 32; bound/=2) {
        if (threadIdx.x < bound) {
            float sum = thread_vals[threadIdx.x];
            sum += thread_vals[threadIdx.x + bound];
            thread_vals[threadIdx.x] = sum;
        }
        __syncthreads();
    }
    for (; bound > 0; bound/=2) {
        if (threadIdx.x < bound) {
            volatile float* vmem = &thread_vals[threadIdx.x];
            volatile float* vmem2 = &thread_vals[threadIdx.x+bound];
            // float sum = *vmem;
            // sum += *vmem2;
            // *vmem = sum;
            *vmem += *vmem2;
        }
    }

    if (threadIdx.x == 0) {
        d_result_buffer[blockIdx.x] = thread_vals[0];
    }
}

__global__ void reduction_gpu_v5(float *d_x, int n, float* d_result_buffer){
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    __shared__ float thread_vals[BLOCK_SIZE];
    float sum = 0.0f;
    if (i < n) sum = d_x[i];
    if (i+blockDim.x < n) sum += d_x[i+blockDim.x];
    thread_vals[threadIdx.x] = sum;
    __syncthreads();
    int bound = BLOCK_SIZE/2;
    for (; bound > 32; bound/=2) {
        if (threadIdx.x < bound) {
            float sum = thread_vals[threadIdx.x];
            sum += thread_vals[threadIdx.x + bound];
            thread_vals[threadIdx.x] = sum;
        }
        __syncthreads();
    }
    volatile float* vmem = thread_vals;
    // for (; bound > 0; bound/=2) {
    //     if (threadIdx.x < bound) {
    //         vmem[threadIdx.x] += vmem[threadIdx.x+bound];
    //     }
    // }
    if (threadIdx.x < 32)  vmem[threadIdx.x] += vmem[threadIdx.x+32];
    if (threadIdx.x < 16)  vmem[threadIdx.x] += vmem[threadIdx.x+16];
    if (threadIdx.x < 8)  vmem[threadIdx.x] += vmem[threadIdx.x+8];
    if (threadIdx.x < 4)  vmem[threadIdx.x] += vmem[threadIdx.x+4];
    if (threadIdx.x < 2)  vmem[threadIdx.x] += vmem[threadIdx.x+2];
    if (threadIdx.x < 1)  vmem[threadIdx.x] += vmem[threadIdx.x+1];

    if (threadIdx.x == 0) {
        d_result_buffer[blockIdx.x] = thread_vals[0];
    }
}

__global__ void reduction_gpu_v6(float *d_x, int n, float* d_result_buffer){
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    __shared__ float thread_vals[BLOCK_SIZE];
    float sum = 0.0f;
    if (i < n) sum = d_x[i];
    if (i+blockDim.x < n) sum += d_x[i+blockDim.x];
    thread_vals[threadIdx.x] = sum;
    __syncthreads();
    int bound = BLOCK_SIZE/2;
    for (; bound > 32; bound/=2) {
        if (threadIdx.x < bound) {
            float sum = thread_vals[threadIdx.x];
            sum += thread_vals[threadIdx.x + bound];
            thread_vals[threadIdx.x] = sum;
        }
        __syncthreads();
    }
    float val = 0.0f;
    if (threadIdx.x < 32) {
        val = thread_vals[threadIdx.x * 2] + thread_vals[threadIdx.x * 2 + 1];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
    }

    if (threadIdx.x == 0) {
        d_result_buffer[blockIdx.x] = val;
    }
}

bool check_result(float cpu_result, float gpu_result) {
    float abs_diff =  fabs(cpu_result - gpu_result);
    std::cout << "CPU result: " << cpu_result << ", GPU result: " << gpu_result << ", abs diff: " << abs_diff << std::endl;
    return abs_diff / fabs(cpu_result) < 1e-4;
}


void calc1(float* x, float result_cpu) {
    float result_gpu;

    float *d_x, *d_result;
    cudaMalloc(&d_x, sizeof(float) * N);
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_x, x, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduction_gpu<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_x, N, d_result);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU reduction time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(&result_gpu, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_result);

    if (check_result(result_cpu, result_gpu)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }
}

void calc2(float* x, float result_cpu) {
    float result_gpu;

    float *d_x, *d_result;
    cudaMalloc(&d_x, sizeof(float) * N);
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_x, x, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduction_gpu_v2<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_x, N, d_result);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU reduction time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(&result_gpu, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_result);

    if (check_result(result_cpu, result_gpu)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }
}


void calc3(float* x, float result_cpu) {
    float result_gpu;

    float *d_x, *d_result_buffer;
    int DATA_SIZE = 2 * BLOCK_SIZE;
    cudaMalloc(&d_x, sizeof(float) * N);
    cudaMalloc(&d_result_buffer, sizeof(float) * (N + DATA_SIZE - 1) / DATA_SIZE);
    cudaMemcpy(d_x, x, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemset(d_result_buffer, 0, sizeof(float) * (N + DATA_SIZE - 1) / DATA_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int len = N;
    for(; len > DATA_SIZE; len = (len + DATA_SIZE - 1) / DATA_SIZE) {
        reduction_gpu_v3<<<(len + DATA_SIZE - 1) / DATA_SIZE, BLOCK_SIZE>>>(d_x, len, d_result_buffer);
        float* tmp = d_x;
        d_x = d_result_buffer;
        d_result_buffer = tmp;
    }
    reduction_gpu_v3<<<1, BLOCK_SIZE>>>(d_x, len, d_result_buffer);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU reduction time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(&result_gpu, d_result_buffer + 0, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_result_buffer);

    if (check_result(result_cpu, result_gpu)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }
}

void calc4(float* x, float result_cpu) {
    float result_gpu;

    float *d_x, *d_result_buffer;
    int DATA_SIZE = 2 * BLOCK_SIZE;
    cudaMalloc(&d_x, sizeof(float) * N);
    cudaMalloc(&d_result_buffer, sizeof(float) * (N + DATA_SIZE - 1) / DATA_SIZE);
    cudaMemcpy(d_x, x, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemset(d_result_buffer, 0, sizeof(float) * (N + DATA_SIZE - 1) / DATA_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int len = N;
    for(; len > DATA_SIZE; len = (len + DATA_SIZE - 1) / DATA_SIZE) {
        reduction_gpu_v4<<<(len + DATA_SIZE - 1) / DATA_SIZE, BLOCK_SIZE>>>(d_x, len, d_result_buffer);
        float* tmp = d_x;
        d_x = d_result_buffer;
        d_result_buffer = tmp;
    }
    reduction_gpu_v4<<<1, BLOCK_SIZE>>>(d_x, len, d_result_buffer);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU reduction time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(&result_gpu, d_result_buffer + 0, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_result_buffer);

    if (check_result(result_cpu, result_gpu)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }
}

void calc5(float* x, float result_cpu) {
    float result_gpu;

    float *d_x, *d_result_buffer;
    int DATA_SIZE = 2 * BLOCK_SIZE;
    cudaMalloc(&d_x, sizeof(float) * N);
    cudaMalloc(&d_result_buffer, sizeof(float) * (N + DATA_SIZE - 1) / DATA_SIZE);
    cudaMemcpy(d_x, x, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemset(d_result_buffer, 0, sizeof(float) * (N + DATA_SIZE - 1) / DATA_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int len = N;
    for(; len > DATA_SIZE; len = (len + DATA_SIZE - 1) / DATA_SIZE) {
        reduction_gpu_v5<<<(len + DATA_SIZE - 1) / DATA_SIZE, BLOCK_SIZE>>>(d_x, len, d_result_buffer);
        float* tmp = d_x;
        d_x = d_result_buffer;
        d_result_buffer = tmp;
    }
    reduction_gpu_v5<<<1, BLOCK_SIZE>>>(d_x, len, d_result_buffer);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU reduction time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(&result_gpu, d_result_buffer + 0, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_result_buffer);

    if (check_result(result_cpu, result_gpu)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }
}


void calc6(float* x, float result_cpu) {
    float result_gpu;

    float *d_x, *d_result_buffer;
    int DATA_SIZE = 2 * BLOCK_SIZE;
    cudaMalloc(&d_x, sizeof(float) * N);
    cudaMalloc(&d_result_buffer, sizeof(float) * (N + DATA_SIZE - 1) / DATA_SIZE);
    cudaMemcpy(d_x, x, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemset(d_result_buffer, 0, sizeof(float) * (N + DATA_SIZE - 1) / DATA_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int len = N;
    for(; len > DATA_SIZE; len = (len + DATA_SIZE - 1) / DATA_SIZE) {
        reduction_gpu_v6<<<(len + DATA_SIZE - 1) / DATA_SIZE, BLOCK_SIZE>>>(d_x, len, d_result_buffer);
        float* tmp = d_x;
        d_x = d_result_buffer;
        d_result_buffer = tmp;
    }
    reduction_gpu_v6<<<1, BLOCK_SIZE>>>(d_x, len, d_result_buffer);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU reduction time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(&result_gpu, d_result_buffer + 0, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_result_buffer);

    if (check_result(result_cpu, result_gpu)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }
}

int main() {
    float* x = new float[N];
    float result_cpu;
    generate_input(x, N);

    reduction_cpu(x, N, &result_cpu);
    calc1(x, result_cpu);
    calc1(x, result_cpu);
    calc2(x, result_cpu);
    calc3(x, result_cpu);
    calc4(x, result_cpu);
    calc5(x, result_cpu);
    std::cout << "calc6:" << std::endl;
    calc6(x, result_cpu);
    delete[] x;

}