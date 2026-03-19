#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#define NUM_BINS 256
#define N (1 << 24)  // 16M elements

// cpu端生成随机输入数据
void generate_input(int* a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = rand() % NUM_BINS;  // 生成0~255的随机数
    }
}

void histogram_cpu(int* a, int* y, int n) {
    for (int i = 0; i < n; i++) {
        y[a[i]]++;
    }
}

__global__ void histogram_gpu(int* a, int* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    // 计算全局线程ID
    if (idx < n ) {
        atomicAdd(&y[a[idx]], 1);
    }
}

__global__ void histogram_gpu2(int* a, int* y, int n) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        int4 data = reinterpret_cast<int4*>(a)[idx / 4];  // 每个线程处理4个元素
        atomicAdd(&y[data.x], 1);
        atomicAdd(&y[data.y], 1);
        atomicAdd(&y[data.z], 1);
        atomicAdd(&y[data.w], 1);
    }
}

__global__ void histogram_gpu3(int* a, int* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int local_y[NUM_BINS];

    for (int i=threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        local_y[i] = 0;
    }
    __syncthreads();

    if (idx < n) {
        atomicAdd(&local_y[a[idx]], 1);
    }
    __syncthreads();
    
    for (int i=threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&y[i], local_y[i]);
    }
}

__global__ void histogram_gpu4(int *a, int* y, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int local_y[NUM_BINS];
    for (int i = threadIdx.x; i<NUM_BINS; i+= blockDim.x) {
        local_y[i] = 0;
    }
    __syncthreads();
    if (tid < n / 4) {
        int4 data = reinterpret_cast<int4*>(a)[tid];
        atomicAdd(&local_y[data.x], 1);
        atomicAdd(&local_y[data.y], 1);
        atomicAdd(&local_y[data.z], 1);
        atomicAdd(&local_y[data.w], 1);
    }
    __syncthreads();
    for (int i= threadIdx.x; i < NUM_BINS; i+= blockDim.x) {
        atomicAdd(&y[i], local_y[i]);
    }
}

bool check_result(int* y_cpu, int* y_gpu) {
    for(int i=0; i<NUM_BINS; i++) {
        if (y_cpu[i] != y_gpu[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "Hello, CUDA Histogram!" << std::endl;

    int* a = new int[N];
    int* y_cpu = new int[NUM_BINS]();
    int* y_gpu = new int[NUM_BINS]();

    generate_input(a, N);
    histogram_cpu(a, y_cpu, N);

    int* d_a;
    int* d_y;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_y, NUM_BINS * sizeof(int));
    cudaMemcpy(d_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, NUM_BINS * sizeof(int));  
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // histogram_gpu<<<(N + NUM_BINS-1) / NUM_BINS, NUM_BINS>>>(d_a, d_y, N);

    // histogram_gpu2<<<(N + NUM_BINS-1) / NUM_BINS, NUM_BINS/4>>>(d_a, d_y, N);
    // histogram_gpu3<<<(N + NUM_BINS-1) / NUM_BINS, NUM_BINS>>>(d_a, d_y, N);
    histogram_gpu4<<<(N + NUM_BINS-1) / NUM_BINS, NUM_BINS/4>>>(d_a, d_y, N);

    cudaDeviceSynchronize();
    cudaMemcpy(y_gpu, d_y, sizeof(int) * NUM_BINS, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU time: " << milliseconds << " ms" << std::endl;
    cudaFree(d_a);
    cudaFree(d_y);
    check_result(y_cpu, y_gpu) ? std::cout << "Results match!" << std::endl : std::cout << "Results do not match!" << std::endl;
    delete[] a;
    delete[] y_cpu;

    return 0;
}