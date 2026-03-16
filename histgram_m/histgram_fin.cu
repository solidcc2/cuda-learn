#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <cmath>
#define N (1<<24)
#define BUCKET_NUM (1<<10)

void generate_input(int* input, int n) {
    for(int i=0; i<n; i++) {
        input[i] = rand() % BUCKET_NUM;
    }
}

void histogram_cpu(int* input, int len, int* histogram) {
    for(int i=0; i<len; i++) {
        histogram[input[i]]++; 
    }
}

__global__ void _cu_histogram_gpu(int* d1, int len, int* histogram) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < len)  atomicAdd(&histogram[d1[tid]], 1);
}

void histogram_gpu(int* input, int len, int* histogram) {
    int* d1, *d2;
    cudaMalloc(&d1, sizeof(int) * len);
    cudaMalloc(&d2, sizeof(int) * len);
    cudaMemset(&d2, 0, sizeof(int) * len);
    cudaMemcpy(d1, input, sizeof(int) * len, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);

    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int block_size = 256;
    _cu_histogram_gpu<<<(N + block_size - 1)/block_size, block_size>>>(d1, N, d2);
    
    cudaMemcpy(histogram, d2, BUCKET_NUM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << "use: " << elapsed << "ms" << std::endl;

    cudaFree(d1);
    cudaFree(d2);

}

bool check(int* histogram_cpu, int* histogram_gpu) {
    for(int i=0; i<BUCKET_NUM; i++) {
        if (histogram_cpu[i] != histogram_gpu[i]) return false;
    }
    return true;
}

int main() {
    int* input = (int*) malloc(sizeof(int) * N);
    generate_input(input, N);
    int* histogram_c, *histogram_g;
    histogram_c = (int*)malloc(sizeof(int) * BUCKET_NUM);
    histogram_cpu(input, N, histogram_c);

    histogram_g = (int*)malloc(sizeof(int) * BUCKET_NUM);
    histogram_gpu(input, N, histogram_g);
    std::cout << (check(histogram_c, histogram_g) ? "right" : "wrong") << std::endl;
    free(histogram_g);
}