#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#define INPUT_LEN 2048
// 基础测试输入
void generate_input(float *x, int n) {
    for(int i=0; i<n; i++) {
        // x[i] = static_cast<float>(rand()) / RAND_MAX;
        x[i] = static_cast<float>(rand());
    }
}

void softmax_cpu(float *input, int n, float* output) {
    float m = *std::max_element(input, input+n);
    float sum = 0.0f;
    for(int i=0; i<n; i++) {
        output[i] = expf(input[i] - m);
        sum += output[i];
    }
    for(int i=0; i<n; i++) {
        output[i] = output[i] / sum;
    }
}

void online_softmax_cpu(float* input, int n, float* output) {
    float m = input[0];
    float sum = 0.0f;
    for(int i = 0; i<n; i++) {
        if (m < input[i]) {
            sum = sum * expf(m - input[i]);
            m = input[i];
        }
        sum += expf(input[i] - m);
    }
    for(int i=0; i<n; i++) {
        output[i] = expf(input[i] - m) / sum;
    }
}

__global__ void _cu_max(float* d_input, int n, float* d_output) {
    extern __shared__ float buf[];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float in1 = tid*2 < n ? d_input[tid * 2]: -INFINITY;
    float in2 = tid*2+1 < n ? d_input[tid * 2 + 1]: -INFINITY;
    buf[threadIdx.x] = max(in1, in2);
    __syncthreads();
    int bound = blockDim.x / 2;
    for(; bound >= 32; bound /= 2) {
        if (threadIdx.x < bound) {
            buf[threadIdx.x] = max(buf[threadIdx.x], buf[threadIdx.x + bound]);
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        float var = buf[threadIdx.x];
        var = max(var, __shfl_down_sync(0xffffffff, var, 16));
        var = max(var, __shfl_down_sync(0xffffffff, var, 8));
        var = max(var, __shfl_down_sync(0xffffffff, var, 4));
        var = max(var, __shfl_down_sync(0xffffffff, var, 2));
        var = max(var, __shfl_down_sync(0xffffffff, var, 1));
        if (threadIdx.x == 0) {
            d_output[blockIdx.x] = var;
        }
    }
}

void _max_gpu(float*& buf1, int n, float*& buf2) {
    int block_size = 256;
    for(; n > block_size*2; n = (n + block_size*2 - 1) / (block_size*2) ){
        _cu_max<<<(n + block_size*2 - 1)/(block_size*2), block_size, block_size*sizeof(float)>>>(buf1, n, buf2);
        std::swap(buf1, buf2); 
    }
    _cu_max<<<1, block_size, block_size*sizeof(float)>>>(buf1, n, buf2);
}

__global__ void _cu_sum(float* buf1, int n, float* buf2) {
    extern __shared__ float buf[];
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float in1 = tid*2 < n ? buf1[tid*2] : 0.0f;
    float in2 = tid*2 + 1 < n ? buf1[tid*2+1] : 0.0f;
    buf[threadIdx.x] = in1 + in2;
    __syncthreads();
    int bound = blockDim.x / 2;
    for(; bound >= 32; bound >>= 1) {
        if (threadIdx.x < bound) {
            buf[threadIdx.x] = buf[threadIdx.x] + buf[threadIdx.x + bound];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        float var = buf[threadIdx.x];
        var += __shfl_down_sync(0xffffffff, var, 16);
        var += __shfl_down_sync(0xffffffff, var, 8);
        var += __shfl_down_sync(0xffffffff, var, 4);
        var += __shfl_down_sync(0xffffffff, var, 2);
        var += __shfl_down_sync(0xffffffff, var, 1);
        if (threadIdx.x == 0) {
            buf2[blockIdx.x] = var;
        }
    }
}

void _sum_gpu(float*& buf1, int n, float*& buf2) {
    int block_size = 256;
    for(; n > block_size*2; n = (n + block_size*2 - 1) / (block_size*2)) {
        _cu_sum<<<(n+block_size*2-1)/(block_size*2), block_size, block_size*sizeof(float)>>>(buf1, n, buf2);
        std::swap(buf1, buf2);
    }
    _cu_sum<<<1, block_size, block_size*sizeof(float)>>>(buf1, n, buf2);
}

__global__ void step1(float* buf1, int n, float* m) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        buf1[tid] = expf(buf1[tid] - *m);    // 可以128位宽优化
    }
}

__global__ void step2(float* buf1, int n, float *sum) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < n) {
        buf1[tid] /= *sum;
    }
}


void softmax_gpu(float* input, int n, float* output) {
    float* d_input;
    cudaMalloc(&d_input, sizeof(float)* n);
    cudaMemcpy(d_input, input, sizeof(float)*n, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    float* buf1, *buf2;
    cudaMalloc(&buf1, sizeof(float) * n);
    cudaMalloc(&buf2, sizeof(float) * n);
    cudaMemcpy(buf1, d_input, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    _max_gpu(buf1, n, buf2);
    cudaMemcpy(buf1, d_input, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    int block_size = 256;
    step1<<<(n + block_size - 1)/ block_size, block_size>>>(buf1, n, &buf2[0]);
    cudaMemcpy(d_input, buf1, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    _sum_gpu(buf1, n, buf2);
    step2<<<(n+block_size-1)/block_size, block_size>>>(d_input, n, &buf2[0]);
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapse;
    cudaEventElapsedTime(&elapse, start, end);
    std::cout << "use: " << elapse << "ms" << std::endl;
    cudaMemcpy(output, d_input, sizeof(float) * n, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(buf1);
    cudaFree(buf2);
}


__global__ void _cu_softmax_single_block(float* input, int n, float* output) {
    float t_m = -INFINITY;
    __shared__ float m, sum;
    extern __shared__ float block[];
    for(int i=threadIdx.x; i < n; i += blockDim.x) {    // 128位宽优化
        t_m = max(input[i], t_m);
    }
    block[threadIdx.x] = t_m;
    __syncthreads();
    int bound = blockDim.x / 2;
    for (; bound >= 32; bound >>= 1) {
        if (threadIdx.x < bound) {
            block[threadIdx.x] = max(block[threadIdx.x], block[threadIdx.x + bound]);
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        float var = block[threadIdx.x];
        var = max(var, __shfl_down_sync(0xffffffff, var, 16));
        var = max(var, __shfl_down_sync(0xffffffff, var, 8));
        var = max(var, __shfl_down_sync(0xffffffff, var, 4));
        var = max(var, __shfl_down_sync(0xffffffff, var, 2));
        var = max(var, __shfl_down_sync(0xffffffff, var, 1));
        if (threadIdx.x == 0) {
            m = var;
        }
    }
    __syncthreads();
    
    for(int i=threadIdx.x; i<n; i+=blockDim.x) { // 128位宽优化
        output[i] = __expf(input[i]-m);
    }
    float t_sum = 0.0f;
    for(int i=threadIdx.x; i < n; i+=blockDim.x) {
        t_sum += output[i];
    }
    block[threadIdx.x] = t_sum;
    __syncthreads();
    bound = blockDim.x / 2;
    for(; bound >= 32; bound >>= 1) {
        if (threadIdx.x < bound) {
            block[threadIdx.x] = block[threadIdx.x] + block[threadIdx.x + bound];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        float var = block[threadIdx.x];
        var += __shfl_down_sync(0xffffffff, var, 16);
        var += __shfl_down_sync(0xffffffff, var, 8);
        var += __shfl_down_sync(0xffffffff, var, 4);
        var += __shfl_down_sync(0xffffffff, var, 2);
        var += __shfl_down_sync(0xffffffff, var, 1);
        if (threadIdx.x == 0) {
            sum = var;
        }
    }
    __syncthreads();
    for(int i=threadIdx.x; i < n; i+=blockDim.x) {
        output[i] /= sum;
    }
}

void softmax_gpu_single_block(float* input, int n, float* output) {
    float* d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float)* n);
    cudaMalloc(&d_output, sizeof(float) * n);
    cudaMemcpy(d_input, input, sizeof(float)*n, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    int block_size = 256;
    _cu_softmax_single_block<<<1, block_size, block_size * sizeof(float)>>>(d_input, n, d_output);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapse;
    cudaEventElapsedTime(&elapse, start, end);
    std::cout << "use: " << elapse << "ms" << std::endl;
    cudaMemcpy(output, d_output, sizeof(float) * n, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void _cu_online_softmax_single_block(float* input, int n, float* output) {
    extern __shared__ float block_buf[];
    float* maxs = block_buf;
    float* sums = block_buf + blockDim.x;
    maxs[threadIdx.x] = -INFINITY;
    sums[threadIdx.x] = 0.0f;
    for(int i=threadIdx.x; i<n; i+=blockDim.x) {
        float in = input[i];
        float m = max(maxs[threadIdx.x], in);
        sums[threadIdx.x] = sums[threadIdx.x] * __expf(maxs[threadIdx.x] - m) + __expf(in - m);
        maxs[threadIdx.x] = m;
    }
    __syncthreads();
    int bound = blockDim.x / 2;
    for(; bound >= 32; bound >>= 1) {
        if (threadIdx.x < bound) {
            float max_ = max(maxs[threadIdx.x], maxs[threadIdx.x + bound]);
            sums[threadIdx.x] = 
                sums[threadIdx.x] * __expf(maxs[threadIdx.x] - max_) +
                sums[threadIdx.x + bound] * __expf(maxs[threadIdx.x + bound] - max_);
            maxs[threadIdx.x] = max_;
        }
        __syncthreads();
    }
    __shared__ float s_sum, s_max;
    if (threadIdx.x < 32) {
        float max_ = maxs[threadIdx.x];
        float sum_ = sums[threadIdx.x];
        float m;
        float max_n, sum_n;
        max_n = __shfl_down_sync(0xffffffff, max_, 16);
        sum_n = __shfl_down_sync(0xffffffff, sum_, 16);
        m = max(max_, max_n);
        sum_ = sum_ * __expf(max_ - m) + sum_n * __expf(max_n - m);
        max_ = m;


        max_n = __shfl_down_sync(0xffffffff, max_, 8);
        sum_n = __shfl_down_sync(0xffffffff, sum_, 8);
        m = max(max_, max_n);
        sum_ = sum_ * __expf(max_ - m) + sum_n * __expf(max_n - m);
        max_ = m;


        max_n = __shfl_down_sync(0xffffffff, max_, 4);
        sum_n = __shfl_down_sync(0xffffffff, sum_, 4);
        m = max(max_, max_n);
        sum_ = sum_ * __expf(max_ - m) + sum_n * __expf(max_n - m);
        max_ = m;


        max_n = __shfl_down_sync(0xffffffff, max_, 2);
        sum_n = __shfl_down_sync(0xffffffff, sum_, 2);
        m = max(max_, max_n);
        sum_ = sum_ * __expf(max_ - m) + sum_n * __expf(max_n - m);
        max_ = m;


        max_n = __shfl_down_sync(0xffffffff, max_, 1);
        sum_n = __shfl_down_sync(0xffffffff, sum_, 1);
        m = max(max_, max_n);
        sum_ = sum_ * __expf(max_ - m) + sum_n * __expf(max_n - m);
        max_ = m;

        if (threadIdx.x == 0) {
            s_sum = sum_;
            s_max = max_;
        }
    }
    __syncthreads();
    for(int i= threadIdx.x; i < n; i+= blockDim.x) {
        output[i] = __expf(input[i] - s_max) / s_sum;
    }
}

void online_softmax_gpu_single_block(float* input, int n, float* output) {
    float* d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float)* n);
    cudaMalloc(&d_output, sizeof(float) * n);
    cudaMemcpy(d_input, input, sizeof(float)*n, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    int block_size = 256;
    _cu_online_softmax_single_block<<<1, block_size, 2 * block_size * sizeof(float)>>>(d_input, n, d_output);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapse;
    cudaEventElapsedTime(&elapse, start, end);
    std::cout << "use: " << elapse << "ms" << std::endl;
    cudaMemcpy(output, d_output, sizeof(float) * n, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}


bool nearly_equal(float a, float b, float rtol = 1e-5f, float atol = 1e-8f) {
    return fabsf(a - b) <= atol + rtol * fmaxf(fabsf(a), fabsf(b));
}


bool check_result(float* base, int n, float* check) {
    for(int i=0; i<n; i++) {
        if (!nearly_equal(base[i], check[i])) {
            return false;
        }
    }
    return true;
}


int main() {
    int input_len = INPUT_LEN;
    float *input = (float*) malloc(sizeof(float) * input_len);
    generate_input(input, input_len);
    float *sm_cpu = (float*) malloc(sizeof(float) * input_len);
    float *osm_cpu = (float*) malloc(sizeof(float) * input_len);
    softmax_cpu(input, input_len, sm_cpu);
    online_softmax_cpu(input, input_len, osm_cpu);
    std::cout << "cpu check: " 
        << (check_result(sm_cpu, input_len, osm_cpu) ? "right" : "wrong") << std::endl;

    float* sm_gpu = (float*)malloc(sizeof(float)* input_len);
    softmax_gpu(input, input_len, sm_gpu);
    std::cout << "softmax_gpu check: "
        << (check_result(sm_cpu, input_len, sm_gpu) ? "right" : "wrong") << std::endl;
    free(sm_gpu);

    float* sm_gpu_single_block = (float*)malloc(sizeof(float)* input_len);
    softmax_gpu_single_block(input, input_len, sm_gpu_single_block);
    std::cout << "sm_gpu_single_block check: "
        << (check_result(sm_cpu, input_len, sm_gpu_single_block) ? "right" : "wrong") << std::endl;
    free(sm_gpu_single_block);

    float* osm_gpu_single_block = (float*)malloc(sizeof(float)* input_len);
    online_softmax_gpu_single_block(input, input_len, osm_gpu_single_block);
    std::cout << "online_softmax_gpu_single_block check: "
        << (check_result(sm_cpu, input_len, osm_gpu_single_block) ? "right" : "wrong") << std::endl;
    free(osm_gpu_single_block);

    free(osm_cpu);
    free(sm_cpu);
    free(input);
}
