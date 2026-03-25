#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cmath>
#include <chrono>

void generate_matrix(float* A, size_t len) {
    for(size_t i=0; i<len; i++) {
        A[i] = (float)rand();
    }
}

bool nearly_equal(float a, float b, float rtol = 1e-5f, float atol = 1e-8f) {
    return fabsf(a - b) <= atol + rtol * fmaxf(fabsf(a), fabsf(b));
}

void transpose_cpu(const float* A, float* O, size_t rows, size_t cols) {
    for(size_t i=0; i<rows; i++) {
        for(size_t j=0; j<cols; j++) {
            O[j * rows + i] = A[i * cols + j];
        }
    }
}
void matmul_T_cpu(const float* A, const float* BT, float* O,
    size_t a, size_t b, size_t c) {
    for(size_t i=0; i<a; i++) {
        for(size_t j=0; j<b; j++) {
            O[i*b + j] = 0.0f;
            for(int k=0; k<c; k++) {
                O[i*b + j] += A[i*c + k] * BT[j*c + k];
            }
        }
    }
}

void matmul_cpu(const float* A,     // a x c
            const float* B,         // c x b
            float* O,         // a x b
            size_t a,  
            size_t b,
            size_t c) {
    float* BT = new float[b * c];
    transpose_cpu(B, BT, c, b);
    matmul_T_cpu(A, BT, O, a, b, c);
    delete[] BT;
}

void online_softmax_local_cpu(float* A, size_t rows, size_t cols) {
    for(int i=0; i<rows; i++) {
        float m = -INFINITY;
        float acc = 0.0f;
        float* line = A + i*cols;
        for(size_t j=0; j<cols; j++) {
            float new_m = std::max(m, line[j]);
            acc = acc * expf(m - new_m) + expf(line[j] - new_m);
            m = new_m;
        }
        for(size_t j=0; j<cols; j++) {
            line[j] = expf(line[j]-m) / acc;
        }
    }
}

void flash_attention_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    auto start = std::chrono::high_resolution_clock::now();
    float* buf = new float[ seq_len * seq_len];
    float sqrt_h = std::sqrt(head_dim);
    int head_size = seq_len * head_dim;
    int batch_elem_num = head_size * num_heads;
    for(int b=0; b<batch_size; b++) {
        const float* Q_b = Q + b * batch_elem_num;
        const float* K_b = K + b * batch_elem_num;
        const float* V_b = V + b * batch_elem_num;
        float * O_b = O + b * batch_elem_num;
        for(int h=0; h<num_heads; h++) {
            const float* Q_h = Q_b + h * head_size;
            const float* K_h = K_b + h * head_size;
            const float* V_h = V_b + h * head_size;
            float * O_h = O_b + h * head_size;
            matmul_T_cpu(Q_h, K_h, buf, seq_len, seq_len, head_dim);
            for(size_t i=0; i<seq_len*seq_len; i++) {
                buf[i] /= sqrt_h;
            }
            online_softmax_local_cpu(buf, seq_len, seq_len);
            matmul_cpu(buf, V_h, O_h, seq_len, head_dim, seq_len);

        }
    }
    delete[] buf;
    auto end = std::chrono::high_resolution_clock::now();
    double use = std::chrono::duration<double, std::milli>(end-start).count();
    std::cout << "flash_attention_cpu use: " << use << "ms" << std::endl;
}

int main() {
    int batch_size = 2;
    int num_heads = 4;
    int seq_len = 128;
    int head_dim = 256;
    float* Q = new float[batch_size * num_heads * seq_len * head_dim];
    float* K = new float[batch_size * num_heads * seq_len * head_dim];
    float* V = new float[batch_size * num_heads * seq_len * head_dim];
    float* O = new float[batch_size * num_heads * seq_len * head_dim];
    generate_matrix(Q, batch_size * num_heads * seq_len * head_dim);
    generate_matrix(K, batch_size * num_heads * seq_len * head_dim);
    generate_matrix(V, batch_size * num_heads * seq_len * head_dim);
    flash_attention_cpu(Q, K, V, O, batch_size, num_heads, seq_len, head_dim);
}
