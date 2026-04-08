#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <string>
#define DEBUG_NUMERIC

namespace {
constexpr int kDebugBatchId = 0;
constexpr int kDebugHeadId = 0;
constexpr int kDebugQTokenId = 0;
constexpr int kDebugChunkId = 0;
constexpr int kDebugWindowElems = 4;
constexpr int kDebugOutElems = 4;
}


__device__ inline float check_nan_val(float x, const char* tag) {
#ifdef DEBUG_NUMERIC
    if (isnan(x)) {
        printf("%s nan: %f tx=%d ty=%d bx=%d by=%d bz=%d\n",
               tag, x, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z);
    }
#endif
    return x;
}

__device__ inline float check_infinite_val(float x, const char* tag) {
#ifdef DEBUG_NUMERIC
    if (!isfinite(x)) {
        printf("%s isfinite: %f tx=%d ty=%d bx=%d by=%d bz=%d\n",
               tag, x, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z);
    }
#endif
    return x;
}

__device__ inline float safe_sub(float a, float b) {
    return (isfinite(a) && isfinite(b)) ? a - b : -INFINITY;     
}

__device__ inline float softmax_div(float a, float b) {
    return (b != 0.0) ? a/b : 0.0;
}

void check_inputs(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int64_t max_seqlen_q,
    torch::Tensor cu_seqlens_q,
    int64_t max_seqlen_k,
    torch::Tensor seqused_k,
    bool causal,
    int64_t window_size_left, 
    int64_t window_size_right,
    torch::Tensor block_table,
    torch::Tensor out) {

    TORCH_CHECK(q.sizes() == out.sizes(), "q and out must have the same shape");
    // TORCH_CHECK(q.is_contiguous(), "q is not contiguous");
    // TORCH_CHECK(k.is_contiguous(), "k is not contiguous");
    // TORCH_CHECK(v.is_contiguous(), "v is not contiguous");
    TORCH_CHECK(q.size(1) == k.size(2), "GQA not support, num_head must equal num_kv_head");
}

#define BLOCK_ROW_V2 8

// block by q_rows
template<typename scalar_t>
__global__ void flash_attn_varlen_with_block_kernel_v2(
    const scalar_t* q,  // total_q x num_heads x head_dim
    const int64_t q_stride0,
    const int64_t q_stride1,
    const int64_t q_stride2, 
    const scalar_t* k,  // num_blocks x block_size x num_kv_heads x head_dim
    const int64_t k_stride0,
    const int64_t k_stride1,
    const int64_t k_stride2,
    const int64_t k_stride3,
    const scalar_t* v,
    const int64_t v_stride0,
    const int64_t v_stride1,
    const int64_t v_stride2,
    const int64_t v_stride3,
    const int64_t max_seqlen_q,
    const int32_t* cu_seqlens_q,
    const int64_t max_seqlen_k,
    const int32_t* seqused_k,
    bool causal,
    int64_t window_size_left,
    int64_t window_size_right,
    const int32_t* block_table,
    const int64_t bt_stride0,
    const int64_t bt_stride1,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    scalar_t* out
) {
    extern __shared__ scalar_t tile[];
    const int q_seq_chunk_size = blockDim.y;
    const int kv_seq_chunk_size = blockDim.y;
    const int window_chunk_size = kv_seq_chunk_size;
    const scalar_t neg_inf =
            static_cast<scalar_t>(-std::numeric_limits<float>::infinity());

    scalar_t* tile_Q = tile;   // q_seq_chunk_size x head_dim
    scalar_t* tile_K = tile_Q + q_seq_chunk_size * head_dim;  // kv_seq_chunk_size x head_dim
    scalar_t* tile_softmax = tile_K + kv_seq_chunk_size * head_dim;     // q_seq_chunk_size x kv_seq_chunk_size
    scalar_t* tile_softmax_m = tile_softmax + q_seq_chunk_size * kv_seq_chunk_size;   // q_seq_chunk_size x window_chunk_size
    scalar_t* tile_softmax_sum = tile_softmax_m + q_seq_chunk_size * window_chunk_size; // q_seq_chunk_size x window_chunk_size
    scalar_t* tile_VT = tile_softmax_sum + q_seq_chunk_size * window_chunk_size;   // head_dim x kv_seq_chunk_size
    scalar_t* tile_out = tile_VT + head_dim * kv_seq_chunk_size; // q_seq_chunk_size x head_dim
    scalar_t* tile_old_softmax_m = tile_out + q_seq_chunk_size * head_dim;    // q_seq_chunk_size x 1
    scalar_t* tile_old_softmax_sum = tile_old_softmax_m + q_seq_chunk_size * 1;  // q_seq_chunk_size x 1
    scalar_t* tile_dotQK = tile_old_softmax_sum + q_seq_chunk_size * 1;   // q_seq_chunk_size x head_dim
    scalar_t* tile_dotPV = tile_dotQK + q_seq_chunk_size * head_dim;        // window_chunk_size * kv_seq_chunk_size
    // init
    if (threadIdx.x == 0) {
        tile_old_softmax_m[threadIdx.y] = neg_inf;
        tile_old_softmax_sum[threadIdx.y] = scalar_t(0);
    }
    if (threadIdx.x < head_dim) {
        tile_out[threadIdx.y * head_dim + threadIdx.x] = scalar_t(0);
    }

    int64_t batch_id = blockIdx.z / num_heads;
    int64_t head_id = blockIdx.z % num_heads;
    int64_t q_token_id = threadIdx.y + blockIdx.y * blockDim.y;
    int64_t q_seqlen = cu_seqlens_q[batch_id+1] - cu_seqlens_q[batch_id];
    int64_t kv_seqlen = seqused_k[batch_id];
    
    // gather tile_Q
    if (q_token_id < q_seqlen && threadIdx.x < head_dim) {
        tile_Q[threadIdx.y * head_dim + threadIdx.x] = check_nan_val(q[
            q_stride0 * (cu_seqlens_q[batch_id] + q_token_id) + 
            q_stride1 * head_id +
            q_stride2 * threadIdx.x
        ], "tile_for_q");
    } else {
        tile_Q[threadIdx.y * head_dim + threadIdx.x] = scalar_t(0);
    }
    // window size 边界调整
    int64_t history_offset = q_seqlen - 1 - q_token_id; // history offset
    // [absolute_range_left, absolute_range_right)
    if (window_size_left == -1 && window_size_right == -1) {
        window_size_left = kv_seqlen;
        window_size_right = kv_seqlen;
    }
    if (causal == true) {
        window_size_right = 0;
    }
    int64_t absolute_range_right = min(kv_seqlen, kv_seqlen - history_offset + window_size_right); // 开区间
    int64_t absolute_range_left = max(int64_t{0}, kv_seqlen - 1 - history_offset - window_size_left);
#ifdef DEBUG_NUMERIC
    if (threadIdx.x == 0 && q_token_id < q_seqlen) {
        printf(
            "range batch=%lld head=%lld q=%lld q_len=%lld kv_len=%lld hist=%lld left=%lld right=%lld win=(%lld,%lld)\n",
            (long long)batch_id,
            (long long)head_id,
            (long long)q_token_id,
            (long long)q_seqlen,
            (long long)kv_seqlen,
            (long long)history_offset,
            (long long)absolute_range_left,
            (long long)absolute_range_right,
            (long long)window_size_left,
            (long long)window_size_right
        );
    }
#endif
    for (int chunk_id = 0; 
            chunk_id < (absolute_range_right - absolute_range_left + kv_seq_chunk_size - 1) / kv_seq_chunk_size; 
            chunk_id++) 
    {
        int64_t virt_kv_token_id = absolute_range_left + chunk_id * kv_seq_chunk_size + threadIdx.y;
        if (virt_kv_token_id < absolute_range_right && threadIdx.x < head_dim) {
            int64_t phy_block_id = block_table[
                bt_stride0 * batch_id + 
                bt_stride1 * (virt_kv_token_id / block_size) // virt block id
            ];
            tile_K[threadIdx.y * head_dim + threadIdx.x] = check_nan_val(k[
                k_stride0 * phy_block_id +
                k_stride1 * (virt_kv_token_id % block_size) +
                k_stride2 * head_id +
                k_stride3 * threadIdx.x
            ], "tile_for_k");
        } else {
            tile_K[threadIdx.y * head_dim + threadIdx.x] = scalar_t(0);
        }
        __syncthreads();

        // tile matmul
        for(int k_tile_row_id = 0; k_tile_row_id < kv_seq_chunk_size; k_tile_row_id ++) {
            scalar_t result = check_nan_val(tile_Q[threadIdx.y * head_dim + threadIdx.x] * tile_K[k_tile_row_id * head_dim + threadIdx.x], "Q_dot_K");
            tile_dotQK[threadIdx.y * head_dim + threadIdx.x] = result;
            __syncthreads();
            for(int bound = blockDim.x / 2; bound >= warpSize && head_dim > warpSize; bound /= 2) {
                if (threadIdx.x < bound) {
                    result = check_nan_val(result + tile_dotQK[threadIdx.y * head_dim + threadIdx.x + bound], "QK_mul_sum_in_block");
                    tile_dotQK[threadIdx.y * head_dim + threadIdx.x] = result;
                }
                __syncthreads();
            }
            if (threadIdx.x < warpSize) {
                result = check_nan_val(result + __shfl_down_sync(0xffffffff, result, 16), "QK_mul_sum_in_warp_16");
                result = check_nan_val(result + __shfl_down_sync(0xffffffff, result, 8), "QK_mul_sum_in_warp_8");
                result = check_nan_val(result + __shfl_down_sync(0xffffffff, result, 4), "QK_mul_sum_in_warp_4");
                result = check_nan_val(result + __shfl_down_sync(0xffffffff, result, 2), "QK_mul_sum_in_warp_2");
                result = check_nan_val(result + __shfl_down_sync(0xffffffff, result, 1), "QK_mul_sum_in_warp_1");
            }
            if (threadIdx.x == 0 && absolute_range_left + chunk_id * window_chunk_size + k_tile_row_id < absolute_range_right) {
                tile_softmax[
                    threadIdx.y * window_chunk_size + 
                    k_tile_row_id
                ] = check_nan_val(result / sqrt(head_dim), "score");
            } else if (threadIdx.x == 0) {
                tile_softmax[
                    threadIdx.y * window_chunk_size + 
                    k_tile_row_id
                ] = neg_inf;
            }
        }
        __syncthreads();
 #ifdef DEBUG_NUMERIC
        if (batch_id == kDebugBatchId && head_id == kDebugHeadId &&
            q_token_id == kDebugQTokenId && chunk_id == kDebugChunkId &&
            threadIdx.x == 0) {
            for (int i = 0; i < min(window_chunk_size, kDebugWindowElems); ++i) {
                printf(
                    "kernel_score batch=%lld head=%lld q=%lld idx=%d kv=%lld score=%f\n",
                    (long long)batch_id,
                    (long long)head_id,
                    (long long)q_token_id,
                    i,
                    (long long)(absolute_range_left + chunk_id * window_chunk_size + i),
                    (float)tile_softmax[threadIdx.y * window_chunk_size + i]
                );
            }
        }
#endif
        
        if (threadIdx.x < window_chunk_size) {
            // init tile_softmax_m
            tile_softmax_m[threadIdx.y * window_chunk_size + threadIdx.x] = tile_softmax[threadIdx.y * window_chunk_size + threadIdx.x];
            // init tile_softmax_sum
            tile_softmax_sum[threadIdx.y * window_chunk_size + threadIdx.x] = 
                (absolute_range_left + chunk_id * window_chunk_size + threadIdx.x < absolute_range_right) ? scalar_t(1.0) : scalar_t(0.0);
        } 
        int bound = blockDim.x / 2;
        for (; bound >= warpSize; bound /= 2){
            if (threadIdx.x + bound < window_chunk_size) {
                scalar_t m_ = (threadIdx.x < window_chunk_size) ? 
                                    tile_softmax_m[threadIdx.y * window_chunk_size + threadIdx.x] : neg_inf;
                scalar_t m_neighbor = (threadIdx.x + bound < window_chunk_size) ? 
                                    tile_softmax_m[threadIdx.y * window_chunk_size + threadIdx.x + bound] : neg_inf;
            
                scalar_t sum_ = (threadIdx.x < window_chunk_size) ? 
                                    tile_softmax_sum[threadIdx.y * window_chunk_size + threadIdx.x] : scalar_t(0);
                scalar_t sum_neighbor = (threadIdx.x + bound < window_chunk_size) ? 
                                    tile_softmax_sum[threadIdx.y * window_chunk_size + threadIdx.x + bound] : scalar_t(0);
                scalar_t merge_m = check_nan_val(max(m_, m_neighbor), "merge_m");
                scalar_t merge_sum = check_nan_val(sum_ * __expf(safe_sub(m_, merge_m)) + 
                                sum_neighbor * __expf(safe_sub(m_neighbor, merge_m)), "merge_sum");
                tile_softmax_m[threadIdx.y * window_chunk_size + threadIdx.x] = merge_m;
                tile_softmax_sum[threadIdx.y * window_chunk_size + threadIdx.x] = merge_sum;
            }         
            __syncthreads();                                       
        }
        
        if (threadIdx.x < warpSize) {
            scalar_t m_, sum_, m_neighbor, sum_neighbor, merge_m;
            m_ = (threadIdx.x < window_chunk_size) ? 
                                    tile_softmax_m[threadIdx.y * window_chunk_size + threadIdx.x] : neg_inf;
            sum_ = (threadIdx.x < window_chunk_size) ? 
                                    tile_softmax_sum[threadIdx.y * window_chunk_size + threadIdx.x] : scalar_t(0);
            m_neighbor = __shfl_down_sync(0xffffffff, m_, 16);
            sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 16);
            unsigned int y_neighbor = __shfl_down_sync(0xffffffff, threadIdx.y, 16);
            assert(y_neighbor == threadIdx.y);
            merge_m = check_nan_val(max(m_, m_neighbor), "warp_merge_m");
            check_infinite_val(sum_, "sum_");
            check_infinite_val(sum_neighbor, "sum_neighbor");
            sum_ = check_infinite_val(sum_ * __expf(safe_sub(m_, merge_m)) + sum_neighbor * __expf(safe_sub(m_neighbor, merge_m)), "warp_merge_sum");
            m_ = merge_m;

            m_neighbor = __shfl_down_sync(0xffffffff, m_, 8);
            sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 8);
            y_neighbor = __shfl_down_sync(0xffffffff, threadIdx.y, 8);
            assert(y_neighbor == threadIdx.y);
            merge_m = check_nan_val(max(m_, m_neighbor), "warp_merge_m");
            check_infinite_val(sum_neighbor, "sum_neighbor");
            sum_ = check_infinite_val(sum_ * __expf(safe_sub(m_, merge_m)) + sum_neighbor * __expf(safe_sub(m_neighbor, merge_m)), "warp_merge_sum");
            m_ = merge_m;

            m_neighbor = __shfl_down_sync(0xffffffff, m_, 4);
            sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 4);
            y_neighbor = __shfl_down_sync(0xffffffff, threadIdx.y, 4);
            assert(y_neighbor == threadIdx.y);
            merge_m = check_nan_val(max(m_, m_neighbor), "warp_merge_m");
            check_infinite_val(sum_neighbor, "sum_neighbor");
            sum_ = check_infinite_val(sum_ * __expf(safe_sub(m_, merge_m)) + sum_neighbor * __expf(safe_sub(m_neighbor, merge_m)), "warp_merge_sum");
            m_ = merge_m;

            m_neighbor = __shfl_down_sync(0xffffffff, m_, 2);
            sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 2);
            y_neighbor = __shfl_down_sync(0xffffffff, threadIdx.y, 2);
            assert(y_neighbor == threadIdx.y);
            merge_m = check_nan_val(max(m_, m_neighbor), "warp_merge_m");
            check_infinite_val(sum_neighbor, "sum_neighbor");
            sum_ = check_infinite_val(sum_ * __expf(safe_sub(m_, merge_m)) + sum_neighbor * __expf(safe_sub(m_neighbor, merge_m)), "warp_merge_sum");
            m_ = merge_m;

            m_neighbor = __shfl_down_sync(0xffffffff, m_, 1);
            sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 1);
            y_neighbor = __shfl_down_sync(0xffffffff, threadIdx.y, 1);
            assert(y_neighbor == threadIdx.y);
            merge_m = check_nan_val(max(m_, m_neighbor), "warp_merge_m");
            check_infinite_val(sum_neighbor, "sum_neighbor");
            sum_ = check_infinite_val(sum_ * __expf(safe_sub(m_, merge_m)) + sum_neighbor * __expf(safe_sub(m_neighbor, merge_m)), "warp_merge_sum");
            m_ = merge_m;

            if (threadIdx.x == 0) { // 复用 首列表达chunk m & sum
                tile_softmax_m[threadIdx.y * window_chunk_size + 0] = m_;
                tile_softmax_sum[threadIdx.y * window_chunk_size + 0] = sum_;
            }
        }

        __syncthreads();
        if (threadIdx.x < window_chunk_size) {
            scalar_t val = tile_softmax[threadIdx.y * window_chunk_size + threadIdx.x];
            tile_softmax[threadIdx.y * window_chunk_size + threadIdx.x] = check_nan_val(
                softmax_div(
                    __expf(safe_sub(val, tile_softmax_m[threadIdx.y * window_chunk_size + 0])), tile_softmax_sum[threadIdx.y * window_chunk_size + 0]),
                "softmax");
        }
        __syncthreads();
#ifdef DEBUG_NUMERIC
        if (batch_id == kDebugBatchId && head_id == kDebugHeadId &&
            q_token_id == kDebugQTokenId && chunk_id == kDebugChunkId &&
            threadIdx.x == 0) {
            for (int i = 0; i < min(window_chunk_size, kDebugWindowElems); ++i) {
                printf(
                    "kernel_prob batch=%lld head=%lld q=%lld idx=%d kv=%lld prob=%f\n",
                    (long long)batch_id,
                    (long long)head_id,
                    (long long)q_token_id,
                    i,
                    (long long)(absolute_range_left + chunk_id * window_chunk_size + i),
                    (float)tile_softmax[threadIdx.y * window_chunk_size + i]
                );
            }
        }
#endif


        // scalar_t old_sum = (q_token_id < q_seqlen) ? tile_old_softmax_sum[threadIdx.y]: scalar_t(0);
        // scalar_t new_sum = (q_token_id < q_seqlen) ? tile_softmax_sum[threadIdx.y * window_chunk_size] : scalar_t(0);
        // scalar_t old_m = (q_token_id < q_seqlen) ? tile_old_softmax_m[threadIdx.y] : neg_inf;
        // scalar_t new_m = (q_token_id < q_seqlen) ? tile_softmax_m[threadIdx.y * window_chunk_size] : neg_inf;
        scalar_t old_sum = tile_old_softmax_sum[threadIdx.y];
        scalar_t new_sum = tile_softmax_sum[threadIdx.y * window_chunk_size];
        scalar_t old_m = tile_old_softmax_m[threadIdx.y];
        scalar_t new_m = tile_softmax_m[threadIdx.y * window_chunk_size];
        scalar_t merge_m = (q_token_id < q_seqlen) ? static_cast<scalar_t>(max(new_m, old_m)) : neg_inf;
        scalar_t merge_sum = (q_token_id < q_seqlen) ? static_cast<scalar_t>(old_sum * __expf(safe_sub(old_m, merge_m)) 
                                                                            + new_sum * __expf(safe_sub(new_m, merge_m))) : scalar_t(0);
#ifdef DEBUG_NUMERIC
        if (threadIdx.x == 0 && q_token_id < q_seqlen) {
            printf(
                "state batch=%lld head=%lld q=%lld chunk=%d old_m=%f old_sum=%f new_m=%f new_sum=%f merge_m=%f merge_sum=%f range=[%lld,%lld)\n",
                (long long)batch_id,
                (long long)head_id,
                (long long)q_token_id,
                chunk_id,
                (float)old_m,
                (float)old_sum,
                (float)new_m,
                (float)new_sum,
                (float)merge_m,
                (float)merge_sum,
                (long long)absolute_range_left,
                (long long)absolute_range_right
            );
        }
#endif

        // tile matmul P @ V
        if (virt_kv_token_id < absolute_range_right && threadIdx.x < head_dim) {   // 转置读入，方便后续对位相乘
            int64_t phy_block_id = block_table[
                bt_stride0 * batch_id + 
                bt_stride1 * (virt_kv_token_id / block_size) // virt block id
            ];
            tile_VT[kv_seq_chunk_size * threadIdx.x + threadIdx.y] = 
                v[
                    v_stride0 * phy_block_id + 
                    v_stride1 * (virt_kv_token_id % block_size) + 
                    v_stride2 * head_id + 
                    v_stride3 * threadIdx.x
                ];
        } else {
            tile_VT[kv_seq_chunk_size * threadIdx.x + threadIdx.y] = scalar_t(0);
        }
        __syncthreads();

        for(int vt_tile_row_id = 0; vt_tile_row_id < head_dim; vt_tile_row_id++) {
            if (threadIdx.x < window_chunk_size) {
                scalar_t result = tile_softmax[threadIdx.y * window_chunk_size + threadIdx.x] *
                                                                                tile_VT[vt_tile_row_id * kv_seq_chunk_size + threadIdx.x];
                tile_dotPV[threadIdx.y * kv_seq_chunk_size + threadIdx.x] = result;                                                                
            }
            __syncthreads();
            for(int bound=blockDim.x/2; bound >= warpSize; bound /= 2) {
                if (threadIdx.x < bound) {
                    scalar_t raw = (threadIdx.x < kv_seq_chunk_size) ? 
                                        tile_dotPV[threadIdx.y * kv_seq_chunk_size + threadIdx.x] : scalar_t(0);
                    scalar_t raw_neighbor = (threadIdx.x + bound < kv_seq_chunk_size) ? 
                                        tile_dotPV[threadIdx.y * kv_seq_chunk_size + threadIdx.x + bound] : scalar_t(0);
                    if (threadIdx.x < kv_seq_chunk_size) {
                        tile_dotPV[threadIdx.y * kv_seq_chunk_size + threadIdx.x] = raw + raw_neighbor;
                    }

                }
                __syncthreads();
            }
            if (threadIdx.x < warpSize) {
                scalar_t result = (threadIdx.x < kv_seq_chunk_size) ? tile_dotPV[threadIdx.y * kv_seq_chunk_size + threadIdx.x] : scalar_t(0);
                result += __shfl_down_sync(0xffffffff, result, 16);
                result += __shfl_down_sync(0xffffffff, result, 8);
                result += __shfl_down_sync(0xffffffff, result, 4);
                result += __shfl_down_sync(0xffffffff, result, 2);
                result += __shfl_down_sync(0xffffffff, result, 1);

                if (threadIdx.x == 0) {
                    scalar_t e = tile_out[threadIdx.y * head_dim + vt_tile_row_id];
                    e = scalar_t(e * softmax_div(old_sum, merge_sum) * __expf(safe_sub(old_m, merge_m)) + result * softmax_div(new_sum, merge_sum) 
                        * __expf(safe_sub(new_m, merge_m)));
                    tile_out[threadIdx.y * head_dim + vt_tile_row_id] = e;
#ifdef DEBUG_NUMERIC
                    if (batch_id == kDebugBatchId && head_id == kDebugHeadId &&
                        q_token_id == kDebugQTokenId && chunk_id == kDebugChunkId &&
                        vt_tile_row_id < kDebugOutElems) {
                        printf(
                            "kernel_out batch=%lld head=%lld q=%lld dim=%d accum=%f\n",
                            (long long)batch_id,
                            (long long)head_id,
                            (long long)q_token_id,
                            vt_tile_row_id,
                            (float)e
                        );
                    }
#endif
                }
            }
        }
        if (threadIdx.x == 0) {
            tile_old_softmax_m[threadIdx.y] = merge_m;
            tile_old_softmax_sum[threadIdx.y] = merge_sum;
        }
        __syncthreads();
    }
    if (cu_seqlens_q[batch_id] + q_token_id < cu_seqlens_q[batch_id+1] && threadIdx.x < head_dim) {
        out[q_stride0 * (cu_seqlens_q[batch_id] + q_token_id) +
                q_stride1 * head_id + 
                q_stride2 * threadIdx.x] 
            = tile_out[threadIdx.y * head_dim + threadIdx.x];
    }
}

template<typename scalar_t>
torch::Tensor flash_attn_varlen_with_block_v2(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int64_t max_seqlen_q,
    torch::Tensor cu_seqlens_q,
    int64_t max_seqlen_k,
    torch::Tensor seqused_k,
    bool causal,
    int64_t window_size_left, 
    int64_t window_size_right,
    torch::Tensor block_table,
    torch::Tensor out
){    
    check_inputs(q, k, v, 
        max_seqlen_q, cu_seqlens_q,
        max_seqlen_k, seqused_k,
        causal,
        window_size_left, window_size_right,
        block_table, 
        out);
    int head_dim = q.size(2);
    auto batch_size = seqused_k.size(0);
    auto num_heads = q.size(1);

    assert(head_dim <= 128);        // 泛化

    int block_row = BLOCK_ROW_V2;
    dim3 block(head_dim, block_row);
    dim3 grid(1, 
            (max_seqlen_q + block_row - 1)/ block_row, 
            batch_size * num_heads
        );
    int tile_size = block.y * head_dim +
                    block.y * head_dim + 
                    block.y * block.y + 
                    block.y * block.y + 
                    block.y * block.y + 
                    head_dim * block.y +
                    block.y * head_dim +
                    block.y * 1 + 
                    block.y * 1 + 
                    block.y * head_dim +
                    block.y * block.y;

    flash_attn_varlen_with_block_kernel_v2<scalar_t><<<grid, block, sizeof(scalar_t) * tile_size>>>(
        q.const_data_ptr<scalar_t>(),
        q.stride(0), q.stride(1), q.stride(2),
        k.const_data_ptr<scalar_t>(),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.const_data_ptr<scalar_t>(),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        max_seqlen_q,
        cu_seqlens_q.const_data_ptr<int32_t>(),
        max_seqlen_k,
        seqused_k.const_data_ptr<int32_t>(),
        causal,
        window_size_left, window_size_right,
        block_table.const_data_ptr<int32_t>(),
        block_table.stride(0), block_table.stride(1),
        q.size(1),
        k.size(2),
        q.size(2),
        k.size(1),
        out.data_ptr<scalar_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_varlen_with_block", &flash_attn_varlen_with_block_v2<float>, "flash attn varlen with block");
}
