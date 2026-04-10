#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_math.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
// #include <cassert>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <string>
#include <limits>

# define TOY_FLASH_ATTN_ASSERT_ON
# define DEBUG_NUMERIC

#ifdef TOY_FLASH_ATTN_ASSERT_ON
#define toy_flash_attn_assert(x) assert(x)
#else
#define toy_flash_attn_assert(x) ((void)0)
#endif

template<typename scalar_t>
__device__ inline scalar_t check_nan_val(scalar_t x, const char* tag) {
#ifdef DEBUG_NUMERIC
    if (isnan(x)) {
        printf("%s nan: %f tx=%d ty=%d bx=%d by=%d bz=%d\n",
            tag, x, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z);
    }
#endif
    return x;
}

template<typename scalar_t>
__device__ inline scalar_t check_non_finite_val(scalar_t x, const char* tag) {
#ifdef DEBUG_NUMERIC
    if (!isfinite(x)) {
        printf("%s non-finite: %f tx=%d ty=%d bx=%d by=%d bz=%d\n",
               tag, x, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z);
    }
#endif
    return x;
}

__host__ __device__ inline uint32_t ceil_pow2_u32(uint32_t x) {
    if (x <= 1) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}
__host__ __device__ inline int64_t round_up(int64_t x, int64_t base) {
    return (x + base - 1) / base * base;
}

// grid(batch_id, q_token_chunk_id, head_id)
template<typename scalar_t, typename inner_scalar_t>
struct FlashAttnTrait {
    static const inner_scalar_t inner_neg_inf =
        -std::numeric_limits<inner_scalar_t>::infinity();

    struct ParamSet {
        scalar_t* q;    // total_q x num_heads x head_dim
        scalar_t* k;    // num_blocks x block_size x num_kv_heads x head_dim
        scalar_t* v;
        scalar_t* out;   // assert out_layout == q_layout
        union {
            int64_t q_stride[3];  
            int64_t out_stride[3];  
        };
        int64_t k_stride[4];
        int64_t v_stride[4];
        int64_t max_seqlen_q;
        int32_t* cu_seqlens_q;  // batch_size + 1
        int64_t max_seqlen_k;
        int32_t* seqused_k;     // batch_size
        int64_t window_size[2];
        int32_t* block_table;
        int64_t bt_stride[2];
        int64_t num_heads;
        int64_t head_dim;
        int64_t block_size;
        bool causal;
        int64_t q_chunk_size;
        int64_t kv_chunk_size;

        __device__ inline int64_t batch_size() const {
            return gridDim.x;
        }
        __device__ inline int64_t batch_id() const {
            return blockIdx.x;
        }
        __device__ inline int64_t head_id() const {
            return blockIdx.z;
        }
        __device__ inline int64_t q_seqlen() const {
            const int64_t bid = batch_id();
            return static_cast<int64_t>(cu_seqlens_q[bid+1] - cu_seqlens_q[bid]);
        }
        __device__ inline int64_t kv_seqlen() const {
            return static_cast<int64_t>(seqused_k[batch_id()]);
        }
        
        // q_token_id bound with threadIdx.y
        __device__ inline int64_t q_token_id() const {
            const int64_t idx = blockIdx.y * blockDim.y + threadIdx.y;
            return idx;
        }
        __device__ inline scalar_t q_at(int64_t q_token_id, int64_t head_pos) const {
            toy_flash_attn_assert(q_token_id >= 0 && q_token_id < q_seqlen());
            toy_flash_attn_assert(head_pos >= 0 && head_pos < head_dim);
            return q[
                q_stride[0] * (q_token_id + cu_seqlens_q[batch_id()]) + 
                q_stride[1] * head_id() + 
                q_stride[2] * head_pos
            ];
        }
        __device__ inline scalar_t& out_at(int64_t out_token_id, int64_t head_pos) const {
            toy_flash_attn_assert(out_token_id >= 0 && out_token_id < q_seqlen());
            toy_flash_attn_assert(head_pos >= 0 && head_pos < head_dim);
            return out[
                out_stride[0] * (out_token_id + cu_seqlens_q[batch_id()]) + 
                out_stride[1] * head_id() + 
                out_stride[2] * head_pos
            ];
        }
        __device__ inline scalar_t k_at(int64_t k_seq_id, int64_t head_pos) const {
            toy_flash_attn_assert(k_seq_id >= 0 && k_seq_id < kv_seqlen());
            toy_flash_attn_assert(head_pos >= 0 && head_pos < head_dim);
            toy_flash_attn_assert(block_size > 0);
            const int64_t virt_block_id = k_seq_id / block_size;
            const int64_t block_off = k_seq_id % block_size;
            const int32_t phy_block_id = block_table[
                bt_stride[0] * batch_id() + 
                bt_stride[1] * virt_block_id
            ];
            return k[
                k_stride[0] * phy_block_id + 
                k_stride[1] * block_off +
                k_stride[2] * head_id() + 
                k_stride[3] * head_pos
            ];
        }
        __device__ inline scalar_t v_at(int64_t v_seq_id, int64_t head_pos) const {
            toy_flash_attn_assert(v_seq_id >= 0 && v_seq_id < kv_seqlen());
            toy_flash_attn_assert(head_pos >= 0 && head_pos < head_dim);
            toy_flash_attn_assert(block_size > 0);
            const int64_t virt_block_id = v_seq_id / block_size;
            const int64_t block_off = v_seq_id % block_size;
            const int32_t phy_block_id = block_table[
                bt_stride[0] * batch_id() + 
                bt_stride[1] * virt_block_id
            ];
            return v[
                v_stride[0] * phy_block_id + 
                v_stride[1] * block_off +
                v_stride[2] * head_id() + 
                v_stride[3] * head_pos
            ];
        }
    };

    struct TileLayout {
        static __device__ __host__ inline TileLayout builder(char* smem, const ParamSet& param) {
            TileLayout layout(param);
            int64_t offset = 0;
            layout.q = reinterpret_cast<scalar_t*>(smem + offset);
            offset += (param.q_chunk_size * param.head_dim) * sizeof(scalar_t);

            offset = round_up(offset, alignof(scalar_t));
            layout.k = reinterpret_cast<scalar_t*>(smem + offset);
            offset += (param.kv_chunk_size * param.head_dim) * sizeof(scalar_t);

            offset = round_up(offset, alignof(scalar_t));
            layout.v = reinterpret_cast<scalar_t*>(smem + offset);
            offset += (param.kv_chunk_size * param.head_dim) * sizeof(scalar_t);

            offset = round_up(offset, alignof(scalar_t));
            layout.out = reinterpret_cast<scalar_t*>(smem + offset);
            offset += (param.q_chunk_size * param.head_dim) * sizeof(scalar_t);
            
            offset = round_up(offset, alignof(scalar_t));
            layout.qk_matmul_reduction = reinterpret_cast<scalar_t*>(smem + offset);
            offset += (param.q_chunk_size * param.head_dim) * sizeof(scalar_t);

            offset = round_up(offset, alignof(inner_scalar_t));
            layout.score_reduction = reinterpret_cast<inner_scalar_t*>(smem + offset);
            offset += (param.q_chunk_size * param.kv_chunk_size) * sizeof(inner_scalar_t);

            offset = round_up(offset, alignof(inner_scalar_t));
            layout.max_reduction = reinterpret_cast<inner_scalar_t*>(smem + offset);
            offset += (param.q_chunk_size * param.kv_chunk_size) * sizeof(inner_scalar_t);

            offset = round_up(offset, alignof(inner_scalar_t));
            layout.sum_reduction = reinterpret_cast<inner_scalar_t*>(smem + offset);
            offset += (param.q_chunk_size * param.kv_chunk_size) * sizeof(inner_scalar_t);
            
            offset = round_up(offset, alignof(inner_scalar_t));
            layout._size = offset;
            
            return layout;
        }

        static __host__ inline size_t size(const ParamSet& param) {
            TileLayout layout = builder((char*)nullptr, param);
            return layout._size;
        }

        __device__ inline scalar_t& q_at(int64_t token_off, int64_t head_pos) {
            toy_flash_attn_assert(token_off >= 0 && token_off < param.q_chunk_size);
            toy_flash_attn_assert(head_pos >= 0 && head_pos < param.head_dim);
            return q[param.head_dim * token_off + head_pos];
        }
        __device__ inline scalar_t& k_at(int64_t seq_off, int64_t head_pos) {
            toy_flash_attn_assert(seq_off >= 0 && seq_off < param.kv_chunk_size);
            toy_flash_attn_assert(head_pos >= 0 && head_pos < param.head_dim);
            return k[param.head_dim * seq_off + head_pos];
        }
        __device__ inline scalar_t& v_at(int64_t seq_off, int64_t head_pos) {
            toy_flash_attn_assert(seq_off >= 0 && seq_off < param.kv_chunk_size);
            toy_flash_attn_assert(head_pos >= 0 && head_pos < param.head_dim);
            return v[param.head_dim * seq_off + head_pos];
        }
        __device__ inline scalar_t& out_at(int64_t token_off, int64_t head_pos) {
            toy_flash_attn_assert(token_off >= 0 && token_off < param.q_chunk_size);
            toy_flash_attn_assert(head_pos >= 0 && head_pos < param.head_dim);
            return out[param.head_dim * token_off + head_pos];
        }

        // tile视角
        __device__ inline scalar_t& qk_matmul_reduction_at(int64_t token_off, int64_t head_pos) {
            toy_flash_attn_assert(token_off >= 0 && token_off < param.q_chunk_size);
            toy_flash_attn_assert(head_pos >= 0 && head_pos < param.head_dim);
            return qk_matmul_reduction[param.head_dim * token_off + head_pos];
        }
        // thread block视角，完成无效thread补齐
        __device__ inline scalar_t qk_matmul_reduction_block_at(int64_t y, int64_t x) {
            return (y < param.q_chunk_size && x < param.head_dim) ? 
                qk_matmul_reduction_at(y, x) : scalar_t(0);
        }

        __device__ inline inner_scalar_t& score_reduction_at(int64_t token_off, int64_t seq_off) {
            toy_flash_attn_assert(token_off >= 0 && token_off < param.q_chunk_size);
            toy_flash_attn_assert(seq_off >= 0 && seq_off < param.kv_chunk_size);
            return score_reduction[param.kv_chunk_size * token_off + seq_off];
        }
        __device__ inline inner_scalar_t& max_reduction_at(int64_t token_off, int64_t seq_off) {
            toy_flash_attn_assert(token_off >= 0 && token_off < param.q_chunk_size);
            toy_flash_attn_assert(seq_off >= 0 && seq_off < param.kv_chunk_size);
            return max_reduction[param.kv_chunk_size * token_off + seq_off];
        }
        __device__ inline inner_scalar_t& sum_reduction_at(int64_t token_off, int64_t seq_off) {
            toy_flash_attn_assert(token_off >= 0 && token_off < param.q_chunk_size);
            toy_flash_attn_assert(seq_off >= 0 && seq_off < param.kv_chunk_size);
            return sum_reduction[param.kv_chunk_size * token_off + seq_off];
        }

        __device__ inline bool is_valid_kv(int64_t q_token_id, int64_t kv_seq_id, int64_t q_kv_offset, const int64_t* kv_win) {
            const int64_t kv_axis = q_token_id - q_kv_offset;
            const int64_t win[2] = {
                max(int64_t(0), kv_axis - kv_win[0]),
                min(param.kv_seqlen()-1, kv_axis + kv_win[1]),
            };
            return kv_seq_id >= win[0] && kv_seq_id <= win[1];
        }
        __device__ inline  scalar_t q_in_tile(int64_t q_token_id, int64_t head_pos) {
            toy_flash_attn_assert(q_token_id / param.q_chunk_size == blockIdx.y);
            toy_flash_attn_assert(head_pos < param.head_dim);
            return q_at(q_token_id % param.q_chunk_size, head_pos);
        }
        __device__ inline scalar_t k_in_tile(int64_t k_seq_id, int64_t chunk_id, int64_t head_pos) {
            toy_flash_attn_assert(k_seq_id / param.kv_chunk_size == chunk_id);
            toy_flash_attn_assert(head_pos < param.head_dim);
            return k_at(k_seq_id % param.kv_chunk_size, head_pos);
        }
        
    private:
        __device__ __host__ TileLayout(const ParamSet& p):param(p) {}

        scalar_t* q;
        scalar_t* k;
        scalar_t* v;
        scalar_t* out;

        scalar_t* qk_matmul_reduction;

        // inner for softmax
        inner_scalar_t* score_reduction;
        inner_scalar_t* max_reduction;
        inner_scalar_t* sum_reduction;
        inner_scalar_t* softmax_reduction;
        inner_scalar_t* last_chunk_max;
        inner_scalar_t* last_chunk_sum;
        inner_scalar_t* last_chunk_softmax;

        size_t _size;

        const ParamSet& param;
    };
    static __global__ void kernel(ParamSet param) {
        extern __shared__ char smem[];
        TileLayout layout = TileLayout::builder(smem, param);
        // load q chunk
        toy_flash_attn_assert(param.q_chunk_size <= blockDim.y);
        toy_flash_attn_assert(param.kv_chunk_size <= blockDim.y);
        toy_flash_attn_assert(param.head_dim <= blockDim.x);
        toy_flash_attn_assert(blockDim.x >= warpSize);
        toy_flash_attn_assert(blockDim.x % warpSize == 0);
        toy_flash_attn_assert(param.kv_chunk_size <= blockDim.x);

        if (threadIdx.y < param.q_chunk_size && threadIdx.x < param.head_dim) {
            layout.q_at(threadIdx.y , threadIdx.x) = 
                (param.q_token_id() < param.q_seqlen()) ? param.q_at(param.q_token_id(), threadIdx.x) : scalar_t(0);

        }
        __syncthreads();
        const int64_t kv_win[2] = {
            param.window_size[0] == -1 ? param.kv_seqlen() : param.window_size[0],
            param.causal ? 0 : 
                param.window_size[1] == -1 ? param.kv_seqlen() : param.window_size[1]
        };
        const int64_t q_kv_offset = param.q_seqlen() - param.kv_seqlen();

        for(int64_t kv_chunk_id = 0; 
                kv_chunk_id < (param.kv_seqlen() + param.kv_chunk_size - 1) / param.kv_chunk_size; kv_chunk_id++) {
            {   // kv_seq_id 作用域 for k tile load
                int64_t kv_seq_id = kv_chunk_id * param.kv_chunk_size + threadIdx.y;
                if (threadIdx.y < param.kv_chunk_size && threadIdx.x < param.head_dim) {
                    layout.k_at(threadIdx.y , threadIdx.x) = 
                        (kv_seq_id < param.kv_seqlen()) ? param.k_at(kv_seq_id, threadIdx.x) : scalar_t(0);
                }
                __syncthreads();
            }

            // Q K matmul

            for(int kv_chunk_off = 0; kv_chunk_off <param.kv_chunk_size; kv_chunk_off ++) {
                int64_t kv_seq_id = kv_chunk_id * param.kv_chunk_size + kv_chunk_off;
                if (threadIdx.y < param.q_chunk_size && threadIdx.x < param.head_dim) {
                    if (param.q_token_id() < param.q_seqlen()) {
                        layout.qk_matmul_reduction_at(threadIdx.y, threadIdx.x) = check_non_finite_val(
                            layout.is_valid_kv(param.q_token_id(), kv_seq_id, q_kv_offset, kv_win) ?
                                    layout.q_in_tile(param.q_token_id(), threadIdx.x) *
                                    layout.k_in_tile(kv_seq_id, kv_chunk_id, threadIdx.x) 
                                : scalar_t(0),
                        "qk_dot");
                    } else {
                        layout.qk_matmul_reduction_at(threadIdx.y, threadIdx.x) = scalar_t(0);
                    }
                }
                
                do {     // while(0) 范围控制 line sum reduction
                    scalar_t val = layout.qk_matmul_reduction_block_at(threadIdx.y, threadIdx.x);
                    // block归约
                    int32_t bound = ceil_pow2_u32(param.head_dim) / 2;
                    for(; bound >= warpSize; bound /= 2) {
                        if (threadIdx.x < bound && threadIdx.y < param.q_chunk_size) {
                            scalar_t neighbor = layout.qk_matmul_reduction_block_at(threadIdx.y, threadIdx.x + bound);
                            val = check_non_finite_val(val+neighbor, "qk_block_reduction");
                            layout.qk_matmul_reduction_at(threadIdx.y, threadIdx.x) = val;
                        }
                        __syncthreads();
                    }
                    if (threadIdx.x >= warpSize || threadIdx.y >= param.q_chunk_size) break;

                    val = check_non_finite_val(val + __shfl_down_sync(0xffffffff, val, 16), "qk_warp_reduction");
                    val = check_non_finite_val(val + __shfl_down_sync(0xffffffff, val, 8), "qk_warp_reduction");
                    val = check_non_finite_val(val + __shfl_down_sync(0xffffffff, val, 4), "qk_warp_reduction");
                    val = check_non_finite_val(val + __shfl_down_sync(0xffffffff, val, 2), "qk_warp_reduction");
                    val = check_non_finite_val(val + __shfl_down_sync(0xffffffff, val, 1), "qk_warp_reduction");
                    if (threadIdx.x == 0) {
                        layout.score_reduction_at(threadIdx.y, kv_chunk_off) = 
                            check_non_finite_val(static_cast<inner_scalar_t>(val) / sqrt((double)param.head_dim), "score");
                    }
                } while(0);
            }
            {
                // online softmax
                bool valid_thread = threadIdx.y < param.q_chunk_size && threadIdx.x < param.kv_chunk_size;
                if(valid_thread) {     // load tile
                    int64_t kv_seq_id = kv_chunk_id * param.kv_chunk_size + threadIdx.x;
                    bool is_valid_kv = layout.is_valid_kv(param.q_token_id(), kv_seq_id, q_kv_offset, kv_win);

                    layout.max_reduction_at(threadIdx.y, threadIdx.x) = 
                        is_valid_kv ? layout.score_reduction_at(threadIdx.y, threadIdx.x) : inner_neg_inf;
                    layout.sum_reduction_at(threadIdx.y, threadIdx.x) = 
                        is_valid_kv ? inner_scalar_t(1.0) : inner_scalar_t(0.0);
                }
                __syncthreads();

                inner_scalar_t max_ = valid_thread ?
                    layout.max_reduction_at(threadIdx.y, threadIdx.x) : inner_neg_inf;
                inner_scalar_t sum_ = valid_thread ?
                    layout.sum_reduction_at(threadIdx.y, threadIdx.x) : inner_scalar_t(0);
                int32_t bound = ceil_pow2_u32(param.kv_chunk_size) / 2;
                for(; bound >= warpSize; bound /= 2) {  // block reduction
                    if (threadIdx.x < bound && threadIdx.y < param.q_chunk_size) {
                        bool valid_neighbor = threadIdx.x + bound < param.q_chunk_size && threadIdx.y < param.q_chunk_size;
                        inner_scalar_t max_neighbor = valid_neighbor ?
                            layout.max_reduction_at(threadIdx.y, threadIdx.x + bound) : inner_neg_inf;
                        inner_scalar_t sum_neighbor = valid_neighbor ? 
                            layout.sum_reduction_at(threadIdx.y, threadIdx.x + bound) : inner_scalar_t(0.0);
                        inner_scalar_t max_merge = max(max_, max_neighbor);
                        inner_scalar_t sum_merge = check_non_finite_val(sum_ * __expf(check_nan_val(max_ - max_merge, "softmax_sum_reduction_block_old")) + 
                                                sum_neighbor * __expf(check_nan_val(max_neighbor - max_merge, "softmax_sum_reduction_block_new")), 
                                                "softmax_sum_reduction_block");
                        max_ = max_merge;
                        sum_ = sum_merge;
                        layout.max_reduction_at(threadIdx.y, threadIdx.x) = max_;
                        layout.sum_reduction_at(threadIdx.y, threadIdx.x) = sum_;
                    }
                    __syncthreads();
                }
                if (threadIdx.x < warpSize) {
                    
                }

            }

        }
    };
}; 