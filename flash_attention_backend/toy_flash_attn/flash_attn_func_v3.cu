#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_math.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <string>

#ifdef TOY_FLASH_ATTN_ASSERT_ON
#define toy_flash_attn_assert(x) assert(x)
#else
#define toy_flash_attn_assert(x) ((void)0)
#endif

// grid(batch_id, q_token_chunk_id, head_id)
template<typename scalar_t, typename inner_scalar_t>
struct FlashAttnTrait {
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
        static __device__ inline TileLayout builder(char* smem, const ParamSet& param) {
            TileLayout layout(param);
            int64_t offset = 0;
            layout.q = reinterpret_cast<scalar_t*>(smem + offset);
            offset += (param.q_chunk_size * param.head_dim) * sizeof(scalar_t);

            layout.k = reinterpret_cast<scalar_t*>(smem + offset);
            offset += (param.kv_chunk_size * param.head_dim) * sizeof(scalar_t);

            layout.v = reinterpret_cast<scalar_t*>(smem + offset);
            offset += (param.kv_chunk_size * param.head_dim) * sizeof(scalar_t);

            layout.out = reinterpret_cast<scalar_t*>(smem + offset);
            offset += (param.q_chunk_size * param.head_dim) * sizeof(scalar_t);
            
            // TODO
            // align
            
            return layout;
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
        __device__ inline bool is_valid_kv(int64_t q_token_id, int64_t kv_seq_id, int64_t q_kv_offset, const int64_t* kv_win) {
            const int64_t kv_axis = q_token_id - q_kv_offset;
            const int64_t win[2] = {
                max(int64_t(0), kv_axis - kv_win[0]),
                min(param.kv_seqlen()-1, kv_axis + kv_win[1]),
            };
            return kv_seq_id >= win[0] && kv_seq_id <= win[1];
        }
        
    private:
        __device__ TileLayout(const ParamSet& p):param(p) {}

        scalar_t* q;
        scalar_t* k;
        scalar_t* v;
        scalar_t* out;

        // inner for softmax
        inner_scalar_t* score_reduction;
        inner_scalar_t* max_reduction;
        inner_scalar_t* sum_reduction;
        inner_scalar_t* softmax_reduction;
        inner_scalar_t* last_chunk_max;
        inner_scalar_t* last_chunk_sum;
        inner_scalar_t* last_chunk_softmax;

        const ParamSet& param;
    };
    static __global__ void kernel(ParamSet param) {
        extern __shared__ char smem[];
        TileLayout layout = TileLayout::builder(smem, param);
        // load q chunk
        toy_flash_attn_assert(param.q_chunk_size <= blockDim.y);
        toy_flash_attn_assert(param.head_dim <= blockDim.x);

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
            
            int64_t kv_seq_id = kv_chunk_id * param.kv_chunk_size + threadIdx.y;
            if (threadIdx.y < param.kv_chunk_size && threadIdx.x < param.head_dim) {
                layout.k_at(threadIdx.y , threadIdx.x) = 
                    (kv_seq_id < param.kv_seqlen()) ? param.k_at(kv_seq_id, threadIdx.x) : scalar_t(0);
            }
            __syncthreads();

            // Q K matmul

            
        }
    };
};