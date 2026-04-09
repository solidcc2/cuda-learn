#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <string>

#ifdef TOY_FLASH_ATTN_ASSERT_ON
#define toy_flash_attn_assert(x) assert(x)
#else
#define toy_flash_attn_assert(x) ((void)0)
#endif


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
    };

    struct TileLayout {
        static __device__ inline TileLayout builder(char* smem, 
                                                int64_t q_chunk_size,
                                                int64_t kv_chunk_size,
                                                int64_t head_dim) {
            TileLayout layout;
            int64_t offset = 0;
            layout.q = reinterpret_cast<scalar_t*>(smem + offset);
            offset += (q_chunk_size * head_dim) * sizeof(scalar_t);

            layout.k = reinterpret_cast<scalar_t*>(smem + offset);
            offset += (kv_chunk_size * head_dim) * sizeof(scalar_t);

            layout.v = reinterpret_cast<scalar_t*>(smem + offset);
            offset += (kv_chunk_size * head_dim) * sizeof(scalar_t);

            layout.out = reinterpret_cast<scalar_t*>(smem + offset);
            offset += (q_chunk_size * head_dim) * sizeof(scalar_t);
            
            // TODO
            // align

            layout.q_chunk_size = q_chunk_size;
            layout.kv_chunk_size = kv_chunk_size;
            layout.head_dim = head_dim;
            
            return layout;
        }

        __device__ inline scalar_t& q_idx(int64_t token_off, int64_t head_pos) {
            toy_flash_attn_assert(token_off >= 0 && token_off < q_chunk_size);
            toy_flash_attn_assert(head_pos >= 0 && head_pos < head_dim);
            return q[head_dim * token_off + head_pos];
        }
        __device__ inline scalar_t& k_idx(int64_t seq_off, int64_t head_pos) {
            toy_flash_attn_assert(seq_off >= 0 && seq_off < kv_chunk_size);
            toy_flash_attn_assert(head_pos >= 0 && head_pos < head_dim);
            return k[head_dim * seq_off + head_pos];
        }
        __device__ inline scalar_t& v_idx(int64_t seq_off, int64_t head_pos) {
            toy_flash_attn_assert(seq_off >= 0 && seq_off < kv_chunk_size);
            toy_flash_attn_assert(head_pos >= 0 && head_pos < head_dim);
            return v[head_dim * seq_off + head_pos];
        }
        __device__ inline scalar_t& out_idx(int64_t token_off, int64_t head_pos) {
            toy_flash_attn_assert(token_off >= 0 && token_off < q_chunk_size);
            toy_flash_attn_assert(head_pos >= 0 && head_pos < head_dim);
            return out[head_dim * token_off + head_pos];
        }
        
    private:
        __device__ TileLayout() = default;

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

        int64_t q_chunk_size;
        int64_t kv_chunk_size;
        int64_t head_dim;
    };
    static __global__ void kernel(ParamSet param) {
        extern __shared__ char smem[];
        TileLayout layout = TileLayout::builder(smem, threadIdx.y, threadIdx.y, param.head_dim);

        
    };
};