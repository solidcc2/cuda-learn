
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <string>
#include <limits>

#include "helper.h"

template<typename scalar_t, typename inner_scalar_t>
struct FlashAttnTrait;

template<typename scalar_t, typename inner_scalar_t>
__global__ void kernel_wrapper(
    typename FlashAttnTrait<scalar_t, inner_scalar_t>::ParamSet param) {
    FlashAttnTrait<scalar_t, inner_scalar_t>::kernel(param);
}

// grid(batch_id, q_token_chunk_id, head_id)
template<typename scalar_t, typename inner_scalar_t>
struct FlashAttnTrait {
    // static const int32_t K_X_STRIDE=16;  // x 维度每线程处理16个数
    // static const int32_t Q_CHUNK_SIZE = 8;
    // static const int32_t KV_CHUNK_SIZE = 64;
    // static const int32_t BLOCK_Y = 64;

    static const int32_t K_X_STRIDE=4;  // x 维度每线程处理16个数
    static const int32_t Q_CHUNK_SIZE = 8;
    static const int32_t KV_CHUNK_SIZE = 8;
    static const int32_t BLOCK_Y = 8;

    struct ParamSet;
    struct TileLayout;
    static __device__ void kernel(ParamSet& param);
    static void check_inputs(
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
    );
    static torch::Tensor flash_attn_varlen_with_block(
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
    );
};

// grid(batch_id, q_token_chunk_id, head_id)
template<typename scalar_t, typename inner_scalar_t>
struct FlashAttnTrait<scalar_t, inner_scalar_t>::ParamSet {
    const scalar_t* q;    // total_q x num_heads x head_dim
    const scalar_t* k;    // num_blocks x block_size x num_kv_heads x head_dim
    const scalar_t* v;
    scalar_t* out;   // assert out_layout == q_layout
    union {
        int64_t q_stride[3];  
        int64_t out_stride[3];  
    };
    int64_t k_stride[4];
    int64_t v_stride[4];
    int64_t max_seqlen_q;
    const int32_t* cu_seqlens_q;  // batch_size + 1
    int64_t max_seqlen_k;
    const int32_t* seqused_k;     // batch_size
    int64_t window_size[2];
    const int32_t* block_table;
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
        const int64_t idx = blockIdx.y * q_chunk_size + threadIdx.y;
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
template<typename scalar_t, typename inner_scalar_t>
struct FlashAttnTrait<scalar_t, inner_scalar_t>::TileLayout {
    static __device__ inline TileLayout builder(char* smem, const ParamSet& param) {
        const int64_t  head_dim_thread_count = (param.head_dim + K_X_STRIDE - 1) / K_X_STRIDE;
        auto layout = layout_init(smem, param);
        toy_flash_attn_assert(smem != nullptr);
        bool valid_thread = threadIdx.y < param.q_chunk_size && threadIdx.x < head_dim_thread_count;
        if (valid_thread) {
            int64_t base = threadIdx.x * K_X_STRIDE;
            #pragma unroll
            for(int64_t lane=0; lane < K_X_STRIDE; lane++) {
                int64_t head_pos = base + lane;
                if (head_pos < param.head_dim) {
                    layout.out_at(threadIdx.y, head_pos) = inner_scalar_t(0);
                }
            }
            if (threadIdx.x == 0) {
                layout.last_chunk_max_at(threadIdx.y) = inner_scalar_t(-INFINITY);
                layout.last_chunk_sum_at(threadIdx.y) = inner_scalar_t(0);
            }
        }
        return layout;
    }

    static __host__ __device__ inline TileLayout layout_init(char* smem, const ParamSet& param) {
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

        offset = round_up(offset, alignof(inner_scalar_t));
        layout.out = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (param.q_chunk_size * param.head_dim) * sizeof(inner_scalar_t);
        
        offset = round_up(offset, alignof(inner_scalar_t));
        layout.qk_matmul_reduction = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (param.q_chunk_size * param.head_dim) * sizeof(inner_scalar_t);

        offset = round_up(offset, alignof(inner_scalar_t));
        layout.sv_matmul_reduction = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (param.q_chunk_size * param.kv_chunk_size) * sizeof(inner_scalar_t);

        offset = round_up(offset, alignof(inner_scalar_t));
        layout.out_reduction = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (param.q_chunk_size * param.head_dim) * sizeof(inner_scalar_t);

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
        layout.softmax_reduction = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (param.q_chunk_size * param.kv_chunk_size) * sizeof(inner_scalar_t);

        offset = round_up(offset, alignof(inner_scalar_t));
        layout.last_chunk_max = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (param.q_chunk_size * 1) * sizeof(inner_scalar_t);
        
        offset = round_up(offset, alignof(inner_scalar_t));
        layout.last_chunk_sum = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (param.q_chunk_size * 1) * sizeof(inner_scalar_t);

        offset = round_up(offset, alignof(inner_scalar_t));
        layout._size = offset;
        return layout;
    }

    static __host__ inline size_t size(const ParamSet& param) {
        TileLayout layout = layout_init((char*)nullptr, param);
        return layout._size;
    }

    // 这组accessor有大量代码冗余，但是考虑清晰，且重复的非常明显，当前复杂度还不是集中在这里，不考虑过度抽象减少冗余
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

    // v 提供转置读取接口，方便和softmax结果对齐，都是行对位相乘，列统一为kv_chunk_off, 内存布局更友好
    __device__ inline scalar_t& v_t_at(int64_t head_pos, int64_t seq_off) {
        toy_flash_attn_assert(seq_off >= 0 && seq_off < param.kv_chunk_size);
        toy_flash_attn_assert(head_pos >= 0 && head_pos < param.head_dim);
        return v[param.kv_chunk_size * head_pos + seq_off];
    }
    __device__ inline inner_scalar_t& out_at(int64_t token_off, int64_t head_pos) {
        toy_flash_attn_assert(token_off >= 0 && token_off < param.q_chunk_size);
        toy_flash_attn_assert(head_pos >= 0 && head_pos < param.head_dim);
        return out[param.head_dim * token_off + head_pos];
    }

    __device__ inline inner_scalar_t& qk_matmul_reduction_at(int64_t token_off, int64_t head_pos) {
        toy_flash_attn_assert(token_off >= 0 && token_off < param.q_chunk_size);
        toy_flash_attn_assert(head_pos >= 0 && head_pos < param.head_dim);
        return qk_matmul_reduction[param.head_dim * token_off + head_pos];
    }

    __device__ inline inner_scalar_t& sv_matmul_reduction_at(int64_t token_off, int64_t seq_off) {
        toy_flash_attn_assert(token_off >= 0 && token_off < param.q_chunk_size);
        toy_flash_attn_assert(seq_off >= 0 && seq_off < param.kv_chunk_size);
        return sv_matmul_reduction[param.kv_chunk_size * token_off + seq_off];
    }

    __device__ inline inner_scalar_t& out_reduction_at(int64_t token_off, int64_t head_pos) {
        toy_flash_attn_assert(token_off >= 0 && token_off < param.q_chunk_size);
        toy_flash_attn_assert(head_pos >= 0 && head_pos < param.head_dim);
        return out_reduction[param.head_dim * token_off + head_pos];
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
    __device__ inline inner_scalar_t& softmax_reduction_at(int64_t token_off, int64_t seq_off) {
        toy_flash_attn_assert(token_off >= 0 && token_off < param.q_chunk_size);
        toy_flash_attn_assert(seq_off >= 0 && seq_off < param.kv_chunk_size);
        return softmax_reduction[param.kv_chunk_size * token_off + seq_off];
    }

    __device__ inline inner_scalar_t& last_chunk_max_at(int64_t token_off) {
        toy_flash_attn_assert(token_off >= 0 && token_off < param.q_chunk_size);
        return last_chunk_max[token_off];
    }

    __device__ inline inner_scalar_t& last_chunk_sum_at(int64_t token_off) {
        toy_flash_attn_assert(token_off >= 0 && token_off < param.q_chunk_size);
        return last_chunk_sum[token_off];
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
    inner_scalar_t* out;

    inner_scalar_t* qk_matmul_reduction;
    inner_scalar_t* sv_matmul_reduction;
    inner_scalar_t* out_reduction;

    // inner for softmax
    inner_scalar_t* score_reduction;
    inner_scalar_t* max_reduction;
    inner_scalar_t* sum_reduction;
    inner_scalar_t* softmax_reduction;  // 理论上可以和score_reduction复用，但是性能差别不大，这部分shared占用不是瓶颈，先分开使逻辑清晰
    inner_scalar_t* last_chunk_max;
    inner_scalar_t* last_chunk_sum;

    size_t _size;

    const ParamSet& param;
};
template<typename scalar_t, typename inner_scalar_t>
__device__ void FlashAttnTrait<scalar_t, inner_scalar_t>::kernel(ParamSet& param) {
    extern __shared__ char smem[];
    TileLayout layout = TileLayout::builder(smem, param);
#ifdef DEBUG_FLASH_ATTN_V3_TRACE
    const bool debug_row =
        param.batch_id() == 0 &&
        param.head_id() == 0 &&
        param.q_token_id() < 2;
#endif
    const int64_t head_dim_thread_count = (param.head_dim + K_X_STRIDE - 1) / K_X_STRIDE;
    const int64_t kv_chunk_size_thread_count = (param.kv_chunk_size + K_X_STRIDE - 1) / K_X_STRIDE;
    toy_flash_attn_assert(param.q_chunk_size <= blockDim.y);
    toy_flash_attn_assert(param.kv_chunk_size <= blockDim.y);
    toy_flash_attn_assert(head_dim_thread_count <= blockDim.x);
    toy_flash_attn_assert(kv_chunk_size_thread_count <= blockDim.x);
    toy_flash_attn_assert((blockDim.x & (blockDim.x - 1)) == 0);  // 2 的幂
    toy_flash_attn_assert(param.q_chunk_size * blockDim.x % warpSize == 0);


    const inner_scalar_t softmax_scale_log2 = M_LOG2Ef32 / sqrt((double)param.head_dim);

    const int64_t kv_win[2] = {
        param.window_size[0] == -1 ? param.kv_seqlen() : param.window_size[0],
        param.causal ? 0 : 
            param.window_size[1] == -1 ? param.kv_seqlen() : param.window_size[1]
    };

    const int64_t q_kv_offset = param.q_seqlen() - param.kv_seqlen();
    const int64_t q_chunk_first = blockIdx.y * param.q_chunk_size;
    const int64_t q_chunk_last = min(q_chunk_first + param.q_chunk_size, param.q_seqlen()) - 1;
    int64_t kv_seq_begin = 0;
    int64_t kv_seq_end = 0;
    if (q_chunk_first <= q_chunk_last) {
        // Per q-chunk we first compute the union of all reachable KV positions,
        // then only visit the overlapping kv chunks. Fine-grained masking stays below.
        kv_seq_begin = max(int64_t(0), q_chunk_first - q_kv_offset - kv_win[0]);
        kv_seq_end = min(param.kv_seqlen(), q_chunk_last - q_kv_offset + 1 + kv_win[1]);
        if (!param.causal && param.window_size[0] == -1 && param.window_size[1] == -1) {
            kv_seq_begin = 0;
            kv_seq_end = param.kv_seqlen();
        }
    }
    const int64_t kv_chunk_begin = kv_seq_begin / param.kv_chunk_size;
    const int64_t kv_chunk_end = (kv_seq_end + param.kv_chunk_size - 1) / param.kv_chunk_size;
    {   // load q chunk 
        bool valid_thread = threadIdx.y < param.q_chunk_size && threadIdx.x < head_dim_thread_count;
        if (valid_thread) {
            int64_t base = threadIdx.x * K_X_STRIDE;
            #pragma unroll
            for (int64_t lane = 0; lane < K_X_STRIDE; ++lane) {
                int64_t head_pos = base + lane;
                if (head_pos < param.head_dim) {
                    layout.q_at(threadIdx.y , head_pos) = 
                        (param.q_token_id() < param.q_seqlen()) ? param.q_at(param.q_token_id(), head_pos) : scalar_t(0);
                }
            }

        }
        __syncthreads();
    }

    for(int64_t kv_chunk_id = kv_chunk_end - 1;
            kv_chunk_id >= kv_chunk_begin; kv_chunk_id--) {
        {   // kv_seq_id 作用域 for k tile load
            int64_t kv_seq_id = kv_chunk_id * param.kv_chunk_size + threadIdx.y;
            bool valid_thread = threadIdx.y < param.kv_chunk_size && threadIdx.x < head_dim_thread_count;
            if (valid_thread) {
                int64_t base = threadIdx.x * K_X_STRIDE;
                #pragma unroll 
                for(int64_t lane = 0; lane < K_X_STRIDE; ++lane) {
                    int64_t head_pos = base + lane;
                    if (head_pos < param.head_dim) {
                        layout.k_at(threadIdx.y , head_pos) = 
                            (kv_seq_id < param.kv_seqlen()) ? param.k_at(kv_seq_id, head_pos) : scalar_t(0);
                    }
                }
            }
            __syncthreads();
        }

        // Q K matmul
        for(int64_t kv_chunk_off = param.kv_chunk_size - 1; kv_chunk_off >= 0; kv_chunk_off--) {
            int64_t kv_seq_id = kv_chunk_id * param.kv_chunk_size + kv_chunk_off;
            {   // QK 对位乘
                bool valid_thread = threadIdx.y < param.q_chunk_size && threadIdx.x < head_dim_thread_count;
                if (valid_thread) {
                    int64_t base = threadIdx.x * K_X_STRIDE;
                    #pragma unroll
                    for (int64_t lane = 0; lane < K_X_STRIDE; lane++) {
                        int64_t head_pos = base + lane;
                        if (head_pos >= param.head_dim) continue;
                        if (param.q_token_id() < param.q_seqlen()) {
                            inner_scalar_t q_elem = layout.q_in_tile(param.q_token_id(), head_pos);
                            inner_scalar_t k_elem = layout.k_in_tile(kv_seq_id, kv_chunk_id, head_pos);
                            if (threadIdx.y == 0 && threadIdx.x == 0 && head_pos == 0) {
                                printf(
                                    "(%u, %u, %u) q=%lld kv_chunk=%lld kv_off=%lld kv_seq=%lld q_elem[0][0]: %f\n",
                                    (unsigned)blockIdx.x,
                                    (unsigned)blockIdx.y,
                                    (unsigned)blockIdx.z,
                                    (long long)param.q_token_id(),
                                    (long long)kv_chunk_id,
                                    (long long)kv_chunk_off,
                                    (long long)kv_seq_id,
                                    (double)q_elem
                                );
                                printf(
                                    "(%u, %u, %u) q=%lld kv_chunk=%lld kv_off=%lld kv_seq=%lld k_elem[0][0]: %f\n",
                                    (unsigned)blockIdx.x,
                                    (unsigned)blockIdx.y,
                                    (unsigned)blockIdx.z,
                                    (long long)param.q_token_id(),
                                    (long long)kv_chunk_id,
                                    (long long)kv_chunk_off,
                                    (long long)kv_seq_id,
                                    (double)k_elem
                                );
                            }
                            inner_scalar_t qk_elem = q_elem * k_elem;
                            layout.qk_matmul_reduction_at(threadIdx.y, head_pos) = 
                                check_non_finite_val(inner_scalar_t(qk_elem), "qk_dot");    // 乘在bf16, 加在fp32
                        } else {
                            layout.qk_matmul_reduction_at(threadIdx.y, head_pos) = inner_scalar_t(0);
                        }
                    }
                }
            }
            __syncthreads();
            if (threadIdx.y==0 && threadIdx.x == 0) {
                printf(
                    "(%u, %u, %u) q=%lld kv_chunk=%lld kv_off=%lld kv_seq=%lld QK dot[0][0]: %f\n",
                    (unsigned)blockIdx.x,
                    (unsigned)blockIdx.y,
                    (unsigned)blockIdx.z,
                    (long long)param.q_token_id(),
                    (long long)kv_chunk_id,
                    (long long)kv_chunk_off,
                    (long long)kv_seq_id,
                    (double)layout.qk_matmul_reduction_at(0, 0)
                );
            }
            
            do {     // while(0) 范围控制 line sum reduction
                bool valid_thread = threadIdx.y < param.q_chunk_size && threadIdx.x < head_dim_thread_count;
                int64_t base = threadIdx.x * K_X_STRIDE;
                inner_scalar_t stride_sum = inner_scalar_t(0);
                #pragma unroll
                for(int64_t lane=0; lane < K_X_STRIDE; lane++) {
                    int64_t head_pos = base + lane;
                    if (head_pos >= param.head_dim) continue;
                    stride_sum += valid_thread ? layout.qk_matmul_reduction_at(threadIdx.y, head_pos) : inner_scalar_t(0);
                }
                inner_scalar_t val = stride_sum;
                if (valid_thread) {
                    layout.qk_matmul_reduction_at(threadIdx.y, base) = stride_sum;
                }

                // block归约
                int32_t bound = ceil_pow2_u32(head_dim_thread_count) / 2;
                for(; bound >= warpSize; bound /= 2) {
                    if (threadIdx.x < bound && threadIdx.y < param.q_chunk_size) {
                        bool valid_pos = threadIdx.y < param.q_chunk_size && threadIdx.x + bound < head_dim_thread_count;
                        int64_t neighbor_pos = (threadIdx.x + bound) * K_X_STRIDE;
                        inner_scalar_t neighbor = valid_pos ? layout.qk_matmul_reduction_at(threadIdx.y, neighbor_pos) : inner_scalar_t(0);
                        val = check_non_finite_val(val+neighbor, "qk_block_reduction");
                        layout.qk_matmul_reduction_at(threadIdx.y, base) = val;
                    }
                    __syncthreads();
                }
                if (threadIdx.x >= warpSize || threadIdx.y >= param.q_chunk_size) break;
                int32_t subgroup = (head_dim_thread_count < warpSize) ? ceil_pow2_u32(head_dim_thread_count) : warpSize;
                switch (subgroup) {
                    case 32:
                        val = check_non_finite_val(val + __shfl_down_sync(0xffffffff, val, 16, subgroup), "qk_warp_reduction");
                        [[fallthrough]];    
                    case 16:
                        val = check_non_finite_val(val + __shfl_down_sync(0xffffffff, val, 8, subgroup), "qk_warp_reduction");
                        [[fallthrough]];    
                    case 8:
                        val = check_non_finite_val(val + __shfl_down_sync(0xffffffff, val, 4, subgroup), "qk_warp_reduction");
                        [[fallthrough]];    
                    case 4:
                        val = check_non_finite_val(val + __shfl_down_sync(0xffffffff, val, 2, subgroup), "qk_warp_reduction");
                        [[fallthrough]];    
                    case 2:
                        val = check_non_finite_val(val + __shfl_down_sync(0xffffffff, val, 1, subgroup), "qk_warp_reduction");
                        [[fallthrough]];    
                    case 1:
                        break;
                    default:
                        toy_flash_attn_assert(false);
                }

                if (threadIdx.x == 0) {
                    bool valid_q_kv_pair = 
                        param.q_token_id() < param.q_seqlen() && layout.is_valid_kv(param.q_token_id(), kv_seq_id, q_kv_offset, kv_win);

#ifdef DEBUG_FLASH_ATTN_V3_TRACE
                    if (debug_row && threadIdx.y < param.q_chunk_size) {
                        printf(
                            "score_mask bid=%lld hid=%lld q=%lld kv=%lld qlen=%lld kvlen=%lld off=%lld "
                            "win=(%lld,%lld) valid=%d val=%f\n",
                            (long long)param.batch_id(),
                            (long long)param.head_id(),
                            (long long)param.q_token_id(),
                            (long long)kv_seq_id,
                            (long long)param.q_seqlen(),
                            (long long)param.kv_seqlen(),
                            (long long)q_kv_offset,
                            (long long)kv_win[0],
                            (long long)kv_win[1],
                            (int)valid_q_kv_pair,
                            (double)val
                        );
                    }
#endif
                    layout.score_reduction_at(threadIdx.y, kv_chunk_off) = valid_q_kv_pair ? 
                        check_non_finite_val(val, "score") : inner_scalar_t(-INFINITY);
                }
            } while(0);
        }
        if (threadIdx.y==0 && threadIdx.x == 0) {
            printf(
                "(%u, %u, %u) q=%lld kv_chunk=%lld score[0][0]: %f\n",
                (unsigned)blockIdx.x,
                (unsigned)blockIdx.y,
                (unsigned)blockIdx.z,
                (long long)param.q_token_id(),
                (long long)kv_chunk_id,
                (double)layout.score_reduction_at(0, 0)
            );
        }
        {   // online softmax
            bool valid_thread = threadIdx.y < param.q_chunk_size && threadIdx.x < kv_chunk_size_thread_count;
            int64_t base = threadIdx.x * K_X_STRIDE;
            if(valid_thread) {     // load tile
                #pragma unroll
                for(int64_t lane = 0; lane < K_X_STRIDE; lane++) {
                    int64_t seq_off = base + lane;
                    if (seq_off < param.kv_chunk_size) {
                        layout.max_reduction_at(threadIdx.y, seq_off) = 
                            layout.score_reduction_at(threadIdx.y, seq_off);
                        layout.sum_reduction_at(threadIdx.y, seq_off) = 
                            layout.score_reduction_at(threadIdx.y, seq_off) != inner_scalar_t(-INFINITY) ?
                                    inner_scalar_t(1.0) : inner_scalar_t(0.0);
                    }
                }
            }
            __syncthreads();

            inner_scalar_t max_ = inner_scalar_t(-INFINITY);
            inner_scalar_t sum_ = inner_scalar_t(0);
            #pragma unroll
            for(int64_t lane=0; lane <K_X_STRIDE; lane++) {
                int64_t seq_off = base + lane;
                if (seq_off < param.kv_chunk_size) {
                    inner_scalar_t max_lane = valid_thread ?
                        layout.max_reduction_at(threadIdx.y, seq_off) : inner_scalar_t(-INFINITY);
                    inner_scalar_t sum_lane = valid_thread ?
                        layout.sum_reduction_at(threadIdx.y, seq_off) : inner_scalar_t(0);
                    inner_scalar_t max_merge = max(max_lane, max_);
                    sum_ = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_lane_old")) + 
                                            sum_lane * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_lane, max_merge), "softmax_sum_reduction_lane_new")), 
                                            "softmax_sum_reduction_lane");
                    max_ = max_merge;
                }
            }
            if (valid_thread) {
                layout.max_reduction_at(threadIdx.y, base) = max_;
                layout.sum_reduction_at(threadIdx.y, base) = sum_;
            }
            __syncthreads();

            int32_t bound = ceil_pow2_u32(kv_chunk_size_thread_count) / 2;
            for(; bound >= warpSize; bound /= 2) {  // block reduction
                if (threadIdx.x < bound && threadIdx.y < param.q_chunk_size) {
                    bool valid_neighbor = threadIdx.x + bound < kv_chunk_size_thread_count && threadIdx.y < param.q_chunk_size;
                    int64_t neighbor_base = (threadIdx.x + bound) * K_X_STRIDE;
                    inner_scalar_t max_neighbor = valid_neighbor ?
                        layout.max_reduction_at(threadIdx.y, neighbor_base) : inner_scalar_t(-INFINITY);
                    inner_scalar_t sum_neighbor = valid_neighbor ? 
                        layout.sum_reduction_at(threadIdx.y, neighbor_base) : inner_scalar_t(0.0);
                    inner_scalar_t max_merge = max(max_, max_neighbor);
                    inner_scalar_t sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_block_old")) + 
                                            sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_block_new")), 
                                            "softmax_sum_reduction_block");
                    max_ = max_merge;
                    sum_ = sum_merge;
                    layout.max_reduction_at(threadIdx.y, base) = max_;
                    layout.sum_reduction_at(threadIdx.y, base) = sum_;
                }
                __syncthreads();
            }
            if (threadIdx.x < warpSize && threadIdx.y < param.q_chunk_size) {
                int32_t subgroup = kv_chunk_size_thread_count < warpSize ? ceil_pow2_u32(kv_chunk_size_thread_count) : warpSize;
                inner_scalar_t max_neighbor;
                inner_scalar_t sum_neighbor;
                inner_scalar_t max_merge;
                inner_scalar_t sum_merge;

                switch(subgroup) {
                    case 32:
                        max_neighbor = __shfl_down_sync(0xffffffff, max_, 16, subgroup);
                        sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 16, subgroup);
                        max_merge = max(max_, max_neighbor);
                        sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_, max_merge), "softmax_sum_reduction_warp_old")) + 
                                                    sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new")), 
                                                    "softmax_sum_reduction_warp");
                        max_ = max_merge;
                        sum_ = sum_merge;
                        [[fallthrough]];
                    case 16:
                        max_neighbor = __shfl_down_sync(0xffffffff, max_, 8, subgroup);
                        sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 8, subgroup);
                        max_merge = max(max_, max_neighbor);
                        sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_, max_merge), "softmax_sum_reduction_warp_old")) + 
                                                    sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new")), 
                                                    "softmax_sum_reduction_warp");
                        max_ = max_merge;
                        sum_ = sum_merge;
                        [[fallthrough]];
                    case 8:
                        max_neighbor = __shfl_down_sync(0xffffffff, max_, 4, subgroup);
                        sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 4, subgroup);
                        max_merge = max(max_, max_neighbor);
                        sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_, max_merge), "softmax_sum_reduction_warp_old")) + 
                                                    sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new")), 
                                                    "softmax_sum_reduction_warp");
                        max_ = max_merge;
                        sum_ = sum_merge;
                        [[fallthrough]];
                    case 4:
                        max_neighbor = __shfl_down_sync(0xffffffff, max_, 2, subgroup);
                        sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 2, subgroup);
                        max_merge = max(max_, max_neighbor);
                        sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_, max_merge), "softmax_sum_reduction_warp_old")) + 
                                                    sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new")), 
                                                    "softmax_sum_reduction_warp");
                        max_ = max_merge;
                        sum_ = sum_merge;
                        [[fallthrough]];
                    case 2:
                        max_neighbor = __shfl_down_sync(0xffffffff, max_, 1, subgroup);
                        sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 1, subgroup);
                        max_merge = max(max_, max_neighbor);
                        sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_, max_merge), "softmax_sum_reduction_warp_old")) + 
                                                    sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new")), 
                                                    "softmax_sum_reduction_warp");
                        max_ = max_merge;
                        sum_ = sum_merge;
                        [[fallthrough]];
                    case 1:
                        break;
                    default:
                        toy_flash_attn_assert(false);
                }

                if (threadIdx.x == 0) {
                    layout.max_reduction_at(threadIdx.y, 0) = max_;
                    layout.sum_reduction_at(threadIdx.y, 0) = sum_;
                }
            } 
            __syncthreads();
#ifdef DEBUG_FLASH_ATTN_V3_TRACE
            if (debug_row && threadIdx.x == 0 && threadIdx.y < param.q_chunk_size) {
                printf(
                    "chunk_softmax bid=%lld hid=%lld q=%lld chunk=%lld max=%f sum=%f\n",
                    (long long)param.batch_id(),
                    (long long)param.head_id(),
                    (long long)param.q_token_id(),
                    (long long)kv_chunk_id,
                    (double)layout.max_reduction_at(threadIdx.y, 0),
                    (double)layout.sum_reduction_at(threadIdx.y, 0)
                );
            }
#endif

            if (threadIdx.y==0 && threadIdx.x == 0) {
                printf(
                    "(%u, %u, %u) q=%lld kv_chunk=%lld max[0][0]: %f\n",
                    (unsigned)blockIdx.x,
                    (unsigned)blockIdx.y,
                    (unsigned)blockIdx.z,
                    (long long)param.q_token_id(),
                    (long long)kv_chunk_id,
                    (double)layout.max_reduction_at(0, 0)
                );
                printf(
                    "(%u, %u, %u) q=%lld kv_chunk=%lld sum[0][0]: %f\n",
                    (unsigned)blockIdx.x,
                    (unsigned)blockIdx.y,
                    (unsigned)blockIdx.z,
                    (long long)param.q_token_id(),
                    (long long)kv_chunk_id,
                    (double)layout.sum_reduction_at(0, 0)
                );
            }

            if (valid_thread) { // block 广播max & sum, 回填softmax
                inner_scalar_t max_ = layout.max_reduction_at(threadIdx.y, 0);
                #pragma unroll
                for(int64_t lane=0; lane < K_X_STRIDE; lane++) {
                    int64_t seq_off = base + lane;
                    if (seq_off < param.kv_chunk_size) {
                        inner_scalar_t score = layout.score_reduction_at(threadIdx.y, seq_off);
                        inner_scalar_t softmax = exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(score, max_), "softmax_exp_sub"));    // 去除sum除法
                        layout.softmax_reduction_at(threadIdx.y, seq_off) = softmax;
                    }
                }
            }
            __syncthreads();
        }
        {   // load V
            bool valid_thread = threadIdx.y < param.kv_chunk_size && threadIdx.x < head_dim_thread_count;
            int64_t kv_seq_id = kv_chunk_id * param.kv_chunk_size + threadIdx.y;
            if (valid_thread) {
                int64_t base = threadIdx.x * K_X_STRIDE;
                #pragma unroll 
                for(int lane=0; lane < K_X_STRIDE; lane++) {
                    int64_t head_pos = base + lane;
                    if (head_pos < param.head_dim) {
                        layout.v_t_at(head_pos , threadIdx.y) = 
                            (kv_seq_id < param.kv_seqlen()) ?
                                param.v_at(kv_seq_id, head_pos) : scalar_t(0);
                    }
                }
            }
            __syncthreads();
        }
        for(int64_t head_off = 0; head_off < param.head_dim; head_off++) {
            bool valid_thread = threadIdx.y < param.q_chunk_size && threadIdx.x < kv_chunk_size_thread_count;
            if (valid_thread) {
                // 乘法低的精度，加法高精度
                int64_t base = threadIdx.x * K_X_STRIDE;
                #pragma unroll
                for(int64_t lane=0; lane < K_X_STRIDE; lane++) {
                    int64_t seq_off = base + lane;
                    if (seq_off < param.kv_chunk_size) {
                        // scalar_t s = static_cast<scalar_t>(layout.softmax_reduction_at(threadIdx.y, seq_off));
                        inner_scalar_t s = layout.softmax_reduction_at(threadIdx.y, seq_off);
                        inner_scalar_t v = layout.v_t_at(head_off, seq_off);
                        inner_scalar_t val = check_non_finite_val(s * v, "softmax_mul_v");
                        layout.sv_matmul_reduction_at(threadIdx.y, seq_off) = inner_scalar_t(val);
                    }
                }
            }
            __syncthreads();

            {   // softmax dot V block reduction
                int64_t base = threadIdx.x * K_X_STRIDE;
                inner_scalar_t out = inner_scalar_t(0);
                #pragma unroll
                for(int64_t lane=0; lane<K_X_STRIDE; lane++) {
                    int64_t seq_off = base + lane;
                    if (seq_off < param.kv_chunk_size) {
                        inner_scalar_t out_lane = valid_thread ? layout.sv_matmul_reduction_at(threadIdx.y, seq_off) : inner_scalar_t(0);
                        out = check_non_finite_val(out + out_lane, "out_reduction_lane");
                    }
                }
                if (valid_thread) {
                    layout.sv_matmul_reduction_at(threadIdx.y, base) = out;
                }

                int64_t bound = ceil_pow2_u32(kv_chunk_size_thread_count)/2;
                for(; bound >= warpSize; bound /= 2) {  // unlikely逻辑， 这里大概率不会走到
                    if (threadIdx.x < bound && threadIdx.y < param.q_chunk_size) {
                        int64_t neighbor_base = (threadIdx.x + bound) * K_X_STRIDE;
                        inner_scalar_t out_neighbor = (threadIdx.x + bound < kv_chunk_size_thread_count) ? 
                            layout.sv_matmul_reduction_at(threadIdx.y, neighbor_base) : inner_scalar_t(0);
                        out = check_non_finite_val(out + out_neighbor, "out_reduction_block");
                        layout.sv_matmul_reduction_at(threadIdx.y, base) = out;
                    }
                    __syncthreads();
                }
                // softmax dot V warp reduction
                if (threadIdx.x < warpSize && threadIdx.y < param.q_chunk_size) {
                    int64_t subgroup = kv_chunk_size_thread_count < warpSize ? ceil_pow2_u32(kv_chunk_size_thread_count) : warpSize;
                    inner_scalar_t out_neighbor;
                    switch(subgroup) {
                        case 32:
                            out_neighbor = __shfl_down_sync(0xffffffff, out, 16, subgroup);
                            out = check_non_finite_val(out + out_neighbor, "out_reduction_warp");
                            [[fallthrough]];
                        case 16:
                            out_neighbor = __shfl_down_sync(0xffffffff, out, 8, subgroup);
                            out = check_non_finite_val(out + out_neighbor, "out_reduction_warp");
                            [[fallthrough]];
                        case 8:
                            out_neighbor = __shfl_down_sync(0xffffffff, out, 4, subgroup);
                            out = check_non_finite_val(out + out_neighbor, "out_reduction_warp");
                            [[fallthrough]];
                        case 4:
                            out_neighbor = __shfl_down_sync(0xffffffff, out, 2, subgroup);
                            out = check_non_finite_val(out + out_neighbor, "out_reduction_warp");
                            [[fallthrough]];
                        case 2:
                            out_neighbor = __shfl_down_sync(0xffffffff, out, 1, subgroup);
                            out = check_non_finite_val(out + out_neighbor, "out_reduction_warp");
                            [[fallthrough]];
                        case 1:
                            break;
                        default:
                            toy_flash_attn_assert(false);
                    }
                    
                    if (threadIdx.x == 0) {
                        layout.out_reduction_at(threadIdx.y, head_off) = out;
#ifdef DEBUG_FLASH_ATTN_V3_TRACE
                        if (debug_row) {
                            printf(
                                "sv_reduce bid=%lld hid=%lld q=%lld chunk=%lld head_off=%lld out=%f\n",
                                (long long)param.batch_id(),
                                (long long)param.head_id(),
                                (long long)param.q_token_id(),
                                (long long)kv_chunk_id,
                                (long long)head_off,
                                (double)out
                            );
                        }
#endif
                    }
                }
                __syncthreads();
            }
        }

        {   // out reduction
            bool valid_thread = threadIdx.y < param.q_chunk_size && threadIdx.x < head_dim_thread_count;
            inner_scalar_t merge_m, merge_sum;
            if (valid_thread) {
                inner_scalar_t new_m = layout.max_reduction_at(threadIdx.y, 0);
                inner_scalar_t new_sum = layout.sum_reduction_at(threadIdx.y, 0);
                inner_scalar_t old_m = layout.last_chunk_max_at(threadIdx.y);
                inner_scalar_t old_sum = layout.last_chunk_sum_at(threadIdx.y);

                merge_m = max(old_m, new_m);
                merge_sum = check_non_finite_val(
                        old_sum * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(old_m, merge_m), "softmax_chunk_reduction_sum_old_sub")) + 
                        new_sum * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(new_m, merge_m), "softmax_chunk_reduction_sum_new_sub"))
                    , "softmax_chunk_reduction_sum");
                
                int64_t base = threadIdx.x * K_X_STRIDE;
                #pragma unroll
                for(int64_t lane=0; lane < K_X_STRIDE; lane++) {
                    int64_t head_pos = base + lane;
                    if (head_pos < param.head_dim) {
                        inner_scalar_t old_e = inner_scalar_t(layout.out_at(threadIdx.y, head_pos));
                        inner_scalar_t new_e = inner_scalar_t(layout.out_reduction_at(threadIdx.y, head_pos));
                        inner_scalar_t merge_e = check_non_finite_val(
                                old_e * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(old_m, merge_m), "out_chunk_reduction_old_sub")) +
                                new_e * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(new_m, merge_m), "out_chunk_reduction_new_sub"))  
                            , "out_chunk_reduction");       // 去除sum除法，统一到最后
                        layout.out_at(threadIdx.y, head_pos) = merge_e;
                    }
                }
#ifdef DEBUG_FLASH_ATTN_V3_TRACE
                if (debug_row && threadIdx.x == 0) {
                    printf(
                        "chunk_merge bid=%lld hid=%lld q=%lld chunk=%lld "
                        "old_m=%f old_sum=%f new_m=%f new_sum=%f "
                        "merge_m=%f merge_sum=%f old_e=%f new_e=%f merge_e=%f\n",
                        (long long)param.batch_id(),
                        (long long)param.head_id(),
                        (long long)param.q_token_id(),
                        (long long)kv_chunk_id,
                        (double)old_m,
                        (double)old_sum,
                        (double)new_m,
                        (double)new_sum,
                        (double)merge_m,
                        (double)merge_sum,
                        (double)old_e,
                        (double)new_e,
                        (double)merge_e
                    );
                }
#endif
            }
            __syncthreads();
            if (valid_thread && threadIdx.x == 0) {
                layout.last_chunk_max_at(threadIdx.y) = merge_m;
                layout.last_chunk_sum_at(threadIdx.y) = merge_sum;
            }
            __syncthreads();
            if (threadIdx.y == 0 && threadIdx.x == 0) {
                printf(
                    "(%u, %u, %u) q=%lld kv_chunk=%lld out_tile[0][0]: %f\n",
                    (unsigned)blockIdx.x,
                    (unsigned)blockIdx.y,
                    (unsigned)blockIdx.z,
                    (long long)param.q_token_id(),
                    (long long)kv_chunk_id,
                    (double)layout.out_at(0, 0)
                );
            }
        }
        __syncthreads();
    } 
    {   // write back out
        bool valid_thread = threadIdx.y < param.q_chunk_size && threadIdx.x < head_dim_thread_count;
        if (param.q_token_id() < param.q_seqlen() && valid_thread) {
            int64_t base = threadIdx.x * K_X_STRIDE;
            #pragma unroll
            for(int64_t lane=0; lane < K_X_STRIDE; lane++) {
                int64_t head_pos = base + lane;
                if (head_pos < param.head_dim) {
                    inner_scalar_t val = layout.out_at(threadIdx.y, head_pos);
                    inner_scalar_t merge_sum = layout.last_chunk_sum_at(threadIdx.y);
                    param.out_at(param.q_token_id(), head_pos) = scalar_t(softmax_div(val, merge_sum));
                }
            }
        }
    }
}
template<typename scalar_t, typename inner_scalar_t>
void FlashAttnTrait<scalar_t, inner_scalar_t>::check_inputs(
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
    TORCH_CHECK(q.size(1) == k.size(2), "GQA not support, num_head must equal num_kv_head");
} 

template<typename scalar_t, typename inner_scalar_t>
torch::Tensor FlashAttnTrait<scalar_t, inner_scalar_t>::flash_attn_varlen_with_block(
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
#ifdef DEBUG_FLASH_ATTN_V3_TRACE
    printf("########### flash attn v3 ###############\n");
#endif
    int head_dim = q.size(2);
    auto batch_size = seqused_k.size(0);
    auto num_heads = q.size(1);
    int64_t block_size = k.size(1);

    assert(head_dim <= 128);        // 泛化

    int q_chunk_size = Q_CHUNK_SIZE;
    int kv_chunk_size = KV_CHUNK_SIZE;
    int blockX = (head_dim + K_X_STRIDE - 1) / K_X_STRIDE;
    int blockY = BLOCK_Y;
    dim3 block(blockX, blockY);
    dim3 grid(batch_size, (max_seqlen_q + q_chunk_size - 1)/q_chunk_size, num_heads);

    ParamSet param = {
        .q = q.const_data_ptr<scalar_t>(),
        .k = k.const_data_ptr<scalar_t>(),
        .v = v.const_data_ptr<scalar_t>(),
        .out = out.data_ptr<scalar_t>(),
        .q_stride = {q.stride(0), q.stride(1), q.stride(2)},
        .k_stride = {k.stride(0), k.stride(1), k.stride(2), k.stride(3)},
        .v_stride = {v.stride(0), v.stride(1), v.stride(2), v.stride(3)},
        .max_seqlen_q = max_seqlen_q,
        .cu_seqlens_q = cu_seqlens_q.const_data_ptr<int32_t>(),
        .max_seqlen_k = max_seqlen_k,
        .seqused_k = seqused_k.const_data_ptr<int32_t>(),
        .window_size = {window_size_left, window_size_right},
        .block_table = block_table.const_data_ptr<int32_t>(),
        .bt_stride = {block_table.stride(0), block_table.stride(1)},
        .num_heads = num_heads,
        .head_dim = head_dim,
        .block_size = block_size,
        .causal = causal,
        .q_chunk_size = q_chunk_size,
        .kv_chunk_size = kv_chunk_size
    };
    kernel_wrapper<scalar_t, inner_scalar_t><<<grid, block, TileLayout::size(param)>>>(param);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_varlen_with_block_v4_bf16fp32", &FlashAttnTrait<at::BFloat16, float>::flash_attn_varlen_with_block, "flash attn varlen with block");
    m.def("flash_attn_varlen_with_block_v4_fp32fp32", &FlashAttnTrait<float, float>::flash_attn_varlen_with_block, "flash attn varlen with block");
    // m.def("flash_attn_varlen_with_block_v3", &FlashAttnTrait<float, float>::flash_attn_varlen_with_block, "flash attn varlen with block");
    // m.def("flash_attn_varlen_with_block_v3", &FlashAttnTrait<at::BFloat16, at::BFloat16>::flash_attn_varlen_with_block, "flash attn varlen with block");
}
