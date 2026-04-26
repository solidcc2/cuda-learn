
#include <__clang_cuda_builtin_vars.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cuda_bf16.h>

#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/pointer.hpp"
#include "cute/stride.hpp"
#include "cute/tensor_impl.hpp"
#include "helper.h"

template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int block_x, int block_y>
struct FlashAttnTrait;

template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int block_x, int block_y>
__global__ void kernel_wrapper(
    typename FlashAttnTrait<scalar_t, inner_scalar_t, head_dim_stride, block_x, block_y>::ParamSet param) {
    FlashAttnTrait<scalar_t, inner_scalar_t, head_dim_stride, block_x, block_y>::kernel(param);
}

// grid(batch_id, q_token_chunk_id, head_id)
template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int block_x, int block_y>
struct FlashAttnTrait {
    static constexpr int32_t MMA_Q_CHUNK_SIZE = 16;
    static constexpr int32_t MMA_K_CHUNK_SIZE = 16;
    static constexpr int32_t MMA_HEAD_CHUNK_SIZE = 16;

    static constexpr int32_t Q_CHUNK_SIZE = MMA_Q_CHUNK_SIZE;
    static constexpr int32_t WARP_SIZE = 32;
    static constexpr int32_t KV_CHUNK_SIZE = MMA_K_CHUNK_SIZE * (block_x * block_y / WARP_SIZE); 

    using D3_const_G_Tensor = decltype(cute::make_tensor(
        cute::make_gmem_ptr((const scalar_t*)0),
        cute::make_shape(int64_t{0}, int64_t{0}, int64_t{0}),
        cute::make_stride(int64_t{0}, int64_t{0}, int64_t{0})
    ));
    using D4_const_G_Tensor = decltype(cute::make_tensor(
        cute::make_gmem_ptr((const scalar_t*)0),
        cute::make_shape(int64_t{0}, int64_t{0}, int64_t{0}, int64_t{0}),
        cute::make_stride(int64_t{0}, int64_t{0}, int64_t{0}, int64_t{0})
    ));


    using D3_G_Tensor = decltype(cute::make_tensor(
        cute::make_gmem_ptr((scalar_t*)0),
        cute::make_shape(int64_t{0}, int64_t{0}, int64_t{0}),
        cute::make_stride(int64_t{0}, int64_t{0}, int64_t{0})
    ));
    using D4_G_Tensor = decltype(cute::make_tensor(
        cute::make_gmem_ptr((scalar_t*)0),
        cute::make_shape(int64_t{0}, int64_t{0}, int64_t{0}, int64_t{0}),
        cute::make_stride(int64_t{0}, int64_t{0}, int64_t{0}, int64_t{0})
    ));

    using D2_S_Tensor = decltype(cute::make_tensor(
        cute::make_smem_ptr((scalar_t*)0),
        cute::make_shape(int64_t{0}, int64_t{0}),
        cute::make_stride(int64_t{0}, int64_t{0})
    ));
        using D2_inner_S_Tensor = decltype(cute::make_tensor(
        cute::make_smem_ptr((inner_scalar_t*)0),
        cute::make_shape(int64_t{0}, int64_t{0}),
        cute::make_stride(int64_t{0}, int64_t{0})
    ));

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

// grid(batch_id, q_token_chunk_id, q_head_id)
template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int block_x, int block_y>
struct FlashAttnTrait<scalar_t, inner_scalar_t, head_dim_stride, block_x, block_y>::ParamSet {
    const scalar_t* q;    // total_q x num_q_heads x head_dim
    const scalar_t* k;    // num_blocks x block_size x num_kv_heads x head_dim
    const scalar_t* v;
    scalar_t* out;
    int64_t q_stride[3];  
    int64_t k_stride[4];
    int64_t v_stride[4];
    int64_t out_stride[3];
    int64_t q_size[3];
    int64_t k_size[4];
    int64_t v_size[4];
    int64_t out_size[3];  
    int64_t max_seqlen_q;
    const int32_t* cu_seqlens_q;  // batch_size + 1
    int64_t max_seqlen_k;
    const int32_t* seqused_k;     // batch_size
    int64_t window_size[2];
    const int32_t* block_table;
    int64_t bt_stride[2];
    int64_t num_q_heads;
    int64_t num_kv_heads;
    int64_t q_per_kv_group;
    int64_t head_dim;
    int64_t block_size;
    int64_t num_kv_block;
    bool causal;

    __device__ inline int64_t batch_size() const {
        return gridDim.x;
    }
    __device__ inline int64_t batch_id() const {
        return blockIdx.x;
    }
    __device__ inline int64_t q_chunk_id() const {
        return blockIdx.y;
    }
    __device__ inline int64_t q_head_id() const {
        return blockIdx.z;
    }
    __device__ inline int64_t kv_head_id() const {
        return q_head_id() / q_per_kv_group;
    }
    __device__ inline int64_t head_id() const {
        return q_head_id();
    }
    __device__ inline int64_t q_seqlen() const {
        const int64_t bid = batch_id();
        return static_cast<int64_t>(cu_seqlens_q[bid+1] - cu_seqlens_q[bid]);
    }
    __device__ inline int64_t kv_seqlen() const {
        return static_cast<int64_t>(seqused_k[batch_id()]);
    }
    
    __device__ inline scalar_t q_at(int64_t q_token_id, int64_t head_pos) const {
        toy_flash_attn_assert(q_token_id >= 0 && q_token_id < q_seqlen());
        toy_flash_attn_assert(head_pos >= 0 && head_pos < head_dim);
        return q[
            q_stride[0] * (q_token_id + cu_seqlens_q[batch_id()]) + 
            q_stride[1] * q_head_id() +
            q_stride[2] * head_pos
        ];
    }
    __device__ inline scalar_t& out_at(int64_t out_token_id, int64_t head_pos) const {
        toy_flash_attn_assert(out_token_id >= 0 && out_token_id < q_seqlen());
        toy_flash_attn_assert(head_pos >= 0 && head_pos < head_dim);
        return out[
            out_stride[0] * (out_token_id + cu_seqlens_q[batch_id()]) + 
            out_stride[1] * q_head_id() +
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
            k_stride[2] * kv_head_id() +
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
            v_stride[2] * kv_head_id() +
            v_stride[3] * head_pos
        ];
    }
};
template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int block_x, int block_y>
struct FlashAttnTrait<scalar_t, inner_scalar_t, head_dim_stride, block_x, block_y>::TileLayout {
    static __device__ inline TileLayout builder(char* smem, const ParamSet& param) {
        TileLayout layout = layout_init(smem, param);
        toy_flash_attn_assert(smem != nullptr);
        constexpr int64_t x_group = head_dim_stride / block_x;    // 每个x线程处理的数量 




        bool valid_thread = threadIdx.y < Q_CHUNK_SIZE;
        if (valid_thread) {
            int64_t base = threadIdx.x * x_group;
            #pragma unroll  // TODO: 这里不能循环展开，考虑静态模板化
            for(int64_t lane=0; lane < x_group; lane++) {
                int64_t head_pos = base + lane;
                layout.out_at(threadIdx.y, head_pos) = inner_scalar_t(0);
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

        toy_flash_attn_assert(head_dim_stride==round_up(param.head_dim, MMA_HEAD_CHUNK_SIZE));  // 对齐到MMA_HEAD_CHUNK_SIZE,便于mma

        int64_t offset = 0;
        layout.q = reinterpret_cast<scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * head_dim_stride) * sizeof(scalar_t);

        offset = round_up(offset, alignof(scalar_t));
        layout.k = reinterpret_cast<scalar_t*>(smem + offset);
        offset += (KV_CHUNK_SIZE * head_dim_stride) * sizeof(scalar_t);

        offset = round_up(offset, alignof(scalar_t));
        layout.v = reinterpret_cast<scalar_t*>(smem + offset);
        offset += (KV_CHUNK_SIZE * head_dim_stride) * sizeof(scalar_t);

        offset = round_up(offset, alignof(inner_scalar_t));
        layout.out = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * head_dim_stride) * sizeof(inner_scalar_t);

        offset = round_up(offset, alignof(inner_scalar_t));
        layout.out_reduction = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * head_dim_stride) * sizeof(inner_scalar_t);

        offset = round_up(offset, alignof(inner_scalar_t));
        layout.score_reduction = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * KV_CHUNK_SIZE) * sizeof(inner_scalar_t);

        offset = round_up(offset, alignof(inner_scalar_t));
        layout.max_reduction = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * KV_CHUNK_SIZE) * sizeof(inner_scalar_t);

        offset = round_up(offset, alignof(inner_scalar_t));
        layout.sum_reduction = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * KV_CHUNK_SIZE) * sizeof(inner_scalar_t);
        
        offset = round_up(offset, alignof(scalar_t));
        layout.softmax_reduction = reinterpret_cast<scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * KV_CHUNK_SIZE) * sizeof(scalar_t);

        offset = round_up(offset, alignof(inner_scalar_t));
        layout.last_chunk_max = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * 1) * sizeof(inner_scalar_t);
        
        offset = round_up(offset, alignof(inner_scalar_t));
        layout.last_chunk_sum = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * 1) * sizeof(inner_scalar_t);

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
        toy_flash_attn_assert(token_off >= 0 && token_off < Q_CHUNK_SIZE);
        toy_flash_attn_assert(head_pos >= 0 && head_pos < head_dim_stride);
        return q[head_dim_stride * token_off + head_pos];
    }
    __device__ inline scalar_t& k_at(int64_t seq_off, int64_t head_pos) {
        toy_flash_attn_assert(seq_off >= 0 && seq_off < KV_CHUNK_SIZE);
        toy_flash_attn_assert(head_pos >= 0 && head_pos < head_dim_stride);
        return k[head_dim_stride * seq_off + head_pos];
    }

    // v 提供转置读取接口，方便和softmax结果对齐，都是行对位相乘，列统一为kv_chunk_off, 内存布局更友好
    __device__ inline scalar_t& v_t_at(int64_t head_pos, int64_t seq_off) {
        toy_flash_attn_assert(seq_off >= 0 && seq_off < KV_CHUNK_SIZE);
        toy_flash_attn_assert(head_pos >= 0 && head_pos < head_dim_stride);
        return v[KV_CHUNK_SIZE * head_pos + seq_off];
    }
    __device__ inline inner_scalar_t& out_at(int64_t token_off, int64_t head_pos) {
        toy_flash_attn_assert(token_off >= 0 && token_off < Q_CHUNK_SIZE);
        toy_flash_attn_assert(head_pos >= 0 && head_pos < head_dim_stride);
        return out[head_dim_stride * token_off + head_pos];
    }

    __device__ inline inner_scalar_t& out_reduction_at(int64_t token_off, int64_t head_pos) {
        toy_flash_attn_assert(token_off >= 0 && token_off < Q_CHUNK_SIZE);
        toy_flash_attn_assert(head_pos >= 0 && head_pos < param.head_dim);
        return out_reduction[head_dim_stride * token_off + head_pos];
    }

    __device__ inline inner_scalar_t& score_reduction_at(int64_t token_off, int64_t seq_off) {
        toy_flash_attn_assert(token_off >= 0 && token_off < Q_CHUNK_SIZE);
        toy_flash_attn_assert(seq_off >= 0 && seq_off < KV_CHUNK_SIZE);
        return score_reduction[KV_CHUNK_SIZE * token_off + seq_off];
    }
    __device__ inline inner_scalar_t& max_reduction_at(int64_t token_off, int64_t seq_off) {
        toy_flash_attn_assert(token_off >= 0 && token_off < Q_CHUNK_SIZE);
        toy_flash_attn_assert(seq_off >= 0 && seq_off < KV_CHUNK_SIZE);
        return max_reduction[KV_CHUNK_SIZE * token_off + seq_off];
    }
    __device__ inline inner_scalar_t& sum_reduction_at(int64_t token_off, int64_t seq_off) {
        toy_flash_attn_assert(token_off >= 0 && token_off < Q_CHUNK_SIZE);
        toy_flash_attn_assert(seq_off >= 0 && seq_off < KV_CHUNK_SIZE);
        return sum_reduction[KV_CHUNK_SIZE * token_off + seq_off];
    }
    __device__ inline scalar_t& softmax_reduction_at(int64_t token_off, int64_t seq_off) {
        toy_flash_attn_assert(token_off >= 0 && token_off < Q_CHUNK_SIZE);
        toy_flash_attn_assert(seq_off >= 0 && seq_off < KV_CHUNK_SIZE);
        return softmax_reduction[KV_CHUNK_SIZE * token_off + seq_off];
    }

    __device__ inline inner_scalar_t& last_chunk_max_at(int64_t token_off) {
        toy_flash_attn_assert(token_off >= 0 && token_off < Q_CHUNK_SIZE);
        return last_chunk_max[token_off];
    }

    __device__ inline inner_scalar_t& last_chunk_sum_at(int64_t token_off) {
        toy_flash_attn_assert(token_off >= 0 && token_off < Q_CHUNK_SIZE);
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
        toy_flash_attn_assert(q_token_id / Q_CHUNK_SIZE == blockIdx.y);
        toy_flash_attn_assert(head_pos < param.head_dim);
        return q_at(q_token_id % Q_CHUNK_SIZE, head_pos);
    }
    __device__ inline scalar_t k_in_tile(int64_t k_seq_id, int64_t chunk_id, int64_t head_pos) {
        toy_flash_attn_assert(k_seq_id / KV_CHUNK_SIZE == chunk_id);
        toy_flash_attn_assert(head_pos < param.head_dim);
        return k_at(k_seq_id % KV_CHUNK_SIZE, head_pos);
    }

    scalar_t* q;
    scalar_t* k;
    scalar_t* v;
    inner_scalar_t* out;

    inner_scalar_t* out_reduction;

    // inner for softmax
    inner_scalar_t* score_reduction;
    inner_scalar_t* max_reduction;
    inner_scalar_t* sum_reduction;
    scalar_t* softmax_reduction;  // 理论上可以和score_reduction复用，但是性能差别不大，这部分shared占用不是瓶颈，先分开使逻辑清晰
    inner_scalar_t* last_chunk_max;
    inner_scalar_t* last_chunk_sum;

private:
    __device__ __host__ TileLayout(const ParamSet& p):param(p) {}

    size_t _size;

    const ParamSet& param;
};
template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int block_x, int block_y>
__device__ void FlashAttnTrait<scalar_t, inner_scalar_t, head_dim_stride, block_x, block_y>::kernel(ParamSet& param) {
    extern __shared__ char smem[];
    TileLayout layout = TileLayout::builder(smem, param);

    toy_flash_attn_assert(blockDim.y == block_y);
    toy_flash_attn_assert(blockDim.x == block_x);

    toy_flash_attn_assert(Q_CHUNK_SIZE == block_y);
    toy_flash_attn_assert(Q_CHUNK_SIZE * block_x % warpSize == 0);

    toy_flash_attn_assert(KV_CHUNK_SIZE % block_x == 0);
    toy_flash_attn_assert((block_x & (block_x - 1)) == 0);  // 2 的幂

    toy_flash_attn_assert(head_dim_stride % block_x == 0);
    toy_flash_attn_assert(head_dim_stride==round_up(param.head_dim, MMA_HEAD_CHUNK_SIZE));
    toy_flash_attn_assert(block_x <= head_dim_stride);
    toy_flash_attn_assert(block_x <= KV_CHUNK_SIZE);
    toy_flash_attn_assert(KV_CHUNK_SIZE >= head_dim_stride);



    const inner_scalar_t softmax_scale_log2 = M_LOG2Ef32 / sqrt((double)param.head_dim);

    const int64_t kv_win[2] = {
        param.window_size[0] == -1 ? param.kv_seqlen() : param.window_size[0],
        param.causal ? 0 : 
            param.window_size[1] == -1 ? param.kv_seqlen() : param.window_size[1]
    };

    const int64_t q_kv_offset = param.q_seqlen() - param.kv_seqlen();
    const int64_t q_chunk_first = blockIdx.y * Q_CHUNK_SIZE;
    const int64_t q_chunk_last = min(q_chunk_first + Q_CHUNK_SIZE, param.q_seqlen()) - 1;
    int64_t kv_seq_begin = 0;
    int64_t kv_seq_end = 0;
    if (q_chunk_first <= q_chunk_last) {
        kv_seq_begin = max(int64_t(0), q_chunk_first - q_kv_offset - kv_win[0]);
        kv_seq_end = min(param.kv_seqlen(), q_chunk_last - q_kv_offset + 1 + kv_win[1]);
        if (!param.causal && param.window_size[0] == -1 && param.window_size[1] == -1) {
            kv_seq_begin = 0;
            kv_seq_end = param.kv_seqlen();
        }
    }

    int64_t batch_id = blockIdx.x;
    int64_t head_id = blockIdx.z;
    int64_t q_chunk_id = blockIdx.y;


    D3_const_G_Tensor gTensor_Q = cute::make_tensor(
        cute::make_gmem_ptr(param.q),
        cute::make_shape(param.q_size[0], param.q_size[1], param.q_size[2]),
        cute::make_stride(param.q_stride[0], param.q_stride[1], param.q_stride[2])
    );
    D3_G_Tensor gTensor_out = cute::make_tensor(
        cute::make_gmem_ptr(param.out),
        cute::make_shape(param.out_size[0], param.out_size[1], param.out_size[2]),
        cute::make_stride(param.out_stride[0], param.out_stride[1], param.out_stride[2])
    );
    D4_const_G_Tensor gTensor_K = cute::make_tensor(
        cute::make_gmem_ptr(param.k),
        cute::make_shape(param.k_size[0], param.k_size[1], param.k_size[2], param.k_size[3]),
        cute::make_stride(param.k_stride[0], param.k_stride[1], param.k_stride[2], param.k_stride[3])
    );
    D4_const_G_Tensor gTensor_V = cute::make_tensor(
        cute::make_gmem_ptr(param.v),
        cute::make_shape(param.v_size[0], param.v_size[1], param.v_size[2], param.v_size[3]),
        cute::make_stride(param.v_stride[0], param.v_stride[1], param.v_stride[2], param.v_stride[3])
    );
    int64_t batch_q_len = param.cu_seqlens_q[param.batch_id()+1] - param.cu_seqlens_q[param.batch_id()];
    D3_const_G_Tensor gTensor_Q_batch = cute::make_tensor(
        cute::make_gmem_ptr(
            param.q + param.q_stride[0] * param.cu_seqlens_q[param.batch_id()]
        ),
        cute::make_shape(int64_t{batch_q_len}, param.q_size[1], param.q_size[2]),
        cute::make_stride(param.q_stride[0], param.q_stride[1], param.q_stride[2])
    );
    D3_G_Tensor gTensor_out_batch = cute::make_tensor(
        cute::make_gmem_ptr(
            param.out + param.out_stride[0] * param.cu_seqlens_q[param.batch_id()]
        ),
        cute::make_shape(int64_t{batch_q_len}, param.out_size[1], param.out_size[2]),
        cute::make_stride(param.out_stride[0], param.out_stride[1], param.out_stride[2])
    );

    auto sTensor_q = cute::make_tensor(
        cute::make_smem_ptr(layout.q),
        cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<head_dim_stride>{}),
        cute::make_stride(cute::Int<head_dim_stride>{}, cute::Int<1>{})
    );
    auto sTensor_k = cute::make_tensor(
        cute::make_smem_ptr(layout.k),
        cute::make_shape(cute::Int<KV_CHUNK_SIZE>{}, cute::Int<head_dim_stride>{}),
        cute::make_stride(cute::Int<head_dim_stride>{}, cute::Int<1>{})
    );
    auto sTensor_v = cute::make_tensor(
        cute::make_smem_ptr(layout.v),
        cute::make_shape(cute::Int<KV_CHUNK_SIZE>{}, cute::Int<head_dim_stride>{}),
        cute::make_stride(cute::Int<head_dim_stride>{}, cute::Int<1>{})
    );
    auto sTensor_out = cute::make_tensor(
        cute::make_smem_ptr(layout.out),
        cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<head_dim_stride>{}),
        cute::make_stride(cute::Int<head_dim_stride>{}, cute::Int<1>{})
    );

    const int64_t kv_chunk_begin = kv_seq_begin / KV_CHUNK_SIZE;
    const int64_t kv_chunk_end = (kv_seq_end + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;
    {   // load q chunk         
        auto gQ_chunk = cute::local_tile(
            gTensor_Q_batch(cute::_, param.head_id(), cute::_),   // 取当前head得到一个2维 tile
            cute::shape(sTensor_q),
            cute::make_coord(param.q_chunk_id(), cute::Int<0>{})
        );
        // 这一版本先不TiledCopy先手写
        for(int linear = threadIdx.x; linear < cute::size(sTensor_q); linear += blockDim.x) {
            auto coord = cute::idx2crd(linear, cute::shape(sTensor_q));
            auto [q_off, head_off] = coord;
            auto q_token_id = Q_CHUNK_SIZE * param.q_chunk_id() + q_off;
            bool valid = q_token_id < param.q_seqlen() && head_off < param.head_dim;
            sTensor_q(coord) = valid ? gQ_chunk(coord) : scalar_t{0};
        }
        __syncthreads();
    }

    for(int64_t kv_chunk_id = kv_chunk_end - 1;
            kv_chunk_id >= kv_chunk_begin; kv_chunk_id--) {
        {   // kv_seq_id 作用域 for k tile load
            toy_flash_attn_assert(head_dim_stride % block_x == 0);

            constexpr int x_group = head_dim_stride / block_x;
            constexpr int y_group = KV_CHUNK_SIZE / block_y;
            int64_t base_x = threadIdx.x * x_group;
            int64_t base_y = threadIdx.y * y_group;
            #pragma unroll 
            for(int64_t lane_x = 0; lane_x < x_group; ++lane_x) {
                int64_t head_pos = base_x + lane_x;
                
                #pragma unroll
                for(int64_t lane_y = 0; lane_y < y_group; ++lane_y) {
                    int64_t kv_seq_off = base_y + lane_y;
                    int64_t kv_seq_id = kv_chunk_id * KV_CHUNK_SIZE + kv_seq_off;
                    bool valid_k = kv_seq_id < param.kv_seqlen() && 
                        head_pos < param.head_dim;
                    layout.k_at(kv_seq_off, head_pos) = valid_k ?
                        param.k_at(kv_seq_id, head_pos) : scalar_t(0);
                }
            }
        }
        __syncthreads();
        {   // QK matmul use mma
            // q_chunk， k_chunk, head_dim_chunk
            nvcuda::wmma::fragment<
                nvcuda::wmma::matrix_a, 
                MMA_Q_CHUNK_SIZE, MMA_K_CHUNK_SIZE, MMA_HEAD_CHUNK_SIZE,    
                typename WmmaElement<scalar_t>::type, 
                nvcuda::wmma::row_major
            > frag_q;
            nvcuda::wmma::fragment<
                nvcuda::wmma::matrix_b, 
                MMA_Q_CHUNK_SIZE, MMA_K_CHUNK_SIZE, MMA_HEAD_CHUNK_SIZE,   
                typename WmmaElement<scalar_t>::type, 
                nvcuda::wmma::col_major
            > frag_k;
            nvcuda::wmma::fragment<
                nvcuda::wmma::accumulator, 
                MMA_Q_CHUNK_SIZE, MMA_K_CHUNK_SIZE, MMA_HEAD_CHUNK_SIZE,   
                float   // 考虑换成 inner_scalar_t?
            > frag_acc;
            nvcuda::wmma::fill_fragment(frag_acc, 0.0f);

            int64_t tid = threadIdx.y * block_x + threadIdx.x;
            toy_flash_attn_assert(block_x * block_y % warpSize == 0);
            int64_t warp_group_id = tid / warpSize;     // warp 做kv方向并行， warp内做head dim循环
            int64_t k_base = warp_group_id * MMA_K_CHUNK_SIZE;
            toy_flash_attn_assert(k_base < KV_CHUNK_SIZE);
            constexpr int64_t round = head_dim_stride / MMA_HEAD_CHUNK_SIZE;
            #pragma unroll
            for(int head_chunk_id=0; head_chunk_id<round; head_chunk_id++) {
                int64_t head_base = head_chunk_id * MMA_HEAD_CHUNK_SIZE;
                auto* q_tile = WmmaElement<scalar_t>::ptr(&layout.q_at(0, head_base));
                auto* k_tile = WmmaElement<scalar_t>::ptr(&layout.k_at(k_base, head_base));
                nvcuda::wmma::load_matrix_sync(frag_q, q_tile, head_dim_stride);
                nvcuda::wmma::load_matrix_sync(frag_k, k_tile, head_dim_stride);
                nvcuda::wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
            }
            auto* score_tile = WmmaElement<inner_scalar_t>::ptr(&layout.score_reduction_at(0, k_base));
            nvcuda::wmma::store_matrix_sync(score_tile, frag_acc, KV_CHUNK_SIZE, nvcuda::wmma::mem_row_major);
            
            __syncthreads();
            constexpr int64_t x_group = KV_CHUNK_SIZE / block_x;
            #pragma unroll
            for(int64_t lane = 0; lane < x_group; lane++) {
                int64_t kv_seq_off = threadIdx.x * x_group + lane;
                int64_t kv_seq_id = kv_chunk_id * KV_CHUNK_SIZE + kv_seq_off;
                bool valid_score = param.q_token_id() < param.q_seqlen() &&  
                    layout.is_valid_kv(param.q_token_id(), kv_seq_id, q_kv_offset, kv_win);
                if (!valid_score) {
                    layout.score_reduction_at(threadIdx.y, kv_seq_off) = inner_scalar_t(-INFINITY);
                }
            }
            __syncthreads();
        }
        {   // online softmax
            constexpr int64_t x_group = KV_CHUNK_SIZE / block_x;
            int64_t base_x = threadIdx.x * x_group;
            {   // init max sum tile
                #pragma unroll
                for(int64_t lane_x = 0; lane_x < x_group; lane_x++) {
                    int64_t seq_off = base_x + lane_x;
                    layout.max_reduction_at(threadIdx.y, seq_off) = 
                        layout.score_reduction_at(threadIdx.y, seq_off);
                    layout.sum_reduction_at(threadIdx.y, seq_off) = 
                        layout.score_reduction_at(threadIdx.y, seq_off) != inner_scalar_t(-INFINITY) ?
                                inner_scalar_t(1.0) : inner_scalar_t(0.0);
                }
            }
            __syncthreads();

            inner_scalar_t max_ = inner_scalar_t(-INFINITY);
            inner_scalar_t sum_ = inner_scalar_t(0);
            int64_t q_off = threadIdx.y;
            #pragma unroll
            for(int64_t lane_x=0; lane_x <x_group; lane_x++) {  // group 归约
                int64_t seq_off = base_x + lane_x;
                inner_scalar_t max_lane = layout.max_reduction_at(q_off, seq_off);
                inner_scalar_t sum_lane = layout.sum_reduction_at(q_off, seq_off);
                inner_scalar_t max_merge = max(max_lane, max_);
                sum_ = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_lane_old")) + 
                                        sum_lane * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_lane, max_merge), "softmax_sum_reduction_lane_new")), 
                                        "softmax_sum_reduction_lane");
                max_ = max_merge;
            }
            layout.max_reduction_at(q_off, base_x) = max_;
            layout.sum_reduction_at(q_off, base_x) = sum_;
            __syncthreads();

            // 这里大概率是不要warp归约的，但是也实现一下，后续可以改为断言
            int32_t bound = block_x / 2;
            #pragma unroll
            for(; bound >= warpSize; bound /= 2) {  // block reduction
                if (threadIdx.x < bound && threadIdx.y < Q_CHUNK_SIZE) {
                    int64_t neighbor_base = (threadIdx.x + bound) * x_group;
                    inner_scalar_t max_neighbor = layout.max_reduction_at(q_off, neighbor_base);
                    inner_scalar_t sum_neighbor = layout.sum_reduction_at(q_off, neighbor_base);
                    inner_scalar_t max_merge = max(max_, max_neighbor);
                    inner_scalar_t sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_block_old")) + 
                                            sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_block_new")), 
                                            "softmax_sum_reduction_block");
                    max_ = max_merge;
                    sum_ = sum_merge;
                    layout.max_reduction_at(q_off, base_x) = max_;
                    layout.sum_reduction_at(q_off, base_x) = sum_;
                }
                __syncthreads();
            }


            if (threadIdx.x < warpSize) {
                int32_t subgroup = block_x < warpSize ? block_x : warpSize;
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
                    layout.max_reduction_at(q_off, 0) = max_;
                    layout.sum_reduction_at(q_off, 0) = sum_;
                }
            }
            __syncthreads();
#ifdef DEBUG_FLASH_ATTN_TRACE
            if (debug_row && threadIdx.x == 0 && threadIdx.y < Q_CHUNK_SIZE) {
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

            { // block 广播max, 回填softmax
                int64_t q_off = threadIdx.y;

                inner_scalar_t max_ = layout.max_reduction_at(q_off, 0);
                #pragma unroll
                for(int64_t lane_x=0; lane_x < x_group; lane_x++) {
                    int64_t seq_off = base_x + lane_x;
                    int64_t kv_seq_id = kv_chunk_id * KV_CHUNK_SIZE + seq_off;
                    bool valid_q_kv_pair =
                        param.q_token_id() < param.q_seqlen() &&
                        layout.is_valid_kv(param.q_token_id(), kv_seq_id, q_kv_offset, kv_win);
                    inner_scalar_t score = layout.score_reduction_at(q_off, seq_off);
                    inner_scalar_t softmax = valid_q_kv_pair
                        ? exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(score, max_), "softmax_exp_sub"))
                        : inner_scalar_t(0);
                    layout.softmax_reduction_at(q_off, seq_off) = scalar_t(softmax);    // 量化为bf16, 准备参与下一步mma计算
                }

            }
            __syncthreads();
        }
        {   // load V
            constexpr int64_t x_group = head_dim_stride / block_x;
            constexpr int64_t y_group = KV_CHUNK_SIZE / block_y;
            int64_t base_x = threadIdx.x * x_group;
            int64_t base_y = threadIdx.y * y_group;
            #pragma unroll
            for(int lane_y=0; lane_y < y_group; lane_y++) {
                int64_t kv_seq_off = base_y + lane_y;
                int64_t kv_seq_id = kv_chunk_id * KV_CHUNK_SIZE + kv_seq_off;

                #pragma unroll 
                for(int lane_x=0; lane_x < x_group; lane_x++) {
                    int64_t head_pos = base_x + lane_x;
                    bool valid_v = kv_seq_id < param.kv_seqlen() && head_pos < param.head_dim;
                    layout.v_t_at(head_pos, kv_seq_off) = 
                        valid_v ? param.v_at(kv_seq_id, head_pos) : scalar_t(0);
                }
            }
        }
        __syncthreads();

        int64_t tid = threadIdx.y * block_x + threadIdx.x;
        int64_t warp_id = tid / warpSize;
        
        nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_a, 
            MMA_Q_CHUNK_SIZE, MMA_HEAD_CHUNK_SIZE, MMA_K_CHUNK_SIZE, 
            typename WmmaElement<scalar_t>::type,
            nvcuda::wmma::row_major
        > frag_P;
        nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_b, 
            MMA_Q_CHUNK_SIZE, MMA_HEAD_CHUNK_SIZE, MMA_K_CHUNK_SIZE,
            typename WmmaElement<scalar_t>::type,
            nvcuda::wmma::col_major
        > frag_V;
        nvcuda::wmma::fragment<
            nvcuda::wmma::accumulator, 
            MMA_Q_CHUNK_SIZE, MMA_HEAD_CHUNK_SIZE, MMA_K_CHUNK_SIZE,
            typename WmmaElement<inner_scalar_t>::type
        > frag_out;
        int64_t kv_chunk_round = KV_CHUNK_SIZE / MMA_K_CHUNK_SIZE;
        int64_t head_off = warp_id * MMA_HEAD_CHUNK_SIZE;

        nvcuda::wmma::fill_fragment(frag_out, 0.0f);
        // 设定有断言 block_x <= head_dim_stride


        if (head_off < head_dim_stride) {
            for(int round_id = 0; round_id < kv_chunk_round; round_id++) {
                int64_t base_v_off = round_id * MMA_K_CHUNK_SIZE;
                auto* tile_p = WmmaElement<scalar_t>::ptr(&layout.softmax_reduction_at(0, base_v_off));
                auto* tile_v = WmmaElement<scalar_t>::ptr(&layout.v_t_at(head_off, base_v_off));
                nvcuda::wmma::load_matrix_sync(frag_P, tile_p, KV_CHUNK_SIZE);
                nvcuda::wmma::load_matrix_sync(frag_V, tile_v, KV_CHUNK_SIZE);
                nvcuda::wmma::mma_sync(frag_out, frag_P, frag_V, frag_out);
            }
            auto* tile_out_reduction = WmmaElement<inner_scalar_t>::ptr(&layout.out_reduction_at(0, head_off));
            nvcuda::wmma::store_matrix_sync(tile_out_reduction, frag_out, head_dim_stride, nvcuda::wmma::mem_row_major);
        }
        __syncthreads();

        {   // out reduction
            inner_scalar_t merge_m, merge_sum;
            inner_scalar_t new_m = layout.max_reduction_at(threadIdx.y, 0);
            inner_scalar_t new_sum = layout.sum_reduction_at(threadIdx.y, 0);
            inner_scalar_t old_m = layout.last_chunk_max_at(threadIdx.y);
            inner_scalar_t old_sum = layout.last_chunk_sum_at(threadIdx.y);

            merge_m = max(old_m, new_m);
            merge_sum = check_non_finite_val(
                    old_sum * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(old_m, merge_m), "softmax_chunk_reduction_sum_old_sub")) + 
                    new_sum * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(new_m, merge_m), "softmax_chunk_reduction_sum_new_sub"))
                , "softmax_chunk_reduction_sum");
                
            int64_t x_group = head_dim_stride / block_x;
            int64_t base = threadIdx.x * x_group;
            #pragma unroll
            for(int64_t lane=0; lane < x_group; lane++) {
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
            if (threadIdx.x == 0) {
                layout.last_chunk_max_at(threadIdx.y) = merge_m;
                layout.last_chunk_sum_at(threadIdx.y) = merge_sum;
            }
        }
        __syncthreads();
    } 
    {   // write back out
        int64_t x_group = head_dim_stride / block_x;
        int64_t base = threadIdx.x * x_group;
        inner_scalar_t merge_sum = layout.last_chunk_sum_at(threadIdx.y);
        #pragma unroll
        for(int64_t lane=0; lane < x_group; lane++) {
            int64_t head_pos = base + lane;
            bool valid_pos = param.q_token_id() < param.q_seqlen() && head_pos < param.head_dim;
            inner_scalar_t val = layout.out_at(threadIdx.y, head_pos);
            if (valid_pos) {
                param.out_at(param.q_token_id(), head_pos) = scalar_t(softmax_div(val, merge_sum));
            }
        }
    }
}
template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int block_x, int block_y>
void FlashAttnTrait<scalar_t, inner_scalar_t, head_dim_stride, block_x, block_y>::check_inputs(
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
    TORCH_CHECK(k.size(2) == v.size(2), "k and v must have the same num_kv_heads");
    TORCH_CHECK(round_up(q.size(2), 16) == head_dim_stride);
    TORCH_CHECK(q.size(1) % k.size(2) == 0,
        "GQA requires num_q_heads to be divisible by num_kv_heads");
} 

template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int block_x, int block_y>
torch::Tensor FlashAttnTrait<scalar_t, inner_scalar_t, head_dim_stride, block_x, block_y>::flash_attn_varlen_with_block(
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
#ifdef DEBUG_FLASH_ATTN_TRACE
    printf("########### flash attn v3 ###############\n");
#endif
    int head_dim = q.size(2);
    auto batch_size = seqused_k.size(0);
    auto num_q_heads = q.size(1);
    auto num_kv_heads = k.size(2);
    auto q_per_kv_group = num_q_heads / num_kv_heads;
    int64_t block_size = k.size(1);

    assert(head_dim <= 128);        // 泛化

    dim3 block(block_x, block_y);
    dim3 grid(batch_size, (max_seqlen_q + Q_CHUNK_SIZE - 1)/Q_CHUNK_SIZE, num_q_heads);

    ParamSet param = {
        .q = q.const_data_ptr<scalar_t>(),
        .k = k.const_data_ptr<scalar_t>(),
        .v = v.const_data_ptr<scalar_t>(),
        .out = out.data_ptr<scalar_t>(),
        .q_stride = {q.stride(0), q.stride(1), q.stride(2)},
        .k_stride = {k.stride(0), k.stride(1), k.stride(2), k.stride(3)},
        .v_stride = {v.stride(0), v.stride(1), v.stride(2), v.stride(3)},
        .out_stride = {out.stride(0), out.stride(1), out.stride(2)},
        .max_seqlen_q = max_seqlen_q,
        .cu_seqlens_q = cu_seqlens_q.const_data_ptr<int32_t>(),
        .max_seqlen_k = max_seqlen_k,
        .seqused_k = seqused_k.const_data_ptr<int32_t>(),
        .window_size = {window_size_left, window_size_right},
        .block_table = block_table.const_data_ptr<int32_t>(),
        .bt_stride = {block_table.stride(0), block_table.stride(1)},
        .num_q_heads = num_q_heads,
        .num_kv_heads = num_kv_heads,
        .q_per_kv_group = q_per_kv_group,
        .head_dim = head_dim,
        .block_size = block_size,
        .causal = causal,
    };
    kernel_wrapper<scalar_t, inner_scalar_t, head_dim_stride, block_x, block_y><<<grid, block, TileLayout::size(param)>>>(param);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_varlen_with_block_v5_64", &FlashAttnTrait<cute::bfloat16_t, float, 64, 8, 16>::flash_attn_varlen_with_block, "flash attn varlen with block");
    // m.def("flash_attn_varlen_with_block_v5_128", &FlashAttnTrait<at::BFloat16, float, 128, 16, 16>::flash_attn_varlen_with_block, "flash attn varlen with block");
}
