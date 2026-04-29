
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <torch/headeronly/util/BFloat16.h>

#include "cute/arch/mma_sm80.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/container/array_subbyte.hpp"
#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/pointer.hpp"
#include "cute/pointer_flagged.hpp"
#include "cute/stride.hpp"
#include "cute/tensor_impl.hpp"
#include "helper.h"

template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int thread_block_size>
struct FlashAttnTrait;

template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int thread_block_size>
__global__ void kernel_wrapper(
    typename FlashAttnTrait<scalar_t, inner_scalar_t, head_dim_stride, thread_block_size>::ParamSet param) {
    FlashAttnTrait<scalar_t, inner_scalar_t, head_dim_stride, thread_block_size>::kernel(param);
}

// grid(batch_id, q_token_chunk_id, head_id)
template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int thread_block_size>
struct FlashAttnTrait {
    // 使用 16x8x16算子
    static constexpr int32_t MMA_Q_CHUNK_SIZE = 16;
    static constexpr int32_t MMA_KV_CHUNK_SIZE = 8;
    static constexpr int32_t MMA_HEAD_CHUNK_SIZE = 16;

    static constexpr int32_t Q_CHUNK_SIZE = MMA_Q_CHUNK_SIZE;
    static constexpr int32_t WARP_SIZE = 32;
    static constexpr int32_t KV_CHUNK_SIZE = MMA_KV_CHUNK_SIZE * thread_block_size / WARP_SIZE; 

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
template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int thread_block_size>
struct FlashAttnTrait<scalar_t, inner_scalar_t, head_dim_stride, thread_block_size>::ParamSet {
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
template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int thread_block_size>
struct FlashAttnTrait<scalar_t, inner_scalar_t, head_dim_stride, thread_block_size>::TileLayout {
    static __device__ inline TileLayout builder(char* smem, const ParamSet& param) {
        TileLayout layout = layout_init(smem, param);
        toy_flash_attn_assert(smem != nullptr);
        
        auto sTensor_out = cute::make_tensor(
            cute::make_smem_ptr(layout.out),
            cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<head_dim_stride>{}),
            cute::make_stride(cute::Int<head_dim_stride>{}, cute::_1{})
        );
        auto sTensor_last_max = cute::make_tensor(
            cute::make_smem_ptr(layout.last_chunk_max),
            cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}),
            cute::make_stride(cute::_1{})
        );
        auto sTensor_last_sum = cute::make_tensor(
            cute::make_smem_ptr(layout.last_chunk_sum),
            cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}),
            cute::make_stride(cute::_1{})
        );
        for(int linear = threadIdx.x; linear < cute::size(sTensor_out); linear += thread_block_size) {
            auto coord = cute::idx2crd(linear, cute::shape(sTensor_out));
            sTensor_out(coord) = inner_scalar_t{0};
        }
        if (threadIdx.x < cute::size(sTensor_last_max)) {
            sTensor_last_max(threadIdx.x) = inner_scalar_t{-INFINITY};
            sTensor_last_sum(threadIdx.x) = inner_scalar_t{0};
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
        layout.warp_max = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * KV_CHUNK_SIZE / WARP_SIZE) * sizeof(inner_scalar_t);

        offset = round_up(offset, alignof(inner_scalar_t));
        layout.warp_sum = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * KV_CHUNK_SIZE / WARP_SIZE) * sizeof(inner_scalar_t);
        
        offset = round_up(offset, alignof(scalar_t));
        layout.softmax_reduction = reinterpret_cast<scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * KV_CHUNK_SIZE) * sizeof(scalar_t);

        offset = round_up(offset, alignof(inner_scalar_t));
        layout.last_chunk_max = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * 1) * sizeof(inner_scalar_t);
        
        offset = round_up(offset, alignof(inner_scalar_t));
        layout.last_chunk_sum = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * 1) * sizeof(inner_scalar_t);

        offset = round_up(offset, alignof(int32_t));
        layout.cache_block_table = reinterpret_cast<int32_t*>(smem + offset);
        offset += ((KV_CHUNK_SIZE / param.block_size) + 1) * sizeof(int32_t);

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
    // inner_scalar_t* max_reduction;
    // inner_scalar_t* sum_reduction;
    inner_scalar_t* warp_sum;
    inner_scalar_t* warp_max;
    scalar_t* softmax_reduction;    // 这里不能和score复用，数据类型不同，一个fp32, 一个bf16
    inner_scalar_t* last_chunk_max;
    inner_scalar_t* last_chunk_sum;

    int32_t* cache_block_table;

private:
    __device__ __host__ TileLayout(const ParamSet& p):param(p) {}

    size_t _size;

    const ParamSet& param;
};
template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int thread_block_size>
__device__ void FlashAttnTrait<scalar_t, inner_scalar_t, head_dim_stride, thread_block_size>::kernel(ParamSet& param) {
    extern __shared__ char smem[];
    TileLayout layout = TileLayout::builder(smem, param);

    toy_flash_attn_assert(blockDim.x == thread_block_size);
    toy_flash_attn_assert(Q_CHUNK_SIZE % MMA_Q_CHUNK_SIZE == 0);
    toy_flash_attn_assert(KV_CHUNK_SIZE % MMA_KV_CHUNK_SIZE == 0);

    toy_flash_attn_assert((thread_block_size & (thread_block_size - 1)) == 0);  // 2 的幂

    toy_flash_attn_assert(head_dim_stride==round_up(param.head_dim, MMA_HEAD_CHUNK_SIZE));

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

    const int64_t kv_chunk_begin = kv_seq_begin / KV_CHUNK_SIZE;
    const int64_t kv_chunk_end = (kv_seq_end + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;

    auto gTensor_Q = cute::make_tensor(
        cute::make_gmem_ptr(param.q),
        cute::make_shape(param.q_size[0], param.q_size[1], param.q_size[2]),
        cute::make_stride(param.q_stride[0], param.q_stride[1], param.q_stride[2])
    );
    auto gTensor_out = cute::make_tensor(
        cute::make_gmem_ptr(param.out),
        cute::make_shape(param.out_size[0], param.out_size[1], param.out_size[2]),
        cute::make_stride(param.out_stride[0], param.out_stride[1], param.out_stride[2])
    );
    auto gTensor_K = cute::make_tensor(
        cute::make_gmem_ptr(param.k),
        cute::make_shape(param.k_size[0], param.k_size[1], param.k_size[2], param.k_size[3]),
        cute::make_stride(param.k_stride[0], param.k_stride[1], param.k_stride[2], param.k_stride[3])
    );
    auto gTensor_V = cute::make_tensor(
        cute::make_gmem_ptr(param.v),
        cute::make_shape(param.v_size[0], param.v_size[1], param.v_size[2], param.v_size[3]),
        cute::make_stride(param.v_stride[0], param.v_stride[1], param.v_stride[2], param.v_stride[3])
    );
    int64_t batch_q_len = param.cu_seqlens_q[param.batch_id()+1] - param.cu_seqlens_q[param.batch_id()];
    auto gTensor_Q_batch = cute::make_tensor(
        cute::make_gmem_ptr(
            param.q + param.q_stride[0] * param.cu_seqlens_q[param.batch_id()]
        ),
        cute::make_shape(int64_t{batch_q_len}, param.q_size[1], param.q_size[2]),
        cute::make_stride(param.q_stride[0], param.q_stride[1], param.q_stride[2])
    );
    auto gTensor_out_batch = cute::make_tensor(
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
    auto sTensor_v_t = cute::make_tensor(
        cute::make_smem_ptr(layout.v),
        cute::make_shape(cute::Int<head_dim_stride>{}, cute::Int<KV_CHUNK_SIZE>{}),
        cute::make_stride(cute::Int<KV_CHUNK_SIZE>{}, cute::Int<1>{})
    );
    auto sTensor_out = cute::make_tensor(
        cute::make_smem_ptr(layout.out),
        cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<head_dim_stride>{}),
        cute::make_stride(cute::Int<head_dim_stride>{}, cute::Int<1>{})
    );
    auto sTensor_score = cute::make_tensor(
        cute::make_smem_ptr(layout.score_reduction),
        cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<KV_CHUNK_SIZE>{}),
        cute::make_stride(cute::Int<KV_CHUNK_SIZE>{}, cute::_1{})
    );
    auto sTensor_warp_max = cute::make_tensor(
        cute::make_smem_ptr(layout.warp_max),
        cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<KV_CHUNK_SIZE/WARP_SIZE>{}),
        cute::make_stride(cute::Int<KV_CHUNK_SIZE/WARP_SIZE>{}, cute::_1{})
    );
    auto sTensor_warp_sum = cute::make_tensor(
        cute::make_smem_ptr(layout.warp_sum),
        cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<KV_CHUNK_SIZE/WARP_SIZE>{}),
        cute::make_stride(cute::Int<KV_CHUNK_SIZE/WARP_SIZE>{}, cute::_1{})
    );
    auto sTensor_softmax = cute::make_tensor(
        cute::make_smem_ptr(layout.softmax_reduction),
        cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<KV_CHUNK_SIZE>{}),
        cute::make_stride(cute::Int<KV_CHUNK_SIZE>{}, cute::_1{})
    );
    auto sTensor_out_reduction = cute::make_tensor(
        cute::make_smem_ptr(layout.out_reduction),
        cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<head_dim_stride>{}),
        cute::make_stride(cute::Int<head_dim_stride>{}, cute::Int<1>{})
    );
    auto sTensor_last_max = cute::make_tensor(
        cute::make_smem_ptr(layout.last_chunk_max),
        cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}),
        cute::make_stride(cute::_1{})
    );
    auto sTensor_last_sum = cute::make_tensor(
        cute::make_smem_ptr(layout.last_chunk_sum),
        cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}),
        cute::make_stride(cute::_1{})
    );


    {   // load q chunk         
        auto gQ_chunk = cute::local_tile(
            gTensor_Q_batch(cute::_, param.q_head_id(), cute::_),   // 取当前head得到一个2维 tile
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

        // for kv chunk do 
        //     k tile load
        //     chunk softmax
        //     v tile load
        //     out chunk merge
        
        // k tile载入不能直接做，依赖物理地址 & 虚拟地址转换
        {   // kv_seq_id 作用域 for k tile load
            // 构建block table chunk 缓存
            auto virt_seq_begin = kv_chunk_id * KV_CHUNK_SIZE;
            auto virt_seq_end = min((kv_chunk_id+1)*KV_CHUNK_SIZE, param.kv_seqlen());
            auto virt_block_begin = virt_seq_begin / param.block_size;    // TODO: block_size需要模板化
            auto virt_block_end = (virt_seq_end + param.block_size - 1) / param.block_size;
            for(int linear = threadIdx.x; virt_block_begin + linear < virt_block_end; linear += blockDim.x) {
                layout.cache_block_table[linear] = param.block_table[
                    param.bt_stride[0] * param.batch_id() +
                    param.bt_stride[1] * (virt_block_begin + linear)
                ];
            }
            __syncthreads();

            for(int linear = threadIdx.x; linear < cute::size(sTensor_k); linear += blockDim.x) {
                auto coord = cute::idx2crd(linear, cute::shape(sTensor_k));
                auto [seq_off, head_off] = coord;
                auto kv_seq_id = virt_seq_begin + seq_off;
                bool valid = kv_seq_id < param.kv_seqlen() && head_off < param.head_dim;
                if (valid) {
                    auto virt_block_id = kv_seq_id / param.block_size;
                    auto block_off = kv_seq_id % param.block_size;

                    auto phy_block_id = 
                        layout.cache_block_table[virt_block_id - virt_block_begin];
                    sTensor_k(coord) = gTensor_K(phy_block_id, block_off, param.kv_head_id(), head_off);
                    sTensor_v_t(head_off, seq_off) = gTensor_V(phy_block_id, block_off, param.kv_head_id(), head_off);
                } else {
                    sTensor_k(coord) = scalar_t{0};// v也一起提前载入，减少一次寻址, v转置载入，方便后面 P @ V
                    sTensor_v_t(head_off, seq_off) = scalar_t{0}; 
                }  
            }
            __syncthreads();
        }
        {   // QK matmul use mma
            // q_chunk， k_chunk, head_dim_chunk
            // 
            auto mma = cute::make_tiled_mma(
                cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>{}, // 固定 MMA_Q_CHUNK_SIZE = Q_CHUNK_SIZE所以shape(0) = 1
                cute::Layout<cute::Shape<cute::_1, cute::Int<KV_CHUNK_SIZE/MMA_KV_CHUNK_SIZE>, cute::_1>>{}  // M N K K方向循环，N方向warp处理，M固定单warp
            );
            auto thr_mma = mma.get_thread_slice(threadIdx.x);
            auto tCsAcc = thr_mma.partition_C(sTensor_score);
            auto tCrAcc = cute::make_fragment_like(tCsAcc);
            cute::clear(tCrAcc);
            for(int i=0; i<head_dim_stride / MMA_HEAD_CHUNK_SIZE; i++){
                auto q_chunk = cute::local_tile(sTensor_q,
                    cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<MMA_HEAD_CHUNK_SIZE>{}),
                    cute::make_coord(cute::_0{}, i)
                );
                auto k_chunk = cute::local_tile(sTensor_k,
                    cute::make_shape(cute::Int<KV_CHUNK_SIZE>{}, cute::Int<MMA_HEAD_CHUNK_SIZE>{}),
                    cute::make_coord(cute::_0{}, i)
                );
                auto tCsQ = thr_mma.partition_A(q_chunk);
                auto tCsK = thr_mma.partition_B(k_chunk);
                auto tCrQ = cute::make_fragment_like(tCsQ);
                auto tCrK = cute::make_fragment_like(tCsK);
                cute::copy(tCsQ, tCrQ);
                cute::copy(tCsK, tCrK);
                cute::gemm(thr_mma, tCrQ, tCrK, tCrAcc);
            }
            cute::copy(tCrAcc, tCsAcc);
            __syncthreads();
            // 结合窗口，过滤掉无效部分
            for(int linear = threadIdx.x; linear < cute::size(sTensor_score); linear += thread_block_size) {
                auto coord = cute::idx2crd(linear, cute::shape(sTensor_score));
                auto [q_off, kv_off] = coord;
                auto q_token_id = Q_CHUNK_SIZE * param.q_chunk_id() + q_off;
                auto kv_seq_id = KV_CHUNK_SIZE * kv_chunk_id + kv_off;
                bool valid_score = q_token_id < param.q_seqlen() &&
                    layout.is_valid_kv(q_token_id, kv_seq_id, q_kv_offset, kv_win);
                if (!valid_score) {
                    sTensor_score(coord) = inner_scalar_t{-INFINITY};
                }
            }
            __syncthreads();
        }
        {   // online softmax

            // 先warp归约求chunk max & sum
            toy_flash_attn_assert(thread_block_size >= KV_CHUNK_SIZE);
            toy_flash_attn_assert(thread_block_size % WARP_SIZE == 0);
            toy_flash_attn_assert(KV_CHUNK_SIZE % WARP_SIZE == 0);
            constexpr int warp_chunk_num = KV_CHUNK_SIZE / WARP_SIZE * Q_CHUNK_SIZE;
            for(int linear=threadIdx.x; linear<cute::size(sTensor_score); linear += thread_block_size) {
                auto coord = cute::idx2crd(linear, cute::shape(sTensor_score));
                auto val = sTensor_score(coord);
                auto max_ = val;
                auto sum_ = val == inner_scalar_t{-INFINITY} ? inner_scalar_t{0} : inner_scalar_t{1};
                auto max_neighbor = __shfl_down_sync(0xffffffff, max_, 16);
                auto sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 16);
                auto max_merge = max(max_, max_neighbor);
                auto sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_warp_old")) + 
                                        sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new")), 
                                        "softmax_sum_reduction_warp");
                max_ = max_merge;
                sum_ = sum_merge;
                
                max_neighbor = __shfl_down_sync(0xffffffff, max_, 8);
                sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 8);
                max_merge = max(max_, max_neighbor);
                sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_warp_old")) + 
                                        sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new")), 
                                        "softmax_sum_reduction_warp");
                max_ = max_merge;
                sum_ = sum_merge;
                                    
                max_neighbor = __shfl_down_sync(0xffffffff, max_, 4);
                sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 4);
                max_merge = max(max_, max_neighbor);
                sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_warp_old")) + 
                                        sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new")), 
                                        "softmax_sum_reduction_warp");
                max_ = max_merge;
                sum_ = sum_merge;

                max_neighbor = __shfl_down_sync(0xffffffff, max_, 2);
                sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 2);
                max_merge = max(max_, max_neighbor);
                sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_warp_old")) + 
                                        sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new")), 
                                        "softmax_sum_reduction_warp");
                max_ = max_merge;
                sum_ = sum_merge;

                max_neighbor = __shfl_down_sync(0xffffffff, max_, 1);
                sum_neighbor = __shfl_down_sync(0xffffffff, sum_, 1);
                max_merge = max(max_, max_neighbor);
                sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_warp_old")) + 
                                        sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new")), 
                                        "softmax_sum_reduction_warp");
                max_ = max_merge;
                sum_ = sum_merge;
                
                int warp_off = linear % WARP_SIZE;
                auto [q_off, seq_off] = coord;
                auto warp_in_row = seq_off / WARP_SIZE;
                if (warp_off == 0) {
                    sTensor_warp_max(q_off, warp_in_row) = max_;
                    sTensor_warp_sum(q_off, warp_in_row) = sum_;
                }
            }
            __syncthreads();

            constexpr auto subgroup = KV_CHUNK_SIZE / WARP_SIZE;
            constexpr auto subgroup_num = WARP_SIZE / subgroup;
            uint32_t base_mask = 0xffffffff;
            toy_flash_attn_assert((subgroup & (subgroup-1)) == 0);  // 2的幂，便于warp内分组
            if constexpr (subgroup == 1) {
                base_mask = 0x1;
            } else if constexpr (subgroup == 2) {
                base_mask = 0x3;
            } else if constexpr (subgroup == 4) {
                base_mask = 0xf;
            } else if constexpr (subgroup == 8) {
                base_mask = 0xff;
            } else if constexpr (subgroup == 16) {
                base_mask = 0xffff;
            } else if constexpr (subgroup == 32) {
                base_mask = 0xffffffff;
            } else {
                toy_flash_attn_assert(false);
            }
            // warp分组归约
            for (int linear = threadIdx.x; linear < cute::size(sTensor_warp_max); linear += thread_block_size) {
                auto coord = cute::idx2crd(linear, cute::shape(sTensor_warp_max));
                // auto warp_id = linear / WARP_SIZE;
                auto warp_off = linear % WARP_SIZE;
                auto group_id = warp_off / subgroup;
                auto mask = base_mask << (group_id * subgroup);
                auto max_ = sTensor_warp_max(coord);
                auto sum_ = sTensor_warp_sum(coord);
                auto max_neighbor = max_;
                auto sum_neighbor = sum_;
                auto max_merge = max_;
                auto sum_merge = sum_;
                switch (subgroup) {
                case 32:
                    max_neighbor = __shfl_down_sync(mask, max_, 16);
                    sum_neighbor = __shfl_down_sync(mask, sum_, 16);
                    max_merge = max(max_, max_neighbor);
                    sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_warp_old2")) + 
                                        sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new2")), 
                                        "softmax_sum_reduction_warp2");
                    max_ = max_merge;
                    sum_ = sum_merge;
                    [[fallthrough]];
                case 16:
                    max_neighbor = __shfl_down_sync(mask, max_, 8);
                    sum_neighbor = __shfl_down_sync(mask, sum_, 8);
                    max_merge = max(max_, max_neighbor);
                    sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_warp_old2")) + 
                                        sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new2")), 
                                        "softmax_sum_reduction_warp2");
                    max_ = max_merge;
                    sum_ = sum_merge;
                    [[fallthrough]];
                case 8:
                    max_neighbor = __shfl_down_sync(mask, max_, 4);
                    sum_neighbor = __shfl_down_sync(mask, sum_, 4);
                    max_merge = max(max_, max_neighbor);
                    sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_warp_old2")) + 
                                        sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new2")), 
                                        "softmax_sum_reduction_warp2");
                    max_ = max_merge;
                    sum_ = sum_merge;
                    [[fallthrough]];
                case 4:
                    max_neighbor = __shfl_down_sync(mask, max_, 2);
                    sum_neighbor = __shfl_down_sync(mask, sum_, 2);
                    max_merge = max(max_, max_neighbor);
                    sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_warp_old2")) + 
                                        sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new2")), 
                                        "softmax_sum_reduction_warp2");
                    max_ = max_merge;
                    sum_ = sum_merge;
                    [[fallthrough]];
                case 2:
                    max_neighbor = __shfl_down_sync(mask, max_, 1);
                    sum_neighbor = __shfl_down_sync(mask, sum_, 1);
                    max_merge = max(max_, max_neighbor);
                    sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_warp_old2")) + 
                                        sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new2")), 
                                        "softmax_sum_reduction_warp2");
                    max_ = max_merge;
                    sum_ = sum_merge;
                    [[fallthrough]];
                case 1:
                    break;
                default:
                    toy_flash_attn_assert(false);
                };
                auto [q_off, col_off] = coord;
                if(col_off == 0) {
                    sTensor_warp_max(q_off, 0) = max_;
                    sTensor_warp_sum(q_off, 0) = sum_;
                }   
            }
            __syncthreads();

            // block 广播max, 回填softmax
            for (int linear = threadIdx.x; linear < cute::size(sTensor_score); linear += thread_block_size) {
                auto coord = cute::idx2crd(linear, cute::shape(sTensor_score));
                auto [q_off, seq_off] = coord;
                auto kv_seq_id = kv_chunk_id * KV_CHUNK_SIZE + seq_off;
                auto q_token_id = param.q_chunk_id() * Q_CHUNK_SIZE + q_off;

                auto max_ = sTensor_warp_max(q_off, 0);
                bool is_valid = layout.is_valid_kv(q_token_id, kv_seq_id, q_kv_offset, kv_win);
                auto score = sTensor_score(coord);
                auto softmax = is_valid ? exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(score, max_), "softmax_exp_sub"))
                    : inner_scalar_t(0);
                sTensor_softmax(coord) = scalar_t{softmax};
            }
            __syncthreads();
        }

        {   // P @ V
            auto mma = cute::make_tiled_mma(
                cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>{},
                cute::Layout<cute::Shape<cute::_1, cute::Int<head_dim_stride/MMA_HEAD_CHUNK_SIZE>, cute::_1>>{}    // 依旧规定MMA_Q_CHUNK_SIZE = Q_CHUNK_SIZE
            );
            auto thr_mma = mma.get_thread_slice(threadIdx.x);

            auto tCs_out_reduction = thr_mma.partition_C(sTensor_out_reduction);
            auto tCr_out_reduction = cute::make_fragment_like(tCs_out_reduction);
            cute::clear(tCr_out_reduction);

            for (int mma_kv_chunk_id=0; mma_kv_chunk_id < KV_CHUNK_SIZE / MMA_KV_CHUNK_SIZE; mma_kv_chunk_id++) {
                auto mma_softmax_chunk = cute::local_tile(sTensor_softmax, 
                    cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<MMA_KV_CHUNK_SIZE>{}),
                    cute::make_coord(cute::_0{}, mma_kv_chunk_id)
                );
                auto mma_v_chunk = cute::local_tile(sTensor_v_t,
                    cute::make_shape(cute::Int<head_dim_stride>{}, cute::Int<MMA_KV_CHUNK_SIZE>{}),
                    cute::make_coord(cute::_0{}, mma_kv_chunk_id)
                );
                auto tCs_softmax = thr_mma.partition_A(mma_softmax_chunk);
                auto tCs_V_T = thr_mma.partition_B(mma_v_chunk);
                auto tCr_softmax = cute::make_fragment_like(tCs_softmax);
                auto tCr_V_T = cute::make_fragment_like(tCs_V_T);
                cute::copy(tCs_softmax, tCr_softmax);
                cute::copy(tCs_V_T, tCr_V_T);

                cute::gemm(thr_mma, tCr_softmax, tCr_V_T, tCr_out_reduction);
            }
            cute::copy(tCr_out_reduction, tCs_out_reduction);
            __syncthreads();
        }
        {   // out reduction
            for(int linear = threadIdx.x; linear < cute::size(sTensor_out_reduction); linear += thread_block_size) {
                auto coord = cute::idx2crd(linear, cute::shape(sTensor_out_reduction));
                auto [q_off, head_off] = coord;
                auto max_new = sTensor_warp_max(q_off, 0);
                auto sum_new = sTensor_warp_sum(q_off, 0);
                auto max_old = sTensor_last_max(q_off);
                auto sum_old = sTensor_last_sum(q_off);
                auto max_merge = max(max_old, max_new);
                auto sum_merge = check_non_finite_val(
                    sum_old * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_old, max_merge), "softmax_chunk_reduction_sum_old_sub")) + 
                    sum_new * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_new, max_merge), "softmax_chunk_reduction_sum_new_sub"))
                , "softmax_chunk_reduction_sum");

                auto o_old = sTensor_out(coord);
                auto o_new = sTensor_out_reduction(coord);
                auto o_merge = check_non_finite_val(
                            o_old * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_old, max_merge), "out_chunk_reduction_old_sub")) +
                            o_new * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_new, max_merge), "out_chunk_reduction_new_sub"))  
                        , "out_chunk_reduction");       // 去除sum除法，统一到最后
                sTensor_out(coord) = o_merge;
                if (head_off == 0) {    // block_x总是横跨多个行，不会存在重复的问题？
                    sTensor_last_max(q_off) = max_merge;
                    sTensor_last_sum(q_off) = sum_merge;
                }
            }
            __syncthreads();
        }
    }
    {
        for(int linear = threadIdx.x; linear < cute::size(sTensor_out); linear += thread_block_size) {
            auto coord = cute::idx2crd(linear, cute::shape(sTensor_out));
            auto [q_off, head_off] = coord;
            auto q_token_id = param.q_chunk_id() * Q_CHUNK_SIZE + q_off;
            bool valid_pos = q_token_id < param.q_seqlen() && head_off < param.head_dim;
            if (valid_pos) {
                auto o = sTensor_out(coord);
                auto sum_ = sTensor_last_sum(q_off); 
                gTensor_out_batch(q_token_id, param.q_head_id(), head_off) = scalar_t(softmax_div(o, sum_)); 
            }
        }
    }
}
template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int thread_block_size>
void FlashAttnTrait<scalar_t, inner_scalar_t, head_dim_stride, thread_block_size>::check_inputs(
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

template<typename scalar_t, typename inner_scalar_t, int head_dim_stride, int thread_block_size>
torch::Tensor FlashAttnTrait<scalar_t, inner_scalar_t, head_dim_stride, thread_block_size>::flash_attn_varlen_with_block(
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

    dim3 block(thread_block_size);
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
        .q_size = {q.size(0), q.size(1), q.size(2)},
        .k_size = {k.size(0), k.size(1), k.size(2), k.size(3)},
        .v_size = {v.size(0), v.size(1), v.size(2), v.size(3)},
        .out_size = {out.size(0), out.size(1), out.size(2)},
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
    kernel_wrapper<scalar_t, inner_scalar_t, head_dim_stride, thread_block_size><<<grid, block, TileLayout::size(param)>>>(param);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_varlen_with_block_v6_64", &FlashAttnTrait<at::BFloat16, float, 64, 128>::flash_attn_varlen_with_block, "flash attn varlen with block");
    // m.def("flash_attn_varlen_with_block_v5_128", &FlashAttnTrait<at::BFloat16, float, 128, 16, 16>::flash_attn_varlen_with_block, "flash attn varlen with block");
}
