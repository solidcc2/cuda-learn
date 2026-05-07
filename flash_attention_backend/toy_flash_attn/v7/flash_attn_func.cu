
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/headeronly/util/BFloat16.h>
#include <type_traits>
#include <cute/tensor.hpp>
#include "helper.h"

template<typename scalar_t, typename inner_scalar_t, int Q_CHUNK_SIZE, int KV_CHUNK_SIZE, int HEAD_DIM_STRIDE>
struct FlashAttnTrait;

template<typename scalar_t, typename inner_scalar_t, int Q_CHUNK_SIZE, int KV_CHUNK_SIZE, int HEAD_DIM_STRIDE>
__global__ void kernel_wrapper(
    typename FlashAttnTrait<scalar_t, inner_scalar_t, Q_CHUNK_SIZE, KV_CHUNK_SIZE, HEAD_DIM_STRIDE>::ParamSet param) {
    FlashAttnTrait<scalar_t, inner_scalar_t, Q_CHUNK_SIZE, KV_CHUNK_SIZE, HEAD_DIM_STRIDE>::kernel(param);
}

// grid(batch_id, q_token_chunk_id, head_id)
template<typename scalar_t, typename inner_scalar_t, int Q_CHUNK_SIZE, int KV_CHUNK_SIZE, int HEAD_DIM_STRIDE>
struct FlashAttnTrait {
    // 使用 16x8x16算子
    static constexpr int32_t MMA_M = 16;
    static constexpr int32_t MMA_N = 8;
    static constexpr int32_t MMA_K = 16;

    static constexpr int32_t THR_NUM = 128;   // 暂定，后续调整到模板去
    static constexpr int32_t WARP_SIZE = 32;
    
    // Q @ K^T 的MMA约束
    static_assert(Q_CHUNK_SIZE % MMA_M == 0);
    static_assert(KV_CHUNK_SIZE % MMA_N == 0);  
    static_assert(HEAD_DIM_STRIDE % MMA_K == 0);

    // P @ V^T 的MMA约束
    static_assert(KV_CHUNK_SIZE % MMA_K == 0);
    static_assert(HEAD_DIM_STRIDE % MMA_N == 0);


    static_assert(std::is_same_v<inner_scalar_t, float>);   // innner_scalar_t暂时强制float

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
template<typename scalar_t, typename inner_scalar_t, int Q_CHUNK_SIZE, int KV_CHUNK_SIZE, int HEAD_DIM_STRIDE>
struct FlashAttnTrait<scalar_t, inner_scalar_t, Q_CHUNK_SIZE, KV_CHUNK_SIZE, HEAD_DIM_STRIDE>::ParamSet {
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
};
template<typename scalar_t, typename inner_scalar_t, int Q_CHUNK_SIZE, int KV_CHUNK_SIZE, int HEAD_DIM_STRIDE>
struct FlashAttnTrait<scalar_t, inner_scalar_t, Q_CHUNK_SIZE, KV_CHUNK_SIZE, HEAD_DIM_STRIDE>::TileLayout {
    static __device__ inline TileLayout builder(char* smem, const ParamSet& param) {
        toy_flash_attn_assert(param.head_dim == HEAD_DIM_STRIDE);
        TileLayout layout = layout_init(smem, param);
        toy_flash_attn_assert(smem != nullptr);
        
        auto sTensor_out = cute::make_tensor(
            cute::make_smem_ptr(layout.out),
            cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<HEAD_DIM_STRIDE>{}),
            cute::make_stride(cute::Int<HEAD_DIM_STRIDE>{}, cute::_1{})
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
        for(int linear = threadIdx.x; linear < cute::size(sTensor_out); linear += THR_NUM) {
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

        int64_t offset = 0;
        layout.q = reinterpret_cast<scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * HEAD_DIM_STRIDE) * sizeof(scalar_t);

        offset = round_up(offset, alignof(scalar_t));
        layout.k = reinterpret_cast<scalar_t*>(smem + offset);
        offset += (KV_CHUNK_SIZE * HEAD_DIM_STRIDE) * sizeof(scalar_t);

        offset = round_up(offset, alignof(scalar_t));
        layout.v = reinterpret_cast<scalar_t*>(smem + offset);
        offset += (KV_CHUNK_SIZE * HEAD_DIM_STRIDE) * sizeof(scalar_t);

        offset = round_up(offset, alignof(inner_scalar_t));
        layout.out = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * HEAD_DIM_STRIDE) * sizeof(inner_scalar_t);

        offset = round_up(offset, alignof(inner_scalar_t));
        layout.out_reduction = reinterpret_cast<inner_scalar_t*>(smem + offset);
        offset += (Q_CHUNK_SIZE * HEAD_DIM_STRIDE) * sizeof(inner_scalar_t);

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

    __device__ inline bool is_valid_kv(int64_t q_token_id, int64_t kv_seq_id, int64_t q_kv_offset, const int64_t* kv_win) {
        const int64_t kv_axis = q_token_id - q_kv_offset;
        const int64_t win[2] = {
            max(int64_t(0), kv_axis - kv_win[0]),
            min(param.kv_seqlen()-1, kv_axis + kv_win[1]),
        };
        return kv_seq_id >= win[0] && kv_seq_id <= win[1];
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
template<typename scalar_t, typename inner_scalar_t, int Q_CHUNK_SIZE, int KV_CHUNK_SIZE, int HEAD_DIM_STRIDE>
__device__ void FlashAttnTrait<scalar_t, inner_scalar_t, Q_CHUNK_SIZE, KV_CHUNK_SIZE, HEAD_DIM_STRIDE>::kernel(ParamSet& param) {
    extern __shared__ char smem[];
    TileLayout layout = TileLayout::builder(smem, param);

    toy_flash_attn_assert(blockDim.x == THR_NUM);
    toy_flash_attn_assert(Q_CHUNK_SIZE % MMA_M == 0);
    toy_flash_attn_assert(KV_CHUNK_SIZE % cutlass::const_max(MMA_N, MMA_K) == 0);

    toy_flash_attn_assert((THR_NUM & (THR_NUM - 1)) == 0);  // 2 的幂

    toy_flash_attn_assert(HEAD_DIM_STRIDE==round_up(param.head_dim, cutlass::const_max(MMA_N, MMA_K)));

    const inner_scalar_t softmax_scale_log2 = M_LOG2Ef32 / sqrt((double)param.head_dim);

    const int64_t kv_win[2] = {
        param.window_size[0] == -1 ? param.kv_seqlen() : param.window_size[0],
        param.causal ? 0 : 
            param.window_size[1] == -1 ? param.kv_seqlen() : param.window_size[1]
    };

    const int64_t q_kv_offset = param.q_seqlen() - param.kv_seqlen();
    const int64_t q_chunk_first = param.q_chunk_id() * Q_CHUNK_SIZE;
    const int64_t q_chunk_last = min(q_chunk_first + Q_CHUNK_SIZE, param.q_seqlen()) - 1;
    int64_t kv_seq_begin = 0;
    int64_t kv_seq_end = 0;
    if (q_chunk_first <= q_chunk_last) {
        kv_seq_begin = max(int64_t(0), q_chunk_first - q_kv_offset - kv_win[0]);
        kv_seq_end = min(param.kv_seqlen(), q_chunk_last - q_kv_offset + 1 + kv_win[1]);
    }

    const int64_t kv_chunk_begin = kv_seq_begin / KV_CHUNK_SIZE;
    const int64_t kv_chunk_end = (kv_seq_end + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;

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
    auto gTensor_Q_batch = cute::make_tensor(
        cute::make_gmem_ptr(
            param.q + param.q_stride[0] * param.cu_seqlens_q[param.batch_id()]
        ),
        cute::make_shape(int64_t{param.q_seqlen()}, param.q_size[1], param.q_size[2]),
        cute::make_stride(param.q_stride[0], param.q_stride[1], param.q_stride[2])
    );
    auto gTensor_out_batch = cute::make_tensor(
        cute::make_gmem_ptr(
            param.out + param.out_stride[0] * param.cu_seqlens_q[param.batch_id()]
        ),
        cute::make_shape(int64_t{param.q_seqlen()}, param.out_size[1], param.out_size[2]),
        cute::make_stride(param.out_stride[0], param.out_stride[1], param.out_stride[2])
    );

    auto sTensor_q = cute::make_tensor(
        cute::make_smem_ptr(layout.q),
        cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<HEAD_DIM_STRIDE>{}),
        cute::make_stride(cute::Int<HEAD_DIM_STRIDE>{}, cute::Int<1>{})
    );
    cute::Layout kv_head_layout = cute::make_layout(
        cute::make_shape(cute::Int<KV_CHUNK_SIZE>{}, cute::Int<HEAD_DIM_STRIDE>{}),
        cute::make_stride(cute::Int<HEAD_DIM_STRIDE>{}, cute::Int<1>{})
    );
    auto k_layout = cute::composition(
        cute::Swizzle<5, 1>{},
        kv_head_layout
    );
    auto sTensor_k = cute::make_tensor(
        cute::make_smem_ptr(layout.k),
        k_layout
    );
    auto sTensor_v_t = cute::make_tensor(
        cute::make_smem_ptr(layout.v),
        cute::make_shape(cute::Int<HEAD_DIM_STRIDE>{}, cute::Int<KV_CHUNK_SIZE>{}),
        cute::make_stride(cute::Int<KV_CHUNK_SIZE>{}, cute::Int<1>{})
    );
    auto sTensor_out = cute::make_tensor(
        cute::make_smem_ptr(layout.out),
        cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<HEAD_DIM_STRIDE>{}),
        cute::make_stride(cute::Int<HEAD_DIM_STRIDE>{}, cute::Int<1>{})
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
        cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<HEAD_DIM_STRIDE>{}),
        cute::make_stride(cute::Int<HEAD_DIM_STRIDE>{}, cute::Int<1>{})
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

    {
        cute::Tensor gQ_chunk = cute::local_tile(
            gTensor_Q_batch(cute::_, param.q_head_id(), cute::_),   // 取当前head得到一个2维 tile
            cute::shape(sTensor_q),
            cute::make_coord(param.q_chunk_id(), cute::Int<0>{})
        );
        cute::Tensor Q_id = cute::make_identity_tensor(cute::shape(gTensor_Q_batch));
        cute::Tensor Q_id_chunk = cute::local_tile(
            Q_id(cute::_, param.q_head_id(), cute::_),
            cute::shape(sTensor_q),
            cute::make_coord(param.q_chunk_id(), cute::Int<0>{})
        );

        constexpr int thr_data = sizeof(cute::uint128_t) / sizeof(scalar_t);
        constexpr int thr_col = HEAD_DIM_STRIDE / thr_data;
        constexpr int thr_row = THR_NUM / thr_col;
        cute::Layout thr_layout = cute::make_layout(
            cute::make_shape(cute::Int<thr_row>{}, cute::Int<thr_col>{}),
            cute::LayoutRight{}
        );
        cute::Layout val_layout = cute::make_layout(
            cute::make_shape(cute::_1{}, cute::Int<thr_data>{})
        );

        cute::TiledCopy copy_a = cute::make_tiled_copy(
            cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, scalar_t>{},
            thr_layout, val_layout
        );
        if (threadIdx.x < cute::size(copy_a)) {
            cute::ThrCopy copy = copy_a.get_slice(threadIdx.x);
            cute::Tensor src = copy.partition_S(gQ_chunk);
            cute::Tensor src_id = copy.partition_S(Q_id_chunk);
            cute::Tensor dst = copy.partition_D(sTensor_q);
            cute::Tensor pred = cute::make_tensor<bool>(cute::shape(src));
            
            CUTE_UNROLL
            for(int i=0; i < cute::size(pred); i++) {
                auto coord = src_id(i);
                pred(i) = cute::elem_less(src_id(i), cute::shape(gTensor_Q_batch));
            }
            cute::fill(dst, scalar_t{0});
            cute::copy_if(pred, src, dst);
            cute::cp_async_fence();
        }
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
                auto kv_seq_id = kv_chunk_id * KV_CHUNK_SIZE + seq_off;
                bool valid = kv_seq_id < param.kv_seqlen() && head_off < param.head_dim;
                if (valid) {
                    auto virt_block_id = kv_seq_id / param.block_size;
                    auto block_off = kv_seq_id % param.block_size;

                    auto phy_block_id = 
                        layout.cache_block_table[virt_block_id - virt_block_begin];
                    // auto phy_block_id = param.block_table[
                    //     param.bt_stride[0] * param.batch_id() +
                    //     param.bt_stride[1] * virt_block_id
                    // ];

                    sTensor_k(coord) = gTensor_K(phy_block_id, block_off, param.kv_head_id(), head_off);
                    sTensor_v_t(head_off, seq_off) = gTensor_V(phy_block_id, block_off, param.kv_head_id(), head_off);
                } else {
                    sTensor_k(coord) = scalar_t{0};// v也一起提前载入，减少一次寻址, v转置载入，方便后面 P @ V
                    sTensor_v_t(head_off, seq_off) = scalar_t{0}; 
                }  
            }
            __syncthreads();
        }
        {
            cute::cp_async_wait<0>();   // Q就绪
            __syncthreads();
        }
        {   // QK matmul use mma
            // q_chunk， k_chunk, head_dim_chunk
            // 
            auto mma = cute::make_tiled_mma(
                cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>{}, // 固定 MMA_Q_CHUNK_SIZE = Q_CHUNK_SIZE所以shape(0) = 1
                cute::Layout<cute::Shape<cute::_1, cute::Int<KV_CHUNK_SIZE/MMA_N>, cute::_1>>{}  // M N K K方向循环，N方向warp处理，M固定单warp
            );
            auto thr_mma = mma.get_thread_slice(threadIdx.x);
            auto tCsAcc = thr_mma.partition_C(sTensor_score);
            auto tCrAcc = cute::make_fragment_like(tCsAcc);
            cute::clear(tCrAcc);
            for(int i=0; i<HEAD_DIM_STRIDE / MMA_K; i++){
                auto q_chunk = cute::local_tile(sTensor_q,
                    cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<MMA_K>{}),
                    cute::make_coord(cute::_0{}, i)
                );
                auto k_chunk = cute::local_tile(sTensor_k,
                    cute::make_shape(cute::Int<KV_CHUNK_SIZE>{}, cute::Int<MMA_K>{}),
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
            for(int linear = threadIdx.x; linear < cute::size(sTensor_score); linear += THR_NUM) {
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
#ifdef DEBUG_FLASH_ATTN_TRACE
            if (param.batch_id() == 0 &&
                param.q_head_id() == 0 &&
                kv_chunk_id == kv_chunk_end - 1 &&
                threadIdx.x == 0) {
                printf("QK_TILE_V6 chunk=%lld\n", (long long)kv_chunk_id);
                for (int q_row = 0; q_row < 2 && q_row < Q_CHUNK_SIZE; ++q_row) {
                    printf("  row %d:", q_row);
                    for (int seq_col = 0; seq_col < 8 && seq_col < KV_CHUNK_SIZE; ++seq_col) {
                        printf(" %0.6f", (double)sTensor_score(q_row, seq_col));
                    }
                    printf("\n");
                }
            }
#endif
        }
        {   // online softmax

            // 先warp归约求chunk max & sum
            toy_flash_attn_assert(THR_NUM >= KV_CHUNK_SIZE);
            toy_flash_attn_assert(THR_NUM % WARP_SIZE == 0);
            toy_flash_attn_assert(KV_CHUNK_SIZE % WARP_SIZE == 0);
            for(int linear=threadIdx.x; linear<cute::size(sTensor_score); linear += THR_NUM) {
                // auto coord = cute::idx2crd(linear, cute::Shape<cute::Int<KV_CHUNK_SIZE>, cute::Int<Q_CHUNK_SIZE>>{});
                // auto coord = cute::idx2crd(linear, cute::shape(sTensor_score));
                auto q_off = linear / KV_CHUNK_SIZE;
                auto seq_off = linear % KV_CHUNK_SIZE;
                auto val = sTensor_score(q_off, seq_off);
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
                // auto [q_off, seq_off] = coord;
                auto warp_in_row = seq_off / WARP_SIZE;
                if (warp_off == 0) {
                    sTensor_warp_max(q_off, warp_in_row) = max_;
                    sTensor_warp_sum(q_off, warp_in_row) = sum_;
                }
            }
            __syncthreads();
#ifdef DEBUG_FLASH_ATTN_TRACE
            if (param.batch_id() == 0 &&
                param.q_head_id() == 0 &&
                kv_chunk_id == kv_chunk_end - 1 &&
                threadIdx.x == 0) {
                printf("WARP_STAGE1_V6 chunk=%lld\n", (long long)kv_chunk_id);
                for (int q_row = 0; q_row < 2 && q_row < Q_CHUNK_SIZE; ++q_row) {
                    printf("  row %d max:", q_row);
                    for (int col = 0; col < KV_CHUNK_SIZE / WARP_SIZE; ++col) {
                        printf(" %0.6f", (double)sTensor_warp_max(q_row, col));
                    }
                    printf("\n");
                    printf("  row %d sum:", q_row);
                    for (int col = 0; col < KV_CHUNK_SIZE / WARP_SIZE; ++col) {
                        printf(" %0.6f", (double)sTensor_warp_sum(q_row, col));
                    }
                    printf("\n");
                }
            }
#endif

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
            for (int linear = threadIdx.x; linear < cute::size(sTensor_warp_max); linear += THR_NUM) {
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
                    max_neighbor = __shfl_down_sync(mask, max_, 16, subgroup);
                    sum_neighbor = __shfl_down_sync(mask, sum_, 16, subgroup);
                    max_merge = max(max_, max_neighbor);
                    sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_warp_old2")) + 
                                        sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new2")), 
                                        "softmax_sum_reduction_warp2");
                    max_ = max_merge;
                    sum_ = sum_merge;
                    [[fallthrough]];
                case 16:
                    max_neighbor = __shfl_down_sync(mask, max_, 8, subgroup);
                    sum_neighbor = __shfl_down_sync(mask, sum_, 8, subgroup);
                    max_merge = max(max_, max_neighbor);
                    sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_warp_old2")) + 
                                        sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new2")), 
                                        "softmax_sum_reduction_warp2");
                    max_ = max_merge;
                    sum_ = sum_merge;
                    [[fallthrough]];
                case 8:
                    max_neighbor = __shfl_down_sync(mask, max_, 4, subgroup);
                    sum_neighbor = __shfl_down_sync(mask, sum_, 4, subgroup);
                    max_merge = max(max_, max_neighbor);
                    sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_warp_old2")) + 
                                        sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new2")), 
                                        "softmax_sum_reduction_warp2");
                    max_ = max_merge;
                    sum_ = sum_merge;
                    [[fallthrough]];
                case 4:
                    max_neighbor = __shfl_down_sync(mask, max_, 2, subgroup);
                    sum_neighbor = __shfl_down_sync(mask, sum_, 2, subgroup);
                    max_merge = max(max_, max_neighbor);
                    sum_merge = check_non_finite_val(sum_ * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_,max_merge), "softmax_sum_reduction_warp_old2")) + 
                                        sum_neighbor * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_neighbor, max_merge), "softmax_sum_reduction_warp_new2")), 
                                        "softmax_sum_reduction_warp2");
                    max_ = max_merge;
                    sum_ = sum_merge;
                    [[fallthrough]];
                case 2:
                    max_neighbor = __shfl_down_sync(mask, max_, 1, subgroup);
                    sum_neighbor = __shfl_down_sync(mask, sum_, 1, subgroup);
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
            for (int linear = threadIdx.x; linear < cute::size(sTensor_score); linear += THR_NUM) {
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
#ifdef DEBUG_FLASH_ATTN_TRACE
            if (param.batch_id() == 0 &&
                param.q_head_id() == 0 &&
                kv_chunk_id == kv_chunk_end - 1 &&
                threadIdx.x == 0) {
                printf("P_TILE_V6 chunk=%lld\n", (long long)kv_chunk_id);
                for (int q_row = 0; q_row < 2 && q_row < Q_CHUNK_SIZE; ++q_row) {
                    printf("  row %d:", q_row);
                    for (int seq_col = 0; seq_col < 8 && seq_col < KV_CHUNK_SIZE; ++seq_col) {
                        printf(" %0.6f", (double)sTensor_softmax(q_row, seq_col));
                    }
                    printf("\n");
                }
            }
#endif
        }
        {    // P @ V
            constexpr int WARP_NUM = THR_NUM / WARP_SIZE;
            constexpr int N_PER_PASS = WARP_NUM * MMA_N;
            static_assert(HEAD_DIM_STRIDE % N_PER_PASS == 0);

            auto mma = cute::make_tiled_mma(
                cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>{},
                cute::Layout<cute::Shape<cute::_1, cute::Int<WARP_NUM>, cute::_1>>{}
            );
            auto thr_mma = mma.get_thread_slice(threadIdx.x);

            for (int n_tile_id = 0; n_tile_id < HEAD_DIM_STRIDE / N_PER_PASS; ++n_tile_id) {
                auto out_chunk = cute::local_tile(
                    sTensor_out_reduction,
                    cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<N_PER_PASS>{}),
                    cute::make_coord(cute::_0{}, n_tile_id)
                );

                auto tCs_out = thr_mma.partition_C(out_chunk);
                auto tCr_out = cute::make_fragment_like(tCs_out);
                cute::clear(tCr_out);

                for (int mma_kv_chunk_id = 0; mma_kv_chunk_id < KV_CHUNK_SIZE / MMA_K; ++mma_kv_chunk_id) {
                    auto softmax_chunk = cute::local_tile(
                        sTensor_softmax,
                        cute::make_shape(cute::Int<Q_CHUNK_SIZE>{}, cute::Int<MMA_K>{}),
                        cute::make_coord(cute::_0{}, mma_kv_chunk_id)
                    );
                    auto v_chunk = cute::local_tile(
                        sTensor_v_t,
                        cute::make_shape(cute::Int<N_PER_PASS>{}, cute::Int<MMA_K>{}),
                        cute::make_coord(n_tile_id, mma_kv_chunk_id)
                    );

                    auto tCs_softmax = thr_mma.partition_A(softmax_chunk);
                    auto tCs_V_T = thr_mma.partition_B(v_chunk);
                    auto tCr_softmax = cute::make_fragment_like(tCs_softmax);
                    auto tCr_V_T = cute::make_fragment_like(tCs_V_T);

                    cute::copy(tCs_softmax, tCr_softmax);
                    cute::copy(tCs_V_T, tCr_V_T);
                    cute::gemm(thr_mma, tCr_softmax, tCr_V_T, tCr_out);
                }

                cute::copy(tCr_out, tCs_out);
            }
            __syncthreads();
        }

        {   // out reduction
            for(int linear = threadIdx.x; linear < cute::size(sTensor_out_reduction); linear += THR_NUM) {
                auto coord = cute::idx2crd(linear, cute::shape(sTensor_out_reduction));
                auto [q_off, head_off] = coord;
                auto max_new = sTensor_warp_max(q_off, 0);
                auto sum_new = sTensor_warp_sum(q_off, 0);
                auto max_old = sTensor_last_max(q_off);
                auto sum_old = sTensor_last_sum(q_off);
                auto max_merge = max(max_old, max_new);

                auto o_old = sTensor_out(coord);
                auto o_new = sTensor_out_reduction(coord);
                auto o_merge = check_non_finite_val(
                            o_old * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_old, max_merge), "out_chunk_reduction_old_sub")) +
                            o_new * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_new, max_merge), "out_chunk_reduction_new_sub"))  
                        , "out_chunk_reduction");       // 去除sum除法，统一到最后
                sTensor_out(coord) = o_merge;
                toy_flash_attn_assert(THR_NUM >= HEAD_DIM_STRIDE);
            }
            __syncthreads();
            toy_flash_attn_assert(THR_NUM >= Q_CHUNK_SIZE);
            if (threadIdx.x < Q_CHUNK_SIZE) {   // TODO, 做成2个buffer
                int q_off = threadIdx.x;
                auto max_new = sTensor_warp_max(q_off, 0);
                auto sum_new = sTensor_warp_sum(q_off, 0);
                auto max_old = sTensor_last_max(q_off);
                auto sum_old = sTensor_last_sum(q_off);
                auto max_merge = max(max_old, max_new);
                auto sum_merge = check_non_finite_val(
                    sum_old * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_old, max_merge), "softmax_chunk_reduction_sum_old_sub")) + 
                    sum_new * exp2f(check_nan_val(softmax_scale_log2 * softmax_sub(max_new, max_merge), "softmax_chunk_reduction_sum_new_sub"))
                , "softmax_chunk_reduction_sum");

                sTensor_last_max(q_off) = max_merge;
                sTensor_last_sum(q_off) = sum_merge;
            }
            __syncthreads();
        }
    }
    {
        for(int linear = threadIdx.x; linear < cute::size(sTensor_out); linear += THR_NUM) {
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
template<typename scalar_t, typename inner_scalar_t, int Q_CHUNK_SIZE, int KV_CHUNK_SIZE, int HEAD_DIM_STRIDE>
void FlashAttnTrait<scalar_t, inner_scalar_t, Q_CHUNK_SIZE, KV_CHUNK_SIZE, HEAD_DIM_STRIDE>::check_inputs(
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
    TORCH_CHECK(round_up(q.size(2), 16) == HEAD_DIM_STRIDE);
    TORCH_CHECK(q.size(1) % k.size(2) == 0,
        "GQA requires num_q_heads to be divisible by num_kv_heads");
} 

template<typename scalar_t, typename inner_scalar_t, int Q_CHUNK_SIZE, int KV_CHUNK_SIZE, int HEAD_DIM_STRIDE>
torch::Tensor FlashAttnTrait<scalar_t, inner_scalar_t, Q_CHUNK_SIZE, KV_CHUNK_SIZE, HEAD_DIM_STRIDE>::flash_attn_varlen_with_block(
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
    auto num_q_heads = q.size(1);
    auto num_kv_heads = k.size(2);
    auto q_per_kv_group = num_q_heads / num_kv_heads;
    int64_t block_size = k.size(1);

    assert(head_dim <= 128);        // 泛化

    dim3 block(THR_NUM);
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
    kernel_wrapper<scalar_t, inner_scalar_t, Q_CHUNK_SIZE, KV_CHUNK_SIZE, HEAD_DIM_STRIDE><<<grid, block, TileLayout::size(param)>>>(param);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_varlen_with_block_v7_64", &FlashAttnTrait<at::BFloat16, float, 16, 32, 64>::flash_attn_varlen_with_block, "flash attn varlen with block");
    // m.def("flash_attn_varlen_with_block_v5_128", &FlashAttnTrait<at::BFloat16, float, 128, 16, 16>::flash_attn_varlen_with_block, "flash attn varlen with block");
}
