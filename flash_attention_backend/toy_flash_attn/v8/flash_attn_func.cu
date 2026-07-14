
#include <cmath>
#include <cstddef>
#include <numeric>
#include <cstdint>
#include <cassert>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/headeronly/util/BFloat16.h>
#include <type_traits>
#include <cute/tensor.hpp>
#include "helper.h"

/*
    scalar_t = at::BFloat16
    inner_scalar_t = float
    THR_NUM = 128
    constexpr int Q_BLOCK = 64; 
    constexpr int KV_SPLIT = 256;
    constexpr int KV_BLOCK = 32;
    constexpr int HEAD_DIM = 64;
 */
#define TEMPLATE_PARAM template<typename scalar_t, typename inner_scalar_t, int THR_NUM, int BLOCK_SIZE, int Q_BLOCK, int KV_SPLIT, int KV_BLOCK, int HEAD_DIM>
#define TEMPLATE_VAL scalar_t, inner_scalar_t, THR_NUM, BLOCK_SIZE, Q_BLOCK, KV_SPLIT, KV_BLOCK, HEAD_DIM

TEMPLATE_PARAM struct FlashAttnTrait;

TEMPLATE_PARAM __global__ void splitkv_kernel(typename FlashAttnTrait<TEMPLATE_VAL>::ParamSet param) {
    FlashAttnTrait<TEMPLATE_VAL>::splitkv_kernel(param);
}

TEMPLATE_PARAM __global__ void splitkv_combine_kernel(typename FlashAttnTrait<TEMPLATE_VAL>::ParamSet param) {
    FlashAttnTrait<TEMPLATE_VAL>::splitkv_combine_kernel(param);
}

TEMPLATE_PARAM struct FlashAttnTrait {
    // 使用 16x8x16算子
    static constexpr int32_t MMA_M = 16;
    static constexpr int32_t MMA_N = 8;
    static constexpr int32_t MMA_K = 16;

    static constexpr int32_t WARP_SIZE = 32;

    static_assert(std::is_same_v<inner_scalar_t, float>);   // inner_scalar_t暂时强制float

    struct ParamSet;
    struct TileLayout;

    // grid(batch_id, q_token_chunk_id, kv_head_id)
    static __device__ __forceinline__ void splitkv_kernel(ParamSet& param);
    static __device__ __forceinline__ void splitkv_combine_kernel(ParamSet& param);
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

TEMPLATE_PARAM
struct FlashAttnTrait<TEMPLATE_VAL>::ParamSet {
    const scalar_t* q;    // total_q x num_q_heads x head_dim
    const scalar_t* k;    // num_blocks x block_size x num_kv_heads x head_dim
    const scalar_t* v;
    scalar_t* out;
    float* split_lse;
    float* split_o;
    int64_t q_stride[3];  
    int64_t k_stride[4];
    int64_t v_stride[4];
    int64_t out_stride[3];
    int64_t split_lse_stride[3];
    int64_t split_o_stride[4];
    int64_t q_size[3];
    int64_t k_size[4];
    int64_t v_size[4];
    int64_t out_size[3];  
    int64_t split_lse_size[3];
    int64_t split_o_size[4];
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
    bool causal;
    int64_t num_kv_splits;

    __device__ __forceinline__ int64_t batch_size() const {
        return num_kv_splits > 1 ? gridDim.z / num_q_heads : gridDim.y;
    }
    __device__ __forceinline__ int64_t batch_id() const {
        return num_kv_splits > 1 ? blockIdx.z / num_q_heads : blockIdx.y;
    }
    __device__ __forceinline__ int64_t kv_split_id() const {
        return num_kv_splits > 1 ? blockIdx.y : 0;
    }
    __device__ __forceinline__ int64_t q_block_id() const {
        return blockIdx.x;
    }
    __device__ __forceinline__ int64_t q_head_id() const {
        return num_kv_splits > 1 ? blockIdx.z % num_q_heads : blockIdx.z;
    }
    __device__ __forceinline__ int64_t kv_head_id() const {
        return q_head_id() / q_per_kv_group;
    }
    __device__ __forceinline__ int64_t q_seqlen() const {
        return _q_seqlen;
    }
    __device__ __forceinline__ int64_t kv_seqlen() const {
        return _kv_seqlen;
    }
    __device__ __forceinline__ void device_init() {
        const int64_t bid = batch_id();
        _q_seqlen = static_cast<int64_t>(cu_seqlens_q[bid+1] - cu_seqlens_q[bid]);
        _kv_seqlen = static_cast<int64_t>(seqused_k[bid]);
    }

    int64_t _q_seqlen=0;
    int64_t _kv_seqlen=0;
};

TEMPLATE_PARAM
struct FlashAttnTrait<TEMPLATE_VAL>::TileLayout {
    static __device__ __forceinline__ TileLayout builder(char* smem, const ParamSet& param) {
        toy_flash_attn_assert(param.head_dim <= HEAD_DIM);
        TileLayout layout = layout_init(smem, param);
        return layout;
    }

    static __host__ __device__ __forceinline__ TileLayout layout_init(char* smem, const ParamSet& param) {
        TileLayout layout(param);

        int64_t offset = 0;
        layout.q = reinterpret_cast<scalar_t*>(smem + offset);
        offset += (Q_BLOCK * HEAD_DIM) * sizeof(scalar_t);

        offset = round_up(offset, alignof(scalar_t));
        layout.k[0] = reinterpret_cast<scalar_t*>(smem + offset);
        offset += (KV_BLOCK * HEAD_DIM) * sizeof(scalar_t);

        offset = round_up(offset, alignof(scalar_t));
        layout.v[0] = reinterpret_cast<scalar_t*>(smem + offset);
        offset += (KV_BLOCK * HEAD_DIM) * sizeof(scalar_t);

        offset = round_up(offset, alignof(scalar_t));
        layout.k[1] = reinterpret_cast<scalar_t*>(smem + offset);
        offset += (KV_BLOCK * HEAD_DIM) * sizeof(scalar_t);

        offset = round_up(offset, alignof(scalar_t));
        layout.v[1] = reinterpret_cast<scalar_t*>(smem + offset);
        offset += (KV_BLOCK * HEAD_DIM) * sizeof(scalar_t);

        offset = round_up(offset, alignof(scalar_t));
        layout.out = reinterpret_cast<scalar_t*>(smem + offset);
        offset += (Q_BLOCK * HEAD_DIM) * sizeof(scalar_t);

        layout._size = offset;
        return layout;
    }

    static __host__ __forceinline__ size_t size(const ParamSet& param) {
        TileLayout layout = layout_init((char*)nullptr, param);
        return layout._size;
    }

    __device__ __forceinline__ bool is_valid_kv(int64_t q_token_id, int64_t kv_seq_id, int64_t q_kv_offset, const int64_t* kv_win) {
        if (q_token_id < 0 || q_token_id >= param.q_seqlen() || kv_seq_id < 0 ||  kv_seq_id >= param.kv_seqlen()) {
            return false;
        }
        const int64_t kv_axis = q_token_id - q_kv_offset;
        const int64_t win[2] = {
            max(int64_t(0), kv_axis - kv_win[0]),
            min(param.kv_seqlen()-1, kv_axis + kv_win[1]),
        };
        return kv_seq_id >= win[0] && kv_seq_id <= win[1];
    }

    __device__ __forceinline__ void clear_by_vec(auto& s_block){
        // 向量化clear
        constexpr int data_per_vec = sizeof(cute::uint128_t) / sizeof(scalar_t);
        auto s_block_vec = cute::recast<cute::uint128_t>(s_block);
        constexpr auto vec_thr_layout = cute::make_layout(
            cute::make_shape(
                cute::Int<THR_NUM / (HEAD_DIM / data_per_vec)>{},
                cute::Int<HEAD_DIM / data_per_vec>{}
            ),
            cute::LayoutRight{}
        );
        auto thr_vec = cute::local_partition(
            s_block_vec,
            vec_thr_layout,
            threadIdx.x
        );
        cute::clear(thr_vec);   
    }
    
    __device__ __forceinline__ void clear_block_smem(int64_t phy_in_kv_block_id, auto&& s_kv) {
        auto s_block_kv = cute::local_tile(
            s_kv,
            cute::Shape<cute::Int<BLOCK_SIZE>, cute::Int<HEAD_DIM>>{},
            cute::make_coord(phy_in_kv_block_id, cute::_0{})
        );

        // 向量化clear
        clear_by_vec(s_block_kv);
    }

    __device__ __forceinline__ void load_block_smem(int64_t phy_block_id, int64_t phy_in_kv_block_id, int64_t block_seq_begin, auto&& g_kv, auto&& s_kv) {
        auto g_block_kv = cute::local_tile(
            g_kv(phy_block_id, cute::_, param.kv_head_id(), cute::_),
            cute::Shape<cute::Int<BLOCK_SIZE>, cute::Int<HEAD_DIM>>{},
            cute::make_coord(cute::_0{}, cute::_0{})
        );
        auto s_block_kv = cute::local_tile(
            s_kv,
            cute::Shape<cute::Int<BLOCK_SIZE>, cute::Int<HEAD_DIM>>{},
            cute::make_coord(phy_in_kv_block_id, cute::_0{})
        );

        // 向量化clear
        clear_by_vec(s_block_kv);
        
        // 拷贝
        auto block_identity = cute::make_identity_tensor(
            cute::Shape<cute::Int<BLOCK_SIZE>, cute::Int<HEAD_DIM>>{}
        );
        constexpr int32_t data_per_thr = sizeof(cute::uint128_t) / sizeof(scalar_t);
        constexpr int32_t thr_col = HEAD_DIM / data_per_thr;
        constexpr int32_t thr_row = THR_NUM / thr_col;
        constexpr auto thr_layout = cute::make_layout(
            cute::make_shape(cute::Int<thr_row>{}, cute::Int<thr_col>{}),
            cute::LayoutRight{}
        );
        constexpr auto val_layout = cute::make_layout(
            cute::make_shape(cute::_1{}, cute::Int<data_per_thr>{}),
            cute::LayoutRight{}
        );
        cute::TiledCopy copy_a = cute::make_tiled_copy(
            cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, scalar_t>{},
            thr_layout, val_layout
        );
        auto thr_copy = copy_a.get_slice(threadIdx.x);
        if (threadIdx.x < cute::size(copy_a)) {
            auto src = thr_copy.partition_S(g_block_kv);
            auto dst = thr_copy.partition_D(s_block_kv);
            auto src_id = thr_copy.partition_S(block_identity);
            int valid_rows = min(int64_t{BLOCK_SIZE}, param.kv_seqlen() - block_seq_begin);
            auto pred = cute::make_tensor<bool>(cute::shape(src));
            for (int i = 0; i < cute::size(pred); i++) {
                pred(i) = cute::get<0>(src_id(i)) < valid_rows;
            }
            cute::copy_if(copy_a, pred, src, dst);
        }
    }

    __device__ __forceinline__ void load_kv_group(int64_t kv_block_id, auto&& gK, auto& gV, auto&& sK, auto&& sV) {
        CUTE_UNROLL
        for(int b=0; b < KV_BLOCK/BLOCK_SIZE; b++){
            int block_seq_begin = KV_SPLIT * param.kv_split_id() + KV_BLOCK * kv_block_id + b * BLOCK_SIZE;
            if (block_seq_begin < param.kv_seqlen()) {
                int phy_block_id = param.block_table[
                    param.bt_stride[0] * param.batch_id() + 
                    param.bt_stride[1] * block_seq_begin / BLOCK_SIZE
                ];

                load_block_smem(phy_block_id, b, block_seq_begin, gK, sK);
                load_block_smem(phy_block_id, b, block_seq_begin, gV, sV);
            } else {
                clear_block_smem(b, sK);
                clear_block_smem(b, sV);
            }
        }
        cute::cp_async_fence();
    }

    __device__ __forceinline__ void load_q_block_smem(int64_t q_block_id, auto&& gQ_b, auto&& sQ) {
        auto q_head = gQ_b(cute::_, param.q_head_id(), cute::_);
        auto q_block = cute::local_tile(
            q_head, 
            cute::Shape<cute::Int<Q_BLOCK>, cute::Int<HEAD_DIM>>{},
            cute::make_coord(q_block_id, cute::_0{})
        );
        auto q_idx = cute::make_identity_tensor(cute::shape(q_head));
        auto q_block_identity = cute::local_tile(
            q_idx, 
            cute::Shape<cute::Int<Q_BLOCK>, cute::Int<HEAD_DIM>>{},
            cute::make_coord(q_block_id, cute::_0{})
        );
        constexpr int32_t data_per_thr = sizeof(cute::uint128_t) / sizeof(scalar_t);
        constexpr int32_t thr_col = HEAD_DIM / data_per_thr;
        constexpr int32_t thr_row = THR_NUM / thr_col;
        constexpr auto thr_layout = cute::make_layout(
            cute::make_shape(cute::Int<thr_row>{}, cute::Int<thr_col>{}),
            cute::LayoutRight{}
        );
        constexpr auto val_layout = cute::make_layout(
            cute::make_shape(cute::_1{}, cute::Int<data_per_thr>{}),
            cute::LayoutRight{}
        );
        cute::TiledCopy copy_a = cute::make_tiled_copy(
            cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, scalar_t>{}, 
            thr_layout, val_layout
        );

        // 向量化clear
        clear_by_vec(sQ);
        auto thr_copy = copy_a.get_slice(threadIdx.x);
        if (threadIdx.x < cute::size(copy_a)) {
            auto src = thr_copy.partition_S(q_block);
            auto dst = thr_copy.partition_D(sQ);
            auto src_id = thr_copy.partition_S(q_block_identity);
            auto pred = cute::make_tensor<bool>(cute::shape(src));
            for(int i=0; i<cute::size(pred); i++){
                pred(i) = cute::elem_less(src_id(i), cute::shape(q_head));
            }
            // cute::clear(dst);
            cute::copy_if(copy_a, pred, src, dst);
        }
    }

    scalar_t* q;
    scalar_t* k[2];
    scalar_t* v[2];
    scalar_t* out;

private:
    __device__ __host__ TileLayout(const ParamSet& p):param(p) {}

    size_t _size;

    const ParamSet& param;
};

TEMPLATE_PARAM __device__ __forceinline__ void FlashAttnTrait<TEMPLATE_VAL>::splitkv_kernel(ParamSet& param) {
    extern __shared__ __align__(16) char smem[];
    param.device_init();
    TileLayout layout = TileLayout::builder(smem, param);

    const inner_scalar_t softmax_scale_log2 = M_LOG2Ef32 / sqrt((float)param.head_dim);

    const int64_t kv_win[2] = {
        param.window_size[0] == -1 ? param.kv_seqlen() : param.window_size[0],
        param.causal ? 0 : 
            param.window_size[1] == -1 ? param.kv_seqlen() : param.window_size[1]
    };

    const int64_t q_kv_offset = param.q_seqlen() - param.kv_seqlen();

    // toy_flash_attn_assert(param.q_size[2] == HEAD_DIM); // 不等于，会导致global memory读取性能劣化，可以测试下
    // toy_flash_attn_assert(param.q_stride[2] == 1);
    // toy_flash_attn_assert(param.k_size[3] == HEAD_DIM);
    // toy_flash_attn_assert(param.k_stride[3] == 1);
    // toy_flash_attn_assert(param.v_size[3] == HEAD_DIM);
    // toy_flash_attn_assert(param.v_stride[3] == 1);
    auto gK = cute::make_tensor(
        cute::make_gmem_ptr(param.k),
        cute::make_shape(param.k_size[0], param.k_size[1], param.k_size[2], param.k_size[3]),
        cute::make_stride(param.k_stride[0], param.k_stride[1], param.k_stride[2], cute::_1{})
    );
    auto gV = cute::make_tensor(
        cute::make_gmem_ptr(param.v),
        cute::make_shape(param.v_size[0], param.v_size[1], param.v_size[2], param.v_size[3]),
        cute::make_stride(param.v_stride[0], param.v_stride[1], param.v_stride[2], cute::_1{})
    );
    auto gQ_b = cute::make_tensor(
        cute::make_gmem_ptr(
            param.q + param.q_stride[0] * param.cu_seqlens_q[param.batch_id()]
        ),
        cute::make_shape(int64_t{param.q_seqlen()}, param.q_size[1], param.q_size[2]),
        cute::make_stride(param.q_stride[0], param.q_stride[1], cute::_1{})
    );
    // auto gOut_b = cute::make_tensor(
    //     cute::make_gmem_ptr(
    //         param.out + param.out_stride[0] * param.cu_seqlens_q[param.batch_id()]
    //     ),
    //     cute::make_shape(int64_t{param.q_seqlen()}, param.out_size[1], param.out_size[2]),
    //     cute::make_stride(param.out_stride[0], param.out_stride[1], param.out_stride[2])
    // );

    auto g_split_lse_batch = cute::make_tensor(
        cute::make_gmem_ptr(
            param.split_lse + param.split_lse_stride[0] * param.cu_seqlens_q[param.batch_id()]
        ),
        cute::make_shape(int64_t{param.q_seqlen()}, param.split_lse_size[1], param.split_lse_size[2]),
        cute::make_stride(param.split_lse_stride[0], param.split_lse_stride[1], param.split_lse_stride[2])
    );
    auto g_split_o_batch = cute::make_tensor(
        cute::make_gmem_ptr(
            param.split_o + param.split_o_stride[0] * param.cu_seqlens_q[param.batch_id()]
        ),
        cute::make_shape(int64_t{param.q_seqlen()}, param.split_o_size[1], param.split_o_size[2], param.split_o_size[3]),
        cute::make_stride(param.split_o_stride[0], param.split_o_stride[1], param.split_o_stride[2], param.split_o_stride[3])
    );

    auto sQ = cute::make_tensor(
        cute::make_smem_ptr(layout.q),
        cute::composition(
            cute::Swizzle<3, 3>{},
            cute::make_layout(
                cute::make_shape(cute::Int<Q_BLOCK>{}, cute::Int<HEAD_DIM>{}),
                cute::LayoutRight{}
            )
        )
    );
    auto k_layout = cute::composition(
            cute::Swizzle<3, 3>{},
            cute::make_layout(
                cute::make_shape(cute::Int<KV_BLOCK>{}, cute::Int<HEAD_DIM>{}),
                cute::LayoutRight{}
            )
        );
    auto v_layout = cute::composition(
            cute::Swizzle<3, 3>(),
            cute::make_layout(
                cute::make_shape(cute::Int<KV_BLOCK>{}, cute::Int<HEAD_DIM>{}),
                cute::LayoutRight{}
            )
        );
    auto vt_layout = cute::composition(
            cute::Swizzle<3, 3>(),
            cute::make_layout(
                cute::make_shape(cute::Int<HEAD_DIM>{}, cute::Int<KV_BLOCK>{}),
                cute::LayoutLeft{}
            )
        );
    std::array _sK = {
        cute::make_tensor(cute::make_smem_ptr(layout.k[0]), k_layout),
        cute::make_tensor(cute::make_smem_ptr(layout.k[1]), k_layout)
    }; 
    std::array _sV = {
        cute::make_tensor( cute::make_smem_ptr(layout.v[0]), v_layout),
        cute::make_tensor( cute::make_smem_ptr(layout.v[1]), v_layout)
    };
    std::array _sVt = {
        cute::make_tensor( _sV[0].data(), vt_layout),
        cute::make_tensor( _sV[1].data(), vt_layout)
    };
    // auto sK = _sK[0];
    // auto sV = _sV[0];
    // auto sVt = _sVt[0];

    // 定义算子 & 线程layout
    auto mma_qk = cute::make_tiled_mma(
        cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>{},
        cute::Layout<cute::Shape<cute::Int<THR_NUM/WARP_SIZE>, cute::_1, cute::_1>>{},
        cute::Tile<cute::Int<Q_BLOCK>, cute::Int<KV_BLOCK>, cute::Int<MMA_K>>{}
    );
    auto mma_pv = cute::make_tiled_mma(
        cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>{},
        cute::Layout<cute::Shape<cute::Int<THR_NUM/WARP_SIZE>, cute::_1, cute::_1>>{},
        cute::Tile<cute::Int<Q_BLOCK>, cute::Int<HEAD_DIM>, cute::Int<MMA_K>>{}
    );
    auto smem_tiled_copy_Q = cute::make_tiled_copy_A(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, scalar_t>{}, mma_qk);
    auto smem_tiled_copy_K = cute::make_tiled_copy_B(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, scalar_t>{}, mma_qk);
    auto smem_tiled_copy_Vt = cute::make_tiled_copy_B(cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, scalar_t>{}, mma_pv);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(threadIdx.x);
    auto thr_mma_qk = mma_qk.get_thread_slice(threadIdx.x);
    auto thr_mma_pv = mma_pv.get_thread_slice(threadIdx.x);

    // 定义寄存器tensor
    auto score = cute::partition_fragment_C(thr_mma_qk, cute::Shape<cute::Int<Q_BLOCK>, cute::Int<KV_BLOCK>>{});
    auto score_rowcol_view = cute::make_tensor(score.data(), convert_layout_acc_rowcol(score.layout()));
    auto score_ids = thr_mma_qk.partition_C(cute::make_identity_tensor(cute::Shape<cute::Int<Q_BLOCK>, cute::Int<KV_BLOCK>>{}));
    auto score_ids_rowcol = cute::make_tensor( score_ids.data(), convert_layout_acc_rowcol(score_ids.layout()));

    auto old_max = cute::make_tensor<float>(cute::Shape<cute::Int<cute::size<0>(score_rowcol_view)>>{});
    auto old_sum = cute::make_tensor<float>(cute::Shape<cute::Int<cute::size<0>(score_rowcol_view)>>{});
    auto acc_o = cute::partition_fragment_C(thr_mma_pv, cute::Shape<cute::Int<Q_BLOCK>, cute::Int<HEAD_DIM>>{});;
    auto acc_o_rowcol = cute::make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
    auto acc_o_ids = thr_mma_pv.partition_C(cute::make_identity_tensor(cute::Shape<cute::Int<Q_BLOCK>, cute::Int<HEAD_DIM>>{}));
    auto acc_o_ids_rowcol = cute::make_tensor(acc_o_ids.data(),convert_layout_acc_rowcol(acc_o_ids.layout()));
    cute::clear(acc_o);
    cute::fill(old_max, -INFINITY);
    cute::clear(old_sum);

    // auto block_max = cute::make_tensor<float>(cute::Shape<cute::Int<cute::size<0>(score_rowcol_view)>>{});
    // auto block_sum = cute::make_tensor<float>(cute::Shape<cute::Int<cute::size<0>(score_rowcol_view)>>{});
    // auto block_o = cute::partition_fragment_C(thr_mma_pv, cute::Shape<cute::Int<Q_BLOCK>, cute::Int<HEAD_DIM>>{});
    // auto block_o_rowcol = cute::make_tensor(block_o.data(),convert_layout_acc_rowcol(block_o.layout()));
    // load sQ
    layout.load_q_block_smem(param.q_block_id(), gQ_b, sQ);
    static_assert(KV_SPLIT % KV_BLOCK == 0);
    static_assert(KV_BLOCK % BLOCK_SIZE == 0);

    if (KV_SPLIT/KV_BLOCK > 0) {
        layout.load_kv_group(0, gK, gV, _sK[0], _sV[0]);
        cute::cp_async_wait<0>();
        __syncthreads();
    }
    for(int kv_block_id = 0; kv_block_id < KV_SPLIT/KV_BLOCK; kv_block_id++)
    {   
        bool has_next_block = kv_block_id != KV_SPLIT/KV_BLOCK-1;
        if (has_next_block) { // prefetch sK & sV
            int kv_nxt_ptr = (kv_block_id + 1) % 2;
            layout.load_kv_group(kv_block_id+1, gK, gV, _sK[kv_nxt_ptr], _sV[kv_nxt_ptr]);
        }

        auto sK = _sK[kv_block_id % 2];
        auto sV = _sV[kv_block_id % 2];
        auto sVt = _sVt[kv_block_id % 2];
        // Q @ K
        cute::clear(score);
        for(int64_t iK = 0; iK < HEAD_DIM / MMA_K; iK++) {
            auto sQ_K = cute::local_tile(sQ,
                cute::make_shape(cute::Int<Q_BLOCK>{}, cute::Int<MMA_K>{}),
                cute::make_coord(cute::_0{}, iK) 
            );
            auto sK_K = cute::local_tile(sK, 
                cute::make_shape(cute::Int<KV_BLOCK>{}, cute::Int<MMA_K>{}),
                cute::make_coord(cute::_0{}, iK)
            ); 
            auto tCrA = thr_mma_qk.partition_fragment_A(sQ_K);
            auto tCrB = thr_mma_qk.partition_fragment_B(sK_K);
            auto tXsA = smem_thr_copy_Q.partition_S(sQ_K);
            auto tXsB = smem_thr_copy_K.partition_S(sK_K);
            auto tXrA = smem_thr_copy_Q.retile_D(tCrA);
            auto tXrB = smem_thr_copy_K.retile_D(tCrB);
            cute::copy(smem_tiled_copy_Q, tXsA, tXrA);
            cute::copy(smem_tiled_copy_K, tXsB, tXrB);
            cute::gemm(thr_mma_qk, tCrA, tCrB, score);
        }

        // score mask
        CUTE_UNROLL
        for (int linear = 0; linear < cute::size(score); linear++) {
            auto coord = score_ids(linear);
            auto [r, c] = coord;
            int64_t q_token_id = param.q_block_id() * Q_BLOCK + r; 
            int64_t kv_token_id = param.kv_split_id() * KV_SPLIT + kv_block_id * KV_BLOCK + c;
            score(linear) = layout.is_valid_kv(q_token_id, kv_token_id, q_kv_offset, kv_win) 
                ? score(linear)
                : -INFINITY;
        }
        // softmax
        CUTE_UNROLL
        for (int r = 0; r < cute::size<0>(score_rowcol_view); r++) {
            auto op_max = [] __device__ (auto a, auto b) {return a>b ? a : b;};
            auto op_sum = [] __device__ (auto a, auto b) {return check_non_finite_val(a+b, "softmax_score_reduce");};
            inner_scalar_t r_max = reduce_thr(score_rowcol_view(r, cute::_), op_max);
            r_max = AllReduce<4>::run(r_max, op_max);   // 4是固定值， 因为m16n8k16算子里，每lane是2行2列4个val, n8/2列 = 4个lane归约
            
            float new_max = op_max(old_max(r), r_max);
            float alpha = old_sum(r) == 0.0f ? 0.0f : exp2f((old_max(r) - new_max) * softmax_scale_log2);

            CUTE_UNROLL
            for (int c = 0; c < cute::size<1>(acc_o_rowcol); ++c) {
                acc_o_rowcol(r, c) *= alpha;
            }

            if (new_max == -INFINITY) {
                cute::fill(score_rowcol_view(r, cute::_), 0.0f);
            } else {
                CUTE_UNROLL
                for (int c = 0; c < cute::size<1>(score_rowcol_view); ++c) {
                    score_rowcol_view(r, c) = exp2f(softmax_sub(score_rowcol_view(r, c), new_max) * softmax_scale_log2);
                }
            }
            inner_scalar_t r_sum = reduce_thr(score_rowcol_view(r, cute::_), op_sum);
            r_sum = AllReduce<4>::run(r_sum, op_sum);
            // old_sum(r) = old_sum(r) * alpha + r_sum;
            old_sum(r) = fmaf(old_sum(r), alpha, r_sum);
            old_max(r) = new_max;
        }

        // P @ V
        auto tCrP_bf16 = convert_fp32_to_bf16(score);
        auto tCrP = cute::make_tensor(
            tCrP_bf16.data(),
            convert_layout_acc_Aregs(tCrP_bf16.layout())
        );

        static_assert(decltype(cute::size<2>(tCrP))::value == KV_BLOCK / MMA_K);
        for(int64_t iK = 0; iK < KV_BLOCK / MMA_K; iK++) {
            auto sVt_K = cute::local_tile(
                sVt,
                cute::make_shape(
                    cute::Int<HEAD_DIM>{},
                    cute::Int<MMA_K>{}
                ),
                cute::make_coord(cute::_0{}, iK)
            );
            // auto tCrA = tCrP(cute::_, cute::_, iK);
            auto tCrB = thr_mma_pv.partition_fragment_B(sVt_K);
            auto tXsB = smem_thr_copy_Vt.partition_S(sVt_K);
            auto tXrB = smem_thr_copy_Vt.retile_D(tCrB);
            cute::copy(smem_tiled_copy_Vt, tXsB, tXrB);
            cute::gemm(thr_mma_pv, 
                tCrP(cute::_, cute::_, iK), 
                tCrB(cute::_, cute::_, cute::_0{}), 
                acc_o);
        }

        if (has_next_block) {
            cute::cp_async_wait<0>();
            __syncthreads();
        }
    }
    if (param.num_kv_splits > 1) {
        // write split lse
        const int32_t lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id % 4 == 0) {
            CUTE_UNROLL
            for (int r = 0; r < cute::size<0>(score_ids_rowcol); r++) {
                auto coord = score_ids_rowcol(r, 0);
                int64_t q_token_id = param.q_block_id() * Q_BLOCK + cute::get<0>(coord);

                if (q_token_id < param.q_seqlen()) {
                    float split_lse = (old_sum(r) == 0.0f)
                        ? -INFINITY
                        : old_max(r) * softmax_scale_log2 + log2f(old_sum(r));
                    g_split_lse_batch(q_token_id, param.q_head_id(), param.kv_split_id()) = split_lse;
                }
            }
        }
        // write split output
        CUTE_UNROLL
        for (int r = 0; r < cute::size<0>(acc_o_rowcol); r++) {
            const inner_scalar_t inv_sum = old_sum(r) > 0.0f ? 1.0f / old_sum(r) : 0.0f;
            CUTE_UNROLL
            for (int c = 0; c < cute::size<1>(acc_o_rowcol); c++) {
                auto coord = acc_o_ids_rowcol(r, c);
                auto [q_token_offset, head_dim_id] = coord;

                const int64_t q_token_id =
                    param.q_block_id() * Q_BLOCK + q_token_offset;

                if (q_token_id < param.q_seqlen() &&
                    head_dim_id < param.head_dim) {
                    g_split_o_batch(
                        q_token_id,
                        param.q_head_id(),
                        param.kv_split_id(),
                        head_dim_id
                    ) = acc_o_rowcol(r, c) * inv_sum;
                }
            }
        }
    } else {
        // num_kv_splits == 1: write directly to out, skip combine
        auto g_out_batch = cute::make_tensor(
            cute::make_gmem_ptr(
                param.out + param.out_stride[0] * param.cu_seqlens_q[param.batch_id()]
            ),
            cute::make_shape(int64_t{param.q_seqlen()}, param.out_size[1], param.out_size[2]),
            cute::make_stride(param.out_stride[0], param.out_stride[1], param.out_stride[2])
        );
        CUTE_UNROLL
        for (int r = 0; r < cute::size<0>(acc_o_rowcol); r++) {
            const inner_scalar_t inv_sum = old_sum(r) > 0.0f ? 1.0f / old_sum(r) : 0.0f;
            CUTE_UNROLL
            for (int c = 0; c < cute::size<1>(acc_o_rowcol); c++) {
                auto coord = acc_o_ids_rowcol(r, c);
                auto [q_token_offset, head_dim_id] = coord;

                const int64_t q_token_id =
                    param.q_block_id() * Q_BLOCK + q_token_offset;

                if (q_token_id < param.q_seqlen() &&
                    head_dim_id < param.head_dim) {
                    g_out_batch(
                        q_token_id,
                        param.q_head_id(),
                        head_dim_id
                    ) = scalar_t(acc_o_rowcol(r, c) * inv_sum);
                }
            }
        }
    }
}

TEMPLATE_PARAM __device__ __forceinline__
void FlashAttnTrait<TEMPLATE_VAL>::splitkv_combine_kernel(ParamSet& param) {
    const int64_t q_token_id = blockIdx.x;
    const int64_t q_head_id = blockIdx.y;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;

    if (q_token_id >= param.out_size[0] ||
        q_head_id >= param.num_q_heads) {
        return;
    }

    auto g_split_lse = cute::make_tensor(
        cute::make_gmem_ptr(param.split_lse),
        cute::make_shape( param.split_lse_size[0], param.split_lse_size[1],  param.split_lse_size[2]),
        cute::make_stride( param.split_lse_stride[0], param.split_lse_stride[1], param.split_lse_stride[2])
    );

    auto g_split_o = cute::make_tensor(
        cute::make_gmem_ptr(param.split_o),
        cute::make_shape( param.split_o_size[0], param.split_o_size[1], param.split_o_size[2], param.split_o_size[3]),
        cute::make_stride( param.split_o_stride[0], param.split_o_stride[1], param.split_o_stride[2], param.split_o_stride[3])
    );

    auto g_out = cute::make_tensor(
        cute::make_gmem_ptr(param.out),
        cute::make_shape(param.out_size[0], param.out_size[1], param.out_size[2] ),
        cute::make_stride( param.out_stride[0], param.out_stride[1], param.out_stride[2])
    );

    auto op_max = [] __device__ (float a, float b) { return a > b ? a : b; };
    auto op_sum = [] __device__ (float a, float b) { return check_non_finite_val(a+b, "splitkv_combine_kernel_sum"); };

    // reduce max lse across splits
    inner_scalar_t local_lse_max = -INFINITY;

    for (int64_t split_id = lane_id; split_id < param.num_kv_splits; split_id += WARP_SIZE) {
        local_lse_max = op_max(local_lse_max, g_split_lse(q_token_id, q_head_id, split_id));
    }

    const inner_scalar_t lse_max = AllReduce<WARP_SIZE>::run(local_lse_max, op_max);

    // all splits are invalid
    if (!isfinite(lse_max)) {
        for (int64_t head_dim_id = lane_id; head_dim_id < param.head_dim; head_dim_id += WARP_SIZE) {
            g_out(q_token_id, q_head_id, head_dim_id) = scalar_t(0.0f);
        }
        return;
    }

    // reduce split weights
    inner_scalar_t local_weight_sum = 0.0f;
    for (int64_t split_id = lane_id; split_id < param.num_kv_splits; split_id += WARP_SIZE) {
        const inner_scalar_t split_lse = g_split_lse(q_token_id, q_head_id, split_id); 
        local_weight_sum += isfinite(split_lse) ? exp2f(split_lse - lse_max) : 0.0f;
    }
    const inner_scalar_t weight_sum = AllReduce<WARP_SIZE>::run(local_weight_sum, op_sum);

    // combine normalized partial outputs
    for (int64_t head_dim_id = lane_id; head_dim_id < param.head_dim; head_dim_id += WARP_SIZE) {
        inner_scalar_t out_value = 0.0f;
        for (int64_t split_id = 0; split_id < param.num_kv_splits; split_id++) {
            const inner_scalar_t split_lse = g_split_lse(q_token_id, q_head_id, split_id);
            if (isfinite(split_lse)) {
                const inner_scalar_t weight = exp2f(split_lse - lse_max) / weight_sum;
                out_value += weight 
                    * g_split_o(
                        q_token_id,
                        q_head_id,
                        split_id,
                        head_dim_id
                    );
            }
        }
        g_out(q_token_id, q_head_id, head_dim_id) =
            scalar_t(out_value);
    }
}

TEMPLATE_PARAM
void FlashAttnTrait<TEMPLATE_VAL>::check_inputs(
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
    TORCH_CHECK(round_up(q.size(2), 16) == HEAD_DIM);
    TORCH_CHECK(q.size(1) % k.size(2) == 0,
        "GQA requires num_q_heads to be divisible by num_kv_heads");
} 

TEMPLATE_PARAM torch::Tensor FlashAttnTrait<TEMPLATE_VAL>::flash_attn_varlen_with_block(
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
    auto total_q = q.size(0);
    auto num_q_heads = q.size(1);
    auto num_kv_heads = k.size(2);
    auto q_per_kv_group = num_q_heads / num_kv_heads;
    int64_t num_kv_splits = (max_seqlen_k + KV_SPLIT - 1) / KV_SPLIT;
    toy_flash_attn_assert(k.size(1) == BLOCK_SIZE);

    static_assert(Q_BLOCK % MMA_M == 0);
    static_assert(KV_BLOCK % std::lcm(MMA_M, MMA_K) == 0);
    static_assert(HEAD_DIM % std::lcm(MMA_M, MMA_K) == 0);
    static_assert(KV_SPLIT % KV_BLOCK == 0);
    toy_flash_attn_assert(head_dim == HEAD_DIM);

    dim3 block(THR_NUM);
    dim3 grid;

    if (num_kv_splits > 1) {
        int grid_x = (max_seqlen_q + Q_BLOCK - 1) / Q_BLOCK;
        int grid_y = (max_seqlen_k + KV_SPLIT - 1) / KV_SPLIT;
        int grid_z = batch_size * num_q_heads;
        grid = dim3(grid_x, grid_y, grid_z);
    } else {
        int grid_x = (max_seqlen_q + Q_BLOCK - 1) / Q_BLOCK;
        int grid_y = batch_size;
        int grid_z = num_q_heads;
        grid = dim3(grid_x, grid_y, grid_z);
    }
    auto split_lse = torch::full(
        {total_q, num_q_heads, num_kv_splits},
        -INFINITY,
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(q.device())
    );
    auto split_o = torch::zeros(
        {total_q, num_q_heads, num_kv_splits, head_dim},
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(q.device())
    );
    ParamSet param = {
        .q = q.const_data_ptr<scalar_t>(),
        .k = k.const_data_ptr<scalar_t>(),
        .v = v.const_data_ptr<scalar_t>(),
        .out = out.data_ptr<scalar_t>(),
        .split_lse = split_lse.data_ptr<float>(),
        .split_o = split_o.data_ptr<float>(),
        .q_stride = {q.stride(0), q.stride(1), q.stride(2)},
        .k_stride = {k.stride(0), k.stride(1), k.stride(2), k.stride(3)},
        .v_stride = {v.stride(0), v.stride(1), v.stride(2), v.stride(3)},
        .out_stride = {out.stride(0), out.stride(1), out.stride(2)},
        .split_lse_stride = {split_lse.stride(0), split_lse.stride(1), split_lse.stride(2)},
        .split_o_stride = {split_o.stride(0), split_o.stride(1), split_o.stride(2), split_o.stride(3)},
        .q_size = {q.size(0), q.size(1), q.size(2)},
        .k_size = {k.size(0), k.size(1), k.size(2), k.size(3)},
        .v_size = {v.size(0), v.size(1), v.size(2), v.size(3)},
        .out_size = {out.size(0), out.size(1), out.size(2)},
        .split_lse_size = {split_lse.size(0), split_lse.size(1), split_lse.size(2)},
        .split_o_size = {split_o.size(0), split_o.size(1), split_o.size(2), split_o.size(3)},
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
        .causal = causal,
        .num_kv_splits = num_kv_splits
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ::splitkv_kernel<TEMPLATE_VAL><<<grid, block, TileLayout::size(param), stream>>>(param);
    // combine
    if (num_kv_splits > 1) {
        dim3 combine_block(WARP_SIZE);
        dim3 combine_grid(total_q, num_q_heads);

        ::splitkv_combine_kernel<TEMPLATE_VAL> <<<combine_grid, combine_block, 0, stream>>>(param);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_varlen_with_block_v8_64", &FlashAttnTrait<at::BFloat16, float, 128, 16, 64, 256, 32, 64>::flash_attn_varlen_with_block, "flash attn varlen with block");
    // m.def("flash_attn_varlen_with_block_v5_128", &FlashAttnTrait<at::BFloat16, float, 128, 16, 16>::flash_attn_varlen_with_block, "flash attn varlen with block");
}
