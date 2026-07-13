#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


# define TOY_FLASH_ATTN_ASSERT_ON
// # define DEBUG_NUMERIC
// # define DEBUG_FLASH_ATTN_TRACE

#ifdef TOY_FLASH_ATTN_ASSERT_ON
#define toy_flash_attn_assert(x) assert(x)
#else
#define toy_flash_attn_assert(x) ((void)0)
#endif

template<typename scalar_t>
__device__ inline scalar_t check_nan_val(scalar_t x, const char* tag) {
#ifdef DEBUG_NUMERIC
    if (isnan(x)) {
#ifdef __CUDA_ARCH__
        printf("%s nan: %f tx=%d ty=%d bx=%d by=%d bz=%d\n",
            tag, static_cast<double>(x), threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z);
#else
        printf("%s nan: %f\n", tag, static_cast<double>(x));
#endif
    }
#endif
    return x;
}

template<typename scalar_t>
__device__ inline scalar_t check_non_finite_val(scalar_t x, const char* tag) {
#ifdef DEBUG_NUMERIC
    if (!isfinite(x)) {
#ifdef __CUDA_ARCH__
        printf("%s non-finite: %f tx=%d ty=%d bx=%d by=%d bz=%d\n",
               tag, static_cast<double>(x), threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z);
#else
        printf("%s non-finite: %f\n", tag, static_cast<double>(x));
#endif
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

// online softmax sub 只需要处理-inf
template<typename scalar_t>
__device__ inline scalar_t softmax_sub(scalar_t a, scalar_t b) { 
    return (isfinite(a) && isfinite(b)) ? a - b : scalar_t{-INFINITY};
}

// online softmax div 除0表示无效位
template<typename scalar_t>
__device__ inline scalar_t softmax_div(scalar_t a, scalar_t b) {
    return (b != scalar_t(0)) ? a/b : scalar_t{0};
}

template<typename scalar_t>
__device__ inline scalar_t valid_mul(scalar_t valid_a, scalar_t b) {
    return valid_a == scalar_t(0) ? scalar_t(0) : valid_a * b;
}

__device__ inline void busy_wait(unsigned long long delay_cycles) {
    unsigned long long start = clock64();
    while (clock64() - start < delay_cycles) {
    }
}
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

template<typename Layout>
__device__ __forceinline__ 
auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(cute::size<0>(acc_layout))::value == 4);
    static_assert(decltype(cute::rank(acc_layout))::value == 3);

    // (4, MMA_M, MMA_N)
    //   ↓ 把第0维4拆成(2,2)
    // ((2,2), MMA_M, MMA_N)
    auto l = cute::logical_divide(
        acc_layout,
        cute::Shape<cute::_2>{}
    );

    // row = (第二个2, MMA_M)
    // col = (第一个2, MMA_N)
    return cute::make_layout(
        cute::make_layout(
            cute::get<0, 1>(l),
            cute::get<1>(l)
        ),
        cute::make_layout(
            cute::get<0, 0>(l),
            cute::get<2>(l)
        )
    );
}

template<typename Tensor>
__device__ __forceinline__ 
auto convert_fp32_to_bf16(Tensor const& x) {
    constexpr int N = decltype(cute::size(x))::value;
    static_assert(N % 2 == 0);
    cutlass::NumericArrayConverter<cute::bfloat16_t, float, N> convert;
    auto out = convert(*reinterpret_cast<cutlass::Array<float, N> const*>(x.data()));
    return cute::make_tensor(
        cute::make_rmem_ptr<cutlass::bfloat16_t>(out.data()),
        x.layout()
    );
}

template <class Layout>
__device__ __forceinline__
auto convert_layout_acc_Aregs(Layout const& acc_layout) {
    static_assert(decltype(cute::rank(acc_layout))::value == 3);
    static_assert(decltype(cute::size<0>(acc_layout))::value == 4);

    auto divided = cute::logical_divide(
        acc_layout,
        cute::Shape<cute::Underscore, cute::Underscore, cute::_2>{}
    );

    return cute::make_layout(
        cute::make_layout(
            cute::get<0>(divided),
            cute::get<2, 0>(divided)
        ),
        cute::get<1>(divided),
        cute::get<2, 1>(divided)
    );
}

template<class Op>
__device__ __forceinline__
auto reduce_thr(auto const& col_tensor, Op op) {
    static_assert(decltype(cute::size(col_tensor))::value > 0);
    auto ret = col_tensor(0);
    CUTE_UNROLL
    for (int c = 1; c < cute::size(col_tensor); ++c)
        ret = op(ret, col_tensor(c));
    return ret;
}

template<int WIDTH>
class AllReduce{
public:
    template<class T, class Op>
    __device__ __forceinline__
    static T run(T x, Op op) {
        x = AllReduce<WIDTH/2>::run(x, op);
        return op(x, __shfl_xor_sync(0xffffffff, x, WIDTH/2));
    }
};
template<>
class AllReduce<2>{
public:
    template<class T, class Op>
    __device__ __forceinline__
    static T run(T x, Op op) {
        return op(x, __shfl_xor_sync(0xffffffff, x, 1));
    }
};