#include "torch/headeronly/util/BFloat16.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


// # define TOY_FLASH_ATTN_ASSERT_ON
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

template<typename T>
struct WmmaElement;

template<>
struct WmmaElement<at::BFloat16> {
    using type = __nv_bfloat16;

    static __device__ inline type* ptr(at::BFloat16* p) {
        return reinterpret_cast<type*>(p);
    }

    static __device__ inline const type* ptr(const at::BFloat16* p) {
        return reinterpret_cast<const type*>(p);
    }
};

template<>
struct WmmaElement<float> {
    using type = float;

    static __device__ inline type* ptr(float* p) {
        return reinterpret_cast<type*>(p);
    }

    static __device__ inline const type* ptr(const float* p) {
        return reinterpret_cast<const type*>(p);
    }
};

