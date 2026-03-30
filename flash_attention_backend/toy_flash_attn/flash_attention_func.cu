#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <string>

void check_inputs(
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
}

template<typename scalar_t>
__global__ void flash_attn_varlen_with_block_kernel(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const int64_t max_seqlen_q,
    const int32_t* cu_seqlens_q,
    const int64_t max_seqlen_k,
    const int32_t* seqused_k,
    bool causal,
    int64_t window_size_left,
    int64_t window_size_right,
    const int32_t* block_table,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    scalar_t* out
) {
    // TODO
}

torch::Tensor flash_attn_varlen_with_block(
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
    dim3 block(3);  // TODO
    dim3 grid(4);   // TODO
    flash_attn_varlen_with_block_kernel<at::BFloat16><<<grid, block>>>(
        q.const_data_ptr<at::BFloat16>(),
        k.const_data_ptr<at::BFloat16>(),
        v.const_data_ptr<at::BFloat16>(),
        max_seqlen_q,
        cu_seqlens_q.const_data_ptr<int32_t>(),
        max_seqlen_k,
        seqused_k.const_data_ptr<int32_t>(),
        causal,
        window_size_left, window_size_right,
        block_table.const_data_ptr<int32_t>(),
        q.size(1),
        k.size(2),
        q.size(2),
        k.size(1),
        out.data_ptr<at::BFloat16>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_varlen_with_block", &flash_attn_varlen_with_block, "flash attn varlen with block");
}
