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
    TORCH_CHECK(q.is_contiguous(), "q is not contiguous");
    TORCH_CHECK(k.is_contiguous(), "k is not contiguous");
    TORCH_CHECK(v.is_contiguous(), "v is not contiguous");
    TORCH_CHECK(q.size(1) == k.size(2), "GQA not support, num_head must equal num_kv_head");
}

#define TILE_ROW 32
#define TILE_COL 32

// 单个batch
template<typename scalar_t>
__global__ void flash_attn_varlen_with_block_kernel(
    const scalar_t* q,  // max_seqlen_q x num_heads x head_dim
    c10::IntArrayRef q_strides,
    const scalar_t* k,  // num_blocks x block_size x num_kv_heads x head_dim
    c10::IntArrayRef k_strides,
    const scalar_t* v,
    c10::IntArrayRef v_strides,
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
    int64_t batch_size,
    scalar_t* out
) {
    extern __shared__ scalar_t* tile; 
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t scale = sqrt(head_dim);
    for(int batch_id = 0; batch_id<batch_size; batch_id++) {
        // Q.transpose(0, 1).matmul(K.permute(1, 2, 0)) / scale
        q_range_l = cu_seqlens_q[batch_id];
        q_range_r = cu_seqlens_q[batch_id+1];
        kv_len = seqused_k[batch_id];
        
        

        // mask window 
    }
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
        q.strides(),
        k.const_data_ptr<at::BFloat16>(),
        k.strides(),
        v.const_data_ptr<at::BFloat16>(),
        v.strides(),
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
        seqused_k.size(0),
        out.data_ptr<at::BFloat16>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_varlen_with_block", &flash_attn_varlen_with_block, "flash attn varlen with block");
}
