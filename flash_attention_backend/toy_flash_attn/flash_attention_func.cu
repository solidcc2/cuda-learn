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
#define BLOCK_ROW 32
#define BLOCK_COL 32

// deprecated
template<typename scalar_t>
__global__ void flash_attn_varlen_with_block_kernel(
    const scalar_t* q,  // total_q x num_heads x head_dim
    const int64_t q_stride0,
    const int64_t q_stride1,
    const int64_t q_stride2, 
    const scalar_t* k,  // num_blocks x block_size x num_kv_heads x head_dim
    const int64_t k_stride0,
    const int64_t k_stride1,
    const int64_t k_stride2,
    const int64_t k_stride3,
    const scalar_t* v,
    const int64_t v_stride0,
    const int64_t v_stride1,
    const int64_t v_stride2,
    const int64_t v_stride3,
    const int64_t max_seqlen_q,
    const int32_t* cu_seqlens_q,
    const int64_t max_seqlen_k,
    const int32_t* seqused_k,
    bool causal,
    int64_t window_size_left,
    int64_t window_size_right,
    const int32_t* block_table,
    const int64_t bt_stride0,
    const int64_t bt_stride1,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    scalar_t* out
) {
    extern __shared__ scalar_t tile[];   // 2 x TILE_ROW x TILE_COL
    scalar_t* tile_A = tile; // BLOCK_ROW x BLOCK_COL
    scalar_t* tile_BT = tile + blockDim.y * blockDim.x; // BLOCK_ROW x BLOCK_COL
    int64_t batch_id = blockIdx.z / num_heads;
    int64_t head_id = blockIdx.z % num_heads;
    int64_t q_seqlen = cu_seqlens_q[batch_id+1] - cu_seqlens_q[batch_id];
    int64_t kv_seqlen = seqused_k[batch_id];
    int64_t row_id = blockDim.y * blockIdx.y + threadIdx.y;
    int64_t col_id = blockDim.x * blockIdx.x + threadIdx.x;
    scalar_t result = 0.0f;
    for(int chunk_id=0; chunk_id<(head_dim + TILE_COL - 1)/TILE_COL; chunk_id++) {
        if (row_id < q_seqlen && chunk_id*TILE_COL+threadIdx.x < head_dim && threadIdx.x < TILE_COL) {
            int64_t token_id = cu_seqlens_q[batch_id] + row_id;
            tile_A[threadIdx.y * TILE_COL + threadIdx.x] = 
                q[
                    q_stride0 * token_id + 
                    q_stride1 * head_id + 
                    q_stride2 * (chunk_id*TILE_COL+threadIdx.x)
                ];
        } else {
            tile_A[threadIdx.y * TILE_COL + threadIdx.x] = 0.0f;
        }
        if (col_id < kv_seqlen && chunk_id*TILE_COL+threadIdx.y < head_dim && threadIdx.y < TILE_COL) {  // 
            int64_t virt_block_id = col_id / block_size;
            int64_t block_offset = col_id % block_size;
            int64_t phy_block_id = 
                block_table[
                    bt_stride0 * batch_id + 
                    bt_stride1 * virt_block_id
                ];
            tile_BT[threadIdx.x *TILE_COL + threadIdx.y] = 
                k[  
                    k_stride0 * phy_block_id + 
                    k_stride1 * block_offset + 
                    k_stride2 * head_id + 
                    k_stride3 * (chunk_id*TILE_COL+threadIdx.y)
                ];
        } else {
            tile_BT[threadIdx.x *TILE_COL + threadIdx.y] = 0.0f;
        }
        __syncthreads();
        if (row_id < q_seqlen && col_id < kv_seqlen) {
            for(i=0; i<TILE_COL; i++) {
                result += tile_A[threadIdx.y * TILE_COL + i] * tile_BT[threadIdx.x * TILE_COL + i];
            }
        
        }
        __syncthreads();
    }
    result /= sqrt(head_dim);

    // 沿着 x 方向做softmax
    scalar_t* softmax_tile = tile;   // TILE_ROW x TILE_COL
 
    softmax_tile[threadIdx.y * TILE_COL + threadIdx.x] = 
        (row_id < q_seqlen && col_id < kv_seqlen) ? result : -INFINITY;
    __syncthreads();
    scalar_t m = softmax_tile[threadIdx.y * TILE_COL + threadIdx.x];
    scalar_t sum = 1.0f;
    
    for(int bound = blockDim.x/2; bound >=warpSize; bound/=2) {
        if (threadIdx.x < bound) {
            scalar_t neighbor = softmax_tile[threadIdx.y * TILE_COL + threadIdx.x + bound];
            scaler_t new_m = max(m, neighbor);
            sum = sum * __expf(m-new_m) + __expf(neighbor-new_m);
            m = new_m;
        }
    }
    scalar_t var = softmax_tile[threadIdx.y * TILE_COL + threadIdx.x];
    if (threadIdx.x < warpSize) {
        scalar_t neighbor = __shfl_down_sync(0xffffffff, var, 16);
        scaler_t new_m = max(m, neighbor);
        sum = sum * __expf(m-new_m) + __expf(neighbor-new_m);
        m = new_m;

        neighbor = __shfl_down_sync(0xffffffff, var, 8);
        scaler_t new_m = max(m, neighbor);
        sum = sum * __expf(m-new_m) + __expf(neighbor-new_m);
        m = new_m;

        neighbor = __shfl_down_sync(0xffffffff, var, 4);
        scaler_t new_m = max(m, neighbor);
        sum = sum * __expf(m-new_m) + __expf(neighbor-new_m);
        m = new_m;

        neighbor = __shfl_down_sync(0xffffffff, var, 2);
        scaler_t new_m = max(m, neighbor);
        sum = sum * __expf(m-new_m) + __expf(neighbor-new_m);
        m = new_m;

        neighbor = __shfl_down_sync(0xffffffff, var, 1);
        scaler_t new_m = max(m, neighbor);
        sum = sum * __expf(m-new_m) + __expf(neighbor-new_m);
        m = new_m;
    }
    tile_A[threadIdx.y * TILE_COL + threadIdx.x] = __expf(var - m) / sum;

    if (col_id < kv_seqlen && row_id < q_seqlen && threadIdx.y < TILE_COL) {
        tile_BT[threadIdx.x * TILE_COL + threadIdx.y] = 0.0f;
    } else {
        tile_BT[threadIdx.x * TILE_COL + threadIdx.y] = 0.0f;
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
    dim3 block(BLOCK_COL, BLOCK_ROW);
    auto batch_size = seqused_k.size(0);
    auto num_heads = q.size(1);
    dim3 grid((max_seqlen_k + BLOCK_COL - 1)/BLOCK_COL,
                (max_seqlen_q + BLOCK_ROW - 1)/BLOCK_ROW, 
                batch_size * num_heads);
    flash_attn_varlen_with_block_kernel<at::BFloat16><<<grid, block, 2 * sizeof(at::BFloat16) * TILE_COL * TILE_ROW>>>(
        q.const_data_ptr<at::BFloat16>(),
        q.stride(0), q.stride(1), q.stride(2),
        k.const_data_ptr<at::BFloat16>(),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.const_data_ptr<at::BFloat16>(),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        max_seqlen_q,
        cu_seqlens_q.const_data_ptr<int32_t>(),
        max_seqlen_k,
        seqused_k.const_data_ptr<int32_t>(),
        causal,
        window_size_left, window_size_right,
        block_table.const_data_ptr<int32_t>(),
        block_table.stride(0), block_table.stride(1),
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
