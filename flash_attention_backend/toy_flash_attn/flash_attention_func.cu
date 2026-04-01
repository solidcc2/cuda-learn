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

#define BLOCK_ROW_V2 8

// block by q_rows
template<typename scalar_t>
__global__ void flash_attn_varlen_with_block_kernel_v2(
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
    extern __shared__ scalar_t tile[];
    scalar_t* tile_Q = tile;    // blockDim.y * head_dim;
    scalar_t* tile_KT = tile + blockDim.y * head_dim; // blockDim.y * head_dim
    int window_size = window_size_right - window_size_left;
    scalar_t* tile_softmax = tile_KT + blockDim.y * head_dim; // blockDim.y * window_size
    scalar_t* tile_softmax_m = tile_softmax + blockDim.y * window_size;     // blockDim.x * blockDim.y
    scalar_t* tile_softmax_sum = tile_softmax_m + blockDim.x * blockDim.y; // blockDim.x * blockDim.y

    int64_t batch_id = blockIdx.z / num_heads;
    int64_t head_id = blockIdx.z % num_heads;
    int64_t q_token_id = threadIdx.y + blockIdx.y * blockDim.y;
    int64_t q_seqlen = cu_seqlens_q[batch_id+1] - cu_seqlens_q[batch_id];
    int64_t kv_seqlen = seqlens_k[batch_id];
    
    // gather tile_Q
    if (q_token_id < q_seqlen && threadIdx.x < head_dim) {
        tile_Q[threadIdx.y * head_dim + threadIdx.x] = q[
            q_stride0 * (cu_seqlens_q[batch_id] + q_token_id) + 
            q_stride1 * head_id +
            q_stride2 * threadIdx.x
        ];
    } else {
        tile_Q[threadIdx.y * head_dim + threadIdx.x] = 0.0;
    }
    // window size 边界调整
    int64_t history_offset = q_seqlen - q_token_id; // history offset
    // [absolute_range_left, absolute_range_right)
    int64_t absolute_range_right = min(kv_seqlen-1, kv_seqlen - history_offset + window_size_right + 1); // 开区间
    int64_t absolute_range_left = max(0, kv_seqlen - history_offset - window_size_left);
    for (int chunk_id = 0; chunk_id < (absolute_range_right - absolute_range_left + blockDim.y - 1) / blockDim.y; chunk_id++) {
        int64_t virt_kv_token_id = absolute_range_left + chunk_id * blockDim.y + threadIdx.y;
        int64_t phy_block_id = block_table[
            bt_stride0 * batch_id,
            bt_stride1 * (virt_kv_token_id / block_size) // virt block id
        ]
        if (virt_kv_token_id < kv_seqlen && threadIdx.x < head_dim) {
            tile_KT[threadidx.y * head_dim + threadIdx.x] = k[
                q_stride0 * phy_block_id +
                q_stride1 * (virt_kv_token_id % block_size) +
                q_stride2 * head_id +
                q_stride3 * threadIdx.x
            ]
        } else {
            tile_KT[threadidx.y * head_dim + threadIdx.x] = 0.0;
        }
        __syncthreads();

        // tile matmul
        for(int b_tile_row_id = 0; b_tile_row_id < blockDim.y; b_tile_row_id ++) {
            scalar_t result = tile_Q[threadIdx.y * head_dim + threadIdx.x] * tile_BT[b_tile_row_id * head_dim + threadIdx.x];
            tile_BT[b_tile_row_id * head_dim + threadIdx.x] = result;
            __syncthreads();
            for(int bound = blockDim.x / 2; bound >= warpSize; bound /= 2) {
                if (threadIdx. x < bound) {
                    result += tile_BT[b_tile_row_id * head_dim + threadIdx.x];
                }
            }
            if (threadIdx.x < warpSize) {
                result += __shfl_down_sync(0xffffffff, result, 16);
                result += __shfl_down_sync(0xffffffff, result, 8);
                result += __shfl_down_sync(0xffffffff, result, 4);
                result += __shfl_down_sync(0xffffffff, result, 2);
                result += __shfl_down_sync(0xffffffff, result, 1);
            }
            if (threadIdx.x == 0 && chunk_id * blockDim.y + b_tile_row_id < window_size) {
                tile_softmax[
                    threadIdx.y * window_size + 
                    chunk_id * blockDim.y + b_tile_row_id
                ] = result / sqrt(head_dim);
            }
        }
        __syncthreads();

        // online softmax
        scalar_t m = tile_softmax[threadIdx.y * window_size + threadIdx.x];
        scalar_t sum = 1.0;
        for(int i = threadIdx.x; i < window_size / blockDim.x; i += blockDim.x) {
            scalar_t val = tile_softmax[threadIdx.y * window_size + i];
            scalar_t new_m = max(val, m);
            sum = sum * __expf(m - new_m) + __expf(val - new_m);
            m = new_m;
        }
        tile_softmax_m[threadIdx.y * blockDim.x + threadIdx.x] = m;
        tile_softmax_sum[threadIdx.y * blockDim.x + threadIdx.x] = sum;
        __syncthreads();
        
        for(int bound < blockDim.x / 2; bound >= warpSize; bound /= 2) {
            if (threadIdx.x < bound) {
                scalar_t neighbor_m = tile_softmax_m[threadIdx.y * blockDim.x + threadIdx.x + bound];
                scalar_t neighbor_sum = tile_softmax_sum[threadIdx.y * blockDim.x + threadIdx.x + bound];
                scalar_t new_m = max(m, neighbor_m);
                sum = sum * __expf(m - new_m) + neighbor_sum * __expf(neighbor_m - new_m); 
                m = new_m;
            }
        }
        if (threadIdx.x < warpSize) {
            scalar_t neighbor_m = __shfl_down_sync(0xffffffff, m, 16);
            scalar_t neighbor_sum = __shfl_down_sync(0xffffffff, sum, 16);
            scalar_t new_m = max(m, neighbor_m);
            sum = sum * __expf(m - new_m) + neighbor_sum * __expf(neighbor_m - new_m); 
            m = new_m;

            neighbor_m = __shfl_down_sync(0xffffffff, m, 8);
            neighbor_sum = __shfl_down_sync(0xffffffff, sum, 8);
            new_m = max(m, neighbor_m);
            sum = sum * __expf(m - new_m) + neighbor_sum * __expf(neighbor_m - new_m); 
            m = new_m;

            neighbor_m = __shfl_down_sync(0xffffffff, m, 4);
            neighbor_sum = __shfl_down_sync(0xffffffff, sum, 4);
            new_m = max(m, neighbor_m);
            sum = sum * __expf(m - new_m) + neighbor_sum * __expf(neighbor_m - new_m); 
            m = new_m;

            neighbor_m = __shfl_down_sync(0xffffffff, m, 2);
            neighbor_sum = __shfl_down_sync(0xffffffff, sum, 2);
            new_m = max(m, neighbor_m);
            sum = sum * __expf(m - new_m) + neighbor_sum * __expf(neighbor_m - new_m); 
            m = new_m;

            neighbor_m = __shfl_down_sync(0xffffffff, m, 1);
            neighbor_sum = __shfl_down_sync(0xffffffff, sum, 1);
            new_m = max(m, neighbor_m);
            sum = sum * __expf(m - new_m) + neighbor_sum * __expf(neighbor_m - new_m); 
            m = new_m;
        }
    }
    

}

torch::Tensor flash_attn_varlen_with_block_v2(
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
    auto num_heads = q.size(1);

    assert(head_dim <= 128);        // 泛化

    int block_row = BLOCK_ROW_V2;
    dim3 block(head_dim, block_row);
    dim3 grid(1, 
            (max_seqlen_q + block_row - 1)/ block_row, 
            batch_size * num_heads
        );

    flash_attn_varlen_with_block_kernel<at::BFloat16><<<grid, block, 2 * sizeof(at::BFloat16) * head_dim * block_row>>>(
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
