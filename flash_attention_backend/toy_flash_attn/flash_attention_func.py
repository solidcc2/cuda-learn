import torch

from typing import List
# layout: (2, num_blocks, block_size, num_kv_heads, head_size)
def flash_attn_varlen_func(
    q: torch.Tensor,    # NHD： total_q x num_head x head_dim
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int|None,
    cu_seqlens_q: List[int]|None,
    max_seqlen_k: int|None,
    cu_seqlens_k: List[int]|None,  # only used for non-paged prefill
    seqused_k: List[int]|None = None,
    # softmax_scale: int|None=None,
    causal=False,
    window_size: tuple[int, int]  | None = None,
    block_table=None,
    # return_softmax_lse=False,
    out: torch.Tensor|None = None,
) -> torch.Tensor: 
    '''
        只考虑decoder, block_table决定是否过cache
    '''
    if block_table is not None:
        flash_attn_varlen_with_block(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, seqused_k, causal, window_size, block_table, out)
    else:
        flash_attn_varlen_without_block(q, k, v, max_seqlen_q,cu_seqlens_q, max_seqlen_k, cu_seqlens_k, causal, window_size, out)

def flash_attn_varlen_with_block(
    q: torch.Tensor,    # NHD： block_num x block_size x num_head x head_dim
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: List[int],
    max_seqlen_k: int,
    seqused_k: List[int],
    # softmax_scale: int|None=None,
    causal=False,
    window_size: tuple[int, int] | None = None,
    block_table=None,
    # return_softmax_lse=False,
    out: torch.Tensor|None = None,
) -> torch.Tensor: 
    pass
    
def flash_attn_varlen_without_block(
    q: torch.Tensor,    # NHD： total_q x num_head x head_dim
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: List[int],
    max_seqlen_k: int,
    cu_seqlens_k: List[int],  # only used for non-paged prefill
    # softmax_scale: int|None=None,
    causal=False,
    window_size: tuple[int, int] | None = None, # 所有batch统一window size
    # return_softmax_lse=False,
    out: torch.Tensor|None = None,
) -> torch.Tensor: 
    '''
        简单版本，直接计算，不经过cache
    '''
    if out is None:
        out = torch.empty_like(q, device=q.device)

    for batch_id in range(len(cu_seqlens_q)-1):
        token_range = (cu_seqlens_q[batch_id], cu_seqlens_q[batch_id+1])
        kv_range = (cu_seqlens_k[batch_id], cu_seqlens_k[batch_id+1])
        # Q @ K^T / sqrt(d)
        K = k[kv_range[0]:kv_range[1]]
        V = v[kv_range[0]:kv_range[1]]
        Q = q[token_range[0]:token_range[1]]
        scale: float = q.shape[2] ** 0.5
        S = Q.transpose(0, 1).matmul(K.permute(1, 2, 0)) / scale
        kv_len = kv_range[1] - kv_range[0]
        batch_win = window_size
        if batch_win is None or batch_win == (-1, -1):
            batch_win = (kv_len, kv_len)
        if causal == True:
            batch_win = (batch_win[0], 0)
        mask = torch.ones(S.shape[1:], dtype=bool, device=S.device)
        for pos, t in enumerate(mask):
            w = (max(pos - batch_win[0], 0), min(pos + 1 + batch_win[1], kv_len))
            t[w[0]:w[1]] = False
        S = S.masked_fill(mask, float("-inf"))

        P = S.softmax(2)
        o = P.matmul(V.transpose(0, 1))
        o = o.transpose(0, 1)
        out[token_range[0]:token_range[1]] = o
    return out
