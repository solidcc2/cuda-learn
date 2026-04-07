from __future__ import annotations

import os
import torch

from pathlib import Path

from torch.utils.cpp_extension import load

_THIS_DIR = Path(__file__).resolve().parent

_CUDA_VALUE_DTYPE = torch.bfloat16
_CUDA_INDEX_DTYPE = torch.int32


def _tensor_debug_str(name: str, tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return f"{name}=None"
    return (
        f"{name}(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, contiguous={tensor.is_contiguous()})"
    )


def _maybe_log_cuda_inputs(**tensors: torch.Tensor | None) -> None:
    if os.environ.get("TOY_FLASH_ATTN_DEBUG", "0") != "1":
        return
    print(
        "[toy_flash_attn] flash_attn_varlen_with_block_cu inputs:",
        ", ".join(_tensor_debug_str(name, tensor) for name, tensor in tensors.items()),
        flush=True,
    )


def _check_cuda_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    expected_device: torch.device,
    expected_dtype: torch.dtype,
) -> None:
    if tensor.device != expected_device:
        raise ValueError(
            f"{name}.device={tensor.device} does not match q.device={expected_device}"
        )
    if tensor.dtype != expected_dtype:
        raise TypeError(
            f"{name}.dtype={tensor.dtype} does not match CUDA op requirement "
            f"{expected_dtype}"
        )


# compile + load cu
_ops = load(
    name="toy_torch_flash_attention_func",
    sources=[
        str(_THIS_DIR / "flash_attention_func.cu"),
    ],
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2"],
    verbose=True,
)

# layout: (2, num_blocks, block_size, num_kv_heads, head_size)
def flash_attn_varlen_func(
    q: torch.Tensor,    # NHD： total_q x num_head x head_dim
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int|None,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int|None,
    cu_seqlens_k: torch.Tensor|None,  # only used for non-paged prefill
    seqused_k: torch.Tensor|None = None,
    # softmax_scale: int|None=None,
    causal=False,
    window_size: tuple[int, int]  | None = None,
    block_table=None,
    # return_softmax_lse=False,
    out: torch.Tensor|None = None,
) -> torch.Tensor: 
    '''
        只考虑decoder, block_table决定是否过cache。
        q和kv尾对齐
    '''
    if block_table is not None:
        return flash_attn_varlen_with_block_cu(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, seqused_k, causal, window_size, block_table, out)
        # return flash_attn_varlen_with_block(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, seqused_k, causal, window_size, block_table, out)
    else:
        return flash_attn_varlen_without_block(q, k, v, max_seqlen_q,cu_seqlens_q, max_seqlen_k, cu_seqlens_k, causal, window_size, out)

def flash_attn_varlen_with_block(
    q: torch.Tensor,    # total_q x num_head x head_dim
    k: torch.Tensor,    # NHD： num_blocks x block_size x num_head x head_dim
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    seqused_k: torch.Tensor,
    # softmax_scale: int|None=None,
    causal=False,
    window_size: tuple[int, int] | None = None,
    block_table=None,
    # return_softmax_lse=False,
    out: torch.Tensor|None = None,
) -> torch.Tensor: 
    assert block_table is not None
    if out is None:
        out = torch.empty_like(q, device=q.device)
    for batch_id in range(len(seqused_k)):
        q_range = (cu_seqlens_q[batch_id].item(), cu_seqlens_q[batch_id+1].item())
        kv_len = seqused_k[batch_id].item()
        kv_range = (0, kv_len)
        Q = q[q_range[0]:q_range[1]]
        block_size = k.shape[1]
        phy_block_ids = [block_table[batch_id][seq_id//block_size] for seq_id in range(*kv_range)]
        offset = [seq_id%block_size for seq_id in range(*kv_range)]
        K = k[phy_block_ids, offset]
        V = v[phy_block_ids, offset]
        head_dim = q.shape[2]
        S = Q.transpose(0, 1).matmul(K.permute(1, 2, 0)) / float(head_dim ** 0.5)

        cur_win = window_size
        if cur_win is None or cur_win == (-1, -1):
            cur_win = (kv_len, kv_len)
        if causal == True:
            cur_win = (cur_win[0], 0)
        mask = torch.ones(S.shape[1:], dtype=bool, device=S.device)
        k_q_offset = K.shape[0] - Q.shape[0]
        for pos, t in enumerate(mask):
            w = (max(0, pos+k_q_offset-cur_win[0]), min(kv_len, pos+k_q_offset+1+cur_win[1]))
            t[w[0]:w[1]] = False
        S = S.masked_fill(mask, float("-inf"))
        P = S.softmax(2)
        o = P.matmul(V.transpose(0, 1)).transpose(0, 1)
        out[q_range[0]:q_range[1]] = o
    return out
        
    
def flash_attn_varlen_without_block(
    q: torch.Tensor,    # NHD： total_q x num_head x head_dim
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,  # only used for non-paged prefill
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
        token_range = (cu_seqlens_q[batch_id].item(), cu_seqlens_q[batch_id+1].item())
        kv_range = (cu_seqlens_k[batch_id].item(), cu_seqlens_k[batch_id+1].item())
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
        k_q_offset = K.shape[0] - Q.shape[0]
        for pos, t in enumerate(mask):
            w = (max(pos + k_q_offset - batch_win[0], 0), min(pos + k_q_offset + 1 + batch_win[1], kv_len))
            t[w[0]:w[1]] = False
        S = S.masked_fill(mask, float("-inf"))

        P = S.softmax(2)
        o = P.matmul(V.transpose(0, 1))
        o = o.transpose(0, 1)
        out[token_range[0]:token_range[1]] = o
    return out


def flash_attn_varlen_with_block_cu(
    q: torch.Tensor,    # total_q x num_head x head_dim
    k: torch.Tensor,    # NHD： num_blocks x block_size x num_head x head_dim
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    seqused_k: torch.Tensor,
    # softmax_scale: int|None=None,
    causal=False,
    window_size: tuple[int, int] | None = None,
    block_table=None,
    # return_softmax_lse=False,
    out: torch.Tensor|None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(q)
    if window_size is None:
        window_size = (-1, -1)
    if block_table is None:
        raise ValueError("block_table must not be None for flash_attn_varlen_with_block_cu")
    if not q.is_cuda:
        raise ValueError("flash_attn_varlen_with_block_cu expects CUDA tensors")

    _maybe_log_cuda_inputs(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        block_table=block_table,
        out=out,
    )

    expected_device = q.device
    _check_cuda_tensor("q", q, expected_device=expected_device, expected_dtype=_CUDA_VALUE_DTYPE)
    _check_cuda_tensor("k", k, expected_device=expected_device, expected_dtype=_CUDA_VALUE_DTYPE)
    _check_cuda_tensor("v", v, expected_device=expected_device, expected_dtype=_CUDA_VALUE_DTYPE)
    _check_cuda_tensor(
        "cu_seqlens_q",
        cu_seqlens_q,
        expected_device=expected_device,
        expected_dtype=_CUDA_INDEX_DTYPE,
    )
    _check_cuda_tensor(
        "seqused_k",
        seqused_k,
        expected_device=expected_device,
        expected_dtype=_CUDA_INDEX_DTYPE,
    )
    _check_cuda_tensor(
        "block_table",
        block_table,
        expected_device=expected_device,
        expected_dtype=_CUDA_INDEX_DTYPE,
    )
    _check_cuda_tensor("out", out, expected_device=expected_device, expected_dtype=_CUDA_VALUE_DTYPE)

    return _ops.flash_attn_varlen_with_block(q, k, v, 
                                    max_seqlen_q, cu_seqlens_q,
                                    max_seqlen_k, seqused_k,
                                    causal, window_size[0], window_size[1],
                                    block_table, 
                                    out)
