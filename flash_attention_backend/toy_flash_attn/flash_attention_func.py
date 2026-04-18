from __future__ import annotations

import os
import torch

from pathlib import Path
from itertools import count

from torch.utils.cpp_extension import load

_THIS_DIR = Path(__file__).resolve().parent

_CUDA_VALUE_DTYPE = torch.bfloat16
_CUDA_INDEX_DTYPE = torch.int32
_DUMP_COUNTER = count()


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


def _maybe_log_with_block_dtypes(
    *,
    batch_id: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    S: torch.Tensor,
    P: torch.Tensor,
    o: torch.Tensor,
) -> None:
    if os.environ.get("TOY_FLASH_ATTN_PRINT_DTYPE", "0") != "1":
        return
    if batch_id != 0:
        return
    fp32_precision = (
        torch.backends.cuda.matmul.fp32_precision
        if hasattr(torch.backends.cuda.matmul, "fp32_precision")
        else "no fp32_precision"
    )
    allow_tf32 = (
        torch.backends.cuda.matmul.allow_tf32
        if hasattr(torch.backends.cuda.matmul, "allow_tf32")
        else "no allow_tf32"
    )
    print(
        "[toy_flash_attn] with_block dtype trace:",
        f"q={q.dtype} k={k.dtype} v={v.dtype} "
        f"Q={Q.dtype} K={K.dtype} V={V.dtype} "
        f"S={S.dtype} P={P.dtype} o={o.dtype} "
        f"fp32_precision={fp32_precision} allow_tf32={allow_tf32}",
        flush=True,
    )


def _clone_for_dump(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_clone_for_dump(item) for item in value)
    if isinstance(value, list):
        return [_clone_for_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_for_dump(item) for key, item in value.items()}
    return value


def _maybe_dump_flash_attn_context(op_name: str, **payload) -> None:
    dump_dir = os.environ.get("TOY_FLASH_ATTN_DUMP_DIR")
    if not dump_dir:
        return

    dump_path = Path(dump_dir).expanduser().resolve()
    dump_path.mkdir(parents=True, exist_ok=True)
    dump_id = next(_DUMP_COUNTER)
    file_path = dump_path / f"{dump_id:05d}_{op_name}.pt"
    torch.save(_clone_for_dump(payload), file_path)
    print(f"[toy_flash_attn] dumped {op_name} context to {file_path}", flush=True)


def _layer_debug_meta(layer) -> dict:
    if layer is None:
        return {}
    return {
        "layer_name": type(layer).__name__,
        "layer_idx": getattr(layer, "layer_idx", None),
    }


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
        str(_THIS_DIR / "v4/flash_attn_func.cu"),
    ],
    extra_include_paths=[
        str(_THIS_DIR / "v4"),
    ],
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O0", "-G"],
    verbose=True,
)

# _ops = load(
#     name="toy_torch_flash_attention_func",
#     sources=[
#         str(_THIS_DIR / "flash_attn_func_v3.cu"),
#     ],
#     extra_cflags=["-O2"],
#     extra_cuda_cflags=["-O2"],
#     verbose=True,
# )

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
    layer=None,
    # return_softmax_lse=False,
    out: torch.Tensor|None = None,
) -> torch.Tensor: 
    '''
        只考虑decoder, block_table决定是否过cache。
        q和kv尾对齐
    '''
    if block_table is not None:
        if os.getenv("TOY_FLASH_ATTN_USE", "bf16") == "reference":
            return flash_attn_varlen_with_block(
                q, k, v,
                max_seqlen_q, cu_seqlens_q,
                max_seqlen_k, seqused_k,
                causal, window_size, block_table, out,
                layer=layer,
            )
        if os.getenv("TOY_FLASH_ATTN_USE", "bf16") == "fp32":
            return flash_attn_varlen_with_block_cu_fp32(
                q, k, v,
                max_seqlen_q, cu_seqlens_q,
                max_seqlen_k, seqused_k,
                causal, window_size, block_table, out,
                layer=layer,
            )
        return flash_attn_varlen_with_block_cu_bf16(
            q, k, v,
            max_seqlen_q, cu_seqlens_q,
            max_seqlen_k, seqused_k,
            causal, window_size, block_table, out,
            layer=layer,
        )
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
    layer=None,
) -> torch.Tensor: 
    # q = q.to(dtype=torch.float32)
    # k = k.to(dtype=torch.float32)
    # v = v.to(dtype=torch.float32)
    # out = out.to(dtype=torch.float32)
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
        _maybe_log_with_block_dtypes(
            batch_id=batch_id,
            q=q,
            k=k,
            v=v,
            Q=Q,
            K=K,
            V=V,
            S=S,
            P=P,
            o=o,
        )
        out[q_range[0]:q_range[1]] = o
    _maybe_dump_flash_attn_context(
        "with_block",
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
        debug_meta=_layer_debug_meta(layer),
        result=out,
    )
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

import hashlib
import torch

def _tensor_bytes(t: torch.Tensor) -> bytes:
    x = t.detach().contiguous().cpu().view(torch.uint8)
    return x.numpy().tobytes()

def _tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha1(_tensor_bytes(t)).hexdigest()[:16]

def _qkv_hash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> str:
    h = hashlib.sha1()
    for t in (q, k, v):
        x = t.detach().contiguous().cpu()
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(_tensor_bytes(x))
    return h.hexdigest()[:16]

def _effective_qkv_hash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
) -> str:
    h = hashlib.sha1()
    block_size = k.shape[1]

    batch_size = len(seqused_k)
    for batch_id in range(batch_size):
        q_begin = cu_seqlens_q[batch_id].item()
        q_end = cu_seqlens_q[batch_id + 1].item()
        kv_len = seqused_k[batch_id].item()

        q_slice = q[q_begin:q_end]
        h.update(_qkv_hash(q_slice, q_slice.new_empty((0,)), q_slice.new_empty((0,))).encode())

        phy_block_ids = [block_table[batch_id][seq_id // block_size] for seq_id in range(kv_len)]
        offsets = [seq_id % block_size for seq_id in range(kv_len)]
        k_slice = k[phy_block_ids, offsets]
        v_slice = v[phy_block_ids, offsets]
        h.update(_qkv_hash(k_slice, v_slice, k_slice.new_empty((0,))).encode())

    return h.hexdigest()[:16]

def flash_attn_varlen_with_block_cu_bf16(
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
    layer=None,
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
    # q = q.to(dtype=torch.float32)
    # k = k.to(dtype=torch.float32)
    # v = v.to(dtype=torch.float32)
    # out = out.to(dtype=torch.float32)
    turn_hash = _effective_qkv_hash(q, k, v, cu_seqlens_q, seqused_k, block_table)

    op_out = _ops.flash_attn_varlen_with_block_v4_bf16fp32(q, k, v,
                                    max_seqlen_q, cu_seqlens_q,
                                    max_seqlen_k, seqused_k,
                                    causal, window_size[0], window_size[1],
                                    block_table,
                                    out)
    _maybe_dump_flash_attn_context(
        "with_block_cu",
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
        debug_meta=_layer_debug_meta(layer),
        result=op_out,
    )
    if op_out.data_ptr() != out.data_ptr():
        out.copy_(op_out.to(dtype=out.dtype))
    torch.cuda.synchronize(q.device)
    output_hash = _tensor_hash(out)
    print(f"[turn hash] {turn_hash}", flush=True)
    print(f"[output hash] {output_hash}", flush=True)
    print("======================= turn end =====================", flush=True)
    os._exit(0)

    return out

def flash_attn_varlen_with_block_cu_fp32(
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
    layer=None,
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
    turn_hash = _effective_qkv_hash(q, k, v, cu_seqlens_q, seqused_k, block_table)
    q_fp32 = q.to(dtype=torch.float32)
    k_fp32 = k.to(dtype=torch.float32)
    v_fp32 = v.to(dtype=torch.float32)
    out_fp32 = torch.empty_like(q_fp32)

    op_out = _ops.flash_attn_varlen_with_block_v4_fp32fp32(q_fp32, k_fp32, v_fp32,
                                    max_seqlen_q, cu_seqlens_q,
                                    max_seqlen_k, seqused_k,
                                    causal, window_size[0], window_size[1],
                                    block_table,
                                    out_fp32)
    _maybe_dump_flash_attn_context(
        "with_block_cu",
        q=q_fp32,
        k=k_fp32,
        v=v_fp32,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
        debug_meta=_layer_debug_meta(layer),
        result=op_out,
    )
    out.copy_(op_out.to(dtype=out.dtype))
    torch.cuda.synchronize(q.device)
    output_hash = _tensor_hash(out)
    print(f"[turn hash] {turn_hash}", flush=True)
    print(f"[output hash] {output_hash}", flush=True)
    print("======================= turn end =====================", flush=True)
    os._exit(0)
    return out
