from __future__ import annotations

import sys

from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from flash_attention_backend.toy_flash_attn.test_utils import (
    make_block_cache,
    make_inputs,
    replay_dump_close,
    run_official_dense,
    run_toy_dense,
    run_toy_paged,
    run_toy_paged_cuda,
)

STRICT_ATOL = 2e-3
STRICT_RTOL = 2e-3
RELAXED_ATOL = 8e-3
RELAXED_RTOL = 8e-3


def _flattened_quantile(values: torch.Tensor, q: float) -> float:
    flat = values.reshape(-1).cpu()
    return float(torch.quantile(flat, q).item())


def _unravel_index(flat_index: int, shape: torch.Size) -> tuple[int, ...]:
    coords: list[int] = []
    remaining = flat_index
    for dim in reversed(shape):
        coords.append(remaining % dim)
        remaining //= dim
    return tuple(reversed(coords))


def _compare_outputs(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    metadata: dict[str, object],
) -> dict[str, object]:
    diff = (lhs.float() - rhs.float()).abs()
    flat_idx = int(diff.argmax().item())
    worst_index = _unravel_index(flat_idx, diff.shape)
    allclose_strict = torch.allclose(lhs, rhs, atol=STRICT_ATOL, rtol=STRICT_RTOL)
    allclose_relaxed = torch.allclose(lhs, rhs, atol=RELAXED_ATOL, rtol=RELAXED_RTOL)

    if allclose_strict:
        status = "pass"
    elif allclose_relaxed:
        status = "sensitive"
    else:
        status = "error"

    return {
        "status": status,
        "supported": True,
        "allclose_strict": bool(allclose_strict),
        "allclose_relaxed": bool(allclose_relaxed),
        "strict_threshold": {"atol": STRICT_ATOL, "rtol": STRICT_RTOL},
        "relaxed_threshold": {"atol": RELAXED_ATOL, "rtol": RELAXED_RTOL},
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "p95_abs_diff": _flattened_quantile(diff, 0.95),
        "p99_abs_diff": _flattened_quantile(diff, 0.99),
        "numel": int(diff.numel()),
        "worst_index": list(worst_index),
        "lhs_value": float(lhs[worst_index].item()),
        "rhs_value": float(rhs[worst_index].item()),
        "metadata": metadata,
    }


def measure_close(
    *,
    q_lens: list[int],
    k_lens: list[int] | None,
    causal: bool,
    window_size: tuple[int, int] | None,
    num_heads: int = 2,
    num_kv_heads: int | None = None,
    head_dim: int = 16,
    dtype: torch.dtype = torch.float16,
    seed: int = 0,
    use_block: bool = False,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = make_inputs(
        q_lens=q_lens,
        k_lens=k_lens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    out_ref = run_official_dense(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        window_size=window_size,
    )

    resolved_k_lens = k_lens if k_lens is not None else q_lens
    if use_block:
        k_cache, v_cache, block_table = make_block_cache(k_dense=k, v_dense=v, k_lens=resolved_k_lens)
        out_toy = run_toy_paged(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            seqused_k=torch.tensor(resolved_k_lens, device=q.device, dtype=torch.int32),
            causal=causal,
            window_size=window_size,
            block_table=block_table,
        )
        path_kind = "paged"
    else:
        out_toy = run_toy_dense(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            window_size=window_size,
        )
        path_kind = "dense"

    return _compare_outputs(
        out_toy,
        out_ref,
        metadata={
            "q_lens": q_lens,
            "k_lens": resolved_k_lens,
            "causal": causal,
            "window_size": window_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype": str(dtype),
            "seed": seed,
            "path_kind": path_kind,
            "reference": "official_fa2",
        },
    )


def measure_close_with_block_cu(
    *,
    q_lens: list[int],
    k_lens: list[int] | None,
    causal: bool,
    window_size: tuple[int, int] | None,
    num_heads: int = 2,
    num_kv_heads: int | None = None,
    head_dim: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    resolved_k_lens = k_lens if k_lens is not None else q_lens
    q, k, v, cu_seqlens_q, _, max_seqlen_q, max_seqlen_k = make_inputs(
        q_lens=q_lens,
        k_lens=resolved_k_lens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    k_cache, v_cache, block_table = make_block_cache(k_dense=k, v_dense=v, k_lens=resolved_k_lens)
    seqused_k = torch.tensor(resolved_k_lens, device=q.device, dtype=torch.int32)

    out_ref = run_toy_paged(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
    )
    out_cu = run_toy_paged_cuda(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
    )

    return _compare_outputs(
        out_cu,
        out_ref,
        metadata={
            "q_lens": q_lens,
            "k_lens": resolved_k_lens,
            "causal": causal,
            "window_size": window_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype": str(dtype),
            "seed": seed,
            "path_kind": "paged_cuda",
            "reference": "python_paged_reference",
        },
    )


def assert_close(
    *,
    q_lens: list[int],
    k_lens: list[int] | None,
    causal: bool,
    window_size: tuple[int, int] | None,
    num_heads: int = 2,
    num_kv_heads: int | None = None,
    head_dim: int = 16,
    dtype: torch.dtype = torch.float16,
    seed: int = 0,
    use_block: bool = False,
) -> None:
    result = measure_close(
        q_lens=q_lens,
        k_lens=k_lens,
        causal=causal,
        window_size=window_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        seed=seed,
        use_block=use_block,
    )
    assert bool(result["allclose_strict"]), result


def assert_close_with_block_cu(
    *,
    q_lens: list[int],
    k_lens: list[int] | None,
    causal: bool,
    window_size: tuple[int, int] | None,
    num_heads: int = 2,
    num_kv_heads: int | None = None,
    head_dim: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> None:
    result = measure_close_with_block_cu(
        q_lens=q_lens,
        k_lens=k_lens,
        causal=causal,
        window_size=window_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        seed=seed,
    )
    if not bool(result["allclose_strict"]):
        raise AssertionError(
            "with_block_cu mismatch: "
            f"max_abs_diff={result['max_abs_diff']:.6f}, "
            f"mean_abs_diff={result['mean_abs_diff']:.6f}, "
            f"worst_index={tuple(result['worst_index'])}, "
            f"out_cu={result['lhs_value']}, "
            f"out_ref={result['rhs_value']}, "
            f"q_lens={result['metadata']['q_lens']}, "
            f"k_lens={result['metadata']['k_lens']}, "
            f"causal={result['metadata']['causal']}, "
            f"window_size={result['metadata']['window_size']}, "
            f"num_heads={result['metadata']['num_heads']}, "
            f"num_kv_heads={result['metadata']['num_kv_heads']}, "
            f"head_dim={result['metadata']['head_dim']}, "
            f"dtype={result['metadata']['dtype']}, "
            f"seed={result['metadata']['seed']}"
        )


__all__ = [
    "RELAXED_ATOL",
    "RELAXED_RTOL",
    "STRICT_ATOL",
    "STRICT_RTOL",
    "assert_close",
    "assert_close_with_block_cu",
    "measure_close",
    "measure_close_with_block_cu",
    "replay_dump_close",
]
