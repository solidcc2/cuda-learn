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
    run_official,
    run_toy_dense,
    run_toy_paged,
    run_toy_paged_cuda,
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
    out_ref = run_official(
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

    if use_block:
        resolved_k_lens = k_lens if k_lens is not None else q_lens
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

    assert torch.allclose(out_toy, out_ref, atol=2e-3, rtol=2e-3)


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

    if not torch.allclose(out_cu, out_ref, atol=2e-3, rtol=2e-3):
        diff = (out_cu.float() - out_ref.float()).abs()
        flat_idx = int(diff.argmax().item())
        worst_index = list(__import__("numpy").unravel_index(flat_idx, diff.shape))
        worst_index_tuple = tuple(worst_index)
        raise AssertionError(
            "with_block_cu mismatch: "
            f"max_abs_diff={diff.max().item():.6f}, "
            f"mean_abs_diff={diff.mean().item():.6f}, "
            f"worst_index={worst_index_tuple}, "
            f"out_cu={out_cu[worst_index_tuple].item()}, "
            f"out_ref={out_ref[worst_index_tuple].item()}, "
            f"q_lens={q_lens}, "
            f"k_lens={resolved_k_lens}, "
            f"causal={causal}, "
            f"window_size={window_size}, "
            f"num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}, "
            f"dtype={dtype}, "
            f"seed={seed}"
        )


__all__ = ["assert_close", "assert_close_with_block_cu", "replay_dump_close"]
