import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

_MODULE_PATH = Path(__file__).with_name("flash_attention_func.py")
_SPEC = spec_from_file_location("toy_flash_attention_func", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
flash_attn_varlen_without_block = _MOD.flash_attn_varlen_without_block
flash_attn_varlen_with_block = _MOD.flash_attn_varlen_with_block
flash_attn_varlen_with_block_cu = _MOD.flash_attn_varlen_with_block_cu


def _require_fa2_cuda() -> None:
    # These tests are intended as black-box parity checks against the
    # official FA2 implementation, so we only run when CUDA + FA2 are visible.
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required for this test.")
    if not is_flash_attn_varlen_func_available():
        raise unittest.SkipTest(
            "Official flash_attn_varlen_func is not available in this environment."
        )
    if get_flash_attn_version() != 2:
        raise unittest.SkipTest("This test expects FA2 to be active.")


def _make_inputs(
    q_lens: list[int],
    k_lens: list[int] | None = None,
    num_heads: int = 2,
    head_dim: int = 16,
    dtype: torch.dtype = torch.float16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
]:
    # `q_lens` and `k_lens` are per-request lengths. When `k_lens` is longer
    # than `q_lens`, tests assume tail-alignment: q[-1] aligns with kv[-1].
    if k_lens is None:
        k_lens = q_lens

    total_q = sum(q_lens)
    total_k = sum(k_lens)
    device = "cuda"
    q = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_k, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_k, num_heads, head_dim, device=device, dtype=dtype)

    cu_seqlens_q = [0]
    for seq_len in q_lens:
        cu_seqlens_q.append(cu_seqlens_q[-1] + seq_len)
    cu_seqlens_k = [0]
    for seq_len in k_lens:
        cu_seqlens_k.append(cu_seqlens_k[-1] + seq_len)

    return (
        q,
        k,
        v,
        torch.tensor(cu_seqlens_q, device=device, dtype=torch.int32),
        torch.tensor(cu_seqlens_k, device=device, dtype=torch.int32),
        max(q_lens),
        max(k_lens),
    )


def _make_block_cache(
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    k_lens: list[int],
    block_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Pack dense K/V into a simple paged cache used by the toy `with_block`
    # path. `block_table[batch_id, logical_block] = physical_block_id`.
    num_heads = k_dense.shape[1]
    head_dim = k_dense.shape[2]
    blocks_per_seq = [(k_len + block_size - 1) // block_size for k_len in k_lens]
    total_blocks = sum(blocks_per_seq)
    max_blocks_per_seq = max(blocks_per_seq)

    k_cache = torch.zeros(
        total_blocks,
        block_size,
        num_heads,
        head_dim,
        device=k_dense.device,
        dtype=k_dense.dtype,
    )
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.full(
        (len(k_lens), max_blocks_per_seq),
        -1,
        device=k_dense.device,
        dtype=torch.long,
    )

    # Shuffle physical block ids so the test exercises logical->physical
    # translation instead of accidentally relying on identity mapping.
    physical_ids = list(range(total_blocks))
    physical_ids = physical_ids[::2] + physical_ids[1::2]

    dense_start = 0
    next_block = 0
    for batch_id, k_len in enumerate(k_lens):
        for logical_block in range(blocks_per_seq[batch_id]):
            physical_block = physical_ids[next_block]
            next_block += 1
            block_table[batch_id, logical_block] = physical_block

            token_start = dense_start + logical_block * block_size
            token_end = min(token_start + block_size, dense_start + k_len)
            valid = token_end - token_start
            k_cache[physical_block, :valid] = k_dense[token_start:token_end]
            v_cache[physical_block, :valid] = v_dense[token_start:token_end]
        dense_start += k_len

    return k_cache, v_cache, block_table


def _run_official(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    window_size: tuple[int, int] | None,
) -> torch.Tensor:
    # Official FA2 reference path.
    return flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        causal=causal,
        window_size=list(window_size) if window_size is not None else None,
        fa_version=2,
    )


def _run_toy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    window_size: tuple[int, int] | None,
) -> torch.Tensor:
    # Dense / non-paged toy path.
    return flash_attn_varlen_without_block(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        causal=causal,
        window_size=window_size,
    )


def _run_toy_with_block(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    seqused_k: torch.Tensor,
    causal: bool,
    window_size: tuple[int, int] | None,
    block_table: torch.Tensor,
) -> torch.Tensor:
    # Paged KV toy path. `seqused_k` is interpreted as per-request valid KV
    # length, and the toy implementation uses the same tail-alignment rule.
    return flash_attn_varlen_with_block(
        q=q,
        k=k_cache,
        v=v_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
    )


def _run_toy_with_block_cu(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    seqused_k: torch.Tensor,
    causal: bool,
    window_size: tuple[int, int] | None,
    block_table: torch.Tensor,
) -> torch.Tensor:
    return flash_attn_varlen_with_block_cu(
        q=q,
        k=k_cache,
        v=v_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
    )


def _assert_close(
    q_lens: list[int],
    k_lens: list[int] | None,
    causal: bool,
    window_size: tuple[int, int] | None,
    num_heads: int = 2,
    head_dim: int = 16,
    dtype: torch.dtype = torch.float16,
    seed: int = 0,
    use_block: bool = False,
) -> None:
    # Compare toy implementation against official FA2 on exactly the same
    # tensors and masking settings.
    case_desc = (
        f"q_lens={q_lens}, k_lens={k_lens}, causal={causal}, "
        f"window_size={window_size}, num_heads={num_heads}, "
        f"head_dim={head_dim}, use_block={use_block}"
    )
    print(f"[RUN ] {case_desc}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = _make_inputs(
        q_lens=q_lens,
        k_lens=k_lens,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    out_ref = _run_official(
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
        # Build a paged cache view for the same dense K/V, then compare the toy
        # block path against the dense official FA2 output.
        k_cache, v_cache, block_table = _make_block_cache(
            k_dense=k,
            v_dense=v,
            k_lens=k_lens if k_lens is not None else q_lens,
        )
        out_toy = _run_toy_with_block(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            seqused_k=torch.tensor(
                k_lens if k_lens is not None else q_lens,
                device=q.device,
                dtype=torch.int32,
            ),
            causal=causal,
            window_size=window_size,
            block_table=block_table,
        )
    else:
        out_toy = _run_toy(
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

    assert out_toy.shape == out_ref.shape
    assert torch.allclose(out_toy, out_ref, atol=2e-2, rtol=2e-2), (
        f"Mismatch with q_lens={q_lens}, k_lens={k_lens}, causal={causal}, "
        f"window_size={window_size}, use_block={use_block}"
    )
    print(f"[PASS] {case_desc}")


def _assert_close_with_block_cu(
    q_lens: list[int],
    k_lens: list[int] | None,
    causal: bool,
    window_size: tuple[int, int] | None,
    num_heads: int = 2,
    head_dim: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> None:
    case_desc = (
        f"q_lens={q_lens}, k_lens={k_lens}, causal={causal}, "
        f"window_size={window_size}, num_heads={num_heads}, "
        f"head_dim={head_dim}, dtype={dtype}, use_block_cu=True"
    )
    print(f"[RUN ] {case_desc}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    q, k, v, cu_seqlens_q, _, max_seqlen_q, max_seqlen_k = _make_inputs(
        q_lens=q_lens,
        k_lens=k_lens,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    k_cache, v_cache, block_table = _make_block_cache(
        k_dense=k,
        v_dense=v,
        k_lens=k_lens if k_lens is not None else q_lens,
    )
    seqused_k = torch.tensor(
        k_lens if k_lens is not None else q_lens,
        device=q.device,
        dtype=torch.int32,
    )

    out_ref = _run_toy_with_block(
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
    out_cu = _run_toy_with_block_cu(
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

    assert out_cu.shape == out_ref.shape
    assert torch.allclose(out_cu, out_ref, atol=3e-2, rtol=3e-2), (
        f"Mismatch with q_lens={q_lens}, k_lens={k_lens}, causal={causal}, "
        f"window_size={window_size}, use_block_cu=True"
    )
    print(f"[PASS] {case_desc}")


class FlashAttentionFuncTest(unittest.TestCase):
    def setUp(self) -> None:
        _require_fa2_cuda()

    def test_without_block_matches_fa2_full_attention(self) -> None:
        # Basic dense varlen self-attention.
        _assert_close(q_lens=[3, 5], k_lens=None, causal=False, window_size=None)

    def test_without_block_matches_fa2_causal_attention(self) -> None:
        _assert_close(q_lens=[3, 5], k_lens=None, causal=True, window_size=None)

    def test_without_block_matches_fa2_local_window(self) -> None:
        _assert_close(q_lens=[4, 6], k_lens=None, causal=False, window_size=(1, 1))

    def test_without_block_matches_fa2_causal_local_window(self) -> None:
        _assert_close(q_lens=[4, 6], k_lens=None, causal=True, window_size=(2, 0))

    def test_without_block_matches_fa2_more_varlen_batches(self) -> None:
        _assert_close(q_lens=[1, 3, 7, 2], k_lens=None, causal=False, window_size=None)

    def test_without_block_matches_fa2_longer_sequences(self) -> None:
        _assert_close(q_lens=[8, 11], k_lens=None, causal=True, window_size=None)

    def test_without_block_matches_fa2_tail_aligned_suffix_query(self) -> None:
        # Dense path with q_len < k_len. This is the important "tail aligned"
        # scenario where q[-1] is interpreted as aligned with kv[-1].
        _assert_close(
            q_lens=[2, 3],
            k_lens=[5, 7],
            causal=True,
            window_size=None,
        )

    def test_with_block_matches_fa2_full_attention(self) -> None:
        # Same semantics as the dense test, but K/V are read from paged cache.
        _assert_close(
            q_lens=[3, 5],
            k_lens=None,
            causal=False,
            window_size=None,
            use_block=True,
        )

    def test_with_block_matches_fa2_causal_attention(self) -> None:
        _assert_close(
            q_lens=[3, 5],
            k_lens=None,
            causal=True,
            window_size=None,
            use_block=True,
        )

    def test_with_block_matches_fa2_tail_aligned_suffix_query(self) -> None:
        # Paged-cache version of the tail-aligned suffix-query case.
        _assert_close(
            q_lens=[2, 3],
            k_lens=[5, 7],
            causal=True,
            window_size=None,
            use_block=True,
        )

    def test_with_block_cu_matches_python_causal_attention_bf16(self) -> None:
        _assert_close_with_block_cu(
            q_lens=[3, 5],
            k_lens=None,
            causal=True,
            window_size=None,
        )

    def test_with_block_cu_matches_python_full_attention_bf16(self) -> None:
        _assert_close_with_block_cu(
            q_lens=[3, 5],
            k_lens=None,
            causal=False,
            window_size=None,
        )

    def test_with_block_cu_matches_python_tail_aligned_suffix_query_bf16(self) -> None:
        _assert_close_with_block_cu(
            q_lens=[2, 3],
            k_lens=[5, 7],
            causal=True,
            window_size=None,
        )

    def test_without_block_matches_fa2_different_head_shapes(self) -> None:
        cases = [
            {"q_lens": [2, 5], "num_heads": 1, "head_dim": 8},
            {"q_lens": [3, 4], "num_heads": 4, "head_dim": 16},
            {"q_lens": [2, 6], "num_heads": 2, "head_dim": 32},
        ]
        for case in cases:
            with self.subTest(case=case):
                _assert_close(
                    q_lens=case["q_lens"],
                    k_lens=None,
                    causal=False,
                    window_size=None,
                    num_heads=case["num_heads"],
                    head_dim=case["head_dim"],
                )

    def test_without_block_matches_fa2_multiple_window_configs(self) -> None:
        cases = [
            {"q_lens": [5, 5], "causal": False, "window_size": (0, 0)},
            {"q_lens": [5, 5], "causal": False, "window_size": (2, 1)},
            {"q_lens": [5, 5], "causal": True, "window_size": (3, 0)},
        ]
        for case in cases:
            with self.subTest(case=case):
                _assert_close(
                    q_lens=case["q_lens"],
                    k_lens=None,
                    causal=case["causal"],
                    window_size=case["window_size"],
                )


if __name__ == "__main__":
    print("Running toy flash attention parity tests against official FA2...")
    unittest.main(verbosity=2)
