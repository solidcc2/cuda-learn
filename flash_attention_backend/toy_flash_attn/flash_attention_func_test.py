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


def _require_fa2_cuda() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required for this test.")
    if not is_flash_attn_varlen_func_available():
        raise unittest.SkipTest(
            "Official flash_attn_varlen_func is not available in this environment."
        )
    if get_flash_attn_version() != 2:
        raise unittest.SkipTest("This test expects FA2 to be active.")


def _make_inputs(
    seq_lens: list[int],
    num_heads: int = 2,
    head_dim: int = 16,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    total = sum(seq_lens)
    device = "cuda"
    q = torch.randn(total, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total, num_heads, head_dim, device=device, dtype=dtype)

    cu_seqlens = [0]
    for seq_len in seq_lens:
        cu_seqlens.append(cu_seqlens[-1] + seq_len)
    cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
    max_seq_len = max(seq_lens)
    return q, k, v, cu_seqlens, max_seq_len


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
    return flash_attn_varlen_without_block(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q.tolist(),
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k.tolist(),
        causal=causal,
        window_size=window_size,
    )


def _assert_close(
    seq_lens: list[int],
    causal: bool,
    window_size: tuple[int, int] | None,
) -> None:
    q, k, v, cu_seqlens, max_seq_len = _make_inputs(seq_lens)
    out_ref = _run_official(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seq_len,
        max_seqlen_k=max_seq_len,
        causal=causal,
        window_size=window_size,
    )
    out_toy = _run_toy(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seq_len,
        max_seqlen_k=max_seq_len,
        causal=causal,
        window_size=window_size,
    )

    assert out_toy.shape == out_ref.shape
    assert torch.allclose(out_toy, out_ref, atol=2e-2, rtol=2e-2), (
        f"Mismatch with seq_lens={seq_lens}, causal={causal}, "
        f"window_size={window_size}"
    )


class FlashAttentionFuncTest(unittest.TestCase):
    def setUp(self) -> None:
        _require_fa2_cuda()

    def test_without_block_matches_fa2_full_attention(self) -> None:
        _assert_close(seq_lens=[3, 5], causal=False, window_size=None)

    def test_without_block_matches_fa2_causal_attention(self) -> None:
        _assert_close(seq_lens=[3, 5], causal=True, window_size=None)

    def test_without_block_matches_fa2_local_window(self) -> None:
        _assert_close(seq_lens=[4, 6], causal=False, window_size=(1, 1))


if __name__ == "__main__":
    unittest.main()
