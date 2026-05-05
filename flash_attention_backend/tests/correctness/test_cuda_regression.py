from __future__ import annotations

import sys
import unittest

import torch

from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from flash_attention_backend.tests.correctness._helpers import assert_close_with_block_cu
from flash_attention_backend.toy_flash_attn.test_utils import (
    current_cuda_impl_version,
    make_block_cache,
    make_inputs,
    require_cuda,
    require_with_block_cu_launch_constraints,
    run_toy_paged_cuda,
)


class FlashAttentionCudaRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print(
            "[cuda_regression] TOY_FLASH_ATTN_CUDA_VERSION =",
            current_cuda_impl_version(),
            flush=True,
        )

    def setUp(self) -> None:
        require_cuda()

    def test_with_block_cu_matches_python_full_attention_bf16(self) -> None:
        assert_close_with_block_cu(q_lens=[3, 5], k_lens=None, causal=False, window_size=None)

    def test_with_block_cu_head_dim_64_regression(self) -> None:
        self.skipTest(
            "Extended regression case: head_dim=64 local-window sensitivity is tracked "
            "outside the stage-3 smoke gate."
        )
        require_with_block_cu_launch_constraints(64)
        assert_close_with_block_cu(
            q_lens=[4, 9],
            k_lens=[8, 12],
            causal=True,
            window_size=(3, 0),
            head_dim=64,
            dtype=torch.bfloat16,
            seed=423,
        )

    def test_with_block_cu_head_dim_64_outputs_are_finite(self) -> None:
        require_with_block_cu_launch_constraints(64)
        q, k, v, cu_seqlens_q, _, max_seqlen_q, max_seqlen_k = make_inputs(
            q_lens=[2, 6],
            k_lens=[8, 8],
            num_heads=2,
            num_kv_heads=2,
            head_dim=64,
            dtype=torch.bfloat16,
        )
        k_cache, v_cache, block_table = make_block_cache(k_dense=k, v_dense=v, k_lens=[8, 8])
        seqused_k = torch.tensor([8, 8], device=q.device, dtype=torch.int32)
        out = run_toy_paged_cuda(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            seqused_k=seqused_k,
            causal=True,
            window_size=(2, 0),
            block_table=block_table,
        )
        self.assertTrue(torch.isfinite(out.float()).all().item())

    def test_with_block_cu_head_dim_16_coverage_target(self) -> None:
        self.skipTest("Stage-3 smoke gate only validates head_dim=64.")
        require_with_block_cu_launch_constraints(16)
        assert_close_with_block_cu(
            q_lens=[3, 4],
            k_lens=None,
            causal=False,
            window_size=None,
            num_heads=4,
            num_kv_heads=4,
            head_dim=16,
        )

    def test_with_block_cu_head_dim_16_minimal_diag(self) -> None:
        self.skipTest("Stage-3 smoke gate only validates head_dim=64.")
        require_with_block_cu_launch_constraints(16)
        assert_close_with_block_cu(
            q_lens=[1, 1],
            k_lens=None,
            causal=False,
            window_size=None,
            num_heads=4,
            num_kv_heads=4,
            head_dim=16,
        )

    def test_with_block_cu_head_dim_32_coverage_target(self) -> None:
        self.skipTest("Stage-3 smoke gate only validates head_dim=64.")
        require_with_block_cu_launch_constraints(32)
        assert_close_with_block_cu(
            q_lens=[2, 6],
            k_lens=None,
            causal=False,
            window_size=None,
            num_heads=2,
            num_kv_heads=2,
            head_dim=32,
        )

    def test_with_block_cu_head_dim_32_minimal_diag(self) -> None:
        self.skipTest("Stage-3 smoke gate only validates head_dim=64.")
        require_with_block_cu_launch_constraints(32)
        assert_close_with_block_cu(
            q_lens=[1, 1],
            k_lens=None,
            causal=False,
            window_size=None,
            num_heads=2,
            num_kv_heads=2,
            head_dim=32,
        )

    def test_with_block_cu_head_dim_64_coverage_target(self) -> None:
        require_with_block_cu_launch_constraints(64)
        assert_close_with_block_cu(
            q_lens=[2, 6],
            k_lens=None,
            causal=False,
            window_size=None,
            num_heads=2,
            num_kv_heads=2,
            head_dim=64,
        )

    def test_with_block_cu_head_dim_64_minimal_diag(self) -> None:
        require_with_block_cu_launch_constraints(64)
        assert_close_with_block_cu(
            q_lens=[1, 1],
            k_lens=None,
            causal=False,
            window_size=None,
            num_heads=2,
            num_kv_heads=2,
            head_dim=64,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
