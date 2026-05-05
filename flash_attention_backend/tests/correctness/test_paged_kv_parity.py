from __future__ import annotations

import sys
import unittest

import torch

from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from flash_attention_backend.tests.correctness._helpers import assert_close
from flash_attention_backend.toy_flash_attn.test_utils import make_block_cache, make_inputs, require_fa2_cuda


class FlashAttentionPagedKvParityTest(unittest.TestCase):
    def setUp(self) -> None:
        require_fa2_cuda()

    def test_paged_kv_block_table_mapping_matches_fa2(self) -> None:
        assert_close(q_lens=[5, 5], k_lens=[8, 12], causal=True, window_size=None, use_block=True)

    def test_tail_aligned_suffix_query_matches_fa2(self) -> None:
        assert_close(q_lens=[2, 3], k_lens=[5, 7], causal=True, window_size=None, use_block=True)

    def test_local_and_causal_window_matches_fa2(self) -> None:
        assert_close(q_lens=[5, 5], k_lens=None, causal=True, window_size=(3, 0), use_block=True)

    def test_block_table_is_non_identity_mapping(self) -> None:
        q, k, v, *_ = make_inputs(q_lens=[2, 2], k_lens=[6, 6], dtype=torch.float16)
        del q, v
        _, _, block_table = make_block_cache(k_dense=k, v_dense=k, k_lens=[6, 6])
        first_row = block_table[0].tolist()
        expected_identity = list(range(len(first_row)))
        self.assertNotEqual(first_row, expected_identity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
