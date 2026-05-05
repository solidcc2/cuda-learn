from __future__ import annotations

import sys
import unittest

from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from flash_attention_backend.tests.correctness._helpers import assert_close
from flash_attention_backend.toy_flash_attn.test_utils import require_fa2_cuda


class FlashAttentionFa2ParityTest(unittest.TestCase):
    def setUp(self) -> None:
        require_fa2_cuda()

    def test_without_block_matches_fa2_full_attention(self) -> None:
        assert_close(q_lens=[3, 5], k_lens=None, causal=False, window_size=None)

    def test_without_block_matches_fa2_causal_local_window(self) -> None:
        assert_close(q_lens=[4, 6], k_lens=None, causal=True, window_size=(2, 0))

    def test_with_block_matches_fa2_full_attention(self) -> None:
        assert_close(q_lens=[3, 5], k_lens=None, causal=False, window_size=None, use_block=True)

    def test_with_block_matches_fa2_tail_aligned_suffix_query(self) -> None:
        assert_close(q_lens=[2, 3], k_lens=[5, 7], causal=True, window_size=None, use_block=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
