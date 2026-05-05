from __future__ import annotations

import sys
import unittest

from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from flash_attention_backend.tests.correctness._helpers import replay_dump_close
from flash_attention_backend.toy_flash_attn.test_utils import (
    STEP0_REPLAY_DUMP,
    VLLM_MATCH_TOP10_WORST_DUMPS,
    iter_replay_dump_paths,
    require_cuda,
    require_dump_path_env,
    resolve_named_dump_paths,
)


class FlashAttentionReplayDumpTest(unittest.TestCase):
    def setUp(self) -> None:
        require_cuda()

    def test_replay_dump_matches_python(self) -> None:
        for dump_path in iter_replay_dump_paths(require_dump_path_env()):
            with self.subTest(dump=str(dump_path)):
                replay_dump_close(dump_path)

    def test_replay_top10_vllm_worst_dumps(self) -> None:
        dump_root = require_dump_path_env()
        if dump_root.is_file():
            dump_root = dump_root.parent
        for dump_path in resolve_named_dump_paths(dump_root, VLLM_MATCH_TOP10_WORST_DUMPS):
            with self.subTest(dump=str(dump_path)):
                replay_dump_close(dump_path)

    def test_replay_step0_matches_python(self) -> None:
        dump_root = require_dump_path_env()
        if dump_root.is_file():
            dump_root = dump_root.parent
        dump_path = resolve_named_dump_paths(dump_root, [STEP0_REPLAY_DUMP])[0]
        replay_dump_close(dump_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
