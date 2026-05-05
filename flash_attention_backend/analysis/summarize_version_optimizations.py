#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flash_attention_backend.bench.common import system_environment, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "kind": "flash_attention_backend.version_optimizations",
        "environment": system_environment(),
        "versions": [
            {
                "version": "baseline",
                "positioning": "Python reference paged attention path.",
                "added_vs_previous": [
                    "Reference implementation for paged KV semantics and correctness comparisons."
                ],
                "code_evidence": [
                    "flash_attention_backend/toy_flash_attn/flash_attention_func.py",
                    "flash_attention_backend/bench/common.py",
                ],
            },
            {
                "version": "v4",
                "positioning": "First custom CUDA paged path exposed through the shared wrapper.",
                "added_vs_previous": [
                    "Registers bf16 and fp32 CUDA aliases.",
                    "Uses the v4 CUDA kernel implementation under toy_flash_attn/v4/."
                ],
                "code_evidence": [
                    "flash_attention_backend/toy_flash_attn/flash_attention_func.py",
                    "flash_attention_backend/toy_flash_attn/v4/flash_attn_func.cu",
                    "flash_attention_backend/toy_flash_attn/v4/helper.h",
                ],
            },
            {
                "version": "v5",
                "positioning": "Custom CUDA paged path using the v5 WMMA/Tensor Core kernel.",
                "added_vs_previous": [
                    "Binds the v5 kernel through the shared bf16 CUDA wrapper alias.",
                    "Switches the implementation source to toy_flash_attn/v5/."
                ],
                "code_evidence": [
                    "flash_attention_backend/toy_flash_attn/flash_attention_func.py",
                    "flash_attention_backend/toy_flash_attn/v5/flash_attn_func.cu",
                    "flash_attention_backend/toy_flash_attn/v5/helper.h",
                ],
            },
            {
                "version": "v6",
                "positioning": "Custom CUDA paged path using the CuTe-based v6 kernel.",
                "added_vs_previous": [
                    "Makes v6 the default CUDA implementation when TOY_FLASH_ATTN_CUDA_VERSION is unset.",
                    "Binds the v6 kernel through the shared bf16 CUDA wrapper alias.",
                    "Current exported specialization is head_dim=64 only."
                ],
                "code_evidence": [
                    "flash_attention_backend/toy_flash_attn/flash_attention_func.py",
                    "flash_attention_backend/toy_flash_attn/v6/flash_attn_func.cu",
                    "flash_attention_backend/toy_flash_attn/v6/helper.h",
                ],
            },
            {
                "version": "official",
                "positioning": "Official FlashAttention decoder path used as the external performance reference.",
                "added_vs_previous": [
                    "Routes to the official backend configuration rather than the custom paged CUDA path.",
                    "Current op-level comparison boundary remains dense-path oriented."
                ],
                "code_evidence": [
                    "flash_attention_backend/bench/common.py",
                    "flash_attention_backend/bench/e2e/run_vllm_e2e.py",
                ],
            },
        ],
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
