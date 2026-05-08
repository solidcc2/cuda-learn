#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from flash_attention_backend.bench.common import apply_version_env, write_json
from flash_attention_backend.bench.op.cases_op import case_payload
from flash_attention_backend.tests.correctness._helpers import (
    RELAXED_ATOL,
    RELAXED_RTOL,
    STRICT_ATOL,
    STRICT_RTOL,
    _compare_outputs,
)
from flash_attention_backend.toy_flash_attn.test_utils import (
    make_block_cache,
    make_inputs,
    require_cuda,
    require_fa2_cuda,
    require_with_block_cu_launch_constraints,
    run_official,
    run_toy_paged_cuda,
)


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="qwen_like_b1_s2048_h64")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def run_case(case_name: str) -> dict[str, Any]:
    case = case_payload(case_name)
    dtype = DTYPE_MAP[case["dtype"]]

    torch.manual_seed(case["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(case["seed"])

    require_cuda()
    require_fa2_cuda()
    require_with_block_cu_launch_constraints(case["head_dim"])

    q, k, v, cu_seqlens_q, _, max_seqlen_q, max_seqlen_k = make_inputs(
        q_lens=case["q_lens"],
        k_lens=case["k_lens"],
        num_heads=case["num_heads"],
        num_kv_heads=case["num_kv_heads"],
        head_dim=case["head_dim"],
        dtype=dtype,
    )
    k_cache, v_cache, block_table = make_block_cache(
        k_dense=k,
        v_dense=v,
        k_lens=case["k_lens"],
        block_size=16,
    )
    seqused_k = torch.tensor(case["k_lens"], device=q.device, dtype=torch.int32)

    out_official = run_official(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        causal=case["causal"],
        window_size=case["window_size"],
        block_table=block_table,
    )

    with apply_version_env("v7"):
        out_v7 = run_toy_paged_cuda(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            seqused_k=seqused_k,
            causal=case["causal"],
            window_size=case["window_size"],
            block_table=block_table,
        )

    result = _compare_outputs(
        out_v7,
        out_official,
        metadata={
            "case_name": case_name,
            "q_lens": case["q_lens"],
            "k_lens": case["k_lens"],
            "num_heads": case["num_heads"],
            "num_kv_heads": case["num_kv_heads"],
            "head_dim": case["head_dim"],
            "causal": case["causal"],
            "window_size": case["window_size"],
            "dtype": case["dtype"],
            "seed": case["seed"],
            "block_size": 16,
            "lhs": "v7_cuda_paged",
            "rhs": "official_fa2_paged",
            "strict_threshold": {"atol": STRICT_ATOL, "rtol": STRICT_RTOL},
            "relaxed_threshold": {"atol": RELAXED_ATOL, "rtol": RELAXED_RTOL},
        },
    )
    result["case_name"] = case_name
    result["kind"] = "flash_attention_backend.op_case_correctness"
    return result


def main() -> None:
    args = parse_args()
    payload = run_case(args.case)
    if args.output_json is not None:
        write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
