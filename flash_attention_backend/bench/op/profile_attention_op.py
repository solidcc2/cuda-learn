from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from flash_attention_backend.bench.common import apply_version_env, write_json
from flash_attention_backend.bench.op.bench_attention_op import (
    _build_inputs,
    _make_runner,
)
from flash_attention_backend.bench.op.cases_op import get_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--backend-only", action="store_true")
    parser.add_argument("--dump-input-meta", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    op_case = get_case(args.case)
    case = {
        "case_name": op_case.name,
        "q_lens": op_case.q_lens,
        "k_lens": op_case.k_lens,
        "num_heads": op_case.num_heads,
        "num_kv_heads": op_case.num_kv_heads,
        "head_dim": op_case.head_dim,
        "causal": op_case.causal,
        "window_size": op_case.window_size,
        "dtype": op_case.dtype,
        "seed": op_case.seed,
    }
    try:
        with apply_version_env(args.version):
            tensors = _build_inputs(case)
            runner = _make_runner(args.version, case, tensors)

            for _ in range(args.warmup):
                runner()
            for _ in range(args.iters):
                runner()
    except unittest.SkipTest as exc:
        raise SystemExit(
            "Op profiling prerequisites are not satisfied: "
            f"{exc}"
        ) from exc

    if args.dump_input_meta is not None:
        write_json(
            args.dump_input_meta,
            {
                "case_name": case["case_name"],
                "version": args.version,
                "backend_only": args.backend_only,
                "metadata": {
                    "q_lens": case["q_lens"],
                    "k_lens": case["k_lens"],
                    "num_heads": case["num_heads"],
                    "num_kv_heads": case["num_kv_heads"],
                    "head_dim": case["head_dim"],
                    "causal": case["causal"],
                    "window_size": case["window_size"],
                    "dtype": case["dtype"],
                    "seed": case["seed"],
                },
            },
        )


if __name__ == "__main__":
    main()
