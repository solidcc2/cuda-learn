from __future__ import annotations

import argparse
import importlib
import sys
import time
import unittest
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from flash_attention_backend.bench.common import (
    VERSION_CONFIGS,
    apply_version_env,
    summarize_ms,
    system_environment,
    version_metadata,
    write_json,
)
from flash_attention_backend.bench.op.cases_op import CASES, get_case


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=sorted(VERSION_CONFIGS), required=True)
    parser.add_argument("--case", choices=sorted(CASES), default=None)
    parser.add_argument("--q-lens", type=int, nargs="+", default=None)
    parser.add_argument("--k-lens", type=int, nargs="+", default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--window-size", type=int, nargs=2, default=None)
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if args.iters <= 0:
        parser.error("--iters must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.case is None and None in {
        tuple(args.q_lens) if args.q_lens is not None else None,
        tuple(args.k_lens) if args.k_lens is not None else None,
        args.num_heads,
        args.num_kv_heads,
        args.head_dim,
        args.dtype,
        args.seed,
    }:
        parser.error("manual mode requires q/k lenses, head counts, head dim, dtype, and seed")
    return args


def _resolved_case(args: argparse.Namespace) -> dict[str, Any]:
    if args.case is not None:
        case = get_case(args.case)
        return {
            "case_name": case.name,
            "q_lens": case.q_lens,
            "k_lens": case.k_lens,
            "num_heads": case.num_heads,
            "num_kv_heads": case.num_kv_heads,
            "head_dim": case.head_dim,
            "causal": case.causal,
            "window_size": case.window_size,
            "dtype": case.dtype,
            "seed": case.seed,
        }
    return {
        "case_name": "manual",
        "q_lens": args.q_lens,
        "k_lens": args.k_lens,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "causal": args.causal,
        "window_size": tuple(args.window_size) if args.window_size is not None else None,
        "dtype": args.dtype,
        "seed": args.seed,
    }


def _build_inputs(case: dict[str, Any]) -> dict[str, Any]:
    test_utils = importlib.import_module("flash_attention_backend.toy_flash_attn.test_utils")
    test_utils.require_cuda()
    dtype = DTYPE_MAP[case["dtype"]]
    torch.manual_seed(case["seed"])
    torch.cuda.manual_seed_all(case["seed"])
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = test_utils.make_inputs(
        q_lens=case["q_lens"],
        k_lens=case["k_lens"],
        num_heads=case["num_heads"],
        num_kv_heads=case["num_kv_heads"],
        head_dim=case["head_dim"],
        dtype=dtype,
    )
    k_cache, v_cache, block_table = test_utils.make_block_cache(
        k_dense=k,
        v_dense=v,
        k_lens=case["k_lens"],
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "seqused_k": torch.tensor(case["k_lens"], device=q.device, dtype=torch.int32),
        "block_table": block_table,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }


def _make_runner(version: str, case: dict[str, Any], tensors: dict[str, Any]):
    test_utils = importlib.import_module("flash_attention_backend.toy_flash_attn.test_utils")
    if version == "official":
        test_utils.require_fa2_cuda()

        def run() -> torch.Tensor:
            return test_utils.run_official(
                q=tensors["q"],
                k=tensors["k"],
                v=tensors["v"],
                cu_seqlens_q=tensors["cu_seqlens_q"],
                cu_seqlens_k=tensors["cu_seqlens_k"],
                max_seqlen_q=tensors["max_seqlen_q"],
                max_seqlen_k=tensors["max_seqlen_k"],
                causal=case["causal"],
                window_size=case["window_size"],
            )

        return run

    if version == "baseline":
        def run() -> torch.Tensor:
            return test_utils.run_toy_paged(
                q=tensors["q"],
                k_cache=tensors["k_cache"],
                v_cache=tensors["v_cache"],
                cu_seqlens_q=tensors["cu_seqlens_q"],
                max_seqlen_q=tensors["max_seqlen_q"],
                max_seqlen_k=tensors["max_seqlen_k"],
                seqused_k=tensors["seqused_k"],
                causal=case["causal"],
                window_size=case["window_size"],
                block_table=tensors["block_table"],
            )

        return run

    test_utils.require_with_block_cu_launch_constraints(case["head_dim"])

    def run() -> torch.Tensor:
        return test_utils.run_toy_paged_cuda(
            q=tensors["q"],
            k_cache=tensors["k_cache"],
            v_cache=tensors["v_cache"],
            cu_seqlens_q=tensors["cu_seqlens_q"],
            max_seqlen_q=tensors["max_seqlen_q"],
            max_seqlen_k=tensors["max_seqlen_k"],
            seqused_k=tensors["seqused_k"],
            causal=case["causal"],
            window_size=case["window_size"],
            block_table=tensors["block_table"],
        )

    return run


def _time_runner(fn, warmup: int, iters: int) -> tuple[list[float], torch.Tensor]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples_ms: list[float] = []
    result = None
    for _ in range(iters):
        started = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    assert result is not None
    return samples_ms, result


def main() -> None:
    args = _parse_args()
    case = _resolved_case(args)
    try:
        with apply_version_env(args.version):
            tensors = _build_inputs(case)
            runner = _make_runner(args.version, case, tensors)
            samples_ms, result = _time_runner(runner, args.warmup, args.iters)
            metrics = summarize_ms(samples_ms)
            total_q = sum(case["q_lens"])
            total_k = sum(case["k_lens"])

            payload = {
                "kind": "flash_attention_backend.op_case",
                "case_name": case["case_name"],
                "version": args.version,
                "path_kind": version_metadata(args.version)["path_kind"],
                "avg_ms": metrics["avg_ms"],
                "min_ms": metrics["min_ms"],
                "max_ms": metrics["max_ms"],
                "std_ms": metrics["std_ms"],
                "iters": args.iters,
                "warmup": args.warmup,
                "tokens_per_s": (total_q * args.iters * 1000.0 / sum(samples_ms)) if samples_ms else None,
                "metadata": {
                    "q_lens": case["q_lens"],
                    "k_lens": case["k_lens"],
                    "batch": len(case["q_lens"]),
                    "num_heads": case["num_heads"],
                    "num_kv_heads": case["num_kv_heads"],
                    "head_dim": case["head_dim"],
                    "causal": case["causal"],
                    "window_size": case["window_size"],
                    "dtype": case["dtype"],
                    "seed": case["seed"],
                    "total_q_tokens": total_q,
                    "total_k_tokens": total_k,
                    "is_gqa": case["num_heads"] != case["num_kv_heads"],
                    "version_description": version_metadata(args.version)["description"],
                    "result_shape": list(result.shape),
                },
                "version_metadata": version_metadata(args.version),
                "environment": system_environment(),
            }
    except unittest.SkipTest as exc:
        raise SystemExit(
            "Op benchmark prerequisites are not satisfied: "
            f"{exc}"
        ) from exc
    print("case_name      =", payload["case_name"])
    print("version        =", payload["version"])
    print("path_kind      =", payload["path_kind"])
    print("avg_ms         =", payload["avg_ms"])
    print("tokens_per_s   =", payload["tokens_per_s"])
    if args.output_json is not None:
        write_json(args.output_json, payload)
        print("output_json    =", args.output_json)


if __name__ == "__main__":
    main()
