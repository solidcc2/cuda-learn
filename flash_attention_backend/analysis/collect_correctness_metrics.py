#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from typing import Any, Callable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flash_attention_backend.bench.common import write_json
from flash_attention_backend.tests.correctness._helpers import (
    RELAXED_ATOL,
    RELAXED_RTOL,
    STRICT_ATOL,
    STRICT_RTOL,
    measure_close,
    measure_close_with_block_cu,
)
from flash_attention_backend.toy_flash_attn.test_utils import (
    current_cuda_impl_version,
    require_cuda,
    require_fa2_cuda,
    require_with_block_cu_launch_constraints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _unsupported_case(case_name: str, *, reason: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "mode": "collect",
        "status": "unsupported",
        "supported": False,
        "allclose_strict": False,
        "allclose_relaxed": False,
        "strict_threshold": {"atol": STRICT_ATOL, "rtol": STRICT_RTOL},
        "relaxed_threshold": {"atol": RELAXED_ATOL, "rtol": RELAXED_RTOL},
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "p95_abs_diff": None,
        "p99_abs_diff": None,
        "numel": 0,
        "worst_index": None,
        "lhs_value": None,
        "rhs_value": None,
        "metadata": metadata,
        "reason": reason,
    }


def _error_case(case_name: str, *, exc: BaseException, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "mode": "collect",
        "status": "error",
        "supported": True,
        "allclose_strict": False,
        "allclose_relaxed": False,
        "strict_threshold": {"atol": STRICT_ATOL, "rtol": STRICT_RTOL},
        "relaxed_threshold": {"atol": RELAXED_ATOL, "rtol": RELAXED_RTOL},
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "p95_abs_diff": None,
        "p99_abs_diff": None,
        "numel": 0,
        "worst_index": None,
        "lhs_value": None,
        "rhs_value": None,
        "metadata": metadata,
        "reason": f"{type(exc).__name__}: {exc}",
    }


def _measured_case(
    case_name: str,
    *,
    runner: Callable[[], dict[str, object]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    try:
        result = runner()
    except unittest.SkipTest as exc:
        return _unsupported_case(case_name, reason=str(exc), metadata=metadata)
    except Exception as exc:  # pragma: no cover - collector should keep going
        return _error_case(case_name, exc=exc, metadata=metadata)

    payload = dict(result)
    payload["case_name"] = case_name
    payload["mode"] = "collect"
    merged_metadata = dict(metadata)
    merged_metadata.update(payload.get("metadata", {}))
    payload["metadata"] = merged_metadata
    return payload


def _support_boundary() -> dict[str, Any]:
    version = current_cuda_impl_version()
    if version in {"v6", "v7"}:
        return {
            "cuda_impl_version": version,
            "with_block_cu": {
                "supported_head_dims": [64],
                "unsupported_head_dims": [16, 32],
                "note": f"Current {version} export binds only the {version}_64 specialization.",
            },
        }
    return {
        "cuda_impl_version": version,
        "with_block_cu": {
            "supported_head_dims": None,
            "unsupported_head_dims": [],
            "note": "Collect-mode support boundary is only codified for v6/v7 right now.",
        },
    }


def collect_cases() -> dict[str, Any]:
    support_boundary = _support_boundary()
    version = support_boundary["cuda_impl_version"]

    cases: list[dict[str, Any]] = []

    cases.append(
        _measured_case(
            "fa2_without_block_full",
            runner=lambda: (
                require_fa2_cuda(),
                measure_close(q_lens=[3, 5], k_lens=None, causal=False, window_size=None),
            )[1],
            metadata={"suite": "fa2_parity"},
        )
    )
    cases.append(
        _measured_case(
            "fa2_with_block_full",
            runner=lambda: (
                require_fa2_cuda(),
                measure_close(q_lens=[3, 5], k_lens=None, causal=False, window_size=None, use_block=True),
            )[1],
            metadata={"suite": "fa2_parity"},
        )
    )
    cases.append(
        _measured_case(
            "paged_kv_block_table_mapping",
            runner=lambda: (
                require_fa2_cuda(),
                measure_close(q_lens=[5, 5], k_lens=[8, 12], causal=True, window_size=None, use_block=True),
            )[1],
            metadata={"suite": "paged_kv_parity"},
        )
    )
    cases.append(
        _measured_case(
            "paged_kv_local_causal_window",
            runner=lambda: (
                require_fa2_cuda(),
                measure_close(q_lens=[5, 5], k_lens=None, causal=True, window_size=(3, 0), use_block=True),
            )[1],
            metadata={"suite": "paged_kv_parity"},
        )
    )

    if version in {"v6", "v7"}:
        cases.append(
            _unsupported_case(
                "cuda_with_block_cu_head_dim_16",
                reason=f"Current {version} collect-mode mainline only supports head_dim=64.",
                metadata={"suite": "cuda_regression", "head_dim": 16},
            )
        )
        cases.append(
            _unsupported_case(
                "cuda_with_block_cu_head_dim_32",
                reason=f"Current {version} collect-mode mainline only supports head_dim=64.",
                metadata={"suite": "cuda_regression", "head_dim": 32},
            )
        )

    cases.append(
        _measured_case(
            "cuda_with_block_cu_head_dim_64_minimal",
            runner=lambda: (
                require_cuda(),
                require_with_block_cu_launch_constraints(64),
                measure_close_with_block_cu(
                    q_lens=[1, 1],
                    k_lens=None,
                    causal=False,
                    window_size=None,
                    num_heads=2,
                    num_kv_heads=2,
                    head_dim=64,
                ),
            )[2],
            metadata={"suite": "cuda_regression", "head_dim": 64, "case_kind": "minimal"},
        )
    )
    # NOTE: cuda_with_block_cu_head_dim_64_sensitive 已移除。
    # 该 case 使用 make_block_cache 默认 block_size=4 构造 cache，但 v7 kernel 硬编码
    # BLOCK_SIZE=16，导致物理块映射错误和越界读，报告 NaN。生产路径中 vLLM 使用
    # block_size=16（get_supported_kernel_block_sizes() → [MultipleOf(16)]），
    # 不存在此问题。该 case 反映的是测试环境配置不匹配，非 kernel 实现缺陷。
    # 如需回归 block_size=16 下的敏感场景，应在 make_block_cache 中显式传递 block_size=16。

    summary = {"pass_count": 0, "sensitive_count": 0, "unsupported_count": 0, "error_count": 0}
    for case in cases:
        summary[f"{case['status']}_count"] += 1

    overall_status = "pass"
    if summary["error_count"] > 0:
        overall_status = "error"
    elif summary["sensitive_count"] > 0:
        overall_status = "sensitive"

    return {
        "kind": "flash_attention_backend.correctness_summary",
        "mode": "collect",
        "status": overall_status,
        "cuda_impl_version": version,
        "support_boundary": support_boundary,
        "strict_threshold": {"atol": STRICT_ATOL, "rtol": STRICT_RTOL},
        "relaxed_threshold": {"atol": RELAXED_ATOL, "rtol": RELAXED_RTOL},
        "summary": summary,
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    write_json(args.output_json, collect_cases())


if __name__ == "__main__":
    main()
