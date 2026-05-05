#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flash_attention_backend.bench.common import system_environment, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-summary", type=Path, required=True)
    parser.add_argument("--op-dir", type=Path, required=True)
    parser.add_argument("--op-output", type=Path, required=True)
    parser.add_argument("--correctness-input", type=Path, required=True)
    parser.add_argument("--correctness-output", type=Path, required=True)
    parser.add_argument("--version-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _aggregate_op_results(op_dir: Path) -> dict[str, Any]:
    benchmarks = []
    for path in sorted(op_dir.glob("*.json")):
        benchmarks.append(_load_json(path))
    return {
        "kind": "flash_attention_backend.op_summary",
        "benchmarks": benchmarks,
        "count": len(benchmarks),
    }


def main() -> None:
    args = parse_args()
    e2e_summary = _load_optional_json(args.e2e_summary)
    op_summary = _aggregate_op_results(args.op_dir)
    correctness_summary = _load_optional_json(args.correctness_input)
    version_summary = _load_optional_json(args.version_summary)

    write_json(args.op_output, op_summary)
    if correctness_summary is not None:
        write_json(args.correctness_output, correctness_summary)

    payload = {
        "kind": "flash_attention_backend.report_inputs",
        "environment": system_environment(),
        "e2e": e2e_summary,
        "op": op_summary,
        "correctness": correctness_summary,
        "version_optimizations": version_summary,
        "profiling": {
            "status": "未采集",
            "note": "Unified analysis entry does not execute profiler runs by default.",
        },
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
