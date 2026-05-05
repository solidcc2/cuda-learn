#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from flash_attention_backend.bench.common import system_environment, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _safe_path(path: Path) -> str:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return resolved.name


def main() -> None:
    args = parse_args()
    json_dir = args.json_dir.resolve()
    entries = []
    for path in sorted(json_dir.glob("*.json")):
        if path == args.output.resolve():
            continue
        entries.append(__import__("json").loads(path.read_text()))
    write_json(
        args.output.resolve(),
        {
            "kind": "flash_attention_backend.e2e_summary",
            "environment": system_environment(),
            "json_dir": _safe_path(json_dir),
            "benchmarks": entries,
        },
    )


if __name__ == "__main__":
    main()
