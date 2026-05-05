#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "bench"
    / "e2e"
    / "parse_perf_eval_logs.py"
)

if __name__ == "__main__":
    print(
        "[compat] forwarding analysis/parse_perf_eval_logs.py -> "
        "bench/e2e/parse_perf_eval_logs.py",
        file=sys.stderr,
    )
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
