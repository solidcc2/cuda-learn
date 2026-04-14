from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run_and_log(cmd: list[str], *, env: dict[str, str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _fresh_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate baseline/custom GPT-2 attention dumps and run drift analysis."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("flash_attention_backend"),
        help="directory that will contain dump dirs and logs",
    )
    parser.add_argument(
        "--base-name",
        default="base_gpt2.pt",
        help="baseline dump directory name",
    )
    parser.add_argument(
        "--custom-name",
        default="bf16_fp32.pt",
        help="custom dump directory name",
    )
    parser.add_argument(
        "--analyze-script",
        type=Path,
        default=Path("flash_attention_backend/toy_flash_attn/analyze_flash_attn_dumps.py"),
        help="analysis script path",
    )
    parser.add_argument(
        "--model-script",
        type=Path,
        default=Path("flash_attention_backend/test_self_flash_attn_backend.py"),
        help="script that runs the real GPT-2 generation path",
    )
    parser.add_argument(
        "--input-threshold",
        type=float,
        default=1e-5,
        help="input threshold passed to the analysis script",
    )
    parser.add_argument(
        "--output-threshold",
        type=float,
        default=1e-3,
        help="output threshold passed to the analysis script",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="top-k worst steps to print in the report",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    output_root = (repo_root / args.output_root).resolve()
    base_dump_dir = output_root / args.base_name
    custom_dump_dir = output_root / args.custom_name
    base_log = output_root / "base_gpt2.log"
    custom_log = output_root / "bf16_fp32.log"
    analyze_log = output_root / "analyze.log"

    _fresh_dir(base_dump_dir)
    _fresh_dir(custom_dump_dir)

    common_env = os.environ.copy()

    base_env = common_env.copy()
    base_env["TOY_FLASH_ATTN_USE_WITH_BLOCK"] = "1"
    base_env["TOY_FLASH_ATTN_DUMP_DIR"] = str(base_dump_dir)
    _run_and_log(
        [sys.executable, str(args.model_script)],
        env=base_env,
        log_path=base_log,
        cwd=repo_root,
    )

    custom_env = common_env.copy()
    custom_env.pop("TOY_FLASH_ATTN_USE_WITH_BLOCK", None)
    custom_env["TOY_FLASH_ATTN_DUMP_DIR"] = str(custom_dump_dir)
    _run_and_log(
        [sys.executable, str(args.model_script)],
        env=custom_env,
        log_path=custom_log,
        cwd=repo_root,
    )

    analyze_env = common_env.copy()
    _run_and_log(
        [
            sys.executable,
            str(args.analyze_script),
            str(base_dump_dir),
            str(custom_dump_dir),
            "--input-threshold",
            str(args.input_threshold),
            "--output-threshold",
            str(args.output_threshold),
            "--top-k",
            str(args.top_k),
        ],
        env=analyze_env,
        log_path=analyze_log,
        cwd=repo_root,
    )

    print(f"base dump dir: {base_dump_dir}")
    print(f"custom dump dir: {custom_dump_dir}")
    print(f"base log: {base_log}")
    print(f"custom log: {custom_log}")
    print(f"analysis report: {analyze_log}")


if __name__ == "__main__":
    main()
