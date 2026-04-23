#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPEED_RE = re.compile(
    r"est\. speed input:\s*([0-9.]+)\s*toks/s,\s*output:\s*([0-9.]+)\s*toks/s"
)
WALL_TIME_RE = re.compile(r"\b([0-9.]+)s/it\b")
KEY_VALUE_RE = re.compile(r"^(model|revision|TOY_FLASH_ATTN_USE|attention_backend|kv_cache_dtype arg|batch_size|max_tokens|cache_config\.cache_dtype|model_config\.dtype|resolved kv torch dtype)\s*=\s*(.*)$")
FINISH_RE = re.compile(r"finish_reason\s*[=:]\s*([A-Za-z0-9_ -]+)")
ABS_PATH_RE = re.compile(r"(?<![\w.~-])/(?:[^\s:'\",]+/?)+")


def _run_text(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return path.name


def _sanitize_text(text: str, repo_root: Path) -> str:
    repo_root_text = str(repo_root.resolve())
    text = text.replace(repo_root_text, ".")
    home = str(Path.home())
    text = text.replace(home, "<home>")
    return ABS_PATH_RE.sub("<abs-path>", text)


def _environment(repo_root: Path) -> dict[str, Any]:
    env: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "git_commit": _run_text(["git", "rev-parse", "HEAD"], cwd=repo_root),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        env["torch_error"] = str(exc)

    try:
        import vllm

        env["vllm"] = getattr(vllm, "__version__", None)
    except Exception as exc:
        env["vllm_error"] = str(exc)

    return env


def _case_from_name(path: Path) -> dict[str, Any]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 4:
        return {"case_name": stem}

    batch_part = parts[-2]
    tokens_part = parts[-1]
    model = parts[-3]
    version = "_".join(parts[:-3])

    case: dict[str, Any] = {
        "case_name": stem,
        "version": version,
        "model_arg": model,
    }
    if batch_part.startswith("b") and batch_part[1:].isdigit():
        case["batch"] = int(batch_part[1:])
    if tokens_part.startswith("t") and tokens_part[1:].isdigit():
        case["max_tokens"] = int(tokens_part[1:])
    return case


def _last_speed(text: str) -> tuple[float | None, float | None]:
    matches = SPEED_RE.findall(text)
    if not matches:
        return None, None
    input_speed, output_speed = matches[-1]
    return float(input_speed), float(output_speed)


def _last_wall_time_s(text: str) -> float | None:
    matches = WALL_TIME_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def _config_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = KEY_VALUE_RE.match(line.strip())
        if match:
            values[match.group(1)] = match.group(2)
    return values


def _finish_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    for match in FINISH_RE.finditer(text):
        reason = match.group(1).strip()
        if reason:
            reasons.append(reason)
    return reasons


def _error_summary(text: str, repo_root: Path) -> str | None:
    markers = [
        "Traceback (most recent call last):",
        "EngineCore encountered a fatal error.",
        "CUDA error:",
        "RuntimeError:",
        "Error:",
        "env:",
        "command not found",
        "No such file or directory",
    ]
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if any(marker in line for marker in markers):
            return _sanitize_text("\n".join(lines[idx : min(len(lines), idx + 12)]), repo_root)
    return None


def _parse_log(path: Path, log_root: Path, repo_root: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    case = _case_from_name(path)
    values = _config_values(text)
    input_speed, output_speed = _last_speed(text)
    wall_time_s = _last_wall_time_s(text)
    batch = case.get("batch")
    error = _error_summary(text, repo_root)

    result: dict[str, Any] = {
        **case,
        "log_path": _repo_relative(path, repo_root),
        "success": error is None,
        "config": values,
        "input_toks_per_s": input_speed,
        "output_toks_per_s": output_speed,
        "wall_time_s": wall_time_s,
        "finish_reasons": _finish_reasons(text),
        "prompt_count": len(re.findall(r"^Prompt\s+[0-9]+:", text, flags=re.MULTILINE)),
        "generated_count": len(re.findall(r"^Generated:", text, flags=re.MULTILINE)),
        "error": error,
    }
    if output_speed is not None and isinstance(batch, int) and batch > 0:
        result["output_toks_per_s_per_request"] = output_speed / batch
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.resolve()
    output = args.output.resolve()
    repo_root = args.repo_root.resolve()

    logs = sorted(log_dir.glob("*.log"))
    payload = {
        "environment": _environment(repo_root),
        "log_dir": _repo_relative(log_dir, repo_root),
        "benchmarks": [_parse_log(path, log_dir, repo_root) for path in logs],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
