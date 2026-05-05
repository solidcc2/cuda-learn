from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


MODEL_CONFIGS = {
    "gpt2": {
        "model": "gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    },
    "qwen": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    },
}

DEFAULT_PROMPT_TEXT = (
    "Summarize flash attention decoder profiling in one sentence with exact wording."
)


@dataclass(frozen=True)
class VersionConfig:
    version: str
    toy_flash_attn_use: str
    attention_backend: str
    kv_cache_dtype: str
    cuda_version: str | None
    path_kind: str
    description: str


VERSION_CONFIGS: dict[str, VersionConfig] = {
    "baseline": VersionConfig(
        version="baseline",
        toy_flash_attn_use="reference",
        attention_backend="CUSTOM",
        kv_cache_dtype="bfloat16",
        cuda_version=None,
        path_kind="paged",
        description="Python reference paged path via custom backend.",
    ),
    "v5": VersionConfig(
        version="v5",
        toy_flash_attn_use="bf16",
        attention_backend="CUSTOM",
        kv_cache_dtype="bfloat16",
        cuda_version="v5",
        path_kind="paged",
        description="Custom CUDA paged path using v5 kernel.",
    ),
    "v6": VersionConfig(
        version="v6",
        toy_flash_attn_use="bf16",
        attention_backend="CUSTOM",
        kv_cache_dtype="bfloat16",
        cuda_version="v6",
        path_kind="paged",
        description="Custom CUDA paged path using v6 kernel.",
    ),
    "official": VersionConfig(
        version="official",
        toy_flash_attn_use="official",
        attention_backend="FLASH_ATTN",
        kv_cache_dtype="auto",
        cuda_version=None,
        path_kind="dense",
        description="Official dense FlashAttention decoder path.",
    ),
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def make_prompt(target_tokens: int, base_text: str = DEFAULT_PROMPT_TEXT) -> str:
    repeated = []
    while len(repeated) < target_tokens:
        repeated.extend(base_text.split())
    return " ".join(repeated[:target_tokens])


def make_prompts(batch_size: int, prompt_len: int) -> list[str]:
    prompts = []
    for i in range(batch_size):
        prompts.append(make_prompt(prompt_len, f"Prompt {i} about GPU attention benchmarking."))
    return prompts


def env_snapshot() -> dict[str, str]:
    keys = [
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOY_FLASH_ATTN_USE",
        "TOY_FLASH_ATTN_CUDA_VERSION",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


@contextmanager
def apply_version_env(version: str) -> Iterator[VersionConfig]:
    config = VERSION_CONFIGS[version]
    original_use = os.environ.get("TOY_FLASH_ATTN_USE")
    original_cuda = os.environ.get("TOY_FLASH_ATTN_CUDA_VERSION")
    os.environ["TOY_FLASH_ATTN_USE"] = config.toy_flash_attn_use
    if config.cuda_version is None:
        os.environ.pop("TOY_FLASH_ATTN_CUDA_VERSION", None)
    else:
        os.environ["TOY_FLASH_ATTN_CUDA_VERSION"] = config.cuda_version
    try:
        yield config
    finally:
        if original_use is None:
            os.environ.pop("TOY_FLASH_ATTN_USE", None)
        else:
            os.environ["TOY_FLASH_ATTN_USE"] = original_use
        if original_cuda is None:
            os.environ.pop("TOY_FLASH_ATTN_CUDA_VERSION", None)
        else:
            os.environ["TOY_FLASH_ATTN_CUDA_VERSION"] = original_cuda


def summarize_ms(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("samples_ms must not be empty")
    avg = sum(samples_ms) / len(samples_ms)
    variance = sum((sample - avg) ** 2 for sample in samples_ms) / len(samples_ms)
    return {
        "avg_ms": avg,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "std_ms": math.sqrt(variance),
    }


def system_environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at_s": time.time(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda"] = torch.version.cuda
        payload["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            payload["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - best effort metadata
        payload["torch_error"] = str(exc)

    try:
        import vllm

        payload["vllm"] = getattr(vllm, "__version__", None)
    except Exception as exc:  # pragma: no cover - best effort metadata
        payload["vllm_error"] = str(exc)

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload["git_commit"] = result.stdout.strip()
    except Exception as exc:  # pragma: no cover - best effort metadata
        payload["git_commit_error"] = str(exc)
    return payload


def version_metadata(version: str) -> dict[str, Any]:
    config = VERSION_CONFIGS[version]
    return asdict(config)
