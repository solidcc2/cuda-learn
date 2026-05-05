from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from vllm import LLM, SamplingParams
from vllm.utils.torch_utils import get_kv_cache_torch_dtype
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from flash_attention_backend.bench.common import (
    MODEL_CONFIGS,
    VERSION_CONFIGS,
    apply_version_env,
    env_snapshot,
    make_prompts,
    summarize_ms,
    system_environment,
    version_metadata,
    write_json,
)
from flash_attention_backend.bench.e2e.cases_e2e import CASES, get_case

register_backend(
    AttentionBackendEnum.CUSTOM,
    "flash_attention_backend.toy_flash_attn.ToyFlashAttentionBackend",
)


def _safe_path(path: Path) -> str:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return resolved.name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=sorted(VERSION_CONFIGS), required=True)
    parser.add_argument("--case", choices=sorted(CASES), default=None)
    parser.add_argument("-m", "--model", choices=sorted(MODEL_CONFIGS), default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=None)
    parser.add_argument("--prompt-len", type=int, default=None)
    parser.add_argument("-t", "--max-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.case is None and None in {
        args.model,
        args.batch_size,
        args.prompt_len,
        args.max_tokens,
        args.max_model_len,
    }:
        parser.error("manual mode requires --model/--batch-size/--prompt-len/--max-tokens/--max-model-len")
    return args


def _resolved_case(args: argparse.Namespace) -> dict[str, Any]:
    if args.case is not None:
        case = get_case(args.case)
        return {
            "case_name": case.name,
            "model": case.model,
            "batch_size": case.batch_size,
            "prompt_len": case.prompt_len,
            "max_tokens": case.max_tokens,
            "max_model_len": case.max_model_len,
        }
    return {
        "case_name": "manual",
        "model": args.model,
        "batch_size": args.batch_size,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
    }


def _configure_download_env(allow_download: bool) -> None:
    if allow_download:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _run_generate(llm: LLM, prompts: list[str], sampling_params: SamplingParams) -> tuple[list[Any], float]:
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return outputs, elapsed_ms


def _require_cuda_runtime() -> None:
    if torch.cuda.is_available():
        return
    raise SystemExit(
        "CUDA is not available in this environment. "
        "Run this benchmark on a machine where the vLLM runtime can see a CUDA device."
    )


def main() -> None:
    args = _parse_args()
    case = _resolved_case(args)
    _configure_download_env(args.allow_download)
    _require_cuda_runtime()
    model_config = MODEL_CONFIGS[case["model"]]

    with apply_version_env(args.version) as version_config:
        llm = LLM(
            model=model_config["model"],
            revision=model_config["revision"],
            tokenizer_revision=model_config["revision"],
            max_model_len=case["max_model_len"],
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=case["batch_size"],
            attention_backend=version_config.attention_backend,
            kv_cache_dtype=version_config.kv_cache_dtype,
        )
        engine = llm.llm_engine
        vconfig = engine.vllm_config
        prompts = make_prompts(case["batch_size"], case["prompt_len"])
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=0.9,
            max_tokens=case["max_tokens"],
        )

        for _ in range(args.warmup):
            _run_generate(llm, prompts, sampling_params)

        timings_ms: list[float] = []
        output_lengths: list[int] = []
        finish_reasons: list[str | None] = []
        prompt_count = 0

        for _ in range(args.repeat):
            outputs, elapsed_ms = _run_generate(llm, prompts, sampling_params)
            timings_ms.append(elapsed_ms)
            prompt_count = len(outputs)
            for output in outputs:
                generated = output.outputs[0]
                output_lengths.append(len(generated.token_ids))
                finish_reasons.append(generated.finish_reason)

        metrics = summarize_ms(timings_ms)
        avg_output_tokens = sum(output_lengths) / len(output_lengths) if output_lengths else 0.0
        total_output_tokens = sum(output_lengths)
        total_elapsed_s = sum(timings_ms) / 1000.0

        payload = {
            "kind": "flash_attention_backend.e2e_case",
            "case_name": case["case_name"],
            "model_arg": case["model"],
            "model_name": model_config["model"],
            "model_revision": model_config["revision"],
            "version": args.version,
            "path_kind": version_config.path_kind,
            "batch_size": case["batch_size"],
            "prompt_len": case["prompt_len"],
            "max_tokens": case["max_tokens"],
            "max_model_len": case["max_model_len"],
            "warmup": args.warmup,
            "repeat": args.repeat,
            "prompt_count": prompt_count,
            "avg_output_tokens_per_prompt": avg_output_tokens,
            "total_output_tokens": total_output_tokens,
            "tokens_per_s": (total_output_tokens / total_elapsed_s) if total_elapsed_s > 0 else None,
            "latency_ms": metrics,
            "finish_reasons": finish_reasons,
            "config": {
                "attention_backend": version_config.attention_backend,
                "kv_cache_dtype_arg": version_config.kv_cache_dtype,
                "cache_config.cache_dtype": str(vconfig.cache_config.cache_dtype),
                "model_config.dtype": str(vconfig.model_config.dtype),
                "resolved_kv_torch_dtype": str(
                    get_kv_cache_torch_dtype(
                        vconfig.cache_config.cache_dtype,
                        vconfig.model_config.dtype,
                    )
                ),
                "gpu_memory_utilization": args.gpu_memory_utilization,
            },
            "version_metadata": version_metadata(args.version),
            "environment": system_environment(),
            "env_vars": env_snapshot(),
        }

        print("case_name               =", payload["case_name"])
        print("model                   =", payload["model_name"])
        print("version                 =", payload["version"])
        print("path_kind               =", payload["path_kind"])
        print("batch_size              =", payload["batch_size"])
        print("prompt_len              =", payload["prompt_len"])
        print("max_tokens              =", payload["max_tokens"])
        print("max_model_len           =", payload["max_model_len"])
        print("warmup                  =", payload["warmup"])
        print("repeat                  =", payload["repeat"])
        print("tokens_per_s            =", payload["tokens_per_s"])
        print("avg_ms                  =", payload["latency_ms"]["avg_ms"])

        if args.output_json is not None:
            write_json(args.output_json, payload)
            print("output_json             =", _safe_path(args.output_json))

        del engine
        del llm


if __name__ == "__main__":
    main()
