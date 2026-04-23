#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BACKEND_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER="${BACKEND_ROOT}/test_self_flash_attn_backend.py"
PARSER="${SCRIPT_DIR}/parse_perf_eval_logs.py"
LOG_DIR="${PERF_EVAL_LOG_DIR:-${SCRIPT_DIR}/perf_logs}"
OUT_JSON="${PERF_EVAL_OUTPUT:-${SCRIPT_DIR}/perf_eval_results.json}"

SMOKE_CASES=(
  "baseline:qwen:1:128"
  "v4:qwen:1:128"
  "v4:qwen:4:128"
  "v5:qwen:1:128"
  "v5:qwen:4:128"
  "official:qwen:1:128"
  "official:qwen:4:128"
  "v4:gpt2:1:128"
  "v5:gpt2:1:128"
  "official:gpt2:1:128"
)

REPORT_CASES=(
  "v4:qwen:1:512"
  "v4:qwen:4:512"
  "v5:qwen:1:512"
  "v5:qwen:4:512"
  "official:qwen:1:512"
  "official:qwen:4:512"
)

STRESS_CASES=(
  "v4:qwen:1:2048"
  "v5:qwen:1:2048"
  "official:qwen:1:2048"
)

DEFAULT_CASES=("${SMOKE_CASES[@]}")

sanitize_log() {
  local log_path="$1"
  "${PYTHON_BIN}" - "${log_path}" "${REPO_ROOT}" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2]).resolve()
text = log_path.read_text(errors="replace")

text = text.replace(str(repo_root), ".")
text = text.replace(str(Path.home()), "<home>")

# Hide remaining absolute local paths from third-party tooling output while
# preserving repo-relative paths and non-path tokens.
abs_path_re = re.compile(r"(?<![\w.~-])/(?:[^\s:'\",]+/?)+")
text = abs_path_re.sub("<abs-path>", text)

log_path.write_text(text)
PY
}

usage() {
  cat <<'EOF'
Usage:
  run_perf_eval.sh [case ...]
  run_perf_eval.sh smoke
  run_perf_eval.sh report
  run_perf_eval.sh stress
  run_perf_eval.sh all

Case format:
  version:model:batch:max_tokens

Suites:
  smoke   -> short 128-token coverage, includes baseline/v4/v5/official
  report  -> 512-token Qwen coverage for v4/v5/official
  stress  -> 2048-token single-batch Qwen coverage for v4/v5/official
  all     -> smoke + report + stress

Note:
  test_self_flash_attn_backend.py currently keeps LLM max_model_len fixed at 512.
  Long-token suites intentionally do not change max_model_len.

Known versions:
  baseline   -> TOY_FLASH_ATTN_USE=reference
  v3         -> TOY_FLASH_ATTN_USE=bf16, TOY_FLASH_ATTN_CUDA_VERSION=v3
  v4         -> TOY_FLASH_ATTN_USE=bf16, TOY_FLASH_ATTN_CUDA_VERSION=v4
  v4_fp32    -> TOY_FLASH_ATTN_USE=fp32, TOY_FLASH_ATTN_CUDA_VERSION=v4
  v5         -> TOY_FLASH_ATTN_USE=bf16, TOY_FLASH_ATTN_CUDA_VERSION=v5
  official   -> TOY_FLASH_ATTN_USE=official

Examples:
  bash flash_attention_backend/analysis/run_perf_eval.sh
  bash flash_attention_backend/analysis/run_perf_eval.sh report
  bash flash_attention_backend/analysis/run_perf_eval.sh v5:qwen:1:128 official:qwen:1:128

Environment:
  PYTHON_BIN             Python executable. Default: python
  PERF_EVAL_LOG_DIR      Log output directory.
  PERF_EVAL_OUTPUT       JSON output path.
EOF
}

run_case() {
  local version="$1"
  local model="$2"
  local batch="$3"
  local max_tokens="$4"
  local log_path="${LOG_DIR}/${version}_${model}_b${batch}_t${max_tokens}.log"

  mkdir -p "${LOG_DIR}"
  echo "[RUN] version=${version} model=${model} batch=${batch} max_tokens=${max_tokens}"
  echo "[LOG] ${log_path}"

  local status=0
  case "${version}" in
    baseline)
      (
        unset TOY_FLASH_ATTN_CUDA_VERSION
        TOY_FLASH_ATTN_USE=reference \
        "${PYTHON_BIN}" "${RUNNER}" -m "${model}" -b "${batch}" -t "${max_tokens}" \
      ) >"${log_path}" 2>&1 || status=$?
      ;;
    v3)
      env \
        TOY_FLASH_ATTN_USE=bf16 \
        TOY_FLASH_ATTN_CUDA_VERSION=v3 \
        "${PYTHON_BIN}" "${RUNNER}" -m "${model}" -b "${batch}" -t "${max_tokens}" \
        >"${log_path}" 2>&1 || status=$?
      ;;
    v4)
      env \
        TOY_FLASH_ATTN_USE=bf16 \
        TOY_FLASH_ATTN_CUDA_VERSION=v4 \
        "${PYTHON_BIN}" "${RUNNER}" -m "${model}" -b "${batch}" -t "${max_tokens}" \
        >"${log_path}" 2>&1 || status=$?
      ;;
    v4_fp32)
      env \
        TOY_FLASH_ATTN_USE=fp32 \
        TOY_FLASH_ATTN_CUDA_VERSION=v4 \
        "${PYTHON_BIN}" "${RUNNER}" -m "${model}" -b "${batch}" -t "${max_tokens}" \
        >"${log_path}" 2>&1 || status=$?
      ;;
    v5)
      env \
        TOY_FLASH_ATTN_USE=bf16 \
        TOY_FLASH_ATTN_CUDA_VERSION=v5 \
        "${PYTHON_BIN}" "${RUNNER}" -m "${model}" -b "${batch}" -t "${max_tokens}" \
        >"${log_path}" 2>&1 || status=$?
      ;;
    official)
      (
        unset TOY_FLASH_ATTN_CUDA_VERSION
        TOY_FLASH_ATTN_USE=official \
        "${PYTHON_BIN}" "${RUNNER}" -m "${model}" -b "${batch}" -t "${max_tokens}" \
      ) >"${log_path}" 2>&1 || status=$?
      ;;
    *)
      echo "unknown version: ${version}" >&2
      return 2
      ;;
  esac
  sanitize_log "${log_path}"
  return "${status}"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  local cases=("$@")
  if [[ ${#cases[@]} -eq 0 ]]; then
    cases=("${DEFAULT_CASES[@]}")
  fi

  local expanded_cases=()
  for case_spec in "${cases[@]}"; do
    case "${case_spec}" in
      smoke)
        expanded_cases+=("${SMOKE_CASES[@]}")
        ;;
      report)
        expanded_cases+=("${REPORT_CASES[@]}")
        ;;
      stress)
        expanded_cases+=("${STRESS_CASES[@]}")
        ;;
      all)
        expanded_cases+=("${SMOKE_CASES[@]}" "${REPORT_CASES[@]}" "${STRESS_CASES[@]}")
        ;;
      *)
        expanded_cases+=("${case_spec}")
        ;;
    esac
  done

  local failed=0
  for case_spec in "${expanded_cases[@]}"; do
    IFS=":" read -r version model batch max_tokens extra <<<"${case_spec}"
    if [[ -z "${version}" || -z "${model}" || -z "${batch}" || -z "${max_tokens}" || -n "${extra:-}" ]]; then
      echo "invalid case spec: ${case_spec}" >&2
      usage >&2
      exit 2
    fi
    if ! run_case "${version}" "${model}" "${batch}" "${max_tokens}"; then
      echo "[FAIL] ${case_spec}" >&2
      failed=1
    fi
  done

  "${PYTHON_BIN}" "${PARSER}" \
    --log-dir "${LOG_DIR}" \
    --output "${OUT_JSON}" \
    --repo-root "${REPO_ROOT}"

  echo "[JSON] ${OUT_JSON}"
  return "${failed}"
}

main "$@"
