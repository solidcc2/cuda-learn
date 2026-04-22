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

DEFAULT_CASES=(
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

usage() {
  cat <<'EOF'
Usage:
  run_perf_eval.sh [case ...]

Case format:
  version:model:batch:max_tokens

Known versions:
  baseline   -> TOY_FLASH_ATTN_USE=reference
  v3         -> TOY_FLASH_ATTN_USE=bf16, TOY_FLASH_ATTN_CUDA_VERSION=v3
  v4         -> TOY_FLASH_ATTN_USE=bf16, TOY_FLASH_ATTN_CUDA_VERSION=v4
  v4_fp32    -> TOY_FLASH_ATTN_USE=fp32, TOY_FLASH_ATTN_CUDA_VERSION=v4
  v5         -> TOY_FLASH_ATTN_USE=bf16, TOY_FLASH_ATTN_CUDA_VERSION=v5
  official   -> TOY_FLASH_ATTN_USE=official

Examples:
  bash flash_attention_backend/analysis/run_perf_eval.sh
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

  case "${version}" in
    baseline)
      env \
        -u TOY_FLASH_ATTN_CUDA_VERSION \
        TOY_FLASH_ATTN_USE=reference \
        "${PYTHON_BIN}" "${RUNNER}" -m "${model}" -b "${batch}" -t "${max_tokens}" \
        >"${log_path}" 2>&1
      ;;
    v3)
      env \
        TOY_FLASH_ATTN_USE=bf16 \
        TOY_FLASH_ATTN_CUDA_VERSION=v3 \
        "${PYTHON_BIN}" "${RUNNER}" -m "${model}" -b "${batch}" -t "${max_tokens}" \
        >"${log_path}" 2>&1
      ;;
    v4)
      env \
        TOY_FLASH_ATTN_USE=bf16 \
        TOY_FLASH_ATTN_CUDA_VERSION=v4 \
        "${PYTHON_BIN}" "${RUNNER}" -m "${model}" -b "${batch}" -t "${max_tokens}" \
        >"${log_path}" 2>&1
      ;;
    v4_fp32)
      env \
        TOY_FLASH_ATTN_USE=fp32 \
        TOY_FLASH_ATTN_CUDA_VERSION=v4 \
        "${PYTHON_BIN}" "${RUNNER}" -m "${model}" -b "${batch}" -t "${max_tokens}" \
        >"${log_path}" 2>&1
      ;;
    v5)
      env \
        TOY_FLASH_ATTN_USE=bf16 \
        TOY_FLASH_ATTN_CUDA_VERSION=v5 \
        "${PYTHON_BIN}" "${RUNNER}" -m "${model}" -b "${batch}" -t "${max_tokens}" \
        >"${log_path}" 2>&1
      ;;
    official)
      env \
        -u TOY_FLASH_ATTN_CUDA_VERSION \
        TOY_FLASH_ATTN_USE=official \
        "${PYTHON_BIN}" "${RUNNER}" -m "${model}" -b "${batch}" -t "${max_tokens}" \
        >"${log_path}" 2>&1
      ;;
    *)
      echo "unknown version: ${version}" >&2
      return 2
      ;;
  esac
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

  for case_spec in "${cases[@]}"; do
    IFS=":" read -r version model batch max_tokens extra <<<"${case_spec}"
    if [[ -z "${version}" || -z "${model}" || -z "${batch}" || -z "${max_tokens}" || -n "${extra:-}" ]]; then
      echo "invalid case spec: ${case_spec}" >&2
      usage >&2
      exit 2
    fi
    run_case "${version}" "${model}" "${batch}" "${max_tokens}"
  done

  "${PYTHON_BIN}" "${PARSER}" \
    --log-dir "${LOG_DIR}" \
    --output "${OUT_JSON}" \
    --repo-root "${REPO_ROOT}"

  echo "[JSON] ${OUT_JSON}"
}

main "$@"
