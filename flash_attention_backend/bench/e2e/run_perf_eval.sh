#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER="${BACKEND_ROOT}/bench/e2e/run_vllm_e2e.py"
PARSER="${BACKEND_ROOT}/bench/e2e/parse_perf_eval_logs.py"
JSON_DIR="${PERF_EVAL_JSON_DIR:-${SCRIPT_DIR}/perf_json}"
LOG_DIR="${PERF_EVAL_LOG_DIR:-${SCRIPT_DIR}/perf_logs}"
OUT_JSON="${PERF_EVAL_OUTPUT:-${SCRIPT_DIR}/perf_eval_results.json}"
WARMUP="${PERF_EVAL_WARMUP:-1}"
REPEAT="${PERF_EVAL_REPEAT:-3}"

usage() {
  cat <<'EOF'
Usage:
  run_perf_eval.sh [suite|case ...]

Suites:
  smoke
  report
  stress
  all

Cases:
  qwen_b1_t128
  qwen_b4_t128
  qwen_b1_t512
  qwen_b4_t512
  qwen_b1_t2048
  gpt2_b1_t128

Environment:
  PYTHON_BIN
  PERF_EVAL_JSON_DIR
  PERF_EVAL_LOG_DIR
  PERF_EVAL_OUTPUT
  PERF_EVAL_WARMUP
  PERF_EVAL_REPEAT
EOF
}

expand_cases() {
  "${PYTHON_BIN}" - "$@" <<'PY'
from flash_attention_backend.bench.e2e.cases_e2e import CASES, SUITES
import sys

args = sys.argv[1:]
if not args:
    args = ["smoke"]

expanded = []
for value in args:
    if value in SUITES:
        expanded.extend(SUITES[value])
    elif value in CASES:
        expanded.append(value)
    else:
        raise SystemExit(f"unknown suite or case: {value}")
print("\n".join(expanded))
PY
}

run_case() {
  local version="$1"
  local case_name="$2"
  local base_name="${version}_${case_name}"
  local json_path="${JSON_DIR}/${base_name}.json"
  local log_path="${LOG_DIR}/${base_name}.log"

  mkdir -p "${JSON_DIR}" "${LOG_DIR}"
  echo "[RUN] version=${version} case=${case_name}"
  "${PYTHON_BIN}" "${RUNNER}" \
    --version "${version}" \
    --case "${case_name}" \
    --warmup "${WARMUP}" \
    --repeat "${REPEAT}" \
    --output-json "${json_path}" \
    >"${log_path}" 2>&1
  echo "[DONE] version=${version} case=${case_name} json=${json_path}"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  mapfile -t cases < <(expand_cases "$@")
  local versions=("baseline" "v5" "v6" "official")
  for case_name in "${cases[@]}"; do
    for version in "${versions[@]}"; do
      run_case "${version}" "${case_name}"
    done
  done

  "${PYTHON_BIN}" "${PARSER}" --json-dir "${JSON_DIR}" --output "${OUT_JSON}"
  echo "[JSON] ${OUT_JSON}"
}

main "$@"
