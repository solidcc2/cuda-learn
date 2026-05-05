#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="light"
SKIP_E2E=0
SKIP_OP=0
SKIP_CORRECTNESS=0
SKIP_VERSION_SUMMARY=0
SKIP_REPORT_INPUTS=0

E2E_RUNNER="${BACKEND_ROOT}/bench/e2e/collect_e2e.sh"
OP_RUNNER="${BACKEND_ROOT}/bench/op/bench_attention_op.py"
CORRECTNESS_RUNNER="${SCRIPT_DIR}/collect_correctness_metrics.py"
VERSION_SUMMARY_RUNNER="${SCRIPT_DIR}/summarize_version_optimizations.py"
REPORT_INPUT_RUNNER="${SCRIPT_DIR}/build_report_inputs.py"

ARTIFACT_ROOT="${PERF_EVAL_ARTIFACT_DIR:-${SCRIPT_DIR}/artifacts}"
E2E_DIR="${ARTIFACT_ROOT}/e2e"
OP_DIR="${ARTIFACT_ROOT}/op"
CORRECTNESS_DIR="${ARTIFACT_ROOT}/correctness"
REPORT_DIR="${ARTIFACT_ROOT}/report"

E2E_JSON_DIR="${E2E_DIR}/perf_json"
E2E_LOG_DIR="${E2E_DIR}/perf_logs"
E2E_OUTPUT="${E2E_DIR}/perf_eval_results.json"
LEGACY_E2E_OUTPUT="${SCRIPT_DIR}/perf_eval_results.json"

OP_CASE_DIR="${OP_DIR}/case_json"
OP_LOG_DIR="${OP_DIR}/logs"
OP_RESULTS="${OP_DIR}/op_results.json"
CORRECTNESS_LOG_DIR="${CORRECTNESS_DIR}/logs"
CORRECTNESS_SUMMARY="${CORRECTNESS_DIR}/correctness_summary.json"

VERSION_SUMMARY_OUTPUT="${REPORT_DIR}/version_optimizations.json"
REPORT_INPUT_OUTPUT="${REPORT_DIR}/report_inputs.json"

WARMUP="${PERF_EVAL_WARMUP:-1}"
REPEAT="${PERF_EVAL_REPEAT:-3}"
OP_WARMUP="${PERF_EVAL_OP_WARMUP:-1}"
OP_ITERS="${PERF_EVAL_OP_ITERS:-3}"

usage() {
  cat <<'EOF'
Usage:
  run_perf_eval.sh [light|full] [--skip-e2e] [--skip-op] [--skip-correctness]
                   [--skip-version-summary] [--skip-report-inputs]

Modes:
  light  Default. Collects smoke-level e2e, minimal op cases, correctness
         metrics, and version implementation summary.
  full   Collects all e2e cases, all standard op cases, correctness metrics,
         and version implementation summary.

Skip flags:
  --skip-e2e
  --skip-op
  --skip-correctness
  --skip-version-summary
  --skip-report-inputs

Environment:
  PYTHON_BIN
  PERF_EVAL_ARTIFACT_DIR
  PERF_EVAL_WARMUP
  PERF_EVAL_REPEAT
  PERF_EVAL_OP_WARMUP
  PERF_EVAL_OP_ITERS
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      light|full)
        MODE="$1"
        shift
        ;;
      --skip-e2e)
        SKIP_E2E=1
        shift
        ;;
      --skip-op)
        SKIP_OP=1
        shift
        ;;
      --skip-correctness)
        SKIP_CORRECTNESS=1
        shift
        ;;
      --skip-version-summary)
        SKIP_VERSION_SUMMARY=1
        shift
        ;;
      --skip-report-inputs)
        SKIP_REPORT_INPUTS=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
}

timed_section() {
  local label="$1"
  shift
  local started
  local ended
  local elapsed_s
  started="$(date +%s)"
  "$@"
  ended="$(date +%s)"
  elapsed_s="$((ended - started))"
  echo "[TIME] ${label} elapsed_s=${elapsed_s}"
}

ensure_dirs() {
  mkdir -p \
    "${E2E_JSON_DIR}" \
    "${E2E_LOG_DIR}" \
    "${OP_CASE_DIR}" \
    "${OP_LOG_DIR}" \
    "${CORRECTNESS_LOG_DIR}" \
    "${REPORT_DIR}"
}

run_e2e() {
  local suite="$1"
  echo "[E2E] suite=${suite}"
  timed_section "e2e suite=${suite}" env \
    PERF_EVAL_JSON_DIR="${E2E_JSON_DIR}" \
    PERF_EVAL_LOG_DIR="${E2E_LOG_DIR}" \
    PERF_EVAL_OUTPUT="${E2E_OUTPUT}" \
    PERF_EVAL_WARMUP="${WARMUP}" \
    PERF_EVAL_REPEAT="${REPEAT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${E2E_RUNNER}" "${suite}"
  cp "${E2E_OUTPUT}" "${LEGACY_E2E_OUTPUT}"
  echo "[E2E] output=${E2E_OUTPUT}"
}

run_op_case() {
  local version="$1"
  local case_name="$2"
  local json_path="${OP_CASE_DIR}/${version}_${case_name}.json"
  local log_path="${OP_LOG_DIR}/${version}_${case_name}.log"
  echo "[OP] version=${version} case=${case_name}"
  timed_section "op version=${version} case=${case_name}" \
    "${PYTHON_BIN}" "${OP_RUNNER}" \
      --version "${version}" \
      --case "${case_name}" \
      --warmup "${OP_WARMUP}" \
      --iters "${OP_ITERS}" \
      --output-json "${json_path}" \
      >"${log_path}" 2>&1
}

run_op() {
  local mode="$1"
  local -a versions=("baseline" "v5" "v6" "official")
  local -a cases
  if [[ "${mode}" == "full" ]]; then
    cases=(
      "qwen_like_b1_s128_h64"
      "qwen_like_b4_s128_h64"
      "qwen_like_b1_s512_h64"
      "qwen_like_b4_s512_h64"
      "qwen_like_b1_s2048_h64"
      "gpt2_like_b1_s128_h64"
      "gqa_case_b1_s128"
      "gqa_case_b4_s512"
    )
  else
    cases=(
      "qwen_like_b1_s128_h64"
      "gpt2_like_b1_s128_h64"
      "gqa_case_b1_s128"
    )
  fi

  for version in "${versions[@]}"; do
    for case_name in "${cases[@]}"; do
      run_op_case "${version}" "${case_name}"
    done
  done
}

run_correctness() {
  local label="correctness_metrics"
  local target="collect_mode"
  local log_path="${CORRECTNESS_LOG_DIR}/${label}.log"
  echo "[CORRECTNESS] target=${target}"
  timed_section "correctness target=${label}" \
    "${PYTHON_BIN}" "${CORRECTNESS_RUNNER}" \
      --output-json "${CORRECTNESS_SUMMARY}" \
      >"${log_path}" 2>&1
}

run_version_summary() {
  echo "[SUMMARY] version implementations"
  timed_section "version_summary" \
    "${PYTHON_BIN}" "${VERSION_SUMMARY_RUNNER}" \
      --output "${VERSION_SUMMARY_OUTPUT}"
}

build_report_inputs() {
  echo "[REPORT] building report inputs"
  timed_section "report_inputs" \
    "${PYTHON_BIN}" "${REPORT_INPUT_RUNNER}" \
      --e2e-summary "${E2E_OUTPUT}" \
      --op-dir "${OP_CASE_DIR}" \
      --correctness-input "${CORRECTNESS_SUMMARY}" \
      --correctness-output "${CORRECTNESS_SUMMARY}" \
      --version-summary "${VERSION_SUMMARY_OUTPUT}" \
      --op-output "${OP_RESULTS}" \
      --output "${REPORT_INPUT_OUTPUT}"
  echo "[REPORT] output=${REPORT_INPUT_OUTPUT}"
}

main() {
  parse_args "$@"
  if [[ "${MODE}" != "light" && "${MODE}" != "full" ]]; then
    echo "unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
  fi

  ensure_dirs
  if [[ "${SKIP_E2E}" -eq 0 ]]; then
    if [[ "${MODE}" == "full" ]]; then
      run_e2e "all"
    else
      run_e2e "smoke"
    fi
  else
    echo "[SKIP] e2e"
  fi
  if [[ "${SKIP_OP}" -eq 0 ]]; then
    run_op "${MODE}"
  else
    echo "[SKIP] op"
  fi
  if [[ "${SKIP_CORRECTNESS}" -eq 0 ]]; then
    run_correctness
  else
    echo "[SKIP] correctness"
  fi
  if [[ "${SKIP_VERSION_SUMMARY}" -eq 0 ]]; then
    run_version_summary
  else
    echo "[SKIP] version_summary"
  fi
  if [[ "${SKIP_REPORT_INPUTS}" -eq 0 ]]; then
    build_report_inputs
  else
    echo "[SKIP] report_inputs"
  fi
}

main "$@"
