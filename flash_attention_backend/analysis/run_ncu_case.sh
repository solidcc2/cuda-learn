#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
NCU_BIN="${NCU_BIN:-ncu}"
NCU_SET="full"
CASE_NAME=""
VERSIONS="v6,official"
OUTPUT_DIR=""

usage() {
  cat <<'EOF'
Usage:
  run_ncu_case.sh --case <case> [--versions v6,official] [--output-dir <dir>] [--set full]

Supported cases:
  qwen_like_b1_s128_h64
  qwen_like_b4_s512_h64
  qwen_like_b1_s2048_h64

Supported versions:
  v6
  official

Environment:
  PYTHON_BIN
  NCU_BIN
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --case)
        CASE_NAME="$2"
        shift 2
        ;;
      --versions)
        VERSIONS="$2"
        shift 2
        ;;
      --output-dir)
        OUTPUT_DIR="$2"
        shift 2
        ;;
      --set)
        NCU_SET="$2"
        shift 2
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

validate_case() {
  case "${CASE_NAME}" in
    qwen_like_b1_s128_h64|qwen_like_b4_s512_h64|qwen_like_b1_s2048_h64)
      ;;
    *)
      echo "unsupported ncu case: ${CASE_NAME}" >&2
      exit 2
      ;;
  esac
}

validate_versions() {
  IFS=',' read -r -a version_list <<<"${VERSIONS}"
  for version in "${version_list[@]}"; do
    case "${version}" in
      v6|official)
        ;;
      *)
        echo "unsupported ncu version: ${version}" >&2
        exit 2
        ;;
    esac
  done
}

run_version() {
  local version="$1"
  local rep_stem="${OUTPUT_DIR}/${version}"
  local rep_path="${rep_stem}.ncu-rep"
  local log_path="${OUTPUT_DIR}/${version}.log"
  local csv_path="${OUTPUT_DIR}/${version}_raw.csv"
  local meta_path="${OUTPUT_DIR}/input_meta_${version}.json"

  echo "[NCU] version=${version} case=${CASE_NAME} set=${NCU_SET}"
  "${NCU_BIN}" \
    --set "${NCU_SET}" \
    --target-processes all \
    --force-overwrite \
    --export "${rep_stem}" \
    --csv \
    --page raw \
    "${PYTHON_BIN}" "${BACKEND_ROOT}/bench/op/profile_attention_op.py" \
      --version "${version}" \
      --case "${CASE_NAME}" \
      --warmup 1 \
      --iters 1 \
      --dump-input-meta "${meta_path}" \
      >"${csv_path}" 2>"${log_path}"

  if [[ ! -f "${rep_path}" ]]; then
    echo "expected ncu report not found: ${rep_path}" >&2
    exit 1
  fi
  echo "[NCU] rep=${rep_path}"
  echo "[NCU] raw_csv=${csv_path}"
  echo "[NCU] log=${log_path}"
  echo "[NCU] input_meta=${meta_path}"
}

main() {
  parse_args "$@"
  if [[ -z "${CASE_NAME}" ]]; then
    echo "--case is required" >&2
    usage >&2
    exit 2
  fi
  validate_case
  validate_versions

  if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${SCRIPT_DIR}/artifacts/ncu/${CASE_NAME}"
  fi
  mkdir -p "${OUTPUT_DIR}"

  if ! command -v "${NCU_BIN}" >/dev/null 2>&1; then
    echo "ncu executable not found: ${NCU_BIN}" >&2
    exit 127
  fi

  IFS=',' read -r -a version_list <<<"${VERSIONS}"
  for version in "${version_list[@]}"; do
    run_version "${version}"
  done
}

main "$@"
