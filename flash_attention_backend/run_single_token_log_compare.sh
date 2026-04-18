#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/linf/code/cuda/flash_attention_backend"
PYTHON_BIN="${PYTHON_BIN:-/home/linf/code/cuda/vllm_env/bin/python}"
RUNNER="${ROOT}/test_single_token_log_compare.py"
COMPARE="${ROOT}/compare_single_token_logs.py"

FP32_LOG="${ROOT}/fp32.log"
BF16_LOG="${ROOT}/bf16.log"
DIFF_LOG="${ROOT}/diff.log"

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/toy_flash_attn_torch_extensions}"
# mkdir -p "${TORCH_EXTENSIONS_DIR}"

# if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
#   TORCH_CUDA_ARCH_LIST="$(${PYTHON_BIN} -c 'import torch; m, n = torch.cuda.get_device_capability(); print(f"{m}.{n}")')"
#   export TORCH_CUDA_ARCH_LIST
# fi

"${PYTHON_BIN}" "${RUNNER}" --impl fp32 > "${FP32_LOG}" 2>&1
"${PYTHON_BIN}" "${RUNNER}" --impl bf16 > "${BF16_LOG}" 2>&1
"${PYTHON_BIN}" "${COMPARE}" --fp32-log "${FP32_LOG}" --bf16-log "${BF16_LOG}" > "${DIFF_LOG}" 2>&1

echo "wrote ${FP32_LOG}"
echo "wrote ${BF16_LOG}"
echo "wrote ${DIFF_LOG}"
