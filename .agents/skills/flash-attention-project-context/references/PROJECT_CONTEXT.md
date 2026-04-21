# flash_attention_backend Project Context Map

This file is private working state for the `flash-attention-project-context` skill.

It is a navigation map, not a source of truth. Before giving conclusions or making edits, verify task-relevant facts against current code, logs, or JSON.

## Contents

- [State Metadata](#state-metadata)
- [Repository Layout](#repository-layout)
- [Environment Variables](#environment-variables)
- [Backend Selection Dependency](#backend-selection-dependency)
- [CUDA Version Loading Dependency](#cuda-version-loading-dependency)
- [v4 Kernel Dependency](#v4-kernel-dependency)
- [GQA Dependency](#gqa-dependency)
- [Benchmark And Report Dependency](#benchmark-and-report-dependency)
- [Quick Verification Patterns](#quick-verification-patterns)

## State Metadata

| Field | Value |
| --- | --- |
| last_verified_commit | `5a5b3a68b52dca7adead05c22e173dae3967438c` |
| refresh_scope | Created from current project files; focused on runtime dispatch, v4 kernel, benchmark, report workflow, and project-context maintenance. |

When current `git rev-parse HEAD` differs from `last_verified_commit`, refresh this map before relying on it.

Refresh mode:

- Incremental refresh: use when only task-local files changed.
- Full refresh: use when switching branches broadly or when dispatch, benchmark, report, CUDA launch constraints, or GQA may have changed.

## Repository Layout

| Area | Role | Key files |
| --- | --- | --- |
| toy attention package | Python wrapper, CUDA extension loading, reference implementation | `flash_attention_backend/toy_flash_attn/flash_attention_func.py` |
| v4 CUDA kernel | Current toy CUDA kernel | `flash_attention_backend/toy_flash_attn/v4/flash_attn_func.cu`, `flash_attention_backend/toy_flash_attn/v4/helper.h` |
| v3 CUDA kernel | Older CUDA implementation | `flash_attention_backend/toy_flash_attn/flash_attn_func_v3.cu` |
| vLLM smoke runner | End-to-end generation entrypoint | `flash_attention_backend/test_self_flash_attn_backend.py` |
| benchmark analysis | Shell orchestration, log parser, JSON output | `flash_attention_backend/analysis/run_perf_eval.sh`, `flash_attention_backend/analysis/parse_perf_eval_logs.py` |
| performance report | Human-facing benchmark report | `flash_attention_backend/docs/PERFORMANCE_EVAL.md` |
| performance report skill | Report-update workflow | `.agents/skills/flash-attention-performance-report/SKILL.md` |

## Environment Variables

Verify these in code before relying on them.

| Variable | Current role | Primary files to check |
| --- | --- | --- |
| `TOY_FLASH_ATTN_USE` | Selects reference / bf16 / fp32 path in wrapper; selects official / custom backend in smoke runner; used by benchmark shell cases | `toy_flash_attn/flash_attention_func.py`, `test_self_flash_attn_backend.py`, `analysis/run_perf_eval.sh` |
| `TOY_FLASH_ATTN_CUDA_VERSION` | Selects v3 or v4 extension at import time | `toy_flash_attn/flash_attention_func.py`, `analysis/run_perf_eval.sh`, `toy_flash_attn/README.md` |
| `TOY_FLASH_ATTN_DEBUG` | Prints wrapper input tensor metadata | `toy_flash_attn/flash_attention_func.py` |
| `TOY_FLASH_ATTN_PRINT_DTYPE` | Prints Python reference dtype trace | `toy_flash_attn/flash_attention_func.py` |
| `TOY_FLASH_ATTN_DUMP_DIR` | Dumps attention context for replay/debug | `toy_flash_attn/flash_attention_func.py`, `toy_flash_attn/README.md` |
| `TOY_FLASH_ATTN_REPLAY_DUMP` | Replay-test input path | `toy_flash_attn/README.md`, tests |
| `PERF_EVAL_LOG_DIR` | Overrides benchmark log directory | `analysis/run_perf_eval.sh` |
| `PERF_EVAL_OUTPUT` | Overrides benchmark JSON output path | `analysis/run_perf_eval.sh` |
| `PYTHON_BIN` | Selects Python executable for benchmark shell | `analysis/run_perf_eval.sh` |

## Backend Selection Dependency

When changing backend selection, check all of:

- `flash_attention_backend/toy_flash_attn/flash_attention_func.py`
- `flash_attention_backend/test_self_flash_attn_backend.py`
- `flash_attention_backend/analysis/run_perf_eval.sh`
- `flash_attention_backend/toy_flash_attn/README.md`
- `flash_attention_backend/docs/PERFORMANCE_EVAL.md`
- `.agents/skills/flash-attention-performance-report/SKILL.md`

Current facts to verify before use:

- `TOY_FLASH_ATTN_USE=reference` routes to Python reference for paged attention.
- `TOY_FLASH_ATTN_USE=fp32` routes to v4 fp32 CUDA wrapper.
- default/custom bf16 path routes to v4/v3 bf16 op depending on loaded CUDA version.
- `TOY_FLASH_ATTN_USE=official` is interpreted by the vLLM smoke runner as official FlashAttention backend.
- The smoke runner may still accept old misspelling `offical`; verify current code before documenting this.

## CUDA Version Loading Dependency

When changing v3/v4 loading or op aliases, check:

- extension `load(...)` block in `toy_flash_attn/flash_attention_func.py`;
- v3 and v4 op names exported by CUDA bindings;
- wrapper calls using `_ops.flash_attn_varlen_with_block_bf16fp32`;
- fp32 wrapper availability check;
- benchmark case mapping in `analysis/run_perf_eval.sh`;
- README wording.

Current facts to verify before use:

- default CUDA implementation version is `v4`;
- v3 registers only bf16 alias and has no fp32 op alias;
- v4 registers bf16 and fp32 op aliases.

## v4 Kernel Dependency

When changing v4 kernel behavior, check:

- `flash_attention_backend/toy_flash_attn/v4/flash_attn_func.cu`
- `flash_attention_backend/toy_flash_attn/v4/helper.h`
- Python wrapper dtype and stride checks;
- unit tests under `flash_attention_backend/toy_flash_attn/`;
- smoke runner with GPT-2 and Qwen.

Fragile facts to verify before use:

- v4 grid is conceptually batch x q chunk x q head.
- GQA maps q heads to kv heads in CUDA and Python reference.
- v4 top constants currently include `K_X_STRIDE`, `Q_CHUNK_SIZE`, `KV_CHUNK_SIZE`, and `BLOCK_Y`.
- Larger tile constants can increase shared memory pressure, especially fp32 path.
- Debug macros live in `v4/helper.h`; current names should be checked directly before documenting.

## GQA Dependency

When changing GQA, check:

- Python reference head expansion in `flash_attention_func.py`;
- `_check_gqa_heads` in `flash_attention_func.py`;
- CUDA q head / kv head mapping in v4 `ParamSet`;
- launch grid z dimension in v4;
- Qwen smoke test path in `test_self_flash_attn_backend.py`;
- report claims about Qwen and GQA support.

## Benchmark And Report Dependency

Benchmark workflow files:

- `flash_attention_backend/analysis/run_perf_eval.sh`
- `flash_attention_backend/analysis/parse_perf_eval_logs.py`
- `flash_attention_backend/analysis/perf_eval_results.json`
- `flash_attention_backend/analysis/perf_logs/*.log`

Report workflow files:

- `flash_attention_backend/docs/PERFORMANCE_EVAL.md`
- `.agents/skills/flash-attention-performance-report/SKILL.md`

When changing benchmark cases or output fields, check:

- default cases in `run_perf_eval.sh`;
- case filename convention expected by `parse_perf_eval_logs.py`;
- JSON fields consumed by performance report;
- report tables and summary math.

Important report rule:

- Do not copy absolute paths from JSON into the report.
- If logs are stale or contain shell/runtime errors, do not treat parsed rows as valid performance data.

## Quick Verification Patterns

Use these searches before answering related questions:

```bash
rg -n "TOY_FLASH_ATTN_USE|TOY_FLASH_ATTN_CUDA_VERSION" flash_attention_backend
rg -n "K_X_STRIDE|Q_CHUNK_SIZE|KV_CHUNK_SIZE|BLOCK_Y" flash_attention_backend/toy_flash_attn/v4
rg -n "official|offical|FLASH_ATTN|CUSTOM" flash_attention_backend/test_self_flash_attn_backend.py flash_attention_backend/analysis
rg -n "PERFORMANCE_EVAL|perf_eval_results|perf_logs" flash_attention_backend
```
