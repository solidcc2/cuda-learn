---
name: flash-attention-performance-report
description: 默认基于已有 benchmark 产物更新 flash_attention_backend 的性能报告；仅在用户明确要求刷新数据时才运行 benchmark。适用于：更新 PERFORMANCE_EVAL.md、解读 analysis/ 下的结构化结果、校验 benchmark JSON、根据本地代码生成“版本实现与优化摘要”、以及在用户明确要求时按轻量或全量模式重跑 benchmark 后生成报告。
---

# Flash Attention Performance Report

用于更新 `flash_attention_backend/docs/PERFORMANCE_EVAL.md`。

默认只使用已有 benchmark 产物或结构化报告输入，不主动运行 benchmark。

## Core Rules

- 默认不运行 benchmark。只有用户明确要求“刷新数据 / 重跑 / 重新采集”时，才允许运行 benchmark。
- Keep report information orthogonal: separate scope, environment, capability boundary, version-implementation summary, correctness, end-to-end performance, op performance, profiling, version analysis, and conclusion.
- Do not write absolute filesystem paths into the report.
- Do not include private local paths such as home directories, cache directories, virtualenv paths, or machine-specific source roots.
- Prefer repo-relative paths when a path is necessary.
- Do not infer missing results. Mark missing or invalid data as `未采集` or `无有效数据`.
- Do not mix future optimization plans into the report. The report should describe measured facts and current-version analysis.
- Treat `perf_eval_results.json` as a measurement artifact, not as truth if the corresponding log shows an error.
- Do not hard-code the report to a historical backend version set. Derive the current version set from benchmark artifacts and current source files.

## Expected Files

- Benchmark shell entry: `flash_attention_backend/analysis/run_perf_eval.sh`
- Log parser: `flash_attention_backend/analysis/parse_perf_eval_logs.py`
- Structured e2e result: `flash_attention_backend/analysis/perf_eval_results.json`
- Analysis plan: `flash_attention_backend/docs/PERF_PLAN.md`
- Optional future report input: `flash_attention_backend/analysis/artifacts/report/report_inputs.json`
- Optional future version summary: `flash_attention_backend/analysis/artifacts/report/version_optimizations.json`
- Report: `flash_attention_backend/docs/PERFORMANCE_EVAL.md`

## 采集策略

根据 `references/benchmark_collection_modes.md` 决定是否需要运行 benchmark。

默认行为：

- 使用已有产物更新报告
- 不主动触发 benchmark

只有当用户明确要求时，才进入以下模式之一：

- `不跑 bench`
- `轻量 bench`
- `全量 bench`

## Benchmark Workflow

Run benchmark collection from the repository root or by using repo-relative paths:

```bash
bash flash_attention_backend/analysis/run_perf_eval.sh
bash flash_attention_backend/analysis/run_perf_eval.sh light
bash flash_attention_backend/analysis/run_perf_eval.sh full
```

Current collection modes:

- `light`: default, collects smoke-level benchmark data
- `full`: collects broader report-grade benchmark data

Supported versions are defined by the current analysis/e2e collection scripts. Before documenting backend semantics, inspect the current script and `test_self_flash_attn_backend.py`; do not assume old environment-variable behavior.

The analysis shell script is the unified collection entrypoint and regenerates `analysis/perf_eval_results.json` together with structured artifacts under `analysis/artifacts/`.

当用户明确要求运行 benchmark 时：

- `轻量模式`：优先 smoke / 快速刷新
- `全量模式`：用于正式刷新报告数据
- 不要在没有用户明确要求的情况下，从“不跑”自动升级到“轻量”或“全量”

## Version Extensibility

Treat backend versions as data discovered from the current repository, not as fixed skill knowledge.

When updating the report:

- Discover requested/default cases from `run_perf_eval.sh`, especially `DEFAULT_CASES`, `usage()`, and the `case "${version}"` mapping.
- Discover measured cases from `perf_eval_results.json` and validate suspicious rows against `analysis/perf_logs/*.log`.
- Discover backend semantics from current source files, especially `test_self_flash_attn_backend.py` and the toy attention wrapper. If a version-specific CUDA source exists, inspect it before writing kernel-specific claims.
- Discover report structure from `docs/PERF_PLAN.md` before preserving old section names.
- Include every version that is in scope for the current report artifacts. Add report rows or version-analysis subsections for new versions when data or source semantics justify them.
- Do not require a skill update for a new version such as `v6` unless the benchmark artifact schema, report structure, or validation workflow changes.
- If version metadata becomes repetitive, prefer adding a repo artifact such as `analysis/perf_eval_versions.json` or a docs reference, then make the skill read that artifact instead of embedding per-version facts.
- 如果用户没有明确要求重跑数据，优先接受“已有但有效”的产物，而不是主动启动新的 benchmark。

## Version Summary Rules

The report now reserves a dedicated section:

- `版本实现与优化摘要`

Rules for that section:

- Build it from local code, not from benchmark numbers.
- Update it only when relevant implementation paths changed.
- Relevant paths are:
  - `toy_flash_attn/flash_attention_func.py`
  - `toy_flash_attn/v4/`
  - `toy_flash_attn/v5/`
  - `toy_flash_attn/v6/`
- If benchmark data changes but the paths above do not change, keep the previous version-summary content.
- Prefer a structured artifact such as `analysis/artifacts/report/version_optimizations.json` over hand-maintained prose when available.

## JSON Review Checklist

Before updating the report:

- Confirm each benchmark has `success: true`.
- Confirm `input_toks_per_s` and `output_toks_per_s` are not null for performance rows.
- Confirm `prompt_count` and `generated_count` match expected batch size.
- Inspect suspicious logs directly, especially when `success` is true but speed fields are null.
- Exclude stale logs caused by old script failures, for example logs containing only shell errors.

## Report Update Mapping

Use these JSON fields for report sections:

- Environment: `environment.python`, `environment.torch`, `environment.cuda`, `environment.gpu`, `environment.vllm`, `environment.git_commit`.
- Runtime config: benchmark `config.model`, `config.model_config.dtype`, `config.cache_config.cache_dtype`, `config.resolved kv torch dtype`.
- End-to-end performance: `input_toks_per_s`, `output_toks_per_s`, `output_toks_per_s_per_request`, `wall_time_s`.
- Case identity: `version`, `model_arg`, `batch`, `max_tokens`.
- Validity: `success`, `error`, `prompt_count`, `generated_count`.

If `analysis/artifacts/report/report_inputs.json` exists, prefer it as the canonical report input and fall back to `perf_eval_results.json` for e2e-only reports.

Do not copy `log_dir` or `log_path` into the report if they contain absolute paths. If needed, use repo-relative names such as `analysis/perf_logs/v4_qwen_b1_t128.log`.

## Report Writing Rules

When filling `PERFORMANCE_EVAL.md`:

- Environment table should contain only hardware/software facts and measured runtime config.
- Feature boundary table should contain capability statements, not performance claims.
- Correctness section should only contain unittest or numerical comparison data. Do not fill it from generation benchmark JSON.
- `版本实现与优化摘要` should contain only code-backed implementation deltas, not measured speedups.
- `E2E Benchmark` and `Op Benchmark` sections should contain only measured benchmark numbers.
- `Profiling` should contain collected profiling evidence or explicit `未采集`.
- Version analysis should explain current observed behavior by discovered version.
- Keep comparison tables shape-compatible with the discovered case set. Add or remove rows/columns based on current measured cases instead of preserving stale version-specific rows.
- Do not add a debug-on/debug-off overhead section to the standard report unless the user explicitly asks for debug overhead analysis.
- Conclusion should be short and derived directly from filled tables.

Use explicit labels:

- `未采集` for data that was not measured.
- `无有效数据` for data that was attempted but invalid.
- `不适用` for fields that do not apply to that version or case.

## Common Pitfalls

- A stale log can remain in `analysis/perf_logs/` and still be parsed into JSON. Check raw logs if a row has null speeds.
- `baseline` may be too slow or may fail before producing vLLM progress output; do not treat empty speed fields as zero.
- `official` backend behavior depends on current `test_self_flash_attn_backend.py`; inspect `_attention_config()` before documenting the exact environment variable mapping.
- `finish_reason` is only available if the runner prints it. If the parser returns an empty list, report it as `未采集`.
- 当前 `analysis/run_perf_eval.sh` 可以统一采集 `e2e`、`op`、`Correctness` 和 `版本实现与优化摘要`；`Profiling` 仍默认不采集，缺失时应明确写 `未采集`。
