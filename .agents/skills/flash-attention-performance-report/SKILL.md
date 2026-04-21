---
name: flash-attention-performance-report
description: Update the flash_attention_backend performance evaluation report from benchmark logs or perf_eval_results.json. Use when refreshing flash_attention_backend/docs/PERFORMANCE_EVAL.md, interpreting analysis/run_perf_eval.sh outputs, validating parsed benchmark JSON, or filling report tables while keeping scope, environment, correctness, performance, version analysis, and conclusions orthogonal and avoiding absolute or private local paths.
---

# Flash Attention Performance Report

Use this skill when the task is to update `flash_attention_backend/docs/PERFORMANCE_EVAL.md` from benchmark outputs under `analysis/`.

## Core Rules

- Keep report information orthogonal: separate scope, environment, feature boundary, correctness, end-to-end performance, version analysis, debug overhead, and conclusion.
- Do not write absolute filesystem paths into the report.
- Do not include private local paths such as home directories, cache directories, virtualenv paths, or machine-specific source roots.
- Prefer repo-relative paths when a path is necessary.
- Do not infer missing results. Mark missing or invalid data as `未采集` or `无有效数据`.
- Do not mix future optimization plans into the report. The report should describe measured facts and current-version analysis.
- Treat `perf_eval_results.json` as a measurement artifact, not as truth if the corresponding log shows an error.

## Expected Files

- Benchmark shell entry: `flash_attention_backend/analysis/run_perf_eval.sh`
- Log parser: `flash_attention_backend/analysis/parse_perf_eval_logs.py`
- Structured result: `flash_attention_backend/analysis/perf_eval_results.json`
- Raw logs: `flash_attention_backend/analysis/perf_logs/*.log`
- Report: `flash_attention_backend/docs/PERFORMANCE_EVAL.md`

## Benchmark Workflow

Run benchmark cases from the repository root or by using repo-relative paths:

```bash
bash flash_attention_backend/analysis/run_perf_eval.sh v4:qwen:1:128 official:qwen:1:128
```

Case format:

```text
version:model:batch:max_tokens
```

Supported versions are defined by `run_perf_eval.sh`. Before documenting backend semantics, inspect the current script and `test_self_flash_attn_backend.py`; do not assume old environment-variable behavior.

The shell script writes raw logs to `analysis/perf_logs/` and then regenerates `analysis/perf_eval_results.json`.

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

Do not copy `log_dir` or `log_path` into the report if they contain absolute paths. If needed, use repo-relative names such as `analysis/perf_logs/v4_qwen_b1_t128.log`.

## Report Writing Rules

When filling `PERFORMANCE_EVAL.md`:

- Environment table should contain only hardware/software facts and measured runtime config.
- Feature boundary table should contain capability statements, not performance claims.
- Correctness section should only contain unittest or numerical comparison data. Do not fill it from generation benchmark JSON.
- Performance section should contain only measured benchmark numbers.
- Version analysis should explain current observed behavior by version.
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
