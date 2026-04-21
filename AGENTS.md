# Repository Instructions

This repository has repo-local Codex skills under `.agents/skills/`.

## Skill Routing

For any task involving `flash_attention_backend` current behavior, runtime configuration, dispatch paths, benchmark/report workflow, CUDA kernel constants, docs, tests, or branch/commit drift, use `flash-attention-project-context` before making conclusions or edits.

Use `flash-attention-performance-report` when updating `flash_attention_backend/docs/PERFORMANCE_EVAL.md` from benchmark logs or `flash_attention_backend/analysis/perf_eval_results.json`.

Use `personal-coding-principles` for code design, refactoring, state/edge-boundary decisions, error-handling strategy, dependency cleanup, or complex logic simplification.

## Available Project Skills

- `flash-attention-project-context` for current project state, dependency checks, environment variables, dispatch paths, CUDA kernel constants, benchmark scripts, docs, tests, and branch/commit drift.
- `flash-attention-performance-report` for updating `flash_attention_backend/docs/PERFORMANCE_EVAL.md` from benchmark logs or `flash_attention_backend/analysis/perf_eval_results.json`.

## Verification

Before relying on project state, verify task-relevant facts against current files, logs, JSON, or git state. Repo-local context is a navigation aid, not the final source of truth.
