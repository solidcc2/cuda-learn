---
name: flash-attention-project-context
description: Maintain and use the flash_attention_backend project context map. Use when answering or changing current project behavior involving environment variables, dispatch paths, CUDA kernel constants, test entrypoints, benchmark scripts, reports, docs, branch/commit drift, or any conclusion that may be stale. Requires reading references/PROJECT_CONTEXT.md and verifying low-cost facts against live repository files, logs, JSON, or git state instead of relying on cached memory.
---

# Flash Attention Project Context

Use this skill for tasks that depend on the current state of `flash_attention_backend`.

## Mandatory Workflow

1. Read `references/PROJECT_CONTEXT.md` as a navigation map.
2. Check the current git commit and compare it with the commit recorded in the context map.
3. If the commit differs, decide whether the context map needs a full refresh or an incremental refresh before relying on it.
4. Verify task-relevant facts in the live repository with `rg`, `sed`, `git diff`, or direct file reads.
5. Treat the context map and chat memory as hints, not final authority.
6. If code and the context map disagree, trust the code and update the context map.
7. If changing behavior, update the context map when dependencies, entrypoints, environment variables, paths, or feature support change.

## Verification Rules

- Low-cost facts must be checked in place before answering. Examples: supported env vars, default benchmark cases, report path, CUDA version selection, debug macro names.
- High-cost facts that require running tests must be labeled clearly if not run.
- Log/report conclusions must be based on current logs or JSON, not memory.
- Do not write absolute paths or private local paths into project docs or reports.
- Prefer repo-relative paths in explanations and state updates.

## Commit Drift Rules

- Record the current git commit in `references/PROJECT_CONTEXT.md` whenever the context map is refreshed.
- If the current commit differs from the recorded commit, do not assume the context map is current.
- Use incremental refresh when the diff touches only a narrow area relevant to the task.
- Use full refresh when the branch changed broadly, the diff is large, or the task touches env vars, dispatch paths, benchmark workflow, report generation, CUDA launch constraints, or GQA.
- After refresh, update the recorded commit and note the refresh scope.

## When To Update The Context Map

Update `references/PROJECT_CONTEXT.md` after changes to:

- `TOY_FLASH_ATTN_USE` semantics.
- `TOY_FLASH_ATTN_CUDA_VERSION` semantics.
- v3/v4 extension loading or op aliasing.
- v4 kernel constants, launch constraints, GQA mapping, dtype path, or debug macros.
- benchmark scripts under `analysis/`.
- report location or report-generation workflow.
- README or docs that describe runtime behavior.
- test entrypoints or model/backend selection logic.

## Context Map Scope

The context map should stay small. It should record:

- concepts and their owning files;
- dependency checks after changing a concept;
- current fragile facts that are easy to misremember;
- verification commands or search patterns.

Do not turn it into a detailed changelog, bug diary, or full architecture document.
