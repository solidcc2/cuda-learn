---
name: flash-attention-performance-report
description: 默认基于已有 benchmark 产物更新 flash_attention_backend 的性能报告；仅在用户明确要求刷新数据时才运行 benchmark。适用于：更新 PERFORMANCE_EVAL.md、解读 analysis/ 下的结构化结果、校验 benchmark JSON、根据本地代码生成"版本实现与优化摘要"、以及在用户明确要求时按轻量或全量模式重跑 benchmark 后生成报告。
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - WebSearch
context: fork
---

# Flash Attention Performance Report

用于更新 `flash_attention_backend/docs/PERFORMANCE_EVAL.md`。

默认只使用已有 benchmark 产物，不主动运行 benchmark。

## 核心规则

- 默认不运行 benchmark。只有用户明确要求"刷新数据 / 重跑 / 重新采集"时，才允许运行。
- 报告内容要正交：范围、环境、能力边界、版本实现摘要、正确性、端到端性能、算子性能、profiling、版本分析、结论，各节分离。
- 不准写绝对路径到报告中。
- 不准写私有本地路径（home、缓存、虚拟环境、机器相关源码根目录）。
- 需要路径时优先用仓库相对路径。
- 不要推断缺失数据。缺失或无效的数据标为 `未采集` 或 `无有效数据`。
- 不要将未来优化计划混入报告。报告只描述测量事实和当前版本分析。
- `analysis/artifacts/e2e/perf_eval_results.json` 是测量产物，如果对应日志有错误则不应视为有效数据。
- 不要将报告硬编码到某个历史版本集。从 benchmark 产物和当前源代码推导当前版本集。

## 涉及文件

- Benchmark 入口：`flash_attention_backend/analysis/run_perf_eval.sh`
- E2E 解析器：`flash_attention_backend/bench/e2e/parse_perf_eval_logs.py`
- 结构化 E2E 结果：`flash_attention_backend/analysis/artifacts/e2e/perf_eval_results.json`
- 分析计划：`flash_attention_backend/docs/PERF_PLAN.md`
- 可选报告输入：`flash_attention_backend/analysis/artifacts/report/report_inputs.json`
- 可选版本摘要：`flash_attention_backend/analysis/artifacts/report/version_optimizations.json`
- 可选 NCU 采集器：`flash_attention_backend/analysis/run_ncu_case.sh`
- 可选 NCU 摘要生成器：`flash_attention_backend/analysis/summarize_ncu_report.py`
- 最终报告：`flash_attention_backend/docs/PERFORMANCE_EVAL.md`

## 采集策略

默认行为：使用已有产物更新报告，不触发 benchmark。

只有当用户明确要求时，才进入以下模式之一：

- `不跑 bench`
- `轻量 bench`
- `全量 bench`
- `轻量 bench + NCU`
- `全量 bench + NCU`

当用户只说"更新报告"或"刷新报告"，没说明是否重跑时，优先用中文选项提示：

1. `仅更新报告（推荐）` — 使用已有产物，不重跑
2. `刷新轻量数据并更新报告` — 运行 `analysis/run_perf_eval.sh`
3. `刷新全量数据并更新报告` — 运行 `analysis/run_perf_eval.sh full`
4. `刷新轻量数据并重跑 NCU` — 运行 `analysis/run_perf_eval.sh --with-ncu --ncu-case <case>`
5. `刷新全量数据并重跑 NCU` — 运行 `analysis/run_perf_eval.sh full --with-ncu --ncu-case <case>`

提示要求：
- 选项说明用中文
- 默认推荐始终是 `仅更新报告（推荐）`
- 用户选择 NCU 但没给 case 时，追问支持的 case

## Benchmark 工作流

从仓库根目录执行：

```bash
bash flash_attention_backend/analysis/run_perf_eval.sh
bash flash_attention_backend/analysis/run_perf_eval.sh light
bash flash_attention_backend/analysis/run_perf_eval.sh full
bash flash_attention_backend/analysis/run_perf_eval.sh light --with-ncu --ncu-case qwen_like_b1_s128_h64
bash flash_attention_backend/analysis/run_perf_eval.sh full --with-ncu --ncu-case qwen_like_b4_s512_h64
```

采集模式：
- `light`（默认）：smoke 级
- `full`：正式报告级
- `--with-ncu`：显式开启 NCU

当用户明确要求运行 benchmark 时：
- `轻量模式`：快速刷新
- `全量模式`：正式刷新报告数据
- 只有用户明确要求 profiling / NCU，才加 `--with-ncu`
- 不要从"不跑"自动升级到"轻量"或"全量"

## 版本扩展性

将后端版本视为从当前仓库发现的数据，而非固定在 skill 中的知识。

更新报告时：
- 从 `run_perf_eval.sh` 发现默认 case 和版本列表
- 从 `analysis/artifacts/e2e/perf_eval_results.json` 发现实际测量的 case
- 从当前源代码发现后端语义
- 从 `docs/PERF_PLAN.md` 发现报告结构
- 在报告产物范围内的每个版本都要覆盖到
- 新版本（如 v6）不需要更新 skill，除非产物 schema、报告结构或校验流程变了
- 如果版本元数据变得重复，优先用仓库内的结构化产物替代

## 版本摘要规则

报告中保留 `版本实现与优化摘要` 一节。规则：
- 从本地代码构建，不从 benchmark 数据构建
- 只在相关实现路径变化时更新
- 相关路径：`flash_attention_func.py`、`v4/`、`v5/`、`v6/`
- 如果 benchmark 数据变了但上述路径没变，保留之前的摘要内容
- 优先用 `analysis/artifacts/report/version_optimizations.json` 而非手写内容

## JSON 校验清单

更新报告前确认：
- 每个 benchmark 有 `success: true`
- 性能行有非空的 `input_toks_per_s` 和 `output_toks_per_s`
- `prompt_count` 和 `generated_count` 匹配预期 batch size
- 对可疑日志直接审查原文，尤其是 `success` 为 true 但速度字段为 null 的情况
- 排除因脚本失败产生的过期日志

## 报告字段映射

- 环境：`environment.python`、`.torch`、`.cuda`、`.gpu`、`.vllm`、`.git_commit`
- 运行时配置：`config.model`、`.dtype`、`.cache_dtype`、`.resolved kv torch dtype`
- 端到端性能：`input_toks_per_s`、`output_toks_per_s`、`output_toks_per_s_per_request`、`wall_time_s`
- case 标识：`version`、`model_arg`、`batch`、`max_tokens`
- 有效性：`success`、`error`、`prompt_count`、`generated_count`

如果 `analysis/artifacts/report/report_inputs.json` 存在，优先作为报告输入；否则回退到 `analysis/artifacts/e2e/perf_eval_results.json`。

如果 `report_inputs.json` 包含有效的 `profiling` 节，用它作为 profiling 输入；否则标 `未采集`。

不要将 `log_dir` 或 `log_path` 的绝对路径写入报告。

## 报告写作规则

填充 `PERFORMANCE_EVAL.md` 时：
- 环境表只包含硬件/软件事实和测量的运行时配置
- 能力边界表只包含能力声明，不含性能声明
- 正确性节只包含单元测试或数值对拍数据
- `版本实现与优化摘要` 只包含代码层面的实现差异，不含测量加速比
- `E2E Benchmark` 和 `Op Benchmark` 节只包含测量的 benchmark 数据
- `Profiling` 节包含 profiling 证据或明确标 `未采集`
- 版本分析基于当前版本数据解释观察到的行为
- 对比表保持与当前测量的 case 集兼容
- 不要往标准报告里加 debug 开销对比节，除非用户明确要求
- 结论简短，直接从填充的表中推导

**增量更新原则**：已有版本结论不变的段落不重写。只做三件事：(1) 更新数值测量；(2) 扩展新版本的 case 行/分析段；(3) 在结论有变化时迭代，无变化不动。不要为已有版本重写结论一致的描述。

明确标签：
- `未采集` — 没有测量
- `无有效数据` — 尝试过但结果无效
- `不适用` — 该版本或 case 不适用

## 常见问题

- 过期的日志文件仍在 `analysis/perf_logs/` 中，可能被解析进 JSON。如果某行速度为 null，直接检查原始日志。
- `baseline` 可能太慢或在 vLLM 输出前就失败了；不要把空速度字段当作 0。
- `official` 后端行为取决于当前 `test_self_flash_attn_backend.py`；在记录精确的环境变量映射前先检查 `_attention_config()`。
- `finish_reason` 只有 runner 打印时才可用。如果解析器返回空列表，标 `未采集`。
- 当前 `analysis/run_perf_eval.sh` 统一采集 e2e、op、correctness 和版本摘要；NCU 只在 `--with-ncu` 时采集，缺失标 `未采集`。
