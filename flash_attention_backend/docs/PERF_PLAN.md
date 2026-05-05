# toy_flash_attn Analysis 与报告方案

## 1. 目标

当前仓库已经拆出三类入口：

- `bench/e2e/`：端到端 benchmark
- `bench/op/`：单接口 benchmark / profiling
- `tests/correctness/`：正确性与回放入口

当前 `analysis/run_perf_eval.sh` 已被定义为 bench 运行与数据采集的统一入口点。

`analysis/` 的职责是统一编排、落产物、生成报告输入。

目标是稳定回答：

1. `baseline / v5 / v6 / official` 端到端差多少
2. `baseline / v5 / v6 / official` 单接口差多少
3. `v6` 的单 kernel profiling 入口是什么
4. 当前 correctness gate 的状态是什么

## 2. analysis 分层职责

`analysis/` 统一入口做三件事：

1. 调度现有入口
2. 统一落结构化产物
3. 组装报告输入

`analysis/` 不应重新实现 benchmark、profiling 或 correctness 逻辑。

## 3. 当前统一入口与建议目录

```text
flash_attention_backend/analysis/
  run_perf_eval.sh
  collect_correctness_metrics.py
  build_report_inputs.py
  summarize_version_optimizations.py

  artifacts/
    e2e/
      perf_json/
      perf_logs/
      perf_eval_results.json
    op/
      case_json/
      logs/
      op_results.json
    correctness/
      logs/
      correctness_summary.json
    profile/
      profile_manifest.json
    report/
      report_inputs.json
      version_optimizations.json
```

## 4. 数据来源映射

### 4.1 E2E

- 统一入口：`analysis/run_perf_eval.sh`
- e2e 子入口：`bench/e2e/collect_e2e.sh`
- 单 case runner：`bench/e2e/run_vllm_e2e.py`
- 汇总器：`bench/e2e/parse_perf_eval_logs.py`

建议落产物：

- `analysis/artifacts/e2e/perf_json/*.json`
- `analysis/artifacts/e2e/perf_logs/*.log`
- `analysis/artifacts/e2e/perf_eval_results.json`

### 4.2 Op

- 统一入口：`analysis/run_perf_eval.sh`
- op 子入口：`bench/op/bench_attention_op.py`
- case 定义：`bench/op/cases_op.py`

建议落产物：

- `analysis/artifacts/op/case_json/*.json`
- `analysis/artifacts/op/logs/*.log`
- `analysis/artifacts/op/op_results.json`

### 4.3 Correctness

- 统一入口：`analysis/run_perf_eval.sh`
- 测试入口：`tests/correctness/test_fa2_parity.py`
- 入口：`tests/correctness/test_paged_kv_parity.py`
- 入口：`tests/correctness/test_cuda_regression.py`
- 入口：`tests/correctness/test_replay_dumps.py`

建议落产物：

- `analysis/artifacts/correctness/logs/*.log`
- `analysis/artifacts/correctness/correctness_summary.json`

### 4.4 Profiling

- 入口：`bench/op/profile_attention_op.py`
- case 定义：`bench/op/cases_op.py`

建议落产物：

- `analysis/artifacts/profile/profile_manifest.json`

当前建议先固定 profiling case 和命令，不在 `analysis/` 内自动跑 `ncu`。

## 5. 报告输入

报告最终只消费一份汇总文件：

- `analysis/artifacts/report/report_inputs.json`

建议字段：

- `environment`
- `capability_boundary`
- `correctness`
- `e2e`
- `op`
- `profiling`
- `version_optimizations`

## 6. 版本实现与优化摘要

报告中新增一章：

- `版本实现与优化摘要`

这一章只来自本地代码，不来自 benchmark 数字。

建议由：

- `analysis/summarize_version_optimizations.py`

生成：

- `analysis/artifacts/report/version_optimizations.json`

建议字段：

- `version`
- `positioning`
- `added_vs_previous`
- `code_evidence`

## 7. 更新规则

### 7.1 何时更新 benchmark 数据

当统一入口重新执行时更新：

- `analysis/run_perf_eval.sh`

### 7.2 何时更新“版本实现与优化摘要”

只有以下代码路径有变动时才更新：

- `toy_flash_attn/flash_attention_func.py`
- `toy_flash_attn/v4/`
- `toy_flash_attn/v5/`
- `toy_flash_attn/v6/`

如果只是 benchmark 数据变化，而上述代码无变动，则不更新该章节。

## 8. 文档结构

`PERFORMANCE_EVAL.md` 采用以下结构：

1. `评估范围`
2. `测试环境`
3. `能力与口径边界`
4. `版本实现与优化摘要`
5. `Correctness Gate`
6. `E2E Benchmark`
7. `Op Benchmark`
8. `Profiling`
9. `版本分析`
10. `分层结论`

原则：

- `能力与口径边界` 只描述能力与可比性
- `版本实现与优化摘要` 只描述代码可证实的实现差异
- `Correctness Gate` 只描述 gate 状态
- `E2E Benchmark` / `Op Benchmark` 只描述测量结果
- `Profiling` 只描述入口、case 和已采集状态
