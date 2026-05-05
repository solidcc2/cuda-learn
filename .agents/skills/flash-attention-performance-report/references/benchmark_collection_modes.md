# Benchmark 采集模式

## 默认规则

默认使用现有产物更新报告，不主动运行 benchmark。

优先读取：

1. `analysis/artifacts/report/report_inputs.json`
2. `analysis/perf_eval_results.json`
3. `analysis/` 下的原始日志

只有当用户明确要求“刷新数据 / 重跑 / 重新采集”时，才运行 benchmark。

## 允许触发 benchmark 的典型表达

- “重跑 benchmark”
- “刷新报告数据”
- “重新采集 e2e 数据”
- “跑一版轻量 bench”
- “跑全量 bench 并生成报告”

如果用户只是要求“更新报告”或“整理报告”，默认不跑 benchmark。

## 采集模式

### 轻量模式

用于快速刷新、smoke 验证、低成本检查。

当前推荐命令：

```bash
bash flash_attention_backend/analysis/run_perf_eval.sh
bash flash_attention_backend/analysis/run_perf_eval.sh light
```

用途：

- 快速刷新 e2e 汇总
- 验证当前链路是否还能工作
- 避免长时间跑数

### 全量模式

用于用户明确要求正式刷新报告数据时。

当前推荐命令：

```bash
bash flash_attention_backend/analysis/run_perf_eval.sh full
```

用途：

- 覆盖更完整的 e2e case
- 生成更接近正式报告的数据基础

## 当前范围限制

当前 `analysis/run_perf_eval.sh` 已作为统一采集入口，负责：

- `e2e`
- `Op Benchmark`
- `Correctness Gate`
- `版本实现与优化摘要`

但 `Profiling` 仍不默认实跑。

- profiling 仍来自 `bench/op/profile_attention_op.py`

如果 profiling 产物不存在，报告中应明确写 `未采集`，不要推断或补造数据。
