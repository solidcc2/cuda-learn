# Benchmark 采集模式

## 默认规则

默认使用现有产物更新报告，不主动运行 benchmark。

优先读取：

1. `analysis/artifacts/report/report_inputs.json`
2. `analysis/artifacts/e2e/perf_eval_results.json`
3. `analysis/` 下的原始日志

只有当用户明确要求“刷新数据 / 重跑 / 重新采集”时，才运行 benchmark。

如果用户没有说清楚刷新范围，优先使用中文选项提示一次：

1. `仅更新报告（推荐）`
2. `刷新轻量数据并更新报告`
3. `刷新全量数据并更新报告`
4. `刷新轻量数据并重跑 NCU`
5. `刷新全量数据并重跑 NCU`

默认推荐：

- `仅更新报告（推荐）`

## 允许触发 benchmark 的典型表达

- “重跑 benchmark”
- “刷新报告数据”
- “重新采集 e2e 数据”
- “跑一版轻量 bench”
- “跑全量 bench 并生成报告”
- “跑一版轻量 bench 并刷新 NCU”
- “重跑 profiling / NCU”

如果用户只是要求“更新报告”或“整理报告”，默认不跑 benchmark。

## 采集模式

### 轻量模式

用于快速刷新、smoke 验证、低成本检查。

当前推荐命令：

```bash
bash flash_attention_backend/analysis/run_perf_eval.sh
bash flash_attention_backend/analysis/run_perf_eval.sh light
bash flash_attention_backend/analysis/run_perf_eval.sh light --with-ncu --ncu-case qwen_like_b1_s128_h64
```

用途：

- 快速刷新 e2e 汇总
- 验证当前链路是否还能工作
- 避免长时间跑数

对应中文选项：

- `刷新轻量数据并更新报告`

### 全量模式

用于用户明确要求正式刷新报告数据时。

当前推荐命令：

```bash
bash flash_attention_backend/analysis/run_perf_eval.sh full
bash flash_attention_backend/analysis/run_perf_eval.sh full --with-ncu --ncu-case qwen_like_b4_s512_h64
```

用途：

- 覆盖更完整的 e2e case
- 生成更接近正式报告的数据基础

对应中文选项：

- `刷新全量数据并更新报告`

## NCU 模式

只有当用户明确要求刷新 profiling / NCU 时才使用。

当前支持的中文选项：

- `刷新轻量数据并重跑 NCU`
- `刷新全量数据并重跑 NCU`

当前支持的 NCU case：

- `qwen_like_b1_s128_h64`
- `qwen_like_b4_s512_h64`
- `qwen_like_b1_s2048_h64`

推荐命令：

```bash
bash flash_attention_backend/analysis/run_perf_eval.sh --with-ncu --ncu-case qwen_like_b1_s128_h64
bash flash_attention_backend/analysis/run_perf_eval.sh full --with-ncu --ncu-case qwen_like_b4_s512_h64
```

## 当前范围限制

当前 `analysis/run_perf_eval.sh` 已作为统一采集入口，负责：

- `e2e`
- `Op Benchmark`
- `Correctness Gate`
- `版本实现与优化摘要`

但 `Profiling` / `NCU` 仍不默认实跑。

- profiling 被测入口仍来自 `bench/op/profile_attention_op.py`
- `NCU` 只在显式传入 `--with-ncu` 时由 `analysis/run_ncu_case.sh` 调用

如果 profiling 产物不存在，报告中应明确写 `未采集`，不要推断或补造数据。
