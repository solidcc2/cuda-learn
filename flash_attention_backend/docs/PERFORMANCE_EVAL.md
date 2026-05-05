# toy_flash_attn 性能评估与分析报告

## 1. 评估范围

本文档基于当前统一采集产物更新，数据来源为：

- `analysis/run_perf_eval.sh`
- `analysis/artifacts/report/report_inputs.json`
- `analysis/artifacts/report/version_optimizations.json`

本次报告使用的是当前统一入口生成的 `light` 级产物，因此覆盖范围是：

- `E2E Benchmark`：smoke 级 case
- `Op Benchmark`：最小标准 case 集
- `Correctness`：collect 模式摘要
- `Profiling`：3 个显式触发的 NCU case

当前报告纳入的版本：

| 版本 | 当前状态 | 说明 |
| --- | --- | --- |
| baseline | 已采集 | Python reference paged path |
| v5 | 已采集 | 自定义 CUDA paged path |
| v6 | 已采集 | 自定义 CUDA paged path，当前默认 CUDA 实现 |
| official | 已采集 | 官方 decoder 对照路径 |

本次报告不再沿用历史 `v4` 表格数据作为当前主结果。

## 2. 测试环境

环境信息来自 `report_inputs.json`。

| 项目 | 数值 |
| --- | --- |
| 平台 | `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35` |
| GPU | `NVIDIA GeForce RTX 4070 Ti SUPER` |
| Python | `3.10.12` |
| PyTorch | `2.10.0+cu128` |
| CUDA toolkit | `12.8` |
| `torch.cuda.is_available()` | `true` |
| vLLM | `0.17.1` |
| git commit | `c8a742c0119dccab8275c259d8f0c180572d0342` |

说明：

- 当前结构化环境信息会记录 GPU 型号，但不会自动记录图形界面程序占用等运行时背景。
- 当前报告不再直接复用旧的 RTX 2050 结果作为本轮绝对值比较基线。

## 3. 能力与口径边界

| 维度 | baseline | v5 | v6 | official |
| --- | --- | --- | --- | --- |
| paged KV path | 支持 | 支持 | 支持 | e2e 对照路径支持 |
| varlen query | 支持 | 支持 | 支持 | 支持 |
| causal mask | 支持 | 支持 | 支持 | 支持 |
| sliding window | 支持 | 支持 | 支持 | 支持 |
| GQA | 支持 | 支持 | 支持 | 支持 |
| bf16 输入 | 支持 | 支持 | 支持 | 支持 |
| 主 correctness 口径 | 支持 | 支持 | `head_dim=64` 主线 | 不按 paged-op 支持矩阵展开 |
| op 比较口径 | paged | paged | paged | dense |

当前报告中的比较边界：

- `baseline`：Python / reference paged path
- `v5 / v6`：自定义 CUDA paged path
- `official`：当前 e2e 路径中的官方 decoder 对照；op 层当前仍应理解为 dense-path 对照，不写成与 paged-op 完全同构

## 4. 版本实现与优化摘要

本节来自 `analysis/artifacts/report/version_optimizations.json`，只记录代码可证实的实现差异。

### baseline

- 版本定位：Python reference paged attention path
- 相对上一版本的新增点：
  - 作为 paged KV 语义与 correctness 对比的参考实现
- 代码证据：
  - `toy_flash_attn/flash_attention_func.py`
  - `bench/common.py`

### v4

- 版本定位：第一版通过统一 wrapper 暴露的自定义 CUDA paged path
- 相对上一版本的新增点：
  - 注册 bf16 和 fp32 CUDA alias
  - 底层实现位于 `toy_flash_attn/v4/`

### v5

- 版本定位：WMMA / Tensor Core 路径的自定义 CUDA paged path
- 相对上一版本的新增点：
  - 通过统一 bf16 CUDA wrapper 绑定 v5 kernel
  - 底层实现切换到 `toy_flash_attn/v5/`

### v6

- 版本定位：基于 CuTe 的自定义 CUDA paged path
- 相对上一版本的新增点：
  - 当 `TOY_FLASH_ATTN_CUDA_VERSION` 未设置时，默认选择 v6
  - 通过统一 bf16 CUDA wrapper 绑定 v6 kernel
  - 当前导出的主 specialization 只有 `head_dim=64`

### official

- 版本定位：官方 FlashAttention decoder 对照路径
- 相对上一版本的新增点：
  - 使用官方 backend 配置而不是自定义 paged CUDA path
  - 当前 op 层比较边界仍以 dense-path 对照理解

## 5. Correctness And Numerical Stability

当前 correctness 数据来自 collect 模式摘要，而不是 unittest hard fail gate。

### 5.1 支持范围

当前 `v6` collect 模式主线支持范围：

| 路径 | 当前支持 |
| --- | --- |
| `with_block_cu` | `head_dim=64` |
| `with_block_cu` 不纳入当前主线 | `head_dim=16/32` |

对应摘要：

- `cuda_impl_version = v6`
- `supported_head_dims = [64]`
- `unsupported_head_dims = [16, 32]`

### 5.2 统计摘要

| 指标 | 数值 |
| --- | --- |
| overall status | `sensitive` |
| pass count | `5` |
| sensitive count | `1` |
| unsupported count | `2` |
| error count | `0` |
| strict threshold | `atol=2e-3, rtol=2e-3` |
| relaxed threshold | `atol=8e-3, rtol=8e-3` |

### 5.3 case 明细

| case | status | max abs diff | mean abs diff | 说明 |
| --- | --- | --- | --- | --- |
| `fa2_without_block_full` | `pass` | `0.0009766` | `0.0001123` | dense 对 official FA2 |
| `fa2_with_block_full` | `pass` | `0.0009766` | `0.0001123` | paged 对 official FA2 |
| `paged_kv_block_table_mapping` | `pass` | `0.0009766` | `0.0001019` | paged block table 语义 |
| `paged_kv_local_causal_window` | `pass` | `0.0019531` | `0.0000867` | paged local+causal |
| `cuda_with_block_cu_head_dim_16` | `unsupported` | 不适用 | 不适用 | 当前 `v6` 主线不支持 |
| `cuda_with_block_cu_head_dim_32` | `unsupported` | 不适用 | 不适用 | 当前 `v6` 主线不支持 |
| `cuda_with_block_cu_head_dim_64_minimal` | `pass` | `0.0000000` | `0.0000000` | `v6 + hd64` 最小 smoke |
| `cuda_with_block_cu_head_dim_64_sensitive` | `sensitive` | `0.0078125` | `0.0010326` | strict 不过，relaxed 可接受 |

当前结论：

- collect 模式下没有结构性错误
- `v6 + head_dim=64` 主线可以进入性能分析
- `head_dim=16/32` 当前应明确视为 `unsupported`

## 6. E2E Benchmark

本节使用 smoke 级 e2e 结果，共 12 个样本。

### 6.1 Case 覆盖

| case | 版本覆盖 |
| --- | --- |
| `gpt2_b1_t128` | baseline / v5 / v6 / official |
| `qwen_b1_t128` | baseline / v5 / v6 / official |
| `qwen_b4_t128` | baseline / v5 / v6 / official |

统一运行参数：

- `prompt_len = 256`
- `max_tokens = 128`
- `warmup = 1`
- `repeat = 3`

### 6.2 样本明细

| case | 版本 | path | output toks/s | avg latency |
| --- | --- | --- | --- | --- |
| `gpt2_b1_t128` | baseline | paged | `3.96` | `32306.23 ms` |
| `gpt2_b1_t128` | v5 | paged | `102.54` | `1248.28 ms` |
| `gpt2_b1_t128` | v6 | paged | `107.73` | `1188.18 ms` |
| `gpt2_b1_t128` | official | dense | `691.53` | `185.10 ms` |
| `qwen_b1_t128` | baseline | paged | `1.83` | `69852.85 ms` |
| `qwen_b1_t128` | v5 | paged | `52.22` | `2451.14 ms` |
| `qwen_b1_t128` | v6 | paged | `53.68` | `2384.31 ms` |
| `qwen_b1_t128` | official | dense | `321.51` | `398.12 ms` |
| `qwen_b4_t128` | baseline | paged | `1.89` | `271356.92 ms` |
| `qwen_b4_t128` | v5 | paged | `208.50` | `2455.59 ms` |
| `qwen_b4_t128` | v6 | paged | `210.86` | `2428.14 ms` |
| `qwen_b4_t128` | official | dense | `1191.08` | `429.86 ms` |

### 6.3 E2E 摘要

| 对比项 | 结果 |
| --- | --- |
| `qwen_b1_t128`：`v6 / v5` | `1.03x` |
| `qwen_b1_t128`：`official / v6` | `5.99x` |
| `qwen_b4_t128`：`v6 / v5` | `1.01x` |
| `qwen_b4_t128`：`official / v6` | `5.65x` |
| `gpt2_b1_t128`：`v6 / v5` | `1.05x` |
| `gpt2_b1_t128`：`official / v6` | `6.42x` |

当前观察：

- `v5` 和 `v6` 在 e2e smoke 上非常接近
- `official` 在当前环境下明显快于自定义 paged 路径
- `baseline` 在 e2e 上极慢，尤其 `qwen_b4_t128`，不适合作为默认轻量模式的重 case

## 7. Op Benchmark

本节使用 light 模式下的 12 个 op 样本。

### 7.1 Case 覆盖

| case | 版本覆盖 |
| --- | --- |
| `gpt2_like_b1_s128_h64` | baseline / v5 / v6 / official |
| `gqa_case_b1_s128` | baseline / v5 / v6 / official |
| `qwen_like_b1_s128_h64` | baseline / v5 / v6 / official |

### 7.2 样本明细

| case | 版本 | path | avg ms | tokens/s |
| --- | --- | --- | --- | --- |
| `gpt2_like_b1_s128_h64` | baseline | paged | `6.4115` | `155.97` |
| `gpt2_like_b1_s128_h64` | v5 | paged | `0.0849` | `11779.49` |
| `gpt2_like_b1_s128_h64` | v6 | paged | `0.0772` | `12960.14` |
| `gpt2_like_b1_s128_h64` | official | dense | `0.8571` | `1166.73` |
| `gqa_case_b1_s128` | baseline | paged | `6.1665` | `162.17` |
| `gqa_case_b1_s128` | v5 | paged | `0.0764` | `13095.29` |
| `gqa_case_b1_s128` | v6 | paged | `0.0769` | `13011.46` |
| `gqa_case_b1_s128` | official | dense | `0.0583` | `17142.56` |
| `qwen_like_b1_s128_h64` | baseline | paged | `5.5660` | `179.66` |
| `qwen_like_b1_s128_h64` | v5 | paged | `0.1178` | `8489.11` |
| `qwen_like_b1_s128_h64` | v6 | paged | `0.0942` | `10612.29` |
| `qwen_like_b1_s128_h64` | official | dense | `0.0974` | `10262.76` |

### 7.3 Op 摘要

| 对比项 | 结果 |
| --- | --- |
| `qwen_like_b1_s128_h64`：`v6 / v5` | `1.25x` |
| `qwen_like_b1_s128_h64`：`official / v6` | `0.97x` |
| `gpt2_like_b1_s128_h64`：`v6 / v5` | `1.10x` |
| `gpt2_like_b1_s128_h64`：`official / v6` | `0.09x` |
| `gqa_case_b1_s128`：`v6 / v5` | `0.99x` |
| `gqa_case_b1_s128`：`official / v6` | `1.32x` |

当前观察：

- op 层并不呈现与 e2e 同样巨大的 `official / v6` 差距
- `qwen_like_b1_s128_h64` 上，`v6` 已接近 `official`
- `gqa_case_b1_s128` 上，`official` 仍快于 `v6`
- `gpt2_like_b1_s128_h64` 上，当前 `official` dense 路径结果明显不同于 paged 路径，说明 comparison boundary 仍需谨慎解释

## 8. Profiling

当前状态：`已采集`

当前 profiling 数据来自显式开启的 NCU 采集，而不是默认统一入口：

- `analysis/run_perf_eval.sh --with-ncu`
- `analysis/run_ncu_case.sh`
- `analysis/summarize_ncu_report.py`

本轮已采集 case：

| case | v6 kernel 时长 | official kernel 时长 | v6 / official |
| --- | --- | --- | --- |
| `qwen_like_b1_s128_h64` | `45.152 us` | `8.576 us` | `5.26x` |
| `qwen_like_b4_s512_h64` | `163.840 us` | `9.952 us` | `16.46x` |
| `qwen_like_b1_s2048_h64` | `635.520 us` | `9.632 us` | `65.98x` |

当前 3 个 case 共同出现的风险标签：

- `underfilled_grid`
- `scheduler_starvation_risk`
- `uncoalesced_global_access_risk`
- `shared_bank_conflict_risk`
- `local_spill_risk`

当前可以直接从摘要读出的事实：

- `v6` 在 3 个采集 case 上都明显慢于 `official`
- 随着 `k_lens` 增大，`v6 / official` 的 kernel 时长倍率显著放大
- `v6` 的 grid 很小，`eligible warps per scheduler` 长期只有 `~0.14`
- `v6` 的 global excessive sectors 和 shared bank conflict 指标明显高于 `official`

边界说明：

- 本轮 NCU 摘要主要稳定提取了 kernel 时长、吞吐、active/eligible warps、grid/block、shared bank conflict 和访存放大指标
- `dram_throughput_pct`、`l2_hit_rate`、`achieved_occupancy_pct` 在当前导出格式下仍未稳定解析，因此本节不把它们写成结论

## 9. 版本分析

### baseline

- e2e 明显极慢，尤其 `qwen_b4_t128`
- op 层主要用于 reference / 语义对照，不是性能目标

### v5

- 已经明显快于 baseline
- e2e 上是当前自定义 paged 路径的稳定基线
- op 层在三类 case 上都进入了亚毫秒级到低毫秒级

### v6

- 当前默认 CUDA 实现是 `v6`
- correctness collect 模式主线已明确收敛到 `head_dim=64`
- e2e 上只比 `v5` 小幅领先
- op 层在 `qwen_like_b1_s128_h64` 上相对 `v5` 有更明显优势
- NCU 显示 `v6` 在当前 3 个 hd64 case 上都明显慢于 `official`
- 当前主要嫌疑已从“单纯算力不足”收窄到访存与调度效率问题

### official

- e2e 上仍是当前 smoke 样本的明显上界
- op 层和 `v6` 的差距不总是大，说明 e2e 差距不应直接等同于单 kernel 差距
- 当前仍应把它理解为对照路径，而不是与 paged-op 完全同构的单接口基线
- 当前 NCU 结果表明，在选定 hd64 case 上，`official` kernel 时长仍显著优于 `v6`

## 10. 分层结论

1. 当前统一采集链已经可以同时产出 `e2e`、`op`、`correctness` 和 `version summary` 的结构化输入。
2. correctness 已从 hard fail gate 收敛为 collect 模式：
   - `v6 + head_dim=64` 主线可用于后续性能分析
   - `head_dim=16/32` 当前应视为 `unsupported`
   - `head_dim=64` 敏感 case 当前表现为 `sensitive`，不是结构性错误
3. e2e smoke 上，`v5` 和 `v6` 基本持平，而 `official` 仍明显快于两者。
4. op 层结果显示，`v6` 并非在所有单接口 case 上都远落后于 `official`；因此 e2e 差距不能只看单一 op 均值。
5. 当前已采到 3 个 hd64 NCU case，`v6 / official` 的 kernel 时长倍率分别约为 `5.26x / 16.46x / 65.98x`，说明 kernel 级差距真实存在。
6. NCU 摘要已把问题方向收窄到访存与调度效率：
   - `eligible warps per scheduler` 很低
   - `underfilled grid` 明显
   - global access 放大与 shared bank conflict 风险持续存在
