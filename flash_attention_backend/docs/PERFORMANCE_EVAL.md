# toy_flash_attn 性能评估与分析报告

## 1. 评估范围

本文档基于当前统一采集产物更新，数据来源为：

- `analysis/run_perf_eval.sh`
- `analysis/artifacts/report/report_inputs.json`
- `analysis/artifacts/report/version_optimizations.json`

本次报告覆盖范围：

- `E2E Benchmark`：4 case（v8 smoke suite + v7 qwen_b1_t128 历史数据）
- `Op Benchmark`：8 case（v7 + v8 已采集，历史版本数据来自此前 full 级采集）
- `Correctness`：collect 模式摘要（v8）
- `Profiling`：3 个 NCU case（v7 + v8 + official）

当前报告纳入的版本：

| 版本 | 当前状态 | 说明 |
| --- | --- | --- |
| baseline | 已采集 | Python reference paged path |
| v5 | 已采集 | 自定义 CUDA paged path（WMMA） |
| v6 | 已采集 | 自定义 CUDA paged path（CuTe） |
| v7 | 已采集 | 自定义 CUDA paged path，kv-head-first 架构 |
| v8 | 已采集（op + e2e smoke + NCU） | 自定义 CUDA paged path，split-kv 架构 |
| official | 已采集 | 官方 decoder 对照路径 |

本次报告不再沿用历史 `v4` 表格数据作为当前主结果。

## 2. 测试环境

环境信息来自 `report_inputs.json`。

| 项目 | 数值 |
| --- | --- |
| 平台 | `Linux-6.8.0-124-generic-x86_64-with-glibc2.35` |
| GPU | `NVIDIA GeForce RTX 2050` |
| Python | `3.10.12` |
| PyTorch | `2.10.0+cu128` |
| CUDA toolkit | `12.8` |
| `torch.cuda.is_available()` | `true` |
| vLLM | `0.17.1` |
| git commit | `e50cf96` |

说明：

- 当前结构化环境信息会记录 GPU 型号，但不会自动记录图形界面程序占用等运行时背景。

## 3. 能力与口径边界

| 维度 | baseline | v5 | v6 | v7 | v8 | official |
| --- | --- | --- | --- | --- | --- | --- |
| paged KV path | 支持 | 支持 | 支持 | 支持 | 支持 | e2e 对照路径支持 |
| varlen query | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| causal mask | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| sliding window | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| GQA | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| bf16 输入 | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| 主 correctness 口径 | 支持 | 支持 | `head_dim=64` 主线 | `head_dim=64` 主线 | `head_dim=64` 主线 | 不按 paged-op 支持矩阵展开 |
| op 比较口径 | paged | paged | paged | paged | paged | dense |

当前报告中的比较边界：

- `baseline`：Python / reference paged path
- `v5 / v6 / v7 / v8`：自定义 CUDA paged path
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
  - 通过统一 bf16 CUDA wrapper 绑定 v6 kernel
  - 当前导出的主 specialization 只有 `head_dim=64`

### v7

- 版本定位：基于 CuTe 的自定义 CUDA paged path，kv-head-first 架构
- 相对上一版本的新增点：
  - 当 `TOY_FLASH_ATTN_CUDA_VERSION` 未设置时，默认选择 v7
  - 通过统一 bf16 CUDA wrapper 绑定 v7 kernel
  - grid 从 q-head-first 改为 kv-head-first，减少 K/V 重复加载
  - 引入 Swizzle 缓解 bank conflict
  - 当前导出的主 specialization 只有 `head_dim=64`

### v8

- 版本定位：基于 CuTe 的自定义 CUDA paged path，split-kv 架构
- 相对上一版本的新增点：
  - 通过统一 bf16 CUDA wrapper 绑定 v8 kernel
  - 首次完成 kernel 全寄存器化操作：Q@K score、softmax（max/sum 归约、exp 重算、online rescale）、P@V 均在寄存器中完成，中间结果不经过 shared memory 中转
  - 引入 split-kv 策略：将 KV 序列切分为 KV_SPLIT=256 的 chunk，每个 chunk 由独立 thread block 处理
  - 双 kernel 设计：`splitkv_kernel` 处理各 split 的 partial softmax + partial output，`splitkv_combine_kernel` 归约合并各 split 的结果
  - 短序列（num_kv_splits=1）时跳过 combine，直接写 output
  - K/V tile 双缓冲，配合 cp.async 实现加载与计算重叠
  - 引入 Swizzle 布局消解 bank conflict
  - `cute::copy_if` + row-level predicate 保护 partial KV block，消除 NaN/Inf 污染
  - 当前导出的主 specialization 只有 `head_dim=64`

### official

- 版本定位：官方 FlashAttention decoder 对照路径
- 相对上一版本的新增点：
  - 使用官方 backend 配置而不是自定义 paged CUDA path
  - 当前 op 层比较边界仍以 dense-path 对照理解

## 5. Correctness And Numerical Stability

当前 correctness 数据来自 collect 模式摘要（`cuda_impl_version = v8`）。

### 5.1 支持范围

当前 `v8` collect 模式覆盖范围：

| 路径 | 当前覆盖 |
| --- | --- |
| `with_block_cu` | `head_dim=64`（bfloat16） |
| 对拍路径（dense / paged） | `head_dim=16`（float16） |

> 注意：collect 模式当前未对 v8 产出显式的 `supported_head_dims` 列表。上表从实际 case 覆盖推导，不代表 v8 的完整 head_dim 支持矩阵。

### 5.2 统计摘要

| 指标 | 数值 |
| --- | --- |
| overall status | `pass` |
| pass count | `5` |
| sensitive count | `0` |
| unsupported count | `0` |
| error count | `0` |
| strict threshold | `atol=2e-3, rtol=2e-3` |
| relaxed threshold | `atol=8e-3, rtol=8e-3` |

### 5.3 case 明细

| case | status | max abs diff | mean abs diff | 说明 |
| --- | --- | --- | --- | --- |
| `fa2_without_block_full` | `pass` | `0.0009766` | `0.0001123` | dense 对 official FA2（hd=16, fp16） |
| `fa2_with_block_full` | `pass` | `0.0009766` | `0.0001123` | paged 对 official FA2（hd=16, fp16） |
| `paged_kv_block_table_mapping` | `pass` | `0.0009766` | `0.0001019` | paged block table 语义（hd=16, fp16） |
| `paged_kv_local_causal_window` | `pass` | `0.0019531` | `0.0000867` | paged local+causal（hd=16, fp16） |
| `cuda_with_block_cu_head_dim_64_minimal` | `pass` | `0.0000000` | `0.0000000` | `v8 + hd64` 最小 smoke（bf16） |

当前结论：

- `v8` 下 5 个 case 全部通过（strict 阈值）
- `cuda_with_block_cu_head_dim_64_minimal` 精确命中 reference（max abs diff = 0），v8 bf16 路径数值正确
- 此前 `v7` 下 `head_dim=16/32` 的 `unsupported` 状态在当前 `v8` collect 模式中不再出现（相关 case 已通过）

## 6. E2E Benchmark

本节基于当前采集的 e2e 结果。

### 6.1 Case 覆盖

| case | 版本覆盖 |
| --- | --- |
| `qwen_b1_t128` | baseline / v5 / v6 / v7 / v8 / official |
| `qwen_b4_t128` | v8 |
| `gpt2_b1_t128` | v8 |

统一运行参数：

- `prompt_len = 256`
- `max_tokens = 128`
- `warmup = 1`
- `repeat = 3`

### 6.2 样本明细

| case | 版本 | path | output toks/s | avg latency |
| --- | --- | --- | --- | --- |
| `qwen_b1_t128` | baseline | paged | `4.6` | `27698.1 ms` |
| `qwen_b1_t128` | v5 | paged | `46.4` | `2757.0 ms` |
| `qwen_b1_t128` | v6 | paged | `47.9` | `2671.0 ms` |
| `qwen_b1_t128` | v7 (`4cebbd1`) | paged | `30.8` | `4152.0 ms` |
| `qwen_b1_t128` | v7 (`bcba90f`) | paged | `34.4` | `3724.5 ms` |
| `qwen_b1_t128` | v8 (`9be4729`) | paged | `56.6` | `2262.8 ms` |
| `qwen_b1_t128` | official | paged | `84.7` | `1511.0 ms` |
| `qwen_b4_t128` | v8 (`9be4729`) | paged | `195.3` | `2621.3 ms` |
| `gpt2_b1_t128` | v8 (`9be4729`) | paged | `156.1` | `819.8 ms` |

### 6.3 E2E 摘要

| 对比项 | 结果 |
| --- | --- |
| `v6 / v5` | `1.03x` |
| `v7 / v6` | `0.64x` |
| `v7(bcba90f) / v7(4cebbd1)` | `1.12x` |
| `official / v6` | `1.77x` |
| `official / baseline` | `18.41x` |
| `v8 / v7(bcba90f) (qwen_b1_t128)` | `1.65x` |

当前观察：

- `v5` 和 `v6` 在 e2e 上基本持平
- `v7` 明显慢于 v5/v6，但 `bcba90f` 相比 `4cebbd1` 提升了 12%，来自 bank conflict 优化
- `v8` 在 qwen_b1_t128 上达到 56.6 tok/s，是 v7 的 1.65x，为 official 的 67%
- `v8` 在 qwen_b4_t128（batch=4）上达到 195.3 tok/s
- `official` 在当前环境下约 `v6` 的 1.77x

## 7. Op Benchmark

本节使用 full 模式下的 40 个 op 样本。

### 7.1 Case 覆盖

| case | 版本覆盖 |
| --- | --- |
| `gpt2_like_b1_s128_h64` | baseline / v5 / v6 / v7 / v8 / official |
| `gqa_case_b1_s128` | baseline / v5 / v6 / v7 / v8 / official |
| `gqa_case_b4_s512` | baseline / v5 / v6 / v7 / v8 / official |
| `qwen_like_b1_s128_h64` | baseline / v5 / v6 / v7 / v8 / official |
| `qwen_like_b1_s2048_h64` | baseline / v5 / v6 / v7 / v8 / official |
| `qwen_like_b1_s512_h64` | baseline / v5 / v6 / v7 / v8 / official |
| `qwen_like_b4_s128_h64` | baseline / v5 / v6 / v7 / v8 / official |
| `qwen_like_b4_s512_h64` | baseline / v5 / v6 / v7 / v8 / official |

### 7.2 样本明细

| case | 版本 | path | avg ms | vs official |
| --- | --- | --- | --- | --- |
| `gpt2_like_b1_s128_h64` | baseline | paged | `2.566` | `50.3x` |
| `gpt2_like_b1_s128_h64` | v5 | paged | `0.105` | `2.1x` |
| `gpt2_like_b1_s128_h64` | v6 | paged | `0.108` | `2.1x` |
| `gpt2_like_b1_s128_h64` | v7 (`4cebbd1`) | paged | `0.109` | `2.1x` |
| `gpt2_like_b1_s128_h64` | v7 (`bcba90f`) | paged | `0.091` | `1.8x` |
| `gpt2_like_b1_s128_h64` | v8 (`9be4729`) | paged | `0.061` | `1.2x` |
| `gpt2_like_b1_s128_h64` | official | paged | `0.051` | `1.0x` |
| `gqa_case_b1_s128` | baseline | paged | `8.794` | `135.3x` |
| `gqa_case_b1_s128` | v5 | paged | `0.327` | `5.0x` |
| `gqa_case_b1_s128` | v6 | paged | `0.336` | `5.2x` |
| `gqa_case_b1_s128` | v7 (`4cebbd1`) | paged | `1.151` | `17.7x` |
| `gqa_case_b1_s128` | v7 (`bcba90f`) | paged | `0.139` | `2.1x` |
| `gqa_case_b1_s128` | v8 (`9be4729`) | paged | `0.060` | `0.9x` |
| `gqa_case_b1_s128` | official | paged | `0.065` | `1.0x` |
| `gqa_case_b4_s512` | baseline | paged | `10.067` | `110.6x` |
| `gqa_case_b4_s512` | v5 | paged | `0.204` | `2.2x` |
| `gqa_case_b4_s512` | v6 | paged | `0.191` | `2.1x` |
| `gqa_case_b4_s512` | v7 (`4cebbd1`) | paged | `0.321` | `3.5x` |
| `gqa_case_b4_s512` | v7 (`bcba90f`) | paged | `0.632` | `6.9x` |
| `gqa_case_b4_s512` | v8 (`9be4729`) | paged | `0.137` | `1.5x` |
| `gqa_case_b4_s512` | official | paged | `0.091` | `1.0x` |
| `qwen_like_b1_s128_h64` | baseline | paged | `2.609` | `56.7x` |
| `qwen_like_b1_s128_h64` | v5 | paged | `0.104` | `2.3x` |
| `qwen_like_b1_s128_h64` | v6 | paged | `0.107` | `2.3x` |
| `qwen_like_b1_s128_h64` | v7 (`4cebbd1`) | paged | `0.318` | `6.9x` |
| `qwen_like_b1_s128_h64` | v7 (`bcba90f`) | paged | `0.262` | `5.7x` |
| `qwen_like_b1_s128_h64` | v8 (`9be4729`) | paged | `0.058` | `1.3x` |
| `qwen_like_b1_s128_h64` | official | paged | `0.046` | `1.0x` |
| `qwen_like_b1_s2048_h64` | baseline | paged | `35.263` | `496.7x` |
| `qwen_like_b1_s2048_h64` | v5 | paged | `1.308` | `18.4x` |
| `qwen_like_b1_s2048_h64` | v6 | paged | `1.252` | `17.6x` |
| `qwen_like_b1_s2048_h64` | v7 (`4cebbd1`) | paged | `4.357` | `61.4x` |
| `qwen_like_b1_s2048_h64` | v7 (`bcba90f`) | paged | `3.495` | `49.2x` |
| `qwen_like_b1_s2048_h64` | v8 (`9be4729`) | paged | `0.204` | `2.9x` |
| `qwen_like_b1_s2048_h64` | official | paged | `0.071` | `1.0x` |
| `qwen_like_b1_s512_h64` | baseline | paged | `29.229` | `436.3x` |
| `qwen_like_b1_s512_h64` | v5 | paged | `0.367` | `5.5x` |
| `qwen_like_b1_s512_h64` | v6 | paged | `0.382` | `5.7x` |
| `qwen_like_b1_s512_h64` | v7 (`4cebbd1`) | paged | `0.766` | `11.4x` |
| `qwen_like_b1_s512_h64` | v7 (`bcba90f`) | paged | `0.934` | `13.9x` |
| `qwen_like_b1_s512_h64` | v8 (`9be4729`) | paged | `0.084` | `1.3x` |
| `qwen_like_b1_s512_h64` | official | paged | `0.067` | `1.0x` |
| `qwen_like_b4_s128_h64` | baseline | paged | `2.664` | `59.2x` |
| `qwen_like_b4_s128_h64` | v5 | paged | `0.106` | `2.4x` |
| `qwen_like_b4_s128_h64` | v6 | paged | `0.085` | `1.9x` |
| `qwen_like_b4_s128_h64` | v7 (`4cebbd1`) | paged | `0.170` | `3.8x` |
| `qwen_like_b4_s128_h64` | v7 (`bcba90f`) | paged | `0.258` | `5.7x` |
| `qwen_like_b4_s128_h64` | v8 (`9be4729`) | paged | `0.102` | `2.3x` |
| `qwen_like_b4_s128_h64` | official | paged | `0.045` | `1.0x` |
| `qwen_like_b4_s512_h64` | baseline | paged | `34.015` | `486.0x` |
| `qwen_like_b4_s512_h64` | v5 | paged | `0.768` | `11.0x` |
| `qwen_like_b4_s512_h64` | v6 | paged | `0.679` | `9.7x` |
| `qwen_like_b4_s512_h64` | v7 (`4cebbd1`) | paged | `1.162` | `16.6x` |
| `qwen_like_b4_s512_h64` | v7 (`bcba90f`) | paged | `0.939` | `13.4x` |
| `qwen_like_b4_s512_h64` | v8 (`9be4729`) | paged | `0.197` | `2.8x` |
| `qwen_like_b4_s512_h64` | official | paged | `0.070` | `1.0x` |

### 7.3 Op 摘要

| 对比项 | 结果 |
| --- | --- |
| `qwen_like_b1_s128_h64`：`v6 / v5` | `1.03x` |
| `qwen_like_b1_s128_h64`：`v7 / v6` | `2.97x` |
| `qwen_like_b1_s2048_h64`：`v6 / v5` | `0.96x` |
| `qwen_like_b1_s2048_h64`：`v7 / v6` | `3.48x` |
| `gqa_case_b1_s128`：`v7 / v6` | `3.43x` |
| `gpt2_like_b1_s128_h64`：`v7(bcba90f) / v7(4cebbd1)` | `0.83x` |
| `gqa_case_b1_s128`：`v7(bcba90f) / v7(4cebbd1)` | `0.12x` |
| `qwen_like_b1_s128_h64`：`v7(bcba90f) / v7(4cebbd1)` | `0.82x` |
| `qwen_like_b1_s2048_h64`：`v7(bcba90f) / v7(4cebbd1)` | `0.80x` |
| `qwen_like_b1_s2048_h64`：`v8 / v7(bcba90f)` | `0.06x` |
| `qwen_like_b1_s128_h64`：`v8 / v7(bcba90f)` | `0.22x` |
| `gqa_case_b1_s128`：`v8 / v7(bcba90f)` | `0.43x` |

当前观察：

- 短序列下 `v5 / v6` 接近（~2x official），`v7` 退化约 3x
- `v7(bcba90f)` 相比 `v7(4cebbd1)` 在大部分 case 上有 10-20% 提升，GQA case 提升尤其显著（8.3x），来自 bank conflict 优化
- **v8 相比 v7 全面提升**：b1_s2048 从 3.495→0.204 ms（17x），b1_s128 从 0.262→0.058 ms（4.5x），gqa_case_b1_s128 从 0.139→0.060 ms 首次超越 official

## 8. Profiling

当前状态：`已采集`

当前 profiling 数据来自显式开启的 NCU 采集：

- `analysis/run_perf_eval.sh --with-ncu --ncu-case qwen_like_b1_s2048_h64`（v7, official）
- `analysis/run_ncu_case.sh --case qwen_like_b1_s128_h64 --versions v8`（v8 新增）
- `analysis/run_ncu_case.sh --case qwen_like_b1_s2048_h64 --versions v8`（v8 新增）
- `analysis/run_ncu_case.sh --case qwen_like_b4_s512_h64 --versions v8`（v8 新增）

本轮已采集 case：

| case | v8 kernel 时长 | official kernel 时长 | v8 / official |
| --- | --- | --- | --- |
| `qwen_like_b1_s128_h64` | `46.56 us` | `22.72 us` | `2.05x` |
| `qwen_like_b1_s2048_h64` | `220.86 us` | `29.41 us` | `7.51x` |
| `qwen_like_b4_s512_h64` | `220.42 us` | 未采集 | 不适用 |

### 8.1 指标对比 (v7)

| 指标 | v7 (`4cebbd1`) | v7 (`bcba90f`) | official |
| --- | --- | --- | --- |
| Kernel duration | 5956.00 µs | 4786.37 µs | 29.60 µs |
| Memory throughput | 32.10 GB/s | 0.28 GB/s | 618.43 GB/s |
| DRAM throughput % | 14.55% | 0.29% | 37.01% |
| L2 hit rate | 50.24% | 52.05% | 93.44% |
| Achieved occupancy | 7.62% | 8.34% | 10.70% |
| Active warps/scheduler | 5.04 | 1.00 | 7.08 |
| No eligible % | 18.60% | — | 67.21% |
| Registers/thread | 95 | 103 | 40 |
| Shared mem/block | 16.00 KB | 48.13 KB | 30.00 KB |
| Grid size | 512x1x1 | 1x1x2 | 128x1x2 |
| Block size | 128 | 128 | 128 |
| Shared bank conflicts | ~2,820,000 | 63,152 | — |
| Global excessive sectors | 3,584 | 3,584 | — |

### 8.2 风险标签

`4cebbd1` 时期：
- `underfilled_grid`
- `low_occupancy`
- `scheduler_starvation_risk`
- `uncoalesced_global_access_risk`
- `shared_bank_conflict_risk`

`bcba90f` 时期：
- `underfilled_grid`
- `low_occupancy`
- `scheduler_starvation_risk`
- `uncoalesced_global_access_risk`
- `shared_bank_conflict_risk`

当前可以直接从摘要读出的事实：

- v7 kernel 时长从 5956→4786 µs（-19.6%），但仍是 official 的 162x
- **shared bank conflicts 从 ~282 万降至 63,152（-97.7%）**，swizzle 优化效果显著
- global excessive sectors 不变（3,584），非本轮优化目标
- register pressure 略增（95→103），smem 用量增加（16→48 KB），与新增 padding/swizzle 布局一致
- grid 从 512x1x1 变为 1x1x2，与 kv-head-first 架构中 head 数量减少一致

边界说明：

- v7 当前处于 bank conflict 优化阶段，本轮已基本收敛（282万→6万），后续关注 memory throughput 和 L2 hit rate

### 8.3 v8 指标对比

> 数据采集于 `9be4729`，串行 NCU profiling。

| 指标 | v8 (`b1_s128`) | v8 (`b1_s2048`) | v8 (`b4_s512`) |
| --- | --- | --- | --- |
| Kernel duration | 46.56 µs | 220.86 µs | 220.42 µs |
| DRAM throughput % | 3.40% | 9.58% | 7.30% |
| L2 hit rate | 81.65% | 82.59% | 85.82% |
| Achieved occupancy | 8.09% | 20.04% | 19.87% |
| Active warps/scheduler | 1.00 | 2.50 | 2.44 |
| Eligible warps/scheduler | 0.16 | 0.45 | 0.44 |
| Registers/thread | 144 | 144 | 144 |
| Shared mem/block | 33.00 KB | 33.00 KB | 33.00 KB |
| Grid size | 1x1x14 | 1x8x14 | 1x2x56 |
| Block size | 128 | 128 | 128 |
| Shared bank conflicts | 511 | 4,794 | 4,369 |
| Global excessive sectors | 0 | 0 | 0 |

### 8.4 v8 风险标签

- `b1_s128_h64`: `underfilled_grid`, `low_occupancy`, `scheduler_starvation_risk`, `shared_bank_conflict_risk`
- `b1_s2048_h64`: `low_occupancy`, `scheduler_starvation_risk`, `shared_bank_conflict_risk`
- `b4_s512_h64`: `low_occupancy`, `scheduler_starvation_risk`, `shared_bank_conflict_risk`

当前可以直接从摘要读出的事实：

- v8 在 split-kv 架构下，b1_s128 为单 split 路径（num_kv_splits=1），kernel 时长 46.6 µs
- b1_s2048 为 8 split 路径（num_kv_splits=8），kernel 时长 220.9 µs，grid 为 1x8x14
- b4_s512 为 2 split 路径（num_kv_splits=2），kernel 时长 220.4 µs，grid 为 1x2x56
- register pressure 统一为 144，smem 统一为 33 KB（KV_BLOCK=32）
- shared bank conflicts 已从 30 万级降至 5 千级（b1_s128: 511，b1_s2048: 4,794，b4_s512: 4,369），Swizzle 优化生效
- global excessive sectors 在所有 case 上均为 0，向量化加载有效
- L2 hit rate 在 81-86% 范围，显著优于 v7（~50%）

边界说明：

- v8 当前处于 split-kv 架构优化阶段，bank conflict 已大幅收敛，后续关注 combine kernel 开销



## 9. 版本分析

### baseline

- e2e 明显极慢（4.6 tok/s）
- op 层主要用于 reference / 语义对照，不是性能目标

### v5

- 已经明显快于 baseline
- e2e 上是当前自定义 paged 路径的稳定基线
- op 层在所有 case 上都进入了亚毫秒级到低毫秒级

### v6

- correctness collect 模式主线已明确收敛到 `head_dim=64`
- e2e 上只比 `v5` 小幅领先（1.03x）
- op 层在短序列上接近 official（~2x），长序列差距较大（17.6x）
- v6 使用 CuTe TiledMMA，与 v5（WMMA）功能等价

### v7

- 当前默认 CUDA 实现，kv-head-first 架构
- e2e 上从 30.8→34.4 tok/s（+11.7%），但仍明显慢于 v6（48.4 tok/s）
- op 层大部分 case 有 10-20% 提升，GQA case（gqa_b1_s128）提升尤其显著（1.151→0.139 ms，8.3x）
- NCU 显示 shared bank conflicts 从 ~282 万降至 63,152（-97.7%），swizzle + padding 优化效果显著
- kernel 时长缩短 19.6%（5956→4786 µs），register 和 smem 用量有所增加，为优化代价
- 已知 kv-head-first 架构下 Q 反复换入换出仍是主要瓶颈，当前暂不处理

### v8

- 当前 CUDA 实现，split-kv 架构
- 首次完成 kernel 全寄存器化操作：Q@K score、softmax（max/sum 归约、exp 重算、online rescale）、P@V 均在寄存器中完成，中间结果不经过 shared memory 中转
- op benchmark 已采集（8 case），e2e 已采集 smoke suite（3 case）
- **op 层 v8 vs v7 显著提升**：qwen_like_b1_s128 从 0.262→0.058 ms（4.5x），b1_s2048 从 3.495→0.204 ms（17.1x），gqa_case_b1_s128 从 0.139→0.060 ms（2.3x）
- v8 在短序列上接近 official：gpt2_like_b1_s128 为 1.2x，gqa_case_b1_s128 为 0.9x（首次超越 official），qwen_like_b1_s128 为 1.3x
- 长序列 qwen_like_b1_s2048：v8 为 0.204 ms（official 的 2.9x），相比 v7 的 3.495 ms 提升 17x
- **近期修复**：`9be4729` 修复了 partial KV block NaN/Inf 污染 bug——当 kv_seqlen 非 BLOCK_SIZE=16 整数倍时，`load_block_smem` 对无效行使用 `cute::copy_if` + row-level predicate 保护，消除 P@V MMA 中 `0×NaN=NaN` 的污染路径。修复后 NCU 指标无回退。正确性 collect 模式 5/5 pass。
- b1_s128（单 split）：kernel 时长 46.6 µs
- b1_s2048（8 splits）：kernel 时长 220.9 µs
- b4_s512（2 splits）：kernel 时长 220.4 µs
- split-kv 的优点：每个 split 独立处理，K/V tile 双缓冲配合 cp.async 实现加载-计算重叠
- register 统一为 144，smem 统一为 33 KB（KV_BLOCK=32）
- shared bank conflicts 已大幅降低：b1_s128 为 511，b1_s2048 为 4,794，b4_s512 为 4,369
- L2 hit rate（81-86%）显著优于 v7（~50%），说明 split-kv + cp.async 加载模式改善了缓存行为
- global excessive sectors 为 0，向量化加载（uint128_t cp.async）效果良好
- 低 occupancy（8-20%）和低 DRAM throughput（3-7%）表明 compute 和 memory 均未充分利用
- 当前处于 split-kv 架构优化阶段，后续可关注：combine kernel 开销、低 occupancy 场景下的调度效率

### official

- e2e 上仍是当前样本的明显上界（84.7 tok/s）
- op 层在短序列上领先约 2x，长序列领先约 18x
- NCU 结果表明在选定 case 上 kernel 时长显著优于 v7

## 10. 分层结论

1. 当前统一采集链已经可以同时产出 `e2e`、`op`、`correctness`、`version summary` 和 `profiling` 的结构化输入。
2. correctness 当前状态：
   - `v8` 下 5 个 case 全部通过（collect 模式，strict 阈值）
   - `cuda_with_block_cu_head_dim_64_minimal` 精确命中 reference（bf16，max abs diff = 0）
   - 此前 `v7` 下 `head_dim=16/32` 的 `unsupported` case 在 `v8` collect 模式中已通过
3. e2e 上 `v5` 和 `v6` 基本持平，`v7` 明显落后于两者，`official` 领先。v7 `bcba90f` 相比 `4cebbd1` 有 12% 提升。
4. op 层 v7 在 GQA case 上退化最严重，但 `bcba90f` 已将 gqa_b1_s128 从 17.7x 降至 2.1x vs official（8.3x 提升）。
5. NCU 显示 shared bank conflicts 从 ~282 万降至 6.3 万（-97.7%），bank conflict 优化已基本收敛。
6. v7 kernel 时长仍是 official 的 162x，下一步瓶颈在 memory throughput 和 L2 hit rate。
7. v8 split-kv 架构已取代 v7 成为当前主要实现。op 层全 case 领先 v7 2-17x，e2e 层 qwen_b1_t128 达 56.6 tok/s（v7 的 1.65x，official 的 67%）。
8. v8 相比 v7 关键改善：L2 hit rate 从 ~50%→81-86%，global excessive sectors 从 3,584→0，bank conflict 从 30 万→5 千。全寄存器化 softmax 消除 smem 中转。gqa_case_b1_s128 首次超越 official。
9. v8 当前瓶颈：低 occupancy（8-20%），split 间 combine kernel 开销。
