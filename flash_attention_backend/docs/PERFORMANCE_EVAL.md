# toy_flash_attn 性能评估与分析报告

## 1. 评估范围

本文档只评估 attention backend 的端到端可运行性、正确性和性能表现。

本次报告依据 `analysis/perf_eval_results.json` 和 `analysis/perf_logs/*.log` 更新。当前 benchmark 脚本支持的版本集合由 `analysis/run_perf_eval.sh` 定义，并提供 `smoke`、`report`、`stress` 分级套件。本次已重跑 `smoke report`，日志覆盖 `baseline`、`v4`、`v5` 和 `official`；未采集 `v3`、`v4_fp32` 和 `stress`。

当前纳入报告分析的实现版本：

| 版本 | 本次数据状态 | 含义 | 入口 |
| --- | --- | --- | --- |
| baseline | 已采集 | Python reference 实现 | `TOY_FLASH_ATTN_USE=reference` |
| v4 | 已采集 | 当前 toy CUDA kernel | `TOY_FLASH_ATTN_USE=bf16`，`TOY_FLASH_ATTN_CUDA_VERSION=v4` |
| v5 | 已采集 | WMMA / Tensor Core toy CUDA kernel | `TOY_FLASH_ATTN_USE=bf16`，`TOY_FLASH_ATTN_CUDA_VERSION=v5` |
| official | 已采集 | vLLM 官方 FlashAttention backend | 见 `test_self_flash_attn_backend.py` 的 `_attention_config()` |

当前不讨论后续优化计划，只记录已有实现的评估口径、结果和原因分析。

## 2. 测试环境

环境信息来自 `perf_eval_results.json`。

| 项目 | 数值 |
| --- | --- |
| 数据生成时间 | 2026-04-23T01:41:36.816772+00:00 |
| GPU | NVIDIA GeForce RTX 2050 |
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |
| vLLM | 0.17.1 |
| Python | 3.10.12 |
| 测量 git commit | `541a77f073e7b8ed5bf84c1a620a076f7e6fa8b9` |
| 模型 | GPT-2, Qwen2.5-0.5B-Instruct |
| dtype | torch.bfloat16 |
| kv_cache_dtype | baseline/v4/v5: bfloat16; official: auto，实际解析为 torch.bfloat16 |
| max_model_len | 512 |
| batch size | 1, 4 |
| max tokens | 128, 512 |

## 3. 功能边界

| 能力 | baseline | v4 | v5 | official |
| --- | --- | --- | --- | --- |
| paged KV cache | 支持 | 支持 | 支持 | 支持 |
| varlen query | 支持 | 支持 | 支持 | 支持 |
| causal mask | 支持 | 支持 | 支持 | 支持 |
| sliding window | 支持 | 支持 | 支持 | 支持 |
| GQA | 支持 | 支持 | 支持 | 支持 |
| bf16 输入 | 支持 | 支持 | 支持 | 支持 |
| fp32 accumulator | PyTorch 行为 | 支持 | 支持 | 官方实现内部策略 |
| head_dim=64 | 支持 | 支持 | 支持 | 支持 |
| Tensor Core / WMMA | 不适用 | 未使用 | 使用 | 官方实现内部策略 |

说明：本节描述当前源码能力边界，不等同于本次 benchmark 是否已有有效性能数据。

## 4. 正确性评估

### 4.1 方法

正确性评估只回答一个问题：当前实现的输出是否能在指定容差下对齐 reference。

记录指标：

| 指标 | 含义 |
| --- | --- |
| max abs diff | 最大绝对误差 |
| mean abs diff | 平均绝对误差 |
| p99 max diff | 多轮测试中 max diff 的 p99 |
| pass rate | 指定阈值下的通过率 |

### 4.2 结果

本次 JSON 只包含端到端生成性能数据，没有包含 unittest 数值对拍数据。因此正确性结果不在本次报告中填充。

| 版本 | case | q_heads | kv_heads | head_dim | causal | window | threshold | pass rate | max diff | mean diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v4 | full attention | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 |
| v4 | causal | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 |
| v4 | GQA | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 |
| v5 | full attention | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 |
| v5 | causal | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 |
| v5 | GQA | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 |

## 5. 端到端性能评估

### 5.1 方法

端到端性能使用 `flash_attention_backend/test_self_flash_attn_backend.py` 运行。本次执行 `smoke report`：`smoke` 使用 128 token，`report` 使用 512 token；两者都保持 runner 中固定的 `max_model_len=512`。

记录指标：

| 指标 | 含义 |
| --- | --- |
| input toks/s | prefill 吞吐 |
| output toks/s | decode 吞吐 |
| output toks/s/request | 多 batch 下的单请求平均 decode 吞吐 |
| wall time | vLLM progress 中解析到的单步耗时 |
| finish reason | 生成结束原因 |

### 5.2 数据有效性

| case 组 | 数量 | 判定 |
| --- | --- | --- |
| baseline qwen b1 t128 | 1 | 有效 |
| GPT-2 b1 t128: v4/v5/official | 3 | 有效 |
| Qwen b1/b4 t128: v4/v5/official | 6 | 有效 |
| Qwen b1/b4 t512: v4/v5/official | 6 | 有效 |

所有本次解析到的 benchmark 都满足 `success: true`，且有效性能行均包含 `input_toks_per_s` 和 `output_toks_per_s`。

### 5.3 Smoke: 128 Token

| 模型 | 版本 | batch | input toks/s | output toks/s | output toks/s/request | wall time | finish reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | baseline | 1 | 1.14 | 18.19 | 18.19 | 7.04s | 未采集 |
| Qwen2.5-0.5B-Instruct | v4 | 1 | 1.53 | 24.50 | 24.50 | 5.23s | 未采集 |
| Qwen2.5-0.5B-Instruct | v5 | 1 | 3.57 | 57.11 | 57.11 | 2.24s | 未采集 |
| Qwen2.5-0.5B-Instruct | official | 1 | 5.31 | 84.91 | 84.91 | 1.51s | 未采集 |
| Qwen2.5-0.5B-Instruct | v4 | 4 | 3.99 | 60.03 | 15.01 | 2.13s | 未采集 |
| Qwen2.5-0.5B-Instruct | v5 | 4 | 13.52 | 203.63 | 50.91 | 2.51s | 未采集 |
| Qwen2.5-0.5B-Instruct | official | 4 | 21.00 | 316.21 | 79.05 | 1.61s | 未采集 |
| GPT-2 | v4 | 1 | 3.46 | 55.39 | 55.39 | 2.31s | 未采集 |
| GPT-2 | v5 | 1 | 8.95 | 143.25 | 143.25 | 未采集 | 未采集 |
| GPT-2 | official | 1 | 18.87 | 301.93 | 301.93 | 未采集 | 未采集 |

### 5.4 Report: 512 Token

| 模型 | 版本 | batch | input toks/s | output toks/s | output toks/s/request | wall time | finish reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | v4 | 1 | 0.19 | 12.04 | 12.04 | 41.87s | 未采集 |
| Qwen2.5-0.5B-Instruct | v5 | 1 | 0.81 | 51.09 | 51.09 | 9.87s | 未采集 |
| Qwen2.5-0.5B-Instruct | official | 1 | 1.35 | 85.26 | 85.26 | 5.91s | 未采集 |
| Qwen2.5-0.5B-Instruct | v4 | 4 | 0.45 | 22.12 | 5.53 | 18.99s | 未采集 |
| Qwen2.5-0.5B-Instruct | v5 | 4 | 2.85 | 168.67 | 42.17 | 2.99s | 未采集 |
| Qwen2.5-0.5B-Instruct | official | 4 | 5.38 | 266.06 | 66.52 | 1.58s | 未采集 |

### 5.5 本次端到端结果摘要

| 对比项 | 结果 |
| --- | --- |
| baseline Qwen batch=1 decode, 128 token | 18.19 toks/s，低于 v4/v5/official，符合 reference 只作正确性参考的定位 |
| v5 相比 v4，Qwen batch=1 decode, 128 token | v5 为 57.11 toks/s，约为 v4 的 2.33 倍 |
| v5 相比 v4，Qwen batch=4 decode, 128 token | v5 为 203.63 toks/s，约为 v4 的 3.39 倍 |
| v5 相比 v4，GPT-2 batch=1 decode, 128 token | v5 为 143.25 toks/s，约为 v4 的 2.59 倍 |
| official 相比 v5，Qwen batch=1 decode, 128 token | official 约为 v5 的 1.49 倍 |
| official 相比 v5，Qwen batch=4 decode, 128 token | official 约为 v5 的 1.55 倍 |
| v5 相比 v4，Qwen batch=1 decode, 512 token | v5 为 51.09 toks/s，约为 v4 的 4.24 倍 |
| v5 相比 v4，Qwen batch=4 decode, 512 token | v5 为 168.67 toks/s，约为 v4 的 7.63 倍 |
| official 相比 v5，Qwen batch=1 decode, 512 token | official 约为 v5 的 1.67 倍 |
| official 相比 v5，Qwen batch=4 decode, 512 token | official 约为 v5 的 1.58 倍 |
| Qwen batch=4 单请求 decode, 512 token | v4 为 5.53 toks/s/request，v5 为 42.17 toks/s/request，official 为 66.52 toks/s/request |
| v4 多 batch 总吞吐, 512 token | Qwen 从 12.04 toks/s 提升到 22.12 toks/s，约 1.84 倍 |
| v5 多 batch 总吞吐, 512 token | Qwen 从 51.09 toks/s 提升到 168.67 toks/s，约 3.30 倍 |
| official 多 batch 总吞吐, 512 token | Qwen 从 85.26 toks/s 提升到 266.06 toks/s，约 3.12 倍 |

## 6. 版本分析

### 6.1 baseline

baseline 的角色是正确性参考，不作为性能目标。

性能特征：

| 维度 | 说明 |
| --- | --- |
| 运行位置 | Python / PyTorch |
| 主要用途 | 对拍、dump、replay |
| 本次数据状态 | Qwen batch=1, 128 token 已采集 |

原因分析：

- baseline 保留了清晰的数学表达，方便定位 mask、GQA、paged KV 和 tail alignment 语义。
- baseline 没有针对 vLLM 真实推理链路做 kernel 级优化。
- 本次 baseline 能端到端生成，但 decode 吞吐低于 v4/v5/official。

### 6.2 v4

v4 的角色是当前 toy CUDA baseline。

性能特征：

| 维度 | 说明 |
| --- | --- |
| CUDA source | `toy_flash_attn/v4/flash_attn_func.cu` |
| storage dtype | bf16 |
| accumulator dtype | fp32 |
| GQA | 支持 |
| Tensor Core | 未使用 |
| 主要用途 | 验证 paged KV / GQA / online softmax / vLLM 接入 |

原因分析：

- v4 的外层 grid 是 `batch x q_chunk x q_head`，多 batch 会增加 block 数，有利于提升 GPU occupancy。
- v4 的 QK 阶段使用标量乘法和手写 reduction，没有使用 Tensor Core / MMA。
- v4 的 softmax 阶段使用手写 online reduction，同步点较多。
- v4 的 PV 阶段按 `head_off` 串行处理 head_dim，head_dim=64 时会产生明显循环开销。
- v4 保留了多个中间 shared tile，例如 `score/max/sum/softmax/qk_reduction/sv_reduction/out_reduction`，shared memory 占用较高。
- 在 512 token 的 Qwen case 中，v4 与 v5/official 的差距比 128 token 更明显。

### 6.3 v5

v5 的角色是 WMMA / Tensor Core toy CUDA kernel。

性能特征：

| 维度 | 说明 |
| --- | --- |
| CUDA source | `toy_flash_attn/v5/flash_attn_func.cu` |
| storage dtype | bf16 |
| accumulator dtype | fp32 |
| GQA | 支持 |
| Tensor Core | 使用 WMMA |
| 当前绑定 | `flash_attn_varlen_with_block_v5_64` |
| 本次数据状态 | 128 token 与 512 token 均已采集 |

原因分析：

- v5 的 QK 阶段使用 WMMA 表达 `Q @ K^T`，输入为 bf16，累加为 fp32。
- v5 的 softmax 阶段仍使用 fp32 计算 max/sum，随后将 softmax numerator 量化为 bf16 供 PV 的 WMMA 路径使用。
- v5 的 PV 阶段使用 WMMA 表达 `P @ V`，输入为 bf16，累加为 fp32。
- v5 的 chunk merge 和最终 out 累积仍保留 fp32 路径，最后写回 bf16。
- 在本次端到端数据中，v5 相比 v4 有明显提升；512 token 下提升幅度更大，但仍低于 official。

### 6.4 official

official 的角色是生产级性能基线。

性能特征：

| 维度 | 说明 |
| --- | --- |
| backend | vLLM 官方 `FLASH_ATTN` |
| kv_cache_dtype | `auto`，本次解析为 torch.bfloat16 |
| 主要用途 | 端到端性能基线 |
| 性能预期 | 快 |

原因分析：

- official backend 使用成熟的 FlashAttention 实现。
- official backend 通常包含更好的 tile 策略、memory pipeline 和 kernel 调度。
- official backend 可作为性能上界参考，但不直接解释 toy kernel 内部瓶颈。

## 7. 结论

| 问题 | 结论 |
| --- | --- |
| baseline 是否能端到端运行 | 能。本次 Qwen batch=1, 128 token 成功生成，但只适合作正确性参考，不作为性能目标。 |
| v4 是否能端到端运行 | 能。本次 GPT-2 和 Qwen2.5-0.5B-Instruct 的 v4 case 均成功生成。 |
| v5 是否能端到端运行 | 能。本次 GPT-2 和 Qwen2.5-0.5B-Instruct 的 v5 case 均成功生成。 |
| v5 是否提升 toy backend 吞吐 | 能。Qwen 512 token 下，batch=1 decode 从 v4 的 12.04 toks/s 提升到 v5 的 51.09 toks/s，batch=4 从 22.12 toks/s 提升到 168.67 toks/s。 |
| toy backend 与 official 的主要差距 | v5 已缩小差距，但仍低于 official；Qwen 512 token 下 official 约为 v5 的 1.67 倍（batch=1）和 1.58 倍（batch=4）。 |
| v4 当前主要瓶颈 | 标量 QK/PV 计算、未使用 Tensor Core、shared memory 中间 tile 较多、同步点和 head_dim 串行循环开销较大。 |
