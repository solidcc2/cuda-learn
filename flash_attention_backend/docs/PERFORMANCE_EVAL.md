# toy_flash_attn 性能评估与分析报告

## 1. 评估范围

本文档只评估 attention backend 的端到端可运行性、正确性和性能表现。

本次报告依据 `analysis/perf_eval_results.json` 和 `analysis/perf_logs/*.log` 更新。当前 benchmark 脚本支持的版本集合由 `analysis/run_perf_eval.sh` 定义；本次已有日志覆盖 `baseline`、`v4`、`v5` 和 `official`，未包含 `v3`、`v4_fp32` 的有效实测日志。

当前纳入报告分析的实现版本：

| 版本 | 本次数据状态 | 含义 | 入口 |
| --- | --- | --- | --- |
| baseline | 无有效数据 | Python reference 实现 | `TOY_FLASH_ATTN_USE=reference` |
| v4 | 已采集 | 当前 toy CUDA kernel | `TOY_FLASH_ATTN_USE=bf16`，`TOY_FLASH_ATTN_CUDA_VERSION=v4` |
| v5 | 已采集 | WMMA / Tensor Core toy CUDA kernel | `TOY_FLASH_ATTN_USE=bf16`，`TOY_FLASH_ATTN_CUDA_VERSION=v5` |
| official | 已采集 | vLLM 官方 FlashAttention backend | 见 `test_self_flash_attn_backend.py` 的 `_attention_config()` |

当前不讨论后续优化计划，只记录已有实现的评估口径、结果和原因分析。

## 2. 测试环境

环境信息来自 `perf_eval_results.json`。

| 项目 | 数值 |
| --- | --- |
| 数据生成时间 | 2026-04-22T07:51:21Z |
| GPU | NVIDIA GeForce RTX 2050 |
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |
| vLLM | 0.17.1 |
| Python | 3.10.12 |
| 测量 git commit | `7f27dbaed7d2947bd14232088261211c607fbec4` |
| 模型 | GPT-2, Qwen2.5-0.5B-Instruct |
| dtype | torch.bfloat16 |
| kv_cache_dtype | v4/v5: bfloat16; official: auto，实际解析为 torch.bfloat16 |
| max_model_len | 512 |
| batch size | 1, 4 |
| max tokens | 128 |

## 3. 功能边界

| 能力 | baseline | v4 | v5 | official |
| --- | --- | --- | --- | --- |
| paged KV cache | 支持 | 支持 | 支持 | 支持 |
| varlen query | 支持 | 支持 | 支持 | 支持 |
| causal mask | 支持 | 支持 | 支持 | 支持 |
| sliding window | 支持 | 支持 | 支持 | 支持 |
| GQA | 支持 | 支持 | 支持 | 支持 |
| bf16 输入 | 不适用 | 支持 | 支持 | 支持 |
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

端到端性能使用 `flash_attention_backend/test_self_flash_attn_backend.py` 运行。

记录指标：

| 指标 | 含义 |
| --- | --- |
| input toks/s | prefill 吞吐 |
| output toks/s | decode 吞吐 |
| output toks/s/request | 多 batch 下的单请求平均 decode 吞吐 |
| wall time | vLLM progress 中解析到的单步耗时 |
| finish reason | 生成结束原因 |

### 5.2 数据有效性

| case | JSON success | prompt/generated | 性能字段 | 判定 |
| --- | --- | --- | --- | --- |
| baseline qwen b1 t128 | true | 0 / 0 | 空 | 无有效数据；raw log 只有 `env -u` shell 错误 |
| v4 gpt2 b1 t128 | true | 1 / 1 | 有效 | 有效 |
| v4 qwen b1 t128 | true | 1 / 1 | 有效 | 有效 |
| v4 qwen b4 t128 | true | 4 / 4 | 有效 | 有效 |
| v5 gpt2 b1 t128 | true | 1 / 1 | 有效 | 有效 |
| v5 qwen b1 t128 | true | 1 / 1 | 有效 | 有效 |
| v5 qwen b4 t128 | true | 4 / 4 | 有效 | 有效 |
| official gpt2 b1 t128 | true | 1 / 1 | 有效 | 有效 |
| official qwen b1 t128 | true | 1 / 1 | 有效 | 有效 |
| official qwen b4 t128 | true | 4 / 4 | 有效 | 有效 |

### 5.3 单 batch 结果

| 模型 | 版本 | batch | max tokens | input toks/s | output toks/s | wall time | finish reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 | v4 | 1 | 128 | 3.52 | 56.29 | 2.27s | 未采集 |
| GPT-2 | v5 | 1 | 128 | 9.54 | 152.58 | 未采集 | 未采集 |
| GPT-2 | official | 1 | 128 | 19.15 | 306.43 | 未采集 | 未采集 |
| Qwen2.5-0.5B-Instruct | v4 | 1 | 128 | 1.54 | 24.62 | 5.20s | 未采集 |
| Qwen2.5-0.5B-Instruct | v5 | 1 | 128 | 3.53 | 56.40 | 2.27s | 未采集 |
| Qwen2.5-0.5B-Instruct | official | 1 | 128 | 5.32 | 85.19 | 1.50s | 未采集 |

### 5.4 多 batch 结果

| 模型 | 版本 | batch | max tokens | input toks/s | output toks/s | output toks/s/request | wall time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | v4 | 1 | 128 | 1.54 | 24.62 | 24.62 | 5.20s |
| Qwen2.5-0.5B-Instruct | v4 | 4 | 128 | 3.99 | 60.14 | 15.04 | 2.13s |
| Qwen2.5-0.5B-Instruct | v5 | 1 | 128 | 3.53 | 56.40 | 56.40 | 2.27s |
| Qwen2.5-0.5B-Instruct | v5 | 4 | 128 | 13.48 | 202.94 | 50.74 | 2.52s |
| Qwen2.5-0.5B-Instruct | official | 1 | 128 | 5.32 | 85.19 | 85.19 | 1.50s |
| Qwen2.5-0.5B-Instruct | official | 4 | 128 | 21.03 | 316.63 | 79.16 | 1.60s |

### 5.5 本次端到端结果摘要

| 对比项 | 结果 |
| --- | --- |
| Qwen batch=1 decode | v4 为 24.62 toks/s，official 为 85.19 toks/s，official 约为 v4 的 3.46 倍 |
| Qwen batch=4 decode | v4 为 60.14 toks/s，official 为 316.63 toks/s，official 约为 v4 的 5.27 倍 |
| GPT-2 batch=1 decode | v4 为 56.29 toks/s，official 为 306.43 toks/s，official 约为 v4 的 5.44 倍 |
| v5 相比 v4，Qwen batch=1 decode | v5 为 56.40 toks/s，约为 v4 的 2.29 倍 |
| v5 相比 v4，Qwen batch=4 decode | v5 为 202.94 toks/s，约为 v4 的 3.37 倍 |
| v5 相比 v4，GPT-2 batch=1 decode | v5 为 152.58 toks/s，约为 v4 的 2.71 倍 |
| official 相比 v5，Qwen batch=1 decode | official 约为 v5 的 1.51 倍 |
| official 相比 v5，Qwen batch=4 decode | official 约为 v5 的 1.56 倍 |
| Qwen batch=4 单请求 decode | v4 为 15.04 toks/s/request，official 为 79.16 toks/s/request，official 约为 v4 的 5.26 倍 |
| v4 多 batch 总吞吐 | Qwen 从 24.62 toks/s 提升到 60.14 toks/s，约 2.44 倍 |
| v5 多 batch 总吞吐 | Qwen 从 56.40 toks/s 提升到 202.94 toks/s，约 3.60 倍 |
| official 多 batch 总吞吐 | Qwen 从 85.19 toks/s 提升到 316.63 toks/s，约 3.72 倍 |
| baseline | 本次 baseline raw log 只有 shell 错误，未形成有效性能数据 |
| v5 | 本次 v5 GPT-2、Qwen batch=1、Qwen batch=4 case 均有有效性能数据 |

## 6. 版本分析

### 6.1 baseline

baseline 的角色是正确性参考，不作为性能目标。

性能特征：

| 维度 | 说明 |
| --- | --- |
| 运行位置 | Python / PyTorch |
| 主要用途 | 对拍、dump、replay |
| 性能预期 | 慢 |
| 本次数据状态 | 无有效数据 |

原因分析：

- baseline 保留了清晰的数学表达，方便定位 mask、GQA、paged KV 和 tail alignment 语义。
- baseline 没有针对 vLLM 真实推理链路做 kernel 级优化。
- 本次 baseline 日志没有进入 Python runner，而是只留下 shell 错误；因此不能从该日志推导性能或可运行性。

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
- v4 的 `KV_CHUNK_SIZE` 和 `Q_CHUNK_SIZE` 会显著影响 shared memory、同步开销和 chunk 循环次数。
- v4 在 debug 宏开启时会有额外开销，例如 device assert 和 `nan/non-finite` 检查。

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
| 本次数据状态 | 已采集 |

原因分析：

- 当前 benchmark 脚本已支持 `TOY_FLASH_ATTN_CUDA_VERSION=v5`，默认 case 中也包含 v5。
- 本次 v5 在 GPT-2 batch=1、Qwen batch=1、Qwen batch=4 case 中均成功生成，并解析到有效吞吐。
- v5 的 QK 阶段使用 WMMA 表达 `Q @ K^T`，输入为 bf16，累加为 fp32。
- v5 的 softmax 阶段仍使用 fp32 计算 max/sum，随后将 softmax numerator 量化为 bf16 供 PV 的 WMMA 路径使用。
- v5 的 PV 阶段使用 WMMA 表达 `P @ V`，输入为 bf16，累加为 fp32。
- v5 的 chunk merge 和最终 out 累积仍保留 fp32 路径，最后写回 bf16。
- 在本次端到端数据中，v5 相比 v4 有明显提升，但仍低于 official。

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

## 7. 调试开销记录

| 版本 | 配置 | input toks/s | output toks/s |
| --- | --- | --- | --- |
| v4 | assert/debug numeric on | 未采集 | 未采集 |
| v4 | assert/debug numeric off, Qwen batch=1 | 1.54 | 24.62 |
| v4 | assert/debug numeric off, Qwen batch=4 | 3.99 | 60.14 |
| v5 | assert/debug numeric off, Qwen batch=1 | 3.53 | 56.40 |
| v5 | assert/debug numeric off, Qwen batch=4 | 13.48 | 202.94 |

说明：

- 性能评估应使用 debug-off 配置。
- debug-on 数据只用于说明调试检查的开销。

## 8. 结论

| 问题 | 结论 |
| --- | --- |
| v4 是否能端到端运行 | 能。本次 GPT-2 和 Qwen2.5-0.5B-Instruct 的 v4 case 均成功生成。 |
| v4 是否支持 GQA | 能。Qwen2.5-0.5B-Instruct 使用 GQA，本次 v4 case 能端到端生成。 |
| v4 多 batch 是否提升总吞吐 | 能。Qwen batch=4 相比 batch=1，decode 总吞吐从 24.62 toks/s 提升到 60.14 toks/s。 |
| v5 是否能端到端运行 | 能。本次 GPT-2 和 Qwen2.5-0.5B-Instruct 的 v5 case 均成功生成。 |
| v5 是否提升 toy backend 吞吐 | 能。本次 Qwen batch=1 decode 从 v4 的 24.62 toks/s 提升到 v5 的 56.40 toks/s，batch=4 从 60.14 toks/s 提升到 202.94 toks/s。 |
| baseline 本次是否有性能结论 | 没有。本次 baseline raw log 只有 shell 错误，不能视为成功性能样本。 |
| toy backend 与 official 的主要差距 | v5 已缩小差距，但仍低于 official；Qwen batch=1 下 official 约为 v5 的 1.51 倍，batch=4 下约为 1.56 倍。 |
| v4 当前主要瓶颈 | 标量 QK/PV 计算、未使用 Tensor Core、shared memory 中间 tile 较多、同步点和 head_dim 串行循环开销较大。 |
