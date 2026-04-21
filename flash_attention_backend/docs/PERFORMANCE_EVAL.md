# toy_flash_attn 性能评估与分析报告

## 1. 评估范围

本文档只评估 attention backend 的端到端可运行性、正确性和性能表现。

当前纳入对比的实现版本：

| 版本 | 含义 | 入口 |
| --- | --- | --- |
| baseline | Python reference 实现 | `TOY_FLASH_ATTN_USE=reference` |
| v4 | 当前 toy CUDA kernel | `TOY_FLASH_ATTN_USE=bf16`，`TOY_FLASH_ATTN_CUDA_VERSION=v4` |
| official | vLLM 官方 FlashAttention backend | 见 `test_self_flash_attn_backend.py` 的 `_attention_config()` |

当前不讨论后续优化计划，只记录已有实现的评估口径、结果和原因分析。

## 2. 测试环境

| 项目 | 数值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2050 |
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |
| vLLM | 0.17.1 |
| Python | 3.10.12 |
| git commit | `5a5b3a68b52dca7adead05c22e173dae3967438c` |
| 模型 | GPT-2, Qwen2.5-0.5B-Instruct |
| dtype | torch.bfloat16 |
| kv_cache_dtype | v4: bfloat16; official: auto，实际解析为 torch.bfloat16 |
| max_model_len | 512 |
| batch size | 1, 4 |
| max tokens | 128 |

## 3. 功能边界

| 能力 | baseline | v4 | official |
| --- | --- | --- | --- |
| paged KV cache | 支持 | 支持 | 支持 |
| varlen query | 支持 | 支持 | 支持 |
| causal mask | 支持 | 支持 | 支持 |
| sliding window | 支持 | 支持 | 支持 |
| GQA | 支持 | 支持 | 支持 |
| bf16 输入 | 支持 | 支持 | 支持 |
| fp32 accumulator | PyTorch 行为 | 支持 | 官方实现内部策略 |
| head_dim=64 | 支持 | 支持 | 支持 |

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

## 5. 端到端性能评估

### 5.1 方法

端到端性能使用 `flash_attention_backend/test_self_flash_attn_backend.py` 运行。

记录指标：

| 指标 | 含义 |
| --- | --- |
| input toks/s | prefill 吞吐 |
| output toks/s | decode 吞吐 |
| output toks/s/request | 多 batch 下的单请求平均 decode 吞吐 |
| wall time | 端到端耗时 |
| finish reason | 生成结束原因 |

### 5.2 单 batch 结果

| 模型 | 版本 | batch | max tokens | input toks/s | output toks/s | wall time | finish reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 | v4 | 1 | 128 | 3.52 | 56.29 | 2.27s | 未采集 |
| GPT-2 | official | 1 | 128 | 19.15 | 306.43 | 未采集 | 未采集 |
| Qwen2.5-0.5B-Instruct | v4 | 1 | 128 | 1.54 | 24.62 | 5.20s | 未采集 |
| Qwen2.5-0.5B-Instruct | official | 1 | 128 | 5.32 | 85.19 | 1.50s | 未采集 |

### 5.3 多 batch 结果

| 模型 | 版本 | batch | max tokens | input toks/s | output toks/s | output toks/s/request | wall time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | v4 | 1 | 128 | 1.54 | 24.62 | 24.62 | 5.20s |
| Qwen2.5-0.5B-Instruct | v4 | 4 | 128 | 3.99 | 60.14 | 15.04 | 2.13s |
| Qwen2.5-0.5B-Instruct | official | 1 | 128 | 5.32 | 85.19 | 85.19 | 1.50s |
| Qwen2.5-0.5B-Instruct | official | 4 | 128 | 21.03 | 316.63 | 79.16 | 1.60s |

### 5.4 本次端到端结果摘要

| 对比项 | 结果 |
| --- | --- |
| Qwen batch=1 decode | v4 为 24.62 toks/s，official 为 85.19 toks/s，official 约为 v4 的 3.46 倍 |
| Qwen batch=4 decode | v4 为 60.14 toks/s，official 为 316.63 toks/s，official 约为 v4 的 5.27 倍 |
| GPT-2 batch=1 decode | v4 为 56.29 toks/s，official 为 306.43 toks/s，official 约为 v4 的 5.44 倍 |
| v4 多 batch 总吞吐 | Qwen 从 24.62 toks/s 提升到 60.14 toks/s，约 2.44 倍 |
| official 多 batch 总吞吐 | Qwen 从 85.19 toks/s 提升到 316.63 toks/s，约 3.72 倍 |
| baseline | 本次 baseline 日志是旧脚本的 `env -u` 错误残留，未形成有效性能数据 |

## 6. 版本分析

### 6.1 baseline

baseline 的角色是正确性参考，不作为性能目标。

性能特征：

| 维度 | 说明 |
| --- | --- |
| 运行位置 | Python / PyTorch |
| 主要用途 | 对拍、dump、replay |
| 性能预期 | 慢 |

原因分析：

- baseline 保留了清晰的数学表达，方便定位 mask、GQA、paged KV 和 tail alignment 语义。
- baseline 没有针对 vLLM 真实推理链路做 kernel 级优化。
- baseline 不适合作为性能对照，只适合作为正确性对照。

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

### 6.3 official

official 的角色是生产级性能基线。

性能特征：

| 维度 | 说明 |
| --- | --- |
| backend | vLLM 官方 `FLASH_ATTN` |
| kv_cache_dtype | `auto` |
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

说明：

- 性能评估应使用 debug-off 配置。
- debug-on 数据只用于说明调试检查的开销。

## 8. 结论

| 问题 | 结论 |
| --- | --- |
| v4 是否能端到端运行 | 能。本次 GPT-2 和 Qwen2.5-0.5B-Instruct 的 v4 case 均成功生成。 |
| v4 是否支持 GQA | 能。Qwen2.5-0.5B-Instruct 使用 GQA，本次 v4 case 能端到端生成。 |
| v4 多 batch 是否提升总吞吐 | 能。Qwen batch=4 相比 batch=1，decode 总吞吐从 24.62 toks/s 提升到 60.14 toks/s。 |
| v4 与 official 的主要差距 | 当前 v4 decode 吞吐明显低于 official；Qwen batch=1 下 official 约为 v4 的 3.46 倍，batch=4 下约为 5.27 倍。 |
| v4 当前主要瓶颈 | 标量 QK/PV 计算、未使用 Tensor Core、shared memory 中间 tile 较多、同步点和 head_dim 串行循环开销较大。 |
