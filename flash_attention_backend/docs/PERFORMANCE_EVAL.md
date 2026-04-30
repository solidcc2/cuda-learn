# toy_flash_attn 性能评估与分析报告

## 1. 评估范围

本文档基于当前 benchmark 产物更新，只记录已验证的有效样本，不补推缺失结果。

本次报告依据：

- `analysis/run_perf_eval.sh`
- `analysis/perf_eval_results.json`
- `analysis/perf_logs/*.log`

当前 benchmark 脚本已纳入以下版本：

| 版本 | 本次脚本状态 | 入口 |
| --- | --- | --- |
| baseline | 已纳入 | `TOY_FLASH_ATTN_USE=reference` |
| v4 | 已纳入 | `TOY_FLASH_ATTN_USE=bf16`，`TOY_FLASH_ATTN_CUDA_VERSION=v4` |
| v5 | 已纳入 | `TOY_FLASH_ATTN_USE=bf16`，`TOY_FLASH_ATTN_CUDA_VERSION=v5` |
| v6 | 已纳入 | `TOY_FLASH_ATTN_USE=bf16`，`TOY_FLASH_ATTN_CUDA_VERSION=v6` |
| official | 已纳入 | `TOY_FLASH_ATTN_USE=official` |

当前套件定义：

- `smoke`：128 token，覆盖 `baseline / v4 / v5 / v6 / official`
- `report`：512 token，覆盖 `v4 / v5 / v6 / official`
- `stress`：2048 token，覆盖 `v4 / v5 / v6 / official`

## 2. 测试环境

环境信息来自当前 `perf_eval_results.json`。

| 项目 | 数值 |
| --- | --- |
| 数据生成时间 | 2026-04-30T10:45:50.501089+00:00 |
| GPU | NVIDIA GeForce RTX 2050 |
| Python | 3.10.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA toolkit | 12.8 |
| `torch.cuda.is_available()` | `true` |
| vLLM | 0.17.1 |
| 测量 git commit | `d08a36aad054a95fed380144b0111b49f07f68b9` |

## 3. 功能边界

| 能力 | baseline | v4 | v5 | v6 | official |
| --- | --- | --- | --- | --- | --- |
| paged KV cache | 支持 | 支持 | 支持 | 支持 | 支持 |
| varlen query | 支持 | 支持 | 支持 | 支持 | 支持 |
| causal mask | 支持 | 支持 | 支持 | 支持 | 支持 |
| sliding window | 支持 | 支持 | 支持 | 支持 | 支持 |
| GQA | 支持 | 支持 | 支持 | 支持 | 支持 |
| bf16 输入 | 支持 | 支持 | 支持 | 支持 | 支持 |
| fp32 accumulator | PyTorch 行为 | 支持 | 支持 | 支持 | 官方实现内部策略 |
| head_dim=64 | 支持 | 支持 | 支持 | 支持 | 支持 |
| Tensor Core / MMA | 不适用 | 未使用 | WMMA | CuTe MMA | 官方实现内部策略 |

说明：本节描述当前源码能力边界，不等同于本次 benchmark 已对所有版本和 case 采集到有效性能数据。

## 4. 正确性评估

本次 JSON 仅包含端到端生成 benchmark，没有独立数值对拍字段。

因此当前报告不填充：

- `max abs diff`
- `mean abs diff`
- `pass rate`

## 5. 端到端性能评估

### 5.1 数据有效性口径

仅当同时满足以下条件时，样本才计入正式性能表：

- `success: true`
- `input_toks_per_s` 与 `output_toks_per_s` 非空
- `prompt_count == batch`
- `generated_count == batch`

本次汇总：

| 指标 | 数值 |
| --- | --- |
| benchmark case 总数 | 25 |
| 有效样本数 | 25 |
| 无效样本数 | 0 |

### 5.2 Case 状态总览

| 版本 | 模型 | batch | max_tokens | 状态 |
| --- | --- | --- | --- | --- |
| baseline | qwen | 1 | 128 | 已采集 |
| v4 | gpt2 | 1 | 128 | 已采集 |
| v4 | qwen | 1 | 128 | 已采集 |
| v4 | qwen | 4 | 128 | 已采集 |
| v4 | qwen | 1 | 512 | 已采集 |
| v4 | qwen | 4 | 512 | 已采集 |
| v4 | qwen | 1 | 2048 | 已采集 |
| v5 | gpt2 | 1 | 128 | 已采集 |
| v5 | qwen | 1 | 128 | 已采集 |
| v5 | qwen | 4 | 128 | 已采集 |
| v5 | qwen | 1 | 512 | 已采集 |
| v5 | qwen | 4 | 512 | 已采集 |
| v5 | qwen | 1 | 2048 | 已采集 |
| v6 | gpt2 | 1 | 128 | 已采集 |
| v6 | qwen | 1 | 128 | 已采集 |
| v6 | qwen | 4 | 128 | 已采集 |
| v6 | qwen | 1 | 512 | 已采集 |
| v6 | qwen | 4 | 512 | 已采集 |
| v6 | qwen | 1 | 2048 | 已采集 |
| official | gpt2 | 1 | 128 | 已采集 |
| official | qwen | 1 | 128 | 已采集 |
| official | qwen | 4 | 128 | 已采集 |
| official | qwen | 1 | 512 | 已采集 |
| official | qwen | 4 | 512 | 已采集 |
| official | qwen | 1 | 2048 | 已采集 |

### 5.3 有效性能样本

| 模型 | 版本 | batch | max_tokens | input toks/s | output toks/s | output toks/s/request | wall time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 | official | 1 | 128 | 19.13 | 306.06 | 306.06 | 未采集 |
| GPT-2 | v4 | 1 | 128 | 3.51 | 56.19 | 56.19 | 2.28s |
| GPT-2 | v5 | 1 | 128 | 10.26 | 164.13 | 164.13 | 未采集 |
| GPT-2 | v6 | 1 | 128 | 10.38 | 166.10 | 166.10 | 未采集 |
| Qwen2.5-0.5B-Instruct | baseline | 1 | 128 | 1.19 | 19.04 | 19.04 | 6.72s |
| Qwen2.5-0.5B-Instruct | official | 1 | 128 | 5.32 | 85.19 | 85.19 | 1.50s |
| Qwen2.5-0.5B-Instruct | v4 | 1 | 128 | 1.54 | 24.68 | 24.68 | 5.19s |
| Qwen2.5-0.5B-Instruct | v5 | 1 | 128 | 3.62 | 57.85 | 57.85 | 2.21s |
| Qwen2.5-0.5B-Instruct | v6 | 1 | 128 | 3.63 | 58.05 | 58.05 | 2.21s |
| Qwen2.5-0.5B-Instruct | official | 4 | 128 | 21.01 | 313.36 | 78.34 | 1.54s |
| Qwen2.5-0.5B-Instruct | v4 | 4 | 128 | 3.98 | 59.98 | 14.99 | 2.13s |
| Qwen2.5-0.5B-Instruct | v5 | 4 | 128 | 13.60 | 204.75 | 51.19 | 2.50s |
| Qwen2.5-0.5B-Instruct | v6 | 4 | 128 | 14.39 | 216.70 | 54.17 | 2.36s |
| Qwen2.5-0.5B-Instruct | official | 1 | 512 | 1.36 | 85.54 | 85.54 | 5.89s |
| Qwen2.5-0.5B-Instruct | v4 | 1 | 512 | 0.19 | 12.08 | 12.08 | 41.72s |
| Qwen2.5-0.5B-Instruct | v5 | 1 | 512 | 0.82 | 51.48 | 51.48 | 9.79s |
| Qwen2.5-0.5B-Instruct | v6 | 1 | 512 | 0.84 | 53.18 | 53.18 | 9.48s |
| Qwen2.5-0.5B-Instruct | official | 4 | 512 | 5.40 | 267.12 | 66.78 | 1.58s |
| Qwen2.5-0.5B-Instruct | v4 | 4 | 512 | 0.45 | 22.22 | 5.55 | 18.90s |
| Qwen2.5-0.5B-Instruct | v5 | 4 | 512 | 2.83 | 167.38 | 41.84 | 3.01s |
| Qwen2.5-0.5B-Instruct | v6 | 4 | 512 | 3.03 | 179.41 | 44.85 | 2.81s |
| Qwen2.5-0.5B-Instruct | official | 1 | 2048 | 1.36 | 85.45 | 85.45 | 5.90s |
| Qwen2.5-0.5B-Instruct | v4 | 1 | 2048 | 0.19 | 12.11 | 12.11 | 41.63s |
| Qwen2.5-0.5B-Instruct | v5 | 1 | 2048 | 0.82 | 51.78 | 51.78 | 9.73s |
| Qwen2.5-0.5B-Instruct | v6 | 1 | 2048 | 0.84 | 52.63 | 52.63 | 9.58s |

### 5.4 本次有效结果摘要

| 对比项 | 结果 |
| --- | --- |
| baseline Qwen batch=1 decode, 128 token | 19.04 toks/s |
| v4 相比 baseline，Qwen batch=1 decode, 128 token | v4 为 24.68 toks/s，约为 baseline 的 1.30 倍 |
| v5 相比 v4，Qwen batch=1 decode, 128 token | v5 为 57.85 toks/s，约为 v4 的 2.34 倍 |
| v6 相比 v5，Qwen batch=1 decode, 128 token | v6 为 58.05 toks/s，约为 v5 的 1.00 倍 |
| official 相比 v6，Qwen batch=1 decode, 128 token | official 为 85.19 toks/s，约为 v6 的 1.47 倍 |
| v5 相比 v4，Qwen batch=4 decode, 128 token | v5 为 204.75 toks/s，约为 v4 的 3.41 倍 |
| v6 相比 v5，Qwen batch=4 decode, 128 token | v6 为 216.70 toks/s，约为 v5 的 1.06 倍 |
| official 相比 v6，Qwen batch=4 decode, 128 token | official 为 313.36 toks/s，约为 v6 的 1.45 倍 |
| v5 相比 v4，Qwen batch=4 decode, 512 token | v5 为 167.38 toks/s，约为 v4 的 7.53 倍 |
| v6 相比 v5，Qwen batch=4 decode, 512 token | v6 为 179.41 toks/s，约为 v5 的 1.07 倍 |
| official 相比 v6，Qwen batch=4 decode, 512 token | official 为 267.12 toks/s，约为 v6 的 1.49 倍 |
| v6 相比 v5，Qwen batch=1 decode, 2048 token | v6 为 52.63 toks/s，约为 v5 的 1.02 倍 |
| official 相比 v6，Qwen batch=1 decode, 2048 token | official 为 85.45 toks/s，约为 v6 的 1.62 倍 |
| v5 相比 v4，GPT-2 batch=1 decode, 128 token | v5 为 164.13 toks/s，约为 v4 的 2.92 倍 |
| v6 相比 v5，GPT-2 batch=1 decode, 128 token | v6 为 166.10 toks/s，约为 v5 的 1.01 倍 |
| official 相比 v6，GPT-2 batch=1 decode, 128 token | official 为 306.06 toks/s，约为 v6 的 1.84 倍 |

## 6. 版本分析

### 6.1 baseline

- 本次 baseline 只有 `qwen / batch=1 / 128 token` 一个样本。
- decode 吞吐为 19.04 toks/s。
- baseline 仍主要用于 reference 和语义对拍，不作为性能目标。

### 6.2 v4

- v4 本轮 6 个 case 全部有效。
- Qwen 上：
  - `batch=1` 时，128/512/2048 token 的 decode 吞吐分别为 24.68 / 12.08 / 12.11 toks/s。
  - `batch=4` 时，128/512 token 的 decode 吞吐分别为 59.98 / 22.22 toks/s。
- 相比 v5/v6/official，v4 在中长序列上的差距仍然很大，尤其是 512 和 2048 token。

### 6.3 v5

- v5 本轮 6 个 case 全部有效，是完整的自定义 backend 基线。
- Qwen 上：
  - `batch=1` 时，128/512/2048 token 的 decode 吞吐分别为 57.85 / 51.48 / 51.78 toks/s。
  - `batch=4` 时，128/512 token 的 decode 吞吐分别为 204.75 / 167.38 toks/s。
- GPT-2 `batch=1 / 128 token` 的 decode 吞吐为 164.13 toks/s。

### 6.4 v6

- v6 本轮 6 个 case 全部有效。
- Qwen 上：
  - `batch=1` 时，128/512/2048 token 的 decode 吞吐分别为 58.05 / 53.18 / 52.63 toks/s。
  - `batch=4` 时，128/512 token 的 decode 吞吐分别为 216.70 / 179.41 toks/s。
- GPT-2 `batch=1 / 128 token` 的 decode 吞吐为 166.10 toks/s。
- 相比 v5，v6 在当前有效样本上是稳定小幅领先：
  - Qwen `batch=1 / 128 token`：1.00 倍
  - Qwen `batch=4 / 128 token`：1.06 倍
  - Qwen `batch=4 / 512 token`：1.07 倍
  - Qwen `batch=1 / 2048 token`：1.02 倍
  - GPT-2 `batch=1 / 128 token`：1.01 倍

### 6.5 official

- official 本轮 6 个 case 全部有效。
- Qwen 上：
  - `batch=1` 时，128/512/2048 token 的 decode 吞吐分别为 85.19 / 85.54 / 85.45 toks/s。
  - `batch=4` 时，128/512 token 的 decode 吞吐分别为 313.36 / 267.12 toks/s。
- GPT-2 `batch=1 / 128 token` 的 decode 吞吐为 306.06 toks/s。
- 当前所有可比 case 上，official 都明显快于 v4/v5/v6。

## 7. 结论

本轮 benchmark 的主要结论是：

1. `baseline / v4 / v5 / v6 / official` 全部目标 case 都已采集到有效数据。
2. `v6` 在本轮所有与 `v5` 可比的有效样本上都略快，但提升幅度普遍较小，约在 1.00 到 1.07 倍之间。
3. `v5` 到 `v6` 的提升主要体现在 Qwen 的 batch=4 case，尤其是 128 和 512 token。
4. `official` 仍是当前性能上界，在所有可比 case 上都显著快于 `v6`。
