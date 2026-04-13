# toy_flash_attn

`toy_flash_attn` 是一个用于验证 paged KV + local/causal mask + online softmax merge 的 toy FlashAttention 实现，包含：

- Python baseline 路径：`flash_attn_varlen_with_block`
- CUDA toy kernel 路径：`flash_attn_varlen_with_block_cu`
- 无 cache 的简单参考路径：`flash_attn_varlen_without_block`
- dump / replay / drift 分析工具

这套代码主要用于两类工作：

- 单层 attention 对拍：确认 `with_block_cu` 和 Python baseline 是否一致
- 放回 vLLM 真实链路后做逐 step dump，对齐分析“从哪里开始漂移”

## 主要文件

- [flash_attention_func.py](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/flash_attention_func.py)
  统一入口、dump、调试打印、Python baseline。
- [flash_attn_func_v3.cu](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/flash_attn_func_v3.cu)
  当前 toy CUDA kernel。
- [flash_attention_func_test.py](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/flash_attention_func_test.py)
  单测、replay、回归 case。
- [analyze_flash_attn_dumps.py](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/analyze_flash_attn_dumps.py)
  比较两份 dump 目录，分析最早输入/输出漂移。
- [impl.py](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/impl.py)
  vLLM backend 接入层。

## 当前调试约定

- load 阶段处理真实边界，越界填 `0`
- dot-product 中间态不混 `-inf`
- score 层统一做 window/causal mask，invalid score 写 `-inf`
- softmax 后 invalid 自然变 `0`
- `do { } while(0)` 可以做局部短路，但不能破坏 `__syncthreads()` 对齐

## 当前入口怎么切换

入口在 [flash_attention_func.py](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/flash_attention_func.py) 的 `flash_attn_varlen_func(...)`。

`block_table is not None` 时会走 paged 路径。当前分发是通过注释切换的：

- 走 CUDA：`flash_attn_varlen_with_block_cu(...)`
- 走 Python baseline：`flash_attn_varlen_with_block(...)`

调试时常见做法：

- 先用 `with_block` 生成 baseline dump
- 再切到 `with_block_cu` 做 replay / 整链路复测

## dump 快照

设置环境变量后，每次 attention 调用都会落一份 `.pt`：

```bash
TOY_FLASH_ATTN_DUMP_DIR=/tmp/fa_dump
```

dump 文件名形如：

```text
00000_with_block.pt
00001_with_block_cu.pt
```

当前 dump payload 主要字段：

- `q`
- `k`
- `v`
- `max_seqlen_q`
- `cu_seqlens_q`
- `max_seqlen_k`
- `seqused_k`
- `causal`
- `window_size`
- `block_table`
- `result`
- `debug_meta`

`debug_meta` 当前记录：

- `layer_name`
- `layer_idx`（如果上层 layer 对象上有）

说明：

- replay / 分析脚本当前要求 dump 必须包含 `result`
- 旧格式 dump 不再兼容，缺字段时需要重新生成

## 怎么生成 baseline / custom dump

### 1. 生成 Python baseline dump

先让 `flash_attn_varlen_func(...)` 分发到 `flash_attn_varlen_with_block(...)`，再运行真实链路，例如：

```bash
TOY_FLASH_ATTN_DUMP_DIR=$(pwd)/flash_attention_backend/base_gpt2.pt \
python -m unittest flash_attention_backend.test_self_flash_attn_backend
```

或运行你的真实 prompt 脚本，只要它会经过这个 backend 即可。

### 2. 生成 CUDA dump

把分发切回 `flash_attn_varlen_with_block_cu(...)`，再跑同一条链路：

```bash
TOY_FLASH_ATTN_DUMP_DIR=$(pwd)/flash_attention_backend/fp32_fp32_gpt2.pt \
python -m unittest flash_attention_backend.test_self_flash_attn_backend
```

目录名只是习惯，不要求固定。

## replay 单测

replay 单测从 `TOY_FLASH_ATTN_REPLAY_DUMP` 读取 dump。

环境变量可以指向：

- 单个 `.pt`
- 一个 dump 目录

例如：

```bash
TOY_FLASH_ATTN_REPLAY_DUMP=$(pwd)/flash_attention_backend/base_gpt2.pt \
python -m unittest \
toy_flash_attn.flash_attention_func_test.FlashAttentionFuncCuKernelHeadDim64RegressionTest.test_with_block_cu_replay_dump_matches_python
```

### 重要 replay case

在 [flash_attention_func_test.py](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/flash_attention_func_test.py)：

- `test_with_block_cu_replay_dump_matches_python`
  对目录里的全部 dump 做 replay。
- `test_with_block_cu_replay_top10_vllm_worst_dumps`
  只回放一组固定 worst dump，适合开 kernel trace。
- `test_with_block_cu_replay_step0_matches_python`
  只检查 `00000_with_block.pt`，适合查最早一步是否分叉。

对应命令：

全量 replay：

```bash
TOY_FLASH_ATTN_REPLAY_DUMP=$(pwd)/flash_attention_backend/base_gpt2.pt \
python -m unittest \
toy_flash_attn.flash_attention_func_test.FlashAttentionFuncCuKernelHeadDim64RegressionTest.test_with_block_cu_replay_dump_matches_python
```

只跑 top10 worst dump：

```bash
TOY_FLASH_ATTN_REPLAY_DUMP=$(pwd)/flash_attention_backend/base_gpt2.pt \
python -m unittest \
toy_flash_attn.flash_attention_func_test.FlashAttentionFuncCuKernelHeadDim64RegressionTest.test_with_block_cu_replay_top10_vllm_worst_dumps
```

只跑 step0：

```bash
TOY_FLASH_ATTN_REPLAY_DUMP=$(pwd)/flash_attention_backend/base_gpt2.pt \
python -m unittest \
toy_flash_attn.flash_attention_func_test.FlashAttentionFuncCuKernelHeadDim64RegressionTest.test_with_block_cu_replay_step0_matches_python
```

如果要同时把日志写到文件：

```bash
TOY_FLASH_ATTN_REPLAY_DUMP=$(pwd)/flash_attention_backend/base_gpt2.pt \
python -m unittest \
toy_flash_attn.flash_attention_func_test.FlashAttentionFuncCuKernelHeadDim64RegressionTest.test_with_block_cu_replay_step0_matches_python \
2>&1 | tee /tmp/step0_check.log
```

## 常用回归单测

### parity / 基础覆盖

- `test_with_block_cu_matches_python_causal_attention_bf16`
- `test_with_block_cu_matches_python_full_attention_bf16`
- `test_with_block_cu_matches_python_tail_aligned_suffix_query_bf16`

### head_dim=64 回归

- `test_with_block_cu_head_dim_64_tail_aligned_causal_local_window_regression`
  固定同一个 regression case，循环 1000 次，主要用于抓偶现问题。
- `test_with_block_cu_head_dim_64_regression_matrix`
  一组 head_dim=64 的组合 case。
- `test_with_block_cu_head_dim_64_outputs_are_finite`
  专门检查输出有没有 NaN/Inf。

### 已知限制

- `test_with_block_cu_known_negative_non_64_case`
  当前显式跳过。原因是 kernel 依赖 `blockDim.x >= 32`，小 head case 还没支持好。

## 分析脚本

脚本：

- [analyze_flash_attn_dumps.py](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/analyze_flash_attn_dumps.py)

用途：

- 比较两份 dump 目录
- 找最早 metadata / 输入 / dense KV / 输出漂移
- 打印 top-k 最坏输出 step
- 打印 top-k 最坏 dense KV 漂移 step

运行方式：

```bash
python flash_attention_backend/toy_flash_attn/analyze_flash_attn_dumps.py \
  flash_attention_backend/base_gpt2.pt \
  flash_attention_backend/fp32_fp32_gpt2.pt \
  > flash_attention_backend/analyze.log 2>&1
```

默认阈值：

- `input_threshold = 1e-5`
- `output_threshold = 1e-3`

可调参数：

```bash
python flash_attention_backend/toy_flash_attn/analyze_flash_attn_dumps.py \
  BASE_DIR OTHER_DIR \
  --input-threshold 1e-5 \
  --output-threshold 1e-3 \
  --top-k 10
```

### 当前脚本会比对什么

直接逐元素比对：

- `q`
- `k`
- `v`
- `cu_seqlens_q`
- `seqused_k`
- `result`

语义比对：

- 用 `block_table + k/v + seqused_k` 还原本次 attention 实际访问的 dense `K/V`
- 再比较 dense `K/V`

一致性检查：

- `causal`
- `window_size`
- `layer_name`
- `layer_idx`

注意：

- `block_table` 不做直接值相等比较
- 因为物理 block 映射每次运行可能不同，真正有意义的是它还原出来的 dense `K/V` 是否一致

## 调试开关

### Python 侧

- `TOY_FLASH_ATTN_DUMP_DIR`
  开启 dump。
- `TOY_FLASH_ATTN_REPLAY_DUMP`
  replay 单测读取的 dump 路径。
- `TOY_FLASH_ATTN_DEBUG=1`
  打印 `with_block_cu` 输入 tensor 基本信息。
- `TOY_FLASH_ATTN_PRINT_DTYPE=1`
  打印 `with_block` 路径上的 dtype / matmul 配置。

### CUDA 侧

在 [flash_attn_func_v3.cu](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/flash_attn_func_v3.cu) 顶部：

- `DEBUG_FLASH_ATTN_V3_TRACE`
  打开后打印：
  - `score_mask`
  - `chunk_softmax`
  - `sv_reduce`
  - `chunk_merge`
- `DEBUG_NUMERIC`
  用于 `nan/non-finite` 检查

建议：

- 全量 replay 时不要开 `DEBUG_FLASH_ATTN_V3_TRACE`
- 先用 `replay_step0` 或 `top10_worst_dumps` 缩小范围后再开

## 建议排查路径

### 1. 单层一致性

- 跑 `with_block_cu` 对拍基础 case
- 先确认没有 NaN/Inf
- 再看 `max diff / mean diff`

### 2. 重放真实链路 dump

- 先生成 `with_block` baseline dump
- 再生成 `with_block_cu` dump
- 用 replay 单测确认具体 dump 的单步差异

### 3. 链路级漂移

- 用 `analyze_flash_attn_dumps.py` 找：
  - 最早输入漂移
  - 最早 dense KV 漂移
  - 最早输出漂移

### 4. 放大分析

- 如果 `step0` 只是小漂移，`step1` 输入就大幅分叉
- 重点确认：
  - 当前调用是否对齐
  - dense KV 是否已分叉
  - 是否是整链路传播，不是单步 kernel 直接跑飞

## 额外说明

- 当前 baseline 和 CUDA 都经常会做 dtype 实验，尤其 `bf16` / `fp32`
- `fp32/fp32` 更适合做归因实验
- 最终是否“工作正常”，仍应以真实链路口径验证
