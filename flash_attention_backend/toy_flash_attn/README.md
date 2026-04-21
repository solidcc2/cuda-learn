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
- [v4/flash_attn_func.cu](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/v4/flash_attn_func.cu)
  当前默认 toy CUDA kernel，支持 bf16/fp32 CUDA 入口和 GQA head 映射。
- [flash_attn_func_v3.cu](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/flash_attn_func_v3.cu)
  旧版 toy CUDA kernel，仅保留 bf16 CUDA 入口，便于回归对照。
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

`block_table is not None` 时会走 paged 路径。当前分发由环境变量控制：

- `TOY_FLASH_ATTN_USE=reference`
  走 Python baseline：`flash_attn_varlen_with_block(...)`
- `TOY_FLASH_ATTN_USE=fp32`
  走 v4 fp32 CUDA debug path：输入先升到 fp32，kernel 内 fp32 计算，再写回 bf16 `out`。
- `TOY_FLASH_ATTN_USE=bf16`，或不设置
  走 bf16 CUDA path：bf16 storage + fp32 accumulator。

CUDA 源文件版本由另一个环境变量控制，必须在 import `flash_attention_func.py` 之前设置：

- `TOY_FLASH_ATTN_CUDA_VERSION=v4`，或不设置
  编译并加载 [v4/flash_attn_func.cu](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/v4/flash_attn_func.cu)。
- `TOY_FLASH_ATTN_CUDA_VERSION=v3`
  编译并加载 [flash_attn_func_v3.cu](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/flash_attn_func_v3.cu)。

注意：

- `TOY_FLASH_ATTN_USE=fp32` 只支持 `TOY_FLASH_ATTN_CUDA_VERSION=v4`。
- v3 只导出 `flash_attn_varlen_with_block_v3`，不支持当前 v4 的 fp32 CUDA 入口。
- 切换 `TOY_FLASH_ATTN_CUDA_VERSION` 后建议使用新的 Python 进程运行，避免已加载扩展复用旧版本。

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
TOY_FLASH_ATTN_USE=reference \
TOY_FLASH_ATTN_DUMP_DIR=$(pwd)/flash_attention_backend/base_gpt2.pt \
python -m unittest flash_attention_backend.test_self_flash_attn_backend
```

或运行你的真实 prompt 脚本，只要它会经过这个 backend 即可。

### 2. 生成 CUDA dump

把分发切回 `flash_attn_varlen_with_block_cu(...)`，再跑同一条链路：

```bash
TOY_FLASH_ATTN_USE=bf16 \
TOY_FLASH_ATTN_DUMP_DIR=$(pwd)/flash_attention_backend/fp32_fp32_gpt2.pt \
python -m unittest flash_attention_backend.test_self_flash_attn_backend
```

目录名只是习惯，不要求固定。

### 3. 一键生成 baseline/custom dump 并跑分析

也可以直接用：

- [run_gpt2_dump_analysis.py](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/run_gpt2_dump_analysis.py)

它会顺序执行：

1. `with_block` 生成 baseline dump
2. `with_block_cu` 生成 custom dump
3. 运行 `analyze_flash_attn_dumps.py` 生成报告

默认命令：

```bash
python flash_attention_backend/toy_flash_attn/run_gpt2_dump_analysis.py
```

默认输出：

- `flash_attention_backend/base_gpt2.pt/`
- `flash_attention_backend/bf16_fp32.pt/`
- `flash_attention_backend/base_gpt2.log`
- `flash_attention_backend/bf16_fp32.log`
- `flash_attention_backend/analyze.log`

可选参数示例：

```bash
python flash_attention_backend/toy_flash_attn/run_gpt2_dump_analysis.py \
  --output-root flash_attention_backend \
  --base-name base_gpt2.pt \
  --custom-name bf16_fp32.pt \
  --input-threshold 1e-5 \
  --output-threshold 1e-3 \
  --top-k 10
```

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

- v4 的可运行 head_dim 受 launch constraints 约束：
  - `ceil(head_dim / K_X_STRIDE)` 必须是 2 的幂
  - `Q_CHUNK_SIZE * ceil(head_dim / K_X_STRIDE)` 必须 warp 对齐
- `test_with_block_cu_known_negative_non_64_case`
  当前已放开为 head_dim=16 的覆盖 case，不再是显式跳过的 known negative。

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
- `TOY_FLASH_ATTN_USE`
  选择 attention 实现路径：
  - `reference`：Python baseline
  - `bf16`：bf16 CUDA path，默认值
  - `fp32`：v4 fp32 CUDA debug path
- `TOY_FLASH_ATTN_CUDA_VERSION`
  选择编译/加载的 CUDA 源文件版本：
  - `v4`：默认，加载 `v4/flash_attn_func.cu`
  - `v3`：加载 `flash_attn_func_v3.cu`
- `TOY_FLASH_ATTN_REPLAY_DUMP`
  replay 单测读取的 dump 路径。
- `TOY_FLASH_ATTN_DEBUG=1`
  打印 `with_block_cu` 输入 tensor 基本信息。
- `TOY_FLASH_ATTN_PRINT_DTYPE=1`
  打印 `with_block` 路径上的 dtype / matmul 配置。

### CUDA 侧

v4 主要调试宏在 [v4/helper.h](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/v4/helper.h) 和 [v4/flash_attn_func.cu](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/v4/flash_attn_func.cu)：

- `DEBUG_FLASH_ATTN_V3_TRACE`
  打开后打印：
  - `score_mask`
  - `chunk_softmax`
  - `sv_reduce`
  - `chunk_merge`
- `DEBUG_NUMERIC`
  用于 `nan/non-finite` 检查

v3 的对应宏仍在 [flash_attn_func_v3.cu](/home/linf/code/cuda/flash_attention_backend/toy_flash_attn/flash_attn_func_v3.cu) 内。

## v4 fp32 shared memory 注意事项

v4 的 `TOY_FLASH_ATTN_USE=fp32` 会使用 `FlashAttnTrait<float, float>`，中间 tile 都是 fp32，占用显著高于 bf16 storage path。

当前 v4 顶部有 4 个会直接影响 block shape 和 shared memory 占用的常量：

```cpp
static const int32_t K_X_STRIDE = ...;
static const int32_t Q_CHUNK_SIZE = ...;
static const int32_t KV_CHUNK_SIZE = ...;
static const int32_t BLOCK_Y = ...;
```

如果把 v4 配成较大的 tile，例如：

```cpp
K_X_STRIDE = 16
Q_CHUNK_SIZE = 16
KV_CHUNK_SIZE = 64
BLOCK_Y = 64
```

fp32 path 很容易因为 dynamic shared memory 超过单 block 默认上限而启动失败，典型错误是：

```text
CUDA error: invalid argument
```

因此在使用 `TOY_FLASH_ATTN_USE=fp32` 做 debug/对拍时，需要修改上述 4 个常量，缩小 tile 占用后再运行。一个保守配置是：

```cpp
static const int32_t K_X_STRIDE = 4;
static const int32_t Q_CHUNK_SIZE = 8;
static const int32_t KV_CHUNK_SIZE = 8;
static const int32_t BLOCK_Y = 8;
```

这个限制来自当前 v4 保留了较多中间 shared tile，例如 `score/max/sum/softmax/qk_reduction/sv_reduction/out_reduction`。后续如果改为 WMMA/CuTe 并复用或删除这些中间 tile，shared memory 压力可以下降。

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
