# CLAUDE.md

本文件为 Claude Code（claude.ai/code）提供仓库指导。

## 构建与测试

> **Python 环境**：所有命令需在 vllm 虚拟环境中执行。先激活环境：`source ../vllm_env/bin/activate`。对 bash 脚本（如 `run_ncu_case.sh`）可通过 `PYTHON_BIN=../vllm_env/bin/python` 指定解释器路径。

### CUDA kernel 编译
CUDA kernel 在 import 时通过 `torch.utils.cpp_extension.load` 进行 JIT 编译。`toy_flash_attn/flash_attention_func.py` 被导入时触发编译，无需单独执行构建步骤。切换版本后首次导入需等待约 30 秒。

### 版本选择（环境变量）
```bash
# 选择 CUDA kernel 版本
export TOY_FLASH_ATTN_CUDA_VERSION=v7    # 可选: v3, v4, v5, v6, v7

# 选择后端实现
export TOY_FLASH_ATTN_USE=bf16            # bf16 CUDA kernel
                                          # reference: Python 参考实现
                                          # fp32: fp32 CUDA kernel（仅 v4 支持）
                                          # official: 官方 flash-attention
```

### 运行正确性测试
```bash
cd flash_attention_backend

# 全部正确性测试
TOY_FLASH_ATTN_CUDA_VERSION=v7 python -m pytest tests/correctness/ -v

# 单个测试类
TOY_FLASH_ATTN_CUDA_VERSION=v7 python -m pytest tests/correctness/test_fa2_parity.py -v

# 单个测试用例
TOY_FLASH_ATTN_CUDA_VERSION=v7 python -m pytest \
  tests/correctness/test_paged_kv_parity.py::FlashAttentionPagedKvParityTest::test_paged_kv_block_table_mapping_matches_fa2 -v
```

### 运行算子级性能测试
```bash
cd flash_attention_backend
python bench/op/bench_attention_op.py --version v7 --case qwen_like_b1_s128_h64
```

### 运行 NCU 性能分析
```bash
cd flash_attention_backend
bash analysis/run_ncu_case.sh --case qwen_like_b1_s2048_h64 --versions v7,official
```

### 端到端 vLLM 推理验证
```bash
cd flash_attention_backend
TOY_FLASH_ATTN_CUDA_VERSION=v7 python test_self_flash_attn_backend.py -m qwen -b 1 -t 128
```

### 完整性能评估流程
```bash
cd flash_attention_backend
bash analysis/run_perf_eval.sh
```

### 调试标志
```bash
export TOY_FLASH_ATTN_DEBUG=1              # 打印 kernel 输入张量信息
export TOY_FLASH_ATTN_DUMP_DIR=/tmp/dumps  # 导出每层 attention 的 I/O，用于离线回放
export TOY_FLASH_ATTN_PRINT_DTYPE=1        # 打印各中间张量的 dtype
```

## 项目架构

### 项目目标
从零实现 Flash Attention forward CUDA kernel，作为自定义 `AttentionBackend`（注册为 `CUSTOM`）接入 vLLM 推理框架。核心目的：与官方 flash-attention 做正确性对拍和性能对标，探索优化空间。

### 目录结构

```
flash_attention_backend/
  toy_flash_attn/                 ← CUDA kernel + Python 封装
    flash_attention_func.py         ← Python 入口，版本分发，debug/dump 基础设施
    backend.py                      ← vLLM AttentionBackend 注册（ToyFlashAttentionBackend）
    impl.py                         ← vLLM AttentionImpl 适配器（ToyFlashAttentionImpl）
    test_utils.py                   ← 通用测试工具：make_inputs, make_block_cache, runner 函数
    v3/ ~ v7/                       ← 各版本 kernel，包含：
      flash_attn_func.cu              ← CUDA kernel（ParamSet, TileLayout, kernel(), launcher, PYBIND11）
      helper.h                        ← 断言宏、nan/isfinite 检查、round_up、softmax_sub/div
  在 v7/ 当前分支实验状态：部分 Q/V/out 操作被注释，只保留单次 K 读取，用于定向测试 global excessive sectors

  tests/correctness/              ← 正确性验证（与官方 FA2 对拍）
    test_fa2_parity.py              ← 非 paged（dense）路径正确性
    test_paged_kv_parity.py         ← paged KV cache 路径正确性
    test_cuda_regression.py         ← CUDA 回归测试
    test_replay_dumps.py            ← 离线 dump 回放调试
    _helpers.py                     ← measure_close / assert_close，严格/宽松阈值
  bench/                          ← 性能测量
    op/bench_attention_op.py        ← 单算子延迟基准
    op/profile_attention_op.py      ← NCU profiling 驱动
    op/cases_op.py                  ← 基准 case 定义（qwen_like 等）
    e2e/                            ← 端到端 vLLM 推理基准
    common.py                       ← VERSION_CONFIGS、apply_version_env、system_environment
  analysis/                       ← 性能数据采集 & 报告生成
    run_perf_eval.sh                ← 统一编排入口
    run_ncu_case.sh                 ← NCU profiling  runner
    summarize_ncu_report.py         ← .ncu-rep → summary.json + SUMMARY.md
    build_report_inputs.py          ← 汇总所有结果到 report_inputs.json
    artifacts/ncu_*/                ← 历史 NCU profiling 存档（summary, CSV, 报告）

cuda_code/                        ← 独立的 CUDA 学习练习（matmul, softmax, histogram, reduction, transpose）
```

### 调用链：import → vLLM 集成

1. vLLM 加载 `toy_flash_attn.backend.ToyFlashAttentionBackend`（注册为 `CUSTOM`）
2. `impl.py` 的 `forward()` 将 vLLM 的 `AttentionLayer` 参数映射到 kernel 接口
3. `flash_attention_func.flash_attn_varlen_func` 根据 `TOY_FLASH_ATTN_CUDA_VERSION` 分发到对应版本
4. 首次调用时 `torch.utils.cpp_extension.load` JIT 编译 `.cu` 文件，生成 `_ops.flash_attn_varlen_with_block_v7_64`
5. **kernel 启动**：grid `(batch_size, q_chunk_count, kv_head_count)`，block 128 线程，共享内存大小由 `TileLayout::size(param)` 计算

### Kernel 架构模式（v7）

模板化的单 kernel 方案，依赖 CuTe 库：
- **`FlashAttnTrait<scalar_t, inner_scalar_t, Q_CHUNK_SIZE, KV_CHUNK_SIZE, HEAD_DIM_STRIDE>`** — 静态 struct，包含所有编译期参数、CuTe 类型定义、kernel 实现
- **`ParamSet`** — 启动参数（张量指针/步长/尺寸、配置），device 端访问器
- **`TileLayout`** — 共享内存布局管理器，基于偏移量计算 SMEM 分配（支持多区域复用以节省空间）
- **`kernel()`** — `__device__` 函数，执行完整 Flash Attention 流水线

### 核心算法流水线
```
for kv_chunk in reverse(kv_chunk_range):          ← 反向遍历以支持 online softmax
    加载 K + V tile（paged KV cache → 物理 block 地址解析）
    for local_q_head in q_per_kv_group:            ← GQA：一个 kv_head 对应多个 q_head
        (可选) 从 gmem 加载 Q tile
        MMA: Q_tile @ K_tile → Score              ← SM80_16x8x16_F32BF16BF16F32_TN
        应用 causal/window mask
        online softmax: Score → P（warp 归约求 max/sum，分组归约，block 广播）
        MMA: P @ V_tile → partial_out
        online softmax 合并：用新 chunk 的 max/sum 更新 running 输出
写回最终 out（除以 sum）到 gmem
```

### kernel 中的 CuTe 组件
- `SM80_CP_ASYNC_CACHEGLOBAL` / `SM80_16x8x16_F32BF16BF16F32_TN` — Ampere 架构的拷贝和 MMA 原语
- `TiledCopy` / `TiledMMA` — warp 级数据分区
- `Swizzle<5, 1>` — KV 共享内存的 bank conflict 消解（ComposedLayout: Swizzle ∘ Layout）
- `cp_async` 流水线 — 异步 gmem→smem 拷贝与计算重叠（v7 中部分使用）

### 已知性能瓶颈（从 devlog 总结）

按优先级排列：
1. **global excessive sectors**（百万级）— paged KV 逐元素读取，LDGSTS 退化为 LD.E.U16 → 目标：物理 block 整块预读 + cp.async 128B 向量化
2. **shared memory bank conflict**（百万级→15万通过 Swizzle 优化后）— 不充分的 swizzle 或布局选择 → 目标：调整 Swizzle 参数或改变 tile 布局
3. **低占用率**（~8%）— b1_s2048_h64 问题规模过小导致，非核心关注点

当前分支 `test_l2_sector` 集中实验第 1 项的 **phy_id 广播**方案。
