# 读vllm/v1/attention/backends/flash_attn.py笔记

## 概念

1. hybrid模型

模型不是纯attention注意力结构。可能有类似mamba的State Space Model。

2. mamba

Mamba 是一类序列模型架构，不走标准 self-attention 那条路，而是基于 state space model (SSM, 状态空间模型) 的思路做的。维护一个随序列推进不断更新的“隐藏状态”，新 token 来了就更新状态，再从状态里产生输出。

3. vllm block size

KV cache的block size, 通常是16的倍数，但是在包含hybrid等特殊情况，会固定为[16, 32, 64]。
（当 hybrid 模型且 Mamba cache 为 float32 时，为了绕开 flash-attention 的已知 NaN 问题，限制 block size 到 16/32/64）

4. sink

LLM在处理时，可能会除了滑动窗口的token外，还保留一定的全局信息用作sink,增强长上下文的处理能力。

5. cascade attention/prefix 复用优化

针对重复token, 分段式优化，减少重复计算。

6. GQA DCP

7. AOT = ahead of time

在cuda调度时，可以为cuda预先定制好调度编排（包括work怎么切，怎么分块，block到子任务怎么映射，scheduler meta怎么组织等），后续直接用，这里准备时间就是aot

8. MLP = Multi-Layer Perceptron

attention后的前馈网络。对每个token的hiddenstate做非线性变化，提升表达能力。通常过程是：
- 先升维
- 再激活
- 再降维。
    ```python
    gate = Linear(x)
    up = Linear(x)
    act = SiLU(gate)
    hidden = act * up
    out = Linear(hidden)
    ```

## QA

1. 多卡推理decoder的处理方式

场景：多卡 decoder attention 中，Q 很小，K/V 按上下文分散在各卡，不能全量 gather K/V。

做法：各卡先用本地 K_i/V_i 计算局部 attention，跨卡只归并 softmax 的 max/sum 等统计量，以及最终局部输出，不显式拼完整 attention 权重。

结论：通信不会像全量搬运 K/V 那样爆炸，但在单 token decode 场景下，跨卡同步仍可能成为时延瓶颈。

2. 大模型量化过程中，会对attention步骤量化吗

常常token 和 Q K V会量化，在载入kernel后，在attention kernel内做scale精度还原，以FP16参与计算，得到结果后再scale回fp8存入cache。

3. 总结flash attention的主要优化手段。

- cascade/prefix优化，对于公共前缀单独处理，减少计算
- dcp多卡/多设备计算（长上下文带宽换显存）
- aot, cuda计算图，尽量将所有能力放在一个计算图内完成（FA3版本开始支持），包括地址变换 & scale & 计算 descale & 反变换 等
- 对于KV cache，允许fp8存入
- 通过sink段，对全局信息的拼接