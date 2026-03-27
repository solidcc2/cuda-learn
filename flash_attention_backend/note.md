# 读vllm/v1/attention/backends/flash_attn笔记

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

6. GQA

允许Query和KV的head数量不同，通常head_Q > head_kv,例如4:1,即4个Q head共用一个head kv。

7. DCP

DCP是分布式计算方式，多卡计算，为seq分片计算后归约。

8. AOT = ahead of time

在cuda调度时，可以为cuda预先定制好调度编排（包括work怎么切，怎么分块，block到子任务怎么映射，scheduler meta怎么组织等），后续直接用，这里准备时间就是aot

9. MLP = Multi-Layer Perceptron

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

10. MLA = Multi-Head Latent Attention

一种不同的attention结构路线。
通常意义下attention是：
```
Q = X @ Wq  # (token, head_dim_q)
K = X @ Wk  # (token, head_dim_k)
V = X @ Wv  # (token, head_dim_v)
delta = softmax(Q @ K^T / sqrt(head_dim)) @ V
```

MLA对KV大小做了压缩，只存储 X @ Wc的结果，减少占用。
```
Q = X @ Wc @ Wq
K = X @ Wc @ Wk # (token, head_dim_k = Wk_col), 只要缓存X @ Wc (token, Wc_col), Wc_col < Wk_col， 计算时通过X @ Wc还原出K
V = X @ Wc @ Wv # 同上
delta = softmax(Q @ K^T / sqrt(head_dim)) @ V
```
**注：**实际MLA比这个复杂，Q和K的head dim都分成两块计算，计算逻辑不同。

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
- mla，节约kv cache和带宽，对标准attention做latent压缩的。


## flash_attn_varlen_func参数含义

1. 核心张量
- q
- k
- v

query, key, value, 形状为[total, num_heads, head_dim],
对于query, total是当前batch内所有query token的集合

对于key/value, total是batch下的集合，paged KV,会分散在block内

2. 序列长度/varlen信息

- max_seqlen_q

batch里多个query序列的最大长度

- cu_seqlens_q

query的累积（cumulative sequence lengths）长度，
形状是[batch_size + 1], 
描述每个样本在扁平化q的起始位置。
query本身应该是[batch_size, token, num_head, head_dim],
如果知道起始位置，实际就是知道每个batch的NHD块的起点

- max_seqlen_k

batch里key序列的最大长度

- cu_seqlen_k

key的累积（cumulative sequence lengths）长度，
仅在非paged prefill时使用（paged是是分block的）

- seqused_k

每条样本实际使用的key长度，
通常用于paged KV/decode路径，与cu_seqlens_k二选一。相当于key的可见历史

3. 额外输入/变体

- q_v

FA3/FA4时，对于MLA时的额外 q分量。普通MHA时不用。是mla计算attention score的一个分块计算因子。

4. softmax/attention行为控制

- dropout_p
dropout概率，推理通常就是0.0f。
训练时为了防止过拟合，概率的将部分softmax结果置0。
``` delta = dropout(Q @ K^T / sqrt(d)) @ V```

- softmax_scale

softmax前对QK^T的缩放系数，通常是 `1/sqrt(head_dim)`

- causal

是否使用causal mask。
    - True：decoder自回归（推理decoder时）
    - False：双向attention

- window_size

滑动窗口的范围，如果是None, (-1, -1)则全局可见。
聊天式 LLM 在用户新增输入后，通常会先把历史上下文与这次输入拼接起来做 prefill，再进入 decode 逐 token 生成回答。
如果模型启用了 sliding window，那么 window_size 会影响 prefill 时每个 token 能看到多少左侧上下文。

- softcap

logits soft cap。
大于0时会对attention logits做soft cap限制。平滑截断：
- c就是soft cap
- S小时，S' = S
- S大时，`S' in [-c, c] `
```
S = Q @ K^T / sqrt(d)
delta = softmax(S)
--------
S' = c * tanh(S/c)
delta = softmax(S')

```

- alibi_slopes

ALiBi偏置参数。可按head / bach*head提供。不是所有FA版本都支持。

- deterministic

是否使用确定性实现。通常是训练/调试相关，推理一般不关心。

GPU上，即使数学公式一样，也可能因为：

    - 并行归约顺序不同
    - 原子操作顺序不同
    - 不同kernel调度路径： 
    - 浮点加法不满足严格结合率
解决办法：

    - 固定归约树/归约顺序
    - 规避多线程无序atomic，改为线程块独立算好再固定顺序合并
    - 多kernel/多split合并顺序，固定split树，merge顺序固定，禁用部分动态调度路径

- return_attn_probs

是否返回attention probabilites。测试用，推理不用。额外返回P矩阵用于调试。
```
P = softmax(Q @ K^T /sqrt(d))
```

5. paged KV/cache相关

- block table

block/page映射表。告诉kernel某条序列的逻辑block对应哪些物理block。

- out

输出Tensor,如果提供kernel会直接写到这里。就是本层token hidden state输出的delta。

6. 额外返回状态

- return_softmax_lse

是否返回softmax_lse。lse = logsumexp,用于cascade/prefix merge等需要合并attention state的场景。

7. FA3/低精度/调度相关

- scheduler_metadata

FA3的调度元数据。是提前生成的schedule/plan, 供底层kernel使用。

- q_descale， k_descale, v_descale

Q & K & V用于低精度输入恢复到计算域。例如fp8 -> bfp16

- num_splits

split-K/ split attention等调度优化的split数。不是所有FA版本都支持。
类似DCP, 就是按照seq切分片序列，然后分别计算 & 归约。

8. 版本选择

- fa_version
指定走FA2/FA3/FA4哪一代实现。

9. sink/CP相关

- s_aux

sink相关辅助输入。用于attention sink特性。
```
softmax(h) = exp(hi) / sum(exp(hi))
=>
softmax(h) = exp(hi) / (sum(exp(hi)) + s_aux) # 相当于投票时允许放弃投票
```

- cp_world_size

context parallel/DCP的world size。多卡上下文并行时使用

- cp_rank

当前节点在CP/DCP中的编号

- cp_tot_seqused_k

CP/DCP场景下，总体有效key长度相关信息。用于多卡上下文分片时的长度计算。

