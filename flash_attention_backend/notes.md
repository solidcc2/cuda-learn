# First Bring-Up Notes

## 最小目标

让 vLLM 在一次最小推理里：

- 成功选中你的 attention backend
- 调到你的实现
- 返回 shape 正确的输出

## 最小落地步骤

1. 把 `backend.py` 和 `impl.py` 对照 vLLM 当前版本的 attention backend 接口放进去。
2. 在 backend 注册表里注册一个新名字，比如 `my_flash_attn_cpu_ref`。
3. 在 `validate_configuration()` 里只允许你当前能支持的配置。
4. 在 `impl.forward()` 里先直接调 `flash_attention_cpu_ref.py`。
5. 用一个很小的模型和很小的 prompt 跑通。

## 优先避开的复杂度

- decode
- paged KV cache
- variable length batching
- GQA / MQA
- fp16 / bf16
- custom CUDA op

## 第二阶段

首次跑通后，再逐步替换：

- Python reference -> torch extension
- dense full attention -> 分块 flash attention
- prefill only -> decode
