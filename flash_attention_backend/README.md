# Flash Attention Backend Minimal Scaffold

这个目录放的是一个“先跑通 vLLM 自定义 attention backend”的最小方案骨架。

目标：

- 先跑通一条最小 attention backend 路径
- 先只考虑 forward
- 先只考虑 prefill
- 先只支持非常窄的配置
- 先允许内部调用一个慢速 reference 实现

建议工作顺序：

1. 先把 `flash_attention_cpu_ref.py` 跑通，并用小张量对拍。
2. 再把 `backend.py` / `impl.py` 搬到 vLLM 的自定义 backend 实现位置。
3. 在 `validate_configuration()` 里严格限制支持范围。
4. 先手动指定 attention backend，验证能选中。
5. 确认 Python reference 能被调用后，再替换成 CUDA / torch extension 版本。

当前建议的最小支持范围：

- 仅 self-attention
- 仅 prefill
- 仅 causal=False 或 causal=True 里的一个
- 仅 `float32`
- 仅 `head_dim` 固定值，例如 `64`
- 仅连续布局张量

目录说明：

- `backend.py`
  vLLM attention backend 的最小骨架
- `impl.py`
  attention 实现骨架，内部调用 reference op
- `flash_attention_cpu_ref.py`
  纯 PyTorch 的最小 reference
- `notes.md`
  迁移到 vLLM 时的落地步骤

注意：

- 这里不是现成可直接 import 的 vLLM 插件包。
- 这是一个“先跑通”的本地骨架，方便你按步骤搬进 vLLM。
- 真正接入时，需要对齐你当前使用的 vLLM 版本接口。
