from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .flash_attention_cpu_ref import flash_attention_cpu_ref
from vllm.v1.attention.backend import AttentionImpl, AttentionMetadata, AttentionType


@dataclass
class MinimalAttentionMetadata(AttentionMetadata):
    causal: bool = False


class MinimalFlashAttentionImpl(AttentionImpl):
    """
    Minimal implementation scaffold.

    Adapt the constructor and forward signature to the exact vLLM version
    you are integrating with.
    """
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.causal = False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor | None = None,
        attn_metadata: MinimalAttentionMetadata | None = None,
        **_: object,
    ) -> torch.Tensor:
        print("MinimalFlashAttentionImpl.forward called")
        causal = self.causal
        if attn_metadata is not None:
            causal = attn_metadata.causal

        out = flash_attention_cpu_ref(
            q=q,
            k=k,
            v=v,
            causal=causal,
            scale=self.scale,
        )

        if output is not None:
            output.copy_(out)
            return output
        return out
