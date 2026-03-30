from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backend import AttentionImpl, AttentionLayer, AttentionType

from .flash_attention_func import flash_attn_varlen_func

if TYPE_CHECKING:
    from .backend import ToyFlashAttentionMetadata


class ToyFlashAttentionImpl(AttentionImpl["ToyFlashAttentionMetadata"]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        assert num_heads > 0
        assert head_size > 0
        assert scale > 0
        assert num_kv_heads is None or num_kv_heads > 0

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("ToyFlashAttentionImpl only supports decoder attention")
        if alibi_slopes is not None:
            raise NotImplementedError("alibi_slopes is not supported")
        if logits_soft_cap is not None:
            raise NotImplementedError("logits_soft_cap is not supported")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported")
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(f"Unsupported kv_cache_dtype: {kv_cache_dtype}")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = None if sliding_window is None else (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ToyFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not supported")

        if attn_metadata is None:
            if output is None:
                return torch.zeros_like(query)
            return output.zero_()

        num_actual_tokens = attn_metadata.num_actual_tokens
        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        causal = attn_metadata.causal

        assert query.ndim == 3, "query must have shape [num_tokens, num_heads, head_size]"
        assert key.ndim == 3 and value.ndim == 3, "key/value must have shape [num_tokens, num_kv_heads, head_size]"
        assert kv_cache.ndim == 5, "kv_cache must have shape [2, num_blocks, block_size, num_kv_heads, head_size]"
        assert query.shape[1] == self.num_heads
        assert query.shape[2] == self.head_size
        assert key.shape == value.shape
        assert key.shape[1] == self.num_kv_heads
        assert key.shape[2] == self.head_size
        assert kv_cache.shape[0] == 2
        assert kv_cache.shape[3] == self.num_kv_heads
        assert kv_cache.shape[4] == self.head_size
        assert cu_seqlens_q.numel() >= 2

        if output is None:
            output = torch.empty_like(query)
        else:
            assert output.shape == query.shape

        key_cache, value_cache = kv_cache[0], kv_cache[1]

        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q.tolist(),
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=None,
            seqused_k=seqused_k.tolist(),
            causal=causal,
            window_size=self.sliding_window,
            block_table=block_table,
            out=output[:num_actual_tokens],
        )
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:      # eager 模式运行，后续可kernel化
        del layer

        key_cache, value_cache = kv_cache[0], kv_cache[1]
        block_size = key_cache.shape[1]
        slots = slot_mapping.to(device=key.device, dtype=torch.long)
        valid_mask = slots >= 0
        if not torch.any(valid_mask):
            return

        slots = slots[valid_mask]
        key = key[valid_mask]
        value = value[valid_mask]
        block_ids = slots // block_size
        block_offsets = slots % block_size
        key_cache[block_ids, block_offsets] = key
        value_cache[block_ids, block_offsets] = value
