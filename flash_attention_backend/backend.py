"""
Minimal vLLM attention backend scaffold.

This file is intentionally conservative and incomplete:
- it shows the class shape you will likely need
- it narrows support aggressively
- it is meant to be adapted to the exact vLLM version you use
"""

from __future__ import annotations

from typing import ClassVar
from vllm.v1.attention.backend import AttentionBackend

import torch


class MinimalFlashAttentionBackend(AttentionBackend):
    """
    Replace/port this class against the exact AttentionBackend base class
    of the vLLM version you are targeting.
    """

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float32]

    def __init__(self):
        print("[custom backend] MinimalFlashAttentionBackend.__init__")
        super().__init__()


    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"
        # return "minimal_flash_attn_cpu_ref"

    @staticmethod
    def get_impl_cls():
        from .impl import MinimalFlashAttentionImpl

        return MinimalFlashAttentionImpl

    @staticmethod
    def get_builder_cls():
        return None

    @staticmethod
    def get_metadata_cls():
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ):
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def validate_configuration(
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype=None,
        block_size: int | None = None,
        is_attention_free: bool | None = None,
        use_mla: bool | None = None,
        use_sparse: bool | None = None,
        use_sinks: bool | None = None,
        has_sink: bool | None = None,
        **_: object,
    ) -> None:
        # if dtype != torch.float32:
        #     raise ValueError("minimal_flash_attn_cpu_ref only supports torch.float32")
        if head_size != 64:
            raise ValueError("minimal_flash_attn_cpu_ref only supports head_size=64")
        if use_mla:
            raise ValueError("MLA is not supported in the minimal backend")
        if use_sparse:
            raise ValueError("Sparse attention is not supported in the minimal backend")
        if use_sinks or has_sink:
            raise ValueError("Sink attention is not supported in the minimal backend")
