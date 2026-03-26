from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImplBase,
    AttentionMetadataBuilder,
    MultipleOf,
    AttentionType,
    DeviceCapability,
    CommonAttentionMetadata
) 
from vllm.config.cache import CacheDType
import torch
from .impl import ToyFlashAttentionImpl

class ToyFlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["AttentionImplBase"]:
        return ToyFlashAttentionImpl 

    @staticmethod
    def get_builder_cls():  # -> Type["AttentionMetadataBuilder"]:
        return ToyFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension == True:
            print("not support include_num_layers_dimension")
            raise NotImplementedError
        return [0, 1, 2, 3, 4]  # 维持NHD

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size < 256 # 直接使用官方支持
    
    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: "CacheDType | None") -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in ["bfloat16"]   # 支持bfp16

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def is_sparse(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability: "DeviceCapability") -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: "DeviceCapability",
    ) -> str | None:
        return "not implement"
    
class ToyFlashAttentionMetadata:
    num_actual_token: int
    max_query_len: int

class ToyFlashAttentionMetadataBuilder(AttentionMetadataBuilder):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ToyFlashAttentionMetadata:
        return ToyFlashAttentionMetadata()
    
    