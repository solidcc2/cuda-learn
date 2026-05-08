from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class OpCase:
    name: str
    q_lens: list[int]
    k_lens: list[int]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    causal: bool
    window_size: tuple[int, int] | None
    dtype: str
    seed: int


CASES: dict[str, OpCase] = {
    "qwen_like_b1_s128_h64": OpCase(
        name="qwen_like_b1_s128_h64",
        q_lens=[1],
        k_lens=[128],
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        causal=True,
        window_size=None,
        dtype="bfloat16",
        seed=0,
    ),
    "qwen_like_b4_s128_h64": OpCase(
        name="qwen_like_b4_s128_h64",
        q_lens=[1, 1, 1, 1],
        k_lens=[128, 128, 128, 128],
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        causal=True,
        window_size=None,
        dtype="bfloat16",
        seed=0,
    ),
    "qwen_like_b1_s512_h64": OpCase(
        name="qwen_like_b1_s512_h64",
        q_lens=[1],
        k_lens=[512],
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        causal=True,
        window_size=None,
        dtype="bfloat16",
        seed=0,
    ),
    "qwen_like_b4_s512_h64": OpCase(
        name="qwen_like_b4_s512_h64",
        q_lens=[1, 1, 1, 1],
        k_lens=[512, 512, 512, 512],
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        causal=True,
        window_size=None,
        dtype="bfloat16",
        seed=0,
    ),
    "qwen_like_b1_s2048_h64": OpCase(
        name="qwen_like_b1_s2048_h64",
        q_lens=[1],
        k_lens=[2048],
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        causal=True,
        window_size=None,
        dtype="bfloat16",
        seed=1,
    ),
    "mha_b1_s2048_h64_h14_kv14": OpCase(
        name="mha_b1_s2048_h64_h14_kv14",
        q_lens=[1],
        k_lens=[2048],
        num_heads=14,
        num_kv_heads=14,
        head_dim=64,
        causal=True,
        window_size=None,
        dtype="bfloat16",
        seed=1,
    ),
    "single_head_b1_s2048_h64_h1_kv1": OpCase(
        name="single_head_b1_s2048_h64_h1_kv1",
        q_lens=[1],
        k_lens=[2048],
        num_heads=1,
        num_kv_heads=1,
        head_dim=64,
        causal=True,
        window_size=None,
        dtype="bfloat16",
        seed=1,
    ),
    "gpt2_like_b1_s128_h64": OpCase(
        name="gpt2_like_b1_s128_h64",
        q_lens=[1],
        k_lens=[128],
        num_heads=12,
        num_kv_heads=12,
        head_dim=64,
        causal=True,
        window_size=None,
        dtype="bfloat16",
        seed=0,
    ),
    "gqa_case_b1_s128": OpCase(
        name="gqa_case_b1_s128",
        q_lens=[1],
        k_lens=[128],
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        causal=True,
        window_size=(64, 0),
        dtype="bfloat16",
        seed=7,
    ),
    "gqa_case_b4_s512": OpCase(
        name="gqa_case_b4_s512",
        q_lens=[1, 1, 1, 1],
        k_lens=[512, 384, 512, 256],
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        causal=True,
        window_size=None,
        dtype="bfloat16",
        seed=9,
    ),
}


def get_case(case_name: str) -> OpCase:
    return CASES[case_name]


def case_payload(case_name: str) -> dict[str, Any]:
    return asdict(get_case(case_name))
