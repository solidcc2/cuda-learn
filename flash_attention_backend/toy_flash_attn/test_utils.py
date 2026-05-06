from __future__ import annotations

import os
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import torch

from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as official_flash_attn_varlen_func

_MODULE_PATH = Path(__file__).with_name("flash_attention_func.py")
_SPEC = spec_from_file_location("toy_flash_attention_func", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

flash_attn_varlen_without_block = _MOD.flash_attn_varlen_without_block
flash_attn_varlen_with_block = _MOD.flash_attn_varlen_with_block
flash_attn_varlen_with_block_cu_bf16 = _MOD.flash_attn_varlen_with_block_cu_bf16

VLLM_MATCH_TOP10_WORST_DUMPS = [
    "00263_with_block.pt",
    "00303_with_block.pt",
    "00195_with_block.pt",
    "00009_with_block.pt",
    "00147_with_block.pt",
    "00339_with_block.pt",
    "00375_with_block.pt",
    "00123_with_block.pt",
    "00371_with_block.pt",
    "00357_with_block.pt",
]
STEP0_REPLAY_DUMP = "00000_with_block.pt"


def current_cuda_impl_version() -> str:
    return os.environ.get("TOY_FLASH_ATTN_CUDA_VERSION", "v7").lower()


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required for this test.")


def require_fa2_cuda() -> None:
    require_cuda()
    if not is_flash_attn_varlen_func_available():
        raise unittest.SkipTest("Official flash_attn_varlen_func is not available.")
    if get_flash_attn_version() != 2:
        raise unittest.SkipTest("This test expects FA2 to be active.")


def require_dump_path_env() -> Path:
    dump_path = os.environ.get("TOY_FLASH_ATTN_REPLAY_DUMP")
    if not dump_path:
        raise unittest.SkipTest("Set TOY_FLASH_ATTN_REPLAY_DUMP to run replay tests.")
    return Path(dump_path).expanduser().resolve()


def iter_replay_dump_paths(dump_path: Path) -> list[Path]:
    if dump_path.is_file():
        return [dump_path]
    if dump_path.is_dir():
        paths = sorted(dump_path.glob("*_with_block.pt"))
        if not paths:
            raise unittest.SkipTest(f"No *_with_block.pt dump files found in {dump_path}")
        return paths
    raise unittest.SkipTest(f"Replay dump path does not exist: {dump_path}")


def resolve_named_dump_paths(base_dir: Path, filenames: list[str]) -> list[Path]:
    if not base_dir.is_dir():
        raise unittest.SkipTest(f"Replay dump base dir does not exist: {base_dir}")
    paths = [base_dir / name for name in filenames]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise unittest.SkipTest("Missing expected replay dump files:\n" + "\n".join(missing))
    return paths


def require_replay_dump_payload(payload: dict[str, Any], dump_path: Path) -> None:
    required_keys = {
        "q",
        "k",
        "v",
        "max_seqlen_q",
        "cu_seqlens_q",
        "max_seqlen_k",
        "seqused_k",
        "causal",
        "window_size",
        "block_table",
        "result",
    }
    missing = sorted(required_keys - payload.keys())
    if missing:
        raise AssertionError(f"Replay dump {dump_path} missing keys: {missing}")


def make_inputs(
    q_lens: list[int],
    k_lens: list[int] | None = None,
    *,
    num_heads: int = 2,
    num_kv_heads: int | None = None,
    head_dim: int = 16,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    if k_lens is None:
        k_lens = q_lens
    if num_kv_heads is None:
        num_kv_heads = num_heads

    total_q = sum(q_lens)
    total_k = sum(k_lens)
    q = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_k, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_k, num_kv_heads, head_dim, device=device, dtype=dtype)

    cu_seqlens_q = [0]
    for seq_len in q_lens:
        cu_seqlens_q.append(cu_seqlens_q[-1] + seq_len)
    cu_seqlens_k = [0]
    for seq_len in k_lens:
        cu_seqlens_k.append(cu_seqlens_k[-1] + seq_len)

    return (
        q,
        k,
        v,
        torch.tensor(cu_seqlens_q, device=device, dtype=torch.int32),
        torch.tensor(cu_seqlens_k, device=device, dtype=torch.int32),
        max(q_lens),
        max(k_lens),
    )


def make_block_cache(
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    k_lens: list[int],
    *,
    block_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_kv_heads = k_dense.shape[1]
    head_dim = k_dense.shape[2]
    blocks_per_seq = [(k_len + block_size - 1) // block_size for k_len in k_lens]
    total_blocks = sum(blocks_per_seq)
    max_blocks_per_seq = max(blocks_per_seq)

    k_cache = torch.zeros(
        total_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=k_dense.device,
        dtype=k_dense.dtype,
    )
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.full(
        (len(k_lens), max_blocks_per_seq),
        -1,
        device=k_dense.device,
        dtype=torch.int32,
    )

    physical_ids = list(range(total_blocks))
    physical_ids = physical_ids[::2] + physical_ids[1::2]

    dense_start = 0
    next_block = 0
    for batch_id, k_len in enumerate(k_lens):
        for logical_block in range(blocks_per_seq[batch_id]):
            physical_block = physical_ids[next_block]
            next_block += 1
            block_table[batch_id, logical_block] = physical_block

            token_start = dense_start + logical_block * block_size
            token_end = min(token_start + block_size, dense_start + k_len)
            valid = token_end - token_start
            k_cache[physical_block, :valid] = k_dense[token_start:token_end]
            v_cache[physical_block, :valid] = v_dense[token_start:token_end]
        dense_start += k_len

    return k_cache, v_cache, block_table


def require_with_block_cu_launch_constraints(head_dim: int) -> None:
    block_dim_x = (head_dim + 3) // 4
    if block_dim_x & (block_dim_x - 1) != 0 or (8 * block_dim_x) % 32 != 0:
        raise unittest.SkipTest(
            "with_block_cu launch constraints require ceil(head_dim / 4) power-of-two and warp-aligned"
        )


def run_official_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    window_size: tuple[int, int] | None,
) -> torch.Tensor:
    return official_flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        causal=causal,
        window_size=list(window_size) if window_size is not None else None,
        fa_version=2,
    )


def run_official(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    seqused_k: torch.Tensor,
    causal: bool,
    window_size: tuple[int, int] | None,
    block_table: torch.Tensor,
) -> torch.Tensor:
    return official_flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=None,
        seqused_k=seqused_k,
        causal=causal,
        window_size=list(window_size) if window_size is not None else None,
        block_table=block_table,
        fa_version=2,
    )


def run_toy_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    window_size: tuple[int, int] | None,
) -> torch.Tensor:
    return flash_attn_varlen_without_block(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        causal=causal,
        window_size=window_size,
    )


def run_toy_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    seqused_k: torch.Tensor,
    causal: bool,
    window_size: tuple[int, int] | None,
    block_table: torch.Tensor,
) -> torch.Tensor:
    return flash_attn_varlen_with_block(
        q=q,
        k=k_cache,
        v=v_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
    )


def run_toy_paged_cuda(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    seqused_k: torch.Tensor,
    causal: bool,
    window_size: tuple[int, int] | None,
    block_table: torch.Tensor,
) -> torch.Tensor:
    return flash_attn_varlen_with_block_cu_bf16(
        q=q,
        k=k_cache,
        v=v_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
    )


def replay_dump_close(dump_path: Path) -> None:
    payload = torch.load(dump_path, map_location="cpu")
    require_replay_dump_payload(payload, dump_path)
    q = payload["q"].cuda()
    k = payload["k"].cuda()
    v = payload["v"].cuda()
    cu_seqlens_q = payload["cu_seqlens_q"].cuda()
    seqused_k = payload["seqused_k"].cuda()
    block_table = payload["block_table"].cuda()
    out_ref = payload["result"].cuda()
    out_cu = flash_attn_varlen_with_block_cu_bf16(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=payload["max_seqlen_q"],
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=payload["max_seqlen_k"],
        seqused_k=seqused_k,
        causal=payload["causal"],
        window_size=payload["window_size"],
        block_table=block_table,
    )
    assert torch.allclose(out_cu, out_ref, atol=2e-3, rtol=2e-3), (
        f"Mismatch when replaying dumped context from {dump_path}"
    )
