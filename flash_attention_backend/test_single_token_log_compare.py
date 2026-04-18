import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def _make_single_token_inputs(
    *,
    num_heads: int = 1,
    head_dim: int = 64,
    block_size: int = 4,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor | int | bool | tuple[int, int]]:
    device = "cuda"
    q = torch.randn(1, num_heads, head_dim, device=device, dtype=dtype)
    k_dense = torch.randn(1, num_heads, head_dim, device=device, dtype=dtype)
    v_dense = torch.randn(1, num_heads, head_dim, device=device, dtype=dtype)

    k_cache = torch.zeros(
        1, block_size, num_heads, head_dim, device=device, dtype=dtype
    )
    v_cache = torch.zeros_like(k_cache)
    k_cache[0, 0] = k_dense[0]
    v_cache[0, 0] = v_dense[0]
    block_table = torch.tensor([[0]], device=device, dtype=torch.int32)

    return {
        "q": q,
        "k": k_cache,
        "v": v_cache,
        "max_seqlen_q": 1,
        "cu_seqlens_q": torch.tensor([0, 1], device=device, dtype=torch.int32),
        "max_seqlen_k": 1,
        "seqused_k": torch.tensor([1], device=device, dtype=torch.int32),
        "causal": True,
        "window_size": (-1, -1),
        "block_table": block_table,
        "out": torch.empty_like(q),
        "layer": None,
    }


def _load_module():
    module_path = ROOT / "toy_flash_attn" / "flash_attention_func.py"
    spec = spec_from_file_location("toy_flash_attention_func_single_token", module_path)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["fp32", "bf16"], required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    mod = _load_module()

    torch.manual_seed(0)
    inputs = _make_single_token_inputs()
    if args.impl == "fp32":
        mod.flash_attn_varlen_with_block_cu_fp32(**inputs)
    else:
        mod.flash_attn_varlen_with_block_cu_bf16(**inputs)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
