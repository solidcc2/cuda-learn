import unittest
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

_MODULE_PATH = Path(__file__).with_name("flash_attention_func.py")
_SPEC = spec_from_file_location("toy_flash_attention_func", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
flash_attn_varlen_without_block = _MOD.flash_attn_varlen_without_block
flash_attn_varlen_with_block = _MOD.flash_attn_varlen_with_block
flash_attn_varlen_with_block_cu = _MOD.flash_attn_varlen_with_block_cu_bf16

_VLLM_MATCH_TOP10_WORST_DUMPS = [
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

_STEP0_REPLAY_DUMP = "00000_with_block.pt"


def _test_verbose() -> bool:
    return os.environ.get("TOY_FLASH_ATTN_TEST_VERBOSE", "0") == "1"


def _test_print(*args, **kwargs) -> None:
    if _test_verbose():
        print(*args, **kwargs)


def _regression_min_pass_rate() -> float:
    return float(os.environ.get("TOY_FLASH_ATTN_REGRESSION_MIN_PASS_RATE", "0"))


def _debug_compare_reference_window(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    *,
    batch_id: int = 0,
    head_id: int = 0,
    q_token_id: int = 0,
    causal: bool,
    window_size: tuple[int, int] | None,
) -> None:
    q_start = cu_seqlens_q[batch_id].item()
    q_end = cu_seqlens_q[batch_id + 1].item()
    q_len = q_end - q_start
    kv_len = seqused_k[batch_id].item()
    block_size = k_cache.shape[1]
    q_row = q[q_start + q_token_id, head_id].float()
    phy_block_ids = [block_table[batch_id][seq_id // block_size] for seq_id in range(kv_len)]
    offsets = [seq_id % block_size for seq_id in range(kv_len)]
    k_dense = k_cache[phy_block_ids, offsets, head_id].float()
    v_dense = v_cache[phy_block_ids, offsets, head_id].float()

    cur_win = window_size
    if cur_win is None or cur_win == (-1, -1):
        cur_win = (kv_len, kv_len)
    if causal:
        cur_win = (cur_win[0], 0)
    k_q_offset = kv_len - q_len
    left = max(0, q_token_id + k_q_offset - cur_win[0])
    right = min(kv_len, q_token_id + k_q_offset + 1 + cur_win[1])

    scores = k_dense.matmul(q_row) / float(q.shape[2] ** 0.5)
    window_scores = scores[left:right]
    probs = window_scores.softmax(0)
    out = probs.unsqueeze(0).matmul(v_dense[left:right]).squeeze(0)

    _test_print(
        f"[DEBUG REF] batch={batch_id} head={head_id} q={q_token_id} "
        f"range=[{left},{right})"
    )
    _test_print("[DEBUG REF] q[:4]:", q_row[:4])
    _test_print("[DEBUG REF] k[:4,:4]:", k_dense[left:right][:4, :4])
    _test_print("[DEBUG REF] scores[:4]:", window_scores[:4])
    _test_print("[DEBUG REF] probs[:4]:", probs[:4])
    _test_print("[DEBUG REF] out[:4]:", out[:4])


def _require_fa2_cuda() -> None:
    # These tests are intended as black-box parity checks against the
    # official FA2 implementation, so we only run when CUDA + FA2 are visible.
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required for this test.")
    if not is_flash_attn_varlen_func_available():
        raise unittest.SkipTest(
            "Official flash_attn_varlen_func is not available in this environment."
        )
    if get_flash_attn_version() != 2:
        raise unittest.SkipTest("This test expects FA2 to be active.")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required for this test.")


def _require_dump_path_env() -> Path:
    dump_path = os.environ.get("TOY_FLASH_ATTN_REPLAY_DUMP")
    if not dump_path:
        raise unittest.SkipTest(
            "Set TOY_FLASH_ATTN_REPLAY_DUMP to a dumped .pt file or dump directory to run this replay test."
        )
    return Path(dump_path).expanduser().resolve()


def _iter_replay_dump_paths(dump_path: Path) -> list[Path]:
    if dump_path.is_file():
        return [dump_path]
    if dump_path.is_dir():
        paths = sorted(dump_path.glob("*_with_block.pt"))
        if not paths:
            raise unittest.SkipTest(f"No *_with_block.pt dump files found in {dump_path}")
        return paths
    raise unittest.SkipTest(f"Replay dump path does not exist: {dump_path}")


def _resolve_named_dump_paths(base_dir: Path, filenames: list[str]) -> list[Path]:
    if not base_dir.is_dir():
        raise unittest.SkipTest(f"Replay dump base dir does not exist: {base_dir}")
    paths = [base_dir / name for name in filenames]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise unittest.SkipTest(
            "Missing expected replay dump files:\n" + "\n".join(missing)
        )
    return paths


def _require_replay_dump_payload(payload: dict, dump_path: Path) -> None:
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
        raise AssertionError(
            f"Replay dump {dump_path} is missing required keys: {missing}. "
            "Please regenerate dumps with the current dump format."
        )


def _make_inputs(
    q_lens: list[int],
    k_lens: list[int] | None = None,
    num_heads: int = 2,
    head_dim: int = 16,
    dtype: torch.dtype = torch.float16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
]:
    # `q_lens` and `k_lens` are per-request lengths. When `k_lens` is longer
    # than `q_lens`, tests assume tail-alignment: q[-1] aligns with kv[-1].
    if k_lens is None:
        k_lens = q_lens

    total_q = sum(q_lens)
    total_k = sum(k_lens)
    device = "cuda"
    q = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_k, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_k, num_heads, head_dim, device=device, dtype=dtype)

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


def _make_block_cache(
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    k_lens: list[int],
    block_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Pack dense K/V into a simple paged cache used by the toy `with_block`
    # path. `block_table[batch_id, logical_block] = physical_block_id`.
    num_heads = k_dense.shape[1]
    head_dim = k_dense.shape[2]
    blocks_per_seq = [(k_len + block_size - 1) // block_size for k_len in k_lens]
    total_blocks = sum(blocks_per_seq)
    max_blocks_per_seq = max(blocks_per_seq)

    k_cache = torch.zeros(
        total_blocks,
        block_size,
        num_heads,
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

    # Shuffle physical block ids so the test exercises logical->physical
    # translation instead of accidentally relying on identity mapping.
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


def _run_official(
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
    # Official FA2 reference path.
    return flash_attn_varlen_func(
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


def _run_toy(
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
    # Dense / non-paged toy path.
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


def _run_toy_with_block(
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
    # Paged KV toy path. `seqused_k` is interpreted as per-request valid KV
    # length, and the toy implementation uses the same tail-alignment rule.
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


def _run_toy_with_block_cu(
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
    return flash_attn_varlen_with_block_cu(
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


def _require_with_block_cu_launch_constraints(head_dim: int) -> None:
    # v4 maps blockDim.x = ceil(head_dim / K_X_STRIDE) with K_X_STRIDE=4.
    # The kernel asserts blockDim.x is a power of two and that
    # q_chunk_size(8) * blockDim.x is warp-aligned.
    block_dim_x = (head_dim + 3) // 4
    if block_dim_x & (block_dim_x - 1) != 0 or (8 * block_dim_x) % 32 != 0:
        raise unittest.SkipTest(
            "with_block_cu launch constraints require "
            "ceil(head_dim / 4) to be a power of two and "
            "8 * ceil(head_dim / 4) to be warp-aligned."
        )


def _run_case_with_block_cu(
    q_lens: list[int],
    k_lens: list[int] | None,
    causal: bool,
    window_size: tuple[int, int] | None,
    num_heads: int = 2,
    head_dim: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    _require_with_block_cu_launch_constraints(head_dim)

    case_desc = (
        f"q_lens={q_lens}, k_lens={k_lens}, causal={causal}, "
        f"window_size={window_size}, num_heads={num_heads}, "
        f"head_dim={head_dim}, dtype={dtype}, use_block_cu=True"
    )
    _test_print(f"[RUN ] {case_desc}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    q, k, v, cu_seqlens_q, _, max_seqlen_q, max_seqlen_k = _make_inputs(
        q_lens=q_lens,
        k_lens=k_lens,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    k_cache, v_cache, block_table = _make_block_cache(
        k_dense=k,
        v_dense=v,
        k_lens=k_lens if k_lens is not None else q_lens,
    )
    seqused_k = torch.tensor(
        k_lens if k_lens is not None else q_lens,
        device=q.device,
        dtype=torch.int32,
    )

    out_ref = _run_toy_with_block(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
    )
    _debug_compare_reference_window(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        block_table=block_table,
        batch_id=0,
        head_id=0,
        q_token_id=0,
        causal=causal,
        window_size=window_size,
    )
    out_cu = _run_toy_with_block_cu(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
    )

    return out_ref, out_cu, case_desc


def _assert_close(
    q_lens: list[int],
    k_lens: list[int] | None,
    causal: bool,
    window_size: tuple[int, int] | None,
    num_heads: int = 2,
    head_dim: int = 16,
    dtype: torch.dtype = torch.float16,
    seed: int = 0,
    use_block: bool = False,
) -> None:
    # Compare toy implementation against official FA2 on exactly the same
    # tensors and masking settings.
    case_desc = (
        f"q_lens={q_lens}, k_lens={k_lens}, causal={causal}, "
        f"window_size={window_size}, num_heads={num_heads}, "
        f"head_dim={head_dim}, use_block={use_block}"
    )
    _test_print(f"[RUN ] {case_desc}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = _make_inputs(
        q_lens=q_lens,
        k_lens=k_lens,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    out_ref = _run_official(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        window_size=window_size,
    )

    if use_block:
        # Build a paged cache view for the same dense K/V, then compare the toy
        # block path against the dense official FA2 output.
        k_cache, v_cache, block_table = _make_block_cache(
            k_dense=k,
            v_dense=v,
            k_lens=k_lens if k_lens is not None else q_lens,
        )
        out_toy = _run_toy_with_block(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            seqused_k=torch.tensor(
                k_lens if k_lens is not None else q_lens,
                device=q.device,
                dtype=torch.int32,
            ),
            causal=causal,
            window_size=window_size,
            block_table=block_table,
        )
    else:
        out_toy = _run_toy(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            window_size=window_size,
        )

    assert out_toy.shape == out_ref.shape
    assert torch.allclose(out_toy, out_ref, atol=2e-3, rtol=2e-3), (
        f"Mismatch with q_lens={q_lens}, k_lens={k_lens}, causal={causal}, "
        f"window_size={window_size}, use_block={use_block}"
    )
    _test_print(f"[PASS] {case_desc}")


def _assert_close_with_block_cu(
    q_lens: list[int],
    k_lens: list[int] | None,
    causal: bool,
    window_size: tuple[int, int] | None,
    num_heads: int = 2,
    head_dim: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> None:
    out_ref, out_cu, case_desc = _run_case_with_block_cu(
        q_lens=q_lens,
        k_lens=k_lens,
        causal=causal,
        window_size=window_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        seed=seed,
    )

    diff = (out_cu.float() - out_ref.float()).abs()
    _test_print("max diff:", diff.max().item())
    _test_print("mean diff:", diff.mean().item())
    _test_print("out_ref[0:5]:", out_ref[:5])
    _test_print("out_cu[0:5]:", out_cu[:5])


    assert out_cu.shape == out_ref.shape
    assert torch.allclose(out_cu, out_ref, atol=2e-3, rtol=2e-3), (
        f"Mismatch with q_lens={q_lens}, k_lens={k_lens}, causal={causal}, "
        f"window_size={window_size}, use_block_cu=True"
    )
    _test_print(f"[PASS] {case_desc}")


def _report_block_cu_regression_rates(
    *,
    name: str,
    iterations: int,
    q_lens: list[int],
    k_lens: list[int] | None,
    causal: bool,
    window_size: tuple[int, int] | None,
    num_heads: int = 2,
    head_dim: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    thresholds: tuple[float, ...] = (1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2),
) -> None:
    pass_counts = {threshold: 0 for threshold in thresholds}
    max_diffs: list[float] = []
    mean_diffs: list[float] = []
    worst_cases: list[tuple[float, float, int]] = []

    for iteration in range(iterations):
        out_ref, out_cu, _ = _run_case_with_block_cu(
            q_lens=q_lens,
            k_lens=k_lens,
            causal=causal,
            window_size=window_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            seed=iteration,
        )
        assert out_cu.shape == out_ref.shape

        diff = (out_cu.float() - out_ref.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        max_diffs.append(max_diff)
        mean_diffs.append(mean_diff)
        worst_cases.append((max_diff, mean_diff, iteration))

        for threshold in thresholds:
            if torch.allclose(out_cu, out_ref, atol=threshold, rtol=threshold):
                pass_counts[threshold] += 1

    worst_cases.sort(reverse=True)
    sorted_max_diffs = sorted(max_diffs)
    p50 = sorted_max_diffs[int(0.50 * (iterations - 1))]
    p90 = sorted_max_diffs[int(0.90 * (iterations - 1))]
    p99 = sorted_max_diffs[int(0.99 * (iterations - 1))]
    avg_mean_diff = sum(mean_diffs) / iterations

    print(f"[REGRESSION] {name}")
    print(
        "  "
        f"iterations={iterations} q_lens={q_lens} k_lens={k_lens} "
        f"causal={causal} window_size={window_size} "
        f"head_dim={head_dim} dtype={dtype}"
    )
    print(
        "  "
        f"max_diff: min={min(max_diffs):.6g} p50={p50:.6g} "
        f"p90={p90:.6g} p99={p99:.6g} max={max(max_diffs):.6g}"
    )
    print(f"  mean_diff_avg={avg_mean_diff:.6g}")
    for threshold in thresholds:
        passed = pass_counts[threshold]
        failed = iterations - passed
        pass_rate = passed / iterations
        fail_rate = failed / iterations
        print(
            "  "
            f"threshold atol=rtol={threshold:g}: "
            f"pass={passed} fail={failed} "
            f"pass_rate={pass_rate:.2%} fail_rate={fail_rate:.2%}"
        )
        min_pass_rate = _regression_min_pass_rate()
        if min_pass_rate > 0:
            assert pass_rate >= min_pass_rate, (
                f"{name} pass_rate={pass_rate:.2%} below "
                f"TOY_FLASH_ATTN_REGRESSION_MIN_PASS_RATE={min_pass_rate:.2%} "
                f"at threshold={threshold:g}"
            )

    print("  worst seeds:")
    for max_diff, mean_diff, iteration in worst_cases[:5]:
        print(
            "  "
            f"seed={iteration} max_diff={max_diff:.6g} "
            f"mean_diff={mean_diff:.6g}"
        )


def _assert_dump_replay_close(dump_path: Path) -> None:
    payload = torch.load(dump_path, map_location="cpu")
    _require_replay_dump_payload(payload, dump_path)
    q = payload["q"].cuda()
    k = payload["k"].cuda()
    v = payload["v"].cuda()
    cu_seqlens_q = payload["cu_seqlens_q"].cuda()
    seqused_k = payload["seqused_k"].cuda()
    block_table = payload["block_table"].cuda()
    out_ref = payload["result"].cuda()
    out_cu = flash_attn_varlen_with_block_cu(
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
    diff = (out_cu.float() - out_ref.float()).abs()
    _test_print(f"[REPLAY] dump={dump_path}")
    _test_print("max diff:", diff.max().item())
    _test_print("mean diff:", diff.mean().item())
    assert out_cu.shape == out_ref.shape
    assert torch.allclose(out_cu, out_ref, atol=2e-3, rtol=2e-3), (
        f"Mismatch when replaying dumped context from {dump_path}"
    )


class FlashAttentionFuncFa2ParityTest(unittest.TestCase):
    def setUp(self) -> None:
        _require_fa2_cuda()

    def test_without_block_matches_fa2_full_attention(self) -> None:
        # Basic dense varlen self-attention.
        _assert_close(q_lens=[3, 5], k_lens=None, causal=False, window_size=None)

    def test_without_block_matches_fa2_causal_attention(self) -> None:
        _assert_close(q_lens=[3, 5], k_lens=None, causal=True, window_size=None)

    def test_without_block_matches_fa2_local_window(self) -> None:
        _assert_close(q_lens=[4, 6], k_lens=None, causal=False, window_size=(1, 1))

    def test_without_block_matches_fa2_causal_local_window(self) -> None:
        _assert_close(q_lens=[4, 6], k_lens=None, causal=True, window_size=(2, 0))

    def test_without_block_matches_fa2_more_varlen_batches(self) -> None:
        _assert_close(q_lens=[1, 3, 7, 2], k_lens=None, causal=False, window_size=None)

    def test_without_block_matches_fa2_longer_sequences(self) -> None:
        _assert_close(q_lens=[8, 11], k_lens=None, causal=True, window_size=None)

    def test_without_block_matches_fa2_tail_aligned_suffix_query(self) -> None:
        # Dense path with q_len < k_len. This is the important "tail aligned"
        # scenario where q[-1] is interpreted as aligned with kv[-1].
        _assert_close(
            q_lens=[2, 3],
            k_lens=[5, 7],
            causal=True,
            window_size=None,
        )

    def test_with_block_matches_fa2_full_attention(self) -> None:
        # Same semantics as the dense test, but K/V are read from paged cache.
        _assert_close(
            q_lens=[3, 5],
            k_lens=None,
            causal=False,
            window_size=None,
            use_block=True,
        )

    def test_with_block_matches_fa2_causal_attention(self) -> None:
        _assert_close(
            q_lens=[3, 5],
            k_lens=None,
            causal=True,
            window_size=None,
            use_block=True,
        )

    def test_with_block_matches_fa2_tail_aligned_suffix_query(self) -> None:
        # Paged-cache version of the tail-aligned suffix-query case.
        _assert_close(
            q_lens=[2, 3],
            k_lens=[5, 7],
            causal=True,
            window_size=None,
            use_block=True,
        )

    def test_without_block_matches_fa2_different_head_shapes(self) -> None:
        cases = [
            {"q_lens": [2, 5], "num_heads": 1, "head_dim": 8},
            {"q_lens": [3, 4], "num_heads": 4, "head_dim": 16},
            {"q_lens": [2, 6], "num_heads": 2, "head_dim": 32},
        ]
        for case in cases:
            with self.subTest(case=case):
                _assert_close(
                    q_lens=case["q_lens"],
                    k_lens=None,
                    causal=False,
                    window_size=None,
                    num_heads=case["num_heads"],
                    head_dim=case["head_dim"],
                )

    def test_without_block_matches_fa2_multiple_window_configs(self) -> None:
        cases = [
            {"q_lens": [5, 5], "causal": False, "window_size": (0, 0)},
            {"q_lens": [5, 5], "causal": False, "window_size": (2, 1)},
            {"q_lens": [5, 5], "causal": True, "window_size": (3, 0)},
        ]
        for case in cases:
            with self.subTest(case=case):
                _assert_close(
                    q_lens=case["q_lens"],
                    k_lens=None,
                    causal=case["causal"],
                    window_size=case["window_size"],
                )


class FlashAttentionFuncCuKernelParityTest(unittest.TestCase):
    def setUp(self) -> None:
        _require_cuda()

    def test_with_block_cu_matches_python_causal_attention_bf16(self) -> None:
        _assert_close_with_block_cu(
            q_lens=[3, 5],
            k_lens=None,
            causal=True,
            window_size=None,
        )

    def test_with_block_cu_matches_python_full_attention_bf16(self) -> None:
        _assert_close_with_block_cu(
            q_lens=[3, 5],
            k_lens=None,
            causal=False,
            window_size=None,
        )

    def test_with_block_cu_known_negative_non_64_case(self) -> None:
        _assert_close_with_block_cu(
            q_lens=[3, 4],
            k_lens=None,
            causal=False,
            window_size=None,
            num_heads=4,
            head_dim=16,
        )


class FlashAttentionFuncCuKernelHeadDim64RegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        _require_cuda()

    def test_with_block_cu_head_dim_64_minimal_no_mask_no_padding_seed_423(self) -> None:
        _assert_close_with_block_cu(
            q_lens=[4, 4],
            k_lens=None,
            causal=False,
            window_size=None,
            head_dim=64,
            dtype=torch.bfloat16,
            seed=423,
        )

    def test_with_block_cu_head_dim_64_minimal_no_mask_no_padding_regression(self) -> None:
        _report_block_cu_regression_rates(
            name="head_dim_64_minimal_no_mask_no_padding",
            iterations=1000,
            q_lens=[4, 4],
            k_lens=None,
            causal=False,
            window_size=None,
            head_dim=64,
            dtype=torch.bfloat16,
        )

    def test_with_block_cu_head_dim_64_tail_aligned_causal_local_window_regression(self) -> None:
        _report_block_cu_regression_rates(
            name="head_dim_64_tail_aligned_causal_local_window",
            iterations=1000,
            q_lens=[4, 9],
            k_lens=[8, 12],
            causal=True,
            window_size=(3, 0),
            head_dim=64,
            dtype=torch.bfloat16,
        )

    def test_with_block_cu_head_dim_64_regression_matrix(self) -> None:
        cases = [
            {
                "q_lens": [2, 6],
                "k_lens": None,
                "causal": False,
                "window_size": None,
                "seed": 0,
            },
            {
                "q_lens": [2, 6],
                "k_lens": None,
                "causal": True,
                "window_size": None,
                "seed": 0,
            },
            {
                "q_lens": [2, 6],
                "k_lens": None,
                "causal": True,
                "window_size": (2, 0),
                "seed": 0,
            },
            {
                "q_lens": [1, 7],
                "k_lens": None,
                "causal": False,
                "window_size": None,
                "seed": 1,
            },
            {
                "q_lens": [3, 5],
                "k_lens": [6, 9],
                "causal": True,
                "window_size": None,
                "seed": 0,
            },
        ]
        for case in cases:
            with self.subTest(case=case):
                _assert_close_with_block_cu(
                    q_lens=case["q_lens"],
                    k_lens=case["k_lens"],
                    causal=case["causal"],
                    window_size=case["window_size"],
                    head_dim=64,
                    dtype=torch.bfloat16,
                    seed=case["seed"],
                )

    def test_with_block_cu_matches_python_tail_aligned_suffix_query_bf16(self) -> None:
        _assert_close_with_block_cu(
            q_lens=[2, 3],
            k_lens=[5, 7],
            causal=True,
            window_size=None,
        )

    def test_with_block_cu_head_dim_64_outputs_are_finite(self) -> None:
        cases = [
            {"q_lens": [2, 6], "k_lens": None, "causal": False, "window_size": None, "seed": 0},
            {"q_lens": [2, 6], "k_lens": None, "causal": True, "window_size": (2, 0), "seed": 0},
            {"q_lens": [3, 5], "k_lens": [6, 9], "causal": True, "window_size": None, "seed": 1},
        ]
        for case in cases:
            with self.subTest(case=case):
                _, out_cu, case_desc = _run_case_with_block_cu(
                    q_lens=case["q_lens"],
                    k_lens=case["k_lens"],
                    causal=case["causal"],
                    window_size=case["window_size"],
                    head_dim=64,
                    dtype=torch.bfloat16,
                    seed=case["seed"],
                )
                self.assertTrue(
                    torch.isfinite(out_cu.float()).all().item(),
                    f"Non-finite values found in CUDA output for {case_desc}",
                )

    def test_with_block_cu_replay_dump_matches_python(self) -> None:
        for dump_path in _iter_replay_dump_paths(_require_dump_path_env()):
            with self.subTest(dump=str(dump_path)):
                _assert_dump_replay_close(dump_path)

    def test_with_block_cu_replay_top10_vllm_worst_dumps(self) -> None:
        dump_root = _require_dump_path_env()
        if dump_root.is_file():
            dump_root = dump_root.parent
        for dump_path in _resolve_named_dump_paths(dump_root, _VLLM_MATCH_TOP10_WORST_DUMPS):
            with self.subTest(dump=str(dump_path)):
                _assert_dump_replay_close(dump_path)

    def test_with_block_cu_replay_step0_matches_python(self) -> None:
        dump_root = _require_dump_path_env()
        if dump_root.is_file():
            dump_root = dump_root.parent
        dump_path = _resolve_named_dump_paths(dump_root, [_STEP0_REPLAY_DUMP])[0]
        _assert_dump_replay_close(dump_path)

    # @unittest.expectedFailure
    def test_with_block_cu_head_dim_coverage_targets(self) -> None:
        cases = [
            # {"q_lens": [2, 5], "num_heads": 1, "head_dim": 8},
            {"q_lens": [3, 4], "num_heads": 4, "head_dim": 16},
            {"q_lens": [2, 6], "num_heads": 2, "head_dim": 32},
            {"q_lens": [2, 6], "num_heads": 2, "head_dim": 64},
        ]
        for case in cases:
            with self.subTest(case=case):
                _assert_close_with_block_cu(
                    q_lens=case["q_lens"],
                    k_lens=None,
                    causal=False,
                    window_size=None,
                    num_heads=case["num_heads"],
                    head_dim=case["head_dim"],
                )


if __name__ == "__main__":
    _test_print("Running toy flash attention parity tests against official FA2...")
    unittest.main(verbosity=2)
