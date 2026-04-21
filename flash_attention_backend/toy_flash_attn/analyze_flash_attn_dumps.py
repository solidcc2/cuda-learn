from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _numeric_prefix(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def _iter_dump_files(dump_dir: Path) -> list[Path]:
    files = sorted(dump_dir.glob("*.pt"), key=_numeric_prefix)
    if not files:
        raise FileNotFoundError(f"no dump files found under {dump_dir}")
    return files


def _to_float_tensor(value: Any) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    if value.dtype.is_floating_point:
        return value.float()
    return value.to(dtype=torch.float32)


def _tensor_diff(a: torch.Tensor | None, b: torch.Tensor | None) -> tuple[float, float] | None:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return None
    diff = (a - b).abs()
    return diff.max().item(), diff.mean().item()


def _resolve_output_key(payload: dict[str, Any]) -> str:
    if "result" not in payload:
        raise KeyError(
            "expected dump payload to contain 'result'. "
            f"Please regenerate dumps with the current format. keys={sorted(payload.keys())}"
        )
    return "result"


def _normalize_window(window_size: Any) -> tuple[int, int] | None:
    if window_size is None:
        return None
    if isinstance(window_size, tuple):
        return window_size
    if isinstance(window_size, list):
        return tuple(window_size)
    return window_size


def _kv_signature(payload: dict[str, Any]) -> tuple[Any, ...]:
    q = payload["q"]
    k = payload["k"]
    return (
        tuple(q.shape),
        tuple(k.shape),
        int(payload["max_seqlen_q"]),
        int(payload["max_seqlen_k"]),
        tuple(payload["cu_seqlens_q"].tolist()),
        tuple(payload["seqused_k"].tolist()),
        bool(payload["causal"]),
        _normalize_window(payload["window_size"]),
    )


@dataclass
class StepStats:
    index: int
    base_file: str
    other_file: str
    meta_match: bool
    layer_name_match: bool | None
    layer_idx_match: bool | None
    base_layer_name: str | None
    other_layer_name: str | None
    base_layer_idx: Any
    other_layer_idx: Any
    causal_match: bool
    window_match: bool
    q_max: float | None
    q_mean: float | None
    k_max: float | None
    k_mean: float | None
    v_max: float | None
    v_mean: float | None
    cu_max: float | None
    cu_mean: float | None
    seqused_max: float | None
    seqused_mean: float | None
    dense_k_max: float | None
    dense_k_mean: float | None
    dense_v_max: float | None
    dense_v_mean: float | None
    out_max: float | None
    out_mean: float | None


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"unexpected payload type at {path}: {type(payload)}")
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
        raise KeyError(
            f"dump {path} is missing required keys {missing}. "
            "Please regenerate dumps with the current format."
        )
    return payload


def _materialize_dense_kv(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    k = payload["k"]
    v = payload["v"]
    block_table = payload["block_table"]
    seqused_k = payload["seqused_k"]
    block_size = k.shape[1]
    dense_k_batches = []
    dense_v_batches = []
    for batch_id in range(seqused_k.numel()):
        kv_len = int(seqused_k[batch_id].item())
        if kv_len == 0:
            dense_k_batches.append(k.new_empty((0,) + k.shape[2:]))
            dense_v_batches.append(v.new_empty((0,) + v.shape[2:]))
            continue
        phy_block_ids = [int(block_table[batch_id][seq_id // block_size].item()) for seq_id in range(kv_len)]
        block_offsets = [seq_id % block_size for seq_id in range(kv_len)]
        dense_k_batches.append(k[phy_block_ids, block_offsets])
        dense_v_batches.append(v[phy_block_ids, block_offsets])
    return torch.cat(dense_k_batches, dim=0), torch.cat(dense_v_batches, dim=0)


def _build_step_stats(index: int, base_path: Path, other_path: Path) -> StepStats:
    base = _load_payload(base_path)
    other = _load_payload(other_path)
    base_debug_meta = base.get("debug_meta") or {}
    other_debug_meta = other.get("debug_meta") or {}
    base_layer_name = base_debug_meta.get("layer_name")
    other_layer_name = other_debug_meta.get("layer_name")
    base_layer_idx = base_debug_meta.get("layer_idx")
    other_layer_idx = other_debug_meta.get("layer_idx")
    base_out = _to_float_tensor(base[_resolve_output_key(base)])
    other_out = _to_float_tensor(other[_resolve_output_key(other)])
    q_diff = _tensor_diff(_to_float_tensor(base["q"]), _to_float_tensor(other["q"]))
    k_diff = _tensor_diff(_to_float_tensor(base["k"]), _to_float_tensor(other["k"]))
    v_diff = _tensor_diff(_to_float_tensor(base["v"]), _to_float_tensor(other["v"]))
    cu_diff = _tensor_diff(_to_float_tensor(base["cu_seqlens_q"]), _to_float_tensor(other["cu_seqlens_q"]))
    seqused_diff = _tensor_diff(_to_float_tensor(base["seqused_k"]), _to_float_tensor(other["seqused_k"]))
    base_dense_k, base_dense_v = _materialize_dense_kv(base)
    other_dense_k, other_dense_v = _materialize_dense_kv(other)
    dense_k_diff = _tensor_diff(_to_float_tensor(base_dense_k), _to_float_tensor(other_dense_k))
    dense_v_diff = _tensor_diff(_to_float_tensor(base_dense_v), _to_float_tensor(other_dense_v))
    out_diff = _tensor_diff(base_out, other_out)
    return StepStats(
        index=index,
        base_file=base_path.name,
        other_file=other_path.name,
        meta_match=_kv_signature(base) == _kv_signature(other),
        layer_name_match=(
            None if base_layer_name is None and other_layer_name is None
            else base_layer_name == other_layer_name
        ),
        layer_idx_match=(
            None if base_layer_idx is None and other_layer_idx is None
            else base_layer_idx == other_layer_idx
        ),
        base_layer_name=base_layer_name,
        other_layer_name=other_layer_name,
        base_layer_idx=base_layer_idx,
        other_layer_idx=other_layer_idx,
        causal_match=bool(base["causal"]) == bool(other["causal"]),
        window_match=_normalize_window(base["window_size"]) == _normalize_window(other["window_size"]),
        q_max=None if q_diff is None else q_diff[0],
        q_mean=None if q_diff is None else q_diff[1],
        k_max=None if k_diff is None else k_diff[0],
        k_mean=None if k_diff is None else k_diff[1],
        v_max=None if v_diff is None else v_diff[0],
        v_mean=None if v_diff is None else v_diff[1],
        cu_max=None if cu_diff is None else cu_diff[0],
        cu_mean=None if cu_diff is None else cu_diff[1],
        seqused_max=None if seqused_diff is None else seqused_diff[0],
        seqused_mean=None if seqused_diff is None else seqused_diff[1],
        dense_k_max=None if dense_k_diff is None else dense_k_diff[0],
        dense_k_mean=None if dense_k_diff is None else dense_k_diff[1],
        dense_v_max=None if dense_v_diff is None else dense_v_diff[0],
        dense_v_mean=None if dense_v_diff is None else dense_v_diff[1],
        out_max=None if out_diff is None else out_diff[0],
        out_mean=None if out_diff is None else out_diff[1],
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "0"
    if abs(value) >= 1e-3:
        return f"{value:.6f}"
    return f"{value:.3e}"


def _summarize_stats(stats: list[StepStats], *, input_threshold: float, output_threshold: float, top_k: int) -> str:
    lines: list[str] = []
    lines.append(f"matched steps: {len(stats)}")
    if not stats:
        return "\n".join(lines)

    first_meta_mismatch = next((s for s in stats if not s.meta_match), None)
    if first_meta_mismatch is None:
        lines.append("metadata signature: all matched")
    else:
        lines.append(
            "first metadata mismatch: "
            f"step={first_meta_mismatch.index} "
            f"base={first_meta_mismatch.base_file} other={first_meta_mismatch.other_file}"
        )

    first_causal_mismatch = next((s for s in stats if not s.causal_match), None)
    if first_causal_mismatch is None:
        lines.append("causal: all matched")
    else:
        lines.append(
            "first causal mismatch: "
            f"step={first_causal_mismatch.index} "
            f"base={first_causal_mismatch.base_file} other={first_causal_mismatch.other_file}"
        )

    first_window_mismatch = next((s for s in stats if not s.window_match), None)
    if first_window_mismatch is None:
        lines.append("window_size: all matched")
    else:
        lines.append(
            "first window_size mismatch: "
            f"step={first_window_mismatch.index} "
            f"base={first_window_mismatch.base_file} other={first_window_mismatch.other_file}"
        )

    first_layer_name_mismatch = next((s for s in stats if s.layer_name_match is False), None)
    if first_layer_name_mismatch is None:
        lines.append("layer_name: all matched or unavailable")
    else:
        lines.append(
            "first layer_name mismatch: "
            f"step={first_layer_name_mismatch.index} "
            f"base={first_layer_name_mismatch.base_layer_name} "
            f"other={first_layer_name_mismatch.other_layer_name}"
        )

    first_layer_idx_mismatch = next((s for s in stats if s.layer_idx_match is False), None)
    if first_layer_idx_mismatch is None:
        lines.append("layer_idx: all matched or unavailable")
    else:
        lines.append(
            "first layer_idx mismatch: "
            f"step={first_layer_idx_mismatch.index} "
            f"base={first_layer_idx_mismatch.base_layer_idx} "
            f"other={first_layer_idx_mismatch.other_layer_idx}"
        )

    first_input_drift = next(
        (
            s
            for s in stats
            if any(
                value is not None and value > input_threshold
                for value in (s.q_max, s.k_max, s.v_max)
            )
        ),
        None,
    )
    if first_input_drift is None:
        lines.append(f"first input drift > {input_threshold}: not found")
    else:
        lines.append(
            "first input drift: "
            f"step={first_input_drift.index} "
            f"q_max={_fmt(first_input_drift.q_max)} "
            f"k_max={_fmt(first_input_drift.k_max)} "
            f"v_max={_fmt(first_input_drift.v_max)}"
        )

    first_cu_drift = next(
        (s for s in stats if s.cu_max is not None and s.cu_max > input_threshold),
        None,
    )
    if first_cu_drift is None:
        lines.append(f"first cu_seqlens_q drift > {input_threshold}: not found")
    else:
        lines.append(
            "first cu_seqlens_q drift: "
            f"step={first_cu_drift.index} "
            f"cu_max={_fmt(first_cu_drift.cu_max)} "
            f"cu_mean={_fmt(first_cu_drift.cu_mean)}"
        )

    first_seqused_drift = next(
        (s for s in stats if s.seqused_max is not None and s.seqused_max > input_threshold),
        None,
    )
    if first_seqused_drift is None:
        lines.append(f"first seqused_k drift > {input_threshold}: not found")
    else:
        lines.append(
            "first seqused_k drift: "
            f"step={first_seqused_drift.index} "
            f"seqused_max={_fmt(first_seqused_drift.seqused_max)} "
            f"seqused_mean={_fmt(first_seqused_drift.seqused_mean)}"
        )

    first_dense_kv_drift = next(
        (
            s for s in stats
            if any(
                value is not None and value > input_threshold
                for value in (s.dense_k_max, s.dense_v_max)
            )
        ),
        None,
    )
    if first_dense_kv_drift is None:
        lines.append(f"first dense kv drift > {input_threshold}: not found")
    else:
        lines.append(
            "first dense kv drift: "
            f"step={first_dense_kv_drift.index} "
            f"dense_k_max={_fmt(first_dense_kv_drift.dense_k_max)} "
            f"dense_v_max={_fmt(first_dense_kv_drift.dense_v_max)}"
        )

    first_output_drift = next(
        (s for s in stats if s.out_max is not None and s.out_max > output_threshold),
        None,
    )
    if first_output_drift is None:
        lines.append(f"first output drift > {output_threshold}: not found")
    else:
        lines.append(
            "first output drift: "
            f"step={first_output_drift.index} "
            f"out_max={_fmt(first_output_drift.out_max)} "
            f"out_mean={_fmt(first_output_drift.out_mean)}"
        )

    valid_out = [s for s in stats if s.out_max is not None and s.out_mean is not None]
    if valid_out:
        avg_out_max = sum(s.out_max for s in valid_out if s.out_max is not None) / len(valid_out)
        avg_out_mean = sum(s.out_mean for s in valid_out if s.out_mean is not None) / len(valid_out)
        lines.append(
            "output diff average: "
            f"max={_fmt(avg_out_max)} mean={_fmt(avg_out_mean)}"
        )

    valid_dense_k = [s for s in stats if s.dense_k_max is not None and s.dense_k_mean is not None]
    if valid_dense_k:
        avg_dense_k_max = sum(s.dense_k_max for s in valid_dense_k if s.dense_k_max is not None) / len(valid_dense_k)
        avg_dense_k_mean = sum(s.dense_k_mean for s in valid_dense_k if s.dense_k_mean is not None) / len(valid_dense_k)
        lines.append(
            "dense K diff average: "
            f"max={_fmt(avg_dense_k_max)} mean={_fmt(avg_dense_k_mean)}"
        )

    valid_dense_v = [s for s in stats if s.dense_v_max is not None and s.dense_v_mean is not None]
    if valid_dense_v:
        avg_dense_v_max = sum(s.dense_v_max for s in valid_dense_v if s.dense_v_max is not None) / len(valid_dense_v)
        avg_dense_v_mean = sum(s.dense_v_mean for s in valid_dense_v if s.dense_v_mean is not None) / len(valid_dense_v)
        lines.append(
            "dense V diff average: "
            f"max={_fmt(avg_dense_v_max)} mean={_fmt(avg_dense_v_mean)}"
        )

    lines.append("")
    lines.append(f"top {top_k} output drift steps:")
    ranked = sorted(
        valid_out,
        key=lambda s: (
            -math.inf if s.out_max is None else s.out_max,
            -math.inf if s.out_mean is None else s.out_mean,
        ),
        reverse=True,
    )[:top_k]
    for s in ranked:
        lines.append(
            f"  step={s.index:04d} "
            f"out_max={_fmt(s.out_max)} out_mean={_fmt(s.out_mean)} "
            f"q_max={_fmt(s.q_max)} q_mean={_fmt(s.q_mean)} "
            f"layer_name={s.base_layer_name!r}/{s.other_layer_name!r} "
            f"layer_idx={s.base_layer_idx!r}/{s.other_layer_idx!r} "
            f"base={s.base_file} other={s.other_file}"
        )

    ranked_dense_kv = sorted(
        [s for s in stats if s.dense_k_max is not None or s.dense_v_max is not None],
        key=lambda s: max(
            -math.inf if s.dense_k_max is None else s.dense_k_max,
            -math.inf if s.dense_v_max is None else s.dense_v_max,
        ),
        reverse=True,
    )[:top_k]
    lines.append("")
    lines.append(f"top {top_k} dense kv drift steps:")
    for s in ranked_dense_kv:
        lines.append(
            f"  step={s.index:04d} "
            f"dense_k_max={_fmt(s.dense_k_max)} dense_k_mean={_fmt(s.dense_k_mean)} "
            f"dense_v_max={_fmt(s.dense_v_max)} dense_v_mean={_fmt(s.dense_v_mean)} "
            f"seqused_max={_fmt(s.seqused_max)} cu_max={_fmt(s.cu_max)} "
            f"base={s.base_file} other={s.other_file}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two flash attention dump directories and report where inputs "
            "or outputs start to drift."
        )
    )
    parser.add_argument("base_dir", type=Path, help="baseline dump directory")
    parser.add_argument("other_dir", type=Path, help="custom dump directory")
    parser.add_argument(
        "--input-threshold",
        type=float,
        default=1e-5,
        help="flag earliest step whose q/k/v max diff exceeds this threshold",
    )
    parser.add_argument(
        "--output-threshold",
        type=float,
        default=1e-3,
        help="flag earliest step whose output max diff exceeds this threshold",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="number of worst output-drift steps to print",
    )
    args = parser.parse_args()

    base_files = _iter_dump_files(args.base_dir)
    other_files = _iter_dump_files(args.other_dir)
    matched = min(len(base_files), len(other_files))

    print(f"base dump count: {len(base_files)}")
    print(f"other dump count: {len(other_files)}")
    print(f"comparing first {matched} steps by numeric order")
    print(f"input threshold: {args.input_threshold}")
    print(f"output threshold: {args.output_threshold}")
    print("")

    stats = [
        _build_step_stats(i, base_files[i], other_files[i])
        for i in range(matched)
    ]
    print(
        _summarize_stats(
            stats,
            input_threshold=args.input_threshold,
            output_threshold=args.output_threshold,
            top_k=args.top_k,
        )
    )


if __name__ == "__main__":
    main()
