#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flash_attention_backend.bench.common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--case", default=None)
    parser.add_argument("--root-dir", type=Path, default=None)
    parser.add_argument("--cases", default=None)
    parser.add_argument("--versions", default="v7,v6,official")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()
    if args.root_dir is None and args.input_dir is None:
        parser.error("either --input-dir or --root-dir is required")
    if args.root_dir is not None and args.input_dir is not None:
        parser.error("--input-dir and --root-dir are mutually exclusive")
    if args.root_dir is None and not args.case:
        parser.error("--case is required when using --input-dir")
    return args


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _parse_number(value: str) -> float | None:
    text = value.strip().replace(",", "")
    if not text or text in {"N/A", "nan", "NaN"}:
        return None
    match = re.match(r"^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _convert_with_unit(value: float | None, unit: str, *, target: str) -> float | None:
    if value is None:
        return None
    unit_norm = unit.strip().lower()
    if target == "duration_us":
        if unit_norm in {"usecond", "us", "µs"}:
            return value
        if unit_norm in {"nsecond", "ns"}:
            return value / 1000.0
        if unit_norm in {"msecond", "ms"}:
            return value * 1000.0
        if unit_norm in {"second", "s"}:
            return value * 1_000_000.0
    if target == "shared_mem_kb":
        if unit_norm in {"kbyte", "kb"}:
            return value
        if unit_norm in {"byte", "bytes", "b"}:
            return value / 1024.0
    return value


def _read_ncu_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    lines = path.read_text(errors="replace").splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith('"ID"') and "Kernel Name" in line:
            header_idx = idx
            break
    if header_idx is None:
        return [], lines

    reader = csv.DictReader(lines[header_idx:])
    rows = []
    for row in reader:
        if not row:
            continue
        if not any((value or "").strip() for value in row.values()):
            continue
        rows.append({str(key): (value or "") for key, value in row.items() if key is not None})
    return rows, lines


def _pick_kernel_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    valid_rows = []
    for row in rows:
        kernel = row.get("Kernel Name", "").strip()
        duration_ns = _parse_number(row.get("gpu__time_duration.sum", ""))
        if kernel and duration_ns is not None:
            valid_rows.append((duration_ns, row))
    if valid_rows:
        valid_rows.sort(key=lambda item: item[0], reverse=True)
        return valid_rows[0][1]
    for row in rows:
        if row.get("Kernel Name", "").strip():
            return row
    return None


def _row_number(row: dict[str, str], aliases: list[str]) -> float | None:
    alias_norms = [_normalize(alias) for alias in aliases]
    normalized_items = [(_normalize(metric_name), value) for metric_name, value in row.items()]
    for alias in alias_norms:
        for metric_norm, value in normalized_items:
            if metric_norm == alias:
                parsed = _parse_number(value)
                if parsed is not None:
                    return parsed
    for alias in alias_norms:
        for metric_norm, value in normalized_items:
            if alias in metric_norm:
                parsed = _parse_number(value)
                if parsed is not None:
                    return parsed
    return None


def _row_text(row: dict[str, str], aliases: list[str]) -> str | None:
    alias_norms = [_normalize(alias) for alias in aliases]
    normalized_items = [(_normalize(metric_name), value) for metric_name, value in row.items()]
    for alias in alias_norms:
        for metric_norm, value in normalized_items:
            if metric_norm == alias:
                text = value.strip()
                if text:
                    return text
    for alias in alias_norms:
        for metric_norm, value in normalized_items:
            if alias in metric_norm:
                text = value.strip()
                if text:
                    return text
    return None


def _parse_triplet(text: str | None) -> list[int] | None:
    if text is None:
        return None
    matches = re.findall(r"\d+", text)
    if not matches:
        return None
    values = [int(value) for value in matches[:3]]
    while len(values) < 3:
        values.append(1)
    return values


def _labels_from_summary(summary: dict[str, Any], log_text: str) -> list[str]:
    labels: list[str] = []
    grid_size = summary.get("grid_size")
    if isinstance(grid_size, list) and len(grid_size) == 3 and grid_size[0] * grid_size[1] * grid_size[2] < 64:
        labels.append("underfilled_grid")

    occupancy = summary.get("achieved_occupancy_pct")
    if isinstance(occupancy, (int, float)) and occupancy < 40.0:
        labels.append("low_occupancy")

    eligible = summary.get("eligible_warps_per_scheduler")
    no_eligible = summary.get("no_eligible_pct")
    if (
        isinstance(eligible, (int, float)) and eligible < 1.0
    ) or (
        isinstance(no_eligible, (int, float)) and no_eligible > 20.0
    ):
        labels.append("scheduler_starvation_risk")

    dram_pct = summary.get("dram_throughput_pct")
    l2_hit = summary.get("l2_hit_rate")
    global_excessive = summary.get("global_memory_excessive_sectors")
    if (
        isinstance(dram_pct, (int, float)) and dram_pct > 50.0
        and isinstance(l2_hit, (int, float)) and l2_hit < 60.0
    ) or (
        isinstance(global_excessive, (int, float)) and global_excessive > 0
    ):
        labels.append("uncoalesced_global_access_risk")

    registers = summary.get("registers_per_thread")
    local_spill = summary.get("local_spill_requests")
    if (
        isinstance(registers, (int, float)) and registers > 160
    ) or (
        isinstance(local_spill, (int, float)) and local_spill > 0
    ):
        labels.append("local_spill_risk")

    bank_conflicts = summary.get("shared_bank_conflicts")
    if (
        isinstance(bank_conflicts, (int, float)) and bank_conflicts > 0
    ) or "bank conflict" in log_text.lower():
        labels.append("shared_bank_conflict_risk")

    return labels


def _summarize_version(case_name: str, version: str, input_dir: Path) -> dict[str, Any] | None:
    csv_path = input_dir / f"{version}_raw.csv"
    log_path = input_dir / f"{version}.log"
    meta_path = input_dir / f"input_meta_{version}.json"
    rep_path = input_dir / f"{version}.ncu-rep"
    if not csv_path.exists():
        return None

    rows, _raw_lines = _read_ncu_rows(csv_path)
    kernel_row = _pick_kernel_row(rows)
    kernel_name = kernel_row.get("Kernel Name", "").strip() if kernel_row is not None else None

    input_meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    cwd = Path.cwd().resolve()

    def _artifact_path(path: Path) -> str | None:
        if not path.exists():
            return None
        try:
            return str(path.resolve().relative_to(cwd))
        except ValueError:
            return path.name

    duration_ns = _row_number(kernel_row or {}, ["gpu__time_duration.sum", "gpu__time_duration.avg"])
    dram_bytes_per_s = _row_number(kernel_row or {}, ["dram__bytes.sum.per_second"])
    shared_mem_bytes = _row_number(
        kernel_row or {},
        [
            "launch__shared_mem_per_block_allocated",
            "launch__shared_mem_per_block",
            "launch__shared_mem_per_block_dynamic",
        ],
    )

    summary = {
        "version": version,
        "case_name": case_name,
        "kernel_name": kernel_name,
        "duration_us": duration_ns / 1000.0 if duration_ns is not None else None,
        "memory_throughput_gbs": (dram_bytes_per_s / 1e9) if dram_bytes_per_s is not None else None,
        "dram_throughput_pct": _row_number(
            kernel_row or {},
            ["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "FBSP.TriageSCG.dramc__throughput.avg.pct_of_peak_sustained_elapsed"],
        ),
        "l2_hit_rate": _row_number(
            kernel_row or {},
            ["lts__average_t_sector_hit_rate_realtime.pct", "lts__t_sector_hit_rate.pct", "l2 hit rate"],
        ),
        "achieved_occupancy_pct": _row_number(
            kernel_row or {},
            ["sm__warps_active.avg.pct_of_peak_sustained_active", "launch__occupancy_per_block_size"],
        ),
        "active_warps_per_scheduler": _row_number(
            kernel_row or {},
            ["smsp__warps_active.avg.per_cycle_active", "active warps per scheduler"],
        ),
        "eligible_warps_per_scheduler": _row_number(
            kernel_row or {},
            ["smsp__warps_eligible.avg.per_cycle_active", "eligible warps per scheduler"],
        ),
        "no_eligible_pct": _row_number(
            kernel_row or {},
            ["warpsampling:smsp__pcsamp_warps_issue_stalled_no_instructions", "smsp__pcsamp_warps_issue_stalled_no_instructions"],
        ),
        "registers_per_thread": _row_number(
            kernel_row or {},
            ["launch__registers_per_thread", "registers per thread"],
        ),
        "dynamic_shared_mem_per_block_kb": (shared_mem_bytes / 1024.0) if shared_mem_bytes is not None else None,
        "grid_size": _parse_triplet(_row_text(kernel_row or {}, ["Grid Size", "launch__grid_size"])),
        "block_size": _parse_triplet(_row_text(kernel_row or {}, ["Block Size", "launch__block_size"])),
        "shared_bank_conflicts": _row_number(
            kernel_row or {},
            [
                "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
                "smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
                "derived__memory_l1_conflicts_shared_nway",
            ],
        ),
        "global_memory_excessive_sectors": _row_number(
            kernel_row or {},
            ["derived__memory_l2_theoretical_sectors_global_excessive"],
        ),
        "local_spill_requests": _row_number(
            kernel_row or {},
            ["derived__local_spilling_requests", "sass__inst_executed_register_spilling"],
        ),
        "input_meta": input_meta,
        "artifacts": {
            "report": _artifact_path(rep_path),
            "raw_csv": _artifact_path(csv_path),
            "log": _artifact_path(log_path),
            "input_meta": _artifact_path(meta_path),
        },
    }
    summary["labels"] = _labels_from_summary(summary, log_text)
    return summary


def _comparison_summary(per_version: list[dict[str, Any]]) -> dict[str, Any]:
    by_version = {item["version"]: item for item in per_version}
    comparison: dict[str, Any] = {"versions": [item["version"] for item in per_version]}
    primary_version = None
    if "v7" in by_version:
        primary_version = "v7"
    elif "v6" in by_version:
        primary_version = "v6"
    if primary_version is not None and "official" in by_version:
        primary = by_version[primary_version]
        official = by_version["official"]
        if isinstance(primary.get("duration_us"), (int, float)) and isinstance(official.get("duration_us"), (int, float)) and official["duration_us"]:
            comparison[f"{primary_version}_vs_official_duration_ratio"] = primary["duration_us"] / official["duration_us"]
        comparison["combined_labels"] = sorted(set(primary.get("labels", [])) | set(official.get("labels", [])))
    return comparison


def _summarize_case(case_name: str, input_dir: Path, versions: list[str]) -> dict[str, Any]:
    per_version = []
    for version in versions:
        item = _summarize_version(case_name, version, input_dir)
        if item is not None:
            per_version.append(item)
    return {
        "kind": "flash_attention_backend.ncu_summary",
        "generated_at_s": time.time(),
        "case_name": case_name,
        "versions": [item["version"] for item in per_version],
        "per_version": per_version,
        "comparison": _comparison_summary(per_version),
    }


def _render_case_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        f"## Case: {summary['case_name']}",
        "",
        f"- versions: `{', '.join(summary['versions'])}`",
        "",
        "| version | kernel | duration(us) | dram % | l2 hit % | occupancy % | eligible warps/sched | shared bank conflicts | global excessive sectors | labels |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in summary["per_version"]:
        lines.append(
            "| {version} | `{kernel}` | {duration} | {dram} | {l2} | {occ} | {eligible} | {bank_conflicts} | {global_excessive} | {labels} |".format(
                version=item["version"],
                kernel=item.get("kernel_name") or "未采集",
                duration=item.get("duration_us") if item.get("duration_us") is not None else "未采集",
                dram=item.get("dram_throughput_pct") if item.get("dram_throughput_pct") is not None else "未采集",
                l2=item.get("l2_hit_rate") if item.get("l2_hit_rate") is not None else "未采集",
                occ=item.get("achieved_occupancy_pct") if item.get("achieved_occupancy_pct") is not None else "未采集",
                eligible=item.get("eligible_warps_per_scheduler") if item.get("eligible_warps_per_scheduler") is not None else "未采集",
                bank_conflicts=item.get("shared_bank_conflicts") if item.get("shared_bank_conflicts") is not None else "未采集",
                global_excessive=item.get("global_memory_excessive_sectors") if item.get("global_memory_excessive_sectors") is not None else "未采集",
                labels=", ".join(item.get("labels", [])) or "无",
            )
        )
    lines.extend(
        [
            "",
            "### Comparison",
            "",
            "```json",
            json.dumps(summary["comparison"], indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return lines


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    if summary["kind"] == "flash_attention_backend.ncu_summary_collection":
        lines = [
            "# NCU Summary Collection",
            "",
            f"- generated_at_s: `{summary['generated_at_s']}`",
            f"- case_count: `{summary['case_count']}`",
            f"- versions: `{', '.join(summary['versions'])}`",
        ]
        if summary.get("selected_cases"):
            lines.append(f"- selected_cases: `{', '.join(summary['selected_cases'])}`")
        lines.append("")
        lines.extend(
            [
                "## Overview",
                "",
                "| case | version | duration(us) | dram % | l2 hit % | occupancy % | eligible warps/sched | shared bank conflicts | global excessive sectors |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for case_summary in summary["cases"]:
            for item in case_summary["per_version"]:
                lines.append(
                    "| {case_name} | {version} | {duration} | {dram} | {l2} | {occ} | {eligible} | {bank_conflicts} | {global_excessive} |".format(
                        case_name=case_summary["case_name"],
                        version=item["version"],
                        duration=item.get("duration_us") if item.get("duration_us") is not None else "未采集",
                        dram=item.get("dram_throughput_pct") if item.get("dram_throughput_pct") is not None else "未采集",
                        l2=item.get("l2_hit_rate") if item.get("l2_hit_rate") is not None else "未采集",
                        occ=item.get("achieved_occupancy_pct") if item.get("achieved_occupancy_pct") is not None else "未采集",
                        eligible=item.get("eligible_warps_per_scheduler") if item.get("eligible_warps_per_scheduler") is not None else "未采集",
                        bank_conflicts=item.get("shared_bank_conflicts") if item.get("shared_bank_conflicts") is not None else "未采集",
                        global_excessive=item.get("global_memory_excessive_sectors") if item.get("global_memory_excessive_sectors") is not None else "未采集",
                    )
                )
        lines.append("")
        for case_summary in summary["cases"]:
            lines.extend(_render_case_table(case_summary))
        return "\n".join(lines)

    lines = [
        f"# NCU Summary: {summary['case_name']}",
        "",
        f"- generated_at_s: `{summary['generated_at_s']}`",
        f"- versions: `{', '.join(summary['versions'])}`",
        "",
        "## Per Version",
        "",
        "| version | kernel | duration(us) | dram % | l2 hit % | occupancy % | eligible warps/sched | shared bank conflicts | global excessive sectors | labels |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in summary["per_version"]:
        lines.append(
            "| {version} | `{kernel}` | {duration} | {dram} | {l2} | {occ} | {eligible} | {bank_conflicts} | {global_excessive} | {labels} |".format(
                version=item["version"],
                kernel=item.get("kernel_name") or "未采集",
                duration=item.get("duration_us") if item.get("duration_us") is not None else "未采集",
                dram=item.get("dram_throughput_pct") if item.get("dram_throughput_pct") is not None else "未采集",
                l2=item.get("l2_hit_rate") if item.get("l2_hit_rate") is not None else "未采集",
                occ=item.get("achieved_occupancy_pct") if item.get("achieved_occupancy_pct") is not None else "未采集",
                eligible=item.get("eligible_warps_per_scheduler") if item.get("eligible_warps_per_scheduler") is not None else "未采集",
                bank_conflicts=item.get("shared_bank_conflicts") if item.get("shared_bank_conflicts") is not None else "未采集",
                global_excessive=item.get("global_memory_excessive_sectors") if item.get("global_memory_excessive_sectors") is not None else "未采集",
                labels=", ".join(item.get("labels", [])) or "无",
            )
        )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            "```json",
            json.dumps(summary["comparison"], indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    versions = [item.strip() for item in args.versions.split(",") if item.strip()]
    if args.root_dir is not None:
        output_json = args.output_json or (args.root_dir / "summary.json")
        output_md = args.output_md or (args.root_dir / "SUMMARY.md")
        selected_cases = None
        if args.cases:
            selected_cases = [item.strip() for item in args.cases.split(",") if item.strip()]
        case_dirs = [path for path in sorted(args.root_dir.iterdir()) if path.is_dir()]
        if selected_cases is not None:
            selected = set(selected_cases)
            case_dirs = [path for path in case_dirs if path.name in selected]
        case_summaries = []
        for case_dir in case_dirs:
            case_summary = _summarize_case(case_dir.name, case_dir, versions)
            if case_summary["per_version"]:
                case_summaries.append(case_summary)
        payload = {
            "kind": "flash_attention_backend.ncu_summary_collection",
            "generated_at_s": time.time(),
            "case_count": len(case_summaries),
            "versions": versions,
            "selected_cases": selected_cases,
            "cases": case_summaries,
        }
    else:
        output_json = args.output_json or (args.input_dir / "summary.json")
        output_md = args.output_md or (args.input_dir / "SUMMARY.md")
        payload = _summarize_case(args.case, args.input_dir, versions)
    write_json(output_json, payload)
    output_md.write_text(_render_summary_markdown(payload) + "\n")


if __name__ == "__main__":
    main()
