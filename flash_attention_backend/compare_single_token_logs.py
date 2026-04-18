#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict
from pathlib import Path


TURN_END = "======================= turn end ====================="

TURN_HASH_RE = re.compile(r"\[turn hash\]\s+(?P<hash>[0-9a-f]+)")
OUTPUT_HASH_RE = re.compile(r"\[output hash\]\s+(?P<hash>[0-9a-f]+)")
VALUE_RE = re.compile(
    r"(?P<key>\(\d+,\s*\d+,\s*\d+\)\s+q=\d+\s+kv_chunk=\d+"
    r"(?:\s+kv_off=\d+\s+kv_seq=\d+)?\s+"
    r"(?:q_elem\[\d+\]\[\d+\]|k_elem\[\d+\]\[\d+\]|QK dot\[\d+\]\[\d+\]|"
    r"score\[\d+\]\[\d+\]|max\[\d+\]\[0\]|sum\[\d+\]\[0\]|out_tile\[\d+\]\[\d+\])):\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def load_turn(path: Path) -> dict:
    turn_hash = None
    output_hash = None
    values = defaultdict(list)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = TURN_HASH_RE.search(line)
            if m:
                turn_hash = m.group("hash")
                continue

            m = OUTPUT_HASH_RE.search(line)
            if m:
                output_hash = m.group("hash")
                continue

            m = VALUE_RE.search(line)
            if m:
                values[m.group("key")].append(m.group("value"))
                continue

            if TURN_END in line:
                break

    return {
        "turn_hash": turn_hash,
        "output_hash": output_hash,
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-log", required=True)
    parser.add_argument("--bf16-log", required=True)
    args = parser.parse_args()

    fp32_turn = load_turn(Path(args.fp32_log))
    bf16_turn = load_turn(Path(args.bf16_log))

    print(f"fp32_turn_hash={fp32_turn['turn_hash']}")
    print(f"bf16_turn_hash={bf16_turn['turn_hash']}")
    print(f"turn_hash_equal={fp32_turn['turn_hash'] == bf16_turn['turn_hash']}")
    print(f"fp32_output_hash={fp32_turn['output_hash']}")
    print(f"bf16_output_hash={bf16_turn['output_hash']}")
    print(f"output_hash_equal={fp32_turn['output_hash'] == bf16_turn['output_hash']}")

    fp32_keys = set(fp32_turn["values"])
    bf16_keys = set(bf16_turn["values"])
    print(f"fp32_keys={len(fp32_keys)}")
    print(f"bf16_keys={len(bf16_keys)}")
    print(f"only_fp32_keys={len(fp32_keys - bf16_keys)}")
    print(f"only_bf16_keys={len(bf16_keys - fp32_keys)}")

    for key in sorted(fp32_keys | bf16_keys):
        if key not in fp32_turn["values"]:
            print(f"[missing in fp32] key={key}")
            continue
        if key not in bf16_turn["values"]:
            print(f"[missing in bf16] key={key}")
            continue

        fp32_vals = fp32_turn["values"][key]
        bf16_vals = bf16_turn["values"][key]
        if len(fp32_vals) != len(bf16_vals):
            print(
                f"[count mismatch] key={key} "
                f"fp32={len(fp32_vals)} bf16={len(bf16_vals)}"
            )

        n = min(len(fp32_vals), len(bf16_vals))
        for occurrence in range(n):
            if fp32_vals[occurrence] != bf16_vals[occurrence]:
                print(
                    f"[value mismatch] key={key} occurrence={occurrence} "
                    f"fp32={fp32_vals[occurrence]} bf16={bf16_vals[occurrence]}"
                )


if __name__ == "__main__":
    main()
