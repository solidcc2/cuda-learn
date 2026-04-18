#!/usr/bin/env python3

import re
from collections import defaultdict


FP32_LOG = "/home/linf/code/cuda/flash_attention_backend/fp32.log"
BF16_LOG = "/home/linf/code/cuda/flash_attention_backend/bf16.log"
TURN_END = "======================= turn end ====================="

TURN_HASH_RE = re.compile(r"\[turn hash\]\s+(?P<hash>[0-9a-f]+)")
OUTPUT_HASH_RE = re.compile(r"\[output hash\]\s+(?P<hash>[0-9a-f]+)")
VALUE_RE = re.compile(
    r"(?P<key>\(\d+,\s*\d+,\s*\d+\)\s+q=\d+\s+kv_chunk=\d+"
    r"(?:\s+kv_off=\d+\s+kv_seq=\d+)?\s+"
    r"(?:q_elem\[0\]\[0\]|k_elem\[0\]\[0\]|QK dot\[0\]\[0\]|score\[0\]\[0\]|max\[0\]\[0\]|sum\[0\]\[0\]|out_tile\[0\]\[0\])):\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def load_turns(path):
    turns = []
    current_turn_hash = None
    current_output_hash = None
    current_values = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = TURN_HASH_RE.search(line)
            if m:
                current_turn_hash = m.group("hash")
                continue

            m = OUTPUT_HASH_RE.search(line)
            if m:
                current_output_hash = m.group("hash")
                continue

            m = VALUE_RE.search(line)
            if m:
                current_values[m.group("key")].append(m.group("value"))
                continue

            if TURN_END in line:
                turns.append({
                    "turn_hash": current_turn_hash,
                    "output_hash": current_output_hash,
                    "values": current_values,
                })
                current_turn_hash = None
                current_output_hash = None
                current_values = defaultdict(list)

    if current_turn_hash is not None or current_output_hash is not None or current_values:
        turns.append({
            "turn_hash": current_turn_hash,
            "output_hash": current_output_hash,
            "values": current_values,
        })

    return turns


def build_hash_index(turns):
    by_hash = defaultdict(list)
    for src_idx, turn in enumerate(turns):
        by_hash[turn["turn_hash"]].append((src_idx, turn))
    return by_hash


def compare_turn_values(fp32_turn, bf16_turn, turn_hash):
    fp32 = fp32_turn["values"]
    bf16 = bf16_turn["values"]

    fp32_keys = set(fp32)
    bf16_keys = set(bf16)

    print(
        f"[turn value keys] turn_hash={turn_hash} "
        f"fp32_keys={len(fp32_keys)} bf16_keys={len(bf16_keys)} "
        f"only_fp32={len(fp32_keys - bf16_keys)} "
        f"only_bf16={len(bf16_keys - fp32_keys)}"
    )

    for key in sorted(fp32_keys | bf16_keys):
        if key not in fp32:
            print(f"[missing in fp32] turn_hash={turn_hash} key={key}")
            continue
        if key not in bf16:
            print(f"[missing in bf16] turn_hash={turn_hash} key={key}")
            continue

        if len(fp32[key]) != len(bf16[key]):
            print(
                f"[count mismatch] turn_hash={turn_hash} key={key} "
                f"fp32={len(fp32[key])} bf16={len(bf16[key])}"
            )

        n = min(len(fp32[key]), len(bf16[key]))
        for occurrence in range(n):
            if fp32[key][occurrence] != bf16[key][occurrence]:
                print(
                    f"[value mismatch] turn_hash={turn_hash} key={key} "
                    f"occurrence={occurrence} "
                    f"fp32={fp32[key][occurrence]} bf16={bf16[key][occurrence]}"
                )


fp32_turns = load_turns(FP32_LOG)
bf16_turns = load_turns(BF16_LOG)

print(f"fp32 turns: {len(fp32_turns)}")
print(f"bf16 turns: {len(bf16_turns)}")

fp32_by_hash = build_hash_index(fp32_turns)
bf16_by_hash = build_hash_index(bf16_turns)

fp32_hashes = set(fp32_by_hash)
bf16_hashes = set(bf16_by_hash)
common_hashes = sorted(fp32_hashes & bf16_hashes)
only_fp32_hashes = sorted(fp32_hashes - bf16_hashes)
only_bf16_hashes = sorted(bf16_hashes - fp32_hashes)

print(f"common turn hashes: {len(common_hashes)}")
print(f"only fp32 turn hashes: {len(only_fp32_hashes)}")
print(f"only bf16 turn hashes: {len(only_bf16_hashes)}")

for h in only_fp32_hashes:
    srcs = [src for src, _ in fp32_by_hash[h]]
    print(f"[turn hash only in fp32] hash={h} srcs={srcs}")

for h in only_bf16_hashes:
    srcs = [src for src, _ in bf16_by_hash[h]]
    print(f"[turn hash only in bf16] hash={h} srcs={srcs}")

pair_idx = 0
for h in common_hashes:
    fp32_list = fp32_by_hash[h]
    bf16_list = bf16_by_hash[h]

    if len(fp32_list) != len(bf16_list):
        print(
            f"[turn hash multiplicity mismatch] hash={h} "
            f"fp32={len(fp32_list)} bf16={len(bf16_list)}"
        )

    n = min(len(fp32_list), len(bf16_list))
    for i in range(n):
        fp32_src, fp32_turn = fp32_list[i]
        bf16_src, bf16_turn = bf16_list[i]
        output_equal = fp32_turn["output_hash"] == bf16_turn["output_hash"]
        print(
            f"[matched turn hash] pair={pair_idx} "
            f"fp32_src={fp32_src} bf16_src={bf16_src} "
            f"turn_hash={h} "
            f"fp32_output_hash={fp32_turn['output_hash']} "
            f"bf16_output_hash={bf16_turn['output_hash']} "
            f"output_equal={output_equal}"
        )
        compare_turn_values(fp32_turn, bf16_turn, h)
        pair_idx += 1
