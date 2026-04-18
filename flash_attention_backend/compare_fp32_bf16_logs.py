#!/usr/bin/env python3

import re
from collections import defaultdict


FP32_LOG = "/home/linf/code/cuda/flash_attention_backend/fp32.log"
BF16_LOG = "/home/linf/code/cuda/flash_attention_backend/bf16.log"
TURN_END = "======================= turn end ====================="

TURN_HASH_RE = re.compile(r"\[turn hash\]\s+(?P<hash>[0-9a-f]+)")
LINE_RE = re.compile(
    r'(?P<key>\(\d+,\s*\d+,\s*\d+\)\s+q=\d+\s+kv_chunk=\d+'
    r'(?:\s+kv_off=\d+\s+kv_seq=\d+)?\s+'
    r'(?:q_elem\[0\]\[0\]|k_elem\[0\]\[0\]|QK dot\[0\]\[0\]|score\[0\]\[0\]|max\[0\]\[0\]|sum\[0\]\[0\]|out_tile\[0\]\[0\])):\s+'
    r'(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)'
)


def load_turns(path):
    turns = []
    current = {
        "hash": None,
        "values": defaultdict(list),
    }

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            hash_match = TURN_HASH_RE.search(line)
            if hash_match:
                current["hash"] = hash_match.group("hash")
                continue

            if TURN_END in line:
                turns.append(current)
                current = {
                    "hash": None,
                    "values": defaultdict(list),
                }
                continue

            value_match = LINE_RE.search(line)
            if value_match:
                current["values"][value_match.group("key")].append(value_match.group("value"))

    if current["hash"] is not None or current["values"]:
        turns.append(current)

    return turns


fp32_turns = load_turns(FP32_LOG)
bf16_turns = load_turns(BF16_LOG)

print(f"fp32 turns: {len(fp32_turns)}")
print(f"bf16 turns: {len(bf16_turns)}")

if len(fp32_turns) != len(bf16_turns):
    print(f"[turn count mismatch] fp32={len(fp32_turns)} bf16={len(bf16_turns)}")

turn_count = min(len(fp32_turns), len(bf16_turns))

for turn_idx in range(turn_count):
    fp32_turn = fp32_turns[turn_idx]
    bf16_turn = bf16_turns[turn_idx]

    fp32_hash = fp32_turn["hash"]
    bf16_hash = bf16_turn["hash"]
    fp32 = fp32_turn["values"]
    bf16 = bf16_turn["values"]

    fp32_keys = set(fp32)
    bf16_keys = set(bf16)

    print(
        f"[turn {turn_idx}] "
        f"fp32_hash={fp32_hash} "
        f"bf16_hash={bf16_hash} "
        f"hash_equal={fp32_hash == bf16_hash} "
        f"fp32_keys={len(fp32_keys)} "
        f"bf16_keys={len(bf16_keys)} "
        f"only_fp32={len(fp32_keys - bf16_keys)} "
        f"only_bf16={len(bf16_keys - fp32_keys)}"
    )

    for key in sorted(fp32_keys | bf16_keys):
        if key not in fp32:
            print(f"[missing in fp32] turn={turn_idx} key={key}")
            continue
        if key not in bf16:
            print(f"[missing in bf16] turn={turn_idx} key={key}")
            continue

        if len(fp32[key]) != len(bf16[key]):
            print(
                f"[count mismatch] turn={turn_idx} key={key} "
                f"fp32={len(fp32[key])} bf16={len(bf16[key])}"
            )

        n = min(len(fp32[key]), len(bf16[key]))
        for i in range(n):
            if fp32[key][i] != bf16[key][i]:
                print(
                    f"[value mismatch] turn={turn_idx} key={key} "
                    f"occurrence={i} fp32={fp32[key][i]} bf16={bf16[key][i]}"
                )
