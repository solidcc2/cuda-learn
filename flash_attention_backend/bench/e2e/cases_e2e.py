from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class E2ECase:
    name: str
    model: str
    batch_size: int
    prompt_len: int
    max_tokens: int
    max_model_len: int


CASES: dict[str, E2ECase] = {
    "qwen_b1_t128": E2ECase(
        name="qwen_b1_t128",
        model="qwen",
        batch_size=1,
        prompt_len=256,
        max_tokens=128,
        max_model_len=2048,
    ),
    "qwen_b4_t128": E2ECase(
        name="qwen_b4_t128",
        model="qwen",
        batch_size=4,
        prompt_len=256,
        max_tokens=128,
        max_model_len=2048,
    ),
    "qwen_b1_t512": E2ECase(
        name="qwen_b1_t512",
        model="qwen",
        batch_size=1,
        prompt_len=768,
        max_tokens=512,
        max_model_len=4096,
    ),
    "qwen_b4_t512": E2ECase(
        name="qwen_b4_t512",
        model="qwen",
        batch_size=4,
        prompt_len=768,
        max_tokens=512,
        max_model_len=4096,
    ),
    "qwen_b1_t2048": E2ECase(
        name="qwen_b1_t2048",
        model="qwen",
        batch_size=1,
        prompt_len=2048,
        max_tokens=2048,
        max_model_len=8192,
    ),
    "gpt2_b1_t128": E2ECase(
        name="gpt2_b1_t128",
        model="gpt2",
        batch_size=1,
        prompt_len=256,
        max_tokens=128,
        max_model_len=1024,
    ),
}

SUITES: dict[str, list[str]] = {
    "smoke": [
        "qwen_b1_t128",
        "qwen_b4_t128",
        "gpt2_b1_t128",
    ],
    "report": [
        "qwen_b1_t512",
        "qwen_b4_t512",
    ],
    "stress": [
        "qwen_b1_t2048",
    ],
    "all": list(CASES),
}


def get_case(case_name: str) -> E2ECase:
    return CASES[case_name]


def case_payload(case_name: str) -> dict[str, int | str]:
    return asdict(get_case(case_name))
