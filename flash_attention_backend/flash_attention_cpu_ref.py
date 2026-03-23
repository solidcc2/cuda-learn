import math
from typing import Optional

import torch


def flash_attention_cpu_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Minimal reference implementation.

    Expected shape:
    - q: [B, H, S, D]
    - k: [B, H, S, D]
    - v: [B, H, S, D]
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be 4D tensors shaped [B, H, S, D]")

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have the same shape in this minimal version")

    _, _, seq_len, head_dim = q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    if causal:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out
