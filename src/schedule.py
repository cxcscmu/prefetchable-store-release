"""Learning-rate schedule: linear warmup, then cosine decay to floor_frac x peak."""

from __future__ import annotations

import math


def lr_multiplier(step: int, total: int, warmup: int, floor_frac: float, kind: str = "cosine") -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    if kind == "cosine":
        prog = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return floor_frac + (1.0 - floor_frac) * 0.5 * (1.0 + math.cos(math.pi * prog))
    raise ValueError(f"unknown schedule {kind!r}")
