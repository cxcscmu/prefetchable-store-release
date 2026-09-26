"""Dense reference frontier and the quantities read off it.

At each token budget the five dense models are fitted with L(N) = A + B * exp(-c * ln(N / N0)); a store model's
dense-equivalent size N_eq is the N at which the fitted frontier reaches its loss. Its parameter overhead is
P / N_eq and its exchange rate is r = ln(N_eq / B) / ln(P / B) for backbone size B and total size P, so that the
overhead equals (P / B)^(1 - r).
"""

from __future__ import annotations

import math

import numpy as np

N0 = 27.7e6
def fit_dense(points: dict) -> tuple:
    """Saturating frontier L(N) = A + B*exp(-c*ln(N/N0)).
    Grid over c, closed-form (A, B) by least squares at each c. Returns (A, B, c, sse)."""
    n = np.array(sorted(points), dtype=float)
    y = np.array([points[k] for k in sorted(points)], dtype=float)
    x = np.log(n / N0)
    best = None
    for c in np.linspace(0.02, 1.0, 981):
        e = np.exp(-c * x)
        M = np.stack([np.ones_like(e), e], 1)
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        sse = float(((M @ coef - y) ** 2).sum())
        if best is None or sse < best[3]:
            best = (float(coef[0]), float(coef[1]), float(c), sse)
    return best


def dense_equiv(loss: float, fit: tuple) -> float:
    A, B, c, _ = fit
    if loss <= A:
        return float("inf")  # below the fitted asymptote: no dense size matches
    return N0 * math.exp(-math.log((loss - A) / B) / c)


def exponent(loss: float, n_b: float, n_total: float, fit: tuple) -> float:
    return math.log(dense_equiv(loss, fit) / n_b) / math.log(n_total / n_b)


