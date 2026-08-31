"""
Helpers for fast-screening HMO validation experiments.

These helpers intentionally keep the statistics lightweight:
the fast V1-V4 scripts aim to decide whether the mechanism looks
promising within a few hours, not to produce final paper tables.
"""
from __future__ import annotations

import numpy as np


def safe_mean(values) -> float:
    """Return the mean of a sequence, or 0.0 when empty."""
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()) if arr.size > 0 else 0.0


def paired_win_rate(lhs, rhs, count_ties_as_half: bool = True) -> float:
    """Compute the paired win rate for lhs against rhs."""
    if len(lhs) != len(rhs):
        raise ValueError("paired_win_rate expects sequences with equal length")
    if not lhs:
        return 0.0

    wins = 0.0
    for left, right in zip(lhs, rhs):
        if left > right:
            wins += 1.0
        elif left == right and count_ties_as_half:
            wins += 0.5
    return wins / len(lhs)


def top_bottom_split(values, top_frac: float = 0.3):
    """Return indices for the bottom and top quantile of a score vector."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    k = max(1, int(np.ceil(arr.size * top_frac)))
    order = np.argsort(arr)
    low = order[:k]
    high = order[-k:]
    return low, high
