"""Retrieval metrics and out-of-scope threshold calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def first_hit_rank(relevant: Sequence[bool]) -> int | None:
    """1-based rank of the first relevant result, or None if there is none."""
    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            return rank
    return None


def hit_rate(ranks: Sequence[int | None], k: int) -> float:
    return float(np.mean([rank is not None and rank <= k for rank in ranks]))


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    return float(np.mean([0.0 if rank is None else 1.0 / rank for rank in ranks]))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; behaves sensibly for small samples and 0 or 100%."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def rejection_rates(
    in_scope: np.ndarray, out_of_scope: np.ndarray, threshold: float
) -> tuple[float, float]:
    """Return (share of in-scope queries answered, share of out-of-scope queries rejected)."""
    return float(np.mean(in_scope >= threshold)), float(np.mean(out_of_scope < threshold))


def calibrate_threshold(in_scope: np.ndarray, out_of_scope: np.ndarray) -> float:
    """Pick the midpoint threshold that maximises balanced accuracy.

    Candidates sit between consecutive observed scores. Ties go to the lowest
    threshold, which answers more real questions.
    """
    scores = np.unique(np.concatenate([in_scope, out_of_scope]))
    candidates = (scores[:-1] + scores[1:]) / 2
    if len(candidates) == 0:
        return float(scores[0])
    balanced = [sum(rejection_rates(in_scope, out_of_scope, t)) / 2 for t in candidates]
    return float(candidates[int(np.argmax(balanced))])
