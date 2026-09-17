"""Comparison metrics: Spearman correlation, bias, and variance of an
estimator's per-memory values against the oracle true values.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def spearman(estimates: dict[str, float], true_values: dict[str, float]) -> float:
    common = [mem_id for mem_id in true_values if mem_id in estimates]
    if len(common) < 2:
        return float("nan")
    x = [estimates[mem_id] for mem_id in common]
    y = [true_values[mem_id] for mem_id in common]
    rho, _p = spearmanr(x, y)
    return float(rho)


def bias_and_variance(
    estimates_per_seed: list[dict[str, float]], true_values: dict[str, float]
) -> tuple[dict[str, float], dict[str, float]]:
    bias: dict[str, float] = {}
    variance: dict[str, float] = {}
    for mem_id, true_v in true_values.items():
        vals = np.array(
            [est[mem_id] for est in estimates_per_seed if mem_id in est], dtype=float
        )
        if vals.size == 0:
            bias[mem_id] = float("nan")
            variance[mem_id] = float("nan")
            continue
        bias[mem_id] = float(np.mean(vals) - true_v)
        variance[mem_id] = float(np.var(vals))
    return bias, variance


def mean_absolute_error(estimates: dict[str, float], true_values: dict[str, float]) -> float:
    """Tie-free complement to Spearman: mean |estimate - true| across memories
    that are candidates in both. Note this is on whatever scale the estimator
    lives on -- Memory Worth (~P(success|included)) is not on the same scale
    as the causal contrast IPS/SNIPS/DR target, so a large MAE for it is
    partly a scale mismatch, not purely inaccuracy; still reported as asked,
    since it's informative about that mismatch itself."""
    common = [mem_id for mem_id in true_values if mem_id in estimates]
    if not common:
        return float("nan")
    diffs = np.array([abs(estimates[mem_id] - true_values[mem_id]) for mem_id in common])
    return float(diffs.mean())


def mean_signed_bias(estimates: dict[str, float], true_values: dict[str, float]) -> float:
    """Mean signed (estimate - true) across memories, for a single seed's
    estimate set (distinct from bias_and_variance, which averages ACROSS
    SEEDS per memory; this averages ACROSS MEMORIES for one seed)."""
    common = [mem_id for mem_id in true_values if mem_id in estimates]
    if not common:
        return float("nan")
    diffs = np.array([estimates[mem_id] - true_values[mem_id] for mem_id in common])
    return float(diffs.mean())


def summarize(values: dict[str, float]) -> dict[str, float]:
    arr = np.array([v for v in values.values() if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
