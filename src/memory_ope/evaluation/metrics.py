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


def summarize(values: dict[str, float]) -> dict[str, float]:
    arr = np.array([v for v in values.values() if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
