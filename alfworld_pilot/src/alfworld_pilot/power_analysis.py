"""Redoes Part A.3's paired-design sample-size calculation with a MEASURED
rho (from ground_truth_runner.paired_correlation on real results) instead
of the simulator-derived one. Same formula as the main package's
ground_truth_power_analysis.py; kept as a small local copy rather than a
cross-venv import since it's a few lines of arithmetic, not shared state.
"""

from __future__ import annotations

import numpy as np

Z_ALPHA_2 = 1.9600
Z_BETA = 0.8416
POWER_CONST = (Z_ALPHA_2 + Z_BETA) ** 2


def n_pairs_needed(delta: float, var_y1: float, var_y0: float, rho: float) -> float:
    var_d = var_y1 + var_y0 - 2 * rho * np.sqrt(var_y1 * var_y0)
    return POWER_CONST * var_d / (delta ** 2)


def redo_with_measured_rho(measured: dict, target_deltas: list[float] = (0.05, 0.10, 0.15)) -> dict:
    rho = measured["rho"]
    var_y1, var_y0 = measured["var_y1"], measured["var_y0"]
    if np.isnan(rho):
        return {"error": "rho is nan (need variance in both arms' outcomes to estimate a correlation)", "measured": measured}
    return {
        "measured": measured,
        "pairs_needed_by_delta": {str(d): n_pairs_needed(d, var_y1, var_y0, rho) for d in target_deltas},
    }
