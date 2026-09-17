"""Shared vectorized mechanics for the synthetic simulators.

Both simulators (task_difficulty, hitchhiker) share one outcome-generating
model: success probability follows a logistic link on the sum of
(utility - 0.5) deviations of the memories actually included, added to a
task-level base rate. This module implements that model plus top-M candidate
selection, rank-based propensities, and a forced-intervention oracle for
computing true per-memory causal values (E[Y|do(Z_m=1)] - E[Y|do(Z_m=0)],
holding everything else at its natural randomized distribution).
"""

from __future__ import annotations

import numpy as np


def top_m_candidate_info(
    similarities: np.ndarray, m: int, propensity_min: float, propensity_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized top-M selection + rank-linear propensity assignment.

    similarities: (N, n_memories) array; -inf marks memories that cannot be
    candidates for that row (e.g. specialists on easy tasks).

    Returns (candidate_mask, propensity), both (N, n_memories).
    """
    n_rows, n_memories = similarities.shape
    order = np.argsort(-similarities, axis=1)
    candidate_idx = order[:, :m]
    rows = np.arange(n_rows)[:, None]

    candidate_mask = np.zeros((n_rows, n_memories), dtype=bool)
    candidate_mask[rows, candidate_idx] = True

    if m > 1:
        rank_prop = propensity_max - (propensity_max - propensity_min) * (np.arange(m) / (m - 1))
    else:
        rank_prop = np.array([propensity_max])
    propensity = np.zeros((n_rows, n_memories), dtype=float)
    propensity[rows, candidate_idx] = rank_prop[None, :]
    return candidate_mask, propensity


def draw_inclusion(propensity: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independent Bernoulli(p) draw per (row, memory) cell; 0 wherever propensity is 0."""
    return (rng.random(propensity.shape) < propensity).astype(int)


def _logit(p: np.ndarray | float) -> np.ndarray:
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def success_probability(
    base_rate: np.ndarray, included: np.ndarray, utility: np.ndarray, beta: float
) -> np.ndarray:
    """P(Y=1) via a logistic link on base rate + beta * sum of included utility deviations."""
    deviation = utility - 0.5
    included_dev_sum = included @ deviation
    return _sigmoid(_logit(base_rate) + beta * included_dev_sum)


def sample_success(prob: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(prob.shape) < prob).astype(int)


def oracle_values(
    candidate_mask: np.ndarray,
    included: np.ndarray,
    base_rate: np.ndarray,
    utility: np.ndarray,
    beta: float,
) -> np.ndarray:
    """True causal value per memory: mean_{contexts where m is a candidate}[P(Y=1|do(Z_m=1)) - P(Y=1|do(Z_m=0))].

    Other memories' inclusion (included, as already drawn) is held fixed at
    its natural randomized value while Z_m is forced to 1 and then to 0 —
    the same forced-in/forced-out logic the ALFWorld pilot will later use.
    """
    n_memories = utility.shape[0]
    deviation = utility - 0.5
    logit_base = _logit(base_rate)
    values = np.full(n_memories, np.nan)

    for mem_idx in range(n_memories):
        mask = candidate_mask[:, mem_idx]
        if not mask.any():
            continue
        included_excl = included[mask].copy()
        included_excl[:, mem_idx] = 0
        dev_sum_excl = included_excl @ deviation
        base_logit = logit_base[mask]
        p1 = _sigmoid(base_logit + beta * (dev_sum_excl + deviation[mem_idx]))
        p0 = _sigmoid(base_logit + beta * dev_sum_excl)
        values[mem_idx] = float(p1.mean() - p0.mean())

    return values


def naive_observational_values(
    candidate_mask: np.ndarray,
    included: np.ndarray,
    base_rate: np.ndarray,
    utility: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Naive observational contrast per memory: mean_{candidates, Z_m=1}[P(Y=1)] -
    mean_{candidates, Z_m=0}[P(Y=1)], using each row's FACTUAL (not forced)
    inclusion draw. Unlike oracle_values, this does not intervene on Z_m --
    it just splits candidate rows by their observed Z_m. Under confounding
    (propensity correlated with a context variable that also drives the
    outcome), this generally differs from the causal oracle_values contrast,
    because conditioning on the factual Z_m=1 subset over-represents
    high-propensity contexts.
    """
    n_memories = utility.shape[0]
    prob = success_probability(base_rate, included, utility, beta)
    values = np.full(n_memories, np.nan)

    for mem_idx in range(n_memories):
        mask = candidate_mask[:, mem_idx]
        if not mask.any():
            continue
        z = included[:, mem_idx]
        pos_mask = mask & (z == 1)
        neg_mask = mask & (z == 0)
        if not pos_mask.any() or not neg_mask.any():
            continue
        values[mem_idx] = float(prob[pos_mask].mean() - prob[neg_mask].mean())

    return values
