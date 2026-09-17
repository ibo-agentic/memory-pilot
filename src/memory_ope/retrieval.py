"""Randomized top-M retrieval with known inclusion propensities.

For a task, candidates are ranked by similarity and the top M are kept. Each
candidate's inclusion propensity p_i is a deterministic linear function of its
similarity rank within that top-M set (most similar -> propensity_max, least
similar -> propensity_min), so p_i is always known exactly and always lies in
[propensity_min, propensity_max]. Inclusion is then an independent Bernoulli
draw per candidate using that p_i.
"""

from __future__ import annotations

import numpy as np


def select_top_m(similarities: dict[str, float], m: int) -> list[str]:
    """Return candidate ids sorted by similarity descending, truncated to m."""
    ranked = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)
    return [mem_id for mem_id, _ in ranked[:m]]


def assign_propensities(
    ranked_candidate_ids: list[str], propensity_min: float, propensity_max: float
) -> dict[str, float]:
    """Linearly scale propensity by rank: rank 0 (most similar) -> max, last -> min."""
    n = len(ranked_candidate_ids)
    if n == 1:
        return {ranked_candidate_ids[0]: propensity_max}
    span = propensity_max - propensity_min
    return {
        mem_id: propensity_max - span * (rank / (n - 1))
        for rank, mem_id in enumerate(ranked_candidate_ids)
    }


def draw_inclusion(propensities: dict[str, float], rng: np.random.Generator) -> dict[str, int]:
    """Independent Bernoulli(p_i) draw for each candidate."""
    return {mem_id: int(rng.random() < p) for mem_id, p in propensities.items()}


def retrieve(
    similarities: dict[str, float],
    m: int,
    propensity_min: float,
    propensity_max: float,
    rng: np.random.Generator,
) -> tuple[list[str], dict[str, float], dict[str, int]]:
    """Full retrieval step: top-M selection, propensity assignment, inclusion draw.

    Returns (candidate_ids, propensities, included) where included[mem_id] in {0, 1}.
    """
    candidate_ids = select_top_m(similarities, m)
    propensities = assign_propensities(candidate_ids, propensity_min, propensity_max)
    included = draw_inclusion(propensities, rng)
    return candidate_ids, propensities, included
