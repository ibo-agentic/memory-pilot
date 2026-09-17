"""Hitchhiker co-retrieval confounding simulator.

One anchor memory (high utility) and one hitchhiker (low utility) whose
similarity scores are tightly correlated, so they are almost always
retrieved together. `independent_retrieval_fraction` controls how often the
hitchhiker's similarity is instead drawn independently, decoupling its
candidacy from the anchor's. A pool of filler memories with random utilities
fills out the rest of the top-M candidate slots.
"""

from __future__ import annotations

import numpy as np

from . import _common

ANCHOR_ID = "anchor"
HITCHHIKER_ID = "hitchhiker"


def _build_ids_and_utility(cfg: dict) -> tuple[list[str], np.ndarray]:
    sc = cfg["simulators"]["hitchhiker"]
    n_filler = sc["n_filler_memories"]
    ids = [ANCHOR_ID, HITCHHIKER_ID] + [f"filler_{i}" for i in range(n_filler)]
    util_rng = np.random.default_rng(sc["utility_seed"])
    filler_utility = util_rng.uniform(0.0, 1.0, size=n_filler)
    utility = np.concatenate([[sc["anchor_utility"], sc["hitchhiker_utility"]], filler_utility])
    return ids, utility


def _simulate_batch(
    cfg: dict,
    rng: np.random.Generator,
    n: int,
    independent_fraction: float,
    propensity_min: float | None = None,
    propensity_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Returns (candidate_mask, propensity, included, base_rate, ids, utility)."""
    sc = cfg["simulators"]["hitchhiker"]
    ids, utility = _build_ids_and_utility(cfg)
    n_filler = sc["n_filler_memories"]
    p_min = cfg["retrieval"]["propensity_min"] if propensity_min is None else propensity_min
    p_max = cfg["retrieval"]["propensity_max"] if propensity_max is None else propensity_max

    filler_sims = rng.random((n, n_filler))
    anchor_sim = rng.random(n)

    indep_mask = rng.random(n) < independent_fraction
    jitter = rng.normal(0.0, 0.01, size=n)
    hitchhiker_sim = np.where(indep_mask, rng.random(n), anchor_sim + jitter)

    similarities = np.concatenate(
        [anchor_sim[:, None], hitchhiker_sim[:, None], filler_sims], axis=1
    )

    candidate_mask, propensity = _common.top_m_candidate_info(similarities, cfg["retrieval"]["M"], p_min, p_max)
    included = _common.draw_inclusion(propensity, rng)
    base_rate = np.full(n, sc["base_success"])
    return candidate_mask, propensity, included, base_rate, ids, utility


def all_ids(cfg: dict) -> list[str]:
    ids, _utility = _build_ids_and_utility(cfg)
    return ids


def generate_episodes(
    cfg: dict,
    seed: int,
    independent_fraction: float | None = None,
    n_episodes: int | None = None,
    propensity_min: float | None = None,
    propensity_max: float | None = None,
) -> list[dict]:
    sc = cfg["simulators"]["hitchhiker"]
    frac = sc["independent_retrieval_fraction"] if independent_fraction is None else independent_fraction
    n = sc["n_episodes"] if n_episodes is None else n_episodes
    rng = np.random.default_rng(seed)
    candidate_mask, propensity, included, base_rate, ids, utility = _simulate_batch(
        cfg, rng, n, frac, propensity_min, propensity_max
    )
    prob = _common.success_probability(base_rate, included, utility, sc["beta"])
    success = _common.sample_success(prob, rng)

    episodes = []
    for i in range(n):
        cand_idx = np.where(candidate_mask[i])[0]
        episodes.append(
            {
                "task_id": i,
                "seed": seed,
                "task_type": "single",
                "independent_retrieval_fraction": frac,
                "candidate_ids": [ids[j] for j in cand_idx],
                "propensities": {ids[j]: float(propensity[i, j]) for j in cand_idx},
                "included": {ids[j]: int(included[i, j]) for j in cand_idx},
                "success": int(success[i]),
            }
        )
    return episodes


def true_values(
    cfg: dict,
    independent_fraction: float | None = None,
    propensity_min: float | None = None,
    propensity_max: float | None = None,
) -> dict[str, float]:
    sc = cfg["simulators"]["hitchhiker"]
    frac = sc["independent_retrieval_fraction"] if independent_fraction is None else independent_fraction
    rng = np.random.default_rng(sc["oracle_seed"])
    candidate_mask, _propensity, included, base_rate, ids, utility = _simulate_batch(
        cfg, rng, sc["oracle_episodes"], frac, propensity_min, propensity_max
    )
    values = _common.oracle_values(candidate_mask, included, base_rate, utility, sc["beta"])
    return {mem_id: float(v) for mem_id, v in zip(ids, values)}


def naive_observational_values(
    cfg: dict,
    independent_fraction: float | None = None,
    propensity_min: float | None = None,
    propensity_max: float | None = None,
) -> dict[str, float]:
    """Naive population contrast: E[Y|Z_m=1] - E[Y|Z_m=0] among candidate
    contexts, using each context's factual (not forced) inclusion draw."""
    sc = cfg["simulators"]["hitchhiker"]
    frac = sc["independent_retrieval_fraction"] if independent_fraction is None else independent_fraction
    rng = np.random.default_rng(sc["oracle_seed"])
    candidate_mask, _propensity, included, base_rate, ids, utility = _simulate_batch(
        cfg, rng, sc["oracle_episodes"], frac, propensity_min, propensity_max
    )
    values = _common.naive_observational_values(candidate_mask, included, base_rate, utility, sc["beta"])
    return {mem_id: float(v) for mem_id, v in zip(ids, values)}
