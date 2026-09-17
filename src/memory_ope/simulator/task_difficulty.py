"""Task-difficulty confounding simulator.

70 generalist memories (utility ~ Uniform[0,1]) and 30 specialist memories
(fixed utility). Easy tasks only ever draw candidates from generalists; hard
tasks draw candidates from the full pool. Specialists are therefore only ever
seen on hard (low base-success) tasks — a memory's observed co-occurrence
with success is confounded by task difficulty, independent of its own causal
contribution.
"""

from __future__ import annotations

import numpy as np

from . import _common


def _build_ids_and_utility(cfg: dict) -> tuple[list[str], np.ndarray]:
    sc = cfg["simulators"]["task_difficulty"]
    n_gen, n_spec = sc["n_generalists"], sc["n_specialists"]
    ids = [f"gen_{i}" for i in range(n_gen)] + [f"spec_{i}" for i in range(n_spec)]
    util_rng = np.random.default_rng(sc["utility_seed"])
    gen_utility = util_rng.uniform(0.0, 1.0, size=n_gen)
    spec_utility = np.full(n_spec, sc["specialist_utility"])
    utility = np.concatenate([gen_utility, spec_utility])
    return ids, utility


def _simulate_batch(
    cfg: dict,
    rng: np.random.Generator,
    n: int,
    propensity_min: float | None = None,
    propensity_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Returns (candidate_mask, propensity, included, base_rate, task_type, ids, utility)."""
    sc = cfg["simulators"]["task_difficulty"]
    ids, utility = _build_ids_and_utility(cfg)
    n_gen = sc["n_generalists"]
    n_memories = len(ids)
    p_min = cfg["retrieval"]["propensity_min"] if propensity_min is None else propensity_min
    p_max = cfg["retrieval"]["propensity_max"] if propensity_max is None else propensity_max

    task_type = (rng.random(n) < sc["p_hard"]).astype(int)  # 0 = easy, 1 = hard

    similarities = rng.random((n, n_memories))
    easy_rows = task_type == 0
    similarities[np.ix_(easy_rows, np.arange(n_gen, n_memories))] = -np.inf

    candidate_mask, propensity = _common.top_m_candidate_info(similarities, cfg["retrieval"]["M"], p_min, p_max)
    included = _common.draw_inclusion(propensity, rng)

    base_rate = np.where(task_type == 0, sc["easy_task_base_success"], sc["hard_task_base_success"])
    return candidate_mask, propensity, included, base_rate, task_type, ids, utility


def all_ids(cfg: dict) -> list[str]:
    ids, _utility = _build_ids_and_utility(cfg)
    return ids


def generate_episodes(
    cfg: dict,
    seed: int,
    n_episodes: int | None = None,
    propensity_min: float | None = None,
    propensity_max: float | None = None,
) -> list[dict]:
    sc = cfg["simulators"]["task_difficulty"]
    n = sc["n_episodes"] if n_episodes is None else n_episodes
    rng = np.random.default_rng(seed)
    candidate_mask, propensity, included, base_rate, task_type, ids, utility = _simulate_batch(
        cfg, rng, n, propensity_min, propensity_max
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
                "task_type": "hard" if task_type[i] == 1 else "easy",
                "candidate_ids": [ids[j] for j in cand_idx],
                "propensities": {ids[j]: float(propensity[i, j]) for j in cand_idx},
                "included": {ids[j]: int(included[i, j]) for j in cand_idx},
                "success": int(success[i]),
            }
        )
    return episodes


def true_values(
    cfg: dict, propensity_min: float | None = None, propensity_max: float | None = None
) -> dict[str, float]:
    """Causal oracle: E[Y|do(Z_m=1)] - E[Y|do(Z_m=0)] among candidate contexts."""
    sc = cfg["simulators"]["task_difficulty"]
    rng = np.random.default_rng(sc["oracle_seed"])
    candidate_mask, _propensity, included, base_rate, _task_type, ids, utility = _simulate_batch(
        cfg, rng, sc["oracle_episodes"], propensity_min, propensity_max
    )
    values = _common.oracle_values(candidate_mask, included, base_rate, utility, sc["beta"])
    return {mem_id: float(v) for mem_id, v in zip(ids, values)}


def naive_observational_values(
    cfg: dict, propensity_min: float | None = None, propensity_max: float | None = None
) -> dict[str, float]:
    """Naive population contrast: E[Y|Z_m=1] - E[Y|Z_m=0] among candidate
    contexts, using each context's factual (not forced) inclusion draw."""
    sc = cfg["simulators"]["task_difficulty"]
    rng = np.random.default_rng(sc["oracle_seed"])
    candidate_mask, _propensity, included, base_rate, _task_type, ids, utility = _simulate_batch(
        cfg, rng, sc["oracle_episodes"], propensity_min, propensity_max
    )
    values = _common.naive_observational_values(candidate_mask, included, base_rate, utility, sc["beta"])
    return {mem_id: float(v) for mem_id, v in zip(ids, values)}
