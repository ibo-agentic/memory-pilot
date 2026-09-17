"""Forced-in/forced-out paired ground truth, matching Part A.3's design:
same task instance AND the same random draws for every OTHER memory's
inclusion, only the target memory's Z is forced to 1 vs 0. Both arms use
temperature=0 (+ a fixed `seed` if the client supports it) -- see
determinism_check.py for verifying that actually holds, which every
ground-truth session logs before trusting its own results.

Only searches among task instances where the target memory is a NATURAL
candidate (matches the causal oracle's own definition: "among episodes
where m is a candidate") -- forcing a memory in when it was never even a
plausible candidate would measure a different, out-of-scope intervention.
"""

from __future__ import annotations

import random

from .episode_runner import run_logged_episode
from .memory_store import Memory, similarity_scores
from .retrieval_shared import retrieve


def _pair_seed_material(memory_id: str, pair_index: int, task_seed: int) -> int:
    """The exact seed both the candidacy probe and the real paired run must
    use for similarity/retrieval draws, so the probe actually predicts what
    the real run will draw (Python's hash() is randomized per-process for
    strings by default, which would silently break that agreement across
    separate calls -- md5 is used instead for a stable, reproducible seed)."""
    import hashlib

    key = f"{memory_id}|{pair_index}|{task_seed}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) & 0xFFFFFFFF


def _is_natural_candidate(
    env, memories: list[Memory], memory_id: str, task_seed: int, pair_index: int, m: int, p_min: float, p_max: float
) -> bool:
    """Probe-only: uses the SAME seed the real run will use for this
    (memory_id, pair_index, task_seed), so it actually predicts real
    candidacy rather than checking an unrelated random draw."""
    _obs, info = env.reset(task_seed=task_seed)
    task_type = info["task_type"]
    probe_rng = random.Random(_pair_seed_material(memory_id, pair_index, task_seed))
    sims = similarity_scores(memories, task_type, probe_rng)
    candidate_ids, _propensities, _included = retrieve(sims, m, p_min, p_max, probe_rng)
    return memory_id in candidate_ids


def run_ground_truth_pair(
    env,
    llm_client,
    memories: list[Memory],
    m: int,
    propensity_min: float,
    propensity_max: float,
    max_steps: int,
    task_seed: int,
    pair_index: int,
    memory_id: str,
) -> tuple[dict, dict]:
    seed_material = _pair_seed_material(memory_id, pair_index, task_seed)
    rng_in = random.Random(seed_material)
    rng_out = random.Random(seed_material)  # identical seed -> identical "everything else" draws in both arms

    ep_in = run_logged_episode(
        env, llm_client, memories, m, propensity_min, propensity_max, max_steps,
        task_id=task_seed, rng=rng_in, forced_inclusion={memory_id: 1},
    )
    ep_out = run_logged_episode(
        env, llm_client, memories, m, propensity_min, propensity_max, max_steps,
        task_id=task_seed, rng=rng_out, forced_inclusion={memory_id: 0},
    )
    for ep, arm in ((ep_in, "forced_in"), (ep_out, "forced_out")):
        ep["ground_truth_pair_index"] = pair_index
        ep["ground_truth_memory_id"] = memory_id
        ep["ground_truth_arm"] = arm
        ep["ground_truth_task_seed"] = task_seed
    return ep_in, ep_out


def run_ground_truth_experiment(
    env,
    llm_client,
    memories: list[Memory],
    m: int,
    propensity_min: float,
    propensity_max: float,
    max_steps: int,
    memory_id: str,
    n_pairs: int,
    start_seed: int = 0,
    max_probe_tries: int = 10_000,
) -> list[tuple[dict, dict]]:
    pairs = []
    seed = start_seed
    tries = 0
    while len(pairs) < n_pairs and tries < max_probe_tries:
        if _is_natural_candidate(env, memories, memory_id, seed, len(pairs), m, propensity_min, propensity_max):
            ep_in, ep_out = run_ground_truth_pair(
                env, llm_client, memories, m, propensity_min, propensity_max, max_steps,
                task_seed=seed, pair_index=len(pairs), memory_id=memory_id,
            )
            pairs.append((ep_in, ep_out))
        seed += 1
        tries += 1
    if len(pairs) < n_pairs:
        raise RuntimeError(
            f"Only found {len(pairs)}/{n_pairs} task seeds where '{memory_id}' is a natural candidate "
            f"after {tries} tries -- this memory may be too rarely retrieved for the requested pair count."
        )
    return pairs


def paired_correlation(pairs: list[tuple[dict, dict]]) -> dict:
    """Empirical rho for the paired design, from REAL results, to redo
    Part A.3's power analysis with a measured (not simulator-derived) value."""
    import numpy as np

    y1 = np.array([ep_in["success"] for ep_in, _ep_out in pairs], dtype=float)
    y0 = np.array([ep_out["success"] for _ep_in, ep_out in pairs], dtype=float)
    if len(pairs) < 2 or np.std(y1) == 0 or np.std(y0) == 0:
        rho = float("nan")
    else:
        rho = float(np.corrcoef(y1, y0)[0, 1])
    return {
        "n_pairs": len(pairs),
        "mean_y1": float(y1.mean()),
        "mean_y0": float(y0.mean()),
        "observed_gap": float(y1.mean() - y0.mean()),
        "var_y1": float(y1.var()),
        "var_y0": float(y0.var()),
        "rho": rho,
    }
