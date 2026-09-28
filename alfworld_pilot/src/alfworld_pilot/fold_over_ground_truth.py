"""Fold-over ground truth (2026-09-29): for a single task instance, draw ONE
random inclusion mask over the 5 target memories, run one episode with that
mask forced, and a second episode with the COMPLEMENT mask forced -- every
target memory flips status between the two arms, so one task instance gives
a paired in/out data point for (up to) all 5 memories at once, instead of
needing a separate pair per memory (ground_truth_runner.run_ground_truth_pair).

WHY THIS MATCHES THE SAME ESTIMAND IPS/SNIPS/DR TARGET (see README's
"Fold-over ground-truth design" section for the full reasoning): episode_runner
.run_logged_episode's `forced_inclusion` override ALREADY only overrides a
memory's status if it's in `candidate_ids` (i.e., was ALREADY a natural top-M
candidate for this task instance) -- passing a mask for all 5 target memories
unconditionally, regardless of whether each is actually a natural candidate
here, is therefore automatically restricted to the natural-candidate subset by
the existing mechanism, with ZERO new logic needed. This is exactly the
condition under which a fold-over's per-memory regression coefficient targets
the same "effect of memory i, marginalizing over the natural distribution of
everything else" quantity that the one-at-a-time design (and IPS/SNIPS/DR,
built from natural retrieval) target.

This does NOT hold if you skip that restriction (e.g. force a memory's status
via a mechanism that bypasses `included`'s candidate-only override) -- doing
so would extend a memory's measured effect beyond where it can ever naturally
occur, which the natural-retrieval-based OPE estimators cannot be validated
against.

CAVEAT, stated plainly: recovering each of the 5 memories' OWN effect from
many fold-over pairs requires a regression decomposition (parse_fail ~ sum of
per-memory inclusion indicators, across many independently-drawn masks) and
assumes the 5 memories' effects are ADDITIVE (no interaction). The
pre-registered co-retrieved pair (pas_correct + irrelevant_3) was chosen
specifically because it's likely to explore an interaction -- under this
design, their individual "main effect" coefficients will not cleanly separate
from their joint interaction. This is an accepted trade-off for the
efficiency gain (see README), not something this module tries to solve.
"""

from __future__ import annotations

import random

from .episode_runner import SimilarityFn, run_logged_episode
from .ground_truth_runner import TaskSource, _pair_seed_material


def draw_fold_over_mask(target_memory_ids: list[str], pair_index: int, task_seed: int) -> dict[str, int]:
    """Deterministic function of (target_memory_ids, pair_index, task_seed) --
    same seeding convention as ground_truth_runner._pair_seed_material, keyed
    by the sorted target id list so the mask is reproducible regardless of
    input ordering."""
    key = "|".join(sorted(target_memory_ids))
    seed_material = _pair_seed_material(key, pair_index, task_seed)
    mask_rng = random.Random(seed_material)
    return {mid: mask_rng.randint(0, 1) for mid in sorted(target_memory_ids)}


def complement_mask(mask: dict[str, int]) -> dict[str, int]:
    return {mid: 1 - v for mid, v in mask.items()}


def run_fold_over_arm(
    task_source: TaskSource,
    llm_client,
    memories: list,
    target_memory_ids: list[str],
    mask: dict[str, int],
    m: int,
    propensity_min: float,
    propensity_max: float,
    max_steps: int,
    task_seed: int,
    pair_index: int,
    arm: str,
    similarity_fn: SimilarityFn | None = None,
) -> dict:
    """Runs ONE arm of a fold-over pair, given an already-computed mask (from
    draw_fold_over_mask, or its complement) -- deliberately independent of
    the other arm, so job_list.py/run_jobs.py can schedule/resume the two
    arms of a pair as two separate, order-independent jobs (they don't need
    to run in the same process or even the same Kaggle session; both derive
    the identical "everything else" natural-draw seed from
    (target_memory_ids, pair_index, task_seed) alone, per
    ground_truth_runner._pair_seed_material, so running arm b days after arm
    a still reproduces the exact natural draw arm a saw for every non-target
    memory)."""
    seed_material = _pair_seed_material("|".join(sorted(target_memory_ids)), pair_index, task_seed)
    rng = random.Random(seed_material)

    env = task_source.build_env(task_seed)
    try:
        ep = run_logged_episode(
            env, llm_client, memories, m, propensity_min, propensity_max, max_steps,
            task_id=task_seed, rng=rng, forced_inclusion=mask, similarity_fn=similarity_fn,
        )
    finally:
        env.close()

    ep["fold_over_pair_index"] = pair_index
    ep["fold_over_mask"] = mask
    ep["fold_over_arm"] = arm
    return ep


def run_fold_over_pair(
    task_source: TaskSource,
    llm_client,
    memories: list,
    target_memory_ids: list[str],
    m: int,
    propensity_min: float,
    propensity_max: float,
    max_steps: int,
    task_seed: int,
    pair_index: int,
    similarity_fn: SimilarityFn | None = None,
) -> tuple[dict, dict]:
    """Convenience wrapper for running (and testing) both arms in one call --
    equivalent to two separate run_fold_over_arm calls with the mask and its
    complement. The job-list runner uses run_fold_over_arm directly, one arm
    per job, since the two arms don't need to run together (see its
    docstring)."""
    mask = draw_fold_over_mask(target_memory_ids, pair_index, task_seed)
    mask_b = complement_mask(mask)
    ep_a = run_fold_over_arm(
        task_source, llm_client, memories, target_memory_ids, mask, m, propensity_min, propensity_max,
        max_steps, task_seed, pair_index, arm="a", similarity_fn=similarity_fn,
    )
    ep_b = run_fold_over_arm(
        task_source, llm_client, memories, target_memory_ids, mask_b, m, propensity_min, propensity_max,
        max_steps, task_seed, pair_index, arm="b", similarity_fn=similarity_fn,
    )
    return ep_a, ep_b
