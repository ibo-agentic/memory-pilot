"""Wires success_within_25/30 into the four estimators, and computes the
pre-registered validation metrics (README's "Pre-registered validation
metrics" section): for each of the 5 ground-truth memories, each
estimator's point estimate vs. ground truth (signed error, and whether
ground truth falls inside a bootstrap 95% CI), reported per task type and
pooled.

Ground truth comes from the fold-over pairs (fold_over_ground_truth.py /
job_list.py): for a target memory, the paired difference (Y_in - Y_out)
averaged over every fold pair where that memory was actually a natural
candidate. Only some fold pairs contribute to any given memory -- by design,
see fold_over_ground_truth.py's module docstring, not a bug.

IMPORTANT, discovered while wiring this up (not caught when the validation
metrics were first pre-registered): memory_worth.compute() returns ONLY
P(Y=1 | included=1) -- a raw single-arm rate, not a Y1-Y0 gap. Every
existing use of it in this project (tests/test_estimators.py's confounding
regression test) compares it to ground truth via Spearman RANK correlation
only, never a direct numeric subtraction, because it isn't on the same scale
as IPS/SNIPS/DR or a ground-truth gap (paper_data.md's own "MW (randomized,
uncorrected)" framing implies the intended comparison IS a naive,
uncorrected P(Y=1|Z=1) - P(Y=1|Z=0) gap, contrasted against IPS's
propensity-corrected version of the same contrast). So the
"Memory Worth" row of the signed-error/CI-coverage table below uses that
naive gap (_memory_worth_naive_gap, defined here, NOT
src/memory_ope/estimators/memory_worth.py's raw return value) -- computed
locally in this module rather than changing memory_worth.py's return shape
for every other existing caller.

Does not modify src/memory_ope/estimators/ otherwise -- ips.compute and
doubly_robust.compute are called completely unchanged.
"""

from __future__ import annotations

import random
from collections import defaultdict
from functools import partial

import numpy as np

from memory_ope.estimators import doubly_robust, ips, memory_worth

from .job_list import TARGET_MEMORY_IDS
from .secondary_outcomes import truncated_success

N_BOOTSTRAP = 200
ALPHA = 0.05


def ground_truth_effects(fold_episodes: list[dict], target_memory_ids: list[str] | None = None) -> dict[str, dict]:
    """fold_episodes: ground_truth_fold_a/b episode records (each has
    fold_over_pair_index, fold_over_arm, success, candidate_ids, included).
    Returns, per memory: n_pairs (how many fold pairs had that memory as a
    natural candidate) and mean_effect (Y_in - Y_out averaged over those
    pairs, None if it was never a natural candidate in this data)."""
    target_memory_ids = target_memory_ids or TARGET_MEMORY_IDS
    by_pair: dict[int, dict[str, dict]] = defaultdict(dict)
    for ep in fold_episodes:
        by_pair[ep["fold_over_pair_index"]][ep["fold_over_arm"]] = ep

    effects: dict[str, list[float]] = defaultdict(list)
    for arms in by_pair.values():
        if "a" not in arms or "b" not in arms:
            continue
        ep_a, ep_b = arms["a"], arms["b"]
        for mid in target_memory_ids:
            if mid not in ep_a["candidate_ids"]:
                continue  # not a natural candidate for this task instance -- no data point, not a zero
            y_in = ep_a["success"] if ep_a["included"][mid] == 1 else ep_b["success"]
            y_out = ep_b["success"] if ep_a["included"][mid] == 1 else ep_a["success"]
            effects[mid].append(y_in - y_out)

    return {
        mid: {
            "n_pairs": len(effects.get(mid, [])),
            "mean_effect": float(np.mean(effects[mid])) if effects.get(mid) else None,
        }
        for mid in target_memory_ids
    }


def _memory_worth_naive_gap(episodes: list[dict]) -> dict[str, float | None]:
    """See module docstring: the naive, uncorrected P(Y=1|Z=1) - P(Y=1|Z=0)
    gap -- NOT memory_worth.compute()'s raw single-arm return value."""
    included_pos: dict[str, int] = defaultdict(int)
    included_total: dict[str, int] = defaultdict(int)
    excluded_pos: dict[str, int] = defaultdict(int)
    excluded_total: dict[str, int] = defaultdict(int)
    for ep in episodes:
        y = ep["success"]
        for mid in ep["candidate_ids"]:
            if ep["included"][mid] == 1:
                included_total[mid] += 1
                included_pos[mid] += y
            else:
                excluded_total[mid] += 1
                excluded_pos[mid] += y

    result: dict[str, float | None] = {}
    for mid in set(included_total) | set(excluded_total):
        p_in = included_pos[mid] / included_total[mid] if included_total[mid] else None
        p_out = excluded_pos[mid] / excluded_total[mid] if excluded_total[mid] else None
        result[mid] = (p_in - p_out) if (p_in is not None and p_out is not None) else None
    return result


def _build_estimators(all_memory_ids: list[str]) -> dict:
    return {
        "memory_worth": _memory_worth_naive_gap,
        "ips": lambda episodes: ips.compute(episodes)[0],
        "snips": lambda episodes: ips.compute(episodes)[1],
        "doubly_robust": partial(doubly_robust.compute, all_memory_ids=all_memory_ids),
    }


def _bootstrap_ci(episodes: list[dict], compute_fn, memory_id: str, n_bootstrap: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float | None, float | None]:
    if not episodes:
        return None, None
    rng = random.Random(seed)
    n = len(episodes)
    estimates = []
    for _ in range(n_bootstrap):
        sample = [episodes[rng.randrange(n)] for _ in range(n)]
        try:
            point = compute_fn(sample).get(memory_id)
        except Exception:  # noqa: BLE001 -- a degenerate bootstrap resample (e.g. no variation) shouldn't crash the whole CI
            point = None
        if point is not None:
            estimates.append(point)
    if len(estimates) < 2:
        return None, None
    lo = float(np.percentile(estimates, 100 * ALPHA / 2))
    hi = float(np.percentile(estimates, 100 * (1 - ALPHA / 2)))
    return lo, hi


def validate_against_ground_truth(
    logging_episodes: list[dict],
    fold_episodes: list[dict],
    all_memory_ids: list[str],
    target_memory_ids: list[str] | None = None,
    outcome_step_caps: list[int | None] | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """outcome_step_caps: None for the primary (unmodified success) outcome,
    plus 25/30 for the pre-registered secondary outcomes -- each reported
    completely separately, never averaged together."""
    target_memory_ids = target_memory_ids or TARGET_MEMORY_IDS
    outcome_step_caps = outcome_step_caps if outcome_step_caps is not None else [None, 25, 30]
    estimators = _build_estimators(all_memory_ids)

    gt = ground_truth_effects(fold_episodes, target_memory_ids)

    results: dict = {"n_logging_episodes": len(logging_episodes), "n_fold_episodes": len(fold_episodes), "ground_truth": gt, "outcomes": {}}
    for cap in outcome_step_caps:
        outcome_name = "primary" if cap is None else f"success_within_{cap}"
        eps = logging_episodes if cap is None else truncated_success(logging_episodes, cap)

        per_estimator = {}
        for name, fn in estimators.items():
            point_estimates = fn(eps) if eps else {}
            per_memory = {}
            for mid in target_memory_ids:
                point = point_estimates.get(mid) if isinstance(point_estimates, dict) else None
                ci_low, ci_high = _bootstrap_ci(eps, fn, mid, n_bootstrap=n_bootstrap) if eps else (None, None)
                truth = gt[mid]["mean_effect"]
                per_memory[mid] = {
                    "point_estimate": point,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ground_truth": truth,
                    "ground_truth_n_pairs": gt[mid]["n_pairs"],
                    "error": (point - truth) if (point is not None and truth is not None) else None,
                    "ground_truth_in_ci": (
                        (ci_low <= truth <= ci_high) if (ci_low is not None and ci_high is not None and truth is not None) else None
                    ),
                }
            per_estimator[name] = per_memory
        results["outcomes"][outcome_name] = per_estimator

    return results


def validate_per_task_type(
    logging_episodes: list[dict],
    fold_episodes: list[dict],
    all_memory_ids: list[str],
    target_memory_ids: list[str] | None = None,
    outcome_step_caps: list[int | None] | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """Same as validate_against_ground_truth, split by task_type (README's
    "reported per task type as well as pooled")."""
    task_types = sorted({ep["task_type"] for ep in logging_episodes} | {ep["task_type"] for ep in fold_episodes})
    per_type = {
        tt: validate_against_ground_truth(
            [ep for ep in logging_episodes if ep["task_type"] == tt],
            [ep for ep in fold_episodes if ep["task_type"] == tt],
            all_memory_ids, target_memory_ids, outcome_step_caps, n_bootstrap,
        )
        for tt in task_types
    }
    pooled = validate_against_ground_truth(logging_episodes, fold_episodes, all_memory_ids, target_memory_ids, outcome_step_caps, n_bootstrap)
    return {"per_task_type": per_type, "pooled": pooled}
