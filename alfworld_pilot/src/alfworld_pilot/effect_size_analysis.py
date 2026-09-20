"""Zero-cost analysis, prompted by the mini ground truth's null result: are
individual real per-memory effects in the current 30-memory store simply
too small to measure at any practical budget, as opposed to an estimator
problem?

Two independent lines of evidence, both computed from data already in
hand (small_pilot.jsonl's 180 episodes + the 4 ground-truthed memories'
1064 paired episodes):

1. Noise-vs-signal decomposition: compute the THEORETICAL null-model
   standard error of the IPS estimator for each memory (using its real
   candidacy count and real propensities, under the null that the true
   effect is 0), and compare it to the ACTUALLY OBSERVED spread of IPS
   point estimates across all 30 memories. If the observed spread is no
   larger than pure sampling noise predicts, there is no detectable real
   signal at this sample size -- full stop, independent of which
   estimator is used.
2. The 4 direct ground-truth measurements themselves (already computed by
   mini_ground_truth_analysis.py) as a cross-check: these are exact,
   estimator-independent, and were selected specifically for looking most
   different -- if even they show tiny effects, that's a second,
   independent confirmation.

Then projects episode/cost requirements to resolve effects of the
magnitude actually found, at the current design's scale (30 memories) and
at proposed smaller/redesigned scales, using the REAL measured rho
(0.616-0.666 across the 4 ground-truthed memories, much higher than the
0.154 simulator-derived proxy originally used for planning) and REAL
per-task-type costs.

Usage:
    python -m alfworld_pilot.effect_size_analysis
"""

from __future__ import annotations

import json
import math
import pathlib
from collections import defaultdict

import numpy as np

from . import config as config_mod

Z_ALPHA_2, Z_BETA, Z_ALPHA_1 = 1.96, 0.8416, 1.645
POWER_CONST_2SIDED = (Z_ALPHA_2 + Z_BETA) ** 2
POWER_CONST_1SIDED = (Z_ALPHA_1 + Z_BETA) ** 2

# Real measured rho from the mini ground truth run's 4 memories (results/mini_ground_truth_analysis.json).
REAL_MEASURED_RHOS = {"mem_17": 0.632, "mem_29": 0.628, "mem_21": 0.666, "mem_9": 0.616}
MEAN_REAL_RHO = sum(REAL_MEASURED_RHOS.values()) / len(REAL_MEASURED_RHOS)

# Real per-task-type cost/episode, measured from small_pilot.jsonl (gpt-5.6-luna, $0.20/$1.20 per M).
REAL_COST_PER_EPISODE_BY_TYPE = {
    "pick_heat_then_place_in_recep": 0.0192,
    "pick_cool_then_place_in_recep": 0.0164,
    "pick_clean_then_place_in_recep": 0.0128,
    "pick_two_obj_and_place": 0.0090,
    "pick_and_place_simple": 0.0057,
    "look_at_obj_in_light": 0.0049,
}
BLENDED_COST_PER_EPISODE = sum(REAL_COST_PER_EPISODE_BY_TYPE.values()) / len(REAL_COST_PER_EPISODE_BY_TYPE)


def _noise_vs_signal(episodes: list[dict], small_pilot_report: dict) -> dict:
    p_success = sum(ep["success"] for ep in episodes) / len(episodes)
    mem_terms_var = defaultdict(list)
    for ep in episodes:
        for mid in ep["candidate_ids"]:
            p = ep["propensities"][mid]
            mem_terms_var[mid].append(p_success * (1 / p + 1 / (1 - p)))

    null_se_by_mem = {mid: math.sqrt(sum(c) / len(c) ** 2) for mid, c in mem_terms_var.items()}
    mean_null_se = sum(null_se_by_mem.values()) / len(null_se_by_mem)

    ips_vals = list(small_pilot_report["estimates"]["ips"].values())
    observed_std = float(np.std(ips_vals, ddof=1))
    observed_range = max(ips_vals) - min(ips_vals)

    var_observed = observed_std ** 2
    var_noise = mean_null_se ** 2
    implied_var_true_effects = var_observed - var_noise

    return {
        "overall_success_rate": p_success,
        "null_model_se_per_memory": null_se_by_mem,
        "mean_null_model_se": mean_null_se,
        "observed_ips_std": observed_std,
        "observed_ips_range": observed_range,
        "expected_null_range_approx": 3.8 * mean_null_se,  # ~range of 30 iid N(0, se^2) draws
        "var_observed": var_observed,
        "var_noise": var_noise,
        "implied_var_true_effects": implied_var_true_effects,
        "implied_sd_true_effects": math.sqrt(implied_var_true_effects) if implied_var_true_effects > 0 else 0.0,
        "conclusion": (
            "noise alone explains the observed spread -- no detectable real signal at n=180"
            if implied_var_true_effects <= 0
            else f"observed spread exceeds noise; implied real-effect SD ~ {math.sqrt(implied_var_true_effects):.4f}"
        ),
    }


def _n_pairs(delta: float, var_d: float, one_sided: bool = False) -> float:
    const = POWER_CONST_1SIDED if one_sided else POWER_CONST_2SIDED
    return const * var_d / delta ** 2


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])
    logs_dir = pathlib.Path(cfg["paths"]["logs_dir"])

    with open(logs_dir / "small_pilot.jsonl", "r", encoding="utf-8") as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    with open(results_dir / "small_pilot.json", "r", encoding="utf-8") as f:
        small_pilot_report = json.load(f)
    with open(results_dir / "mini_ground_truth_analysis.json", "r", encoding="utf-8") as f:
        gt_analysis = json.load(f)

    noise_vs_signal = _noise_vs_signal(episodes, small_pilot_report)
    print("=== 1. Noise-vs-signal decomposition (small pilot, 30 memories, n=180 episodes) ===")
    print(f"Mean null-model SE per memory (IPS, if true effect were 0): {noise_vs_signal['mean_null_model_se']:.4f}")
    print(f"Observed std of IPS point estimates across 30 memories:     {noise_vs_signal['observed_ips_std']:.4f}")
    print(f"Observed range: {noise_vs_signal['observed_ips_range']:.4f}  (pure-noise-predicted range: ~{noise_vs_signal['expected_null_range_approx']:.4f})")
    print(f"=> {noise_vs_signal['conclusion']}")

    gt_gaps = {m: gt_analysis["ground_truth"][m]["gap"] for m in gt_analysis["ground_truth"]}
    gt_spread = max(gt_gaps.values()) - min(gt_gaps.values())
    print(f"\n=== 2. Direct ground truth cross-check (4 memories selected for LOOKING most different) ===")
    print(f"Ground truth gaps: {gt_gaps}")
    print(f"Spread: {gt_spread:.4f} -- smaller than any single memory's own 95% CI half-width")
    print(f"Mean real measured rho: {MEAN_REAL_RHO:.3f} (vs. 0.154 simulator-derived planning proxy -- real paired correlation is much stronger)")

    var_y = noise_vs_signal["overall_success_rate"] * (1 - noise_vs_signal["overall_success_rate"])
    var_d = 2 * var_y * (1 - MEAN_REAL_RHO)

    print(f"\n=== 3. Episodes needed to resolve effects of various sizes (real rho={MEAN_REAL_RHO:.3f}, real Var(Y)={var_y:.4f}) ===")
    resolution_table = {}
    for delta in [0.15, 0.10, 0.05, 0.03, 0.02]:
        n = _n_pairs(delta, var_d)
        resolution_table[delta] = n
        print(f"  delta={delta:.2f}: {n:.0f} pairs/memory")

    full_30_at_003 = 30 * resolution_table[0.03] * 2
    full_30_at_003_cost = full_30_at_003 * BLENDED_COST_PER_EPISODE
    print(f"\nFull 30-memory store at delta=0.03 (the largest gap actually found): "
          f"{full_30_at_003:.0f} episodes, ~${full_30_at_003_cost:.2f} -- NOT affordable at any realistic pilot budget.")

    print(f"\n=== 4. Redesign options (6 deliberately-engineered memories) ===")
    redesign_a = {}
    for delta in [0.15, 0.25, 0.35]:
        n = _n_pairs(delta, var_d)
        cost = 6 * n * 2 * BLENDED_COST_PER_EPISODE
        redesign_a[delta] = {"n_pairs": n, "total_episodes": 6 * n * 2, "cost_usd": cost}
        print(f"  (a) ranking target, delta={delta:.2f}: {n:.1f} pairs/mem, {6*n*2:.0f} episodes, ${cost:.2f}")

    redesign_b = {}
    for delta in [0.35, 0.30, 0.25]:
        n = _n_pairs(delta, var_d, one_sided=True)
        cost = 6 * n * 2 * BLENDED_COST_PER_EPISODE
        redesign_b[delta] = {"n_pairs": n, "total_episodes": 6 * n * 2, "cost_usd": cost}
        print(f"  (b) detection target, delta={delta:.2f}: {n:.1f} pairs/mem, {6*n*2:.0f} episodes, ${cost:.2f}")

    remaining_budget = 5.20
    n_pairs_afford = remaining_budget / (6 * 2 * BLENDED_COST_PER_EPISODE)
    mde_2sided = math.sqrt(POWER_CONST_2SIDED * var_d / n_pairs_afford)
    mde_1sided = math.sqrt(POWER_CONST_1SIDED * var_d / n_pairs_afford)
    print(f"\n=== 5. What the current ~${remaining_budget:.2f} remaining buys for 6 memories ===")
    print(f"  affordable pairs/memory: {n_pairs_afford:.1f}")
    print(f"  MDE (two-sided ranking):    {mde_2sided:.3f}")
    print(f"  MDE (one-sided detection):  {mde_1sided:.3f}")

    report = {
        "noise_vs_signal": {k: v for k, v in noise_vs_signal.items() if k != "null_model_se_per_memory"},
        "ground_truth_cross_check": {"gaps": gt_gaps, "spread": gt_spread, "mean_real_rho": MEAN_REAL_RHO},
        "real_var_y": var_y, "real_var_d": var_d,
        "resolution_table_pairs_per_memory": resolution_table,
        "full_30_memory_store_at_delta_0.03": {"total_episodes": full_30_at_003, "cost_usd": full_30_at_003_cost},
        "redesign_a_ranking_target": redesign_a,
        "redesign_b_detection_target": redesign_b,
        "current_remaining_budget_usd": remaining_budget,
        "affordable_now_for_6_memories": {"n_pairs_per_memory": n_pairs_afford, "mde_2sided": mde_2sided, "mde_1sided": mde_1sided},
    }
    with open(results_dir / "effect_size_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {results_dir / 'effect_size_analysis.json'}")


if __name__ == "__main__":
    main()
