"""Check: at Stage-2 scale (n_episodes=1000), which design choices give the
most signal? Grid over (n_memories in {30, 50}) x (propensity range in
{[0.1,0.9] default, [0.3,0.7] closer to uniform}), reporting:
  - Spearman vs causal oracle (mean + 95% seed interval, 20 seeds)
  - precision@top5 / precision@bottom5: of the estimator's top-5 (resp.
    bottom-5) ranked memories, what fraction are actually in the causal
    oracle's true top-5 (resp. bottom-5)? A coarser, more forgiving metric
    than full Spearman -- directly relevant to Part A.3's memory-selection
    strategy, which only needs to identify extremes, not a full ranking.

Usage:
    python -m memory_ope.evaluation.signal_boost_check
"""

from __future__ import annotations

import copy
import json
import pathlib

import numpy as np

from .. import config as config_mod
from ..estimators import doubly_robust, ips, memory_worth
from ..evaluation import metrics
from ..simulator import task_difficulty

N_EPISODES = 1000
MEMORY_COUNTS = [30, 50]  # keeping the 70:30 generalist:specialist ratio
PROPENSITY_RANGES = [(0.1, 0.9), (0.3, 0.7)]
ESTIMATOR_NAMES = ["memory_worth", "ips", "snips", "doubly_robust"]
K = 5


def _precision_at_k(estimates: dict[str, float], true_values: dict[str, float], k: int, top: bool) -> float:
    common = [m for m in true_values if m in estimates]
    true_sorted = sorted(common, key=lambda m: true_values[m], reverse=top)
    est_sorted = sorted(common, key=lambda m: estimates[m], reverse=top)
    true_set = set(true_sorted[:k])
    est_set = set(est_sorted[:k])
    return len(true_set & est_set) / k


def _interval(vals: list[float]) -> dict[str, float]:
    arr = np.array(vals, dtype=float)
    return {"mean": float(np.mean(arr)), "p2_5": float(np.percentile(arr, 2.5)), "p97_5": float(np.percentile(arr, 97.5))}


def _run_cell(cfg: dict, n_memories: int, p_min: float, p_max: float) -> dict:
    n_gen = round(n_memories * 0.7)
    n_spec = n_memories - n_gen
    scaled = copy.deepcopy(cfg)
    scaled["simulators"]["task_difficulty"]["n_generalists"] = n_gen
    scaled["simulators"]["task_difficulty"]["n_specialists"] = n_spec

    true_vals = task_difficulty.true_values(scaled, propensity_min=p_min, propensity_max=p_max)
    all_ids = task_difficulty.all_ids(scaled)
    dr_l2_c = scaled["estimators"]["dr_l2_C"]
    seed_count = scaled["seed_count"]

    per_est = {e: {"spearman": [], "prec_top5": [], "prec_bottom5": []} for e in ESTIMATOR_NAMES}
    for seed in range(seed_count):
        episodes = task_difficulty.generate_episodes(scaled, seed, n_episodes=N_EPISODES, propensity_min=p_min, propensity_max=p_max)
        mw = memory_worth.compute(episodes)
        ips_est, snips_est = ips.compute(episodes)
        dr_est = doubly_robust.compute(episodes, all_ids, l2_c=dr_l2_c, cross_fit=True)
        estimates = {"memory_worth": mw, "ips": ips_est, "snips": snips_est, "doubly_robust": dr_est}
        for est_name in ESTIMATOR_NAMES:
            e = estimates[est_name]
            per_est[est_name]["spearman"].append(metrics.spearman(e, true_vals))
            per_est[est_name]["prec_top5"].append(_precision_at_k(e, true_vals, K, top=True))
            per_est[est_name]["prec_bottom5"].append(_precision_at_k(e, true_vals, K, top=False))

    return {
        "n_memories": n_memories,
        "n_generalists": n_gen,
        "n_specialists": n_spec,
        "propensity_min": p_min,
        "propensity_max": p_max,
        "estimators": {e: {m: _interval(v) for m, v in d.items()} for e, d in per_est.items()},
    }


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    cells = []
    for n_memories in MEMORY_COUNTS:
        for p_min, p_max in PROPENSITY_RANGES:
            cells.append(_run_cell(cfg, n_memories, p_min, p_max))

    report = {"n_episodes": N_EPISODES, "k": K, "cells": cells}
    with open(results_dir / "signal_boost_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for cell in cells:
        label = f"n_memories={cell['n_memories']} propensity=[{cell['propensity_min']},{cell['propensity_max']}]"
        print(f"\n=== {label} (n_episodes={N_EPISODES}, {cfg['seed_count']} seeds) ===")
        print(f"{'estimator':<16}{'spearman':>28}{'prec@top5':>14}{'prec@bottom5':>14}")
        for est_name in ESTIMATOR_NAMES:
            r = cell["estimators"][est_name]
            sp = r["spearman"]
            print(
                f"{est_name:<16}"
                f"{sp['mean']:>8.3f} [{sp['p2_5']:>6.3f},{sp['p97_5']:>6.3f}]"
                f"{r['prec_top5']['mean']:>14.2f}"
                f"{r['prec_bottom5']['mean']:>14.2f}"
            )

    print(f"\nWrote {results_dir / 'signal_boost_check.json'}")


if __name__ == "__main__":
    main()
