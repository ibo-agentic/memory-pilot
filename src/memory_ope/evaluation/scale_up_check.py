"""Stage 2 re-plan support: at the ALREADY-chosen Stage 2 setting (30
memories, propensity [0.3, 0.7] -- signal_boost_check's winning cell, and
what alfworld_pilot/config.yaml actually uses), sweep n_episodes across the
range Part C is now considering (1000 baseline vs. 3000/4000/5000) to show
how much the wider real-cost budget actually buys in expected signal, and
at which point a propensity-corrected estimator's 95% seed interval
reliably excludes zero on a SINGLE run (not just the 20-seed mean).

Usage:
    python -m memory_ope.evaluation.scale_up_check
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .. import config as config_mod
from ..estimators import doubly_robust, ips, memory_worth
from ..evaluation import metrics
from ..simulator import task_difficulty, hitchhiker

N_EPISODES_GRID = [1000, 3000, 4000, 5000]
N_MEMORIES = 30
N_GENERALISTS = round(N_MEMORIES * 0.7)
N_SPECIALISTS = N_MEMORIES - N_GENERALISTS
PROPENSITY_MIN, PROPENSITY_MAX = 0.3, 0.7
ESTIMATOR_NAMES = ["memory_worth", "ips", "snips", "doubly_robust"]


def _interval(vals: list[float]) -> dict[str, float]:
    arr = np.array(vals, dtype=float)
    return {"mean": float(np.mean(arr)), "p2_5": float(np.percentile(arr, 2.5)), "p97_5": float(np.percentile(arr, 97.5))}


def _excludes_zero(iv: dict[str, float]) -> bool:
    return iv["p2_5"] > 0.0 or iv["p97_5"] < 0.0


def _run_task_difficulty(cfg: dict, n_episodes: int) -> dict:
    import copy

    scaled = copy.deepcopy(cfg)
    scaled["simulators"]["task_difficulty"]["n_generalists"] = N_GENERALISTS
    scaled["simulators"]["task_difficulty"]["n_specialists"] = N_SPECIALISTS

    true_vals = task_difficulty.true_values(scaled, propensity_min=PROPENSITY_MIN, propensity_max=PROPENSITY_MAX)
    all_ids = task_difficulty.all_ids(scaled)
    dr_l2_c = scaled["estimators"]["dr_l2_C"]
    seed_count = scaled["seed_count"]

    per_est = {e: [] for e in ESTIMATOR_NAMES}
    for seed in range(seed_count):
        episodes = task_difficulty.generate_episodes(
            scaled, seed, n_episodes=n_episodes, propensity_min=PROPENSITY_MIN, propensity_max=PROPENSITY_MAX
        )
        mw = memory_worth.compute(episodes)
        ips_est, snips_est = ips.compute(episodes)
        dr_est = doubly_robust.compute(episodes, all_ids, l2_c=dr_l2_c, cross_fit=True)
        estimates = {"memory_worth": mw, "ips": ips_est, "snips": snips_est, "doubly_robust": dr_est}
        for est_name in ESTIMATOR_NAMES:
            per_est[est_name].append(metrics.spearman(estimates[est_name], true_vals))

    return {est: _interval(v) for est, v in per_est.items()}


def _run_hitchhiker(cfg: dict, n_episodes: int) -> dict:
    import copy

    scaled = copy.deepcopy(cfg)
    scaled["simulators"]["hitchhiker"]["n_filler_memories"] = N_MEMORIES - 2  # + anchor + hitchhiker = N_MEMORIES

    true_vals = hitchhiker.true_values(scaled, propensity_min=PROPENSITY_MIN, propensity_max=PROPENSITY_MAX)
    all_ids = hitchhiker.all_ids(scaled)
    dr_l2_c = scaled["estimators"]["dr_l2_C"]
    seed_count = scaled["seed_count"]

    per_est = {e: [] for e in ESTIMATOR_NAMES}
    for seed in range(seed_count):
        episodes = hitchhiker.generate_episodes(
            scaled, seed, n_episodes=n_episodes, propensity_min=PROPENSITY_MIN, propensity_max=PROPENSITY_MAX
        )
        mw = memory_worth.compute(episodes)
        ips_est, snips_est = ips.compute(episodes)
        dr_est = doubly_robust.compute(episodes, all_ids, l2_c=dr_l2_c, cross_fit=True)
        estimates = {"memory_worth": mw, "ips": ips_est, "snips": snips_est, "doubly_robust": dr_est}
        for est_name in ESTIMATOR_NAMES:
            per_est[est_name].append(metrics.spearman(estimates[est_name], true_vals))

    return {est: _interval(v) for est, v in per_est.items()}


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    report = {
        "setting": {"n_memories": N_MEMORIES, "propensity_min": PROPENSITY_MIN, "propensity_max": PROPENSITY_MAX},
        "n_episodes_grid": N_EPISODES_GRID,
        "task_difficulty": {},
        "hitchhiker": {},
    }

    print(f"=== Scale-up check: n_memories={N_MEMORIES}, propensity=[{PROPENSITY_MIN},{PROPENSITY_MAX}], {cfg['seed_count']} seeds ===\n")
    for sim_name, run_fn in (("task_difficulty", _run_task_difficulty), ("hitchhiker", _run_hitchhiker)):
        print(f"--- {sim_name} ---")
        print(f"{'n_episodes':<12}{'estimator':<16}{'spearman [95% interval]':<32}{'excludes 0?'}")
        for n_episodes in N_EPISODES_GRID:
            result = run_fn(cfg, n_episodes)
            report[sim_name][str(n_episodes)] = result
            for est_name in ESTIMATOR_NAMES:
                iv = result[est_name]
                excl = "YES" if _excludes_zero(iv) else "no"
                print(f"{n_episodes:<12}{est_name:<16}{iv['mean']:>7.3f} [{iv['p2_5']:>6.3f},{iv['p97_5']:>6.3f}]      {excl}")
        print()

    with open(results_dir / "scale_up_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {results_dir / 'scale_up_check.json'}")


if __name__ == "__main__":
    main()
