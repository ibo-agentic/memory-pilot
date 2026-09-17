"""Small-data regime (n = 250, 500, 1000, 2000 episodes) for task_difficulty
-- the realistic order of magnitude for the ALFWorld pilot. Reports Spearman
AND the tie-free mean absolute error / mean signed bias against the causal
oracle, mean + 95% seed-percentile interval over 20 seeds, for all four
estimators.

Usage:
    python -m memory_ope.evaluation.small_data_results
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .. import config as config_mod
from ..estimators import doubly_robust, ips, memory_worth
from ..evaluation import metrics
from ..simulator import task_difficulty

N_VALUES = [250, 500, 1000, 2000]
ESTIMATOR_NAMES = ["memory_worth", "ips", "snips", "doubly_robust"]


def _interval(vals: list[float]) -> dict[str, float]:
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "p2_5": float(np.percentile(arr, 2.5)),
        "p97_5": float(np.percentile(arr, 97.5)),
    }


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    true_vals = task_difficulty.true_values(cfg)
    all_ids = task_difficulty.all_ids(cfg)
    dr_l2_c = cfg["estimators"]["dr_l2_C"]
    seed_count = cfg["seed_count"]

    report = {}
    for n in N_VALUES:
        per_est = {e: {"spearman": [], "mae": [], "bias": []} for e in ESTIMATOR_NAMES}
        for seed in range(seed_count):
            episodes = task_difficulty.generate_episodes(cfg, seed, n_episodes=n)
            mw = memory_worth.compute(episodes)
            ips_est, snips_est = ips.compute(episodes)
            dr_est = doubly_robust.compute(episodes, all_ids, l2_c=dr_l2_c, cross_fit=True)
            estimates = {"memory_worth": mw, "ips": ips_est, "snips": snips_est, "doubly_robust": dr_est}
            for est_name in ESTIMATOR_NAMES:
                e = estimates[est_name]
                per_est[est_name]["spearman"].append(metrics.spearman(e, true_vals))
                per_est[est_name]["mae"].append(metrics.mean_absolute_error(e, true_vals))
                per_est[est_name]["bias"].append(metrics.mean_signed_bias(e, true_vals))

        report[n] = {est_name: {metric: _interval(vals) for metric, vals in d.items()} for est_name, d in per_est.items()}

    with open(results_dir / "small_data_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for n in N_VALUES:
        print(f"\n=== n_episodes = {n} (task_difficulty, {seed_count} seeds, mean [95% interval]) ===")
        print(f"{'estimator':<16}{'spearman':>26}{'MAE':>26}{'bias':>26}")
        for est_name in ESTIMATOR_NAMES:
            r = report[n][est_name]
            sp, mae, bias = r["spearman"], r["mae"], r["bias"]
            print(
                f"{est_name:<16}"
                f"{sp['mean']:>8.3f} [{sp['p2_5']:>6.3f},{sp['p97_5']:>6.3f}]"
                f"{mae['mean']:>8.3f} [{mae['p2_5']:>6.3f},{mae['p97_5']:>6.3f}]"
                f"{bias['mean']:>8.3f} [{bias['p2_5']:>6.3f},{bias['p97_5']:>6.3f}]"
            )

    print(f"\nWrote {results_dir / 'small_data_results.json'}")


if __name__ == "__main__":
    main()
