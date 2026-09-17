"""Check: run task_difficulty at roughly the scale planned for the ALFWorld
pilot -- 50 memories (scaled down from 100, keeping the 70:30
generalist:specialist ratio -> 35:15), M=10 candidates (unchanged),
n_episodes=1000 (the main-logging-phase size from the Stage 2 budget
estimate), same propensity range. Reports Spearman, MAE, and bias with 95%
seed-percentile intervals over 20 seeds.

Usage:
    python -m memory_ope.evaluation.stage2_scale_check
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
N_GENERALISTS = 35
N_SPECIALISTS = 15
ESTIMATOR_NAMES = ["memory_worth", "ips", "snips", "doubly_robust"]


def _interval(vals: list[float]) -> dict[str, float]:
    arr = np.array(vals, dtype=float)
    return {"mean": float(np.mean(arr)), "p2_5": float(np.percentile(arr, 2.5)), "p97_5": float(np.percentile(arr, 97.5))}


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    scaled_cfg = copy.deepcopy(cfg)
    scaled_cfg["simulators"]["task_difficulty"]["n_generalists"] = N_GENERALISTS
    scaled_cfg["simulators"]["task_difficulty"]["n_specialists"] = N_SPECIALISTS
    # M and propensity range are untouched (cfg["retrieval"], unchanged).

    true_vals = task_difficulty.true_values(scaled_cfg)
    all_ids = task_difficulty.all_ids(scaled_cfg)
    dr_l2_c = scaled_cfg["estimators"]["dr_l2_C"]
    seed_count = scaled_cfg["seed_count"]

    per_est = {e: {"spearman": [], "mae": [], "bias": []} for e in ESTIMATOR_NAMES}
    for seed in range(seed_count):
        episodes = task_difficulty.generate_episodes(scaled_cfg, seed, n_episodes=N_EPISODES)
        mw = memory_worth.compute(episodes)
        ips_est, snips_est = ips.compute(episodes)
        dr_est = doubly_robust.compute(episodes, all_ids, l2_c=dr_l2_c, cross_fit=True)
        estimates = {"memory_worth": mw, "ips": ips_est, "snips": snips_est, "doubly_robust": dr_est}
        for est_name in ESTIMATOR_NAMES:
            e = estimates[est_name]
            per_est[est_name]["spearman"].append(metrics.spearman(e, true_vals))
            per_est[est_name]["mae"].append(metrics.mean_absolute_error(e, true_vals))
            per_est[est_name]["bias"].append(metrics.mean_signed_bias(e, true_vals))

    report = {
        "n_memories": N_GENERALISTS + N_SPECIALISTS,
        "n_generalists": N_GENERALISTS,
        "n_specialists": N_SPECIALISTS,
        "M": cfg["retrieval"]["M"],
        "n_episodes": N_EPISODES,
        "propensity_min": cfg["retrieval"]["propensity_min"],
        "propensity_max": cfg["retrieval"]["propensity_max"],
        "seed_count": seed_count,
        "estimators": {e: {m: _interval(v) for m, v in d.items()} for e, d in per_est.items()},
    }

    with open(results_dir / "stage2_scale_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"task_difficulty at Stage-2 scale: {report['n_memories']} memories "
        f"({N_GENERALISTS} generalist + {N_SPECIALISTS} specialist), M={report['M']}, "
        f"n_episodes={N_EPISODES}, propensity=[{report['propensity_min']},{report['propensity_max']}], "
        f"{seed_count} seeds\n"
    )
    print(f"{'estimator':<16}{'spearman':>26}{'MAE':>26}{'bias':>26}")
    for est_name in ESTIMATOR_NAMES:
        r = report["estimators"][est_name]
        sp, mae, bias = r["spearman"], r["mae"], r["bias"]
        print(
            f"{est_name:<16}"
            f"{sp['mean']:>8.3f} [{sp['p2_5']:>6.3f},{sp['p97_5']:>6.3f}]"
            f"{mae['mean']:>8.3f} [{mae['p2_5']:>6.3f},{mae['p97_5']:>6.3f}]"
            f"{bias['mean']:>8.3f} [{bias['p2_5']:>6.3f},{bias['p97_5']:>6.3f}]"
        )

    print(f"\nWrote {results_dir / 'stage2_scale_check.json'}")


if __name__ == "__main__":
    main()
