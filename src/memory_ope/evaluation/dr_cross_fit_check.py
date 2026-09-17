"""Check #6: doubly robust with vs without cross-fitting. Without
cross-fitting, the outcome model is fit and evaluated on the same data, which
can let an overfit model quietly absorb some of the treatment signal into
mu(X,Z) and bias the AIPW correction term. Cross-fitting (K=5 folds here)
evaluates mu only out-of-fold.

Usage:
    python -m memory_ope.evaluation.dr_cross_fit_check
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .. import config as config_mod
from ..estimators import doubly_robust
from ..evaluation import metrics
from ..simulator import hitchhiker, task_difficulty


def _run(name: str, cfg: dict) -> dict:
    if name == "task_difficulty":
        true_vals = task_difficulty.true_values(cfg)
        all_ids = task_difficulty.all_ids(cfg)
        gen = task_difficulty.generate_episodes
    else:
        true_vals = hitchhiker.true_values(cfg)
        all_ids = hitchhiker.all_ids(cfg)
        gen = hitchhiker.generate_episodes

    dr_l2_c = cfg["estimators"]["dr_l2_C"]
    seed_count = cfg["seed_count"]

    corr_cf, corr_no_cf = [], []
    est_cf_by_seed, est_no_cf_by_seed = [], []
    for seed in range(seed_count):
        episodes = gen(cfg, seed)
        dr_cf = doubly_robust.compute(episodes, all_ids, l2_c=dr_l2_c, cross_fit=True)
        dr_no_cf = doubly_robust.compute(episodes, all_ids, l2_c=dr_l2_c, cross_fit=False)
        corr_cf.append(metrics.spearman(dr_cf, true_vals))
        corr_no_cf.append(metrics.spearman(dr_no_cf, true_vals))
        est_cf_by_seed.append(dr_cf)
        est_no_cf_by_seed.append(dr_no_cf)

    bias_cf, var_cf = metrics.bias_and_variance(est_cf_by_seed, true_vals)
    bias_no_cf, var_no_cf = metrics.bias_and_variance(est_no_cf_by_seed, true_vals)

    return {
        "cross_fit": {
            "spearman_mean": float(np.mean(corr_cf)),
            "spearman_std": float(np.std(corr_cf)),
            "bias_mean": metrics.summarize(bias_cf)["mean"],
            "variance_mean": metrics.summarize(var_cf)["mean"],
        },
        "no_cross_fit": {
            "spearman_mean": float(np.mean(corr_no_cf)),
            "spearman_std": float(np.std(corr_no_cf)),
            "bias_mean": metrics.summarize(bias_no_cf)["mean"],
            "variance_mean": metrics.summarize(var_no_cf)["mean"],
        },
    }


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    report = {"task_difficulty": _run("task_difficulty", cfg), "hitchhiker": _run("hitchhiker", cfg)}

    with open(results_dir / "dr_cross_fit_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for sim_name, r in report.items():
        print(f"\n=== {sim_name} (doubly robust, {cfg['seed_count']} seeds, default n_episodes) ===")
        print(f"{'variant':<16}{'spearman mean':>15}{'spearman std':>15}{'bias mean':>12}{'variance mean':>15}")
        for variant in ("cross_fit", "no_cross_fit"):
            v = r[variant]
            print(
                f"{variant:<16}{v['spearman_mean']:>15.3f}{v['spearman_std']:>15.3f}"
                f"{v['bias_mean']:>12.3f}{v['variance_mean']:>15.5f}"
            )

    print(f"\nWrote {results_dir / 'dr_cross_fit_check.json'}")


if __name__ == "__main__":
    main()
