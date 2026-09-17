"""Check #2: with a very large number of episodes, IPS/SNIPS/DR's Spearman
correlation with the causal oracle should approach 1 (they are consistent
estimators of that quantity under correct known propensities). Memory Worth
is not expected to approach 1 -- it targets a different quantity (P(success |
included)), not the causal contrast.

Usage:
    python -m memory_ope.evaluation.large_n_sanity
"""

from __future__ import annotations

import json
import pathlib

from .. import config as config_mod
from ..estimators import doubly_robust, ips, memory_worth
from ..evaluation import metrics
from ..simulator import hitchhiker, task_difficulty

N_LARGE = 200_000
SEED = 0


def _run(name: str, cfg: dict) -> dict:
    if name == "task_difficulty":
        true_vals = task_difficulty.true_values(cfg)
        all_ids = task_difficulty.all_ids(cfg)
        episodes = task_difficulty.generate_episodes(cfg, seed=SEED, n_episodes=N_LARGE)
    else:
        true_vals = hitchhiker.true_values(cfg)
        all_ids = hitchhiker.all_ids(cfg)
        episodes = hitchhiker.generate_episodes(cfg, seed=SEED, n_episodes=N_LARGE)

    mw = memory_worth.compute(episodes)
    ips_est, snips_est = ips.compute(episodes)
    dr_est = doubly_robust.compute(episodes, all_ids, l2_c=cfg["estimators"]["dr_l2_C"], cross_fit=True)

    return {
        "n_episodes": N_LARGE,
        "memory_worth_spearman": metrics.spearman(mw, true_vals),
        "ips_spearman": metrics.spearman(ips_est, true_vals),
        "snips_spearman": metrics.spearman(snips_est, true_vals),
        "doubly_robust_spearman": metrics.spearman(dr_est, true_vals),
    }


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    report = {"task_difficulty": _run("task_difficulty", cfg), "hitchhiker": _run("hitchhiker", cfg)}

    with open(results_dir / "large_n_sanity.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for sim_name, r in report.items():
        print(f"\n=== {sim_name} (n={r['n_episodes']}, seed={SEED}) vs causal oracle ===")
        print(f"  memory_worth   spearman = {r['memory_worth_spearman']:.4f}")
        print(f"  ips            spearman = {r['ips_spearman']:.4f}")
        print(f"  snips          spearman = {r['snips_spearman']:.4f}")
        print(f"  doubly_robust  spearman = {r['doubly_robust_spearman']:.4f}")

    print(f"\nWrote {results_dir / 'large_n_sanity.json'}")


if __name__ == "__main__":
    main()
