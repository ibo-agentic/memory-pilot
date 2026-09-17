"""Check #4: sweep how extreme the inclusion propensities are (near-uniform
0.5-ish vs near-deterministic 0.98-ish), holding everything else fixed, and
see how each estimator's accuracy and effective sample size change.

This is a DIFFERENT axis from the hitchhiker simulator's
"independent_retrieval_fraction" -- see the module docstring below for why.

Usage:
    python -m memory_ope.evaluation.randomness_sweep
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import config as config_mod
from ..estimators import diagnostics, doubly_robust, ips, memory_worth
from ..evaluation import metrics
from ..simulator import task_difficulty

# (propensity_min, propensity_max) pairs, from near-uniform/50-50 retrieval
# to near-deterministic retrieval.
PROPENSITY_RANGES = [
    (0.45, 0.55),
    (0.3, 0.7),
    (0.1, 0.9),   # current default
    (0.05, 0.95),
    (0.02, 0.98),
]
ESTIMATOR_NAMES = ["memory_worth", "ips", "snips", "doubly_robust"]

EXPLANATION = """
"Independent" is being used for two unrelated things in this project, and
check #4 is about the one that hasn't been swept yet:

1. INCLUSION independence (retrieval.py / _common.draw_inclusion): given a
   candidate's propensity p_i, its inclusion Z_i ~ Bernoulli(p_i) is drawn
   independently of every other candidate's draw, for every episode. This is
   fixed by design and is NOT what this sweep varies -- it's already true
   regardless of how spread out p_i is.

2. CANDIDACY correlation (hitchhiker.py's independent_retrieval_fraction):
   controls how often the hitchhiker's *similarity score* (hence whether it
   ends up in the top-M at all) is drawn independently of the anchor's,
   versus jittered to track it. This affects whether two memories tend to be
   CANDIDATES together, not whether their inclusion draws are independent
   given candidacy (which they always are, per #1).

This sweep is about a third, separate axis: how far propensity_min/max sit
from 0.5. Propensities near 0.5 mean retrieval is close to a fair coin flip
(most exploratory, best-behaved importance weights). Propensities near 0/1
mean retrieval is close to deterministic top-M (closer to plain top-k),
which is more realistic/less exploratory but gives IPS-style estimators much
higher-variance importance weights (1/p or 1/(1-p) blows up) and lower
effective sample size.
"""


def _run_range(cfg: dict, p_min: float, p_max: float) -> dict:
    seed_count = cfg["seed_count"]
    dr_l2_c = cfg["estimators"]["dr_l2_C"]
    all_ids = task_difficulty.all_ids(cfg)
    true_vals = task_difficulty.true_values(cfg, propensity_min=p_min, propensity_max=p_max)

    corr = {est: [] for est in ESTIMATOR_NAMES}
    ess_pos_all, ess_neg_all = [], []

    for seed in range(seed_count):
        episodes = task_difficulty.generate_episodes(cfg, seed, propensity_min=p_min, propensity_max=p_max)
        mw = memory_worth.compute(episodes)
        ips_est, snips_est = ips.compute(episodes)
        dr_est = doubly_robust.compute(episodes, all_ids, l2_c=dr_l2_c, cross_fit=True)
        estimates = {"memory_worth": mw, "ips": ips_est, "snips": snips_est, "doubly_robust": dr_est}
        for est_name in ESTIMATOR_NAMES:
            corr[est_name].append(metrics.spearman(estimates[est_name], true_vals))

        ess = diagnostics.effective_sample_size(episodes)
        ess_summary = diagnostics.summarize_ess(ess)
        ess_pos_all.append(ess_summary["mean_ess_pos_ratio"])
        ess_neg_all.append(ess_summary["mean_ess_neg_ratio"])

    return {
        "propensity_min": p_min,
        "propensity_max": p_max,
        "spearman": {est: {"mean": float(np.mean(v)), "std": float(np.std(v))} for est, v in corr.items()},
        "mean_ess_pos_ratio": float(np.mean(ess_pos_all)),
        "mean_ess_neg_ratio": float(np.mean(ess_neg_all)),
    }


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    rows = [_run_range(cfg, p_min, p_max) for p_min, p_max in PROPENSITY_RANGES]
    report = {"explanation": EXPLANATION, "rows": rows}

    with open(results_dir / "randomness_sweep.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(EXPLANATION)
    print(f"{'range':>14}{'ess_pos_ratio':>15}{'ess_neg_ratio':>15}" + "".join(f"{e:>16}" for e in ESTIMATOR_NAMES))
    for row in rows:
        label = f"[{row['propensity_min']},{row['propensity_max']}]"
        line = f"{label:>14}{row['mean_ess_pos_ratio']:>15.3f}{row['mean_ess_neg_ratio']:>15.3f}"
        for est_name in ESTIMATOR_NAMES:
            line += f"{row['spearman'][est_name]['mean']:>16.3f}"
        print(line)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [f"[{r['propensity_min']},{r['propensity_max']}]" for r in rows]
    for est_name in ESTIMATOR_NAMES:
        axes[0].plot(labels, [r["spearman"][est_name]["mean"] for r in rows], marker="o", label=est_name)
    axes[0].set_ylabel("spearman vs causal oracle")
    axes[0].set_xlabel("[propensity_min, propensity_max]")
    axes[0].set_title("Accuracy vs propensity extremity")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(fontsize=8)

    axes[1].plot(labels, [r["mean_ess_pos_ratio"] for r in rows], marker="o", label="ESS ratio (included arm)")
    axes[1].plot(labels, [r["mean_ess_neg_ratio"] for r in rows], marker="s", label="ESS ratio (excluded arm)")
    axes[1].set_ylabel("mean ESS / n (per memory)")
    axes[1].set_xlabel("[propensity_min, propensity_max]")
    axes[1].set_title("Effective sample size vs propensity extremity")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "randomness_sweep.png", dpi=130)
    plt.close(fig)

    print(f"\nWrote {results_dir / 'randomness_sweep.json'} and randomness_sweep.png")


if __name__ == "__main__":
    main()
