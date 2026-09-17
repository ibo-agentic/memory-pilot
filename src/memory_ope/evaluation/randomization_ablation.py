"""Check: separate what randomization ALONE fixes from what the IPS/DR
propensity correction fixes.

Two retrieval policies, same top-M candidate selection in both:
  (a) deterministic top-k: every candidate is deterministically included
      (implemented by calling the existing retrieval code with
      propensity_min = propensity_max = 1.0, so the Bernoulli draw is
      degenerate and always returns 1 -- no new mechanism, just a
      degenerate case of the same code path).
  (b) our randomized retrieval: propensities in [propensity_min, propensity_max]
      as configured, independent Bernoulli inclusion per candidate.

Memory Worth is computed under BOTH (its formula never looks at propensity,
so this isolates whether merely having randomized inclusion -- independent
of any propensity-based correction -- changes which episodes end up in its
success-rate calculation enough to help). IPS/SNIPS/DR are only defined
under (b): under (a), every candidate has Z=1 always, so the "not included"
importance weight 1/(1-p) is 1/0 -- there is no counterfactual data at all,
which is exactly the point (a fixed top-k policy cannot support off-policy
correction; this is not a bug to route around).

For hitchhiker, independent_retrieval_fraction is held at 0.0 in BOTH arms
(anchor and hitchhiker always candidates together, matching Simsek's 0%
independence case) so the only axis varying between (a) and (b) is
deterministic-vs-randomized inclusion, not candidacy correlation.

Usage:
    python -m memory_ope.evaluation.randomization_ablation
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .. import config as config_mod
from ..estimators import doubly_robust, ips, memory_worth
from ..evaluation import metrics
from ..simulator import hitchhiker, task_difficulty


def _run_task_difficulty(cfg: dict) -> dict:
    seed_count = cfg["seed_count"]
    dr_l2_c = cfg["estimators"]["dr_l2_C"]
    all_ids = task_difficulty.all_ids(cfg)

    true_det = task_difficulty.true_values(cfg, propensity_min=1.0, propensity_max=1.0)
    true_rand = task_difficulty.true_values(cfg)

    mw_det_corr, mw_rand_corr = [], []
    ips_corr, snips_corr, dr_corr = [], [], []
    for seed in range(seed_count):
        det_episodes = task_difficulty.generate_episodes(cfg, seed, propensity_min=1.0, propensity_max=1.0)
        rand_episodes = task_difficulty.generate_episodes(cfg, seed)

        mw_det = memory_worth.compute(det_episodes)
        mw_rand = memory_worth.compute(rand_episodes)
        mw_det_corr.append(metrics.spearman(mw_det, true_det))
        mw_rand_corr.append(metrics.spearman(mw_rand, true_rand))

        ips_est, snips_est = ips.compute(rand_episodes)
        dr_est = doubly_robust.compute(rand_episodes, all_ids, l2_c=dr_l2_c, cross_fit=True)
        ips_corr.append(metrics.spearman(ips_est, true_rand))
        snips_corr.append(metrics.spearman(snips_est, true_rand))
        dr_corr.append(metrics.spearman(dr_est, true_rand))

    return _summarize(mw_det_corr, mw_rand_corr, ips_corr, snips_corr, dr_corr)


def _run_hitchhiker(cfg: dict) -> dict:
    seed_count = cfg["seed_count"]
    dr_l2_c = cfg["estimators"]["dr_l2_C"]
    all_ids = hitchhiker.all_ids(cfg)

    true_det = hitchhiker.true_values(cfg, independent_fraction=0.0, propensity_min=1.0, propensity_max=1.0)
    true_rand = hitchhiker.true_values(cfg, independent_fraction=0.0)
    true_gap_det = true_det["anchor"] - true_det["hitchhiker"]
    true_gap_rand = true_rand["anchor"] - true_rand["hitchhiker"]

    mw_det_corr, mw_rand_corr = [], []
    ips_corr, snips_corr, dr_corr = [], [], []

    pair_estimators = ["memory_worth_deterministic_topk", "memory_worth_randomized", "ips", "snips", "doubly_robust"]
    anchor_above = {e: [] for e in pair_estimators}
    gaps = {e: [] for e in pair_estimators}

    for seed in range(seed_count):
        det_episodes = hitchhiker.generate_episodes(
            cfg, seed, independent_fraction=0.0, propensity_min=1.0, propensity_max=1.0
        )
        rand_episodes = hitchhiker.generate_episodes(cfg, seed, independent_fraction=0.0)

        mw_det = memory_worth.compute(det_episodes)
        mw_rand = memory_worth.compute(rand_episodes)
        mw_det_corr.append(metrics.spearman(mw_det, true_det))
        mw_rand_corr.append(metrics.spearman(mw_rand, true_rand))

        ips_est, snips_est = ips.compute(rand_episodes)
        dr_est = doubly_robust.compute(rand_episodes, all_ids, l2_c=dr_l2_c, cross_fit=True)
        ips_corr.append(metrics.spearman(ips_est, true_rand))
        snips_corr.append(metrics.spearman(snips_est, true_rand))
        dr_corr.append(metrics.spearman(dr_est, true_rand))

        per_estimator_values = {
            "memory_worth_deterministic_topk": mw_det,
            "memory_worth_randomized": mw_rand,
            "ips": ips_est,
            "snips": snips_est,
            "doubly_robust": dr_est,
        }
        for est_name, vals in per_estimator_values.items():
            a, h = vals["anchor"], vals["hitchhiker"]
            anchor_above[est_name].append(1.0 if a > h else 0.0)
            gaps[est_name].append(a - h)

    pair_metrics = {}
    for est_name in pair_estimators:
        true_gap = true_gap_det if est_name == "memory_worth_deterministic_topk" else true_gap_rand
        pair_metrics[est_name] = {
            "fraction_anchor_above_hitchhiker": float(np.mean(anchor_above[est_name])),
            "mean_estimated_gap": float(np.mean(gaps[est_name])),
            "true_gap": float(true_gap),
        }

    result = _summarize(mw_det_corr, mw_rand_corr, ips_corr, snips_corr, dr_corr)
    result["pair_metrics"] = pair_metrics
    return result


def _summarize(mw_det, mw_rand, ips_c, snips_c, dr_c) -> dict:
    return {
        "memory_worth_deterministic_topk": metrics.summarize({i: v for i, v in enumerate(mw_det)}),
        "memory_worth_randomized": metrics.summarize({i: v for i, v in enumerate(mw_rand)}),
        "ips_randomized": metrics.summarize({i: v for i, v in enumerate(ips_c)}),
        "snips_randomized": metrics.summarize({i: v for i, v in enumerate(snips_c)}),
        "doubly_robust_randomized": metrics.summarize({i: v for i, v in enumerate(dr_c)}),
    }


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    report = {
        "task_difficulty": _run_task_difficulty(cfg),
        "hitchhiker": _run_hitchhiker(cfg),
    }

    with open(results_dir / "randomization_ablation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for sim_name, r in report.items():
        print(f"\n=== {sim_name} (spearman vs each regime's own causal oracle, {cfg['seed_count']} seeds) ===")
        for key, v in r.items():
            if key == "pair_metrics":
                continue
            print(f"  {key:<32} mean={v['mean']:.3f}  std={v['std']:.3f}")

    print(f"\n=== hitchhiker pair metrics (anchor vs hitchhiker, {cfg['seed_count']} seeds) ===")
    for est_name, pm in report["hitchhiker"]["pair_metrics"].items():
        print(
            f"  {est_name:<32} fraction_anchor_above={pm['fraction_anchor_above_hitchhiker']:.2f}  "
            f"estimated_gap={pm['mean_estimated_gap']:.4f}  true_gap={pm['true_gap']:.4f}"
        )

    print(f"\nWrote {results_dir / 'randomization_ablation.json'}")


if __name__ == "__main__":
    main()
