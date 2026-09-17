"""Check: does selecting ground-truth memories (e.g. "biggest MW-vs-DR
disagreement") from the SAME log used to compute the estimates being
judged introduce a winner's-curse-style selection bias?

Design chosen: SPLIT the log in half by episode index (first half = set A,
second half = set B; a fixed rule decided in advance, not tuned after
looking at results). Selection (which memories go into the expensive
forced-in/forced-out ground-truth rerun set) is done using ONLY set A's
estimates. The estimates that get compared against ground truth -- the
"official" Stage 2 result -- are computed from ONLY set B. This means the
memories were never chosen by looking at the same numbers that get
evaluated against ground truth.

This check demonstrates the effect empirically: select the top-K
MW-vs-doubly_robust disagreement memories using set A, then compare that
same disagreement measured in-sample (on A again) vs out-of-sample (on B)
-- in-sample disagreement for the selected memories should be inflated
relative to out-of-sample, since A's own noise is what got them selected in
the first place.

Disagreement is measured by RANK, not raw value: memory_worth (~0.5 scale)
and doubly_robust (~0.03-0.06 scale) live on different scales entirely, so
a raw |MW - DR| difference is dominated by that constant offset for every
memory regardless of whether they actually rank it differently -- it would
select almost arbitrarily rather than picking real disagreements. Using
each estimator's percentile rank within its own distribution makes the
comparison scale-free and is what "disagreement" should mean here: do MW
and DR put this memory in a different part of the ranking.

Usage:
    python -m memory_ope.evaluation.ground_truth_selection_bias_check
"""

from __future__ import annotations

import copy
import json
import pathlib

import numpy as np

from .. import config as config_mod
from ..estimators import doubly_robust, memory_worth
from ..simulator import task_difficulty

N_EPISODES_TOTAL = 2000  # split into two halves of 1000 each
N_MEMORIES = 30  # per signal_boost_check's recommendation
PROPENSITY_MIN, PROPENSITY_MAX = 0.3, 0.7
TOP_K = 5
N_SEEDS = 20


def _percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    ids = list(values.keys())
    order = sorted(ids, key=lambda m: values[m])
    n = len(order)
    return {mem_id: rank / (n - 1) for rank, mem_id in enumerate(order)} if n > 1 else {order[0]: 0.5}


def _split_episodes(cfg: dict, seed: int) -> tuple[list[dict], list[dict]]:
    episodes = task_difficulty.generate_episodes(
        cfg, seed, n_episodes=N_EPISODES_TOTAL, propensity_min=PROPENSITY_MIN, propensity_max=PROPENSITY_MAX
    )
    half = N_EPISODES_TOTAL // 2
    return episodes[:half], episodes[half:]


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    n_gen = round(N_MEMORIES * 0.7)
    n_spec = N_MEMORIES - n_gen
    scaled = copy.deepcopy(cfg)
    scaled["simulators"]["task_difficulty"]["n_generalists"] = n_gen
    scaled["simulators"]["task_difficulty"]["n_specialists"] = n_spec
    all_ids = task_difficulty.all_ids(scaled)
    dr_l2_c = scaled["estimators"]["dr_l2_C"]

    in_sample_disagreement = []
    out_sample_disagreement = []
    selected_memories_by_seed = []

    for seed in range(N_SEEDS):
        set_a, set_b = _split_episodes(scaled, seed)

        mw_a = memory_worth.compute(set_a)
        dr_a = doubly_robust.compute(set_a, all_ids, l2_c=dr_l2_c, cross_fit=True)
        mw_b = memory_worth.compute(set_b)
        dr_b = doubly_robust.compute(set_b, all_ids, l2_c=dr_l2_c, cross_fit=True)

        common = [m for m in all_ids if m in mw_a and m in dr_a]
        rank_mw_a = _percentile_ranks({m: mw_a[m] for m in common})
        rank_dr_a = _percentile_ranks({m: dr_a[m] for m in common})
        disagreement_a = {m: abs(rank_mw_a[m] - rank_dr_a[m]) for m in common}
        selected = sorted(common, key=lambda m: disagreement_a[m], reverse=True)[:TOP_K]
        selected_memories_by_seed.append(selected)

        common_b = [m for m in common if m in mw_b and m in dr_b]
        rank_mw_b = _percentile_ranks({m: mw_b[m] for m in common_b})
        rank_dr_b = _percentile_ranks({m: dr_b[m] for m in common_b})

        in_sample = np.mean([disagreement_a[m] for m in selected])
        out_sample = np.mean([abs(rank_mw_b[m] - rank_dr_b[m]) for m in selected if m in rank_mw_b])
        in_sample_disagreement.append(float(in_sample))
        out_sample_disagreement.append(float(out_sample))

    report = {
        "design": "fixed split by episode index (first half = selection set A, second half = evaluation set B), decided in advance",
        "n_episodes_total": N_EPISODES_TOTAL,
        "n_memories": N_MEMORIES,
        "propensity_range": [PROPENSITY_MIN, PROPENSITY_MAX],
        "top_k": TOP_K,
        "n_seeds": N_SEEDS,
        "in_sample_disagreement_mean": float(np.mean(in_sample_disagreement)),
        "out_of_sample_disagreement_mean": float(np.mean(out_sample_disagreement)),
        "inflation_ratio": float(np.mean(in_sample_disagreement) / np.mean(out_sample_disagreement)),
        "per_seed": {
            "in_sample": in_sample_disagreement,
            "out_of_sample": out_sample_disagreement,
        },
    }

    with open(results_dir / "ground_truth_selection_bias_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Selected top-{TOP_K} MW-vs-DR RANK-disagreement memories using set A, then re-measured disagreement:")
    print(f"  in-sample (on A, same data used to select)  mean |rank_MW-rank_DR| = {report['in_sample_disagreement_mean']:.4f}")
    print(f"  out-of-sample (on B, independent data)       mean |rank_MW-rank_DR| = {report['out_of_sample_disagreement_mean']:.4f}")
    print(f"  inflation ratio (in-sample / out-of-sample)  = {report['inflation_ratio']:.2f}x")
    print(f"\nWrote {results_dir / 'ground_truth_selection_bias_check.json'}")


if __name__ == "__main__":
    main()
