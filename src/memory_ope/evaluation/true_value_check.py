"""Check #1: compare the causal oracle, the naive observational contrast, and
the raw utility parameter, for both simulators. Confirms true_values() is not
just echoing the utility parameter, and quantifies how much the naive
(factual-conditioning) contrast diverges from the causal (forced-intervention)
one under the designed confounding.

Standardized ground truth for all other checks: the CAUSAL oracle.

Usage:
    python -m memory_ope.evaluation.true_value_check
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.stats import spearmanr

from .. import config as config_mod
from ..simulator import hitchhiker, task_difficulty


def _compare(name: str, causal: dict, naive: dict, utility: dict) -> dict:
    ids = list(causal.keys())
    causal_arr = np.array([causal[i] for i in ids])
    naive_arr = np.array([naive[i] for i in ids])
    utility_arr = np.array([utility[i] for i in ids])

    rho_causal_naive, _ = spearmanr(causal_arr, naive_arr)
    rho_causal_utility, _ = spearmanr(causal_arr, utility_arr)
    rho_naive_utility, _ = spearmanr(naive_arr, utility_arr)

    return {
        "simulator": name,
        "n_memories": len(ids),
        "spearman_causal_vs_naive": float(rho_causal_naive),
        "spearman_causal_vs_utility": float(rho_causal_utility),
        "spearman_naive_vs_utility": float(rho_naive_utility),
        "mean_abs_diff_causal_minus_naive": float(np.mean(np.abs(causal_arr - naive_arr))),
        "max_abs_diff_causal_minus_naive": float(np.max(np.abs(causal_arr - naive_arr))),
        "causal_values": causal,
        "naive_values": naive,
        "utility_values": utility,
    }


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    td_ids, td_utility_arr = task_difficulty._build_ids_and_utility(cfg)
    td_utility = {mem_id: float(v) for mem_id, v in zip(td_ids, td_utility_arr)}
    td_report = _compare(
        "task_difficulty",
        task_difficulty.true_values(cfg),
        task_difficulty.naive_observational_values(cfg),
        td_utility,
    )

    hh_ids, hh_utility_arr = hitchhiker._build_ids_and_utility(cfg)
    hh_utility = {mem_id: float(v) for mem_id, v in zip(hh_ids, hh_utility_arr)}
    hh_report = _compare(
        "hitchhiker",
        hitchhiker.true_values(cfg),
        hitchhiker.naive_observational_values(cfg),
        hh_utility,
    )

    report = {"task_difficulty": td_report, "hitchhiker": hh_report}
    with open(results_dir / "true_value_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for r in (td_report, hh_report):
        print(f"\n=== {r['simulator']} ===")
        print(f"  spearman(causal, naive)    = {r['spearman_causal_vs_naive']:.3f}")
        print(f"  spearman(causal, utility)  = {r['spearman_causal_vs_utility']:.3f}")
        print(f"  spearman(naive, utility)   = {r['spearman_naive_vs_utility']:.3f}")
        print(f"  mean |causal - naive|      = {r['mean_abs_diff_causal_minus_naive']:.4f}")
        print(f"  max  |causal - naive|      = {r['max_abs_diff_causal_minus_naive']:.4f}")

    print(f"\nWrote {results_dir / 'true_value_check.json'}")


if __name__ == "__main__":
    main()
