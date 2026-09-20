"""Zero-cost: selects which memories go into the mini ground-truth rerun,
using the SAME split-by-episode-index design already validated (root
package's ground_truth_selection_bias_check.py) to avoid the ~1.87x
selection-bias inflation that design demonstrated: selecting from the same
data you then cite as evidence overstates disagreement.

Design (fixed in advance, matching the established pattern):
  - Set A = first half of small_pilot.jsonl (episodes 0-89) -- used ONLY to
    SELECT which memories disagree most between memory_worth and ips.
  - Set B = second half (episodes 90-179) -- used ONLY to compute the
    "official" memory_worth/ips/snips/doubly_robust estimates that get
    reported and compared against ground truth.

Disagreement is measured by RANK (percentile rank within each estimator's
own distribution), not raw value -- memory_worth (~0.4-0.9 scale) and ips
(~-0.5 to 0.4 scale) live on different scales entirely, so a raw
difference would be dominated by that offset regardless of real rank
disagreement.

Usage:
    python -m alfworld_pilot.mini_ground_truth_selection
"""

from __future__ import annotations

import json
import pathlib

from memory_ope.estimators import doubly_robust, ips, memory_worth

from . import config as config_mod

TOP_K = 5


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    episodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def _percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    ids = list(values.keys())
    order = sorted(ids, key=lambda m: values[m])
    n = len(order)
    return {mem_id: rank / (n - 1) for rank, mem_id in enumerate(order)} if n > 1 else {order[0]: 0.5}


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])
    logs_dir = pathlib.Path(cfg["paths"]["logs_dir"])
    all_mem_ids = [f"mem_{i}" for i in range(cfg["memory_store"]["n_memories"])]

    episodes = _load_jsonl(logs_dir / "small_pilot.jsonl")
    half = len(episodes) // 2
    set_a, set_b = episodes[:half], episodes[half:]
    print(f"n_episodes total={len(episodes)}, set A (selection)={len(set_a)}, set B (evaluation)={len(set_b)}")

    mw_a = memory_worth.compute(set_a)
    ips_a, _snips_a = ips.compute(set_a)
    common_a = [m for m in all_mem_ids if m in mw_a and m in ips_a]
    rank_mw_a = _percentile_ranks({m: mw_a[m] for m in common_a})
    rank_ips_a = _percentile_ranks({m: ips_a[m] for m in common_a})
    disagreement_a = {m: abs(rank_mw_a[m] - rank_ips_a[m]) for m in common_a}
    selected = sorted(common_a, key=lambda m: disagreement_a[m], reverse=True)[:TOP_K]

    print(f"\nTop-{TOP_K} memory_worth-vs-ips RANK disagreement (set A, n={len(set_a)}):")
    for m in selected:
        print(f"  {m}: mw={mw_a[m]:.3f} (rank {rank_mw_a[m]:.2f})  ips={ips_a[m]:.4f} (rank {rank_ips_a[m]:.2f})  disagreement={disagreement_a[m]:.2f}")

    # "Official" estimates from the HELD-OUT set B, for later comparison against ground truth.
    mw_b = memory_worth.compute(set_b)
    ips_b, snips_b = ips.compute(set_b)
    dr_b = doubly_robust.compute(set_b, all_mem_ids, l2_c=1.0, cross_fit=True)

    def _rank_of(m, est):
        order = sorted(est.keys(), key=lambda x: est[x], reverse=True)
        return order.index(m) + 1 if m in order else None

    print(f"\n'Official' set-B estimates (n={len(set_b)}) for the selected memories:")
    official = {}
    for m in selected:
        official[m] = {
            "memory_worth": {"value": mw_b.get(m), "rank": _rank_of(m, mw_b)},
            "ips": {"value": ips_b.get(m), "rank": _rank_of(m, ips_b)},
            "snips": {"value": snips_b.get(m), "rank": _rank_of(m, snips_b)},
            "doubly_robust": {"value": dr_b.get(m), "rank": _rank_of(m, dr_b)},
        }
        print(f"  {m}: mw={mw_b.get(m):.3f}(#{official[m]['memory_worth']['rank']})  "
              f"ips={ips_b.get(m):.4f}(#{official[m]['ips']['rank']})  "
              f"snips={snips_b.get(m):.4f}(#{official[m]['snips']['rank']})  "
              f"dr={dr_b.get(m):.4f}(#{official[m]['doubly_robust']['rank']})")

    report = {
        "design": "fixed split by episode index (first half=selection set A, second half=evaluation set B)",
        "n_episodes_total": len(episodes),
        "n_set_a": len(set_a),
        "n_set_b": len(set_b),
        "top_k": TOP_K,
        "selected_memories": selected,
        "set_a_disagreement": {m: disagreement_a[m] for m in selected},
        "official_set_b_estimates": official,
    }
    with open(results_dir / "mini_ground_truth_selection.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {results_dir / 'mini_ground_truth_selection.json'}")


if __name__ == "__main__":
    main()
