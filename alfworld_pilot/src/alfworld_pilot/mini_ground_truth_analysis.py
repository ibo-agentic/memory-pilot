"""Final analysis for the mini ground truth run: for each of the 4
selected memories, computes the ground-truth causal value (paired
mean(Y_forced_in) - mean(Y_forced_out)) with a 95% CI using the ACTUAL
correlation rho measured from these real paired episodes (not the old
task_difficulty-simulator-derived proxy used for planning), then compares
against the memory_worth/ips/snips/doubly_robust "official" set-B
estimates already saved by mini_ground_truth_selection.py.

"Which estimator's ranking matches ground truth" is evaluated as pairwise
concordance over the C(4,2)=6 pairs among the 4 ground-truthed memories:
for each pair (m_a, m_b), does the estimator order them the same way
ground truth does? This is the well-defined comparison available at n=4
memories -- a full Spearman over 30 memories isn't possible since only 4
have ground truth.

Usage:
    python -m alfworld_pilot.mini_ground_truth_analysis
"""

from __future__ import annotations

import itertools
import json
import pathlib

import numpy as np

from . import config as config_mod

Z_95 = 1.96
MEMORIES = ["mem_17", "mem_29", "mem_21", "mem_9"]


def _load_pairs(log_path: pathlib.Path) -> list[tuple[dict, dict]]:
    lines = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        ep_in, ep_out = lines[i], lines[i + 1]
        assert ep_in["ground_truth_arm"] == "forced_in" and ep_out["ground_truth_arm"] == "forced_out"
        pairs.append((ep_in, ep_out))
    return pairs


def _ground_truth_stats(pairs: list[tuple[dict, dict]]) -> dict:
    y1 = np.array([ep_in["success"] for ep_in, _ in pairs], dtype=float)
    y0 = np.array([ep_out["success"] for _, ep_out in pairs], dtype=float)
    n = len(pairs)
    mean_y1, mean_y0 = float(y1.mean()), float(y0.mean())
    gap = mean_y1 - mean_y0
    var_y1, var_y0 = float(y1.var(ddof=1)), float(y0.var(ddof=1))
    if n < 2 or np.std(y1) == 0 or np.std(y0) == 0:
        rho = 0.0  # degenerate case (e.g. all-success or all-failure in one arm) -- fall back
        rho_is_degenerate = True  # to the conservative unpaired (rho=0) variance, not undefined/NaN
    else:
        rho = float(np.corrcoef(y1, y0)[0, 1])
        rho_is_degenerate = False
    var_d = var_y1 + var_y0 - 2 * rho * (var_y1 * var_y0) ** 0.5
    var_d = max(var_d, 0.0)  # guard against tiny negative from floating point when rho is close to 1
    se = (var_d / n) ** 0.5 if n > 0 else float("nan")
    ci_low, ci_high = gap - Z_95 * se, gap + Z_95 * se
    return {
        "n_pairs": n, "mean_y1": mean_y1, "mean_y0": mean_y0, "gap": gap,
        "var_y1": var_y1, "var_y0": var_y0, "rho": rho, "rho_is_degenerate_fallback": rho_is_degenerate,
        "se": se, "ci_95": [ci_low, ci_high], "ci_excludes_zero": ci_low > 0 or ci_high < 0,
    }


def _pairwise_concordance(gt: dict[str, float], est: dict[str, float]) -> dict:
    concordant, total = 0, 0
    detail = []
    for a, b in itertools.combinations(gt.keys(), 2):
        gt_diff = gt[a] - gt[b]
        est_diff = est[a] - est[b]
        if gt_diff == 0 or est_diff == 0:
            continue
        total += 1
        match = (gt_diff > 0) == (est_diff > 0)
        concordant += int(match)
        detail.append({"pair": [a, b], "gt_orders_higher": a if gt_diff > 0 else b, "est_orders_higher": a if est_diff > 0 else b, "match": match})
    return {"concordant": concordant, "total": total, "detail": detail}


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])
    logs_dir = pathlib.Path(cfg["paths"]["logs_dir"])

    with open(results_dir / "mini_ground_truth_selection.json", "r", encoding="utf-8") as f:
        selection = json.load(f)
    official = selection["official_set_b_estimates"]

    ground_truth = {}
    for mem_id in MEMORIES:
        pairs = _load_pairs(logs_dir / f"ground_truth_{mem_id}.jsonl")
        stats = _ground_truth_stats(pairs)
        ground_truth[mem_id] = stats
        print(f"\n=== {mem_id} ===")
        print(f"  n_pairs={stats['n_pairs']}  mean(Y_in)={stats['mean_y1']:.3f}  mean(Y_out)={stats['mean_y0']:.3f}")
        print(f"  ground truth gap = {stats['gap']:+.4f}  95% CI [{stats['ci_95'][0]:+.4f}, {stats['ci_95'][1]:+.4f}]"
              f"  {'(excludes zero)' if stats['ci_excludes_zero'] else '(includes zero -- not significant)'}")
        print(f"  measured rho={stats['rho']:.3f}" + (" (degenerate fallback to 0)" if stats["rho_is_degenerate_fallback"] else ""))
        o = official[mem_id]
        print(f"  memory_worth={o['memory_worth']['value']:.3f} (rank #{o['memory_worth']['rank']})   "
              f"ips={o['ips']['value']:.4f} (rank #{o['ips']['rank']})   "
              f"snips={o['snips']['value']:.4f} (rank #{o['snips']['rank']})   "
              f"doubly_robust={o['doubly_robust']['value']:.4f} (rank #{o['doubly_robust']['rank']})")

    gt_gaps = {m: ground_truth[m]["gap"] for m in MEMORIES}
    estimator_names = ["memory_worth", "ips", "snips", "doubly_robust"]
    concordance = {}
    print(f"\n=== Pairwise concordance with ground truth (over the {len(MEMORIES)} memories, {len(MEMORIES)*(len(MEMORIES)-1)//2} pairs) ===")
    for est_name in estimator_names:
        est_values = {m: official[m][est_name]["value"] for m in MEMORIES}
        conc = _pairwise_concordance(gt_gaps, est_values)
        concordance[est_name] = conc
        print(f"  {est_name:<16}: {conc['concordant']}/{conc['total']} pairs match ground truth's ordering")

    n_ci_exclude_zero = sum(1 for m in MEMORIES if ground_truth[m]["ci_excludes_zero"])
    best_estimator = max(estimator_names, key=lambda e: concordance[e]["concordant"])
    best_score = concordance[best_estimator]["concordant"]
    tied = [e for e in estimator_names if concordance[e]["concordant"] == best_score]

    print(f"\n=== Verdict ===")
    print(f"{n_ci_exclude_zero}/{len(MEMORIES)} memories have a 95% CI excluding zero (a real, resolved effect).")
    if len(tied) > 1:
        print(f"TIE between {tied} at {best_score}/{concordance[best_estimator]['total']} concordant pairs -- no single estimator is clearly best at this n.")
    else:
        print(f"Best pairwise match: {best_estimator} ({best_score}/{concordance[best_estimator]['total']}).")

    report = {
        "memories": MEMORIES,
        "ground_truth": ground_truth,
        "official_set_b_estimates": {m: official[m] for m in MEMORIES},
        "pairwise_concordance": concordance,
        "n_memories_ci_excludes_zero": n_ci_exclude_zero,
    }
    with open(results_dir / "mini_ground_truth_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {results_dir / 'mini_ground_truth_analysis.json'}")


if __name__ == "__main__":
    main()
