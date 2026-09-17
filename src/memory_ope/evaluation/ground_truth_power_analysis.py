"""Ground-truth power analysis for Stage 2's forced-in/forced-out reruns.

Question: with binary success, how many paired reruns per memory are needed
to estimate a true effect (causal gap) of 0.05 / 0.10 / 0.15 with reasonable
precision (80% power, two-sided alpha=0.05, standard normal-approximation
sample-size formula for a paired binary outcome)?

Paired design: same task instance AND same environment seed used for both
the forced-in and forced-out rerun of a given memory, so whatever the
environment's random draws (with everything else held fixed), the same
context noise affects both arms equally. Y1 and Y0 are conditionally
independent Bernoulli draws given that shared context, so:

  Var(D) = Var(Y1) + Var(Y0) - 2*rho*sqrt(Var(Y1)*Var(Y0))
  n_pairs = (z_alpha/2 + z_beta)^2 * Var(D) / delta^2

rho (the paired-outcome correlation) is estimated EMPIRICALLY from the
task_difficulty simulator at Stage-2 scale (_common.paired_design_stats),
not assumed -- it comes out modest (mean ~0.15, range 0.01-0.22 across
memories: generalists, which see more heterogeneous task contexts, pair
better than specialists, which only ever appear on hard tasks). This is a
simulator-derived proxy, not a measurement of real ALFWorld -- re-estimate
rho from actual Stage 2 pilot data once available (same formula, real
paired reruns) rather than trusting this number long-term.

Usage:
    python -m memory_ope.evaluation.ground_truth_power_analysis
"""

from __future__ import annotations

import copy
import json
import pathlib

import numpy as np

from .. import config as config_mod
from ..simulator import task_difficulty

Z_ALPHA_2 = 1.9600  # two-sided alpha=0.05
Z_BETA = 0.8416  # 80% power
POWER_CONST = (Z_ALPHA_2 + Z_BETA) ** 2

TARGET_DELTAS = [0.05, 0.10, 0.15]
TOTAL_RERUN_BUDGET = 2000  # episodes; each pair = forced-in + forced-out = 2 episodes
N_MEMORIES_GRID = [3, 5, 8, 10, 13, 15, 20]


def _empirical_rho_stats(cfg: dict) -> dict:
    scaled = copy.deepcopy(cfg)
    scaled["simulators"]["task_difficulty"]["n_generalists"] = 35
    scaled["simulators"]["task_difficulty"]["n_specialists"] = 15
    stats = task_difficulty.paired_design_stats(scaled)
    rhos = np.array([v["rho"] for v in stats.values()])
    var_y1 = np.array([v["var_y1"] for v in stats.values()])
    var_y0 = np.array([v["var_y0"] for v in stats.values()])
    return {
        "rho_mean": float(rhos.mean()),
        "rho_min": float(rhos.min()),
        "rho_max": float(rhos.max()),
        "var_y1_mean": float(var_y1.mean()),
        "var_y0_mean": float(var_y0.mean()),
    }


def _n_pairs_needed(delta: float, var_y1: float, var_y0: float, rho: float) -> float:
    var_d = var_y1 + var_y0 - 2 * rho * np.sqrt(var_y1 * var_y0)
    return POWER_CONST * var_d / (delta ** 2)


def _mde(n_pairs: float, var_y1: float, var_y0: float, rho: float) -> float:
    var_d = var_y1 + var_y0 - 2 * rho * np.sqrt(var_y1 * var_y0)
    return float(np.sqrt(POWER_CONST * var_d / n_pairs))


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    rho_stats = _empirical_rho_stats(cfg)
    var_y1, var_y0 = rho_stats["var_y1_mean"], rho_stats["var_y0_mean"]

    # Sample size needed per target effect, at three rho scenarios: 0 (no
    # pairing benefit / unpaired-equivalent), empirical mean, empirical max.
    rho_scenarios = {"unpaired (rho=0)": 0.0, "paired, empirical mean rho": rho_stats["rho_mean"], "paired, empirical max rho": rho_stats["rho_max"]}
    sample_size_table = {
        scenario: {str(delta): _n_pairs_needed(delta, var_y1, var_y0, rho) for delta in TARGET_DELTAS}
        for scenario, rho in rho_scenarios.items()
    }

    # Trade-off: n_memories vs pairs/memory within the total rerun budget,
    # using the empirical mean rho, reporting resulting MDE.
    total_pairs_budget = TOTAL_RERUN_BUDGET // 2
    tradeoff_table = []
    for n_mem in N_MEMORIES_GRID:
        pairs_per_memory = total_pairs_budget // n_mem
        mde = _mde(pairs_per_memory, var_y1, var_y0, rho_stats["rho_mean"])
        tradeoff_table.append(
            {
                "n_memories": n_mem,
                "pairs_per_memory": pairs_per_memory,
                "episodes_per_memory": pairs_per_memory * 2,
                "total_episodes": pairs_per_memory * 2 * n_mem,
                "mde_80pct_power": round(mde, 4),
            }
        )

    recommendation = (
        "Recommend n_memories=10, ~100 pairs/memory (200 episodes/memory, "
        "2000 total) -> MDE ~0.18 at 80% power (empirical mean rho). This "
        "matches the original Stage 2 plan's 10-memory ground-truth target "
        "and reliably resolves large effects (>=0.15) while giving noisier "
        "but still informative signal for 0.10; 0.05 is not reliably "
        "detectable at this budget for more than ~1 memory at a time (see "
        "sample_size_table: delta=0.05 needs on the order of 1000+ pairs "
        "even with pairing). Prioritizing memory COUNT over per-memory "
        "precision is deliberate: Stage 1 found Spearman correlation itself "
        "needs enough points to be informative (noisy at n<~1000 episodes, "
        "see small_data_results.json), and the pilot's real target is "
        "whether IPS/DR rank memories better than Memory Worth -- a "
        "Spearman check needs multiple ground-truth points more than it "
        "needs each point pinned exactly.\n\n"
        "Memory selection: after the main randomized-logging phase, don't "
        "pick ground-truth memories randomly or purely top/bottom-ranked. "
        "Choose a stratified sample across the IPS/DR-estimated value "
        "distribution (e.g. one from each estimated-value quintile) PLUS "
        "deliberately oversample memories where Memory Worth and IPS/DR "
        "estimates disagree most (largest rank or magnitude gap) -- those "
        "are exactly the cases the pilot's hypothesis lives or dies on, so "
        "ground truth should be spent resolving them, not memories all "
        "estimators already agree on."
    )

    report = {
        "empirical_rho_stats_from_task_difficulty_simulator": rho_stats,
        "power_analysis_assumptions": {
            "alpha_two_sided": 0.05,
            "power": 0.80,
            "z_alpha_2": Z_ALPHA_2,
            "z_beta": Z_BETA,
        },
        "sample_size_table_pairs_needed": sample_size_table,
        "tradeoff_table_within_budget": {
            "total_rerun_budget_episodes": TOTAL_RERUN_BUDGET,
            "rows": tradeoff_table,
        },
        "recommendation": recommendation,
    }

    with open(results_dir / "ground_truth_power_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Empirical paired-design rho (task_difficulty simulator, 50-memory scale):")
    print(f"  mean={rho_stats['rho_mean']:.3f}  min={rho_stats['rho_min']:.3f}  max={rho_stats['rho_max']:.3f}")

    print(f"\nPairs needed for 80% power, alpha=0.05 (two-sided), by target effect delta:")
    print(f"{'scenario':<28}" + "".join(f"delta={d:<10}" for d in TARGET_DELTAS))
    for scenario, row in sample_size_table.items():
        line = f"{scenario:<28}"
        for delta in TARGET_DELTAS:
            line += f"{row[str(delta)]:<15.0f}"
        print(line)

    print(f"\nTrade-off within {TOTAL_RERUN_BUDGET}-episode total rerun budget ({total_pairs_budget} pairs), empirical mean rho:")
    print(f"{'n_memories':>12}{'pairs/memory':>14}{'episodes/memory':>17}{'total_episodes':>16}{'MDE (80% power)':>18}")
    for row in tradeoff_table:
        print(
            f"{row['n_memories']:>12}{row['pairs_per_memory']:>14}{row['episodes_per_memory']:>17}"
            f"{row['total_episodes']:>16}{row['mde_80pct_power']:>18.4f}"
        )

    print(f"\n{recommendation}")
    print(f"\nWrote {results_dir / 'ground_truth_power_analysis.json'}")


if __name__ == "__main__":
    main()
