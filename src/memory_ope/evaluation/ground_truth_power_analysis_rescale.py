"""Re-run ground_truth_power_analysis.py's n_memories/pairs-per-memory
trade-off at Part C's SCALED-UP rerun budget (5000 episodes, matching the
new main_randomized_logging episode count -- see CLAUDE.md's Part C
re-plan), instead of the original 2000-episode budget. Reuses the original
script's rho estimation and MDE formula unchanged (`_empirical_rho_stats`,
`_mde`) -- only the budget and n_memories grid differ -- and writes to its
own results file so the original 2000-budget table stays intact as a
historical comparison point.

Usage:
    python -m memory_ope.evaluation.ground_truth_power_analysis_rescale
"""

from __future__ import annotations

import json
import pathlib

from .. import config as config_mod
from .ground_truth_power_analysis import Z_ALPHA_2, Z_BETA, _empirical_rho_stats, _mde

NEW_TOTAL_RERUN_BUDGET = 5000  # episodes; matches the new main_randomized_logging scale
N_MEMORIES_GRID = [10, 13, 15, 18, 20, 25, 30]


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    rho_stats = _empirical_rho_stats(cfg)
    var_y1, var_y0 = rho_stats["var_y1_mean"], rho_stats["var_y0_mean"]

    total_pairs_budget = NEW_TOTAL_RERUN_BUDGET // 2
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

    report = {
        "empirical_rho_stats_from_task_difficulty_simulator": rho_stats,
        "power_analysis_assumptions": {"alpha_two_sided": 0.05, "power": 0.80, "z_alpha_2": Z_ALPHA_2, "z_beta": Z_BETA},
        "old_budget_episodes": 2000,
        "new_budget_episodes": NEW_TOTAL_RERUN_BUDGET,
        "tradeoff_table_within_new_budget": {"total_rerun_budget_episodes": NEW_TOTAL_RERUN_BUDGET, "rows": tradeoff_table},
    }

    with open(results_dir / "ground_truth_power_analysis_rescale.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Empirical paired-design rho (unchanged from original analysis): mean={rho_stats['rho_mean']:.3f}")
    print(f"\nTrade-off within NEW {NEW_TOTAL_RERUN_BUDGET}-episode total rerun budget ({total_pairs_budget} pairs), empirical mean rho:")
    print(f"{'n_memories':>12}{'pairs/memory':>14}{'episodes/memory':>17}{'total_episodes':>16}{'MDE (80% power)':>18}")
    for row in tradeoff_table:
        print(
            f"{row['n_memories']:>12}{row['pairs_per_memory']:>14}{row['episodes_per_memory']:>17}"
            f"{row['total_episodes']:>16}{row['mde_80pct_power']:>18.4f}"
        )
    print(f"\n(for comparison, the old 2000-episode budget's best n_memories=10 row gave MDE~0.177)")
    print(f"\nWrote {results_dir / 'ground_truth_power_analysis_rescale.json'}")


if __name__ == "__main__":
    main()
