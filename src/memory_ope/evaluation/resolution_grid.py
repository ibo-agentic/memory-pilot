"""Episodes and cost needed to resolve a per-memory causal effect, across a
grid of (target effect delta, number of memories, paired-outcome
correlation rho) -- generalizes the single-point 85,463-episode/$970
calculation in effect_size_analysis.py into a reusable table/figure for the
paper, via power_formula.episodes_and_cost().

Fixed assumptions (stated explicitly, not buried):
  - p = 0.6611111111111111, the REAL overall success rate measured in the
    180-episode small pilot (alfworld_pilot/results/small_pilot.json).
  - cost_per_episode = the REAL blended per-task-type cost measured from
    that same pilot (same BLENDED_COST_PER_EPISODE as
    effect_size_analysis.py: mean of 6 real per-task-type costs, $0.20/$1.20
    per M tokens, openai/gpt-5.6-luna).
  - alpha=0.05 two-sided, power=0.80 (standard, matches every other power
    calculation in this project).
  - The (delta, rho) heatmap panel is evaluated at n_memories=30 (this
    project's actual real-ALFWorld setting); the (n_memories) line panel is
    evaluated at this project's actual real measured (delta=0.03,
    rho=0.6355) point.

Zero API spend -- pure arithmetic over the closed-form formula.

Usage:
    python -m memory_ope.evaluation.resolution_grid
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from . import power_formula as pf
from .. import config as config_mod

P_REAL = 0.6611111111111111  # overall success rate, small_pilot.json
REAL_COST_PER_EPISODE_BY_TYPE = {  # same as effect_size_analysis.py
    "pick_heat_then_place_in_recep": 0.0192,
    "pick_cool_then_place_in_recep": 0.0164,
    "pick_clean_then_place_in_recep": 0.0128,
    "pick_two_obj_and_place": 0.0090,
    "pick_and_place_simple": 0.0057,
    "look_at_obj_in_light": 0.0049,
}
COST_PER_EPISODE = sum(REAL_COST_PER_EPISODE_BY_TYPE.values()) / len(REAL_COST_PER_EPISODE_BY_TYPE)

# The real ALFWorld point this project actually measured (effect_size_analysis.json):
# largest gap found (delta), the natural 30-memory store size, and the real mean
# measured paired-outcome correlation across the 4 ground-truthed memories.
ALFWORLD_POINT = {"delta": 0.03, "n_memories": 30, "rho": 0.6355}

# Table grid (deliverable 4): full 3-D grid, for pulling any (delta,
# n_memories, rho) slice out of results/resolution_grid.json.
DELTA_GRID = [round(d, 4) for d in np.linspace(0.01, 0.20, 20)]
N_MEMORIES_GRID = [10, 20, 30, 40, 50, 75, 100]
RHO_GRID = [0.0, 0.2, 0.4, 0.6, 0.8]

# Finer grids used only for the smooth heatmap panel (not part of the table).
DELTA_FINE = np.linspace(0.01, 0.20, 60)
RHO_FINE = np.linspace(0.0, 0.8, 60)
N_MEMORIES_FINE = np.linspace(10, 100, 60)


def build_table() -> list[dict]:
    rows = []
    for delta in DELTA_GRID:
        for n_mem in N_MEMORIES_GRID:
            for rho in RHO_GRID:
                rows.append(pf.episodes_and_cost(delta, n_mem, rho, P_REAL, COST_PER_EPISODE))
    return rows


def _make_figure(results_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Left panel: cost heatmap over (delta, rho), n_memories=30 fixed ---
    ax = axes[0]
    cost_grid = np.empty((len(RHO_FINE), len(DELTA_FINE)))
    for i, rho in enumerate(RHO_FINE):
        for j, delta in enumerate(DELTA_FINE):
            cost_grid[i, j] = pf.episodes_and_cost(delta, 30, rho, P_REAL, COST_PER_EPISODE)["cost_usd"]
    im = ax.pcolormesh(
        DELTA_FINE, RHO_FINE, cost_grid, shading="auto", cmap="viridis",
        norm=matplotlib.colors.LogNorm(vmin=cost_grid.min(), vmax=cost_grid.max()),
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("cost to resolve (USD, log scale)")
    ax.set_xlabel("target effect size delta")
    ax.set_ylabel("paired-outcome correlation rho")
    ax.set_title("Cost to resolve one memory's effect\n(30-memory store, real p=0.661, real cost/episode)")
    alfworld_cost = pf.episodes_and_cost(
        ALFWORLD_POINT["delta"], ALFWORLD_POINT["n_memories"], ALFWORLD_POINT["rho"], P_REAL, COST_PER_EPISODE
    )["cost_usd"]
    ax.scatter(
        [ALFWORLD_POINT["delta"]], [ALFWORLD_POINT["rho"]],
        marker="*", s=260, color="white", edgecolor="black", linewidth=1.2, zorder=5,
    )
    ax.annotate(
        f"this project's real point\n(delta=0.03, rho=0.636)\n-> ${alfworld_cost:,.0f}",
        xy=(ALFWORLD_POINT["delta"], ALFWORLD_POINT["rho"]),
        xytext=(0.09, 0.35),
        color="white",
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color="white", linewidth=1.0),
    )

    # --- Right panel: cost vs n_memories at the real (delta, rho) point ---
    ax2 = axes[1]
    costs_by_n = [
        pf.episodes_and_cost(ALFWORLD_POINT["delta"], n, ALFWORLD_POINT["rho"], P_REAL, COST_PER_EPISODE)["cost_usd"]
        for n in N_MEMORIES_FINE
    ]
    ax2.plot(N_MEMORIES_FINE, costs_by_n, color="#3b6fa0", linewidth=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("number of memories in the store")
    ax2.set_ylabel("cost to resolve every memory (USD, log scale)")
    ax2.set_title("Cost vs. store size\n(delta=0.03, rho=0.636 -- this project's real measured point)")
    ax2.scatter([30], [alfworld_cost], marker="*", s=260, color="#d9822b", edgecolor="black", linewidth=1.0, zorder=5)
    ax2.annotate(f"N=30 -> ${alfworld_cost:,.0f}", xy=(30, alfworld_cost), xytext=(38, alfworld_cost * 1.4), fontsize=9)
    ax2.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(results_dir / "resolution_grid.png", dpi=140)
    plt.close(fig)


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    table = build_table()
    alfworld_point_full = pf.episodes_and_cost(
        ALFWORLD_POINT["delta"], ALFWORLD_POINT["n_memories"], ALFWORLD_POINT["rho"], P_REAL, COST_PER_EPISODE
    )

    report = {
        "assumptions": {
            "p_real_success_rate": P_REAL,
            "cost_per_episode_usd": COST_PER_EPISODE,
            "cost_per_episode_by_task_type": REAL_COST_PER_EPISODE_BY_TYPE,
            "alpha_two_sided": 0.05,
            "power": 0.80,
        },
        "grid_axes": {"delta": DELTA_GRID, "n_memories": N_MEMORIES_GRID, "rho": RHO_GRID},
        "alfworld_point": {**ALFWORLD_POINT, **alfworld_point_full},
        "table": table,
    }
    with open(results_dir / "resolution_grid.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _make_figure(results_dir)

    print(f"This project's real point: delta={ALFWORLD_POINT['delta']}, n_memories={ALFWORLD_POINT['n_memories']}, "
          f"rho={ALFWORLD_POINT['rho']} -> {alfworld_point_full['n_pairs_per_memory']:.1f} pairs/memory, "
          f"{alfworld_point_full['total_episodes']:.0f} episodes, ${alfworld_point_full['cost_usd']:.2f}")
    print(f"Wrote {results_dir / 'resolution_grid.json'} ({len(table)} grid rows) and {results_dir / 'resolution_grid.png'}")


if __name__ == "__main__":
    main()
