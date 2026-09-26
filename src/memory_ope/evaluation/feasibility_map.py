"""Feasibility map: where per-memory causal valuation is affordable, and
where it isn't, under realistic real-dollar budget ceilings.

This project supports valuing memories via two genuinely different routes,
and this script maps feasibility for BOTH:

  (A) GROUND TRUTH (paired forced-in/forced-out reruns): resolves every
      memory's causal value directly and exactly, at a statistical cost
      given by power_formula.py. Depends on (n_memories, target effect
      delta, paired-outcome correlation rho) -- NOT on M or propensity
      range, since a forced rerun does not go through ordinary randomized
      retrieval at all.
  (B) OBSERVATIONAL / OFF-POLICY (SNIPS on ordinary randomized logs):
      estimates every memory's value from logs the agent would produce
      anyway, at a statistical cost that DOES depend on (n_memories, M,
      propensity range) as well as episode count -- this is the axis this
      project's own signal_boost_check.py / stage2_scale_check.py /
      scale_up_check.py already explored at a few fixed settings; this
      script generalizes that into a proper (N, M, propensity) x episode
      sweep using the SAME task_difficulty simulator and estimator code.

      Uses SNIPS, not doubly_robust, purely for compute-budget reasons
      (measured ~10x faster than DR at n=100,000 episodes -- see this
      script's git history / commit message for the timing numbers): the
      core paper's own results (dr_cross_fit_check.json, stage1_report.json)
      already show SNIPS and doubly_robust track each other closely
      (Spearman 0.92 in the real small_pilot log, near-identical accuracy in
      every Stage 1 simulation table), so this substitution does not change
      the qualitative feasibility boundary, only which exact estimator
      produced it.

Both routes' episode counts are converted to dollars using this project's
REAL measured cost per episode from the small pilot (openai/gpt-5.6-luna,
$0.20/$1.20 per M tokens) -- the same BLENDED_COST_PER_EPISODE used
throughout effect_size_analysis.py / resolution_grid.py. Route (B)'s
"cost" is conceptually softer than route (A)'s: observational logging can
piggyback on episodes the agent runs anyway for normal operation, whereas
ground-truth reruns are additional, dedicated spend on top of that -- this
script reports both on the same dollar axis for comparability, but the
distinction is real and stated explicitly in the writeup, not glossed over.

Reference points (Şimşek 2026, CMI, MemAudit, this project's own ALFWorld
pilot) are hardcoded from numbers verified by fetching each paper directly
(URLs and exact quoted numbers in REFERENCE_POINTS below) -- never
estimated where a real number could be found, and marked NOT COMPARABLE
where a paper's design doesn't map onto this map's axes.

Zero API spend -- pure simulation + arithmetic.

Usage:
    python -m memory_ope.evaluation.feasibility_map
"""

from __future__ import annotations

import copy
import json
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics, power_formula as pf
from .. import config as config_mod
from ..estimators import ips
from ..simulator import task_difficulty

# ---------------------------------------------------------------------------
# Shared real-measured constants (same as effect_size_analysis.py / resolution_grid.py)
P_REAL = 0.6611111111111111
REAL_COST_PER_EPISODE_BY_TYPE = {
    "pick_heat_then_place_in_recep": 0.0192,
    "pick_cool_then_place_in_recep": 0.0164,
    "pick_clean_then_place_in_recep": 0.0128,
    "pick_two_obj_and_place": 0.0090,
    "pick_and_place_simple": 0.0057,
    "look_at_obj_in_light": 0.0049,
}
COST_PER_EPISODE = sum(REAL_COST_PER_EPISODE_BY_TYPE.values()) / len(REAL_COST_PER_EPISODE_BY_TYPE)
RHO_REAL = 0.6355  # mean measured paired-outcome correlation, 4 ground-truthed memories
BUDGETS = [100.0, 1_000.0, 10_000.0]

# ---------------------------------------------------------------------------
# Route A (ground truth / power_formula): N grid for the MDE-vs-N curves.
N_GRID_A = np.unique(np.round(np.geomspace(5, 300, 40)).astype(int))

# ---------------------------------------------------------------------------
# Route B (observational / simulator): full sweep grid.
N_GRID_B = [10, 30, 50, 100, 200]  # keeps 70:30 generalist:specialist ratio
M_GRID_B = [5, 10, 20]
PROPENSITY_GRID_B = [(0.1, 0.9), (0.3, 0.7)]
EPISODE_GRID_B = [500, 1000, 2000, 5000, 10000, 20000, 50000]
SEED_COUNT_B = 10

# ---------------------------------------------------------------------------
# Reference points. Every number here was either (1) verified by fetching the
# source paper directly (URL given, exact quote in the comment) or (2)
# measured by this project itself. Nothing here is guessed.
REFERENCE_POINTS = {
    "this_project_alfworld": {
        "label": "This project (ALFWorld pilot)",
        "n_memories": 30,
        "M": 10,
        "propensity_range": [0.3, 0.7],
        "route": "ground_truth",
        "rho": RHO_REAL,
        "target_delta": 0.03,  # largest real gap actually found, effect_size_analysis.json
        "real_spend_usd": 16.0508,
        "spend_needed_to_resolve_usd": 968.59,
        "source": "This repo: alfworld_pilot/results/effect_size_analysis.json, results/paper_data.md §6.1",
        "note": "Real spend so far ($16.05) resolves 0 of 30 memories to the 0.03 "
        "effect ceiling with adequate power; ~$969 would be needed. Plotted at "
        "its ACTUAL spend to show it lands on the infeasible side of its own budget.",
    },
    "simsek_2026_task_difficulty": {
        "label": "Şimşek 2026, task-difficulty setup",
        "n_memories": 100,  # "70 generalist memories ... and 30 specialist memories" (arXiv 2604.12007)
        "M": None,
        "propensity_range": None,
        "route": "not_comparable",
        "real_spend_usd": 0.0,
        "source": "arXiv:2604.12007 (\"When to Forget: A Memory Governance Primitive\", "
        "Baris Simsek), https://arxiv.org/html/2604.12007 -- quote: \"70 generalist "
        "memories with U*~Uniform(0,1) and 30 specialist memories with U*=0.85\"; "
        "10,000 episodes, 20 seeds (same page).",
        "note": "PURE SIMULATION, zero real LLM calls, zero dollar cost -- the paper "
        "never leaves the synthetic DGP, so it never faces the budget question this "
        "map is about. Plotted as a reference for store size (N=100) only; its "
        "'cost' is definitionally $0 at any N, which is why it is not on route A/B's "
        "cost axis in the same sense as the other points.",
    },
    "simsek_2026_hitchhiker": {
        "label": "Şimşek 2026, co-retrieval setup",
        "n_memories": None,  # paper reports only "one anchor + one hitchhiker"; NOT a stated total pool size
        "M": None,
        "propensity_range": None,
        "route": "not_comparable",
        "real_spend_usd": 0.0,
        "source": "arXiv:2604.12007, https://arxiv.org/html/2604.12007 -- quote: \"One "
        "'anchor' memory (U*=0.90) and one 'hitchhiker' (U*=0.05) are always retrieved "
        "together except in a controlled fraction of episodes\"; independence "
        "fractions tested at 0%, 10%, 30%, 100%.",
        "note": "The paper does NOT state a total memory-pool size for this setup -- "
        "this project's own hitchhiker.py ADDS 30 filler memories as its own design "
        "choice to build a realistic top-M retrieval pool around the anchor/hitchhiker "
        "pair; that 30 is OUR addition, not a number Simsek 2026 reports. Excluded "
        "from the N-axis plot for this reason -- flagging rather than estimating.",
    },
    "cmi_2026": {
        "label": "CMI (Causal Intervention-Based Memory Selection), Causal-LoCoMo",
        "n_memories": 5.6,  # 491 memory entries / 87 examples, averaged
        "M": None,
        "propensity_range": None,
        "route": "single_shot_no_power",
        "real_spend_usd": None,
        "source": "arXiv:2605.17641 (\"Causal Intervention-Based Memory Selection for "
        "Long-Horizon LLM Agents\", Saksham Srivastava), "
        "https://arxiv.org/html/2605.17641 -- quotes: \"491 memory entries: 89 useful "
        "memories, 348 irrelevant memories, and 54 harmful memories\" across "
        "\"87 filtered evaluation examples\" (of 100 generated); \"For each candidate "
        "memory ... CMI compares model behavior under three controlled conditions\" "
        "(no-memory, with-memory, perturbed-memory); response model GPT-4.1, judge "
        "model GPT-5, retrieval via text-embedding-3-large. No API cost reported.",
        "note": "~5.6 candidate memories/example, 87 examples, ONE judged intervention "
        "per candidate memory (3 conditions, no repeated trials for statistical "
        "power). This is a fundamentally different, much cheaper design point: CMI "
        "substitutes a single LLM-as-judge causal check for the repeated real-world "
        "trials this project's power formula assumes are needed for a statistically "
        "resolvable estimate. Not placed on the (N, budget) feasibility curves for "
        "that reason -- it isn't attempting the same statistical guarantee.",
    },
    "memaudit_2026_eval_protocol": {
        "label": "MemAudit (package-oracle eval protocol for memory writing)",
        "n_memories": None,
        "M": None,
        "propensity_range": None,
        "route": "not_comparable",
        "real_spend_usd": 0.907,
        "source": "arXiv:2605.02199 (\"MemAudit: An Exact Package-Oracle Evaluation "
        "Protocol for Budgeted Long-Term LLM Memory Writing\"), "
        "https://arxiv.org/html/2605.02199 -- quote: \"The primary Natural-200 export "
        "used 681 API calls, 1,978,327 tokens, and about $0.907\"; budget measured in "
        "normalized word-equivalent units, B=2,4,8,16 (fractional 0.01-0.20) and "
        "B=30,60,100 for natural packages; models: google/gemini-3.1-flash-lite-preview "
        "(temp 0) and anthropic/claude-sonnet-4.5, via OpenRouter.",
        "note": "This paper's 'budget' (B) is a PER-MEMORY WORD LIMIT for how much a "
        "memory-writing system may write, not a memory-COUNT (N) -- a different axis "
        "entirely, not this map's N. Cited only as a real-dollar SCALE reference "
        "(~$0.91 for 200 examples of a memory-WRITING evaluation) to contrast against "
        "this project's ~$970 needed for a 30-memory causal-VALUATION resolution: "
        "writing-quality evaluation and causal-effect estimation are differently "
        "expensive tasks, not directly comparable point-for-point.",
    },
}

# Two DIFFERENT arXiv papers share the name "MemAudit" (2605.23723, poisoning/
# security auditing, and 2605.02199, the writing-budget evaluation protocol
# above) -- the latter was judged the topically relevant one for this map
# (memory EVALUATION under a budget), per the request; flagging this
# disambiguation explicitly in case the intended paper was the other one.
MEMAUDIT_AMBIGUITY_NOTE = (
    "Two 2026 arXiv papers are both titled 'MemAudit': (1) arXiv:2605.23723, "
    "'Post-hoc Auditing of Poisoned Agent Memory via Causal Attribution and "
    "Structural Anomaly Detection' (security/poisoning focus), and (2) "
    "arXiv:2605.02199, 'An Exact Package-Oracle Evaluation Protocol for "
    "Budgeted Long-Term LLM Memory Writing' (evaluation-protocol focus). "
    "This script uses (2), since it is about memory EVALUATION under a "
    "budget -- topically closest to this paper. If the intended reference "
    "was (1), its abstract reports no store-size/evaluation-scale numbers "
    "comparable to this map's axes (it reports attack-success-rate "
    "reductions, e.g. '70% to 0%', not a memory count or episode budget)."
)


def route_a_mde_curves() -> dict:
    """Ground-truth route: for each budget, the smallest resolvable effect
    (MDE) as a function of n_memories, at the real measured (p, rho,
    cost/episode)."""
    from scipy.stats import norm

    z = norm.ppf(0.975) + norm.ppf(0.80)
    sigma_d2 = pf.var_of_paired_difference(P_REAL, RHO_REAL)
    curves = {}
    for budget in BUDGETS:
        mdes = []
        for n_mem in N_GRID_A:
            n_pairs_affordable = budget / (2 * n_mem * COST_PER_EPISODE)
            if n_pairs_affordable < 2:
                mdes.append(float("nan"))
                continue
            mde = float(np.sqrt(z**2 * sigma_d2 / n_pairs_affordable))
            mdes.append(mde)
        curves[f"budget_{int(budget)}"] = {"n_memories": N_GRID_A.tolist(), "mde": mdes}
    return curves


def _snips_ci_excludes_zero(cfg: dict, n_episodes: int, seed_count: int) -> bool:
    true_vals = task_difficulty.true_values(cfg)
    spearmans = []
    for seed in range(seed_count):
        episodes = task_difficulty.generate_episodes(cfg, seed, n_episodes=n_episodes)
        _ips_est, snips_est = ips.compute(episodes)
        spearmans.append(metrics.spearman(snips_est, true_vals))
    arr = np.array(spearmans)
    p2_5 = float(np.percentile(arr, 2.5))
    return p2_5 > 0.0, float(np.mean(arr)), p2_5, float(np.percentile(arr, 97.5))


def route_b_sweep(base_cfg: dict) -> list[dict]:
    rows = []
    t_start = time.time()
    n_total = len(N_GRID_B) * len(M_GRID_B) * len(PROPENSITY_GRID_B)
    combo_i = 0
    for n_mem in N_GRID_B:
        n_gen = round(n_mem * 0.7)
        n_spec = n_mem - n_gen
        for m in M_GRID_B:
            if m > n_mem:
                continue  # M can't exceed the pool size
            for p_min, p_max in PROPENSITY_GRID_B:
                combo_i += 1
                cfg = copy.deepcopy(base_cfg)
                cfg["simulators"]["task_difficulty"]["n_generalists"] = n_gen
                cfg["simulators"]["task_difficulty"]["n_specialists"] = n_spec
                cfg["retrieval"]["M"] = m
                cfg["retrieval"]["propensity_min"] = p_min
                cfg["retrieval"]["propensity_max"] = p_max

                threshold_episodes = None
                trace = []
                for n_ep in EPISODE_GRID_B:
                    excludes_zero, mean_sp, p2_5, p97_5 = _snips_ci_excludes_zero(cfg, n_ep, SEED_COUNT_B)
                    trace.append({"n_episodes": n_ep, "mean_spearman": mean_sp, "p2_5": p2_5, "p97_5": p97_5})
                    if excludes_zero:
                        threshold_episodes = n_ep
                        break  # found the smallest tested n that resolves this combo -- stop, don't test larger (slower) n

                cost_at_threshold = threshold_episodes * COST_PER_EPISODE if threshold_episodes else None
                rows.append(
                    {
                        "n_memories": n_mem,
                        "M": m,
                        "propensity_min": p_min,
                        "propensity_max": p_max,
                        "threshold_episodes": threshold_episodes,
                        "cost_at_threshold_usd": cost_at_threshold,
                        "resolved_within_tested_range": threshold_episodes is not None,
                        "trace": trace,
                    }
                )
                elapsed = time.time() - t_start
                if cost_at_threshold is not None:
                    outcome = f"-> threshold={threshold_episodes} episodes (${cost_at_threshold:.2f})"
                else:
                    outcome = "-> NOT RESOLVED in tested range"
                print(
                    f"  [{combo_i}/{n_total}] N={n_mem:>3} M={m:>2} prop=[{p_min},{p_max}] "
                    f"{outcome}  ({elapsed:.0f}s elapsed)"
                )
    return rows


def _make_figure(route_a: dict, route_b: list[dict], results_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    # --- Panel A: ground-truth MDE(N) curves + reference points ---
    ax = axes[0]
    colors = {"budget_100": "#c96a4f", "budget_1000": "#3b6fa0", "budget_10000": "#4f9d69"}
    labels = {"budget_100": "$100", "budget_1000": "$1,000", "budget_10000": "$10,000"}
    for key, curve in route_a.items():
        ns = np.array(curve["n_memories"])
        mdes = np.array(curve["mde"])
        ax.plot(ns, mdes, label=f"budget = {labels[key]}", color=colors[key], linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of memories in the store (N)")
    ax.set_ylabel("smallest resolvable effect (MDE), 80% power")
    ax.set_title("Ground-truth route: feasibility boundary\n(real p=0.661, real ρ=0.636, real cost/episode)")

    # Our own point
    rp = REFERENCE_POINTS["this_project_alfworld"]
    ax.scatter([rp["n_memories"]], [rp["target_delta"]], marker="*", s=280, color="black", zorder=6)
    ax.annotate(
        "our point:\nN=30, Δ=0.03\n($969 needed;\n$16.05 actually spent)",
        xy=(rp["n_memories"], rp["target_delta"]), xytext=(45, 0.05), fontsize=8,
        arrowprops=dict(arrowstyle="->", linewidth=0.8),
    )
    # Simsek (N known, cost always $0 -- shown as a vertical marker, not on the cost curves)
    sim = REFERENCE_POINTS["simsek_2026_task_difficulty"]
    ax.axvline(sim["n_memories"], color="gray", linestyle=":", linewidth=1)
    ax.text(sim["n_memories"] * 1.03, 0.011, "Şimşek 2026\n(N=100, pure sim,\ncost=$0 at any Δ)", fontsize=7.5, color="dimgray")

    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)

    # --- Panel B: observational route, cost-at-threshold vs N, one line per (M, propensity) ---
    ax2 = axes[1]
    style_by_prop = {(0.1, 0.9): "--", (0.3, 0.7): "-"}
    color_by_m = {5: "#c96a4f", 10: "#3b6fa0", 20: "#4f9d69"}
    for m in M_GRID_B:
        for prop in PROPENSITY_GRID_B:
            xs, ys = [], []
            for row in route_b:
                if row["M"] == m and (row["propensity_min"], row["propensity_max"]) == prop:
                    if row["cost_at_threshold_usd"] is not None:
                        xs.append(row["n_memories"])
                        ys.append(row["cost_at_threshold_usd"])
            if xs:
                ax2.plot(
                    xs, ys, style_by_prop[prop], color=color_by_m[m], marker="o", markersize=4,
                    label=f"M={m}, prop={list(prop)}",
                )
    for budget in BUDGETS:
        ax2.axhline(budget, color="black", linewidth=0.6, linestyle=":")
        ax2.text(N_GRID_B[-1] * 1.02, budget, f"${int(budget):,}", fontsize=7.5, va="center")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(top=BUDGETS[-1] * 3)
    ax2.set_xlabel("number of memories in the store (N)")
    ax2.set_ylabel("cost for SNIPS 95% CI to exclude zero (USD)")
    ax2.set_title("Observational route: cost vs. N, M, propensity range\n(task_difficulty simulator, SNIPS)")
    ax2.legend(fontsize=6.5, ncol=2, loc="lower right")
    ax2.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(results_dir / "feasibility_map.png", dpi=140)
    plt.close(fig)


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    print("=== Route A: ground truth (power_formula) ===")
    route_a = route_a_mde_curves()

    print("=== Route B: observational (simulator + SNIPS) ===")
    route_b = route_b_sweep(cfg)

    report = {
        "assumptions": {
            "p_real": P_REAL,
            "rho_real": RHO_REAL,
            "cost_per_episode_usd": COST_PER_EPISODE,
            "budgets_usd": BUDGETS,
            "alpha_two_sided": 0.05,
            "power": 0.80,
        },
        "route_a_ground_truth_mde_curves": route_a,
        "route_b_observational_sweep": route_b,
        "reference_points": REFERENCE_POINTS,
        "memaudit_ambiguity_note": MEMAUDIT_AMBIGUITY_NOTE,
    }
    with open(results_dir / "feasibility_map.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _make_figure(route_a, route_b, results_dir)
    print(f"\nWrote {results_dir / 'feasibility_map.json'} and {results_dir / 'feasibility_map.png'}")


if __name__ == "__main__":
    main()
