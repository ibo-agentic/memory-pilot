"""CLI: run all four estimators over the logged episodes for both simulators,
compare against the oracle true values, and write results/stage1_report.json
plus summary plots.

Usage:
    python -m memory_ope.evaluation.run_eval
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import config as config_mod
from ..estimators import doubly_robust, ips, memory_worth
from ..logging_utils import read_episodes
from ..simulator import hitchhiker, task_difficulty
from . import metrics

ESTIMATOR_NAMES = ["memory_worth", "ips", "snips", "doubly_robust"]


def _estimate_all(episodes: list[dict], all_memory_ids: list[str], dr_l2_c: float) -> dict[str, dict[str, float]]:
    mw = memory_worth.compute(episodes)
    ips_est, snips_est = ips.compute(episodes)
    dr_est = doubly_robust.compute(episodes, all_memory_ids, l2_c=dr_l2_c, cross_fit=True)
    return {"memory_worth": mw, "ips": ips_est, "snips": snips_est, "doubly_robust": dr_est}


def _run_simulator(name: str, cfg: dict, logs_dir: pathlib.Path) -> dict:
    if name == "task_difficulty":
        true_vals = task_difficulty.true_values(cfg)
        all_ids = task_difficulty.all_ids(cfg)
    else:
        true_vals = hitchhiker.true_values(cfg)
        all_ids = hitchhiker.all_ids(cfg)

    dr_l2_c = cfg["estimators"]["dr_l2_C"]
    seed_count = cfg["seed_count"]

    per_estimator_seed_estimates: dict[str, list[dict[str, float]]] = {e: [] for e in ESTIMATOR_NAMES}
    per_estimator_seed_corr: dict[str, list[float]] = {e: [] for e in ESTIMATOR_NAMES}

    for seed in range(seed_count):
        log_path = logs_dir / f"{name}_seed{seed}.jsonl"
        if not log_path.exists():
            raise FileNotFoundError(
                f"Missing log {log_path}. Run `python -m memory_ope.simulator.run_simulation` first."
            )
        episodes = list(read_episodes(log_path))
        estimates = _estimate_all(episodes, all_ids, dr_l2_c)
        for est_name in ESTIMATOR_NAMES:
            per_estimator_seed_estimates[est_name].append(estimates[est_name])
            per_estimator_seed_corr[est_name].append(metrics.spearman(estimates[est_name], true_vals))

    result = {"true_values": true_vals, "estimators": {}}
    for est_name in ESTIMATOR_NAMES:
        bias, variance = metrics.bias_and_variance(per_estimator_seed_estimates[est_name], true_vals)
        result["estimators"][est_name] = {
            "correlation_per_seed": per_estimator_seed_corr[est_name],
            "correlation_summary": metrics.summarize(
                {i: v for i, v in enumerate(per_estimator_seed_corr[est_name])}
            ),
            "bias": bias,
            "bias_summary": metrics.summarize(bias),
            "variance": variance,
            "variance_summary": metrics.summarize(variance),
        }
    return result


def _run_hitchhiker_sweep(cfg: dict) -> dict:
    """Sweep independent_retrieval_fraction and report SCALE-FREE metrics for
    the anchor-vs-hitchhiker confound (raw "gap" is misleading across
    estimators that live on different scales -- e.g. Memory Worth's ~0.5 vs
    IPS's ~0.04 -- since a bigger gap in absolute terms isn't necessarily a
    more accurate one):

    (a) fraction of seeds where the estimator ranks anchor above hitchhiker
        (should be 1.0 if the estimator reliably separates them at all)
    (b) pool-wide Spearman correlation with the causal oracle (same metric
        used in the main per-simulator report, but tracked per fraction)

    The raw estimated values and gap are still saved for reference, but are
    no longer the primary comparison.
    """
    sc = cfg["simulators"]["hitchhiker"]
    fracs = sc["independent_retrieval_fraction_sweep"]
    dr_l2_c = cfg["estimators"]["dr_l2_C"]
    all_ids = hitchhiker.all_ids(cfg)
    seed_count = cfg["seed_count"]

    sweep = {est_name: {"anchor": [], "hitchhiker": []} for est_name in ESTIMATOR_NAMES}
    fraction_anchor_above_hitchhiker = {est_name: [] for est_name in ESTIMATOR_NAMES}
    pool_spearman_mean = {est_name: [] for est_name in ESTIMATOR_NAMES}
    true_anchor_per_frac = []
    true_hitchhiker_per_frac = []

    for frac in fracs:
        true_vals = hitchhiker.true_values(cfg, independent_fraction=frac)
        true_anchor_per_frac.append(true_vals["anchor"])
        true_hitchhiker_per_frac.append(true_vals["hitchhiker"])

        per_est_estimates = {est_name: [] for est_name in ESTIMATOR_NAMES}
        per_est_corr = {est_name: [] for est_name in ESTIMATOR_NAMES}
        per_est_anchor_above = {est_name: [] for est_name in ESTIMATOR_NAMES}

        for seed in range(seed_count):
            episodes = hitchhiker.generate_episodes(cfg, seed, independent_fraction=frac)
            estimates = _estimate_all(episodes, all_ids, dr_l2_c)
            for est_name in ESTIMATOR_NAMES:
                per_est_estimates[est_name].append(estimates[est_name])
                per_est_corr[est_name].append(metrics.spearman(estimates[est_name], true_vals))
                e = estimates[est_name]
                if "anchor" in e and "hitchhiker" in e:
                    per_est_anchor_above[est_name].append(1.0 if e["anchor"] > e["hitchhiker"] else 0.0)

        for est_name in ESTIMATOR_NAMES:
            anchor_vals = [e["anchor"] for e in per_est_estimates[est_name] if "anchor" in e]
            hh_vals = [e["hitchhiker"] for e in per_est_estimates[est_name] if "hitchhiker" in e]
            sweep[est_name]["anchor"].append(float(np.mean(anchor_vals)) if anchor_vals else float("nan"))
            sweep[est_name]["hitchhiker"].append(float(np.mean(hh_vals)) if hh_vals else float("nan"))
            fraction_anchor_above_hitchhiker[est_name].append(float(np.mean(per_est_anchor_above[est_name])))
            pool_spearman_mean[est_name].append(float(np.mean(per_est_corr[est_name])))

    return {
        "fractions": fracs,
        "fraction_anchor_above_hitchhiker": fraction_anchor_above_hitchhiker,
        "pool_spearman_mean": pool_spearman_mean,
        # Reference only -- scale-dependent, not the primary comparison (see docstring).
        "sweep": sweep,
        "true_anchor_per_frac": true_anchor_per_frac,
        "true_hitchhiker_per_frac": true_hitchhiker_per_frac,
        "raw_utility_anchor": sc["anchor_utility"],
        "raw_utility_hitchhiker": sc["hitchhiker_utility"],
    }


def _print_summary(report: dict) -> None:
    for sim_name in ("task_difficulty", "hitchhiker"):
        print(f"\n=== {sim_name} ===")
        print(f"{'estimator':<16}{'spearman mean':>15}{'spearman std':>15}{'bias mean':>12}{'variance mean':>15}")
        for est_name in ESTIMATOR_NAMES:
            e = report[sim_name]["estimators"][est_name]
            print(
                f"{est_name:<16}"
                f"{e['correlation_summary']['mean']:>15.3f}"
                f"{e['correlation_summary']['std']:>15.3f}"
                f"{e['bias_summary']['mean']:>12.3f}"
                f"{e['variance_summary']['mean']:>15.5f}"
            )

    print("\n=== hitchhiker recoverability sweep: scale-free metrics ===")
    sweep = report["hitchhiker_sweep"]
    print(f"{'estimator':<16}" + "".join(f"frac={f:<10}" for f in sweep["fractions"]))
    print("fraction of seeds where anchor is ranked above hitchhiker:")
    for est_name in ESTIMATOR_NAMES:
        row = f"{est_name:<16}"
        for v in sweep["fraction_anchor_above_hitchhiker"][est_name]:
            row += f"{v:<15.2f}"
        print(row)
    print("\npool-wide spearman vs causal oracle (mean over seeds):")
    for est_name in ESTIMATOR_NAMES:
        row = f"{est_name:<16}"
        for v in sweep["pool_spearman_mean"][est_name]:
            row += f"{v:<15.3f}"
        print(row)

    print("\n(reference only, scale-dependent -- see docstring) mean estimate and gap:")
    for est_name in ESTIMATOR_NAMES:
        print(f"\n{est_name}:")
        for frac, a, h, ta, th in zip(
            sweep["fractions"],
            sweep["sweep"][est_name]["anchor"],
            sweep["sweep"][est_name]["hitchhiker"],
            sweep["true_anchor_per_frac"],
            sweep["true_hitchhiker_per_frac"],
        ):
            print(
                f"  independent_fraction={frac:<5} anchor_est={a:.3f}  hitchhiker_est={h:.3f}  "
                f"gap={a-h:.3f}  true_gap={ta-th:.3f}"
            )


def _make_plots(report: dict, results_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, sim_name in zip(axes, ("task_difficulty", "hitchhiker")):
        biases = [list(report[sim_name]["estimators"][e]["bias"].values()) for e in ESTIMATOR_NAMES]
        ax.boxplot(biases, tick_labels=ESTIMATOR_NAMES)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"Per-memory bias — {sim_name}")
        ax.set_ylabel("estimate - true value")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(results_dir / "bias_by_estimator.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sweep = report["hitchhiker_sweep"]
    for est_name in ESTIMATOR_NAMES:
        axes[0].plot(sweep["fractions"], sweep["fraction_anchor_above_hitchhiker"][est_name], marker="o", label=est_name)
    axes[0].axhline(1.0, color="gray", linewidth=0.6, linestyle=":")
    axes[0].axhline(0.5, color="gray", linewidth=0.6, linestyle=":")
    axes[0].set_xlabel("independent retrieval fraction")
    axes[0].set_ylabel("fraction of seeds: anchor ranked above hitchhiker")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Scale-free metric (a)")
    axes[0].legend(fontsize=8)

    for est_name in ESTIMATOR_NAMES:
        axes[1].plot(sweep["fractions"], sweep["pool_spearman_mean"][est_name], marker="o", label=est_name)
    axes[1].set_xlabel("independent retrieval fraction")
    axes[1].set_ylabel("pool-wide spearman vs causal oracle")
    axes[1].set_title("Scale-free metric (b)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "hitchhiker_recoverability.png", dpi=130)
    plt.close(fig)


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    logs_dir = pathlib.Path(cfg["paths"]["logs_dir"])
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    report = {
        "task_difficulty": _run_simulator("task_difficulty", cfg, logs_dir),
        "hitchhiker": _run_simulator("hitchhiker", cfg, logs_dir),
        "hitchhiker_sweep": _run_hitchhiker_sweep(cfg),
    }

    with open(results_dir / "stage1_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_summary(report)
    _make_plots(report, results_dir)
    print(f"\nWrote {results_dir / 'stage1_report.json'} and plots to {results_dir}")


if __name__ == "__main__":
    main()
