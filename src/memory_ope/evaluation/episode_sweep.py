"""Check #3: Spearman correlation (mean + 95% seed interval, over 20 seeds)
against the causal oracle, as a function of the number of logged episodes,
for all four estimators on both simulators. Reports where IPS/DR overtake
Memory Worth.

Usage:
    python -m memory_ope.evaluation.episode_sweep
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
from ..evaluation import metrics
from ..simulator import hitchhiker, task_difficulty

EPISODE_COUNTS = [500, 1000, 2000, 5000, 10000, 50000]
ESTIMATOR_NAMES = ["memory_worth", "ips", "snips", "doubly_robust"]


def _generate(name: str, cfg: dict, seed: int, n: int) -> list[dict]:
    if name == "task_difficulty":
        return task_difficulty.generate_episodes(cfg, seed, n_episodes=n)
    return hitchhiker.generate_episodes(cfg, seed, n_episodes=n)


def _true_values(name: str, cfg: dict) -> dict[str, float]:
    if name == "task_difficulty":
        return task_difficulty.true_values(cfg)
    return hitchhiker.true_values(cfg)


def _all_ids(name: str, cfg: dict) -> list[str]:
    if name == "task_difficulty":
        return task_difficulty.all_ids(cfg)
    return hitchhiker.all_ids(cfg)


def _run_simulator(name: str, cfg: dict) -> dict:
    true_vals = _true_values(name, cfg)
    all_ids = _all_ids(name, cfg)
    dr_l2_c = cfg["estimators"]["dr_l2_C"]
    seed_count = cfg["seed_count"]

    corr_by_n = {est: {n: [] for n in EPISODE_COUNTS} for est in ESTIMATOR_NAMES}

    for n in EPISODE_COUNTS:
        for seed in range(seed_count):
            episodes = _generate(name, cfg, seed, n)
            mw = memory_worth.compute(episodes)
            ips_est, snips_est = ips.compute(episodes)
            dr_est = doubly_robust.compute(episodes, all_ids, l2_c=dr_l2_c, cross_fit=True)
            estimates = {"memory_worth": mw, "ips": ips_est, "snips": snips_est, "doubly_robust": dr_est}
            for est_name in ESTIMATOR_NAMES:
                corr_by_n[est_name][n].append(metrics.spearman(estimates[est_name], true_vals))

    summary = {}
    for est_name in ESTIMATOR_NAMES:
        means, lo, hi = [], [], []
        for n in EPISODE_COUNTS:
            vals = np.array(corr_by_n[est_name][n])
            means.append(float(np.mean(vals)))
            lo.append(float(np.percentile(vals, 2.5)))
            hi.append(float(np.percentile(vals, 97.5)))
        summary[est_name] = {"mean": means, "p2_5": lo, "p97_5": hi, "per_seed": {n: corr_by_n[est_name][n] for n in EPISODE_COUNTS}}

    return {"episode_counts": EPISODE_COUNTS, "estimators": summary}


def _find_overtake_points(report: dict) -> dict:
    """First episode count at which mean IPS / DR spearman exceeds mean Memory Worth spearman."""
    out = {}
    for sim_name, r in report.items():
        mw_means = r["estimators"]["memory_worth"]["mean"]
        out[sim_name] = {}
        for est_name in ("ips", "snips", "doubly_robust"):
            est_means = r["estimators"][est_name]["mean"]
            overtake_n = None
            for n, mw_m, est_m in zip(r["episode_counts"], mw_means, est_means):
                if est_m > mw_m:
                    overtake_n = n
                    break
            out[sim_name][est_name] = overtake_n
    return out


def _make_plot(report: dict, results_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, sim_name in zip(axes, ("task_difficulty", "hitchhiker")):
        r = report[sim_name]
        for est_name in ESTIMATOR_NAMES:
            s = r["estimators"][est_name]
            ax.plot(r["episode_counts"], s["mean"], marker="o", label=est_name)
            ax.fill_between(r["episode_counts"], s["p2_5"], s["p97_5"], alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel("episodes")
        ax.set_ylabel("spearman vs causal oracle")
        ax.set_title(sim_name)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "episode_sweep.png", dpi=130)
    plt.close(fig)


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    report = {"task_difficulty": _run_simulator("task_difficulty", cfg), "hitchhiker": _run_simulator("hitchhiker", cfg)}
    overtake = _find_overtake_points(report)

    with open(results_dir / "episode_sweep.json", "w", encoding="utf-8") as f:
        json.dump({"report": report, "overtake_points": overtake}, f, indent=2)

    for sim_name, r in report.items():
        print(f"\n=== {sim_name} ===")
        print(f"{'n_episodes':>10}" + "".join(f"{e:>16}" for e in ESTIMATOR_NAMES))
        for i, n in enumerate(r["episode_counts"]):
            row = f"{n:>10}"
            for est_name in ESTIMATOR_NAMES:
                row += f"{r['estimators'][est_name]['mean'][i]:>16.3f}"
            print(row)
        print(f"  overtake points (first n where mean > memory_worth mean): {overtake[sim_name]}")

    _make_plot(report, results_dir)
    print(f"\nWrote {results_dir / 'episode_sweep.json'} and episode_sweep.png")


if __name__ == "__main__":
    main()
