"""Step-cap sweep, zero cost: task_type_difficulty_check.py found that at
env.max_steps=30, an OPTIMAL scripted policy only solves pick_two_obj_and_place
13% of the time -- at that rate we'd mostly be measuring "did this episode
draw an impossible task", not "did this memory help". Reuses ONE expert run
per task type (task_type_difficulty_check.run_expert_difficulty_proxy,
recording steps-to-solve at a generous ceiling) and evaluates solvability at
several candidate caps from that single run, rather than re-running the
expert once per candidate cap.

Usage:
    python -m alfworld_pilot.step_cap_check
"""

from __future__ import annotations

import json
import pathlib

from . import config as config_mod
from .env_factory import load_real_alfworld_config
from .env_interface import TASK_TYPES
from .task_type_difficulty_check import check_task_type_coverage, run_expert_difficulty_proxy

CANDIDATE_CAPS = [30, 40, 50]
N_SAMPLES_PER_TYPE = 40
# A cap is "acceptable" if every task type clears this floor -- below it, a
# task type's failures are dominated by "ran out of steps" rather than by
# what the agent (or its memories) actually did.
MIN_ACCEPTABLE_SOLVABILITY_PCT = 50.0


def solvability_at_cap(steps_list: list[int], cap: int) -> float:
    if not steps_list:
        return 0.0
    return 100 * sum(1 for s in steps_list if s <= cap) / len(steps_list)


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])
    real_cfg = load_real_alfworld_config(cfg["env"]["real_alfworld_config_path"])

    coverage = check_task_type_coverage(real_cfg, split=cfg["env"].get("real_split", "train"))
    by_type_files = coverage.pop("_by_type_files")

    # our_step_cap arg only affects the (unused here) pct_solvable_within_our_step_cap field;
    # what we actually use is each game's raw steps_list, evaluated at every candidate cap below.
    difficulty = run_expert_difficulty_proxy(real_cfg, by_type_files, our_step_cap=30, n_samples=N_SAMPLES_PER_TYPE)

    table = {}
    for tt in TASK_TYPES:
        steps_list = difficulty[tt]["steps_list"]
        table[tt] = {str(cap): round(solvability_at_cap(steps_list, cap), 1) for cap in CANDIDATE_CAPS}

    print(f"=== Step-cap solvability sweep (ALFWorld's own scripted expert, n={N_SAMPLES_PER_TYPE}/type) ===\n")
    print(f"{'task_type':<32}" + "".join(f"{'cap='+str(c):>10}" for c in CANDIDATE_CAPS))
    for tt in TASK_TYPES:
        row = f"{tt:<32}"
        for cap in CANDIDATE_CAPS:
            row += f"{table[tt][str(cap)]:>9.0f}%"
        print(row)

    # Recommendation: smallest candidate cap where every task type clears the floor.
    recommended_cap = None
    for cap in CANDIDATE_CAPS:
        worst = min(table[tt][str(cap)] for tt in TASK_TYPES)
        if worst >= MIN_ACCEPTABLE_SOLVABILITY_PCT:
            recommended_cap = cap
            break

    worst_by_cap = {cap: min(table[tt][str(cap)] for tt in TASK_TYPES) for cap in CANDIDATE_CAPS}
    print(f"\nWorst-task-type solvability by cap: {worst_by_cap}")
    if recommended_cap is not None:
        worst_tt = min(TASK_TYPES, key=lambda tt: table[tt][str(recommended_cap)])
        print(
            f"\nRecommendation: env.max_steps = {recommended_cap} -- smallest candidate cap where "
            f"every task type clears {MIN_ACCEPTABLE_SOLVABILITY_PCT:.0f}% optimal-policy solvability "
            f"(worst case: {worst_tt} at {table[worst_tt][str(recommended_cap)]:.0f}%). Still a REAL "
            f"difficulty gap vs. the easiest types (~90-100% at this cap), just not a near-guaranteed "
            f"failure for one task type."
        )
    else:
        print(f"\nNo candidate cap in {CANDIDATE_CAPS} clears {MIN_ACCEPTABLE_SOLVABILITY_PCT:.0f}% for every task type.")

    report = {
        "candidate_caps": CANDIDATE_CAPS,
        "n_samples_per_type": N_SAMPLES_PER_TYPE,
        "min_acceptable_solvability_pct": MIN_ACCEPTABLE_SOLVABILITY_PCT,
        "solvability_pct_by_task_type_and_cap": table,
        "worst_task_type_solvability_by_cap": worst_by_cap,
        "recommended_cap": recommended_cap,
    }
    with open(results_dir / "step_cap_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {results_dir / 'step_cap_check.json'}")


if __name__ == "__main__":
    main()
