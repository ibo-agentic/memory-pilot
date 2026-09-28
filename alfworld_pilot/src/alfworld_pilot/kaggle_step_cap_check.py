"""Kaggle step-cap check (2026-09-29, Task 3) -- NOT to be confused with the
existing, unrelated step_cap_check.py (a pre-Kaggle scripted-expert
solvability sweep for the PAID run's own config). This one is specific to
the Kaggle replication's LOCAL LLM agent: using ONLY existing
capability-check and Kaggle timing-probe data (no new runs), how many
successful episodes actually needed more than 20/25/30/40 steps, and what
would lowering env.max_steps from 50 cost in success rate vs. save in
wall-clock time?

Data sources, used for two different things because only one of them has
real timing:
  - results_kaggle/capability_check_all_types_zeroshot_withmem.json (60
    episodes, 6 task types x 10, zero-shot + memories -- the exact condition
    timing_probe.py measures) plus
    results_kaggle/capability_check_pick_and_place_simple_zeroshot_withmem.json's
    task_id 10-19 (10 MORE pick_and_place_simple episodes not already in the
    all-types file, which only has that type's task_id 0-9) -- 70 episodes
    total, deduplicated. Gives success + steps_taken + hit_step_cap per
    episode, but NO wall-clock timing (confirmed absent from that file's
    schema by prompt_length_check.py's own log-format check).
  - The real Kaggle Mode B throughput the user reported after running
    kaggle/timing_probe.ipynb (~136 s/episode per worker, zero-shot +
    memories, pick_and_place_simple) -- real timing, but for a DIFFERENT set
    of episodes (task_id 1000-1019 / 2000-2019, not the capability check's
    0-19) with an unknown steps_taken distribution of its own.

Because no single dataset has both real Kaggle timing AND steps_taken for
the SAME episodes, this script derives an approximate seconds/step by
dividing the real Kaggle per-episode time by the capability check's own mean
steps/episode for the identical condition (136s / 22.8 steps ~= 5.96 s/step)
-- an assumption that per-step cost is roughly uniform across task
instances (each step is one LLM call of broadly similar prompt-processing
cost), not a direct joint measurement. This is stated explicitly wherever
the resulting numbers are used; it is the best honestly-available estimate,
not a precise one.
"""

from __future__ import annotations

import json
import pathlib

ALFWORLD_PILOT_DIR = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = ALFWORLD_PILOT_DIR / "results_kaggle"

KAGGLE_MODE_B_SECONDS_PER_EPISODE = 136.0  # real, user-reported Mode B (2 workers, T4 x2) per-worker mean


def load_pooled_episodes() -> list[dict]:
    with open(RESULTS_DIR / "capability_check_all_types_zeroshot_withmem.json", "r", encoding="utf-8") as f:
        all_types = json.load(f)
    episodes = []
    for tt, r in all_types["results"].items():
        episodes.extend(r["episodes"])

    with open(RESULTS_DIR / "capability_check_pick_and_place_simple_zeroshot_withmem.json", "r", encoding="utf-8") as f:
        pas = json.load(f)
    pas_episodes = pas["results"]["pick_and_place_simple"]["episodes"]
    already_have_ids = {e["task_id"] for e in episodes if e["task_type"] == "pick_and_place_simple"}
    episodes.extend(e for e in pas_episodes if e["task_id"] not in already_have_ids)

    return episodes


def steps_distribution(episodes: list[dict]) -> dict:
    successes = [e for e in episodes if e["success"]]
    steps = sorted(e["steps_taken"] for e in successes)
    n = len(steps)

    def pct(k: int) -> int:
        idx = min(int(round(k / 100 * (n - 1))), n - 1)
        return steps[idx]

    return {
        "n_episodes_total": len(episodes),
        "n_successful": n,
        "min": steps[0] if n else None,
        "p25": pct(25) if n else None,
        "median": pct(50) if n else None,
        "p75": pct(75) if n else None,
        "p90": pct(90) if n else None,
        "max": steps[-1] if n else None,
        "all_success_steps_sorted": steps,
        "n_success_over_20": sum(1 for s in steps if s > 20),
        "n_success_over_25": sum(1 for s in steps if s > 25),
        "n_success_over_30": sum(1 for s in steps if s > 30),
        "n_success_over_40": sum(1 for s in steps if s > 40),
    }


def cap_impact(episodes: list[dict], cap: int, seconds_per_step: float) -> dict:
    n_total = len(episodes)
    n_success_orig = sum(1 for e in episodes if e["success"])

    lost_successes = [e for e in episodes if e["success"] and e["steps_taken"] > cap]
    shortened_failures = [e for e in episodes if not e["success"] and e["steps_taken"] > cap]
    unaffected = n_total - len(lost_successes) - len(shortened_failures)

    total_steps_saved = sum(e["steps_taken"] - cap for e in lost_successes + shortened_failures)
    mean_seconds_saved_per_episode = (total_steps_saved * seconds_per_step) / n_total

    n_success_new = n_success_orig - len(lost_successes)
    return {
        "cap": cap,
        "n_total": n_total,
        "n_success_original": n_success_orig,
        "success_rate_original": n_success_orig / n_total,
        "n_lost_successes": len(lost_successes),
        "lost_success_task_ids": [(e["task_type"], e["task_id"]) for e in lost_successes],
        "n_shortened_failures": len(shortened_failures),
        "n_unaffected": unaffected,
        "n_success_new": n_success_new,
        "success_rate_new": n_success_new / n_total,
        "change_in_success_rate": (n_success_new - n_success_orig) / n_total,
        "mean_seconds_saved_per_episode": mean_seconds_saved_per_episode,
    }


def main() -> None:
    episodes = load_pooled_episodes()
    print(f"Pooled {len(episodes)} zero-shot + memories episodes across capability-check files.")

    dist = steps_distribution(episodes)
    print(f"\nSteps-taken distribution among {dist['n_successful']} successful episodes "
          f"(of {dist['n_episodes_total']} total):")
    print(f"  min={dist['min']}, p25={dist['p25']}, median={dist['median']}, "
          f"p75={dist['p75']}, p90={dist['p90']}, max={dist['max']}")
    print(f"  successes needing >20 steps: {dist['n_success_over_20']}")
    print(f"  successes needing >25 steps: {dist['n_success_over_25']}")
    print(f"  successes needing >30 steps: {dist['n_success_over_30']}")
    print(f"  successes needing >40 steps: {dist['n_success_over_40']}")

    pick_and_place_eps = [e for e in episodes if e["task_type"] == "pick_and_place_simple"]
    mean_steps_pick_and_place = sum(e["steps_taken"] for e in pick_and_place_eps) / len(pick_and_place_eps)
    implied_seconds_per_step = KAGGLE_MODE_B_SECONDS_PER_EPISODE / mean_steps_pick_and_place
    print(f"\nImplied seconds/step (Kaggle Mode B {KAGGLE_MODE_B_SECONDS_PER_EPISODE}s/episode / "
          f"capability-check mean {mean_steps_pick_and_place:.1f} steps/episode, pick_and_place_simple, "
          f"same zero-shot+memories condition): {implied_seconds_per_step:.2f} s/step "
          "-- APPROXIMATE, not a direct joint measurement (see module docstring).")

    results = {"pooled_n": len(episodes), "steps_distribution": dist, "implied_seconds_per_step": implied_seconds_per_step, "caps": []}

    print("\nCap impact (pooled, all 6 task types):")
    for cap in [25, 30, 40]:
        impact = cap_impact(episodes, cap, implied_seconds_per_step)
        results["caps"].append(impact)
        print(f"  cap={cap}: success_rate {impact['success_rate_original']:.3f} -> {impact['success_rate_new']:.3f} "
              f"(change {impact['change_in_success_rate']:+.3f}, {impact['n_lost_successes']} lost successes), "
              f"mean time saved/episode = {impact['mean_seconds_saved_per_episode']:.1f}s")
        if impact["lost_success_task_ids"]:
            print(f"    lost successes: {impact['lost_success_task_ids']}")

    out_path = RESULTS_DIR / "kaggle_step_cap_check.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
