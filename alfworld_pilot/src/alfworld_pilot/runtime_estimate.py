"""Wall-clock time estimate for the full Part C plan (10030 episodes: 50
memory-store construction + 5000 main logging + 4980 ground truth).

Two components, measured/assumed separately since they have very different
character:
  1. LOCAL overhead (env construction, .reset()/.step(), retrieval) --
     measured directly with the mock LLM (so it isolates local compute from
     network latency), real ALFWorld env, at env.max_steps=50.
  2. LLM call latency -- NOT yet measured (no real LLM calls made before
     Part C's model-selection test); a stated, clearly-labeled assumption
     here, to be replaced with the real measured value once Part C runs.

Usage:
    python -m alfworld_pilot.runtime_estimate [--llm-latency-s 1.5]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from . import config as config_mod

# Measured 2026-09-19 (real ALFWorld, mock LLM, env.max_steps=50; see CLAUDE.md):
#   - one-time full-dataset walk (long-lived env construction): ~3.1s
#   - long-lived env, sequential .reset(), 5 real episodes, full 50-step cap
#     each (mock LLM never wins against real admissible-command ordering):
#     1.18s-6.98s/episode, mean 3.67s
#   - single-game (gamefile_path) construct+reset (NOT stepped): mean 0.52s
# The chunked/resumable design (checkpointed_runner.py) builds a FRESH env
# per episode rather than reusing one long-lived env -- its local-overhead
# tax is the single-game construct cost on top of the same stepping cost:
MEASURED_LONG_LIVED_SEC_PER_EPISODE = 3.67
MEASURED_SINGLE_GAME_CONSTRUCT_SEC = 0.52
CHUNKED_LOCAL_SEC_PER_EPISODE = MEASURED_LONG_LIVED_SEC_PER_EPISODE + MEASURED_SINGLE_GAME_CONSTRUCT_SEC

CALLS_PER_EPISODE_WORST_CASE = 50  # every episode runs to the full step cap -- an upper bound;
# a real agent that actually succeeds sometimes will finish (and stop calling the LLM) early,
# so true average calls/episode will be <= this once Part C's model is chosen and used for real


def estimate(n_episodes: int, llm_latency_s: float, calls_per_episode: float = CALLS_PER_EPISODE_WORST_CASE) -> dict:
    local_sec = n_episodes * CHUNKED_LOCAL_SEC_PER_EPISODE
    llm_sec = n_episodes * calls_per_episode * llm_latency_s
    total_sec = local_sec + llm_sec
    return {
        "n_episodes": n_episodes,
        "calls_per_episode_assumed": calls_per_episode,
        "llm_latency_s_assumed": llm_latency_s,
        "local_overhead_hours": round(local_sec / 3600, 2),
        "llm_call_hours_serial": round(llm_sec / 3600, 2),
        "total_hours_serial": round(total_sec / 3600, 2),
        "total_days_serial": round(total_sec / 86400, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm-latency-s", type=float, default=1.5, help="ASSUMED per-call LLM latency, seconds (pre-Part-C placeholder)")
    args = p.parse_args()

    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])
    plan = cfg["episode_plan"]
    total_episodes = plan["memory_store_construction"] + plan["main_randomized_logging"] + plan["ground_truth_reruns"]

    result = estimate(total_episodes, args.llm_latency_s)
    result["note"] = (
        "llm_latency_s_assumed is a STATED ASSUMPTION, not a measurement -- no real LLM calls have "
        "been made yet. Replace with Part C's real measured per-call latency once available. "
        "calls_per_episode_assumed=50 is a worst-case upper bound (every episode runs the full step "
        "cap); a real agent that sometimes succeeds early will use fewer calls on average."
    )

    print(f"Full plan: {total_episodes} episodes (serial execution, chunked/resumable design)")
    print(f"  local overhead:  {result['local_overhead_hours']:.1f} hours")
    print(f"  LLM call time:   {result['llm_call_hours_serial']:.1f} hours (at {args.llm_latency_s}s/call assumed, {CALLS_PER_EPISODE_WORST_CASE} calls/episode worst case)")
    print(f"  TOTAL:           {result['total_hours_serial']:.1f} hours ({result['total_days_serial']:.1f} days) if run serially, one call at a time")
    print("\nLLM latency dominates total wall time by roughly two orders of magnitude over local "
          "overhead -- at this scale, running purely serially would take multiple days. Parallelizing "
          "across several worker processes (each with its own env + a shared cost-tracker/cache) is "
          "worth considering for the real run once a model is chosen; not built in this session.")

    with open(results_dir / "runtime_estimate.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {results_dir / 'runtime_estimate.json'}")


if __name__ == "__main__":
    main()
