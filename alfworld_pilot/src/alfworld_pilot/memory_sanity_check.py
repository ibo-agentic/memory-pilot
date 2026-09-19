"""Part C, memory sanity check (~30 real ALFWorld episodes, ~$0.30 real
spend, openai/gpt-5.6-luna): before spending more on a larger pilot, check
whether including memories changes anything at all.

Paired design: for each of N_PAIRS real ALFWorld games (the SAME game for
both arms of a pair, via RealAlfredEnv's gamefile_path), run two episodes
with IDENTICAL retrieval draws (candidate_ids, similarity scores, and
propensities all come from the same rng seed) but different inclusion:
  - "with_memories": normal randomized inclusion (each candidate included
    per its own propensity draw, exactly as a real logging episode)
  - "baseline": every candidate forced to included=0 -- no memories at all
Holding the task instance and candidate selection fixed between arms means
game-to-game difficulty differences cancel out of the with/without
comparison, which matters a lot at only 15 pairs.

Two stopping mechanisms, layered:
  (a) HARD: the shared CostTracker's hard_cap_usd (config.yaml,
      cost_control.hard_cap_usd) -- a call that would exceed it raises
      CostCapExceeded before being made, same mechanism as everywhere else.
  (b) SOFT: this script's own SANITY_CHECK_SOFT_BUDGET_USD, so this small
      check doesn't eat into the larger pilot's separately-budgeted spend
      even though both draw from the same underlying cost-tracker state
      file (and therefore the same real OpenRouter balance).

Usage:
    python -m alfworld_pilot.memory_sanity_check
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random

from . import config as config_mod
from .cache import LLMCache
from .cost_tracker import CostCapExceeded, CostTracker
from .env_factory import build_task_source
from .episode_runner import run_logged_episode
from .llm_client import OpenRouterClient
from .memory_store import build_mock_store

N_PAIRS = 15  # 30 episodes total
SANITY_CHECK_SOFT_BUDGET_USD = 0.50  # this step's own target -- separate from the shared hard cap


def _pair_seed(pair_index: int) -> int:
    key = f"memory_sanity|{pair_index}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) & 0xFFFFFFFF


def _run_pair(cfg, task_source, llm_client, memories, all_mem_ids, pair_index):
    seed = _pair_seed(pair_index)
    rng_with = random.Random(seed)
    rng_base = random.Random(seed)  # identical seed -> identical candidate_ids/similarity/propensities

    env_with = task_source.build_env(pair_index)
    try:
        ep_with = run_logged_episode(
            env_with, llm_client, memories, m=cfg["retrieval"]["M"], propensity_min=cfg["retrieval"]["propensity_min"],
            propensity_max=cfg["retrieval"]["propensity_max"], max_steps=cfg["env"]["max_steps"], task_id=pair_index, rng=rng_with,
        )
    finally:
        env_with.close()

    env_base = task_source.build_env(pair_index)
    try:
        ep_base = run_logged_episode(
            env_base, llm_client, memories, m=cfg["retrieval"]["M"], propensity_min=cfg["retrieval"]["propensity_min"],
            propensity_max=cfg["retrieval"]["propensity_max"], max_steps=cfg["env"]["max_steps"], task_id=pair_index, rng=rng_base,
            forced_inclusion={mid: 0 for mid in all_mem_ids},
        )
    finally:
        env_base.close()

    return ep_with, ep_base


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])
    logs_dir = pathlib.Path(cfg["paths"]["logs_dir"])

    memories = build_mock_store(
        cfg["memory_store"]["n_memories"], cfg["memory_store"]["lesson_min_tokens"], cfg["memory_store"]["lesson_max_tokens"], seed=0
    )
    all_mem_ids = [m.mem_id for m in memories]

    cache = LLMCache(cfg["cache"]["dir"])
    cost_state_path = logs_dir / "cost_tracker_state.json"
    pricing = cfg["llm"]["pricing_per_million_tokens"]
    cost_tracker = CostTracker(cfg["cost_control"]["hard_cap_usd"], pricing["input"], pricing["output"], state_path=cost_state_path)
    llm_client = OpenRouterClient(
        model_id=cfg["llm"]["model_id"], base_url=cfg["llm"]["base_url"], api_key_env_var=cfg["llm"]["api_key_env_var"],
        reasoning_enabled=cfg["llm"]["reasoning_enabled"], temperature=cfg["llm"]["temperature"],
        max_output_tokens=cfg["llm"]["max_output_tokens"], cache=cache, cost_tracker=cost_tracker,
    )
    task_source = build_task_source(cfg)

    spend_at_start = cost_tracker.total_spend_usd
    print(f"Starting spend (cumulative, shared cost-tracker state): ${spend_at_start:.4f}")

    pairs = []
    stop_reason = None
    for i in range(N_PAIRS):
        if cost_tracker.total_spend_usd - spend_at_start >= SANITY_CHECK_SOFT_BUDGET_USD:
            stop_reason = f"soft budget (${SANITY_CHECK_SOFT_BUDGET_USD}) reached after {len(pairs)} pairs"
            break
        try:
            ep_with, ep_base = _run_pair(cfg, task_source, llm_client, memories, all_mem_ids, i)
        except CostCapExceeded as e:
            stop_reason = f"HARD cost cap hit after {len(pairs)} pairs: {e}"
            break
        pairs.append((ep_with, ep_base))
        n_inc = sum(ep_with["included"].values())
        print(
            f"pair {i}: with_memories success={ep_with['success']} n_included={n_inc} "
            f"calls={ep_with['n_llm_calls']} parse_fail={ep_with['parse_failures']} | "
            f"baseline success={ep_base['success']} calls={ep_base['n_llm_calls']} parse_fail={ep_base['parse_failures']} "
            f"| cumulative spend=${cost_tracker.total_spend_usd:.4f}"
        )

    n_pairs_done = len(pairs)
    if n_pairs_done == 0:
        print(f"\nNo pairs completed ({stop_reason}). Nothing to report.")
        return

    with_eps = [w for w, _b in pairs]
    base_eps = [b for _w, b in pairs]

    success_with = sum(e["success"] for e in with_eps) / n_pairs_done
    success_base = sum(e["success"] for e in base_eps) / n_pairs_done

    # Pool both arms for the parse-failure/prompt-length correlation: baseline episodes are
    # real n_included=0 data points (not synthetic), which is exactly the low end we need.
    all_eps = with_eps + base_eps
    n_included_list = [sum(e["included"].values()) for e in all_eps]
    parse_fail_list = [e["parse_failures"] for e in all_eps]
    avg_prompt_len_list = [e["total_input_tokens"] / e["n_llm_calls"] if e["n_llm_calls"] else 0.0 for e in all_eps]

    corr_included_vs_parsefail = _pearson([float(x) for x in n_included_list], [float(x) for x in parse_fail_list])
    corr_promptlen_vs_parsefail = _pearson(avg_prompt_len_list, [float(x) for x in parse_fail_list])

    total_calls = sum(e["n_llm_calls"] for e in all_eps)
    total_in = sum(e["total_input_tokens"] for e in all_eps)
    total_out = sum(e["total_output_tokens"] for e in all_eps)
    avg_in_per_call = total_in / total_calls if total_calls else 0.0
    avg_out_per_call = total_out / total_calls if total_calls else 0.0
    avg_calls_per_episode = total_calls / len(all_eps)

    plan = cfg["episode_plan"]
    total_plan_episodes = plan["memory_store_construction"] + plan["main_randomized_logging"] + plan["ground_truth_reruns"]
    projected_calls = total_plan_episodes * avg_calls_per_episode
    projected_cost = (projected_calls * avg_in_per_call) / 1e6 * pricing["input"] + (projected_calls * avg_out_per_call) / 1e6 * pricing["output"]

    this_step_spend = cost_tracker.total_spend_usd - spend_at_start

    print(f"\n=== Memory sanity check: {n_pairs_done} pairs ({n_pairs_done * 2} episodes) ===")
    if stop_reason:
        print(f"Stopped early: {stop_reason}")
    print(f"success rate WITH memories:    {success_with:.2f} ({sum(e['success'] for e in with_eps)}/{n_pairs_done})")
    print(f"success rate BASELINE (none):  {success_base:.2f} ({sum(e['success'] for e in base_eps)}/{n_pairs_done})")
    print(f"correlation(n_included, parse_failures), n={len(all_eps)}: {corr_included_vs_parsefail}")
    print(f"correlation(avg_prompt_tokens, parse_failures), n={len(all_eps)}: {corr_promptlen_vs_parsefail}")
    print(f"avg input tokens/call:  {avg_in_per_call:.1f}")
    print(f"avg output tokens/call: {avg_out_per_call:.1f}")
    print(f"avg calls/episode:      {avg_calls_per_episode:.1f}")
    print(f"full-plan ({total_plan_episodes} episodes) projected cost @ gpt-5.6-luna real rates: ${projected_cost:.2f}")
    print(f"\nThis step's spend: ${this_step_spend:.4f}")
    print(f"Cumulative spend (shared cost-tracker state): ${cost_tracker.total_spend_usd:.4f} / ${cfg['cost_control']['hard_cap_usd']} hard cap")

    if success_with == success_base:
        print("\n*** Memories made NO difference to success rate at this sample size. ***")

    report = {
        "n_pairs": n_pairs_done,
        "stop_reason": stop_reason,
        "success_rate_with_memories": success_with,
        "success_rate_baseline": success_base,
        "correlation_n_included_vs_parse_failures": corr_included_vs_parsefail,
        "correlation_prompt_len_vs_parse_failures": corr_promptlen_vs_parsefail,
        "avg_input_tokens_per_call": avg_in_per_call,
        "avg_output_tokens_per_call": avg_out_per_call,
        "avg_calls_per_episode": avg_calls_per_episode,
        "full_plan_episodes": total_plan_episodes,
        "full_plan_projected_cost_usd": round(projected_cost, 2),
        "this_step_spend_usd": round(this_step_spend, 4),
        "cumulative_spend_usd": round(cost_tracker.total_spend_usd, 4),
        "pairs": [
            {
                "pair_index": i,
                "with_memories": {"success": w["success"], "n_included": sum(w["included"].values()), "n_calls": w["n_llm_calls"], "parse_failures": w["parse_failures"], "steps_taken": w["steps_taken"]},
                "baseline": {"success": b["success"], "n_calls": b["n_llm_calls"], "parse_failures": b["parse_failures"], "steps_taken": b["steps_taken"]},
            }
            for i, (w, b) in enumerate(pairs)
        ],
    }
    with open(results_dir / "memory_sanity_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {results_dir / 'memory_sanity_check.json'}")


if __name__ == "__main__":
    main()
