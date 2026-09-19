"""Part C: 5-episode real-ALFWorld model-selection test across 3 candidate
OpenRouter models. REAL API SPEND.

Model ids and pricing fetched directly from https://openrouter.ai/api/v1/models
on 2026-09-19 (re-check before relying on this long-term -- OpenRouter pricing
changes): explicitly the PAID `inclusionai/ling-3.0-flash-vl` variant, not the
`:free` one that also exists in the listing.

Hard-capped at $10.0 TOTAL across all 3 models combined (one shared,
disk-persisted CostTracker) -- a call that would push cumulative spend over
$10 raises CostCapExceeded before it's made, and this script stops
immediately (mid-model if necessary) and reports exactly what completed.

All 3 models get the SAME 5 task_ids (== the same 5 real ALFWorld games,
via env_factory's long-lived-env sequential .reset()) for a fair,
controlled comparison -- not 5 independent random games per model.

Usage:
    python -m alfworld_pilot.model_selection_test
"""

from __future__ import annotations

import json
import pathlib
import random

from . import config as config_mod
from .cache import LLMCache
from .cost_tracker import CostCapExceeded, CostTracker
from .env_factory import build_env_factory
from .episode_runner import run_logged_episode
from .llm_client import OpenRouterClient
from .memory_store import build_mock_store

# Fetched from OpenRouter's /api/v1/models on 2026-09-19 -- re-check before reuse.
CANDIDATE_MODELS = {
    "gpt-5.6-luna": {"model_id": "openai/gpt-5.6-luna", "pricing_input": 0.20, "pricing_output": 1.20},
    "ling-3.0-flash-vl": {"model_id": "inclusionai/ling-3.0-flash-vl", "pricing_input": 0.06, "pricing_output": 0.18},
    "gemini-3.8-flash": {"model_id": "google/gemini-3.8-flash", "pricing_input": 0.75, "pricing_output": 3.75},
}

N_EPISODES_PER_MODEL = 5
TEST_HARD_CAP_USD = 10.0


def _run_one_model(cfg: dict, model_name: str, model_info: dict, memories: list, cache: LLMCache, cost_tracker: CostTracker) -> dict:
    episodes = []
    spend_before = cost_tracker.total_spend_usd
    env = None
    try:
        llm_client = OpenRouterClient(
            model_id=model_info["model_id"],
            base_url=cfg["llm"]["base_url"],
            api_key_env_var=cfg["llm"]["api_key_env_var"],
            reasoning_enabled=cfg["llm"]["reasoning_enabled"],
            temperature=cfg["llm"]["temperature"],
            max_output_tokens=cfg["llm"]["max_output_tokens"],
            cache=cache,
            cost_tracker=cost_tracker,
        )
        env_factory = build_env_factory(cfg)
        env = env_factory()
        rng = random.Random(0)

        for task_id in range(N_EPISODES_PER_MODEL):
            ep = run_logged_episode(
                env, llm_client, memories, m=cfg["retrieval"]["M"], propensity_min=cfg["retrieval"]["propensity_min"],
                propensity_max=cfg["retrieval"]["propensity_max"], max_steps=cfg["env"]["max_steps"], task_id=task_id, rng=rng,
            )
            episodes.append(ep)
    except CostCapExceeded as e:
        return {
            "model_name": model_name, "model_id": model_info["model_id"], "stopped_early": True,
            "stop_reason": str(e), "episodes_completed": len(episodes), "episodes": episodes,
            "spend_this_model_usd": round(cost_tracker.total_spend_usd - spend_before, 4),
        }
    except Exception as e:  # ANY model-level failure (bad model_id, rejects reasoning:disabled with a
        # 400, network error, ...) must not prevent testing the OTHER candidate models, or hide what
        # DID complete (if anything) before it.
        return {
            "model_name": model_name, "model_id": model_info["model_id"], "stopped_early": False,
            "api_error": f"{type(e).__name__}: {e}", "episodes_completed": len(episodes), "episodes": episodes,
            "spend_this_model_usd": round(cost_tracker.total_spend_usd - spend_before, 4),
        }
    finally:
        if env is not None:
            env.close()

    n_calls = sum(ep["n_llm_calls"] for ep in episodes)
    total_in = sum(ep["total_input_tokens"] for ep in episodes)
    total_out = sum(ep["total_output_tokens"] for ep in episodes)
    total_parse_failures = sum(ep["parse_failures"] for ep in episodes)
    return {
        "model_name": model_name,
        "model_id": model_info["model_id"],
        "stopped_early": False,
        "episodes_completed": len(episodes),
        "success_rate": sum(ep["success"] for ep in episodes) / len(episodes),
        "n_calls": n_calls,
        "avg_input_tokens_per_call": total_in / n_calls if n_calls else 0.0,
        "avg_output_tokens_per_call": total_out / n_calls if n_calls else 0.0,
        "avg_calls_per_episode": n_calls / len(episodes),
        "total_parse_failures": total_parse_failures,
        "spend_this_model_usd": round(cost_tracker.total_spend_usd - spend_before, 4),
        "episodes": episodes,
    }


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])
    logs_dir = pathlib.Path(cfg["paths"]["logs_dir"])

    memories = build_mock_store(
        cfg["memory_store"]["n_memories"], cfg["memory_store"]["lesson_min_tokens"], cfg["memory_store"]["lesson_max_tokens"], seed=0
    )
    cache = LLMCache(cfg["cache"]["dir"])
    cost_state_path = logs_dir / "model_selection_test_cost_state.json"

    results = {}
    last_cost_tracker = None
    for model_name, model_info in CANDIDATE_MODELS.items():
        # Fresh CostTracker per model (different pricing), but pointed at the SAME
        # state_path -- cumulative total_spend_usd (dollars) correctly carries over,
        # since each past call's cost was already computed with its own model's price.
        cost_tracker = CostTracker(TEST_HARD_CAP_USD, model_info["pricing_input"], model_info["pricing_output"], state_path=cost_state_path)
        last_cost_tracker = cost_tracker
        if cost_tracker.total_spend_usd >= TEST_HARD_CAP_USD:
            print(f"Stopping before {model_name}: cumulative test spend ${cost_tracker.total_spend_usd:.4f} already at/over ${TEST_HARD_CAP_USD} cap.")
            break

        print(f"\n=== {model_name} ({model_info['model_id']}) ===")
        result = _run_one_model(cfg, model_name, model_info, memories, cache, cost_tracker)
        results[model_name] = result

        if result["stopped_early"]:
            print(f"STOPPED EARLY after {result['episodes_completed']} episodes: {result['stop_reason']}")
            print(f"Cumulative test spend so far: ${cost_tracker.total_spend_usd:.4f}")
            break

        if "api_error" in result:
            print(f"  API ERROR after {result['episodes_completed']} episodes: {result['api_error']}")
            print(f"  spend before failing:   ${result['spend_this_model_usd']:.4f}")
            continue  # try the other candidate models -- this model's own failure isn't fatal to the test

        print(f"  success_rate: {result['success_rate']:.2f}")
        print(f"  avg input tokens/call:  {result['avg_input_tokens_per_call']:.1f}")
        print(f"  avg output tokens/call: {result['avg_output_tokens_per_call']:.1f}")
        print(f"  avg calls/episode:      {result['avg_calls_per_episode']:.1f}")
        print(f"  parse failures:         {result['total_parse_failures']}")
        print(f"  spend this model:       ${result['spend_this_model_usd']:.4f}")
        print(f"  cumulative test spend:  ${cost_tracker.total_spend_usd:.4f} / ${TEST_HARD_CAP_USD} cap")

    plan = cfg["episode_plan"]
    total_plan_episodes = plan["memory_store_construction"] + plan["main_randomized_logging"] + plan["ground_truth_reruns"]
    for model_name, result in results.items():
        if result["stopped_early"] or "api_error" in result:
            continue
        model_info = CANDIDATE_MODELS[model_name]
        projected_calls = total_plan_episodes * result["avg_calls_per_episode"]
        projected_in = projected_calls * result["avg_input_tokens_per_call"]
        projected_out = projected_calls * result["avg_output_tokens_per_call"]
        result["full_plan_projection"] = {
            "total_plan_episodes": total_plan_episodes,
            "projected_calls": projected_calls,
            "projected_cost_usd": round(
                projected_in / 1e6 * model_info["pricing_input"] + projected_out / 1e6 * model_info["pricing_output"], 2
            ),
        }
        print(f"\n{model_name}: full-plan ({total_plan_episodes} episodes) projected cost = ${result['full_plan_projection']['projected_cost_usd']:.2f} (REAL tokenizer, not word-count proxy)")

    total_spend = last_cost_tracker.total_spend_usd if last_cost_tracker is not None else 0.0
    print(f"\nTOTAL test spend across all models run: ${total_spend:.4f}")

    with open(results_dir / "model_selection_test.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "total_test_spend_usd": total_spend}, f, indent=2)
    print(f"\nWrote {results_dir / 'model_selection_test.json'}")


if __name__ == "__main__":
    main()
