"""Measure mode: run a small number of episodes, record REAL input/output
tokens per call (from the client's actual usage, not the estimate used for
the pre-call cost-cap check), and project total cost for the full episode
plan in config.yaml's `episode_plan` section.

Usage (mock backend, no cost, exercised by the test suite):
    python -m alfworld_pilot.measure_mode
"""

from __future__ import annotations

import json
import pathlib
import random

from . import config as config_mod
from .cache import LLMCache
from .cost_tracker import CostTracker
from .env_interface import MockAlfredEnv
from .episode_runner import run_logged_episode
from .memory_store import build_mock_store


def run_measure_mode(cfg: dict, llm_client, env_factory, n_episodes: int) -> dict:
    memories = build_mock_store(
        cfg["memory_store"]["n_memories"], cfg["memory_store"]["lesson_min_tokens"], cfg["memory_store"]["lesson_max_tokens"], seed=0
    )
    rng = random.Random(0)

    per_call_input = []
    per_call_output = []
    calls_per_episode = []
    episodes = []
    for task_id in range(n_episodes):
        env = env_factory()
        ep = run_logged_episode(
            env, llm_client, memories,
            m=cfg["retrieval"]["M"], propensity_min=cfg["retrieval"]["propensity_min"], propensity_max=cfg["retrieval"]["propensity_max"],
            max_steps=cfg["env"]["max_steps"], task_id=task_id, rng=rng,
        )
        episodes.append(ep)
        for step in ep["steps"]:
            per_call_input.append(step["input_tokens"])
            per_call_output.append(step["output_tokens"])
        calls_per_episode.append(ep["n_llm_calls"])

    n_calls = len(per_call_input)
    avg_input = sum(per_call_input) / n_calls if n_calls else 0.0
    avg_output = sum(per_call_output) / n_calls if n_calls else 0.0
    avg_calls_per_episode = sum(calls_per_episode) / len(calls_per_episode) if calls_per_episode else 0.0

    plan = cfg["episode_plan"]
    total_plan_episodes = plan["memory_store_construction"] + plan["main_randomized_logging"] + plan["ground_truth_reruns"]
    total_plan_calls = total_plan_episodes * avg_calls_per_episode
    total_plan_input_tokens = total_plan_calls * avg_input
    total_plan_output_tokens = total_plan_calls * avg_output

    return {
        "n_episodes_measured": n_episodes,
        "n_calls_measured": n_calls,
        "avg_input_tokens_per_call": avg_input,
        "avg_output_tokens_per_call": avg_output,
        "avg_calls_per_episode": avg_calls_per_episode,
        "success_rate_measured": sum(ep["success"] for ep in episodes) / len(episodes) if episodes else 0.0,
        "episode_plan": plan,
        "total_plan_episodes": total_plan_episodes,
        "projected_total_calls": total_plan_calls,
        "projected_total_input_tokens": total_plan_input_tokens,
        "projected_total_output_tokens": total_plan_output_tokens,
    }


def project_cost(measured: dict, price_per_million_input: float, price_per_million_output: float) -> float:
    return (
        measured["projected_total_input_tokens"] / 1e6 * price_per_million_input
        + measured["projected_total_output_tokens"] / 1e6 * price_per_million_output
    )


def main() -> None:
    from .mock_llm import MockLLMClient

    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    llm_client = MockLLMClient(strategy="scripted_success", seed=0)
    measured = run_measure_mode(cfg, llm_client, MockAlfredEnv, cfg["cost_control"]["measure_mode_n_episodes"])

    with open(results_dir / "measure_mode_mock.json", "w", encoding="utf-8") as f:
        json.dump(measured, f, indent=2)

    print(f"Measured over {measured['n_episodes_measured']} MOCK episodes ({measured['n_calls_measured']} calls):")
    print(f"  avg input tokens/call:  {measured['avg_input_tokens_per_call']:.1f}")
    print(f"  avg output tokens/call: {measured['avg_output_tokens_per_call']:.1f}")
    print(f"  avg calls/episode:      {measured['avg_calls_per_episode']:.1f}")
    print(f"  success rate:           {measured['success_rate_measured']:.2f}")
    print(f"\nProjected for the full plan ({measured['total_plan_episodes']} episodes):")
    print(f"  total calls:  {measured['projected_total_calls']:.0f}")
    print(f"  total input tokens:  {measured['projected_total_input_tokens']:.0f}")
    print(f"  total output tokens: {measured['projected_total_output_tokens']:.0f}")
    print("\n(mock tokens are word counts, not real tokenization -- this is a pipeline smoke test, not a real cost estimate. Rerun with a real client + cheap model for real numbers, within the cost cap.)")
    print(f"\nWrote {results_dir / 'measure_mode_mock.json'}")


if __name__ == "__main__":
    main()
