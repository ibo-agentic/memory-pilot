"""Stage 2 (ALFWorld) budget estimate. Pure arithmetic on stated assumptions
-- makes NO API calls. Every number below is a stated assumption, not a
measurement; re-check them once real ALFWorld + a real agent prompt exist,
since actual token counts depend on prompt engineering not yet done.

Pricing snapshot (OpenRouter, fetched 2026-09-17, itself subject to change --
re-check at Stage 2 time): DeepSeek V4.1 Flash $0.15/M in, $0.60/M out;
inclusionAI Ling 3.0 Flash VL $0.06/M in, $0.18/M out; Inception Mercury 2.5
$0.04/M in, $0.15/M out.

Usage:
    python -m memory_ope.evaluation.stage2_budget_estimate
"""

from __future__ import annotations

import json
import pathlib

from .. import config as config_mod

# --- Per-episode LLM-call assumptions -------------------------------------
ASSUMPTIONS = {
    "max_steps_cap": 50,  # standard ALFWorld episode step cap
    "avg_steps_if_success": 12,  # typical successful ReAct trajectory length in published ALFWorld runs
    "avg_steps_if_failure": 50,  # failed episodes usually run to the cap before giving up
    "assumed_success_rate": 0.5,  # rough guess for a cheap, not fine-tuned model; adjust once measured
    "avg_prompt_tokens_per_call_scenarios": {
        "low": 1000,
        "base": 2000,
        "high": 4000,
    },
    "avg_completion_tokens_per_call": 60,  # a ReAct Thought+Action is typically short
}

# --- Episode-count plan for the pilot -------------------------------------
EPISODE_PLAN = {
    "memory_store_construction": 50,  # cold-start episodes (no retrieval) to seed the ~50-trajectory store
    "main_randomized_logging": 1000,  # main OPE logging phase; Stage 1's small-data check (n=250..2000)
    # showed IPS/SNIPS/DR are already near-unbiased at n=250 but Spearman
    # is too noisy to be conclusive below ~1000-2000 -- 1000 is a floor,
    # 2000 is safer if budget allows.
    "ground_truth_reruns": {
        "n_memories": 10,
        "n_arms": 2,  # forced-in, forced-out
        "n_base_tasks": 6,
        "n_seeds_per_task": 4,
    },
}

PRICING_PER_MILLION_TOKENS = {
    "deepseek-v4.1-flash": {"input": 0.15, "output": 0.60},
    "ling-3.0-flash-vl": {"input": 0.06, "output": 0.18},
    "mercury-2.5": {"input": 0.04, "output": 0.15},
}


def _avg_calls_per_episode() -> float:
    a = ASSUMPTIONS
    return a["assumed_success_rate"] * a["avg_steps_if_success"] + (1 - a["assumed_success_rate"]) * a["avg_steps_if_failure"]


def _total_episodes() -> int:
    gt = EPISODE_PLAN["ground_truth_reruns"]
    gt_episodes = gt["n_memories"] * gt["n_arms"] * gt["n_base_tasks"] * gt["n_seeds_per_task"]
    return EPISODE_PLAN["memory_store_construction"] + EPISODE_PLAN["main_randomized_logging"] + gt_episodes


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    avg_calls = _avg_calls_per_episode()
    total_episodes = _total_episodes()
    total_calls = total_episodes * avg_calls

    cost_table = {}
    for scenario, prompt_tokens in ASSUMPTIONS["avg_prompt_tokens_per_call_scenarios"].items():
        input_tokens_m = total_calls * prompt_tokens / 1e6
        output_tokens_m = total_calls * ASSUMPTIONS["avg_completion_tokens_per_call"] / 1e6
        cost_table[scenario] = {
            "avg_prompt_tokens_per_call": prompt_tokens,
            "total_input_tokens_millions": round(input_tokens_m, 2),
            "total_output_tokens_millions": round(output_tokens_m, 2),
            "cost_by_model_usd": {
                model: round(input_tokens_m * price["input"] + output_tokens_m * price["output"], 2)
                for model, price in PRICING_PER_MILLION_TOKENS.items()
            },
        }

    report = {
        "assumptions": ASSUMPTIONS,
        "episode_plan": EPISODE_PLAN,
        "total_episodes": total_episodes,
        "avg_llm_calls_per_episode": round(avg_calls, 1),
        "total_llm_calls_estimate": round(total_calls),
        "pricing_per_million_tokens_usd": PRICING_PER_MILLION_TOKENS,
        "cost_table": cost_table,
    }

    with open(results_dir / "stage2_budget_estimate.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Total episodes planned: {total_episodes}")
    print(f"  memory_store_construction: {EPISODE_PLAN['memory_store_construction']}")
    print(f"  main_randomized_logging:   {EPISODE_PLAN['main_randomized_logging']}")
    gt = EPISODE_PLAN["ground_truth_reruns"]
    gt_n = gt["n_memories"] * gt["n_arms"] * gt["n_base_tasks"] * gt["n_seeds_per_task"]
    print(f"  ground_truth_reruns:       {gt_n}  (= {gt['n_memories']} memories x {gt['n_arms']} arms x {gt['n_base_tasks']} tasks x {gt['n_seeds_per_task']} seeds)")
    print(f"\nAvg LLM calls/episode: {avg_calls:.1f} (= {ASSUMPTIONS['assumed_success_rate']}*{ASSUMPTIONS['avg_steps_if_success']} + {1-ASSUMPTIONS['assumed_success_rate']}*{ASSUMPTIONS['avg_steps_if_failure']})")
    print(f"Total LLM calls estimate: ~{total_calls:,.0f}")

    print(f"\n{'scenario':<8}{'prompt tok/call':>16}" + "".join(f"{m:>22}" for m in PRICING_PER_MILLION_TOKENS))
    for scenario, row in cost_table.items():
        line = f"{scenario:<8}{row['avg_prompt_tokens_per_call']:>16}"
        for model in PRICING_PER_MILLION_TOKENS:
            line += f"{'$' + str(row['cost_by_model_usd'][model]):>22}"
        print(line)

    print(f"\nWrote {results_dir / 'stage2_budget_estimate.json'}")


if __name__ == "__main__":
    main()
