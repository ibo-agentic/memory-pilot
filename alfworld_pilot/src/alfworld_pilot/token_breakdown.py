"""Breaks a single ReAct call's prompt down into its component pieces
(system prompt, retrieved memories, step history, admissible actions) so
the token-cost assumption isn't just one opaque "~2000 tokens" number --
see react_agent._build_prompt for the exact sections. Word counts here
(not real tokenizer counts) are a cheap proxy consistent with the rest of
Part B's mock measurements; real token counts come from measure_mode once
a real client is used.

Usage:
    python -m alfworld_pilot.token_breakdown
"""

from __future__ import annotations

import json
import pathlib
import random

from . import config as config_mod
from .env_interface import MockAlfredEnv
from .memory_store import build_mock_store, similarity_scores
from .react_agent import SYSTEM_PROMPT, StepRecord, _build_prompt
from .retrieval_shared import retrieve


def _word_count(text: str) -> int:
    return len(text.split())


def breakdown_for_step(memories, memory_texts: list[str], history: list[StepRecord], goal_obs: str, admissible_actions: list[str]) -> dict:
    messages = _build_prompt(goal_obs, memory_texts, history, admissible_actions)
    system_tokens = _word_count(SYSTEM_PROMPT)
    memory_tokens = sum(_word_count(t) for t in memory_texts)
    history_tokens = sum(_word_count(s.observation) + _word_count(s.action) for s in history)
    admissible_tokens = _word_count(str(admissible_actions))
    total_from_sections = system_tokens + memory_tokens + history_tokens + admissible_tokens
    total_actual = sum(_word_count(m["content"]) for m in messages)
    return {
        "system_prompt_tokens": system_tokens,
        "memories_tokens": memory_tokens,
        "n_memories_included": len(memory_texts),
        "history_tokens": history_tokens,
        "n_history_steps": len(history),
        "admissible_actions_tokens": admissible_tokens,
        "sum_of_sections": total_from_sections,
        "actual_prompt_tokens": total_actual,  # includes goal/labels/formatting not itemized above
    }


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    memories = build_mock_store(cfg["memory_store"]["n_memories"], cfg["memory_store"]["lesson_min_tokens"], cfg["memory_store"]["lesson_max_tokens"], seed=0)
    rng = random.Random(0)
    env = MockAlfredEnv()
    goal_obs, info = env.reset(task_seed=0)
    task_type = info["task_type"]
    admissible = info["admissible_commands"][0]

    sims = similarity_scores(memories, task_type, rng)
    candidate_ids, _props, included = retrieve(sims, cfg["retrieval"]["M"], cfg["retrieval"]["propensity_min"], cfg["retrieval"]["propensity_max"], rng)
    mem_by_id = {mm.mem_id: mm for mm in memories}
    memory_texts = [mem_by_id[mid].text for mid in candidate_ids if included[mid] == 1]

    breakdowns = {}
    fake_history: list[StepRecord] = []
    for step_idx in [0, 2, 4, 6]:
        while len(fake_history) < step_idx:
            fake_history.append(StepRecord(
                observation=f"You did something at step {len(fake_history)}.", admissible_actions=admissible,
                thought="thinking", action=admissible[0], action_was_admissible=True,
                input_tokens=0, output_tokens=0, cached=False,
            ))
        breakdowns[f"step_{step_idx}"] = breakdown_for_step(memories, memory_texts, fake_history, goal_obs, admissible)

    report = {
        "note": "word counts (proxy for tokens), not a real tokenizer -- consistent with measure_mode's mock numbers",
        "n_memories_in_store": len(memories),
        "lesson_token_range_config": [cfg["memory_store"]["lesson_min_tokens"], cfg["memory_store"]["lesson_max_tokens"]],
        "steps": breakdowns,
    }
    with open(results_dir / "token_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"{'section':<28}" + "".join(f"step_{s:<10}" for s in [0, 2, 4, 6]))
    for section in ("system_prompt_tokens", "memories_tokens", "history_tokens", "admissible_actions_tokens", "actual_prompt_tokens"):
        row = f"{section:<28}"
        for s in [0, 2, 4, 6]:
            row += f"{breakdowns[f'step_{s}'][section]:<15}"
        print(row)
    print(f"\nWrote {results_dir / 'token_breakdown.json'}")


if __name__ == "__main__":
    main()
