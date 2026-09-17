"""Runs one episode end to end: randomized memory retrieval (exactly
Stage 1's mechanism, via retrieval_shared) -> ReAct agent loop -> a JSONL-
ready episode log with the same core schema as Stage 1's logs (task_id,
candidate_ids, propensities, included, success) plus ALFWorld-specific
fields (task_type, steps_taken, hit_step_cap, parse_failures, token/cost
totals) for the confounding check the task-type logging is meant to enable.
"""

from __future__ import annotations

import random
from dataclasses import asdict

from . import react_agent
from .llm_client import LLMClient
from .memory_store import Memory, similarity_scores
from .retrieval_shared import retrieve


def run_logged_episode(
    env,
    llm_client: LLMClient,
    memories: list[Memory],
    m: int,
    propensity_min: float,
    propensity_max: float,
    max_steps: int,
    task_id: int,
    rng: random.Random,
    forced_inclusion: dict[str, int] | None = None,
) -> dict:
    """forced_inclusion, if given, overrides the drawn `included` dict for
    the listed memory ids (used by the forced-in/forced-out ground-truth
    mode) while everything else (candidacy, propensities, other memories'
    natural draws) is untouched."""
    _obs, info = env.reset(task_seed=task_id)
    task_type = info["task_type"]

    sims = similarity_scores(memories, task_type, rng)
    candidate_ids, propensities, included = retrieve(sims, m, propensity_min, propensity_max, rng)

    if forced_inclusion:
        included = dict(included)
        included.update({k: v for k, v in forced_inclusion.items() if k in included})

    mem_by_id = {mem.mem_id: mem for mem in memories}
    included_texts = [mem_by_id[mid].text for mid in candidate_ids if included[mid] == 1]

    result = react_agent.run_episode(env, llm_client, included_texts, max_steps)

    total_input_tokens = sum(s.input_tokens for s in result.steps)
    total_output_tokens = sum(s.output_tokens for s in result.steps)

    return {
        "task_id": task_id,
        "task_type": task_type,
        "candidate_ids": candidate_ids,
        "propensities": propensities,
        "included": included,
        "success": int(result.success),
        "steps_taken": len(result.steps),
        "hit_step_cap": result.hit_step_cap,
        "parse_failures": result.parse_failures,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "n_llm_calls": len(result.steps),
        "n_cache_hits": sum(1 for s in result.steps if s.cached),
        "steps": [asdict(s) for s in result.steps],
    }
