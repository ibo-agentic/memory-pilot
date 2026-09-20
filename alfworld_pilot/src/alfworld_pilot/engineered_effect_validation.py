"""Redesign (b), STEP 1: effect-size validation. Before committing to the
full 6-memory detection experiment, measure the REAL effect size of ONE
harmful and ONE helpful engineered memory with a small number of pairs --
if even a deliberately-engineered, plausible-but-wrong memory doesn't move
success by roughly 0.2+, the detection design won't work either, and the
rest of the budget shouldn't be spent chasing it.

Ground truth for an engineered memory MUST restrict the task source to
that memory's own applicable task type (see engineered_memory_store.py's
docstring for why) -- reuses WeightedRealTaskSource with all-but-one
weight zeroed, rather than a new task-source class.

Usage:
    python -m alfworld_pilot.engineered_effect_validation
"""

from __future__ import annotations

import json
import pathlib

from . import config as config_mod
from .cache import LLMCache
from .checkpointed_runner import run_ground_truth_phase_checkpointed
from .cost_tracker import CostCapExceeded, CostTracker
from .engineered_memory_store import build_engineered_store
from .env_factory import load_real_alfworld_config
from .env_interface import TASK_TYPES
from .llm_client import OpenRouterClient
from .mini_ground_truth_analysis import _ground_truth_stats, _load_pairs
from .weighted_task_source import WeightedRealTaskSource

N_PAIRS_VALIDATION = 20  # ~$1.50 for 2 memories (pick_heat_then_place_in_recep, ~$0.0192/episode real rate)
VALIDATION_MEMORIES = ["mem_helpful_heat", "mem_harmful_heat"]
CHUNK_SIZE = 5


def _pinned_task_source(real_cfg: dict, split: str, task_type: str) -> WeightedRealTaskSource:
    weights = {tt: 0.0 for tt in TASK_TYPES}
    weights[task_type] = 1.0
    return WeightedRealTaskSource(real_cfg, split=split, task_type_weights=weights)


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])
    logs_dir = pathlib.Path(cfg["paths"]["logs_dir"])

    memories = build_engineered_store()
    cache = LLMCache(cfg["cache"]["dir"])
    # DEDICATED state file, not the shared production tracker (logs/cost_tracker_state.json,
    # already at $14.80 cumulative -- reusing it here with hard_cap_usd=13.00 would refuse
    # every call instantly, since it's already "over" that cap). This experiment gets its
    # own fresh-from-$0 budget, per the user's explicit "$15 for this" framing.
    cost_state_path = logs_dir / "engineered_experiment_cost_state.json"
    pricing = cfg["llm"]["pricing_per_million_tokens"]
    cost_tracker = CostTracker(cfg["cost_control"]["hard_cap_usd"], pricing["input"], pricing["output"], state_path=cost_state_path)
    llm_client = OpenRouterClient(
        model_id=cfg["llm"]["model_id"], base_url=cfg["llm"]["base_url"], api_key_env_var=cfg["llm"]["api_key_env_var"],
        reasoning_enabled=cfg["llm"]["reasoning_enabled"], temperature=cfg["llm"]["temperature"],
        max_output_tokens=cfg["llm"]["max_output_tokens"], cache=cache, cost_tracker=cost_tracker,
    )
    real_cfg = load_real_alfworld_config(cfg["env"]["real_alfworld_config_path"])
    split = cfg["env"].get("real_split", "train")

    spend_at_start = cost_tracker.total_spend_usd
    print(f"Starting spend (cumulative, shared cost-tracker state): ${spend_at_start:.4f}")

    results = {}
    for mem_id in VALIDATION_MEMORIES:
        task_type = memories[[m.mem_id for m in memories].index(mem_id)].task_type
        task_source = _pinned_task_source(real_cfg, split, task_type)
        log_path = logs_dir / f"engineered_gt_{mem_id}.jsonl"
        state_path = logs_dir / f"engineered_gt_{mem_id}_state.json"
        print(f"\n=== {mem_id} (task_type={task_type}), target {N_PAIRS_VALIDATION} pairs ===")
        capped = False
        while True:
            try:
                n_new = run_ground_truth_phase_checkpointed(
                    task_source, llm_client, memories, cfg["retrieval"]["M"], cfg["retrieval"]["propensity_min"],
                    cfg["retrieval"]["propensity_max"], cfg["env"]["max_steps"], memory_id=mem_id, n_pairs=N_PAIRS_VALIDATION,
                    log_path=log_path, state_path=state_path, max_new=CHUNK_SIZE,
                )
            except CostCapExceeded as e:
                print(f"  HARD cost cap hit: {e}")
                capped = True
                break
            with open(state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            print(f"  {st['pairs_found']}/{N_PAIRS_VALIDATION} pairs, cumulative spend=${cost_tracker.total_spend_usd:.4f}")
            if n_new == 0:
                break  # target reached
        if capped:
            break
        pairs = _load_pairs(log_path)
        stats = _ground_truth_stats(pairs)
        results[mem_id] = stats
        print(f"  n_pairs={stats['n_pairs']}  mean(Y_in)={stats['mean_y1']:.3f}  mean(Y_out)={stats['mean_y0']:.3f}")
        print(f"  gap={stats['gap']:+.4f}  95% CI [{stats['ci_95'][0]:+.4f}, {stats['ci_95'][1]:+.4f}]  "
              f"{'(excludes zero)' if stats['ci_excludes_zero'] else '(includes zero)'}")
        print(f"  spend so far this run: ${cost_tracker.total_spend_usd - spend_at_start:.4f}")

    this_run_spend = cost_tracker.total_spend_usd - spend_at_start
    print(f"\n=== Validation summary ===")
    for mem_id, stats in results.items():
        flag = "LARGE ENOUGH (>=0.2)" if abs(stats["gap"]) >= 0.2 else "TOO SMALL (<0.2)"
        print(f"  {mem_id}: gap={stats['gap']:+.4f}  [{stats['ci_95'][0]:+.4f}, {stats['ci_95'][1]:+.4f}]  -- {flag}")
    print(f"\nThis run's spend: ${this_run_spend:.4f}")
    print(f"Cumulative: ${cost_tracker.total_spend_usd:.4f} / ${cfg['cost_control']['hard_cap_usd']} hard cap")

    with open(results_dir / "engineered_effect_validation.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "this_run_spend_usd": this_run_spend, "cumulative_spend_usd": cost_tracker.total_spend_usd}, f, indent=2)
    print(f"\nWrote {results_dir / 'engineered_effect_validation.json'}")


if __name__ == "__main__":
    main()
