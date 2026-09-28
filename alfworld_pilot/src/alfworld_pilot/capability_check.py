"""Pre-Kaggle capability check: can Qwen2.5-7B-Instruct actually solve
ALFWorld at all? Run BEFORE committing further Kaggle-replication effort --
if the base agent's success rate is near zero regardless of memories, no
memory effect could ever show up, and the whole replication is moot before
it starts. This is exactly why the paid run needed 50-70% success (its own
memory sanity check / ceiling-effect fix, see alfworld_pilot/README.md).

Two memory conditions, both supported:
  --with-memories OFF (default): EMPTY memory store (no memories retrieved
    or included at all) -- isolates BASE agent capability, decoupled from
    kaggle_memory_store.py's mixed correct/harmful/partial content, since a
    harmful memory dragging success down would conflate "the model can't do
    ALFWorld" with "some of our memories are actively wrong."
  --with-memories ON: the real kaggle_memory_store.py store + real
    embedding retrieval (same M/propensity as kaggle_config.yaml), so
    n_included varies episode to episode -- needed to check whether parse
    failures scale with memory count or prompt length (the same check the
    paid run ran on openai/gpt-5.6-luna in memory_sanity_check.py, which
    came back 0.008 / 0.17). This matters more than the raw parse rate: a
    confound here would make the configuration unusable for measuring
    memory effects regardless of how good the raw success rate looks.

Also classifies every parse-failure fallback action (react_agent.py falls
back to admissible_actions[0] when the model's output doesn't parse) as
"harmful" (an arbitrary put-object action, or closing a receptacle just
opened for a pending action) or "neutral" (everything else -- at worst
wastes a step, doesn't destroy task-relevant state) -- see
_classify_fallback for the exact, stated rule.

Zero real API spend (local model only).

Usage (from alfworld_pilot/, using the .venv-kaggle venv):
    PYTHONPATH=src .venv-kaggle/bin/python -m alfworld_pilot.capability_check \\
        --task-type pick_and_place_simple --n-episodes 20                          # zero-shot, no memories
    PYTHONPATH=src .venv-kaggle/bin/python -m alfworld_pilot.capability_check \\
        --task-type pick_and_place_simple --n-episodes 20 --few-shot              # few-shot, no memories
    PYTHONPATH=src .venv-kaggle/bin/python -m alfworld_pilot.capability_check \\
        --task-type pick_and_place_simple --n-episodes 20 --with-memories         # zero-shot, real memories
    PYTHONPATH=src .venv-kaggle/bin/python -m alfworld_pilot.capability_check \\
        --task-type pick_and_place_simple --n-episodes 20 --few-shot --with-memories
    PYTHONPATH=src .venv-kaggle/bin/python -m alfworld_pilot.capability_check \\
        --all-types --n-episodes 10 [--few-shot] [--with-memories]                # step 4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from dataclasses import asdict

import numpy as np

from . import config as config_mod
from . import react_agent
from .cache import LLMCache
from .cost_tracker import CostTracker
from .env_factory import load_real_alfworld_config
from .env_interface import TASK_TYPES
from .local_model_client import LocalTransformersClient
from .weighted_task_source import WeightedRealTaskSource


def _single_type_source(real_cfg: dict, split: str, task_type: str) -> WeightedRealTaskSource:
    """Reuses WeightedRealTaskSource with all-but-one weight zeroed -- the
    same pattern the paid run's engineered_effect_validation.py used to
    restrict ground truth to one applicable task type."""
    weights = {tt: (1.0 if tt == task_type else 0.0) for tt in TASK_TYPES}
    return WeightedRealTaskSource(real_cfg, split=split, task_type_weights=weights)


def _object_name(action: str, verb: str) -> str | None:
    """Extracts 'egg 1' from 'open egg 1' / 'close egg 1' style actions, or
    None if the action doesn't start with that verb."""
    prefix = f"{verb} "
    if not action.startswith(prefix):
        return None
    return action[len(prefix):].strip()


def _classify_fallback(step_index: int, actions: list[str]) -> str:
    """Classifies ONE fallback action (actions[step_index], known to be a
    parse-failure fallback = admissible_actions[0]) as "harmful" or
    "neutral", by an explicit, mechanical rule -- not a learned or perfect
    judgment of task relevance:

      - starts with "put " -> HARMFUL: admissible_actions[0] is not chosen
        for task relevance, so a fallback put arbitrarily places a held
        object, which is very unlikely to be the correct final placement
        and can end the chance to place it correctly later without
        re-picking it up.
      - starts with "close " AND the immediately preceding action opened
        the SAME object -> HARMFUL: undoes state (an open receptacle) that
        the very next intended action (e.g. a heat/cool/clean/put) likely
        still needed.
      - everything else (go to, open, look, examine, take, or a close that
        doesn't immediately follow opening the same object) -> NEUTRAL: at
        worst wastes a step; doesn't destroy already-made progress.
    """
    action = actions[step_index]
    if action.startswith("put "):
        return "harmful"
    closed = _object_name(action, "close")
    if closed is not None and step_index > 0:
        prev_opened = _object_name(actions[step_index - 1], "open")
        if prev_opened == closed:
            return "harmful"
    return "neutral"


def run_episodes(
    task_type: str,
    n_episodes: int,
    llm_client,
    real_cfg: dict,
    split: str,
    max_steps: int,
    few_shot_text: str | None,
    memories: list | None,
    embedder,
    m: int,
    propensity_min: float,
    propensity_max: float,
    start_seed: int = 0,
) -> list[dict]:
    from .embedding_retrieval import embedding_similarity_scores
    from memory_ope.retrieval import retrieve as ope_retrieve

    source = _single_type_source(real_cfg, split, task_type)
    episodes = []
    for i in range(n_episodes):
        task_id = start_seed + i
        env = source.build_env(task_id)
        try:
            obs, info = env.reset(task_seed=task_id)
            included_texts: list[str] = []
            n_included = 0
            if memories:
                rng = random.Random(task_id)
                sims = embedding_similarity_scores(memories, obs, embedder)
                candidate_ids, _propensities, included = ope_retrieve(sims, m, propensity_min, propensity_max, rng)
                mem_by_id = {mem.mem_id: mem for mem in memories}
                included_texts = [mem_by_id[mid].text for mid in candidate_ids if included[mid] == 1]
                n_included = len(included_texts)
            result = react_agent.run_episode(
                env, llm_client, included_texts, max_steps, obs, info, few_shot_text=few_shot_text
            )
        finally:
            env.close()
        steps = [asdict(s) for s in result.steps]
        actions = [s["action"] for s in steps]
        fallback_classes = [
            _classify_fallback(idx, actions) for idx, s in enumerate(steps) if not s["action_was_admissible"]
        ]
        avg_input_tokens = float(np.mean([s["input_tokens"] for s in steps])) if steps else 0.0
        total_oom_retries = sum(s["oom_retries"] for s in steps)
        ep = {
            "task_id": task_id,
            "task_type": task_type,
            "success": int(result.success),
            "steps_taken": len(result.steps),
            "hit_step_cap": result.hit_step_cap,
            "parse_failures": result.parse_failures,
            "n_included": n_included,
            "avg_input_tokens": avg_input_tokens,
            "total_oom_retries": total_oom_retries,
            "fallback_classes": fallback_classes,
            "failure_mode": None,
            "actions": actions,
        }
        if not ep["success"]:
            ep["failure_mode"] = _classify_failure_mode(steps)
        episodes.append(ep)
        tag = "SUCCESS" if ep["success"] else f"fail ({ep['failure_mode']})"
        print(
            f"    [{task_type}] episode {i + 1}/{n_episodes} (task_id={task_id}): {tag}, "
            f"steps={ep['steps_taken']}, parse_failures={ep['parse_failures']}, n_included={n_included}"
        )
    return episodes


def _classify_failure_mode(steps: list[dict]) -> str:
    actions = [s["action"] for s in steps]
    n = len(actions)
    if n == 0:
        return "no_steps"
    parse_fail_rate = sum(1 for s in steps if not s["action_was_admissible"]) / n
    unique_ratio = len(set(actions)) / n
    took_object = any(a.startswith("take ") for a in actions)
    attempted_terminal = any(a.startswith(("put ", "heat ", "cool ", "clean ")) for a in actions)

    if parse_fail_rate > 0.15:
        return "format_issues"
    if unique_ratio < 0.35:
        return "looping_repetition"
    if not took_object:
        return "never_found_object"
    if took_object and not attempted_terminal:
        return "picked_up_but_stuck"
    return "attempted_but_incomplete"


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    n_success = sum(e["success"] for e in episodes)
    total_steps = sum(e["steps_taken"] for e in episodes)
    total_parse_failures = sum(e["parse_failures"] for e in episodes)
    from collections import Counter

    failure_modes = Counter(e["failure_mode"] for e in episodes if not e["success"])

    n_included_arr = np.array([e["n_included"] for e in episodes], dtype=float)
    parse_failures_arr = np.array([e["parse_failures"] for e in episodes], dtype=float)
    avg_tokens_arr = np.array([e["avg_input_tokens"] for e in episodes], dtype=float)

    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    all_fallbacks = [c for e in episodes for c in e["fallback_classes"]]
    from collections import Counter as _Counter

    fallback_counts = _Counter(all_fallbacks)
    n_fallbacks = len(all_fallbacks)

    return {
        "n_episodes": n,
        "success_rate": n_success / n if n else float("nan"),
        "n_success": n_success,
        "parse_failure_rate_per_step": total_parse_failures / total_steps if total_steps else float("nan"),
        "mean_steps_taken": total_steps / n if n else float("nan"),
        "failure_mode_counts": dict(failure_modes),
        "correlation_n_included_vs_parse_failures": _safe_corr(n_included_arr, parse_failures_arr),
        "correlation_avg_prompt_tokens_vs_parse_failures": _safe_corr(avg_tokens_arr, parse_failures_arr),
        "n_included_range": [float(n_included_arr.min()), float(n_included_arr.max())] if n else None,
        "fallback_action_counts": dict(fallback_counts),
        "fallback_harmful_fraction": (fallback_counts.get("harmful", 0) / n_fallbacks) if n_fallbacks else None,
        "n_fallback_actions": n_fallbacks,
    }


def build_local_client(cfg: dict):
    llm_cfg = cfg["llm"]
    cache = LLMCache(cfg["cache"]["dir"])
    cost_tracker = CostTracker(cfg["cost_control"]["hard_cap_usd"], 0.0, 0.0)
    return LocalTransformersClient(
        model_id=llm_cfg["model_id"],
        revision=llm_cfg["revision"],
        temperature=llm_cfg["temperature"],
        max_output_tokens=llm_cfg["max_output_tokens"],
        cache=cache,
        cost_tracker=cost_tracker,
        device=llm_cfg.get("device", "cuda"),
        quantize_4bit=llm_cfg.get("quantize_4bit", True),
    ), cost_tracker


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-type", choices=TASK_TYPES, default=None)
    p.add_argument("--all-types", action="store_true")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--few-shot", action="store_true")
    p.add_argument("--with-memories", action="store_true")
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=50)
    args = p.parse_args()
    if not args.task_type and not args.all_types:
        raise SystemExit("Pass --task-type <type> or --all-types")

    cfg_path = pathlib.Path(__file__).resolve().parents[2] / "kaggle_config.yaml"
    cfg = config_mod.load_config(cfg_path)
    config_mod.ensure_dirs(cfg)
    real_cfg = load_real_alfworld_config(cfg["env"]["real_alfworld_config_path"])
    split = cfg["env"]["real_split"]

    few_shot_text = react_agent.FEW_SHOT_EXAMPLES if args.few_shot else None
    print(f"Prompting mode: {'FEW-SHOT (2 examples)' if args.few_shot else 'ZERO-SHOT'}")
    print(f"Memory condition: {'REAL kaggle_memory_store + embedding retrieval' if args.with_memories else 'EMPTY (base capability check)'}")
    print(f"Model: {cfg['llm']['model_id']}@{cfg['llm']['revision']}, max_steps={args.max_steps}")

    memories = None
    embedder = None
    if args.with_memories:
        from .embedding_retrieval import SentenceEmbedder
        from .kaggle_memory_store import build_kaggle_store

        memories = build_kaggle_store()
        embedder = SentenceEmbedder(model_name=cfg["embedding"]["model_name"], revision=cfg["embedding"].get("revision"), device="cpu")

    client, cost_tracker = build_local_client(cfg)

    task_types = TASK_TYPES if args.all_types else [args.task_type]
    all_results = {}
    for tt in task_types:
        print(f"\n=== {tt} ({args.n_episodes} episodes) ===")
        episodes = run_episodes(
            tt, args.n_episodes, client, real_cfg, split, args.max_steps, few_shot_text,
            memories, embedder, cfg["retrieval"]["M"], cfg["retrieval"]["propensity_min"], cfg["retrieval"]["propensity_max"],
            start_seed=args.start_seed,
        )
        summary = summarize(episodes)
        all_results[tt] = {"summary": summary, "episodes": episodes}
        print(f"  -> success_rate={summary['success_rate']:.2f} ({summary['n_success']}/{summary['n_episodes']}), "
              f"parse_failure_rate={summary['parse_failure_rate_per_step']:.3f}, "
              f"mean_steps={summary['mean_steps_taken']:.1f}")
        if summary["failure_mode_counts"]:
            print(f"  -> failure modes: {summary['failure_mode_counts']}")
        if args.with_memories:
            print(f"  -> corr(n_included, parse_failures)={summary['correlation_n_included_vs_parse_failures']}, "
                  f"corr(avg_prompt_tokens, parse_failures)={summary['correlation_avg_prompt_tokens_vs_parse_failures']}, "
                  f"n_included range={summary['n_included_range']}")
        if summary["n_fallback_actions"]:
            print(f"  -> fallback actions: {summary['fallback_action_counts']} "
                  f"(harmful fraction={summary['fallback_harmful_fraction']:.2f})")

    if len(task_types) > 1:
        overall_n = sum(r["summary"]["n_episodes"] for r in all_results.values())
        overall_success = sum(r["summary"]["n_success"] for r in all_results.values())
        print(f"\n=== OVERALL across {len(task_types)} task types ===")
        print(f"  success_rate={overall_success / overall_n:.2f} ({overall_success}/{overall_n})")

    out_dir = pathlib.Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "all_types" if args.all_types else args.task_type
    mode = "fewshot" if args.few_shot else "zeroshot"
    mem_tag = "withmem" if args.with_memories else "nomem"
    out_path = out_dir / f"capability_check_{tag}_{mode}_{mem_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Cost tracker (should be $0): {cost_tracker.summary()}")


if __name__ == "__main__":
    main()
