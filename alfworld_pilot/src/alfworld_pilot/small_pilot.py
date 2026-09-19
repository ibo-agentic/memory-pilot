"""Part C, small pilot (~$2 real spend, checkpointed): logs as many real
ALFWorld episodes as fit in this step's soft budget, using
WeightedRealTaskSource (deliberately oversamples harder task types --
see weighted_task_source.py -- so overall success doesn't sit pinned near
ceiling the way the memory sanity check's naive game-indexing did), then
runs all four Stage 1 estimators (memory_worth, ips, snips, doubly_robust)
on the resulting log.

Checkpointed via checkpointed_runner.run_logging_phase_checkpointed --
interrupting and rerunning this script resumes from the log's current
length rather than starting over. The actual stop is whichever comes
first: a nominal episode ceiling (comfortably above what $2 can buy at any
plausible per-episode rate, so it's not meant to bind), this step's own
soft budget, or the shared CostTracker's hard cap (config.yaml,
cost_control.hard_cap_usd).

No ground truth exists yet for real ALFWorld's per-memory causal values
(that needs the ground-truth phase, not run in this pilot) -- the report
below is about what the four estimators PRODUCE and how much they agree
with each other, not about which one is closer to truth.

Usage:
    python -m alfworld_pilot.small_pilot
"""

from __future__ import annotations

import json
import pathlib

from memory_ope.estimators import doubly_robust, ips, memory_worth

from . import config as config_mod
from .cache import LLMCache
from .checkpointed_runner import run_logging_phase_checkpointed
from .cost_tracker import CostCapExceeded, CostTracker
from .env_factory import load_real_alfworld_config
from .llm_client import OpenRouterClient
from .memory_store import build_mock_store
from .weighted_task_source import WeightedRealTaskSource

PILOT_SOFT_BUDGET_USD = 2.00
NOMINAL_EPISODE_CEILING = 400  # comfortably above what $2 can buy at any plausible real per-episode rate
CHUNK_SIZE = 5  # check cumulative spend after every N episodes
DR_L2_C = 1.0  # matches the root package's config/config.yaml default (estimators.dr_l2_C)


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    episodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def _spearman(a: dict[str, float], b: dict[str, float]) -> float | None:
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return None
    import numpy as np
    from scipy.stats import spearmanr

    rho, _p = spearmanr([a[m] for m in common], [b[m] for m in common])
    return float(rho) if not np.isnan(rho) else None


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

    real_cfg = load_real_alfworld_config(cfg["env"]["real_alfworld_config_path"])
    task_source = WeightedRealTaskSource(real_cfg, split=cfg["env"].get("real_split", "train"))

    log_path = logs_dir / "small_pilot.jsonl"
    spend_at_start = cost_tracker.total_spend_usd
    print(f"Starting spend (cumulative, shared cost-tracker state): ${spend_at_start:.4f}")

    stop_reason = None
    while True:
        try:
            n_new = run_logging_phase_checkpointed(
                task_source, llm_client, memories, cfg["retrieval"]["M"], cfg["retrieval"]["propensity_min"],
                cfg["retrieval"]["propensity_max"], cfg["env"]["max_steps"], NOMINAL_EPISODE_CEILING, log_path, max_new=CHUNK_SIZE,
            )
        except CostCapExceeded as e:
            stop_reason = f"HARD cost cap: {e}"
            break
        spend_so_far = cost_tracker.total_spend_usd - spend_at_start
        n_done = 0
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                n_done = sum(1 for line in f if line.strip())
        print(f"  {n_done} episodes logged, this-step spend=${spend_so_far:.4f}, cumulative=${cost_tracker.total_spend_usd:.4f}")
        if n_new == 0:
            stop_reason = f"reached nominal ceiling of {NOMINAL_EPISODE_CEILING} episodes"
            break
        if spend_so_far >= PILOT_SOFT_BUDGET_USD:
            stop_reason = f"soft budget (${PILOT_SOFT_BUDGET_USD}) reached"
            break

    episodes = _load_jsonl(log_path)
    n_episodes = len(episodes)
    this_step_spend = cost_tracker.total_spend_usd - spend_at_start
    print(f"\n=== Small pilot: {n_episodes} episodes logged ===")
    print(f"Stop reason: {stop_reason}")
    print(f"This step's spend: ${this_step_spend:.4f}")
    print(f"Cumulative spend (shared cost-tracker state): ${cost_tracker.total_spend_usd:.4f} / ${cfg['cost_control']['hard_cap_usd']} hard cap")

    if n_episodes == 0:
        print("\nNo episodes logged -- nothing to estimate.")
        return

    success_rate = sum(ep["success"] for ep in episodes) / n_episodes
    print(f"Overall success rate: {success_rate:.2f}")

    task_type_counts: dict[str, int] = {}
    for ep in episodes:
        task_type_counts[ep["task_type"]] = task_type_counts.get(ep["task_type"], 0) + 1
    print(f"Task-type distribution this run: {task_type_counts}")

    mw = memory_worth.compute(episodes)
    ips_est, snips_est = ips.compute(episodes)
    dr = doubly_robust.compute(episodes, all_mem_ids, l2_c=DR_L2_C, cross_fit=True)

    estimators = {"memory_worth": mw, "ips": ips_est, "snips": snips_est, "doubly_robust": dr}
    pairwise_spearman = {}
    names = list(estimators.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pairwise_spearman[f"{a}_vs_{b}"] = _spearman(estimators[a], estimators[b])

    print(f"\n=== Estimator agreement (pairwise Spearman across {len(all_mem_ids)} memories) ===")
    for pair, rho in pairwise_spearman.items():
        print(f"  {pair}: {rho}")

    print(f"\n=== Top 5 / bottom 5 memories by doubly_robust ===")
    dr_sorted = sorted(dr.items(), key=lambda kv: kv[1], reverse=True)
    for mem_id, val in dr_sorted[:5]:
        print(f"  TOP  {mem_id}: dr={val:.4f} mw={mw.get(mem_id):.3f} ips={ips_est.get(mem_id):.4f} snips={snips_est.get(mem_id):.4f}")
    for mem_id, val in dr_sorted[-5:]:
        print(f"  BOT  {mem_id}: dr={val:.4f} mw={mw.get(mem_id):.3f} ips={ips_est.get(mem_id):.4f} snips={snips_est.get(mem_id):.4f}")

    report = {
        "n_episodes": n_episodes,
        "stop_reason": stop_reason,
        "this_step_spend_usd": round(this_step_spend, 4),
        "cumulative_spend_usd": round(cost_tracker.total_spend_usd, 4),
        "overall_success_rate": success_rate,
        "task_type_distribution": task_type_counts,
        "estimates": estimators,
        "pairwise_spearman": pairwise_spearman,
    }
    with open(results_dir / "small_pilot.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {results_dir / 'small_pilot.json'}")


if __name__ == "__main__":
    main()
