"""20-episode wall-clock timing probe for the Kaggle GPU target (2026-09-28,
Task C). NOT RUN YET -- per the explicit instruction not to start any Kaggle
work this round, this script is provided to run ON KAGGLE (or wherever the
actual target GPU is available), not executed locally as part of this check.

Why this exists: nothing in this project's pipeline (local_model_client.py,
cache.py, cost_tracker.py, capability_check.py) records wall-clock duration
anywhere -- confirmed by grepping all of them for time/timestamp/duration/
elapsed before writing this script. The existing capability-check logs
(results_kaggle/capability_check_*.json) have zero timing fields, on any
GPU, so "average seconds per episode" cannot be computed from data that
already exists; it has to be measured. The local 8GB card these episodes
actually ran on also isn't a stand-in for Kaggle's T4/P100 target even if it
HAD been timed -- different GPU, different throughput -- so this needs to
run on the real target hardware to be trustworthy.

What it measures: wall-clock seconds per episode (zero-shot + memories, same
config as capability_check.py --with-memories), split out from one-time
model-load time (so a slow first load doesn't inflate the per-episode
figure), for a fresh set of task_ids (start_seed defaults to 1000, past the
capability check's own 0-19/0-9 ranges, so this doesn't just replay
already-cached responses and report near-zero fake latency).

Usage (on Kaggle, from alfworld_pilot/, in an environment with alfworld +
transformers + the pinned Qwen2.5-7B-Instruct + bitsandbytes installed):
    PYTHONPATH=src python -m alfworld_pilot.timing_probe --n-episodes 20
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import time

from . import config as config_mod
from . import react_agent
from .capability_check import _single_type_source, build_local_client
from .embedding_retrieval import SentenceEmbedder, embedding_similarity_scores
from .env_factory import load_real_alfworld_config
from .kaggle_memory_store import build_kaggle_store

ALFWORLD_PILOT_DIR = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-type", default="pick_and_place_simple")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--start-seed", type=int, default=1000)
    args = p.parse_args()

    cfg_path = ALFWORLD_PILOT_DIR / "kaggle_config.yaml"
    cfg = config_mod.load_config(cfg_path)
    real_cfg = load_real_alfworld_config(cfg["env"]["real_alfworld_config_path"])
    split = cfg["env"]["real_split"]
    source = _single_type_source(real_cfg, split, args.task_type)

    memories = build_kaggle_store()
    embedder = SentenceEmbedder(
        model_name=cfg["embedding"]["model_name"], revision=cfg["embedding"].get("revision"), device="cpu"
    )

    t_load_start = time.monotonic()
    client, cost_tracker = build_local_client(cfg)
    model_load_seconds = time.monotonic() - t_load_start
    print(f"Model load time (one-time, not counted per-episode): {model_load_seconds:.1f}s")

    from memory_ope.retrieval import retrieve as ope_retrieve

    m = cfg["retrieval"]["M"]
    prop_min = cfg["retrieval"]["propensity_min"]
    prop_max = cfg["retrieval"]["propensity_max"]
    max_steps = cfg["env"]["max_steps"]

    episode_timings = []
    for i in range(args.n_episodes):
        task_id = args.start_seed + i
        env = source.build_env(task_id)
        try:
            t0 = time.monotonic()
            obs, info = env.reset(task_seed=task_id)
            rng = random.Random(task_id)
            sims = embedding_similarity_scores(memories, obs, embedder)
            candidate_ids, _props, included = ope_retrieve(sims, m, prop_min, prop_max, rng)
            mem_by_id = {mem.mem_id: mem for mem in memories}
            included_texts = [mem_by_id[mid].text for mid in candidate_ids if included[mid] == 1]
            result = react_agent.run_episode(env, client, included_texts, max_steps, obs, info)
            elapsed = time.monotonic() - t0
        finally:
            env.close()
        steps_taken = len(result.steps)
        episode_timings.append(
            {
                "task_id": task_id,
                "seconds": elapsed,
                "steps_taken": steps_taken,
                "seconds_per_step": elapsed / steps_taken if steps_taken else None,
            }
        )
        print(f"  episode {i + 1}/{args.n_episodes}: {elapsed:.1f}s, {steps_taken} steps, {elapsed / max(steps_taken, 1):.2f}s/step")

    seconds_list = [e["seconds"] for e in episode_timings]
    mean_seconds = sum(seconds_list) / len(seconds_list)
    episodes_per_30h = (30 * 3600) / mean_seconds

    print(f"\nMean seconds/episode: {mean_seconds:.1f}")
    print(f"Episodes that fit in 30 GPU-hours (one week's quota): {episodes_per_30h:.0f}")
    print(f"Episodes that fit in 60 GPU-hours (two weeks' quota): {2 * episodes_per_30h:.0f}")

    out = {
        "args": vars(args),
        "model_load_seconds": model_load_seconds,
        "episode_timings": episode_timings,
        "mean_seconds_per_episode": mean_seconds,
        "episodes_per_30_gpu_hours": episodes_per_30h,
    }
    out_path = ALFWORLD_PILOT_DIR / "results_kaggle" / "timing_probe.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
