"""Wall-clock timing probe for the Kaggle GPU target (2026-09-28, extended
2026-09-29). NOT RUN ON KAGGLE by anything in this repo -- this script is
provided to run ON KAGGLE (or wherever the actual target GPU is available),
per the explicit instruction not to start any Kaggle work locally. It has
been smoke-tested locally on a single (non-T4) GPU purely to confirm the
script itself runs correctly end to end -- those local numbers are NOT
Kaggle throughput estimates (different GPU, different throughput).

Why this exists: nothing in this project's pipeline (local_model_client.py,
cache.py, cost_tracker.py, capability_check.py) records wall-clock duration
anywhere -- confirmed by grepping all of them for time/timestamp/duration/
elapsed before writing this script. So "average seconds per episode" cannot
be computed from data that already exists; it has to be measured, on the
real target hardware.

What it measures, per episode: wall-clock seconds, steps_taken, total input
tokens, total output tokens, and seconds/step -- plus a mean AND median
seconds/episode summary (median is less sensitive than mean to the rare very
long, looping episode this project's own capability check already found).

Two backends (--backend {transformers,vllm}): the default `transformers`
backend (local_model_client.LocalTransformersClient, already validated
end-to-end) and an optional `vllm` backend (vllm_client.VLLMClient, added
2026-09-29 -- see that module's docstring for the T4/AWQ compatibility
assessment; NOT locally verified, meant to be probed on Kaggle itself).

Multi-GPU (--workers N, N>1): spawns N subprocesses (this same module,
recursively, with --workers omitted so each subprocess takes the plain
single-process path), one per CUDA device (CUDA_VISIBLE_DEVICES=0..N-1),
each running a disjoint slice of episodes to its own output file, then
merges the results and reports combined throughput vs. single-worker
throughput. The slicing (split_episode_range) and merging (merge_worker_reports)
are pure functions with no GPU dependency, covered by
tests/test_timing_probe_helpers.py -- the actual multi-GPU subprocess
execution path itself could not be exercised locally (this development
environment has exactly one GPU) and must be verified for real on Kaggle's
T4 x2 session; see kaggle/timing_probe.ipynb.

Usage (on Kaggle, from alfworld_pilot/, in an environment with alfworld +
transformers (+ vllm, if using that backend) + the pinned model(s) installed):
    PYTHONPATH=src python -m alfworld_pilot.timing_probe --n-episodes 20
    PYTHONPATH=src python -m alfworld_pilot.timing_probe --n-episodes 20 --workers 2
    PYTHONPATH=src python -m alfworld_pilot.timing_probe --n-episodes 20 --backend vllm
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import statistics
import subprocess
import sys
import time

from . import config as config_mod
from . import react_agent
from .capability_check import _single_type_source, build_local_client
from .embedding_retrieval import SentenceEmbedder, embedding_similarity_scores
from .env_factory import load_real_alfworld_config
from .kaggle_memory_store import build_kaggle_store

ALFWORLD_PILOT_DIR = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = ALFWORLD_PILOT_DIR / "results_kaggle" / "timing_probe.json"


def split_episode_range(n_episodes: int, n_workers: int, start_seed: int) -> list[dict]:
    """Disjoint (start_seed, n_episodes) slices for n_workers, covering
    exactly n_episodes total task_ids with no overlap and no gap -- e.g.
    split_episode_range(20, 3, 0) -> [{'worker_id': 0, 'start_seed': 0,
    'n_episodes': 7}, {'worker_id': 1, 'start_seed': 7, 'n_episodes': 7},
    {'worker_id': 2, 'start_seed': 14, 'n_episodes': 6}] (remainder spread
    across the first workers, one extra episode each)."""
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if n_episodes < n_workers:
        raise ValueError(f"n_episodes={n_episodes} must be >= n_workers={n_workers}")
    base = n_episodes // n_workers
    remainder = n_episodes % n_workers
    slices = []
    cursor = start_seed
    for i in range(n_workers):
        count = base + (1 if i < remainder else 0)
        slices.append({"worker_id": i, "start_seed": cursor, "n_episodes": count})
        cursor += count
    return slices


def merge_worker_reports(reports: list[dict], parallel_wall_clock_seconds: float) -> dict:
    """Combines N single-worker timing reports (each with an
    "episode_timings" list, as written by run_single_process below) into one
    report, using the ORCHESTRATOR's own measured wall-clock span for the
    parallel run (not a derived max of self-reported per-worker times) --
    that's the number that actually answers "how long did the whole batch
    take running concurrently," which is what combined throughput needs.
    Pure function, no GPU/subprocess dependency -- see
    tests/test_timing_probe_helpers.py."""
    all_timings = [t for r in reports for t in r["episode_timings"]]
    if not all_timings:
        raise ValueError("No episode timings to merge")
    seconds_list = [t["seconds"] for t in all_timings]
    total_episodes = len(all_timings)

    mean_per_episode = statistics.mean(seconds_list)
    single_worker_eps_per_sec = 1.0 / mean_per_episode
    combined_eps_per_sec = total_episodes / parallel_wall_clock_seconds
    speedup = combined_eps_per_sec / single_worker_eps_per_sec

    return {
        "n_workers": len(reports),
        "total_episodes": total_episodes,
        "parallel_wall_clock_seconds": parallel_wall_clock_seconds,
        "mean_seconds_per_episode": mean_per_episode,
        "median_seconds_per_episode": statistics.median(seconds_list),
        "single_worker_episodes_per_second": single_worker_eps_per_sec,
        "combined_episodes_per_second": combined_eps_per_sec,
        "speedup_vs_single_worker": speedup,
        "combined_episodes_per_30_gpu_hours": combined_eps_per_sec * 30 * 3600,
        "episode_timings": all_timings,
    }


def summarize(episode_timings: list[dict]) -> dict:
    seconds_list = [e["seconds"] for e in episode_timings]
    mean_seconds = statistics.mean(seconds_list)
    median_seconds = statistics.median(seconds_list)
    return {
        "n_episodes": len(episode_timings),
        "mean_seconds_per_episode": mean_seconds,
        "median_seconds_per_episode": median_seconds,
        "episodes_per_30_gpu_hours_mean": (30 * 3600) / mean_seconds,
        "episodes_per_30_gpu_hours_median": (30 * 3600) / median_seconds,
    }


def build_llm_client(cfg: dict, backend: str):
    if backend == "transformers":
        return build_local_client(cfg)
    if backend == "vllm":
        from .vllm_client import build_vllm_client

        return build_vllm_client(cfg)
    raise ValueError(f"Unknown backend: {backend!r} (expected 'transformers' or 'vllm')")


def run_single_process(args: argparse.Namespace) -> dict:
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
    client, cost_tracker = build_llm_client(cfg, args.backend)
    model_load_seconds = time.monotonic() - t_load_start
    print(f"[worker {args.worker_id}] model load time (one-time, not counted per-episode): {model_load_seconds:.1f}s")

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
        total_input_tokens = sum(s.input_tokens for s in result.steps)
        total_output_tokens = sum(s.output_tokens for s in result.steps)
        episode_timings.append(
            {
                "task_id": task_id,
                "seconds": elapsed,
                "steps_taken": steps_taken,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "seconds_per_step": elapsed / steps_taken if steps_taken else None,
            }
        )
        print(
            f"[worker {args.worker_id}] episode {i + 1}/{args.n_episodes}: {elapsed:.1f}s, {steps_taken} steps, "
            f"{total_input_tokens} input tok, {total_output_tokens} output tok, {elapsed / max(steps_taken, 1):.2f}s/step"
        )

    summary = summarize(episode_timings)
    print(
        f"[worker {args.worker_id}] mean={summary['mean_seconds_per_episode']:.1f}s, "
        f"median={summary['median_seconds_per_episode']:.1f}s/episode, "
        f"episodes/30 GPU-hr (mean-based)={summary['episodes_per_30_gpu_hours_mean']:.0f}"
    )

    return {
        "args": vars(args),
        "backend": args.backend,
        "model_load_seconds": model_load_seconds,
        "episode_timings": episode_timings,
        "summary": summary,
    }


def run_multi_worker(args: argparse.Namespace) -> None:
    slices = split_episode_range(args.n_episodes, args.workers, args.start_seed)
    out_paths = []
    procs = []
    t_start = time.monotonic()
    for sl in slices:
        out_path = ALFWORLD_PILOT_DIR / "results_kaggle" / f"timing_probe_worker{sl['worker_id']}.json"
        out_paths.append(out_path)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(sl["worker_id"])
        cmd = [
            sys.executable, "-m", "alfworld_pilot.timing_probe",
            "--task-type", args.task_type,
            "--n-episodes", str(sl["n_episodes"]),
            "--start-seed", str(sl["start_seed"]),
            "--backend", args.backend,
            "--worker-id", str(sl["worker_id"]),
            "--out-path", str(out_path),
        ]
        print(f"Spawning worker {sl['worker_id']} on CUDA_VISIBLE_DEVICES={sl['worker_id']}: "
              f"{sl['n_episodes']} episodes from start_seed={sl['start_seed']}")
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(pathlib.Path(__file__).resolve().parents[2])))

    for proc, sl in zip(procs, slices):
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"timing_probe worker {sl['worker_id']} exited with code {ret}")
    parallel_wall_clock = time.monotonic() - t_start

    reports = []
    for out_path in out_paths:
        with open(out_path, "r", encoding="utf-8") as f:
            reports.append(json.load(f))

    merged = merge_worker_reports(reports, parallel_wall_clock)
    print(f"\n=== Combined {args.workers}-worker throughput ===")
    print(f"Parallel wall-clock: {parallel_wall_clock:.1f}s for {merged['total_episodes']} episodes")
    print(f"Mean seconds/episode (per worker): {merged['mean_seconds_per_episode']:.1f}, "
          f"median: {merged['median_seconds_per_episode']:.1f}")
    print(f"Single-worker episodes/sec: {merged['single_worker_episodes_per_second']:.4f}")
    print(f"Combined episodes/sec: {merged['combined_episodes_per_second']:.4f}")
    print(f"Speedup vs. single worker: {merged['speedup_vs_single_worker']:.2f}x (ideal for {args.workers} workers: {args.workers:.2f}x)")
    print(f"Combined episodes per 30 GPU-hours: {merged['combined_episodes_per_30_gpu_hours']:.0f}")

    out_path = ALFWORLD_PILOT_DIR / "results_kaggle" / "timing_probe_combined.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"\nWrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-type", default="pick_and_place_simple")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--start-seed", type=int, default=1000)
    p.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    p.add_argument("--workers", type=int, default=1, help="If >1, spawns this many single-GPU subprocesses on disjoint episode slices and merges results.")
    p.add_argument("--worker-id", type=int, default=0, help=argparse.SUPPRESS)  # set by run_multi_worker when spawning subprocesses
    p.add_argument("--out-path", type=str, default=None, help=argparse.SUPPRESS)  # set by run_multi_worker when spawning subprocesses
    args = p.parse_args()

    if args.workers > 1:
        run_multi_worker(args)
        return

    report = run_single_process(args)
    out_path = pathlib.Path(args.out_path) if args.out_path else DEFAULT_OUT_PATH
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
