"""Resumable job-list runner (2026-09-29, extended 2026-09-30 for 2-GPU
support): reads a job list written by job_list.py, skips whatever's already
in the results log, and executes the rest up to a time budget or max_new
cap -- flushing after every single job (not just at session end) so a crash
loses at most one in-flight job, the same safety margin
checkpointed_runner.py already gives the paid run's own logging/ground-truth
phases.

Deliberately reuses episode_runner.run_logged_episode (logging jobs) and
fold_over_ground_truth.run_fold_over_arm (ground-truth fold jobs) unchanged
-- this module is purely the "which jobs are left, and how much time is
left" orchestration layer.

Has its own CLI (`python -m alfworld_pilot.run_jobs ...`) so
multi_worker_phase0.py can spawn ONE process per GPU exactly the way
timing_probe.py's --workers already does (CUDA_VISIBLE_DEVICES set per
subprocess) -- each invocation is handed its OWN job-list file (already
filtered to that worker's assigned subset) and its OWN results file, and
knows nothing about its sibling worker.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
import time

from . import config as config_mod
from .capability_check import build_local_client
from .env_factory import load_real_alfworld_config
from .episode_runner import SimilarityFn, run_logged_episode
from .fold_over_ground_truth import run_fold_over_arm
from .ground_truth_runner import TaskSource
from .job_list import TARGET_MEMORY_IDS, load_job_list
from .kaggle_memory_store import build_kaggle_store
from .weighted_task_source import WeightedRealTaskSource

ALFWORLD_PILOT_DIR = pathlib.Path(__file__).resolve().parents[2]

DEFAULT_READY_WAIT_TIMEOUT_SECONDS = 1200.0  # 20 min -- matches multi_worker_phase0's watchdog
READY_POLL_INTERVAL_SECONDS = 2.0


def _print_startup_diagnostics(worker_id: int | None) -> None:
    """Prints CUDA_VISIBLE_DEVICES, `nvidia-smi -L`, and free RAM, all
    explicitly flushed -- a real Kaggle hang showed NO per-worker output at
    all (Python's default block buffering when stdout isn't a TTY, i.e. when
    piped/redirected by a subprocess parent, only flushes on a full buffer or
    process exit -- neither happened during a 2+ hour hang). print(...,
    flush=True) here, plus -u on the subprocess's own python invocation
    (multi_worker_phase0.py), together make sure this always shows up
    immediately in the notebook instead of vanishing into an unflushed
    buffer during exactly the kind of hang this is meant to help diagnose."""
    print(f"[worker {worker_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}", flush=True)
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        print(f"[worker {worker_id}] nvidia-smi -L:\n{result.stdout.strip()}", flush=True)
    except Exception as e:  # noqa: BLE001 -- diagnostics must never block a real run
        print(f"[worker {worker_id}] nvidia-smi -L failed: {e!r}", flush=True)
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    print(f"[worker {worker_id}] {line.strip()}", flush=True)
                    break
    except Exception as e:  # noqa: BLE001 -- e.g. not on Linux; diagnostics must never block a real run
        print(f"[worker {worker_id}] could not read /proc/meminfo: {e!r}", flush=True)


def ready_marker_path(ready_dir: str | pathlib.Path, worker_id: int) -> pathlib.Path:
    return pathlib.Path(ready_dir) / f"worker{worker_id}.ready"


def write_ready_marker(ready_dir: str | pathlib.Path, worker_id: int) -> None:
    path = ready_marker_path(ready_dir, worker_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(time.time()), encoding="utf-8")
    print(f"[worker {worker_id}] model loaded onto its GPU -- wrote ready marker {path}", flush=True)


def wait_for_ready_marker(
    ready_dir: str | pathlib.Path, wait_for_worker_id: int, this_worker_id: int | None, timeout_seconds: float = DEFAULT_READY_WAIT_TIMEOUT_SECONDS
) -> None:
    """Staggered loading (2026-09-30, after a real Kaggle hang): a real
    session had both workers fetch/load the ~15GB model at the same moment,
    and loading stalled at 42% for 2+ hours with no error -- most likely CPU/
    disk contention between two simultaneous cold loads, since a LATER run
    against already-downloaded weights (timing_probe.py's --workers) worked
    fine. Worker N waits here for worker N-1's ready marker (written by
    write_ready_marker once ITS model is fully loaded) before starting its
    own load. Gives up and proceeds anyway after `timeout_seconds` (default
    20 min, matching multi_worker_phase0's watchdog) rather than waiting
    forever if the marker never appears -- the watchdog is the real
    backstop for a genuinely hung sibling, not this wait."""
    path = ready_marker_path(ready_dir, wait_for_worker_id)
    print(f"[worker {this_worker_id}] waiting for worker {wait_for_worker_id} to finish loading before starting my own load...", flush=True)
    t_start = time.monotonic()
    while not path.exists():
        if time.monotonic() - t_start > timeout_seconds:
            print(
                f"[worker {this_worker_id}] WARNING: gave up waiting for worker {wait_for_worker_id}'s ready "
                f"marker after {timeout_seconds:.0f}s -- proceeding with my own load anyway.", flush=True,
            )
            return
        time.sleep(READY_POLL_INTERVAL_SECONDS)
    print(f"[worker {this_worker_id}] worker {wait_for_worker_id} finished loading -- starting my own load now.", flush=True)


def completed_job_ids(results_path: str | pathlib.Path) -> set[str]:
    path = pathlib.Path(results_path)
    if not path.exists():
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["job_id"])
    return ids


def _run_one_job(
    job: dict,
    task_source: TaskSource,
    llm_client,
    memories: list,
    m: int,
    propensity_min: float,
    propensity_max: float,
    max_steps: int,
    similarity_fn: SimilarityFn | None,
    worker_id: int | None,
) -> dict:
    condition = job["condition"]
    if condition == "logging":
        rng = random.Random(job["seed"])
        env = task_source.build_env(job["task_id"])
        try:
            ep = run_logged_episode(
                env, llm_client, memories, m, propensity_min, propensity_max, max_steps,
                task_id=job["task_id"], rng=rng, similarity_fn=similarity_fn,
            )
        finally:
            env.close()
    elif condition in ("ground_truth_fold_a", "ground_truth_fold_b"):
        arm = "a" if condition.endswith("_a") else "b"
        ep = run_fold_over_arm(
            task_source, llm_client, memories, TARGET_MEMORY_IDS, job["mask"], m, propensity_min, propensity_max,
            max_steps, job["task_id"], job["pair_index"], arm, similarity_fn=similarity_fn,
        )
    else:
        raise ValueError(f"Unknown job condition: {condition!r}")
    ep["job_id"] = job["job_id"]
    # Which GPU/process ran this episode (multi_worker_phase0.py assigns one
    # worker_id per CUDA_VISIBLE_DEVICES) -- None for a plain single-worker
    # run, so this field is always present but doesn't imply multi-GPU.
    ep["worker_id"] = worker_id
    return ep


def run_jobs(
    job_list_path: str | pathlib.Path,
    results_path: str | pathlib.Path,
    task_source: TaskSource,
    llm_client,
    memories: list,
    m: int,
    propensity_min: float,
    propensity_max: float,
    max_steps: int,
    similarity_fn: SimilarityFn | None = None,
    max_new: int | None = None,
    time_budget_seconds: float | None = None,
    worker_id: int | None = None,
) -> int:
    """Returns the number of NEWLY completed jobs this call made. Safe to
    call again (same job_list_path/results_path) after any interruption --
    already-completed job_ids (by job_id, not by position) are skipped, so
    completed jobs are neither re-run nor lost regardless of how many times
    this is called or in how many separate processes/sessions.

    worker_id, if given, is stamped onto every episode this call produces
    (see _run_one_job) -- purely for logging which GPU/process ran it,
    doesn't affect which jobs get run."""
    jobs = load_job_list(job_list_path)
    results_path = pathlib.Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_job_ids(results_path)

    t_start = time.monotonic()
    n_new = 0
    with open(results_path, "a", encoding="utf-8") as f:
        for job in jobs:
            if job["job_id"] in done:
                continue
            if max_new is not None and n_new >= max_new:
                break
            if time_budget_seconds is not None and (time.monotonic() - t_start) >= time_budget_seconds:
                break
            ep_t_start = time.monotonic()
            ep = _run_one_job(job, task_source, llm_client, memories, m, propensity_min, propensity_max, max_steps, similarity_fn, worker_id)
            ep_seconds = time.monotonic() - ep_t_start
            f.write(json.dumps(ep) + "\n")
            f.flush()
            n_new += 1
            # One line per finished episode, flushed immediately -- a real
            # Kaggle session sat silent for 1h+ with no per-episode signal
            # that anything was actually happening (only a final summary at
            # chunk end). print(..., flush=True) plus -u on the subprocess's
            # own python invocation (multi_worker_phase0.py) together
            # guarantee this shows up right away instead of sitting in an
            # unflushed buffer.
            print(
                f"[worker {worker_id}] episode done: job_id={job['job_id']} success={ep.get('success')} "
                f"steps={ep.get('steps_taken')} seconds={ep_seconds:.1f}",
                flush=True,
            )
    return n_new


def main() -> None:
    """CLI entry point for one worker subprocess (spawned by
    multi_worker_phase0.py, one per GPU, exactly like timing_probe.py's
    --workers) -- also usable directly for a plain single-worker run.
    `--backend mock` runs the zero-cost mock env/LLM instead of the real
    Kaggle config, used to test the multi-GPU orchestration's own splitting/
    merging/crash-handling logic locally without needing real GPUs or
    ALFWorld (see tests/test_multi_worker_phase0.py)."""
    p = argparse.ArgumentParser()
    p.add_argument("--job-list-path", required=True)
    p.add_argument("--results-path", required=True)
    p.add_argument("--time-budget-seconds", type=float, default=None)
    p.add_argument("--max-new", type=int, default=None)
    p.add_argument("--worker-id", type=int, default=None)
    p.add_argument("--backend", choices=["real", "mock"], default="real")
    p.add_argument("--ready-dir", type=str, default=None, help="Directory for staggered-loading ready markers (multi-GPU only)")
    p.add_argument("--wait-for-worker-id", type=int, default=None, help="Wait for this worker's ready marker before loading the model")
    p.add_argument("--fail-after", type=int, default=None, help=argparse.SUPPRESS)  # test-only: simulate a crash after N completed jobs
    p.add_argument("--simulate-hang-seconds", type=float, default=None, help=argparse.SUPPRESS)  # test-only: simulate a hung load
    args = p.parse_args()

    _print_startup_diagnostics(args.worker_id)

    if args.simulate_hang_seconds is not None:
        # Test-only: stands in for a real hung model load, so the watchdog
        # in multi_worker_phase0.py can be exercised without needing a real
        # GPU/model to actually hang.
        print(f"[worker {args.worker_id}] SIMULATING A HANG for {args.simulate_hang_seconds:.0f}s", flush=True)
        time.sleep(args.simulate_hang_seconds)
        print(f"[worker {args.worker_id}] hang simulation finished", flush=True)

    if args.wait_for_worker_id is not None and args.ready_dir is not None:
        wait_for_ready_marker(args.ready_dir, args.wait_for_worker_id, args.worker_id)

    if args.backend == "mock":
        from .ground_truth_runner import MockTaskSource
        from .memory_store import build_mock_store
        from .mock_llm import MockLLMClient

        task_source = MockTaskSource()
        llm_client = MockLLMClient(strategy="scripted_success", seed=0)
        memories = build_mock_store(30, 100, 300, seed=0)
        m, prop_min, prop_max, max_steps = 10, 0.3, 0.7, 30
    else:
        cfg_path = ALFWORLD_PILOT_DIR / "kaggle_config.yaml"
        cfg = config_mod.load_config(cfg_path)
        real_cfg = load_real_alfworld_config(cfg["env"]["real_alfworld_config_path"])
        task_source = WeightedRealTaskSource(
            real_cfg, split=cfg["env"]["real_split"], task_type_weights=cfg["env"]["task_type_weights"]
        )
        llm_client, _cost_tracker = build_local_client(cfg)
        memories = build_kaggle_store()
        m = cfg["retrieval"]["M"]
        prop_min, prop_max = cfg["retrieval"]["propensity_min"], cfg["retrieval"]["propensity_max"]
        max_steps = cfg["env"]["max_steps"]

    if args.ready_dir is not None and args.worker_id is not None:
        write_ready_marker(args.ready_dir, args.worker_id)

    if args.fail_after is not None:
        # Test-only crash simulation: run in small increments, exiting
        # non-zero once `fail_after` jobs have been completed BY THIS
        # PROCESS (not cumulatively across restarts), to verify the
        # orchestrator keeps the sibling worker going and keeps this
        # worker's already-flushed results.
        completed_by_this_process = 0
        while completed_by_this_process < args.fail_after:
            n = run_jobs(
                args.job_list_path, args.results_path, task_source, llm_client, memories, m, prop_min, prop_max, max_steps,
                max_new=1, worker_id=args.worker_id,
            )
            if n == 0:
                break  # nothing left to run at all
            completed_by_this_process += n
        print(f"[worker {args.worker_id}] SIMULATED CRASH after {completed_by_this_process} job(s)")
        raise SystemExit(1)

    n_new = run_jobs(
        args.job_list_path, args.results_path, task_source, llm_client, memories, m, prop_min, prop_max, max_steps,
        time_budget_seconds=args.time_budget_seconds, max_new=args.max_new, worker_id=args.worker_id,
    )
    print(f"[worker {args.worker_id}] completed {n_new} new job(s)")


if __name__ == "__main__":
    main()
