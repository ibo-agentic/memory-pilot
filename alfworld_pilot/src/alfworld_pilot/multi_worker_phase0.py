"""2-GPU support for the Phase 0 pilot (2026-09-30) -- reuses
timing_probe.py's own worker pattern, already verified on real Kaggle (1.89x
combined speedup): one subprocess per GPU (CUDA_VISIBLE_DEVICES=0/1), each
running run_jobs.py's own CLI against its own job-list subset and its own
results file, spawned fresh once per checkpoint-interval chunk so the outer
time-guard/checkpoint loop (run_phase0_pilot.run_session_with_checkpoints)
stays exactly the same for 1 worker or N.

Splitting is recomputed FRESH every chunk from whatever's currently pending
across ALL workers' result files combined -- never a persisted assignment --
so a restart with a different pending set (one worker made more progress
than the other, or one crashed) always gets a freshly (re)balanced split,
never a stale one.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

from .job_list import load_job_list, write_job_list
from .run_jobs import completed_job_ids


def group_into_units(jobs: list[dict]) -> list[list[dict]]:
    """Groups jobs into indivisible assignment units: a ground_truth_fold_a/
    _b pair (matched by pair_index) is ONE unit, so both arms always land on
    the same worker (keeps a pair's two episodes co-located for simpler log
    correlation and debugging, per instruction -- not required for
    correctness, since each arm's own seeding is independent of where it
    runs). A logging job is its own unit. Order is deterministic given the
    same input jobs in the same order."""
    units: list[list[dict]] = []
    pair_units: dict[int, list[dict]] = {}
    pair_order: list[int] = []
    for job in jobs:
        if job["condition"] in ("ground_truth_fold_a", "ground_truth_fold_b"):
            pair_index = job["pair_index"]
            if pair_index not in pair_units:
                pair_units[pair_index] = []
                pair_order.append(pair_index)
            pair_units[pair_index].append(job)
        else:
            units.append([job])
    for pair_index in pair_order:
        units.append(pair_units[pair_index])
    return units


def split_units_across_workers(units: list[list[dict]], n_workers: int) -> list[list[dict]]:
    """Round-robin assigns whole units (see group_into_units) across
    n_workers, flattening each worker's assigned units back into a flat job
    list. Deterministic given the same units in the same order."""
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    worker_units: list[list[list[dict]]] = [[] for _ in range(n_workers)]
    for i, unit in enumerate(units):
        worker_units[i % n_workers].append(unit)
    return [[job for unit in w_units for job in unit] for w_units in worker_units]


def run_multi_gpu_chunk(
    full_job_list_path: str | pathlib.Path,
    worker_results_paths: list[pathlib.Path],
    chunk_budget_seconds: float,
    work_dir: str | pathlib.Path,
    backend: str = "real",
    extra_worker_args: list[list[str]] | None = None,
) -> int:
    """Runs ONE chunk of work split across len(worker_results_paths) GPUs,
    reusing run_jobs.py's own CLI (one subprocess per GPU,
    CUDA_VISIBLE_DEVICES=0..N-1 -- the same pattern timing_probe.py's
    --workers already verified on real Kaggle). See module docstring for why
    the split is recomputed fresh every call, never persisted.

    If a worker subprocess exits non-zero (crashes), this does NOT abort or
    retry -- the other worker(s) still run their own chunk to completion,
    and whatever the crashed worker completed before dying is already
    safely flushed in its own results file (run_jobs.py flushes after every
    job) and is kept, not lost. The crash is printed as a warning, never
    silently swallowed.

    extra_worker_args, if given, is a list of length len(worker_results_paths)
    of extra CLI args appended to each worker's own command -- used by tests
    to inject e.g. --fail-after for crash simulation; None means no extra
    args for any worker.

    Returns the total number of newly completed jobs across all workers this
    chunk (computed by diffing each worker's own completed-job count before
    and after, so it's correct regardless of which worker did how much)."""
    n_workers = len(worker_results_paths)
    all_jobs = load_job_list(full_job_list_path)
    already_done: set[str] = set()
    for results_path in worker_results_paths:
        already_done |= completed_job_ids(results_path)
    pending = [j for j in all_jobs if j["job_id"] not in already_done]

    if not pending:
        return 0

    units = group_into_units(pending)
    worker_job_lists = split_units_across_workers(units, n_workers)

    work_dir = pathlib.Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    n_before = sum(len(completed_job_ids(p)) for p in worker_results_paths)

    procs: list[subprocess.Popen | None] = []
    for worker_id, worker_jobs in enumerate(worker_job_lists):
        if not worker_jobs:
            procs.append(None)  # nothing assigned to this worker this chunk
            continue

        worker_job_list_path = work_dir / f"chunk_worker{worker_id}_jobs.json"
        write_job_list(worker_jobs, worker_job_list_path)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker_id)
        cmd = [
            sys.executable, "-m", "alfworld_pilot.run_jobs",
            "--job-list-path", str(worker_job_list_path),
            "--results-path", str(worker_results_paths[worker_id]),
            "--time-budget-seconds", str(chunk_budget_seconds),
            "--worker-id", str(worker_id),
            "--backend", backend,
        ]
        if extra_worker_args is not None:
            cmd += extra_worker_args[worker_id]
        procs.append(subprocess.Popen(cmd, env=env))

    # Wait for EVERY spawned worker, even if one exits early/non-zero --
    # a crashed worker must never stop its sibling from finishing its chunk.
    for worker_id, proc in enumerate(procs):
        if proc is None:
            continue
        ret = proc.wait()
        if ret != 0:
            print(
                f"[multi_worker_phase0] WARNING: worker {worker_id} exited with code {ret} (crashed) -- "
                "its already-completed jobs are kept; continuing with the other worker(s)."
            )

    n_after = sum(len(completed_job_ids(p)) for p in worker_results_paths)
    return n_after - n_before


def merge_worker_results(worker_results_paths: list[pathlib.Path], merged_path: pathlib.Path) -> int:
    """Concatenates every worker's results file into one merged JSONL file
    (job_id-deduplicated defensively, though the split guarantees disjoint
    job_ids across workers by construction) -- used to produce a single
    combined log for validate_per_task_type and for pushing one checkpoint.
    Returns the number of episodes written."""
    seen: set[str] = set()
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(merged_path, "w", encoding="utf-8") as out:
        for results_path in worker_results_paths:
            if not pathlib.Path(results_path).exists():
                continue
            with open(results_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ep = json.loads(line)
                    if ep["job_id"] in seen:
                        continue
                    seen.add(ep["job_id"])
                    out.write(line if line.endswith("\n") else line + "\n")
                    n_written += 1
    return n_written
