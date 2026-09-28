"""Resumable job-list runner (2026-09-29): reads a job list written by
job_list.py, skips whatever's already in the results log, and executes the
rest up to a time budget or max_new cap -- flushing after every single job
(not just at session end) so a crash loses at most one in-flight job, the
same safety margin checkpointed_runner.py already gives the paid run's own
logging/ground-truth phases.

Deliberately reuses episode_runner.run_logged_episode (logging jobs) and
fold_over_ground_truth.run_fold_over_arm (ground-truth fold jobs) unchanged
-- this module is purely the "which jobs are left, and how much time is
left" orchestration layer.
"""

from __future__ import annotations

import json
import pathlib
import random
import time

from .episode_runner import SimilarityFn, run_logged_episode
from .fold_over_ground_truth import run_fold_over_arm
from .ground_truth_runner import TaskSource
from .job_list import TARGET_MEMORY_IDS, load_job_list


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
) -> int:
    """Returns the number of NEWLY completed jobs this call made. Safe to
    call again (same job_list_path/results_path) after any interruption --
    already-completed job_ids (by job_id, not by position) are skipped, so
    completed jobs are neither re-run nor lost regardless of how many times
    this is called or in how many separate processes/sessions."""
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
            ep = _run_one_job(job, task_source, llm_client, memories, m, propensity_min, propensity_max, max_steps, similarity_fn)
            f.write(json.dumps(ep) + "\n")
            f.flush()
            n_new += 1
    return n_new
