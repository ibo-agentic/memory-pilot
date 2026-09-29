"""Tests for multi_worker_phase0.py -- 2-GPU support for the Phase 0 pilot.

group_into_units/split_units_across_workers are pure and tested directly.
run_multi_gpu_chunk actually SPAWNS REAL SUBPROCESSES (run_jobs.py's own
CLI, exactly as it will run on Kaggle with CUDA_VISIBLE_DEVICES=0/1) --
using `--backend mock` means these subprocesses have zero GPU/ALFWorld
dependency, so the actual splitting/merging/resume/crash-handling mechanics
around the real subprocess-per-GPU pattern (the same one timing_probe.py's
--workers already verified on real Kaggle, 1.89x combined speedup) are
tested for real, not just simulated in-process."""

from __future__ import annotations

import json

from alfworld_pilot.job_list import generate_job_list, write_job_list
from alfworld_pilot.multi_worker_phase0 import (
    group_into_units,
    merge_worker_results,
    run_multi_gpu_chunk,
    split_units_across_workers,
)
from alfworld_pilot.run_jobs import completed_job_ids


def _logging_job(i):
    return {"job_id": f"log_{i}", "condition": "logging", "task_id": i, "pair_index": None, "mask": None, "seed": i}


def _fold_job(pair_index, arm):
    return {
        "job_id": f"gtfold_{pair_index}_{arm}",
        "condition": f"ground_truth_fold_{arm}",
        "task_id": 1_000_000 + pair_index,
        "pair_index": pair_index,
        "mask": {"pas_correct": 1 if arm == "a" else 0},
        "seed": 1_000_000 + pair_index,
    }


# --- group_into_units / split_units_across_workers: pure, fast ---


def test_group_into_units_keeps_fold_pair_together():
    jobs = [_fold_job(0, "a"), _fold_job(0, "b"), _logging_job(0)]
    units = group_into_units(jobs)
    assert len(units) == 2
    pair_unit = next(u for u in units if len(u) == 2)
    assert {j["job_id"] for j in pair_unit} == {"gtfold_0_a", "gtfold_0_b"}


def test_group_into_units_logging_jobs_are_their_own_unit():
    jobs = [_logging_job(0), _logging_job(1)]
    units = group_into_units(jobs)
    assert units == [[jobs[0]], [jobs[1]]]


def test_split_units_across_workers_never_splits_a_fold_pair():
    jobs = [_fold_job(i, arm) for i in range(6) for arm in ("a", "b")]
    units = group_into_units(jobs)
    worker_jobs = split_units_across_workers(units, n_workers=2)
    for pair_index in range(6):
        arms_present = [
            w for w, jobs_w in enumerate(worker_jobs)
            if any(j["pair_index"] == pair_index for j in jobs_w)
        ]
        assert len(arms_present) == 1  # both arms landed on exactly one worker


def test_split_units_across_workers_disjoint_and_covers_everything():
    jobs = [_logging_job(i) for i in range(9)]
    units = group_into_units(jobs)
    worker_jobs = split_units_across_workers(units, n_workers=2)
    all_ids = [j["job_id"] for w in worker_jobs for j in w]
    assert sorted(all_ids) == sorted(j["job_id"] for j in jobs)
    assert len(all_ids) == len(set(all_ids))


def test_split_units_across_workers_roughly_balanced():
    jobs = [_logging_job(i) for i in range(10)]
    units = group_into_units(jobs)
    worker_jobs = split_units_across_workers(units, n_workers=2)
    assert abs(len(worker_jobs[0]) - len(worker_jobs[1])) <= 1


# --- run_multi_gpu_chunk: real subprocesses, mock backend ---


def test_run_multi_gpu_chunk_splits_runs_and_completes_everything(tmp_path):
    jobs = generate_job_list(n_logging=10, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)

    worker_results = [tmp_path / "worker0.jsonl", tmp_path / "worker1.jsonl"]
    n_new = run_multi_gpu_chunk(job_list_path, worker_results, chunk_budget_seconds=60.0, work_dir=tmp_path / "work", backend="mock")

    assert n_new == 10
    all_done = completed_job_ids(worker_results[0]) | completed_job_ids(worker_results[1])
    assert all_done == {j["job_id"] for j in jobs}
    # Both workers actually did something -- this is a real split, not one
    # worker doing everything.
    assert completed_job_ids(worker_results[0])
    assert completed_job_ids(worker_results[1])


def test_run_multi_gpu_chunk_logs_which_worker_ran_each_episode(tmp_path):
    jobs = generate_job_list(n_logging=6, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)

    worker_results = [tmp_path / "worker0.jsonl", tmp_path / "worker1.jsonl"]
    run_multi_gpu_chunk(job_list_path, worker_results, chunk_budget_seconds=60.0, work_dir=tmp_path / "work", backend="mock")

    for expected_worker_id, path in enumerate(worker_results):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    assert json.loads(line)["worker_id"] == expected_worker_id


def test_run_multi_gpu_chunk_merge_produces_combined_log(tmp_path):
    jobs = generate_job_list(n_logging=8, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)

    worker_results = [tmp_path / "worker0.jsonl", tmp_path / "worker1.jsonl"]
    run_multi_gpu_chunk(job_list_path, worker_results, chunk_budget_seconds=60.0, work_dir=tmp_path / "work", backend="mock")

    merged_path = tmp_path / "merged.jsonl"
    n_written = merge_worker_results(worker_results, merged_path)
    assert n_written == 8
    with open(merged_path, "r", encoding="utf-8") as f:
        merged_ids = {json.loads(line)["job_id"] for line in f if line.strip()}
    assert merged_ids == {j["job_id"] for j in jobs}


def test_run_multi_gpu_chunk_resume_after_partial_session_no_duplicates(tmp_path):
    jobs = generate_job_list(n_logging=12, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)
    worker_results = [tmp_path / "worker0.jsonl", tmp_path / "worker1.jsonl"]

    # "Session" 1: a very short chunk budget -- may or may not complete
    # everything depending on scheduling, but should make some progress.
    first_new = run_multi_gpu_chunk(job_list_path, worker_results, chunk_budget_seconds=60.0, work_dir=tmp_path / "work", backend="mock")
    assert first_new == 12  # generous budget: this mock workload finishes in one chunk

    # Re-run "session 2" against the SAME job list/results files (simulating
    # a restart) -- must be a no-op, not a re-run.
    second_new = run_multi_gpu_chunk(job_list_path, worker_results, chunk_budget_seconds=60.0, work_dir=tmp_path / "work", backend="mock")
    assert second_new == 0

    all_ids = []
    for path in worker_results:
        with open(path, "r", encoding="utf-8") as f:
            all_ids.extend(json.loads(line)["job_id"] for line in f if line.strip())
    assert len(all_ids) == len(set(all_ids)) == 12  # no duplicates across the two chunks


def test_run_multi_gpu_chunk_resume_with_different_pending_set_rebalances(tmp_path):
    # Simulates worker 0 having already finished a bunch of jobs from a
    # PREVIOUS chunk (so this chunk's pending set is skewed) -- the split
    # must be recomputed fresh from what's actually pending now, not
    # repeat a stale 50/50 assignment.
    jobs = generate_job_list(n_logging=10, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)
    worker_results = [tmp_path / "worker0.jsonl", tmp_path / "worker1.jsonl"]

    # Pre-seed worker 0's results file as if it already did 8 of the 10 jobs.
    with open(worker_results[0], "w", encoding="utf-8") as f:
        for job in jobs[:8]:
            f.write(json.dumps({"job_id": job["job_id"], "success": 1}) + "\n")

    n_new = run_multi_gpu_chunk(job_list_path, worker_results, chunk_budget_seconds=60.0, work_dir=tmp_path / "work", backend="mock")

    assert n_new == 2  # only the 2 genuinely-remaining jobs
    all_done = completed_job_ids(worker_results[0]) | completed_job_ids(worker_results[1])
    assert all_done == {j["job_id"] for j in jobs}


def test_run_multi_gpu_chunk_one_worker_crashing_keeps_the_others_progress(tmp_path):
    jobs = generate_job_list(n_logging=10, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)
    worker_results = [tmp_path / "worker0.jsonl", tmp_path / "worker1.jsonl"]

    # Worker 0 "crashes" after completing exactly 1 job; worker 1 runs normally.
    n_new = run_multi_gpu_chunk(
        job_list_path, worker_results, chunk_budget_seconds=60.0, work_dir=tmp_path / "work", backend="mock",
        extra_worker_args=[["--fail-after", "1"], []],
    )

    worker0_done = completed_job_ids(worker_results[0])
    worker1_done = completed_job_ids(worker_results[1])
    assert len(worker0_done) == 1  # crashed worker's one completed job is kept, not lost
    assert len(worker1_done) >= 1  # the other worker kept going
    assert n_new == len(worker0_done) + len(worker1_done)
    assert worker0_done.isdisjoint(worker1_done)


def test_run_multi_gpu_chunk_session_still_progresses_after_a_crash_on_next_call(tmp_path):
    jobs = generate_job_list(n_logging=10, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)
    worker_results = [tmp_path / "worker0.jsonl", tmp_path / "worker1.jsonl"]

    run_multi_gpu_chunk(
        job_list_path, worker_results, chunk_budget_seconds=60.0, work_dir=tmp_path / "work", backend="mock",
        extra_worker_args=[["--fail-after", "1"], []],
    )
    # Next chunk: no more crashing -- the session (across chunks) should
    # still finish everything, including re-assigning whatever worker 0
    # didn't get to before it "crashed."
    n_new_second_chunk = run_multi_gpu_chunk(job_list_path, worker_results, chunk_budget_seconds=60.0, work_dir=tmp_path / "work", backend="mock")

    all_done = completed_job_ids(worker_results[0]) | completed_job_ids(worker_results[1])
    assert all_done == {j["job_id"] for j in jobs}
    assert n_new_second_chunk > 0
