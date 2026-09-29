"""Tests for run_phase0_pilot.run_session_with_checkpoints -- the chunking/
stopping logic behind the time guard and periodic checkpointing (README's
"Time guard" and "Periodic checkpoint" sections), generalized (2026-09-30)
to take an injectable run_chunk_fn so the SAME loop covers both the
single-worker (run_jobs in-process) and multi-worker (subprocess-spawning,
see test_multi_worker_phase0.py) cases. Uses the mock backend and a fake
checkpoint_fn so this is fully local, no Kaggle/GPU dependency --
kaggle_session.copy_in/copy_out_and_version themselves stay untestable
outside a real session, but this loop's own correctness doesn't depend on
them at all (checkpoint_fn is injected)."""

from __future__ import annotations

import time

from alfworld_pilot.ground_truth_runner import MockTaskSource
from alfworld_pilot.job_list import generate_job_list, write_job_list
from alfworld_pilot.memory_store import build_mock_store
from alfworld_pilot.mock_llm import MockLLMClient
from alfworld_pilot.run_jobs import completed_job_ids, run_jobs
from alfworld_pilot.run_phase0_pilot import run_session_with_checkpoints

M, P_MIN, P_MAX, MAX_STEPS = 10, 0.3, 0.7, 30


def _memories():
    return build_mock_store(30, 100, 300, seed=0)


class _CountingCheckpoint:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1


class _SlowMockLLMClient(MockLLMClient):
    """Adds a small real delay per call so time-budget/chunking behavior is
    observable without needing to mock time.monotonic()."""

    def __init__(self, delay_seconds: float, **kwargs):
        super().__init__(**kwargs)
        self.delay_seconds = delay_seconds

    def complete(self, messages, stop=None):
        time.sleep(self.delay_seconds)
        return super().complete(messages, stop)


def _single_worker_chunk_fn(job_list_path, results_path, task_source, llm_client, memories):
    def _run_chunk(chunk_budget: float) -> int:
        return run_jobs(job_list_path, results_path, task_source, llm_client, memories, M, P_MIN, P_MAX, MAX_STEPS, time_budget_seconds=chunk_budget)
    return _run_chunk


def test_completes_all_jobs_with_a_generous_budget_and_checkpoints_at_least_once(tmp_path):
    jobs = generate_job_list(n_logging=5, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)
    results_path = tmp_path / "results.jsonl"
    checkpoint = _CountingCheckpoint()

    run_chunk_fn = _single_worker_chunk_fn(job_list_path, results_path, MockTaskSource(), MockLLMClient(strategy="scripted_success", seed=0), _memories())
    total_new = run_session_with_checkpoints(run_chunk_fn, time_budget_seconds=60.0, checkpoint_interval_seconds=60.0, checkpoint_fn=checkpoint)

    assert total_new == 5
    assert completed_job_ids(results_path) == {j["job_id"] for j in jobs}
    assert checkpoint.calls >= 1


def test_stops_immediately_without_checkpointing_if_budget_already_exhausted(tmp_path):
    jobs = generate_job_list(n_logging=5, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)
    results_path = tmp_path / "results.jsonl"
    checkpoint = _CountingCheckpoint()

    run_chunk_fn = _single_worker_chunk_fn(job_list_path, results_path, MockTaskSource(), MockLLMClient(strategy="scripted_success", seed=0), _memories())
    total_new = run_session_with_checkpoints(run_chunk_fn, time_budget_seconds=0.0, checkpoint_interval_seconds=60.0, checkpoint_fn=checkpoint)

    assert total_new == 0
    assert checkpoint.calls == 0
    assert completed_job_ids(results_path) == set()


def test_checkpoints_after_every_chunk_not_just_at_the_end(tmp_path):
    jobs = generate_job_list(n_logging=6, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)
    results_path = tmp_path / "results.jsonl"
    checkpoint = _CountingCheckpoint()

    # Each job takes ~50ms; a 120ms checkpoint interval should force several
    # chunks (roughly 2 jobs/chunk) before the whole job list is exhausted.
    llm = _SlowMockLLMClient(delay_seconds=0.05, strategy="scripted_success", seed=0)
    run_chunk_fn = _single_worker_chunk_fn(job_list_path, results_path, MockTaskSource(), llm, _memories())
    total_new = run_session_with_checkpoints(run_chunk_fn, time_budget_seconds=30.0, checkpoint_interval_seconds=0.12, checkpoint_fn=checkpoint)

    assert total_new == 6
    assert completed_job_ids(results_path) == {j["job_id"] for j in jobs}
    assert checkpoint.calls > 1  # more than just a single final checkpoint


def test_resuming_a_second_call_does_not_duplicate_or_lose_jobs(tmp_path):
    jobs = generate_job_list(n_logging=6, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)
    results_path = tmp_path / "results.jsonl"

    # First "session": a short overall budget interrupts partway through.
    llm = _SlowMockLLMClient(delay_seconds=0.05, strategy="scripted_success", seed=0)
    memories = _memories()
    run_chunk_fn = _single_worker_chunk_fn(job_list_path, results_path, MockTaskSource(), llm, memories)
    checkpoint1 = _CountingCheckpoint()
    first_total = run_session_with_checkpoints(run_chunk_fn, time_budget_seconds=0.15, checkpoint_interval_seconds=0.15, checkpoint_fn=checkpoint1)
    assert 0 < first_total < 6

    # Second "session": fresh call, same job list/results path, generous budget.
    checkpoint2 = _CountingCheckpoint()
    second_total = run_session_with_checkpoints(run_chunk_fn, time_budget_seconds=30.0, checkpoint_interval_seconds=30.0, checkpoint_fn=checkpoint2)

    assert first_total + second_total == 6
    assert completed_job_ids(results_path) == {j["job_id"] for j in jobs}

    import json
    with open(results_path, "r", encoding="utf-8") as f:
        job_ids_in_results = [json.loads(line)["job_id"] for line in f if line.strip()]
    assert len(job_ids_in_results) == len(set(job_ids_in_results)) == 6


def test_run_chunk_fn_receives_the_chunk_budget_not_the_overall_budget():
    seen_budgets = []

    def _run_chunk(chunk_budget: float) -> int:
        seen_budgets.append(chunk_budget)
        return 0  # stop after the first chunk

    run_session_with_checkpoints(_run_chunk, time_budget_seconds=100.0, checkpoint_interval_seconds=10.0, checkpoint_fn=lambda: None)

    assert seen_budgets == [10.0]  # the interval, not the full 100s budget
