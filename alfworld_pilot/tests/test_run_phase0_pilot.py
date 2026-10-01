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

import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from alfworld_pilot.ground_truth_runner import MockTaskSource
from alfworld_pilot.job_list import generate_job_list, write_job_list
from alfworld_pilot.memory_store import build_mock_store
from alfworld_pilot.mock_llm import MockLLMClient
from alfworld_pilot.multi_worker_phase0 import WatchdogTimeoutError
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


def test_watchdog_timeout_still_checkpoints_before_reraising():
    checkpoint = _CountingCheckpoint()

    def _run_chunk(chunk_budget: float) -> int:
        raise WatchdogTimeoutError("worker(s) [0] hung for over 1s with no progress (0 job(s) completed)")

    with pytest.raises(WatchdogTimeoutError):
        run_session_with_checkpoints(_run_chunk, time_budget_seconds=100.0, checkpoint_interval_seconds=10.0, checkpoint_fn=checkpoint)

    assert checkpoint.calls == 1  # saved whatever was done before the exception propagated


def test_run_chunk_fn_receives_the_chunk_budget_not_the_overall_budget():
    seen_budgets = []

    def _run_chunk(chunk_budget: float) -> int:
        seen_budgets.append(chunk_budget)
        return 0  # stop after the first chunk

    run_session_with_checkpoints(_run_chunk, time_budget_seconds=100.0, checkpoint_interval_seconds=10.0, checkpoint_fn=lambda: None)

    assert seen_budgets == [10.0]  # the interval, not the full 100s budget


def test_first_chunk_uses_its_own_shorter_interval_then_falls_back(tmp_path):
    seen_budgets = []
    checkpoint = _CountingCheckpoint()

    def _run_chunk(chunk_budget: float) -> int:
        seen_budgets.append(chunk_budget)
        return 1 if len(seen_budgets) < 4 else 0  # keep going for 3 chunks, then stop

    run_session_with_checkpoints(
        _run_chunk, time_budget_seconds=1000.0, checkpoint_interval_seconds=100.0, checkpoint_fn=checkpoint,
        first_checkpoint_interval_seconds=5.0,
    )

    # Only the FIRST chunk uses the short 30-min-style interval -- every chunk
    # after that falls back to the normal (longer) checkpoint_interval_seconds.
    assert seen_budgets == [5.0, 100.0, 100.0, 100.0]
    assert checkpoint.calls == 4


def test_omitting_first_checkpoint_interval_matches_old_single_interval_behavior():
    seen_budgets = []

    def _run_chunk(chunk_budget: float) -> int:
        seen_budgets.append(chunk_budget)
        return 1 if len(seen_budgets) < 2 else 0

    run_session_with_checkpoints(_run_chunk, time_budget_seconds=1000.0, checkpoint_interval_seconds=10.0, checkpoint_fn=lambda: None)

    assert seen_budgets == [10.0, 10.0]  # no special-cased first chunk when omitted


def test_main_pushes_an_initial_checkpoint_right_after_generating_the_job_list(tmp_path, monkeypatch):
    """Session 1 was lost on real Kaggle because no checkpoint existed until
    the first full chunk finished -- it died before that ever happened, and
    no Dataset was ever created. This confirms main() now pushes a checkpoint
    of the job list ALONE, before predownload_model or any episode ever runs,
    with every heavy dependency (model download, real ALFWorld, real episode
    running) mocked out so this runs in milliseconds."""
    import alfworld_pilot.run_phase0_pilot as rpp

    calls = []
    monkeypatch.setattr(rpp, "KAGGLE_WORKING", tmp_path)

    def _fake_copy_in(slug, logs_dir, cache_dir):
        calls.append(("copy_in", slug))

    def _fake_copy_out(slug, logs_dir, cache_dir, message):
        calls.append(("copy_out", message))
        # The job list must already be on disk by the time the FIRST push happens.
        assert (logs_dir / "phase0_job_list.json").exists()

    def _fake_predownload(cfg):
        calls.append(("predownload_model", None))

    def _fake_run_jobs(*args, **kwargs):
        calls.append(("run_jobs", None))
        return 0  # nothing new -- ends the session after one (empty) chunk

    with patch.object(rpp, "copy_in", _fake_copy_in), \
         patch.object(rpp, "copy_out_and_version", _fake_copy_out), \
         patch.object(rpp, "predownload_model", _fake_predownload), \
         patch.object(rpp, "load_real_alfworld_config", MagicMock()), \
         patch.object(rpp, "WeightedRealTaskSource", MagicMock()), \
         patch.object(rpp, "build_local_client", MagicMock(return_value=(MagicMock(), MagicMock(summary=lambda: "n/a")))), \
         patch.object(rpp, "run_jobs", _fake_run_jobs):
        monkeypatch.setattr(sys, "argv", [
            "run_phase0_pilot",
            "--dataset-slug", "someone/some-dataset",
            "--n-logging", "2",
            "--n-gt-pairs", "0",
            "--time-budget-seconds", "5",
            "--first-checkpoint-interval-seconds", "5",
            "--checkpoint-interval-seconds", "5",
            "--workers", "1",
        ])
        rpp.main()

    call_names = [c[0] for c in calls]
    # The initial job-list-only push happens before predownload_model AND
    # before any job ever runs -- not just "a checkpoint happened eventually".
    assert call_names.index("copy_out") < call_names.index("predownload_model")
    assert call_names.index("copy_out") < call_names.index("run_jobs")
    assert calls[call_names.index("copy_out")][1] == "Phase 0 pilot: initial checkpoint (job list only, no episodes yet)"


def test_main_does_not_push_a_second_initial_checkpoint_when_resuming(tmp_path, monkeypatch):
    """On a later session the job list already exists -- the initial,
    job-list-only push should only ever happen on the FIRST session that
    generates it, not be repeated every time main() runs."""
    import alfworld_pilot.run_phase0_pilot as rpp
    from alfworld_pilot.job_list import generate_job_list, write_job_list

    monkeypatch.setattr(rpp, "KAGGLE_WORKING", tmp_path)
    logs_dir = tmp_path / "logs_kaggle"
    logs_dir.mkdir(parents=True)
    write_job_list(generate_job_list(n_logging=2, n_gt_pairs=0), logs_dir / "phase0_job_list.json")

    copy_out_calls = []

    def _fake_copy_out(slug, logs_dir, cache_dir, message):
        copy_out_calls.append(message)

    def _fake_run_jobs(*args, **kwargs):
        return 0

    with patch.object(rpp, "copy_in", MagicMock()), \
         patch.object(rpp, "copy_out_and_version", _fake_copy_out), \
         patch.object(rpp, "predownload_model", MagicMock()), \
         patch.object(rpp, "load_real_alfworld_config", MagicMock()), \
         patch.object(rpp, "WeightedRealTaskSource", MagicMock()), \
         patch.object(rpp, "build_local_client", MagicMock(return_value=(MagicMock(), MagicMock(summary=lambda: "n/a")))), \
         patch.object(rpp, "run_jobs", _fake_run_jobs):
        monkeypatch.setattr(sys, "argv", [
            "run_phase0_pilot",
            "--dataset-slug", "someone/some-dataset",
            "--n-logging", "2",
            "--n-gt-pairs", "0",
            "--time-budget-seconds", "5",
            "--first-checkpoint-interval-seconds", "5",
            "--checkpoint-interval-seconds", "5",
            "--workers", "1",
        ])
        rpp.main()

    assert "Phase 0 pilot: initial checkpoint (job list only, no episodes yet)" not in copy_out_calls
    assert copy_out_calls.count("Phase 0 pilot: periodic checkpoint") == 1  # the one regular (empty) chunk's checkpoint
