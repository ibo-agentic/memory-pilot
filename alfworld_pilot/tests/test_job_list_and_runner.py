"""Tests for job_list.py (deterministic job generation) and run_jobs.py (the
resumable runner) -- these are what the tiny handoff test
(handoff_test.py) exercises for real, but the core "skip what's already
done, never duplicate, never lose a job" logic is unit-tested here directly
against the mock backend so it runs in milliseconds."""

from __future__ import annotations

import json

from alfworld_pilot.ground_truth_runner import MockTaskSource
from alfworld_pilot.job_list import generate_job_list, load_job_list, write_job_list
from alfworld_pilot.memory_store import build_mock_store
from alfworld_pilot.mock_llm import MockLLMClient
from alfworld_pilot.run_jobs import completed_job_ids, run_jobs

M, P_MIN, P_MAX, MAX_STEPS = 10, 0.3, 0.7, 30


def test_generate_job_list_is_deterministic():
    jobs1 = generate_job_list(n_logging=5, n_gt_pairs=3)
    jobs2 = generate_job_list(n_logging=5, n_gt_pairs=3)
    assert jobs1 == jobs2


def test_generate_job_list_counts_and_ids():
    jobs = generate_job_list(n_logging=5, n_gt_pairs=3)
    logging_jobs = [j for j in jobs if j["condition"] == "logging"]
    gt_a = [j for j in jobs if j["condition"] == "ground_truth_fold_a"]
    gt_b = [j for j in jobs if j["condition"] == "ground_truth_fold_b"]
    assert len(logging_jobs) == 5
    assert len(gt_a) == 3
    assert len(gt_b) == 3
    assert len({j["job_id"] for j in jobs}) == len(jobs)  # every job_id unique


def test_generate_job_list_task_ids_disjoint_between_logging_and_gt():
    jobs = generate_job_list(n_logging=5, n_gt_pairs=3)
    logging_ids = {j["task_id"] for j in jobs if j["condition"] == "logging"}
    gt_ids = {j["task_id"] for j in jobs if j["condition"].startswith("ground_truth")}
    assert logging_ids.isdisjoint(gt_ids)


def test_generate_job_list_fold_pair_masks_are_complements():
    jobs = generate_job_list(n_logging=0, n_gt_pairs=2)
    by_pair = {}
    for j in jobs:
        by_pair.setdefault(j["pair_index"], {})[j["condition"]] = j["mask"]
    for pair_index, arms in by_pair.items():
        mask_a = arms["ground_truth_fold_a"]
        mask_b = arms["ground_truth_fold_b"]
        for mid in mask_a:
            assert mask_a[mid] == 1 - mask_b[mid]


def test_write_and_load_job_list_roundtrip(tmp_path):
    jobs = generate_job_list(n_logging=3, n_gt_pairs=1)
    path = tmp_path / "jobs.json"
    write_job_list(jobs, path)
    loaded = load_job_list(path)
    assert loaded == jobs


def _memories():
    return build_mock_store(30, 100, 300, seed=0)


def test_run_jobs_completes_all_jobs_in_one_call(tmp_path):
    jobs = generate_job_list(n_logging=5, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)
    results_path = tmp_path / "results.jsonl"

    llm = MockLLMClient(strategy="scripted_success", seed=0)
    n_new = run_jobs(job_list_path, results_path, MockTaskSource(), llm, _memories(), M, P_MIN, P_MAX, MAX_STEPS)

    assert n_new == 5
    assert completed_job_ids(results_path) == {j["job_id"] for j in jobs}


def test_run_jobs_max_new_then_resume_no_duplicates_no_loss(tmp_path):
    jobs = generate_job_list(n_logging=5, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)
    results_path = tmp_path / "results.jsonl"
    llm = MockLLMClient(strategy="scripted_success", seed=0)

    n_first = run_jobs(job_list_path, results_path, MockTaskSource(), llm, _memories(), M, P_MIN, P_MAX, MAX_STEPS, max_new=3)
    assert n_first == 3
    assert len(completed_job_ids(results_path)) == 3

    # Simulates a new Kaggle session: fresh call, same job list + results path.
    n_second = run_jobs(job_list_path, results_path, MockTaskSource(), llm, _memories(), M, P_MIN, P_MAX, MAX_STEPS, max_new=10)
    assert n_second == 2  # only the remaining 2, not all 5 again

    done_ids = completed_job_ids(results_path)
    assert done_ids == {j["job_id"] for j in jobs}

    # No job ran twice: exactly one result line per job_id.
    with open(results_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    job_ids_in_results = [r["job_id"] for r in lines]
    assert len(job_ids_in_results) == len(set(job_ids_in_results)) == 5


def test_run_jobs_calling_again_with_nothing_left_is_a_noop(tmp_path):
    jobs = generate_job_list(n_logging=2, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)
    results_path = tmp_path / "results.jsonl"
    llm = MockLLMClient(strategy="scripted_success", seed=0)

    run_jobs(job_list_path, results_path, MockTaskSource(), llm, _memories(), M, P_MIN, P_MAX, MAX_STEPS)
    n_again = run_jobs(job_list_path, results_path, MockTaskSource(), llm, _memories(), M, P_MIN, P_MAX, MAX_STEPS)
    assert n_again == 0


def test_run_jobs_prints_one_flushed_line_per_finished_episode(tmp_path, capsys):
    jobs = generate_job_list(n_logging=3, n_gt_pairs=0)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)
    results_path = tmp_path / "results.jsonl"
    llm = MockLLMClient(strategy="scripted_success", seed=0)

    run_jobs(job_list_path, results_path, MockTaskSource(), llm, _memories(), M, P_MIN, P_MAX, MAX_STEPS, worker_id=1)

    out = capsys.readouterr().out
    printed_lines = [line for line in out.splitlines() if "episode done" in line]
    assert len(printed_lines) == 3  # one per finished episode, not just a final summary
    for job in jobs:
        assert any(f"job_id={job['job_id']}" in line for line in printed_lines)
    for line in printed_lines:
        assert line.startswith("[worker 1]")
        assert "success=" in line and "steps=" in line and "seconds=" in line


def test_run_jobs_handles_ground_truth_fold_jobs(tmp_path):
    jobs = generate_job_list(n_logging=0, n_gt_pairs=2)
    job_list_path = tmp_path / "jobs.json"
    write_job_list(jobs, job_list_path)
    results_path = tmp_path / "results.jsonl"
    llm = MockLLMClient(strategy="scripted_success", seed=0)

    n_new = run_jobs(job_list_path, results_path, MockTaskSource(), llm, _memories(), M, P_MIN, P_MAX, MAX_STEPS)
    assert n_new == 4  # 2 pairs x 2 arms

    with open(results_path, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]
    for r in results:
        assert "fold_over_pair_index" in r
        assert "fold_over_mask" in r
        assert r["fold_over_arm"] in ("a", "b")
