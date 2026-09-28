"""Tests for timing_probe.py's pure, GPU-free helper functions
(split_episode_range, merge_worker_reports, summarize) -- the actual
multi-GPU subprocess execution path (run_multi_worker) can't be exercised
here (this development environment has exactly one GPU, and a real 2-GPU
run must happen on Kaggle), but the slicing/merging logic it depends on is
fully covered."""

from __future__ import annotations

import pytest

from alfworld_pilot.timing_probe import check_no_cache_hits, merge_worker_reports, split_episode_range, summarize


def test_split_episode_range_even():
    slices = split_episode_range(20, 2, start_seed=0)
    assert slices == [
        {"worker_id": 0, "start_seed": 0, "n_episodes": 10},
        {"worker_id": 1, "start_seed": 10, "n_episodes": 10},
    ]


def test_split_episode_range_uneven_remainder_spread_across_first_workers():
    slices = split_episode_range(20, 3, start_seed=0)
    assert [s["n_episodes"] for s in slices] == [7, 7, 6]
    assert sum(s["n_episodes"] for s in slices) == 20


def test_split_episode_range_slices_are_disjoint_and_contiguous():
    slices = split_episode_range(37, 4, start_seed=100)
    covered = []
    for s in slices:
        covered.extend(range(s["start_seed"], s["start_seed"] + s["n_episodes"]))
    assert covered == list(range(100, 137))  # exactly one pass over the whole range, no gaps/overlaps


def test_split_episode_range_rejects_more_workers_than_episodes():
    with pytest.raises(ValueError):
        split_episode_range(2, 5, start_seed=0)


def test_split_episode_range_rejects_zero_workers():
    with pytest.raises(ValueError):
        split_episode_range(10, 0, start_seed=0)


def _fake_report(task_ids: list[int], seconds: list[float]) -> dict:
    return {
        "episode_timings": [
            {"task_id": tid, "seconds": s, "steps_taken": 10, "input_tokens": 800, "output_tokens": 20, "seconds_per_step": s / 10}
            for tid, s in zip(task_ids, seconds)
        ]
    }


def test_merge_worker_reports_combines_all_episodes():
    reports = [
        _fake_report([0, 1], [10.0, 12.0]),
        _fake_report([2, 3], [11.0, 9.0]),
    ]
    merged = merge_worker_reports(reports, parallel_wall_clock_seconds=15.0)
    assert merged["total_episodes"] == 4
    assert merged["n_workers"] == 2
    assert merged["mean_seconds_per_episode"] == pytest.approx((10 + 12 + 11 + 9) / 4)
    assert merged["median_seconds_per_episode"] == pytest.approx(10.5)


def test_merge_worker_reports_speedup_reflects_parallel_wall_clock():
    # 2 workers, each doing 2 episodes at 10s/episode sequentially (20s each
    # if run alone) but running CONCURRENTLY in 20s wall-clock total ->
    # combined throughput = 4 episodes / 20s = 0.2 eps/s, vs. single-worker
    # 1/10 = 0.1 eps/s -> speedup should be ~2x (perfect parallel scaling).
    reports = [
        _fake_report([0, 1], [10.0, 10.0]),
        _fake_report([2, 3], [10.0, 10.0]),
    ]
    merged = merge_worker_reports(reports, parallel_wall_clock_seconds=20.0)
    assert merged["single_worker_episodes_per_second"] == pytest.approx(0.1)
    assert merged["combined_episodes_per_second"] == pytest.approx(0.2)
    assert merged["speedup_vs_single_worker"] == pytest.approx(2.0)


def test_merge_worker_reports_rejects_empty():
    with pytest.raises(ValueError):
        merge_worker_reports([{"episode_timings": []}], parallel_wall_clock_seconds=1.0)


def test_check_no_cache_hits_passes_when_clean():
    timings = [{"task_id": 0, "n_cache_hits": 0}, {"task_id": 1, "n_cache_hits": 0}]
    check_no_cache_hits(timings)  # should not raise


def test_check_no_cache_hits_raises_on_any_cache_hit():
    timings = [{"task_id": 0, "n_cache_hits": 0}, {"task_id": 1, "n_cache_hits": 3}]
    with pytest.raises(RuntimeError, match="cache hit"):
        check_no_cache_hits(timings)


def test_check_no_cache_hits_missing_field_treated_as_zero():
    # Older timing reports without n_cache_hits shouldn't spuriously raise.
    timings = [{"task_id": 0}, {"task_id": 1}]
    check_no_cache_hits(timings)  # should not raise


def test_summarize_mean_and_median_diverge_on_a_skewed_episode():
    # Mirrors this project's own finding (capability_check.py) that a few
    # long, looping episodes can pull the mean well above the median.
    timings = [{"seconds": s} for s in [5.0, 5.0, 5.0, 5.0, 100.0]]
    summary = summarize(timings)
    assert summary["median_seconds_per_episode"] == pytest.approx(5.0)
    assert summary["mean_seconds_per_episode"] == pytest.approx(24.0)
    assert summary["episodes_per_30_gpu_hours_median"] > summary["episodes_per_30_gpu_hours_mean"]
