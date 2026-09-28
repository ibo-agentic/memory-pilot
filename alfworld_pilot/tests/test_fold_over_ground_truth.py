"""Tests for fold_over_ground_truth.py -- the multi-memory paired design
(README's "Fold-over ground-truth design" section)."""

from __future__ import annotations

from alfworld_pilot.fold_over_ground_truth import (
    complement_mask,
    draw_fold_over_mask,
    run_fold_over_pair,
)
from alfworld_pilot.ground_truth_runner import MockTaskSource
from alfworld_pilot.memory_store import build_mock_store
from alfworld_pilot.mock_llm import MockLLMClient

TARGET_IDS = ["mem_0", "mem_1", "mem_2"]
M, P_MIN, P_MAX, MAX_STEPS = 10, 0.3, 0.7, 30


def test_complement_mask_flips_every_bit():
    mask = {"a": 1, "b": 0, "c": 1}
    comp = complement_mask(mask)
    assert comp == {"a": 0, "b": 1, "c": 0}


def test_draw_fold_over_mask_is_deterministic():
    m1 = draw_fold_over_mask(TARGET_IDS, pair_index=0, task_seed=42)
    m2 = draw_fold_over_mask(TARGET_IDS, pair_index=0, task_seed=42)
    assert m1 == m2


def test_draw_fold_over_mask_order_independent():
    m1 = draw_fold_over_mask(["mem_0", "mem_1", "mem_2"], pair_index=0, task_seed=42)
    m2 = draw_fold_over_mask(["mem_2", "mem_0", "mem_1"], pair_index=0, task_seed=42)
    assert m1 == m2


def test_draw_fold_over_mask_varies_by_pair_index():
    masks = {i: draw_fold_over_mask(TARGET_IDS, pair_index=i, task_seed=42) for i in range(10)}
    assert len(set(tuple(sorted(m.items())) for m in masks.values())) > 1  # not all identical


def test_run_fold_over_pair_every_target_memory_flips():
    memories = build_mock_store(30, 100, 300, seed=0)
    llm = MockLLMClient(strategy="scripted_success", seed=0)
    ep_a, ep_b = run_fold_over_pair(
        MockTaskSource(), llm, memories, TARGET_IDS, M, P_MIN, P_MAX, MAX_STEPS, task_seed=7, pair_index=0
    )
    for mid in TARGET_IDS:
        if mid in ep_a["candidate_ids"]:  # only forced if a natural candidate, by design
            assert ep_a["included"][mid] == 1 - ep_b["included"][mid]


def test_run_fold_over_pair_non_target_memories_identical_between_arms():
    memories = build_mock_store(30, 100, 300, seed=0)
    llm = MockLLMClient(strategy="scripted_success", seed=0)
    ep_a, ep_b = run_fold_over_pair(
        MockTaskSource(), llm, memories, TARGET_IDS, M, P_MIN, P_MAX, MAX_STEPS, task_seed=7, pair_index=0
    )
    non_target_ids = set(ep_a["candidate_ids"]) - set(TARGET_IDS)
    for mid in non_target_ids:
        assert ep_a["included"].get(mid) == ep_b["included"].get(mid)


def test_run_fold_over_pair_same_task_instance_both_arms():
    memories = build_mock_store(30, 100, 300, seed=0)
    llm = MockLLMClient(strategy="scripted_success", seed=0)
    ep_a, ep_b = run_fold_over_pair(
        MockTaskSource(), llm, memories, TARGET_IDS, M, P_MIN, P_MAX, MAX_STEPS, task_seed=7, pair_index=0
    )
    assert ep_a["task_type"] == ep_b["task_type"]
    assert ep_a["candidate_ids"] == ep_b["candidate_ids"]


def test_run_fold_over_pair_records_mask_and_pair_index():
    memories = build_mock_store(30, 100, 300, seed=0)
    llm = MockLLMClient(strategy="scripted_success", seed=0)
    ep_a, ep_b = run_fold_over_pair(
        MockTaskSource(), llm, memories, TARGET_IDS, M, P_MIN, P_MAX, MAX_STEPS, task_seed=7, pair_index=3
    )
    assert ep_a["fold_over_pair_index"] == 3
    assert ep_b["fold_over_pair_index"] == 3
    assert ep_a["fold_over_mask"] == complement_mask(ep_b["fold_over_mask"])
    assert ep_a["fold_over_arm"] == "a"
    assert ep_b["fold_over_arm"] == "b"
