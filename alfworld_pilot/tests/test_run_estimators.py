"""Tests for run_estimators.py -- ground truth recovery from fold-over
pairs, the memory_worth naive-gap fix, and the end-to-end validation
structure (README's "Pre-registered validation metrics" section)."""

from __future__ import annotations

from alfworld_pilot.run_estimators import (
    _memory_worth_naive_gap,
    ground_truth_effects,
    validate_against_ground_truth,
    validate_per_task_type,
)

ALL_IDS = ["mem_a", "mem_b", "mem_c"]


def _fold_ep(pair_index, arm, candidate_ids, included, success, task_type="pick_and_place_simple", task_id=0):
    return {
        "fold_over_pair_index": pair_index,
        "fold_over_arm": arm,
        "candidate_ids": candidate_ids,
        "included": included,
        "success": success,
        "task_type": task_type,
        "task_id": task_id,
        "propensities": {mid: 0.5 for mid in candidate_ids},
    }


def test_ground_truth_effects_recovers_a_planted_effect():
    # mem_a: always success=1 when included, 0 when not -> planted effect = 1.0
    fold_episodes = []
    for i in range(5):
        fold_episodes.append(_fold_ep(i, "a", ["mem_a"], {"mem_a": 1}, success=1, task_id=i))
        fold_episodes.append(_fold_ep(i, "b", ["mem_a"], {"mem_a": 0}, success=0, task_id=i))

    gt = ground_truth_effects(fold_episodes, target_memory_ids=["mem_a"])
    assert gt["mem_a"]["n_pairs"] == 5
    assert gt["mem_a"]["mean_effect"] == 1.0


def test_ground_truth_effects_zero_effect_memory():
    fold_episodes = []
    for i in range(4):
        fold_episodes.append(_fold_ep(i, "a", ["mem_a"], {"mem_a": 1}, success=1, task_id=i))
        fold_episodes.append(_fold_ep(i, "b", ["mem_a"], {"mem_a": 0}, success=1, task_id=i))  # same outcome either way

    gt = ground_truth_effects(fold_episodes, target_memory_ids=["mem_a"])
    assert gt["mem_a"]["mean_effect"] == 0.0


def test_ground_truth_effects_only_counts_natural_candidates():
    fold_episodes = [
        _fold_ep(0, "a", ["mem_a"], {"mem_a": 1}, success=1, task_id=0),
        _fold_ep(0, "b", ["mem_a"], {"mem_a": 0}, success=0, task_id=0),
        # mem_b never appears in candidate_ids here -- not a natural candidate for this pair.
    ]
    gt = ground_truth_effects(fold_episodes, target_memory_ids=["mem_a", "mem_b"])
    assert gt["mem_a"]["n_pairs"] == 1
    assert gt["mem_b"]["n_pairs"] == 0
    assert gt["mem_b"]["mean_effect"] is None


def test_ground_truth_effects_incomplete_pair_is_skipped():
    # Only arm "a" present for pair 0 -- no complete pair, contributes nothing.
    fold_episodes = [_fold_ep(0, "a", ["mem_a"], {"mem_a": 1}, success=1, task_id=0)]
    gt = ground_truth_effects(fold_episodes, target_memory_ids=["mem_a"])
    assert gt["mem_a"]["n_pairs"] == 0


def test_memory_worth_naive_gap_basic():
    episodes = [
        {"success": 1, "candidate_ids": ["mem_a"], "included": {"mem_a": 1}},
        {"success": 0, "candidate_ids": ["mem_a"], "included": {"mem_a": 1}},
        {"success": 1, "candidate_ids": ["mem_a"], "included": {"mem_a": 0}},
        {"success": 1, "candidate_ids": ["mem_a"], "included": {"mem_a": 0}},
    ]
    gap = _memory_worth_naive_gap(episodes)
    # P(Y=1|Z=1) = 0.5 (1 of 2), P(Y=1|Z=0) = 1.0 (2 of 2) -> gap = -0.5
    assert gap["mem_a"] == -0.5


def test_memory_worth_naive_gap_none_when_one_arm_missing():
    episodes = [{"success": 1, "candidate_ids": ["mem_a"], "included": {"mem_a": 1}}]
    gap = _memory_worth_naive_gap(episodes)
    assert gap["mem_a"] is None


def _synthetic_logging_and_fold(n=30):
    import random

    rng = random.Random(0)
    logging_episodes = []
    for i in range(n):
        included = {"mem_a": rng.randint(0, 1), "mem_b": rng.randint(0, 1)}
        success = 1 if (included["mem_a"] == 1 and rng.random() < 0.7) or rng.random() < 0.3 else 0
        logging_episodes.append(
            {
                "task_id": i,
                "task_type": "pick_and_place_simple" if i % 2 == 0 else "look_at_obj_in_light",
                "candidate_ids": ["mem_a", "mem_b"],
                "propensities": {"mem_a": 0.5, "mem_b": 0.5},
                "included": included,
                "success": success,
                "steps_taken": rng.randint(1, 40),
            }
        )

    fold_episodes = []
    for i in range(10):
        in_arm_success = 1 if rng.random() < 0.7 else 0
        out_arm_success = 1 if rng.random() < 0.3 else 0
        fold_episodes.append(_fold_ep(i, "a", ["mem_a", "mem_b"], {"mem_a": 1, "mem_b": 0}, in_arm_success, task_id=1000 + i))
        fold_episodes.append(_fold_ep(i, "b", ["mem_a", "mem_b"], {"mem_a": 0, "mem_b": 1}, out_arm_success, task_id=1000 + i))
    return logging_episodes, fold_episodes


def test_validate_against_ground_truth_structure():
    logging_episodes, fold_episodes = _synthetic_logging_and_fold()
    result = validate_against_ground_truth(logging_episodes, fold_episodes, ALL_IDS, target_memory_ids=["mem_a", "mem_b"], n_bootstrap=20)

    assert set(result["outcomes"].keys()) == {"primary", "success_within_25", "success_within_30"}
    for outcome_name, per_estimator in result["outcomes"].items():
        assert set(per_estimator.keys()) == {"memory_worth", "memory_worth_raw", "ips", "snips", "doubly_robust"}
        for est_name, per_memory in per_estimator.items():
            assert set(per_memory.keys()) == {"mem_a", "mem_b"}
            if est_name == "memory_worth_raw":
                # Author's-original-form reporting (README's Phase 0 plan
                # dual-reporting note): no ground-truth comparison fields,
                # since a raw P(success|included) rate isn't on the gap
                # scale ground truth is.
                for mid, row in per_memory.items():
                    assert "point_estimate" in row and "ci_low" in row and "ci_high" in row
                continue
            for mid, row in per_memory.items():
                assert "point_estimate" in row and "ci_low" in row and "ci_high" in row
                assert "ground_truth" in row and "error" in row and "ground_truth_in_ci" in row


def test_memory_worth_dual_reporting_differs_from_naive_gap():
    # README's Phase 0 plan: Memory Worth reported both in the author's
    # original form (raw P(success|included) rate) and as the naive gap --
    # these must actually be different quantities, not the same value twice.
    logging_episodes, fold_episodes = _synthetic_logging_and_fold()
    result = validate_against_ground_truth(logging_episodes, fold_episodes, ALL_IDS, target_memory_ids=["mem_a", "mem_b"], n_bootstrap=20)
    primary = result["outcomes"]["primary"]
    for mid in ["mem_a", "mem_b"]:
        raw = primary["memory_worth_raw"][mid]["point_estimate"]
        gap = primary["memory_worth"][mid]["point_estimate"]
        assert raw is not None and gap is not None
        assert raw != gap  # a rate (~0-1) is never equal to a gap (~-1 to 1) except by coincidence at 0
        assert 0.0 <= raw <= 1.0  # raw is a probability
        assert -1.0 <= gap <= 1.0  # gap is a difference of two probabilities


def test_validate_per_task_type_has_pooled_and_per_type():
    logging_episodes, fold_episodes = _synthetic_logging_and_fold()
    result = validate_per_task_type(logging_episodes, fold_episodes, ALL_IDS, target_memory_ids=["mem_a", "mem_b"], n_bootstrap=20)
    assert "pooled" in result
    assert "per_task_type" in result
    assert "pick_and_place_simple" in result["per_task_type"]
    assert "look_at_obj_in_light" in result["per_task_type"]
