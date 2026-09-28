"""Tests for secondary_outcomes.truncated_success -- the pre-registered
success_within_25/30 outcome (see README's "Pre-registered secondary
outcome" section)."""

from __future__ import annotations

import pytest

from alfworld_pilot.secondary_outcomes import truncated_success


def test_success_within_cap_stays_a_success():
    episodes = [{"task_id": 0, "success": 1, "steps_taken": 10}]
    out = truncated_success(episodes, step_cap=25)
    assert out[0]["success"] == 1


def test_success_beyond_cap_becomes_a_failure():
    episodes = [{"task_id": 0, "success": 1, "steps_taken": 30}]
    out = truncated_success(episodes, step_cap=25)
    assert out[0]["success"] == 0


def test_success_exactly_at_cap_stays_a_success():
    episodes = [{"task_id": 0, "success": 1, "steps_taken": 25}]
    out = truncated_success(episodes, step_cap=25)
    assert out[0]["success"] == 1


def test_original_failure_stays_a_failure_regardless_of_steps_taken():
    episodes = [{"task_id": 0, "success": 0, "steps_taken": 5}]
    out = truncated_success(episodes, step_cap=25)
    assert out[0]["success"] == 0


def test_does_not_mutate_the_input_list():
    episodes = [{"task_id": 0, "success": 1, "steps_taken": 30}]
    truncated_success(episodes, step_cap=25)
    assert episodes[0]["success"] == 1  # original untouched


def test_other_fields_pass_through_unchanged():
    episodes = [{"task_id": 0, "success": 1, "steps_taken": 10, "task_type": "pick_and_place_simple", "candidate_ids": ["a", "b"]}]
    out = truncated_success(episodes, step_cap=25)
    assert out[0]["task_type"] == "pick_and_place_simple"
    assert out[0]["candidate_ids"] == ["a", "b"]


def test_rejects_invalid_step_cap():
    with pytest.raises(ValueError):
        truncated_success([{"success": 1, "steps_taken": 1}], step_cap=0)


def test_works_on_a_realistic_batch_of_episodes():
    episodes = [
        {"task_id": 0, "success": 1, "steps_taken": 6},
        {"task_id": 1, "success": 0, "steps_taken": 50},
        {"task_id": 2, "success": 1, "steps_taken": 46},
        {"task_id": 3, "success": 1, "steps_taken": 4},
    ]
    out25 = truncated_success(episodes, step_cap=25)
    assert [e["success"] for e in out25] == [1, 0, 0, 1]  # task_id 2's 46-step success is truncated away

    out30 = truncated_success(episodes, step_cap=30)
    assert [e["success"] for e in out30] == [1, 0, 0, 1]  # still truncated at 30 (46 > 30)
