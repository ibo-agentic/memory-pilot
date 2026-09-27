"""Tests for weighted_task_source.WeightedRealTaskSource -- in particular
that a configured task_type_weights dict actually produces a draw mix close
to the intended proportions over many task_ids (Task B, 2026-09-28: verifies
the mechanism kaggle_config.yaml's env.task_type_weights relies on before
trusting any pooled-success-rate estimate built on top of it).

Uses a monkeypatched list_real_game_files (fake gamefile path strings, one
per task type, enough for RealAlfredEnv.task_type_from_gamefile's substring
match to work) so this test is fast and doesn't depend on a real ALFWorld
dataset being present -- WeightedRealTaskSource's actual game-selection
logic isn't what's under test here, only its task-TYPE draw distribution.
"""

from __future__ import annotations

from collections import Counter

import pytest

from alfworld_pilot import weighted_task_source
from alfworld_pilot.env_interface import TASK_TYPES
from alfworld_pilot.weighted_task_source import WeightedRealTaskSource


def _fake_game_files() -> list[str]:
    # A handful of distinctly-pathed fake files per task type -- enough for
    # task_type_from_gamefile's substring check, never actually opened.
    return [f"/fake/{tt}/{i:03d}/game.tw-pddl" for tt in TASK_TYPES for i in range(5)]


@pytest.fixture
def patched_game_files(monkeypatch):
    monkeypatch.setattr(weighted_task_source, "list_real_game_files", lambda config, split: _fake_game_files())


KAGGLE_EASY_TASK_WEIGHTS = {
    "pick_and_place_simple": 0.40,
    "look_at_obj_in_light": 0.35,
    "pick_clean_then_place_in_recep": 0.07,
    "pick_heat_then_place_in_recep": 0.07,
    "pick_cool_then_place_in_recep": 0.06,
    "pick_two_obj_and_place": 0.05,
}


def test_probability_for_matches_normalized_weights(patched_game_files):
    source = WeightedRealTaskSource({}, split="train", task_type_weights=KAGGLE_EASY_TASK_WEIGHTS)
    total = sum(KAGGLE_EASY_TASK_WEIGHTS.values())
    for tt, w in KAGGLE_EASY_TASK_WEIGHTS.items():
        assert source.probability_for(tt) == pytest.approx(w / total)


def test_sampled_mix_matches_weights_over_many_draws(patched_game_files):
    source = WeightedRealTaskSource({}, split="train", task_type_weights=KAGGLE_EASY_TASK_WEIGHTS)
    n_draws = 20_000
    counts = Counter(source.task_type(task_id) for task_id in range(n_draws))

    total_w = sum(KAGGLE_EASY_TASK_WEIGHTS.values())
    for tt, w in KAGGLE_EASY_TASK_WEIGHTS.items():
        intended = w / total_w
        actual = counts.get(tt, 0) / n_draws
        # 20k draws gives a binomial std of well under 0.01 for every one of
        # these probabilities; 0.02 is a generous, non-flaky tolerance.
        assert actual == pytest.approx(intended, abs=0.02), (
            f"{tt}: intended={intended:.3f}, realized={actual:.3f}"
        )

    combined_easy = (counts.get("pick_and_place_simple", 0) + counts.get("look_at_obj_in_light", 0)) / n_draws
    assert 0.70 <= combined_easy <= 0.80

    combined_other = 1.0 - combined_easy
    assert 0.20 <= combined_other <= 0.30


def test_sampling_weight_is_deterministic_function_of_task_id(patched_game_files):
    source = WeightedRealTaskSource({}, split="train", task_type_weights=KAGGLE_EASY_TASK_WEIGHTS)
    for task_id in [0, 1, 2, 100, 12345]:
        w1 = source.sampling_weight(task_id)
        w2 = source.sampling_weight(task_id)
        assert w1 == w2
        assert w1 == pytest.approx(source.probability_for(source.task_type(task_id)))
