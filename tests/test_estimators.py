import numpy as np
import pytest

from memory_ope import config as config_mod
from memory_ope.estimators import doubly_robust, ips, memory_worth
from memory_ope.evaluation import metrics
from memory_ope.simulator import task_difficulty


def test_memory_worth_hand_example():
    episodes = [
        {"candidate_ids": ["m"], "propensities": {"m": 0.5}, "included": {"m": 1}, "success": 1},
        {"candidate_ids": ["m"], "propensities": {"m": 0.5}, "included": {"m": 1}, "success": 0},
        {"candidate_ids": ["m"], "propensities": {"m": 0.5}, "included": {"m": 0}, "success": 1},
    ]
    mw = memory_worth.compute(episodes)
    assert mw["m"] == pytest.approx(0.5)  # 1 success, 1 failure among included episodes


def test_memory_worth_defaults_to_half_with_no_inclusion_data():
    episodes = [
        {"candidate_ids": ["m"], "propensities": {"m": 0.5}, "included": {"m": 0}, "success": 1},
    ]
    mw = memory_worth.compute(episodes)
    assert mw["m"] == pytest.approx(0.5)


def test_snips_matches_ips_when_all_propensities_equal():
    rng = np.random.default_rng(42)
    episodes = []
    for _ in range(5000):
        z = int(rng.random() < 0.5)
        y = int(rng.random() < (0.6 if z else 0.4))
        episodes.append(
            {"candidate_ids": ["m"], "propensities": {"m": 0.5}, "included": {"m": z}, "success": y}
        )
    ips_est, snips_est = ips.compute(episodes)
    # With constant propensity 0.5, SNIPS's per-arm normalization reduces to
    # dividing by (count * 0.5)/0.5 = count, i.e. it should match IPS closely.
    assert ips_est["m"] == pytest.approx(snips_est["m"], abs=1e-9)


def test_doubly_robust_close_to_ips_on_confounded_toy_data():
    rng = np.random.default_rng(7)
    episodes = []
    for _ in range(8000):
        task_type = "hard" if rng.random() < 0.5 else "easy"
        p = 0.8 if task_type == "hard" else 0.2
        z = int(rng.random() < p)
        base = 0.6 if task_type == "hard" else 0.3
        y = int(rng.random() < min(max(base + 0.15 * z, 0.0), 1.0))
        episodes.append(
            {
                "candidate_ids": ["m"],
                "task_type": task_type,
                "propensities": {"m": p},
                "included": {"m": z},
                "success": y,
            }
        )
    ips_est, _snips_est = ips.compute(episodes)
    dr_est = doubly_robust.compute(episodes, ["m"], l2_c=1.0)
    assert abs(ips_est["m"] - dr_est["m"]) < 0.05


def test_ips_and_dr_beat_memory_worth_on_task_difficulty_confounding():
    """Regression test for the core research claim: under the task-difficulty
    confound, Memory Worth's rank correlation with true value is markedly
    worse than IPS/doubly-robust's."""
    cfg = config_mod.load_config()
    true_vals = task_difficulty.true_values(cfg)
    episodes = task_difficulty.generate_episodes(cfg, seed=0)

    mw = memory_worth.compute(episodes)
    ips_est, _snips_est = ips.compute(episodes)
    dr_est = doubly_robust.compute(episodes, task_difficulty.all_ids(cfg), l2_c=cfg["estimators"]["dr_l2_C"])

    corr_mw = metrics.spearman(mw, true_vals)
    corr_ips = metrics.spearman(ips_est, true_vals)
    corr_dr = metrics.spearman(dr_est, true_vals)

    assert corr_ips > corr_mw
    assert corr_dr > corr_mw
