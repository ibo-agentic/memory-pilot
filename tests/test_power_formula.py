"""Verifies memory_ope.evaluation.power_formula against the two existing,
independently-written implementations already in this repo
(ground_truth_power_analysis.py's simulator-based one, and
effect_size_analysis.py's real-rho one), and documents the 2x
pairs-vs-episodes discrepancy raised while cross-checking a hand
calculation against effect_size_analysis.json.

Note on tolerance: this repo's existing scripts hardcode rounded z-values
(Z_ALPHA_2=1.96, Z_BETA=0.8416); power_formula.py computes them at full
precision via scipy.stats.norm.ppf. This produces a ~0.01% difference in
n_pairs (e.g. 1424.375 here vs. 1424.390 in effect_size_analysis.json) --
immaterial for any planning or reporting purpose, but real, so comparisons
below use a relative tolerance rather than exact equality.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from memory_ope.evaluation import power_formula as pf

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_var_of_paired_difference_matches_effect_size_analysis_var_d():
    # From alfworld_pilot/results/effect_size_analysis.json:
    # real_var_y = 0.22404320987654322, real_var_d = 0.1633275, mean_real_rho = 0.6355
    p = 0.6611111111111111
    rho = 0.6355000000000001
    assert pf.var_of_paired_difference(p, rho) == pytest.approx(0.1633275, rel=1e-6)


def test_n_pairs_needed_matches_effect_size_analysis_json():
    result_path = REPO_ROOT / "alfworld_pilot" / "results" / "effect_size_analysis.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))

    p = data["noise_vs_signal"]["overall_success_rate"]
    rho = data["ground_truth_cross_check"]["mean_real_rho"]
    assert pf.var_of_paired_difference(p, rho) == pytest.approx(data["real_var_d"], rel=1e-6)

    expected = data["resolution_table_pairs_per_memory"]
    for delta_str, expected_n_pairs in expected.items():
        delta = float(delta_str)
        got = pf.n_pairs_needed(delta, p, rho)
        assert got == pytest.approx(expected_n_pairs, rel=2e-3), (
            f"delta={delta}: power_formula.n_pairs_needed={got}, "
            f"effect_size_analysis.json={expected_n_pairs}"
        )


def test_full_30_memory_store_episodes_and_cost_match_reported_figures():
    result_path = REPO_ROOT / "alfworld_pilot" / "results" / "effect_size_analysis.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))

    p = data["noise_vs_signal"]["overall_success_rate"]
    rho = data["ground_truth_cross_check"]["mean_real_rho"]
    n_pairs_003 = pf.n_pairs_needed(0.03, p, rho)
    eps = pf.total_episodes(n_pairs_003, n_memories=30)

    expected = data["full_30_memory_store_at_delta_0.03"]
    assert eps == pytest.approx(expected["total_episodes"], rel=2e-3)

    # BLENDED_COST_PER_EPISODE from effect_size_analysis.py, averaged over
    # the 6 real per-task-type costs measured from small_pilot.jsonl.
    per_type_costs = [0.0192, 0.0164, 0.0128, 0.0090, 0.0057, 0.0049]
    blended_cost = sum(per_type_costs) / len(per_type_costs)
    cost = pf.cost_usd(eps, blended_cost)
    assert cost == pytest.approx(expected["cost_usd"], rel=2e-3)


def test_ground_truth_power_analysis_reproduces_original_10_memory_plan():
    # results/ground_truth_power_analysis.json's chosen plan: n_memories=10,
    # ~100 pairs/memory, empirical simulator rho=0.15360170223240582,
    # var_y1_mean=0.23732377213042205 ~= var_y0_mean=0.23446407573350928.
    # power_formula assumes a single shared p; approximate p via
    # var = p(1-p) => p = 0.5 - sqrt(0.25 - var) (the smaller root, matching
    # this project's p<0.5-ish "hard task" base rates in that simulator).
    result_path = REPO_ROOT / "results" / "ground_truth_power_analysis.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))
    rho = data["empirical_rho_stats_from_task_difficulty_simulator"]["rho_mean"]
    var_y = (
        data["empirical_rho_stats_from_task_difficulty_simulator"]["var_y1_mean"]
        + data["empirical_rho_stats_from_task_difficulty_simulator"]["var_y0_mean"]
    ) / 2
    p = 0.5 - (0.25 - var_y) ** 0.5

    row = next(r for r in data["tradeoff_table_within_budget"]["rows"] if r["n_memories"] == 10)
    # Solve n_pairs -> delta (MDE) the same way ground_truth_power_analysis._mde does:
    # delta = sqrt(power_const * var_d / n_pairs)
    import math

    from scipy.stats import norm

    z_a, z_b = norm.ppf(1 - 0.05 / 2), norm.ppf(0.80)
    var_d = pf.var_of_paired_difference(p, rho)
    mde = math.sqrt((z_a + z_b) ** 2 * var_d / row["pairs_per_memory"])
    assert mde == pytest.approx(row["mde_80pct_power"], rel=5e-2)


def test_doubled_incorrect_formula_reproduces_the_hand_calc_discrepancy():
    """Documents WHY the discrepancy exists: reproduces the user's ~2849
    pairs / ~170,927 episode hand calculation exactly via the WRONG
    (two-independent-sample) formula, confirming the 2x gap is fully
    explained by an extra leading factor of 2 that does not belong in a
    paired one-sample design -- not by a pairs-vs-episodes mislabeling."""
    p = 0.6611111111111111
    rho = 0.6355000000000001

    correct_n_pairs = pf.n_pairs_needed(0.03, p, rho)
    wrong_n_pairs = pf._n_pairs_needed_doubled_INCORRECT(0.03, p, rho)

    assert wrong_n_pairs == pytest.approx(2 * correct_n_pairs, rel=1e-9)
    assert wrong_n_pairs == pytest.approx(2849, rel=2e-3)

    correct_episodes = pf.total_episodes(correct_n_pairs, n_memories=30)
    wrong_episodes = pf.total_episodes(wrong_n_pairs, n_memories=30)
    assert correct_episodes == pytest.approx(85463, rel=2e-3)
    assert wrong_episodes == pytest.approx(170927, rel=2e-3)
