import numpy as np
import pytest

from memory_ope import retrieval
from memory_ope.simulator import _common


def test_top_m_selects_highest_similarity():
    sims = {"a": 0.9, "b": 0.1, "c": 0.5, "d": 0.7}
    assert retrieval.select_top_m(sims, 2) == ["a", "d"]


def test_propensities_within_bounds_and_monotonic_in_rank():
    ranked = ["a", "b", "c", "d", "e"]
    props = retrieval.assign_propensities(ranked, propensity_min=0.1, propensity_max=0.9)
    values = [props[m] for m in ranked]
    assert values[0] == pytest.approx(0.9)
    assert values[-1] == pytest.approx(0.1)
    assert all(0.1 - 1e-9 <= v <= 0.9 + 1e-9 for v in values)
    assert values == sorted(values, reverse=True)


def test_empirical_inclusion_rate_matches_propensity():
    rng = np.random.default_rng(0)
    p = 0.35
    n = 20000
    propensity = np.full((n, 1), p)
    included = _common.draw_inclusion(propensity, rng)
    empirical_rate = included.mean()
    assert abs(empirical_rate - p) < 0.01


def test_retrieve_end_to_end_propensities_in_config_bounds():
    rng = np.random.default_rng(1)
    sims = {f"m{i}": rng.random() for i in range(30)}
    candidate_ids, propensities, included = retrieval.retrieve(
        sims, m=10, propensity_min=0.1, propensity_max=0.9, rng=rng
    )
    assert len(candidate_ids) == 10
    assert set(propensities.keys()) == set(candidate_ids)
    assert set(included.keys()) == set(candidate_ids)
    assert all(0.1 - 1e-9 <= p <= 0.9 + 1e-9 for p in propensities.values())
    assert all(z in (0, 1) for z in included.values())
