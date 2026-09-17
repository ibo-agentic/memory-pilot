"""Core requested test: IPS recovers the true causal effect of a memory when
the logged inclusion propensities are correct, even under confounding that
would bias the naive "success rate when included" estimate.

Toy DGP: a binary context c in {0,1} (P(c=1)=0.5) that drives both the
memory's known inclusion propensity and the task's base success rate -- the
same confounding shape used in the real task-difficulty simulator, reduced
to one memory and one context variable so the true effect is exact and
closed-form.

  p(c)    = 0.2 if c == 0 else 0.8      (known propensity, logged exactly)
  base(c) = 0.3 if c == 0 else 0.6
  Y ~ Bernoulli(base(c) + EFFECT * Z)   where Z ~ Bernoulli(p(c))

True causal effect E[Y(1)] - E[Y(0)] = EFFECT exactly (additive, so the
context marginalizes out). The naive P(Y=1 | Z=1) is biased upward because
c == 1 (higher base rate) is over-represented among Z == 1 episodes.
"""

from __future__ import annotations

import numpy as np

from memory_ope.estimators import ips, memory_worth

EFFECT = 0.15
BASE = {0: 0.3, 1: 0.6}
PROPENSITY = {0: 0.2, 1: 0.8}
MEM_ID = "m"


def _generate_episodes(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(n):
        c = int(rng.random() < 0.5)
        p = PROPENSITY[c]
        z = int(rng.random() < p)
        y_prob = BASE[c] + EFFECT * z
        y = int(rng.random() < y_prob)
        episodes.append(
            {
                "candidate_ids": [MEM_ID],
                "propensities": {MEM_ID: p},
                "included": {MEM_ID: z},
                "success": y,
            }
        )
    return episodes


def test_ips_unbiased_under_confounding():
    n_per_seed = 20000
    n_seeds = 30
    ips_estimates = []
    snips_estimates = []
    for seed in range(n_seeds):
        episodes = _generate_episodes(n_per_seed, seed)
        ips_est, snips_est = ips.compute(episodes)
        ips_estimates.append(ips_est[MEM_ID])
        snips_estimates.append(snips_est[MEM_ID])

    mean_ips = float(np.mean(ips_estimates))
    mean_snips = float(np.mean(snips_estimates))

    # Averaged over many seeds, IPS/SNIPS should land within a tight band of
    # the exact true effect -- this is the unbiasedness property under
    # correct, known propensities.
    assert abs(mean_ips - EFFECT) < 0.01, f"IPS mean {mean_ips} deviates from true effect {EFFECT}"
    assert abs(mean_snips - EFFECT) < 0.01, f"SNIPS mean {mean_snips} deviates from true effect {EFFECT}"

    # Per-seed variance should shrink as n grows (sanity check, not a tight bound).
    assert np.std(ips_estimates) < 0.05


def test_memory_worth_is_not_the_causal_effect_under_this_confounding():
    """Memory Worth estimates P(Y=1 | Z=1), a different, confounded quantity
    from the causal contrast -- it should NOT land near EFFECT."""
    episodes = _generate_episodes(50000, seed=123)
    mw = memory_worth.compute(episodes)[MEM_ID]
    assert abs(mw - EFFECT) > 0.1
