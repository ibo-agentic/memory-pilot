"""General sample-size / episode-count / cost formula for detecting a
per-memory causal effect with paired forced-in/forced-out episodes.

This is the one-sample paired-difference normal-approximation formula
already implemented ad hoc in two places in this repo
(`memory_ope.evaluation.ground_truth_power_analysis` and
`alfworld_pilot.effect_size_analysis`) -- this module is the reusable,
general-purpose version of the same math, parameterized directly by
(delta, p, rho) as requested for the paper, and verified against both
existing call sites in tests/test_power_formula.py.

DERIVATION (see results/power_formula_derivation.tex for the LaTeX writeup)

Design: for a given memory, run n paired episodes. Pair i forces the memory
IN for one episode (outcome Y1_i) and OUT for the other (outcome Y0_i),
holding the task instance, environment seed, and every other memory's
inclusion identical between the two arms of the pair -- this is exactly
`ground_truth_runner`'s paired design (Stage 1's `_common.paired_design_stats`
and Stage 2's `ground_truth_runner.paired_correlation`).

Let D_i = Y1_i - Y0_i. E[D_i] = delta (the memory's true causal effect).
Because Y1_i, Y0_i are Bernoulli, and approximating both arms' marginal
success probability as a common p (reasonable when delta is small relative
to p(1-p), which holds throughout this project's real settings: p~0.5-0.85,
delta~0.01-0.2):

    Var(Y1) = Var(Y0) = p(1-p)
    Cov(Y1, Y0) = rho * sqrt(Var(Y1)*Var(Y0)) = rho * p(1-p)
    Var(D) = Var(Y1) + Var(Y0) - 2*Cov(Y1,Y0) = 2*p*(1-p)*(1-rho) =: sigma_d^2

By the CLT, Dbar_n ~ N(delta, sigma_d^2 / n) for n reasonably large. Testing
H0: delta=0 vs H1: delta!=0 (two-sided) or delta>0 (one-sided, e.g. the
"detect a harmful memory" framing) at level alpha with power (1-beta), the
standard ONE-SAMPLE z-test sample-size formula gives:

    n_pairs = (z_{alpha/2} + z_beta)^2 * sigma_d^2 / delta^2      [two-sided]
    n_pairs = (z_{alpha}   + z_beta)^2 * sigma_d^2 / delta^2      [one-sided]

Each pair = 2 episodes (forced-in + forced-out). For a store of n_memories
independent memories, each ground-truthed to the same n_pairs:

    total_episodes = 2 * n_memories * n_pairs

*** THE DISCREPANCY, reported honestly (raised while cross-checking a hand
calculation against effect_size_analysis.json's 85,463-episode figure) ***

A hand-derivation using n_pairs = 2*(z_a/2+z_beta)^2*sigma_d^2/delta^2 (an
EXTRA leading factor of 2 beyond what's implemented here) gives exactly 2x
this module's (and both existing scripts') n_pairs -- e.g. at
p=0.6611, rho=0.6355, delta=0.03: this formula gives ~1424 pairs/memory
(85,463 total episodes for 30 memories, matching effect_size_analysis.json
exactly); the doubled formula gives ~2849 pairs/memory (170,927 episodes).

The doubled formula is the sample-size formula for a *two-INDEPENDENT-sample*
comparison of proportions, each group of size n, with per-group variance
sigma^2: there, Var(Xbar1 - Xbar2) = sigma^2/n + sigma^2/n = 2*sigma^2/n
(two independent, uncorrelated sources of sampling variance), giving
n_per_group = 2*(z_a/2+z_beta)^2*sigma^2/delta^2.

That leading "2" is not a general constant to always include -- it exists
specifically because, in the unpaired case, sigma^2 is a PER-GROUP variance
that has not yet been combined across the two groups. In OUR paired design,
sigma_d^2 = 2*p*(1-p)*(1-rho) is already Var(D), i.e. it has ALREADY
combined both arms' variance (via the "2*p*(1-p)" term) and their
correlation (via "(1-rho)"). Applying the two-sample formula's extra
leading 2 on top of sigma_d^2 double-counts that combination step, inflating
n_pairs (and hence total_episodes and cost) by exactly 2x.

CONCLUSION: this repo's existing implementation (memory_worth of
ground_truth_power_analysis.py's `_n_pairs_needed`, and
effect_size_analysis.py's `_n_pairs`) is CORRECT -- it is the standard
one-sample paired-difference formula, which is the right test for a paired
design. effect_size_analysis.json's "85,463 episodes (~$970)" figure is
correct as reported, and it counts EPISODES (= 2 * n_memories * n_pairs),
not pairs -- pairs alone would be 30 * 1424.39 = 42,731.7 (still not what
either the correct or the doubled hand-calculation produced as a "pairs"
figure, ruling out a pairs-vs-episodes mislabeling as the source of the 2x
discrepancy). The 2x gap is fully and only explained by the extra leading
factor of 2 in the hand derivation, which belongs to a different (unpaired,
two-sample) design than the one actually used here.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def var_of_paired_difference(p: float, rho: float) -> float:
    """sigma_d^2 = Var(Y1 - Y0) for two Bernoulli(p) arms with paired
    correlation rho, approximating Var(Y1) = Var(Y0) = p(1-p) (exact when
    both arms share the same marginal success probability; a good
    approximation otherwise as long as the true effect delta is small
    relative to p(1-p), which holds throughout this project's real and
    planned settings)."""
    return 2.0 * p * (1.0 - p) * (1.0 - rho)


def n_pairs_needed(
    delta: float,
    p: float,
    rho: float,
    alpha: float = 0.05,
    power: float = 0.80,
    one_sided: bool = False,
) -> float:
    """Pairs per memory needed for `power` (default 80%) to detect a true
    paired-difference effect of `delta`, at significance level `alpha`
    (default two-sided 0.05), given a common per-arm success probability
    `p` and paired-outcome correlation `rho`.

    Matches (and is verified against, in tests/test_power_formula.py)
    ground_truth_power_analysis.py's `_n_pairs_needed` (which takes
    var_y1/var_y0 separately rather than a single p) and
    effect_size_analysis.py's `_n_pairs`.
    """
    z_a = norm.ppf(1.0 - alpha) if one_sided else norm.ppf(1.0 - alpha / 2.0)
    z_b = norm.ppf(power)
    sigma_d2 = var_of_paired_difference(p, rho)
    return (z_a + z_b) ** 2 * sigma_d2 / (delta ** 2)


def _n_pairs_needed_doubled_INCORRECT(
    delta: float, p: float, rho: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """The two-independent-sample formula, included ONLY to document and
    reproduce the 2x discrepancy described in this module's docstring --
    NOT the right formula for this project's paired design. Do not use this
    for any real planning number; it exists so tests/test_power_formula.py
    can assert it reproduces the erroneous ~2849-pairs/~170,927-episode
    hand-calculation exactly, confirming the source of the discrepancy."""
    z_a = norm.ppf(1.0 - alpha / 2.0)
    z_b = norm.ppf(power)
    sigma_d2 = var_of_paired_difference(p, rho)
    return 2.0 * (z_a + z_b) ** 2 * sigma_d2 / (delta ** 2)


def total_episodes(n_pairs: float, n_memories: float) -> float:
    """Each pair = one forced-in + one forced-out episode; n_memories
    independent memories, each ground-truthed to n_pairs pairs."""
    return 2.0 * n_memories * n_pairs


def cost_usd(total_episodes_: float, cost_per_episode: float) -> float:
    return total_episodes_ * cost_per_episode


def episodes_and_cost(
    delta: float,
    n_memories: float,
    rho: float,
    p: float,
    cost_per_episode: float,
    alpha: float = 0.05,
    power: float = 0.80,
    one_sided: bool = False,
) -> dict:
    """Convenience wrapper: n_pairs -> total_episodes -> cost, in one call,
    for building grids (see resolution_grid.py)."""
    n_pairs = n_pairs_needed(delta, p, rho, alpha=alpha, power=power, one_sided=one_sided)
    eps = total_episodes(n_pairs, n_memories)
    return {
        "delta": delta,
        "n_memories": n_memories,
        "rho": rho,
        "p": p,
        "n_pairs_per_memory": n_pairs,
        "total_episodes": eps,
        "cost_usd": cost_usd(eps, cost_per_episode),
    }
