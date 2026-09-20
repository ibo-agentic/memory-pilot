# memory-ope-pilot

## Goal

Test whether off-policy evaluation (OPE) can estimate the causal value of
each stored memory in an LLM agent from ordinary logs, better than the
"Memory Worth" baseline from Şimşek 2026, arXiv 2604.12007 ("When to Forget:
A Memory Governance Primitive").

## Value definition (causal)

For memory m: `value(m) = E[success | m included] - E[success | m not included]`,
averaged over episodes where m was a retrieval candidate. Because inclusion
is an independent Bernoulli draw with a **known** propensity, this is the
causal contrast E[Y(1)] - E[Y(0)], not just an observational difference in
means — that distinction is the whole point of comparing IPS/DR against
Memory Worth.

## Retrieval design

- Top-M candidates per task by embedding similarity (M = 10).
- Each candidate is included independently with a known propensity p_i in
  [0.1, 0.9] (more similar -> higher p_i; linear in similarity rank within
  the top-M).
- Every episode logs: task id, candidate ids, p_i per candidate, which were
  included, and success (0/1). Logs are JSONL, one line per episode.

## Estimators

1. **Memory Worth** (baseline, from the paper): `hits+(m) / (hits+(m) + hits-(m))`
   — success rate over episodes where m was included. Defaults to 0.5 with
   no data. This is the *naive* / observational estimate — biased whenever
   propensity correlates with a confounder that also drives success.
2. **IPS**: mean over candidate episodes of `Z*Y/p - (1-Z)*Y/(1-p)`.
3. **SNIPS**: same but each arm's weight normalized by its own sum of
   importance weights instead of by episode count.
4. **Doubly robust**: IPS term plus a correction from a shared logistic
   regression outcome model fit on task features + the full inclusion-
   indicator vector, queried with each memory's indicator forced to 1 and to
   0 (holding everything else at its observed value).

## Source-of-truth for the paper

Verified by fetching arXiv 2604.12007 directly (April 2026, author Baris
Simsek). Key facts that shaped this pilot's design:
- MW's actual formula matches estimator #1 above exactly.
- The paper never tests IPS/SNIPS/DR — only MW vs. a "no-update" baseline
  and an oracle-weighted variant. This pilot is genuinely new work, not a
  reproduction.
- Its two synthetic setups (task-difficulty confounding: 70 generalists
  utility~Uniform[0,1] + 30 specialists utility=0.85, easy tasks base
  success 0.78 draw only generalists, hard tasks base success 0.28 mix both;
  hitchhiker: anchor utility=0.90 + hitchhiker utility=0.05 co-retrieved
  except an independent-retrieval fraction, ~30% independence needed for MW
  to separate them) are reproduced as `src/memory_ope/simulator/task_difficulty.py`
  and `hitchhiker.py`.
- No ALFWorld or real-LLM-agent experiment exists in the paper — Stage 2 is
  new empirical work, not something to match against a paper result.

## Repo layout

- `pyproject.toml` — makes `memory_ope` a `pip install -e .` package (added
  2026-09-18, under WSL2 Ubuntu); `alfworld_pilot/retrieval_shared.py`
  imports it directly now, no cross-venv `sys.path` shim.
- `config/config.yaml` — all tunables for both stages.
- `src/memory_ope/simulator/` — Stage 1 synthetic DGPs + Monte Carlo oracle
  for true per-memory causal values (forced-in/forced-out on the natural
  randomized distribution of everything else — the same logic Stage 2 will
  use for its real ground truth).
- `src/memory_ope/estimators/` — the four estimators above.
- `src/memory_ope/evaluation/` — Spearman correlation / bias / variance
  comparison against the oracle, across 20 seeds; writes
  `results/stage1_report.json` + plots.
- `alfworld_pilot/` — Stage 2, real ALFWorld installed and running as of
  2026-09-18 (see that section below and `alfworld_pilot/README.md` for the
  install path, a real textworld/Python-3.13+ incompatibility found and
  patched, and two real bugs in this package's own env-wiring code found
  once ALFWorld was actually exercised).
- `tests/` — includes an explicit check that IPS is unbiased for the true
  causal contrast under confounding when propensities are correct
  (`test_ips_unbiased.py`).

## Stage 1 result (as of the last `run_eval` run)

- **task_difficulty**: Memory Worth's Spearman correlation with true value
  is *negative* (~-0.29), replicating the paper's own reported failure mode
  (ρ≈-0.33). IPS/SNIPS/DR are all positive (0.30-0.45), DR best.
- **hitchhiker**: Memory Worth systematically undershoots the true
  anchor-vs-hitchhiker gap (especially at low independent-retrieval
  fractions), while IPS/SNIPS/DR track the true gap closely and roughly
  independent of that fraction. On pool-wide Spearman across all 32
  memories (dominated by 30 unconfounded filler memories), MW's lower
  variance actually gives it a *higher* pool-wide correlation than IPS —
  the real story here is the anchor/hitchhiker-specific gap, not the
  pool-wide ranking. See "Design review" below for the scale-free version
  of this comparison.

Re-run with `python -m memory_ope.simulator.run_simulation` then
`python -m memory_ope.evaluation.run_eval` (see README.md) — regenerate this
section if those numbers change materially.

## Design review (2026-09-17): true-value definition and deeper checks

Before scoping Stage 2, ran a review pass covering: how "true value" is
actually computed, a large-N sanity check, an episode-count sweep, a
propensity-extremity sweep with effective sample size, scale-free hitchhiker
metrics, and doubly-robust cross-fitting. All scripts live in
`src/memory_ope/evaluation/`, all outputs in `results/`.

### 1. True value definition (`true_value_check.py` -> `results/true_value_check.json`)

Two candidate ground-truth definitions exist and are NOT the same quantity
under confounding:

- **Causal oracle** (`oracle_values` in `_common.py`): `E[Y|do(Z_m=1)] -
  E[Y|do(Z_m=0)]`, forcing inclusion while everything else varies at its
  natural randomized rate. This is what IPS/SNIPS/DR are mathematically
  consistent for under known, correct propensities.
- **Naive observational** (`naive_observational_values`): `E[Y|Z_m=1] -
  E[Y|Z_m=0]`, splitting candidate episodes by their *factual* inclusion
  draw, no forcing. This is what a population-scale Memory-Worth-style
  difference would converge to, and it is itself confounded (that's the
  whole premise of the task-difficulty simulator).

Measured divergence (task_difficulty): all 30 specialists share the same
utility (0.85) and should therefore have near-identical true value; under
the causal oracle they do (std 0.00002), but under the naive definition
they don't (std 0.00097, i.e. real confounding-induced spread across
"identical" memories). Pool-wide, `spearman(causal, naive) = 0.946` for
task_difficulty and `0.999` for hitchhiker — close but not 1.0.

**Decision (confirmed with the user): the causal oracle is the standard
ground truth** for every check below and for Stage 2's forced-in/forced-out
ground truth, since it's what the estimators actually target and what the
pilot's stated goal ("causal value") means. `true_values()` was already
using the causal oracle, not U* — the naive version was added alongside it
for this comparison, not as a replacement.

### 2. Large-N sanity check (`large_n_sanity.py` -> `results/large_n_sanity.json`)

At n=200,000 episodes (1 seed), Spearman vs. the causal oracle:

| | memory_worth | ips | snips | doubly_robust |
|---|---|---|---|---|
| task_difficulty | -0.177 | 0.838 | 0.915 | 0.920 |
| hitchhiker | 0.987 | 0.945 | 0.982 | 0.980 |

IPS/SNIPS/DR climb toward 1 as expected (SNIPS/DR clearly ahead of plain
IPS — self-normalization and the outcome-model correction both help).
They don't hit exactly 1.0 because many generalist/filler memories have
utilities close to 0.5 (true causal value near 0), so their relative rank
order is inherently unstable at any finite n — this is a property of the
DGP's near-ties, not an estimator bug.

### 3. Episode-count sweep (`episode_sweep.py` -> `results/episode_sweep.json`, `episode_sweep.png`)

n in {500, 1000, 2000, 5000, 10000, 50000}, 20 seeds each, mean + 95th/2.5th
seed-percentile band, vs. causal oracle:

- **task_difficulty**: memory_worth is negative at every n tested (-0.38 at
  n=500 to -0.21 at n=50000). IPS/SNIPS/DR are positive from n=500 onward
  and climb steadily (IPS: 0.09 -> 0.62; DR: 0.11 -> 0.76). **Overtake point
  is n=500 (the smallest tested) for all three** — because memory_worth is
  confounded in the wrong *direction* here, not just noisier.
- **hitchhiker**: memory_worth's pool-wide Spearman is *higher* than
  IPS/SNIPS/DR at every n tested, growing from 0.33 (n=500) to 0.96
  (n=50000) — it never gets overtaken pool-wide in this range. This is the
  same "MW wins on the unconfounded majority of the pool" effect noted
  above; see check 5 for why pool-wide Spearman is the wrong lens for the
  hitchhiker pair specifically.

### 4. Randomness (propensity extremity) sweep + effective sample size (`randomness_sweep.py`)

Clarifying the terminology first (this was genuinely ambiguous): "inclusion
independence" (each candidate's Bernoulli draw is independent of every
other candidate's, always true by construction) is different from
hitchhiker's "independent_retrieval_fraction" (whether the hitchhiker's
*similarity score*, hence candidacy, tracks the anchor's), which is
different again from *this* sweep's axis: how far propensity_min/max sit
from 0.5.

Swept `(propensity_min, propensity_max)` from `[0.45,0.55]` (near-coin-flip)
to `[0.02,0.98]` (near-deterministic), task_difficulty simulator, default
n=10000, 20 seeds:

| range | mean ESS ratio | memory_worth | ips | snips | doubly_robust |
|---|---|---|---|---|---|
| [0.45,0.55] | 0.996 | -0.295 | 0.398 | 0.527 | 0.558 |
| [0.3,0.7] | 0.932 | -0.297 | 0.368 | 0.502 | 0.526 |
| [0.1,0.9] (default) | 0.652 | -0.292 | 0.301 | 0.423 | 0.450 |
| [0.05,0.95] | 0.487 | -0.294 | 0.248 | 0.363 | 0.393 |
| [0.02,0.98] | 0.340 | -0.292 | 0.193 | 0.308 | 0.324 |

Textbook behavior: effective sample size (Kish ESS ratio, averaged per
memory) drops sharply as propensities approach 0/1, and IPS/SNIPS/DR
accuracy degrades monotonically with it. Memory Worth is flat across this
sweep (it doesn't use propensities at all, so it's insensitive to this axis
— but also never benefits from more-random retrieval the way OPE does).

### 5. Fair hitchhiker metric (`run_eval.py`'s `hitchhiker_sweep`, scale-free now)

Replaced the raw "gap" comparison (misleading — memory_worth's values live
on a ~0.5 scale, IPS/SNIPS/DR's on a ~0.04 scale, so gap size isn't
comparable across estimators) with two scale-free metrics, swept over
`independent_retrieval_fraction`:

- **(a) fraction of seeds where anchor is ranked above hitchhiker**: at
  n=10000/20 seeds, memory_worth, snips, and doubly_robust all get this
  right 100% of the time at every fraction tested. Plain IPS drops to 95%
  at fractions 0.1-0.2 (occasional seed-level rank flips from its higher
  variance) — the one place raw IPS is measurably less reliable than the
  variance-reduced variants here.
- **(b) pool-wide Spearman vs. causal oracle**: confirms the check-3
  finding — memory_worth (~0.85) > snips/doubly_robust (~0.75) > ips
  (~0.63), flat across the fraction sweep.

Net: memory_worth reliably gets the *direction* right for this specific
pair even under heavy co-retrieval confounding, it just compresses the
*magnitude* of the gap it estimates (see the reference-only gap numbers
still saved in `stage1_report.json`). The pilot's original framing (MW
"fails to separate" anchor and hitchhiker) is directionally right about
magnitude but was not true of rank-order at this episode count — worth
keeping in mind when writing up the pilot's conclusions.

### 6. Doubly robust cross-fitting (`doubly_robust.py`, `dr_cross_fit_check.py`)

`doubly_robust.compute()` now cross-fits by default (`cross_fit=True`,
K=5 `KFold`): the outcome model for each episode is only ever evaluated
out-of-fold. `cross_fit=False` keeps the old fit-and-evaluate-on-everything
behavior, kept only for this comparison.

| | task_difficulty spearman | hitchhiker spearman |
|---|---|---|
| cross_fit=True | 0.450 | 0.750 |
| cross_fit=False | 0.451 | 0.752 |

No material difference in this regime (regularized logistic regression,
C=1.0, ample data relative to feature count) — cross-fitting is now the
default because it's the methodologically correct choice going forward
(matters more with a more flexible outcome model or less data), not because
it changed today's numbers.

### 7'. Randomization ablation (2026-09-17 follow-up): what does randomization alone fix?

(`randomization_ablation.py` -> `results/randomization_ablation.json`)

Isolates two things that were previously conflated: does merely having
*some* randomized inclusion (independent of whether it's propensity-
corrected) change Memory Worth's accuracy, versus does the IPS/DR
propensity correction fix it. Two policies, same top-M candidate selection
in both: (a) deterministic top-k (every candidate deterministically
included -- implemented as the existing retrieval code with
`propensity_min = propensity_max = 1.0`, a degenerate case of the same
mechanism, not new code) vs (b) our randomized retrieval. Memory Worth
computed under both (evaluated against each regime's own causal oracle,
since the oracle itself shifts slightly with the retrieval regime); IPS/
SNIPS/DR only under (b) -- under (a), "not included while candidate" never
occurs, so their importance weight `1/(1-p)` is `1/0`, which is not a bug to
route around, it's the point (a fixed top-k policy has no counterfactual
data to correct with). Hitchhiker's `independent_retrieval_fraction` held
at 0.0 in both arms (Simsek's 0%-independence case) so only the
deterministic-vs-randomized axis varies.

| | memory_worth (top-k) | memory_worth (randomized) | ips | snips | doubly_robust |
|---|---|---|---|---|---|
| task_difficulty | -0.242 | -0.292 | 0.301 | 0.423 | 0.450 |
| hitchhiker | 0.840 | 0.850 | 0.633 | 0.758 | 0.756 |

**Finding, corrected 2026-09-17 (see Part A.1 below for the sharper version
that changes this): pool-wide Spearman makes randomization look like it
does almost nothing (-0.24 vs -0.29; 0.84 vs 0.85, within noise). But the
anchor-vs-hitchhiker PAIR metric tells a different, more informative story
— randomizing inclusion alone (with no propensity correction at all) does
meaningfully help specifically for the hitchhiker-style co-retrieval
confound, just not for the task-difficulty-style candidacy confound.** See
"Part A.1" for the numbers and the mechanistic reason why these two
confound types respond differently to randomization.

### 7''. Small-data regime, n in {250, 500, 1000, 2000} (2026-09-17 follow-up)

(`small_data_results.py` -> `results/small_data_results.json`, task_difficulty
only, 20 seeds, mean [95% seed-percentile interval])

| n | metric | memory_worth | ips | snips | doubly_robust |
|---|---|---|---|---|---|
| 250 | spearman | -0.31 [-0.44,-0.18] | 0.05 [-0.21,0.20] | 0.08 [-0.12,0.28] | 0.08 [-0.13,0.31] |
| 250 | MAE | 0.50 | 0.29 | 0.20 | 0.21 |
| 250 | bias | 0.50 | -0.02 | -0.02 | -0.02 |
| 1000 | spearman | -0.43 [-0.54,-0.36] | 0.11 [-0.07,0.30] | 0.17 [0.02,0.31] | 0.17 [0.03,0.34] |
| 1000 | MAE | 0.51 | 0.14 | 0.10 | 0.10 |
| 1000 | bias | 0.51 | 0.00 | 0.00 | 0.00 |
| 2000 | spearman | -0.42 [-0.50,-0.36] | 0.15 [-0.02,0.33] | 0.24 [0.14,0.37] | 0.24 [0.12,0.41] |
| 2000 | MAE | 0.51 | 0.10 | 0.07 | 0.07 |
| 2000 | bias | 0.51 | 0.00 | 0.00 | 0.00 |

Key nuance for Stage 2 planning: **IPS/SNIPS/DR's bias is already ~0 at
n=250** (unbiasedness kicks in immediately, as it should since it doesn't
depend on sample size) — but their **Spearman correlation is too noisy to
be conclusive below roughly n=1000-2000** (95% interval crosses zero at
n=250-500). MAE shrinks smoothly with n throughout. Memory Worth's MAE/bias
(~0.5) reflects it living on a different scale than the causal contrast
(P(success|included) vs. a probability difference near 0), not pure
inaccuracy — but its Spearman is negative and stable-negative at every n
tested, which IS a real ranking failure, not a scale artifact.
**Implication for Stage 2**: don't expect a clean Spearman signal below
~1000 logged episodes; MAE/bias will look reasonable much earlier but
Spearman is the metric that actually answers "does this reorder memories
correctly," so budget for it.

### Stage 2 budget estimate (2026-09-17, no API calls made)

(`stage2_budget_estimate.py` -> `results/stage2_budget_estimate.json`)

All figures below are **stated assumptions**, not measurements — re-check
once a real ALFWorld agent prompt exists. ALFWorld caps episodes at 50
steps; assumed 12 steps for a successful trajectory, full 50 for a failed
one, 50% assumed success rate for a cheap, non-fine-tuned model -> **~31
LLM calls/episode average** (one call per ReAct step; memory retrieval
itself is embedding-based, not an LLM call).

Episode plan: 50 (cold-start memory-store construction) + 1000 (main
randomized-retrieval logging — floor set by the small-data finding above;
2000 would give a cleaner Spearman signal if budget allows) + 480
(ground-truth reruns: 10 memories x 2 arms x 6 base tasks x 4 seeds) =
**1530 episodes -> ~47,000 LLM calls**.

Token cost per call is the biggest uncertainty (ReAct history grows every
step, so average prompt length across an episode depends heavily on
prompt/memory-formatting choices not yet made) — swept low/base/high
(1000/2000/4000 avg prompt tokens/call, 60 completion tokens/call) against
three cheap OpenRouter models (pricing fetched 2026-09-17, re-check at
Stage 2 time — OpenRouter pricing changes):

| scenario | prompt tok/call | deepseek-v4.1-flash ($0.15/$0.60 per M) | ling-3.0-flash-vl ($0.06/$0.18 per M) | mercury-2.5 ($0.04/$0.15 per M) |
|---|---|---|---|---|
| low | 1000 | $8.82 | $3.36 | $2.32 |
| base | 2000 | $15.94 | $6.20 | $4.22 |
| high | 4000 | $30.17 | $11.90 | $8.02 |

Even the high-end scenario on the priciest of these three models is under
$35 for the whole pilot — cost is not the binding constraint here; getting
a real agent + ALFWorld installed and a sensible memory-retrieval prompt
format is the actual work.

## Part A follow-up (2026-09-17): finishing Stage 1 checks before scoping Stage 2

### A.1 Hitchhiker pair metric in the randomization ablation

(`randomization_ablation.py` -> `results/randomization_ablation.json`, `pair_metrics`)

Extended the ablation to report the anchor-vs-hitchhiker pair specifically
(not just pool-wide Spearman), for both retrieval regimes:

| estimator | fraction seeds anchor>hitchhiker | mean estimated gap | true gap |
|---|---|---|---|
| memory_worth (deterministic top-k) | 0.90 | 0.0046 | 0.0836 |
| memory_worth (randomized, uncorrected) | 1.00 | 0.0324 | 0.0841 |
| ips (randomized) | 1.00 | 0.0859 | 0.0841 |
| snips (randomized) | 1.00 | 0.0798 | 0.0841 |
| doubly_robust (randomized) | 1.00 | 0.0805 | 0.0841 |

**This corrects last round's "randomization alone does almost nothing"
claim** — that was true pool-wide but not for this pair. Under
deterministic top-k, Memory Worth reproduces Simsek's original 0%-
independence failure almost exactly: the gap collapses to 0.0046 (18x
smaller than true), and it even gets the *direction* wrong 10% of the time.
Switching to randomized inclusion — with NO propensity correction at all —
already recovers a real (if still 2.6x-too-small) gap of 0.0324 and fixes
the direction 100% of the time. Only the fully propensity-corrected
estimators (IPS/SNIPS/DR) get the magnitude right (~0.08-0.09 vs true
0.0841).

**Why this differs from task_difficulty** (where randomization alone truly
didn't help, -0.242 vs -0.292): the two confounds work through different
mechanisms. task_difficulty's confound is about *candidacy* — which
memories even show up as candidates correlates with task context, and
randomizing *inclusion given candidacy* never touches that. hitchhiker's
confound is about *co-inclusion* — anchor and hitchhiker being included in
lockstep — and randomizing inclusion directly breaks that lockstep, which
is the specific mechanism creating the confound. **Takeaway: whether
randomization alone helps depends on whether the confound lives in the
candidacy step or the inclusion step; only the propensity correction fixes
both reliably.**

### A.2 Stage 2 scale simulation

(`stage2_scale_check.py` -> `results/stage2_scale_check.json`; 50 memories
[35 generalist + 15 specialist, keeping the 70:30 ratio], M=10, n=1000,
propensity=[0.1,0.9], 20 seeds)

| estimator | spearman | MAE | bias |
|---|---|---|---|
| memory_worth | -0.399 [-0.527,-0.307] | 0.509 | 0.509 |
| ips | 0.138 [-0.104,0.395] | 0.102 | 0.000 |
| snips | 0.212 [-0.022,0.447] | 0.072 | -0.002 |
| doubly_robust | 0.214 [-0.025,0.434] | 0.071 | -0.002 |

Consistent with (slightly better than) the 100-memory/n=1000 result in the
small-data table below — fewer memories at fixed M=10 means each candidate
episode covers a larger share of the pool, mildly helping signal. Memory
Worth stays reliably negative (CI excludes zero); IPS/SNIPS/DR stay
essentially unbiased with wide but positive-mean Spearman CIs (SNIPS/DR's
low end dips just below zero, -0.02 to -0.03 — a reminder that at this
realistic Stage 2 scale, a SINGLE seed's Spearman is not yet a reliable
readout on its own; the estimator's value is in the mean pattern and the
bias/MAE numbers, not any one run's rank correlation).

### A.3 Ground-truth power analysis

(`ground_truth_power_analysis.py` -> `results/ground_truth_power_analysis.json`)

Paired design: same task instance AND environment seed for both the
forced-in and forced-out rerun of a memory, so shared context noise cancels
out of the difference. Estimated the resulting correlation `rho` between
the two arms' outcomes **empirically from the task_difficulty simulator**
(via `_common.paired_design_stats`, added this round) rather than assuming
a number: **mean rho = 0.154, range 0.010 (specialists, which only ever see
hard-task contexts, so less context variance to correlate on) to 0.217
(generalists, which see more heterogeneous contexts)**. This is a
simulator-derived proxy, not a real-ALFWorld measurement — re-estimate rho
from actual Stage 2 paired reruns once available.

Pairs needed for 80% power / two-sided alpha=0.05, by target effect:

| scenario | delta=0.05 | delta=0.10 | delta=0.15 |
|---|---|---|---|
| unpaired (rho=0) | 1481 | 370 | 165 |
| paired, empirical mean rho | 1254 | 313 | 139 |
| paired, empirical max rho | 1160 | 290 | 129 |

Pairing helps (15-22% fewer reruns needed at this rho) but modestly — the
simulator's shared-context variance is a small share of total outcome
variance. It's free (same cost either way), so always pair, but don't
expect it to be a dramatic fix on its own.

Trade-off within a 2000-episode (1000-pair) total rerun budget, empirical mean rho:

| n_memories | pairs/memory | total episodes | MDE @ 80% power |
|---|---|---|---|
| 3 | 333 | 1998 | 0.097 |
| 5 | 200 | 2000 | 0.125 |
| 8 | 125 | 2000 | 0.158 |
| **10** | **100** | **2000** | **0.177** |
| 13 | 76 | 1976 | 0.203 |
| 15 | 66 | 1980 | 0.218 |
| 20 | 50 | 2000 | 0.250 |

**Recommendation: n_memories=10, ~100 pairs/memory (200 episodes/memory,
2000 total)** — matches the original Stage 2 plan's 10-memory target,
reliably resolves effects >=0.15, gives noisier-but-informative signal at
0.10. delta=0.05 is not reliably resolvable for more than ~1 memory within
this budget at all (even paired, ~1254 pairs = 2508 episodes for ONE
memory) — that's a hard budget constraint, not a design flaw to fix.
Prioritizing memory *count* over per-memory precision is deliberate: Stage
1's small-data results (below) show Spearman itself needs enough points
and enough episodes to mean anything, so ground truth should spread across
more memories rather than nail fewer of them exactly.

**Memory selection strategy**: after the main randomized-logging phase,
don't pick ground-truth memories randomly or by top/bottom rank alone.
Choose a stratified sample across the IPS/DR-estimated value distribution
(e.g. one per quintile) **plus deliberately oversample memories where
Memory Worth and IPS/DR disagree most** (largest rank or magnitude gap) —
those are exactly the cases this pilot's hypothesis lives or dies on, so
ground truth should resolve them, not the memories every estimator already
agrees on.

## Pre-Part-B checks (2026-09-17, same day, before scoping ALFWorld build)

### Check 1: do A.2's intervals exclude zero?

Read directly from `results/stage2_scale_check.json`: **memory_worth's
95% interval excludes zero (-0.527, -0.307, reliably negative); none of
IPS/SNIPS/DR's do** (ips: [-0.104, 0.395]; snips: [-0.022, 0.447];
doubly_robust: [-0.025, 0.434]). Important caveat for Stage 2: a real
pilot only gets ONE seed's worth of data, not 20 to average over — at this
scale, a single real run could plausibly show a near-zero or even
slightly negative Spearman for IPS/SNIPS/DR purely by chance, even though
the underlying mean is positive. This motivated check 2.

### Check 2: signal-boost grid (`signal_boost_check.py` -> `results/signal_boost_check.json`)

n_episodes=1000, 20 seeds, grid over n_memories in {30, 50} x propensity
range in {[0.1,0.9] default, [0.3,0.7] closer to uniform}, plus a coarser
precision@top5/bottom5 metric (of the estimator's top/bottom 5 ranked
memories, what fraction are in the causal oracle's true top/bottom 5):

| n_memories | propensity | memory_worth spearman | doubly_robust spearman | dr prec@top5 | dr prec@bottom5 |
|---|---|---|---|---|---|
| 30 | [0.1,0.9] | -0.311 [-0.456,-0.194] | 0.226 [-0.078,0.513] | 0.23 | 0.24 |
| **30** | **[0.3,0.7]** | -0.323 [-0.481,-0.224] | **0.321 [0.148,0.566]** | 0.26 | 0.25 |
| 50 | [0.1,0.9] | -0.399 [-0.527,-0.307] | 0.214 [-0.025,0.434] | 0.19 | 0.16 |
| 50 | [0.3,0.7] | -0.404 [-0.551,-0.284] | 0.285 [-0.041,0.572] | 0.18 | 0.15 |

**Recommendation: 30 memories, propensity range [0.3, 0.7].** This is the
only cell where a propensity-corrected estimator's 95% interval excludes
zero (doubly_robust: 0.148-0.566) — i.e. the only setting where a single
real run would reliably show a positive correlation, not just the 20-seed
mean. Fewer memories helps (less multiple-comparison noise per candidate
episode); propensity closer to 0.5 helps (better effective sample size,
consistent with the earlier randomness_sweep finding). **Used this
setting (30 memories, propensity [0.3,0.7]) for Part B.**

Aside worth noting: Memory Worth's precision@bottom5 is exactly 0.00 in
*every* cell — it never identifies even one of the true worst memories.
Mechanistic reason: on easy tasks (78% base success, generalist-only
candidates), even a genuinely low-value generalist tends to co-occur with
success just from the high base rate, so MW's absolute level is dominated
by the easy/hard base-rate mixture a memory happens to land in, not its
own (much smaller) causal contribution — the signal that would let MW
find the worst memories is swamped by that base-rate noise.

### Check 3: ground-truth selection bias (`ground_truth_selection_bias_check.py`)

**Design chosen: fixed split by episode index** — first half of a log
(set A) used only to *select* which memories go into the expensive
ground-truth rerun set; second half (set B) used only to compute the
"official" estimates that get compared against that ground truth. Decided
in advance, not tuned after looking at results. (Considered a fixed a
priori rule with no data at all — e.g. "always ground-truth the 10 highest
and lowest MW-scored memories" — but that can't adapt to where MW and
IPS/DR actually disagree, which is the whole point of the pilot; the
split preserves adaptivity while removing the same-data selection bias.)

Demonstrated the effect empirically: selected the top-5 memories by
MW-vs-doubly_robust **rank** disagreement (percentile rank within each
estimator's own distribution — a raw value difference is meaningless here
since MW lives on a ~0.5 scale and DR on a ~0.03-0.06 scale) using set A,
then re-measured that disagreement on A again (in-sample) vs. B
(out-of-sample):

- in-sample (same data used to select): mean rank disagreement = 0.672
- out-of-sample (independent data): mean rank disagreement = 0.360
- **inflation ratio: 1.87x**

Confirms the concern directly: picking "biggest disagreement" memories
from a log and then re-citing that same log's disagreement as evidence
overstates it by ~87% here. The split design is necessary, not just
defensive.

### 7. IPS unbiasedness test

`tests/test_ips_unbiased.py` — a hand-built two-context confounded DGP
(context-dependent propensity AND context-dependent base rate, same shape
as task_difficulty but reduced to one memory) where the true causal effect
is exactly 0.15 by construction. Averaged over 30 seeds x 20,000 episodes,
IPS and SNIPS both land within 0.01 of 0.15; Memory Worth (which estimates
a different, confounded quantity — P(success|included), not a contrast) is
asserted to be off by more than 0.1.

## Part B (2026-09-17): Stage 2 infrastructure built, NO real API calls made

Full design, code, and a 17-test mock-only suite live in `alfworld_pilot/`
(its own README there has the details). Summary here for future sessions.

### ALFWorld install: deferred, real blocker (not invented, not worked around)

Attempted the official `pip install alfworld` directly in an isolated
Python 3.11 venv (`alfworld_pilot/.venv/`, separate from this package's 3.14
venv — ALFWorld's dependency chain is much older/riskier). It fails
building `jericho` (a TextWorld dependency needing a native C build): no
cmake/MSVC on this Windows machine, and even with one, `textworld`'s
install runs a Linux-only `setup.sh` that fetches a Linux Inform7 binary.
Neither Docker nor a WSL Linux distro was installed. **Decision (with the
user): defer real ALFWorld, build everything else against a mock env that
implements ALFWorld's exact real API** (confirmed from the official repo +
the original ReAct paper's `alfworld.ipynb`) so swapping in real ALFWorld
later (via WSL2 or Docker, user's choice when ready) needs no code changes
— only a config switch. `env_interface.RealAlfredEnv` is written, gated
behind a clear `ImportError` until alfworld is actually installed.

### Design

- **Randomized retrieval**: reused verbatim, not reimplemented —
  `alfworld_pilot/retrieval_shared.py` adds the root package's `src/` to
  `sys.path` (cross-venv, since the two packages use separate Pythons) and
  imports `memory_ope.retrieval.retrieve` directly.
- **Memories**: 30 short "lessons" (100-300 tokens, one per real ALFWorld
  task type, tagged and repeated), not full trajectories, per spec. Mock
  similarity is topic-aware (same task type -> higher base similarity +
  noise) as a stand-in for a real embedding model — swapping in real
  embeddings later doesn't require changing retrieval.py or episode_runner.py,
  only `memory_store.similarity_scores`.
- **Settings**: 30 memories, propensity range [0.3, 0.7], M=10 — per
  `signal_boost_check.py`'s recommendation (the only grid cell where a
  propensity-corrected estimator's interval excluded zero), as instructed.
- **Episode cap**: 30 steps (below ALFWorld's usual 50-step research
  norm, deliberately, to control cost) — one LLM call per step (Thought +
  Action together in a single completion, not two separate calls, so the
  Stage 2 budget estimate's "~1 call/step" assumption stays accurate).
- **LLM client**: `OpenRouterClient` (OpenAI-compatible, `openai` SDK) —
  refuses to construct with `model_id=None` or any id containing "latest";
  sends `reasoning: {enabled: false}` (OpenRouter's documented unified
  reasoning control; some models reject this with a mandatory-reasoning
  400 error — surfaced, not swallowed, so Part C can see which models do
  that); temperature=0 + an optional `seed` param when supported.
- **Caching**: every request payload (model+messages+params) hashed to a
  cache key; cache hits never touch the cost tracker.
- **Cost control**: `CostTracker.check_before_call` runs BEFORE every real
  call using a pre-call token estimate (chars/4), raising `CostCapExceeded`
  before spending rather than after; `hard_cap_usd` starts at $10 in config.
- **Determinism**: `determinism_check.check_determinism` makes two real
  duplicate calls (bypassing cache, `complete_no_cache`) and compares text
  — verifies temperature=0(+seed) actually holds rather than assuming it.
  Every ground-truth session should run this probe before trusting its
  results.
- **Ground truth (paired, per Part A.3)**: `ground_truth_runner.py` forces
  a memory in/out while holding task instance AND every other memory's
  random draws identical between arms (same derived seed for both), and —
  a bug this session's mock testing actually caught — only searches task
  instances where the target memory is a NATURAL candidate under that
  *same* seed (the candidacy probe must use the identical seed the real
  run will use, or it predicts nothing real; originally it used an
  unrelated seed and silently produced ground-truth pairs missing their
  own target memory from `candidate_ids`). `paired_correlation()` computes
  rho from real results; `power_analysis.redo_with_measured_rho()` reruns
  Part A.3's sample-size formula with it once real pairs exist (Part C).
- **Task-type logging**: every episode logs `task_type` (one of
  ALFWorld's 6 official types, extracted from `info['extra.gamefile']`'s
  path in the real backend, from the mock template directly in the mock
  backend) — enables checking whether task-type difficulty reproduces the
  task_difficulty simulator's confound in real data.

### Bugs the mock testing caught before any real spend (the whole point of testing mock-first)

1. `MockTaskTemplate.steps_to_win` was hand-specified and didn't match
   `winning_action`'s actual position in `admissible_actions` for 5 of 6
   templates — no mock episode could ever win (100% silent failure, hidden
   step-cap timeouts). Fixed by deriving it (`admissible_actions.index(...)
   + 1`) instead of hand-entering it, plus a `__post_init__` assertion so
   this class of bug fails loudly if it ever regresses.
2. The `pick_two_obj_and_place` template's `admissible_actions` list was
   literally missing its own `winning_action` string (a copy-paste typo) —
   same silent-failure symptom, caught by the same fix's assertion.
3. `ground_truth_runner`'s natural-candidacy probe used a different random
   seed (`task_seed` alone) than the actual paired run (`hash((memory_id,
   pair_index, task_seed))`) — the probe was checking a candidacy draw
   that had nothing to do with what the real run would draw. Also switched
   away from Python's built-in `hash()` for the seed material, since
   string hashing is randomized per-process by default and would silently
   break reproducibility across runs. Fixed with one shared
   `_pair_seed_material()` (stable `hashlib.md5`-based) used by both the
   probe and the real run.
4. An early test asserting "random actions shouldn't reliably win" failed
   — not a pipeline bug, but a revealing one: at the realistic 30-step cap
   with short (4-6 step) mock templates, blind random search has enough
   retries to often stumble onto the right sequence by chance. Fixed the
   *test*, not the pipeline (used a zero-slack step cap to actually
   distinguish "follows the plan" from "guesses"), and noted this as a
   real fact about the mock's difficulty at the realistic cap.

### Mock test results

17/17 tests pass (`alfworld_pilot/tests/`, run via its own Python 3.11
venv): scripted-success LLM wins every mock episode; random LLM fails
reliably only under a tight step budget (see bug 4 above); malformed-output
LLM exercises the parse-failure fallback without crashing; episode log
schema matches Stage 1's core fields (`candidate_ids`, `propensities`,
`included`, `success`) plus `task_type`; paired ground truth holds every
other memory's inclusion and full candidate set identical between forced-
in/forced-out arms; ground-truth experiment only selects task instances
where the target memory is a genuine candidate; cache/cost-tracker/
determinism-check unit tests pass; `OpenRouterClient` correctly refuses
`model_id=None` and any "latest"-aliased id without a network call.

### Token estimates (mock; real numbers come from measure_mode with a real client)

`measure_mode.py` over 5 mock episodes (23 calls, 100% success):
avg 602.9 input tokens/call, 14.7 output tokens/call, 4.6 calls/episode.
Projected for the now-larger full plan (50 store-construction + 1000 main
logging + 2000 ground-truth reruns [10 memories x 100 pairs x 2 arms, up
from the original 480 — see Part A.3's revised recommendation] = 3050
episodes): ~14,030 calls, ~8.46M input tokens, ~0.21M output tokens. These
are word counts, not a real tokenizer, and the mock's 4.6 calls/episode is
far below the ~31 calls/episode assumed in the original budget estimate
(mock templates are much shorter than real ALFWorld tasks) — **treat this
as a pipeline smoke test, not a cost projection; rerun measure_mode with a
real client once Part C picks a model** for real numbers.

`token_breakdown.py`'s per-call decomposition (word counts): system prompt
~49 tokens (fixed), retrieved memories ~340 tokens (2 memories included in
this example, 100-300 tokens each), admissible actions list ~26 tokens
(fixed per template), step history growing ~10 tokens/step in this
synthetic example (real ALFWorld observations are more verbose, so expect
faster real growth) — total prompt ~450-520 tokens across the first 6
steps in this example. This decomposition, not just a single "~2000
tokens" guess, is what should be re-measured once real ALFWorld episodes
are running.

### What's still needed before Part C

1. Exact pinned model ids + current OpenRouter prices for the 3 candidate
   models (GPT-5.6 Luna, Ling 3.0 Flash VL, Gemini 3.8 Flash) — not yet
   checked, per the plan ("only after I say go").
2. `config.yaml`'s `llm.model_id` set to one of them.
3. A real determinism-check run before trusting any real ground-truth pair.

## Part B, real ALFWorld (2026-09-18): installed and running under WSL2 Ubuntu, still no LLM API calls made

Moved from the mock-only environment described above to real ALFWorld,
under a fresh WSL2 Ubuntu venv with `build-essential` installed (the
earlier native-Windows attempt's blocker). Also made `memory_ope` a real
`pip install -e .` package (`pyproject.toml`, repo root) — `alfworld_pilot/
retrieval_shared.py` now just imports it directly, no more cross-venv
`sys.path` shim, and both packages' test suites (11 + 17 = 28 tests) run
from the one shared venv. Full details, including the exact patch and bug
fixes, are in `alfworld_pilot/README.md`; summary here:

- **`pip install alfworld`** (base install, no `[full]`/`[vis]` — those pull
  in ai2thor/torch/opencv for the embodied/visual THOR backend, unused by
  this text-only pilot) built cleanly against `textworld[pddl]` once a C
  toolchain was available. `alfworld-download` fetched ~2.3GB of game/PDDL/
  logic/detector data into `~/.cache/alfworld`.
- **Real, non-obvious blocker found and fixed**: `textworld` 1.7.0's PDDL
  grammar engine relies on a `locals().update(...); eval(...)` trick that
  Python 3.13's PEP 667 ("Consistent views of namespaces") permanently
  broke — `NameError` on the very first `env.reset()`, before any of this
  project's own code even runs. No upstream fix exists yet. Patched via a
  small, documented monkeypatch (`alfworld_pilot/src/alfworld_pilot/
  _textworld_py313_compat.py`) that passes an explicit `eval()` namespace
  instead — the PEP 667-documented migration path, applied to the one call
  site affected (confirmed via grep: `locals().update(` appears exactly
  once in the installed package).
- **Two real bugs in this project's own code**, never caught because
  `RealAlfredEnv` had never actually been run before real ALFWorld existed:
  (1) `getattr(alfworld_env, env_type)` instead of `get_environment
  (env_type)` — the class is imported locally inside `get_environment`, not
  exposed as a module attribute; (2) a **double `env.reset()` per episode**
  (once in `episode_runner.py` for `task_type`/retrieval, once again inside
  `react_agent.run_episode`) — harmless-looking on `MockAlfredEnv` in
  existing tests (each test constructs a fresh mock env right before a
  single `run_logged_episode` call) but actually meant the task memories
  were retrieved FOR never matched the task instance actually PLAYED, and
  would have silently burned two real ALFWorld games per logged episode.
  Fixed by having `react_agent.run_episode` take the already-fetched
  `obs`/`info` instead of resetting internally — exactly one `reset()` per
  episode now. Also fixed: `extra.gamefile` (source of `task_type`) is only
  populated by TextWorld on `reset()`, not on `step()` (`None` mid-episode)
  — `task_type` is now cached from `reset()` and reused.
- **`env_factory.py`** (new) selects `MockAlfredEnv`/`RealAlfredEnv` from
  `config.yaml`'s `env.backend` (now defaults to `real`; the unit test
  suite still imports `MockAlfredEnv` directly, independent of that
  default) and constructs the env ONCE for reuse across episodes via
  repeated `.reset()` calls — constructing fresh per episode would re-walk
  real ALFWorld's entire game-file dataset (thousands of directories) every
  time.
- **2 real ALFWorld episodes, `MockLLMClient`, zero API spend** (`python -m
  alfworld_pilot.measure_mode`, `env.backend: real`): both episodes ran a
  genuinely different real ALFWorld game end to end (confirmed by
  inspecting each episode's opening observation) and hit the 30-step cap
  without winning — expected and unimportant here, since `MockLLMClient`'s
  `scripted_success` strategy assumes the MOCK env's simplifying "admissible
  actions list order = winning sequence" property, which doesn't hold for
  real ALFWorld's actual (much larger, unordered) admissible-commands list.
  The point of this run was confirming the real env <-> agent <-> logging
  plumbing works end to end, which it does. Measured: **avg 1054.7 input /
  14.0 output tokens per call, 30.0 calls/episode** (both episodes ran the
  full step cap) — vs. the OLD mock-env numbers (602.9 in / 14.7 out, 4.6
  calls/episode) from Part B's original mock-only measurement: real
  ALFWorld's actual room/object descriptions are far more verbose than the
  mock's synthetic templates, and its longer, harder tasks mean more calls/
  episode before hitting the step cap. Recorded in `results/
  measure_mode_real.json` (kept alongside the original `measure_mode_mock.
  json` for comparison, not overwriting it). Updated cost projection for
  the full 3050-episode plan using these real-env numbers, same 3-model
  OpenRouter pricing snapshot as `stage2_budget_estimate.py`: **$15.24
  (deepseek-v4.1-flash) / $6.02 (ling-3.0-flash-vl) / $4.05 (mercury-2.5)**
  — still well under any reasonable budget, and these are now real-env-
  measured token counts rather than the mock-env proxy or the original
  pure-assumption budget estimate.
- **Known real limitation, NOT fixed this session**: `ground_truth_runner`'s
  paired forced-in/forced-out design needs `env.reset(task_seed=...)` to
  seek back to the SAME task instance twice. `MockAlfredEnv` supports this
  by construction; real ALFWorld's `env.reset()` ignores `task_seed`
  entirely and just advances to the next game in sequence, so real-backend
  ground-truth pairs would land on two different task instances today. This
  needs solving (likely via TextWorld's lower-level `env.load(gamefile)`
  against `AlfredTWEnv.game_files`) before Part C's real ground-truth reruns
  can run — flagged, not solved, since it was out of this session's scope
  (swap the env backend in and smoke-test it, not build ground-truth-on-
  real-ALFWorld).

## Part C prep (2026-09-19): both pre-Part-C blockers solved, plan rescaled, difficulty confound confirmed in real data

Four items requested before Part C spend. Still zero real LLM API calls.
Full details in `alfworld_pilot/README.md`; summary here.

### 1. Paired ground truth against real ALFWorld: SOLVED

The previous session's flagged limitation (`ground_truth_runner` needed
`env.reset(task_seed=...)` to seek back to a specific task instance, which
real ALFWorld doesn't support) is fixed. Verified FIRST, empirically, before
writing any pipeline code: registering one specific `game.tw-pddl` file
directly (`textworld.gym.register_games([gamefile], ...)`) gives
byte-identical resets/steps across independently-built envs and repeated
resets of the same env. `RealAlfredEnv` now takes an optional
`gamefile_path` that restricts it to exactly one game (reusing
`AlfredTWEnv`'s own real `init_env()`, not a reimplementation — skips its
expensive `collect_game_files()` walk via `AlfredTWEnv.__new__`).
`ground_truth_runner.py` now routes both backends through a small
`TaskSource` interface (`MockTaskSource`, `RealTaskSource`) instead of a
bare `env`; `RealTaskSource` maps a `task_seed` to a specific game file by
stable index, so forced-in/forced-out (and repeated candidacy probes)
agree on "the same task" by construction. Verified end to end against real
ALFWorld: identical `candidate_ids`, identical other-memory inclusion,
identical `task_type`, identical first-step observation and admissible
actions between the forced-in and forced-out arms of a real pair.

Operational finding along the way: constructing many single-game
`RealAlfredEnv`s without closing them grows memory substantially (~1.25GB
after ~100 unclosed constructions in a run that had to be killed) — not
from subprocess spawning (`asynchronous=True` is a documented no-op at
`batch_size=1`); root cause not fully tracked down, but adding
`RealAlfredEnv.close()` and calling it after every single-game episode
(`run_ground_truth_pair`, `task_type_difficulty_check.py`) keeps growth far
more modest. **Flagged for Part C**: the real ground-truth phase will
construct on the order of ~10,000 single-game envs; monitor memory and
chunk the run (process restarts every N pairs) if growth reappears despite
closing.

### 2. Python version: rebuilt on 3.11, PEP 667 patch dropped

ALFWorld's own docs only ever claim "Python 3.9+" (its quickstart pins
`python=3.9`); nothing targets 3.13+, where the previous session's
`textworld` incompatibility lives. Installed Python 3.11 via `uv python
install 3.11` (user-space, no sudo — this Ubuntu release is too new for
`apt`'s python3.10/3.11 packages, and `sudo apt-add-repository` needs an
interactive password this non-interactive session doesn't have). Rebuilt
`.venv` on it (`uv venv --python 3.11`, `ensurepip` since uv-built pythons
ship no `pip` script, then reinstalled everything). Confirmed empirically:
the same quickstart that raised `NameError` on 3.14 runs with **zero
patches** on 3.11. Deleted `_textworld_py313_compat.py` and its one call
site in `RealAlfredEnv.__init__`.

### 3. Scaled-up plan: 3050 -> 10030 episodes, new cost projection

`scale_up_check.py` (new, root package) swept n_episodes in {1000, 3000,
4000, 5000} at this pilot's actual setting (30 memories, propensity
[0.3,0.7]) on both Stage 1 simulators, 20 seeds:

| n_episodes | task_difficulty: ips 95% CI | task_difficulty: DR 95% CI | hitchhiker: ips 95% CI | hitchhiker: DR 95% CI |
|---|---|---|---|---|
| 1000 | [-0.088, 0.404] (crosses 0) | [0.148, 0.566] | [0.013, 0.534] | [0.298, 0.621] |
| 3000 | [0.145, 0.639] | [0.284, 0.758] | [0.265, 0.731] | [0.520, 0.806] |
| 4000 | [0.147, 0.678] | [0.360, 0.777] | [0.396, 0.766] | [0.505, 0.821] |
| 5000 | [0.167, 0.685] | [0.337, 0.824] | [0.220, 0.701] | [0.435, 0.813] |

**n=3000 is where plain IPS's interval stops crossing zero on
task_difficulty** (the harder of the two confounds for it); SNIPS/DR
already excluded zero from n=1000. 5000 gives the best mean Spearman of the
range tested (DR: 0.547 at 3000 -> 0.640 at 5000 on task_difficulty) at
negligible added real-measured cost, so that's what was chosen, at the top
of the requested 3000-5000 range. Full table: `results/scale_up_check.json`.

Ground-truth pairs rescaled to match: `ground_truth_power_analysis_
rescale.py` reruns the original power analysis's trade-off table (same
empirical rho=0.154) at a 5000-episode rerun budget instead of the original
2000 — **15 memories x 166 pairs/memory (4980 episodes) gives MDE~0.137 at
80% power**, better than the original 10-memory/100-pair plan's MDE~0.177
on both memory count AND resolution. `results/
ground_truth_power_analysis_rescale.json` has the full n_memories grid; the
original 2000-budget file is untouched for comparison.

New plan: 50 (store construction, unchanged) + 5000 (main logging, was
1000) + 4980 (ground truth, was 2000) = **10030 episodes** (was 3050).
`alfworld_pilot/config.yaml`'s `episode_plan` and `ground_truth` sections
updated to match. Rerunning `measure_mode.py` with the SAME real-measured
per-call rates (1054.7 in / 14.0 out tokens/call, 30.0 calls/episode --
unchanged, since only the episode count changed) against the new plan:
**$50.13 (deepseek-v4.1-flash) / $19.80 (ling-3.0-flash-vl) / $13.33
(mercury-2.5)** — all still under the $100 budget, deepseek using about
half of it. `cost_control.hard_cap_usd` raised 10.0 -> 75.0 accordingly.
The root package's `stage2_budget_estimate.py` (pure pre-real-ALFWorld
assumptions, 1530 episodes) is now superseded by this real-measured
projection and kept only as a historical reference.

### 4. Task types: coverage confirmed, real difficulty spread measured at zero cost

Confirmed against the REAL dataset (not just the mock) that all 6 official
ALFWorld task types are present (population: 22.2% pick_and_place_simple,
8.7% look_at_obj_in_light, 18.3% pick_clean_then_place_in_recep, 12.9%
pick_heat_then_place_in_recep, 15.0% pick_cool_then_place_in_recep, 22.9%
pick_two_obj_and_place) and correctly parsed by `RealAlfredEnv.
task_type_from_gamefile` (`episode_runner.py` already logs `task_type` per
episode; this verifies the parsing that field depends on).

Difficulty spread measured with ALFWorld's own built-in scripted expert
(`info['extra.expert_plan']`, zero LLM calls, n=30 games/type): mean steps
to solve ranges from 13.1 (look_at_obj_in_light) to **43.6
(pick_two_obj_and_place)** — and at this pilot's `env.max_steps: 30` cap, an
OPTIMAL scripted policy can only finish `pick_two_obj_and_place` **13% of
the time**, versus 73-90% for every other task type. This is the real-data
analogue of the task_difficulty simulator's confound (task-type-correlated
base success rate, decoupled from any memory's causal value) — and since
task type also determines which memories are similarity-relevant
candidates, it couples task difficulty to retrieval exactly as the
simulator models. Left `max_steps` unchanged (an explicit prior cost
control) rather than fixing it as a side effect of this check — flagging it
as a deliberate call worth making, not silently changing it. Full table
and per-type sample counts: `results/task_type_difficulty_check.json`.

## Part C, round 2 (2026-09-19, same day): step cap fixed, a real resource leak found and mitigated, checkpointing built, model selection run with real spend

Full details in `alfworld_pilot/README.md`. Summary:

**Step cap raised 30 -> 50.** `step_cap_check.py` swept optimal-policy
solvability at 30/40/50 steps: at 30, `pick_two_obj_and_place` solved only
18% of the time (vs. 78-92% for other types) -- not a difficulty gap, a
near-guaranteed structural failure. 40 barely helped (22%); only 50
(ALFWorld's own standard cap) brought every type to ~100%. Re-measured real
cost at the new cap: full 10030-episode plan now projects to $25.85-$97.19
(word-count proxy) depending on model, vs. $13-50 at the old cap.
`hard_cap_usd` raised 75 -> 90.

**Real, upstream resource leak found and mitigated.** Running the
difficulty/step-cap checks at a larger sample crashed twice with "No space
left on device". Root cause: `fast_downward` (a `textworld`/`alfworld`
dependency) `dlopen()`s a fresh ~32.5MB copy of its shared library on every
PDDL game load and never `dlclose()`s it -- this machine's 5.8GB tmpfs
`/tmp` fills after ~178 loads; confirmed the leak is scoped to the process
(killing it reclaimed everything). Mitigated two ways: pointing `TMPDIR` at
disk-backed storage (951GB free vs. 5.8GB tmpfs), and a new
subprocess-per-chunk supervisor (`run_chunked.py`) that restarts the
process periodically, built on new checkpointing infrastructure
(`checkpointed_runner.py`: resumable logging/ground-truth phases, JSONL
append+flush, per-task_id-seeded randomization so a crash+resume run
reproduces an uninterrupted one byte-for-byte). `CostTracker` now persists
its state across restarts too (was purely in-memory before -- a real gap,
since a restarted process would otherwise start counting spend from $0).

**Wall-clock estimate**: local overhead ~11.7 hours for the full plan;
LLM call latency (pre-Part-C assumption: 1.5s/call, 50 calls/episode worst
case) would add ~209 hours -- latency dominates by ~2 orders of magnitude.
Serial execution: ~9 days. Parallelization flagged as needed before the
real production run, not built yet.

**Part C model-selection test run, REAL SPEND ($0.063 total)**: 5 real
ALFWorld episodes each for `openai/gpt-5.6-luna`, `inclusionai/ling-3.0-
flash-vl` (the paid variant), `google/gemini-3.8-flash`. Results:
- `gpt-5.6-luna`: 60% success (3/5), 1594.2/33.3 avg in/out tokens per
  call, 25.6 calls/episode, 5 parse failures. Full-plan projection: **$92.13**.
- `ling-3.0-flash-vl`: hit a transient 429 (upstream rate limit) after 2
  episodes on the first attempt -- caught cleanly by the test's per-model
  exception handling, which moved on rather than crashing. Retried once;
  the LLM cache made gpt-5.6-luna's replay free and ling only needed 3 new
  episodes' worth of real calls. Second attempt: 60% success (3/5),
  1718.9/74.2 avg in/out tokens per call, 29.0 calls/episode, but a much
  higher parse-failure rate (40/145 ≈ 28% vs. gpt-5.6-luna's ≈4%). Full-plan
  projection: **$33.88**.
- `gemini-3.8-flash`: failed immediately, every attempt, with `400
  Reasoning is mandatory for this endpoint and cannot be disabled` -- a
  real, clean incompatibility with this pilot's `reasoning_enabled: false`
  design (llm_client.py's docstring specifically calls out this failure
  mode as something to surface, not swallow or silently work around).
  Zero spend, zero episodes. Not retried with reasoning enabled -- that
  changes the cost/latency profile and is a real design decision for the
  user, not something to change unilaterally mid-test.

Net: 2 of 3 candidates work; they differ meaningfully on cost ($92 vs $34
for the full plan) and format-compliance (4% vs 28% parse failures) --
worth weighing explicitly when picking a model for the real production run.
The $10 test spending cap was never close to being hit by real usage
(~$0.06 total); what it DID validate is that the exception handling for two
different real failure modes (rate limit, incompatible API requirement)
works as intended.

**Still not done before the full production run**: pick a model,
`determinism_check` against it, decide on parallelization, and run via
`run_chunked.py` (not a single long process) with `TMPDIR` on disk-backed
storage.

## Part C, round 3 (2026-09-19, same day): budget cut to $3.84, model picked, memory sanity check run

User's real remaining OpenRouter balance dropped to $3.84 (from the $0.063
already spent in round 2). `cost_control.hard_cap_usd` lowered 90 -> 3.00
(a real buffer below the true remaining balance). Full details in
`alfworld_pilot/README.md`; summary:

**Model decision: `openai/gpt-5.6-luna`**, not `inclusionai/ling-3.0-flash-
vl` despite the latter being ~3x cheaper for the full plan ($33.88 vs.
$92.13) -- ling's 28% parse-failure rate (vs. gpt-5.6-luna's ~4%) is
disqualifying because a format-failure rate that scales with prompt size
would scale with how many memories got included, confounding the exact
causal contrast this pilot measures. `google/gemini-3.8-flash` was never a
real candidate (fails outright on mandatory reasoning). `config.yaml`
updated: `llm.model_id`, a new `llm.pricing_per_million_tokens` field, and
the lowered hard cap.

**Memory sanity check, REAL SPEND $0.1875** (`memory_sanity_check.py`, 15
paired episodes = 30 total, real ALFWorld, gpt-5.6-luna). Paired design:
same real game AND same rng seed for both arms of a pair (via
`RealAlfredEnv.gamefile_path`), so candidate_ids/similarity/propensities
are identical -- only actual inclusion differs (normal randomized
inclusion vs. every candidate forced to 0). Results:
- Success rate WITH memories 93% (14/15) vs. BASELINE (no memories) 80%
  (12/15). Of 15 pairs, 13 concordant; of the 2 discordant, BOTH favored
  memories, none the other way -- small and not statistically decisive at
  this n (McNemar exact p=0.25), but directionally consistent: memories
  are not doing nothing, which is what this check existed to rule out.
- Parse failures essentially uncorrelated with memory count (r=0.008,
  n=30) and only weakly correlated with prompt length (r=0.17, likely
  driven by a few individual games that were hard for the model in BOTH
  arms, not by memory count itself) -- no evidence format failures scale
  with how many memories got included.
- Real tokens/call (1035.0 in / 34.9 out) and the resulting full-plan
  projection ($62.90) are pooled across the with-memories and
  zero-memory-baseline arms, so likely underestimate real production cost
  (production episodes are almost all with-memories) -- flagged as
  provisional, to be superseded by the next step's cleaner number.
- Spend: $0.1875 (target ~$0.30), well under the shared $3.00 hard cap.

**Budget reality**: the full 10030-episode plan's real-measured cost
($63-92) is no longer affordable against a $3.84 balance -- re-planning
against the real remaining budget is the next step, not running the full
plan as previously scoped.

## Part C, round 4 (2026-09-19, same day): ceiling-effect diagnosis and fix, zero new spend

93% success (memory sanity check) is too close to ceiling to resolve
per-memory effects of 0.10-0.15 (what ground truth can detect at this
budget). Diagnosed and fixed using data already in hand -- no new API
spend for this round. Full details in `alfworld_pilot/README.md`.

**Per-task-type breakdown** (`memory_sanity_check_breakdown.py`, task_type
recovered from the deterministic game-index mapping, zero LLM calls):
13/15 sanity-check pairs were `pick_and_place_simple` (the easiest type),
2/15 `pick_two_obj_and_place` (the hardest), 0/15 from the other 4 types.
Root cause: `task_id % len(game_files)` over `list_real_game_files`'s raw
filesystem-walk order happens to concentrate early indices on one type --
an accident of directory traversal, not a deliberate sample. The 93%/80%
figures are "93%/80% on an 84%-easy-type sample," and the 2 discordant
with/baseline pairs (the whole "memories help" signal so far) were both
the easy type -- `pick_two_obj_and_place` had only 2 samples, too few to
say anything about it specifically.

**Three options considered for reaching 50-70% success:**
1. Unseen/harder ALFWorld split -- uncertain benefit for a zero-shot LLM
   agent (the seen/unseen distinction is about RL-training exposure, which
   doesn't apply here); not recommended as the primary fix.
2. Lower `env.max_steps` (35/40) -- quantified and rejected:
   `step_cap_check.py` rerun with cap=35 added shows `pick_two_obj_and_place`
   solvability at 10% (cap=30) / 12.5% (35) / 17.5% (40) / 100% (50) by an
   OPTIMAL policy, while every other type stays 72-95% across 30-40 --
   this reintroduces the exact single-task-type structural failure the
   30->50 step-cap fix was built to remove.
3. **Weight task-type sampling toward harder types (chosen)**: new
   `weighted_task_source.WeightedRealTaskSource` draws each episode's task
   type with probability proportional to that type's mean steps-to-solve
   (13.1-43.6, from the real n=30/type measurement), then a game of that
   type -- both pure functions of `task_id`, composing with existing
   checkpointing unchanged. Trade-off: moves the logged population away
   from ALFWorld's natural task-type proportions, which matters for a
   deployment benchmark but not much for a methods pilot whose whole
   premise is already "does OPE see through a task-type confound" (a
   bigger deliberate confound is a harder test, not an invalid one). Keeps
   `env.max_steps=50`, so the structural-failure fix stays intact.

## Part C, round 5 (2026-09-20): small pilot run, REAL SPEND $2.00, ~$1.59 of $3.84 balance left

`small_pilot.py`: 180 real ALFWorld episodes via `WeightedRealTaskSource`,
checkpointed, stopped at its own $2.00 soft budget (shared cost-tracker
state; cumulative $2.1908 / $3.00 hard cap). Full details in
`alfworld_pilot/README.md`.

**The ceiling fix worked**: overall success 66% (target was 50-70%), all 6
task types represented (vs. 2/6 in the earlier naive-indexed sanity
check), skewed toward harder types as designed (`pick_two_obj_and_place`
51 episodes, down to `pick_cool_then_place_in_recep` 15).

**All four estimators ran cleanly** on the resulting log against all 30
memories (no NaNs). Pairwise Spearman: memory_worth vs ips = **-0.164**
(negative), memory_worth vs snips/doubly_robust ~0.29 (weak positive),
**snips vs doubly_robust = 0.922** (very strong agreement). This is the
qualitative pattern the whole pilot exists to detect -- MW diverging from
the propensity-corrected estimators, which cluster together -- showing up
in a REAL log for the first time. **Caveat, explicit and important**:
Stage 1's `small_data_results.py` found Spearman correlations stay too
noisy to be conclusive below ~1000-2000 episodes even in the best-case
simulated setting (bias is near-zero much earlier, but rank correlation
swings widely seed-to-seed). At n=180, this result is **suggestive and
consistent with the hypothesis, not a statistically decisive confirmation
of it** -- and with no real-ALFWorld ground truth yet (that needs the
ground-truth phase), there's no way yet to say which estimator's ranking
is actually closer to correct, only that they disagree in the predicted
direction. Full per-memory estimates: `results/small_pilot.json`.

**Remaining real budget**: ~$1.59 of the original $3.84. The full
10030-episode plan (~$63-92 projected) remains unaffordable as scoped --
a from-real-numbers re-plan is the next step.

## Part C, round 6 (2026-09-20): mini ground truth run — the decisive check, REAL SPEND $12.60, result is a genuine null

User topped up OpenRouter credit to ~$17.80. Full details in
`alfworld_pilot/README.md`; summary here.

**Selection** (zero new spend): reused the validated split-by-episode-index
design (`ground_truth_selection_bias_check.py`'s ~1.87x inflation finding)
on the small pilot's 180 episodes -- set A (first 90) selected the top-5
memory_worth-vs-ips RANK-disagreement memories (percentile rank within
each estimator's own distribution): `mem_17`, `mem_29`, `mem_21`, `mem_9`,
`mem_10`. Set B (second 90) computed the "official" estimates compared
against ground truth later.

**Real per-task-type cost mattered a lot**: 2 of 5 selected memories
(`mem_17`, `mem_29`) are tagged `pick_two_obj_and_place` (~$0.009/episode
in practice); 2 more (`mem_21`, `mem_9`) are tagged `pick_heat_then_place_
in_recep`, which turned out to be the MOST expensive type in practice
(~$0.019/episode) despite not being hardest by the scripted-expert
measure -- real agent behavior doesn't track that ranking. Full power (133
pairs/memory for MDE=0.15) across all 5 would have cost ~$19.36, over the
~$13.81 then-available. **User's call: drop `mem_10`, keep full 133-pair
power on the remaining 4**, raising `hard_cap_usd` to 18.00 (~$15.00
corrected estimate using real per-memory costs).

**Run**: via `run_chunked.py`'s subprocess-per-chunk supervisor (already
built/tested), one memory at a time, cheapest first (so a cap-triggered
stop would leave complete results for some memories, not partial for
all). All 4 completed 133/133 pairs: `mem_17` $2.68, `mem_29` $2.70,
`mem_21` $3.58, `mem_9` $3.03 (both cheaper-than-worst-case). Total: 1064
real episodes, **$12.6049**, well under the $18 cap. Took ~10 hours
wall-clock, run strictly sequentially (not parallelized, to avoid a real
race condition in the shared file-based cost-tracker state that
concurrent processes would hit).

**Result: all 4 memories show a statistically null effect.** Every 95%
CI (using the REAL measured paired-correlation rho, 0.62-0.67 -- much
higher than the 0.154 simulator-derived planning proxy) includes zero.
The four point estimates themselves span only 0.053 (-0.023 to +0.030) --
smaller than any individual estimate's own CI half-width, and far below
the ~0.15-0.19 MDE this design targeted.

**Verdict, strict and honest: too noisy to tell, and that itself is the
finding.** IPS achieved perfect (5/5) pairwise concordance with ground
truth's ordering of the 4 memories (MW/SNIPS/DR: 3/5 each) -- but this is
**NOT "IPS wins"**: the ground-truth differences being ordered are
statistically indistinguishable from each other and from zero, so getting
their order "right" carries little evidential weight, and IPS's own
set-A estimates are what flagged these memories as disagreements in the
first place (a share of the same selection-driven noise). The more
informative conclusion: these 4 memories, selected as "biggest MW-vs-IPS
disagreement" from a noisy 90-episode half-sample, likely fell victim to
the SAME selection-bias-inflation effect `ground_truth_selection_bias_
check.py` already demonstrated on synthetic data (~1.87x) -- apparent
disagreement in a small selection sample doesn't reliably indicate a real
underlying effect. This is a genuine, useful finding (validates a risk
this project flagged in advance), not a resolution of "which estimator is
right" -- that would need either a larger main-logging phase before
selection (less noise-prone), more ground-truth pairs (unaffordable here),
or accepting these 4 memories may simply have small true effects.

**Total real spend across the whole project: $14.8585** ($0.0628
model-selection test + $14.7957 shared production tracker: $0.1875 sanity
check + $2.0033 small pilot + $12.6049 mini ground truth). ~$5.20 of the
$17.80 balance remains -- the full 10030-episode plan remains unaffordable
as scoped.

## Part C, round 7 (2026-09-20): effect-size analysis -- confirms the null is a design problem, zero new spend

User's read on the mini ground truth null: individual memories in the
current 30-memory store likely have near-zero effect, so there's nothing
to rank -- a design problem, not an estimator problem. `effect_size_
analysis.py` confirms this with two independent lines of evidence
(details in `alfworld_pilot/README.md`):

1. **Noise-vs-signal decomposition** (all 30 memories, n=180): the
   theoretical null-model SE of the IPS estimator per memory (0.2202,
   using each memory's real candidacy count and propensities) is
   AT LEAST as large as the actually observed spread of IPS estimates
   across all 30 memories (std 0.2063) -- method-of-moments implies a
   NEGATIVE real-effect variance, i.e. noise alone explains everything
   observed, independent of which estimator is used.
2. **The 4 direct ground-truth measurements** (selected for looking most
   different) show a spread of just 0.053, smaller than any one of their
   own CIs -- a second, independent confirmation.

**Both lines agree: typical real per-memory effects in this store are ~0,
with an upper bound around 0.03-0.05.** Resolving the full 30-memory
store at delta=0.03 (the largest gap actually found) would need ~85,463
episodes (~$970) -- not measurable at any realistic budget. Real measured
rho (0.636 mean) is much higher than the 0.154 simulator-derived planning
proxy, which is favorable (fewer pairs needed per unit of resolution than
originally planned) but doesn't change the underlying finding.

**Redesign proposals (NOT run):**
- (a) 6 deliberately helpful/misleading memories, ranking target: 57
  pairs/memory for delta=0.15 (~$7.75), or 10.5 pairs for delta=0.35
  (~$1.42) if the engineered effect is large.
- (b) Same store, reframed as DETECTION (is this memory harmful?) rather
  than ranking -- one-sided test + a deliberately larger assumed effect
  needs much less power: 8.2 pairs/memory (~$1.12) at delta=0.35.
  **Recommended over (a)** as more budget-efficient and more likely to
  succeed, since it doesn't require fighting the same tiny-natural-effect
  problem that produced this session's null.
- Neither's effect size is measured yet -- a cheap validation check
  (~$0.10-0.20, same pattern as the earlier memory sanity check) is
  recommended before committing to either at full power.
- At the current ~$5.20 remaining, 6 memories could afford ~38 pairs/memory
  (MDE ~0.16-0.18) -- plausible if the engineered effects land in the
  assumed 0.2-0.4 range, unverified until tried.
