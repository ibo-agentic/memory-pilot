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

- `config/config.yaml` — all tunables for both stages.
- `src/memory_ope/simulator/` — Stage 1 synthetic DGPs + Monte Carlo oracle
  for true per-memory causal values (forced-in/forced-out on the natural
  randomized distribution of everything else — the same logic Stage 2 will
  use for its real ground truth).
- `src/memory_ope/estimators/` — the four estimators above.
- `src/memory_ope/evaluation/` — Spearman correlation / bias / variance
  comparison against the oracle, across 20 seeds; writes
  `results/stage1_report.json` + plots.
- `alfworld_pilot/` — Stage 2 skeleton only, gated on Stage 1 approval. Use
  the official ALFWorld repo (github.com/alfworld/alfworld) for install
  steps when it's implemented — do not invent them.
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

### 7. IPS unbiasedness test

`tests/test_ips_unbiased.py` — a hand-built two-context confounded DGP
(context-dependent propensity AND context-dependent base rate, same shape
as task_difficulty but reduced to one memory) where the true causal effect
is exactly 0.15 by construction. Averaged over 30 seeds x 20,000 episodes,
IPS and SNIPS both land within 0.01 of 0.15; Memory Worth (which estimates
a different, confounded quantity — P(success|included), not a contrast) is
asserted to be off by more than 0.1.
