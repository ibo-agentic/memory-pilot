# Stage 2 — ALFWorld pilot

## Phase 0 design snapshot and gaps (2026-09-29), before anything is run

**Current design** (`kaggle_config.yaml`, `kaggle_memory_store.py`, `kaggle_session.py`):
- **Main logging episodes**: `episode_plan.main_randomized_logging: 500` — **stale**,
  set before any real throughput was known; real Mode B throughput (~1,502
  episodes/30 GPU-hr) means 500 is a significant underuse of a single week's quota
  and should be recalibrated before Phase 3, not treated as fixed.
- **Task-type weights**: `env.task_type_weights` — 75% `pick_and_place_simple` +
  `look_at_obj_in_light`, 25% the other four, `pick_two_obj_and_place` lowest (Task B
  above); expected pooled success 45.6%.
- **Exploration/randomization**: `retrieval.M=10`, `propensity_min/max=0.3/0.7` (top-M
  real-embedding retrieval + randomized Bernoulli inclusion, unchanged from the paid
  run's mechanism); `env.max_steps=50` (confirmed identical to the paid run's own
  `config.yaml` by direct inspection — both say 50, the Kaggle run needed no change);
  memory store = `kaggle_memory_store.build_kaggle_store()` (21 memories, 4
  categories x 6 task types + 3 irrelevant).
- **Ground truth**: `ground_truth.n_memories: 6`, `pairs_per_memory: 30` — both
  **provisional placeholders**, not yet backed by a selection or a power check against
  real throughput/rho.
- **Cross-session resume**: `kaggle_session.py` — copy in from an attached Kaggle
  Dataset, run `run_chunked.py` time-boxed (default 11h, under the 12h session cap),
  copy out + push a new Dataset version. Explicitly documented in its own docstring as
  **untested outside a real Kaggle session** — a real, unverified risk for any
  multi-session campaign.

**Gaps flagged for the actual goal (validating IPS/SNIPS/DR against ground truth on a
real agent)**:
1. **No selection mechanism exists for which 6 memories to ground-truth.** The paid
   run's `mini_ground_truth_selection.py` (stratified + disagreement-oversampled, by
   memory_worth-vs-ips rank disagreement) is hard-wired to the paid run's own
   `small_pilot.jsonl` and can't run before a Kaggle main-logging dataset exists —
   this is a real chicken-and-egg gap: selection needs logging data, but which
   memories to prioritize is normally decided before running the full campaign. Needs
   a Kaggle-pointed equivalent, or an explicit decision to select post-hoc from
   Phase 3's own logs.
2. **500-episode logging budget is stale** (above) — should be recalibrated against
   real Mode B throughput before Phase 3, the same way the paid run rescaled its own
   `episode_plan` after `measure_mode`.
3. **No orchestration for running ground truth across multiple memories** —
   `run_chunked.py ground_truth` takes one `--memory-id` per invocation; getting all 6
   selected memories ground-truthed means 6 separate manually-tracked invocations
   (and re-invocations across sessions), with no single command or script tying them
   together yet.
4. **`rho` for Qwen is still unmeasured** (same caveat as the Timing results section
   below) — `ground_truth.pairs_per_memory: 30` was set before any real Qwen rho or
   throughput number existed, so its own implied power hasn't been checked against
   what's now known.
5. **`success_within_25/30` (this file's next section) isn't wired into any
   ground-truth or estimator invocation yet** — the secondary outcome is implemented
   and tested, but running the estimators against it is a manual step (call
   `truncated_success()` on a logged dataset before passing it to an estimator), not
   yet part of any Phase 3/4 script.
6. **`kaggle_session.py`'s Dataset-versioning handoff is untested** (above) — Phase
   0/1 should verify this specifically, since a silent failure there would look like
   "the next session started fresh" rather than an obvious error.

None of these block Phase 0 itself (an environment/compatibility probe), but several
(#1, #2, #3) should be resolved before Phase 3 (main logging) is actually launched,
and #6 before relying on any multi-session campaign.

## Pre-registered secondary outcome (2026-09-29): success_within_25 / success_within_30

Checked first, before adding anything: does the agent's prompt ever show `max_steps`,
remaining steps, or anything else that depends on the step limit? **No.**
`react_agent.py`'s `_build_prompt` (the only place the prompt is assembled) takes
`goal_obs, memory_texts, history, admissible_actions, few_shot_text` — no step count or
budget anywhere in its signature or body. `run_episode`'s loop (`for _step_idx in
range(max_steps):`, `react_agent.py:211`) uses `max_steps` only as a Python loop
bound; `_step_idx` is never passed into `_build_prompt` (`react_agent.py:212`) or
`SYSTEM_PROMPT` (`react_agent.py:26-32`, also step-count-free). The agent has no way
to know how many steps it has left.

Because of that, a **secondary outcome is worth pre-registering now, before any
Phase 0 data exists**: `success_within_25` (and `success_within_30`) — did the episode
succeed within the first 25 (or 30) steps of its already-logged trajectory, computed
by truncating post-hoc (a pure function of `success` + `steps_taken`, needs no new
logging fields, and can be applied retroactively to any already-logged episode,
including the paid run's).

**Prediction**: memory effects (per-memory Δ in success probability) will be **larger**
under the 25-step secondary outcome than under the primary 50-step outcome, because an
agent with no visibility into its remaining budget has less room to wander, recover
from a bad early guess, and still land on the right sequence of actions by step 50 — a
memory's guidance (or a harmful memory's misdirection) has comparatively more of its
effect "locked in" by step 25, before self-correction has as much chance to wash it
out. This is the same directional logic as the step-cap check above (long, looping
episodes are exactly where the memory-vs-episode-length confound analysis found the
most parse-failure noise), applied to effect size rather than parse-failure rate.

**What would count as support**: per-memory |Δ| estimates under `success_within_25`
are, on average across the ground-truthed set, larger than under the primary outcome,
with confidence intervals that don't just track the primary outcome's own (wider
CIs from lower absolute success rates at 25 steps could inflate point estimates
without being real — support requires the *pattern* to hold, not one noisy memory).

**What would count as against**: the two outcomes' per-memory effects are
statistically indistinguishable, or the 50-step outcome shows larger/equal effects —
i.e., the "self-correction" story is wrong or too small to matter at this model's
scale.

Implementation: `secondary_outcomes.py`'s `truncated_success(episodes, step_cap)`
returns a copy of any episode list with `"success"` replaced by the truncated outcome,
so every existing, unmodified OPE estimator (`src/memory_ope/estimators/`) can be run
against it by just swapping which list it's given — no estimator code changes, and it
works on already-logged episodes (this project's or the paid run's), not just future
runs.

## Kaggle timing results (2026-09-29) — real numbers, two bugs fixed, one open risk flagged

### 1. Real throughput

Mode B (2 workers, T4 x2, transformers backend, zero-shot + memories, `pick_and_place_simple`):
**~136 s/episode per worker, 1.89x combined speedup, ~1,502 episodes per 30 GPU-hours.**
(Mode A single-GPU and Mode C vLLM numbers aren't recorded here — vLLM was not tested,
see below — Mode B is the throughput this project will actually plan around, since the
real replication run is always multi-GPU on Kaggle's T4 x2.)

Detectable effect (80% power, `power_formula.py`, `n_memories=6`, `p=0.456`, and — still
**`rho=0.6355`, borrowed from the paid run's measured GPT-5.6-Luna correlation, not yet
measured for Qwen** — same caveat as the Timing estimate section above, now with real
throughput instead of an illustrative episode-budget table):

| budget | total episodes | pairs/memory | MDE (Δ, 80% power) |
|---|---|---|---|
| 1 week (30 GPU-hr) | ~1,501 | 125 | 0.107 |
| 2 weeks (60 GPU-hr) | ~3,002 | 250 | 0.075 |
| 4 weeks (120 GPU-hr) | ~6,004 | 501 | 0.053 |

vLLM: **not tested** — the notebook's compatibility probe requires a real pinned
revision, and `llm_vllm.revision` was still the `null` placeholder, so `VLLMClient`
correctly refused to construct rather than run against an unpinned model. Marked
skipped, not failed — the AWQ-fits-comfortably reasoning in `vllm_client.py`'s
docstring is unchanged and untested either way.

### 2. Two real bugs found running this on Kaggle, both fixed

- **Worker subprocess `PYTHONPATH`**: `timing_probe.py --workers 2` set
  `CUDA_VISIBLE_DEVICES` on each subprocess's environment but inherited whatever
  `PYTHONPATH` the parent had — and the parent was launched with `PYTHONPATH=src`
  (relative to `alfworld_pilot/`, so only `alfworld_pilot`'s own `src/`, missing
  repo-root `src/` where `memory_ope` lives). Every worker subprocess failed to
  import `memory_ope.retrieval`. Fixed: `run_multi_worker` now builds each worker's
  `PYTHONPATH` explicitly (`_worker_pythonpath()`, both source roots, absolute paths),
  regardless of what the parent inherited. The notebook's four `PYTHONPATH=src` cells
  (Mode A/B/C and the parity check) are fixed the same way (`PYTHONPATH=src:../src`).
- **Cache hits silently faking timing**: rerunning with the same seeds replayed
  cached `LLMCache` responses for some steps instead of doing a real generation,
  making those episodes' wall-clock time meaningless — and nothing caught it; the
  numbers just looked suspiciously fast. Fixed: every episode now records
  `n_cache_hits` (from `StepRecord.cached`, already tracked per step but never
  surfaced here before), and `check_no_cache_hits()` raises before any summary is
  computed if even one call anywhere in the run was a cache hit — no silent partial
  numbers, a hard refusal naming the affected episodes. Unit-tested
  (`tests/test_timing_probe_helpers.py`).

### 3. Step-cap check (existing logs only, no new runs)

Pooled 70 zero-shot + memories capability-check episodes (6 task types x 10 from
`capability_check_all_types_zeroshot_withmem.json`, plus 10 more
`pick_and_place_simple` episodes — task_id 10-19 — from the dedicated file that
aren't already in the all-types one). 27/70 (38.6%) succeeded.

Steps-taken distribution among the 27 successes: min=3, p25=4, **median=7**, p75=18,
p90=35, max=47.

| threshold | successes needing more steps than this |
|---|---|
| >20 steps | 4 |
| >25 steps | 4 |
| >30 steps | 4 |
| >40 steps | 3 |

(No successful episode landed in (25, 30] — that's why caps 25 and 30 show identical
success-rate impact below; it's a real gap in this data, not a bug.)

**Seconds/step is approximate, stated plainly**: no dataset has both real Kaggle
timing AND steps_taken for the *same* episodes (the capability check has no timing
field; the Kaggle timing run used different task_ids with an unmeasured step-count
distribution of their own). Derived `implied_seconds_per_step = 136s / 22.8 steps
(capability check's own pick_and_place_simple mean, same condition) ≈ 5.96 s/step`,
assuming per-step cost is roughly uniform across task instances — a bridged estimate,
not a joint measurement.

| cap | success rate | Δ success rate | lost successes | mean time saved/episode |
|---|---|---|---|---|
| 25 | 0.386 → 0.329 | −0.057 | 4 | 97.9 s |
| 30 | 0.386 → 0.329 | −0.057 | 4 | 77.9 s |
| 40 | 0.386 → 0.343 | −0.043 | 3 | 38.3 s |

At this sample size, none of these caps look like a clear win: cap=40 gives the
smallest success-rate loss but also the smallest time saved; cap=25/30 roughly
double the time saved but at a larger, already-non-trivial success-rate cost (−0.057
on a pooled base rate of 0.386 is proportionally large). **Recommendation: keep
`max_steps=50` for now** — the time saved is modest relative to the successes it
costs, and n=70 (4 lost successes) is too small to trust the exact trade-off point;
revisit once a real run gives a much larger sample. Full numbers:
`results_kaggle/kaggle_step_cap_check.json` (script: `kaggle_step_cap_check.py` —
distinct from the existing, unrelated `step_cap_check.py`, a pre-Kaggle scripted-
expert solvability sweep for the paid run's own config).

### 4. OOM warnings on long episodes — explanation, no pipeline change

Kaggle logs showed repeated PyTorch CUDA caching-allocator OOM messages on long
episodes (60-86k *cumulative* input tokens across all of that episode's steps — this
matches the same long, looping episodes already flagged in the prompt-length confound
check and step-cap analysis above, e.g. a 46-50-step episode at ~1,700+ tokens/step
compounds to exactly this range; no single request comes anywhere near 60k tokens).

**Can this change model outputs or determinism?**
- If PyTorch's caching allocator recovers internally (frees its own cached-but-unused
  blocks and retries) — the normal case when the failure is fragmentation rather than
  genuinely-full memory — this is pure memory bookkeeping and does not touch computed
  values. Harmless to outputs.
- If it does **not** recover, `model.generate()` raises, and nothing in this
  pipeline currently catches it — the whole process crashes. This matches a real,
  already-documented local crash earlier in this project (`CUBLAS_STATUS_EXECUTION_FAILED`
  at episode 11/20 of the heaviest local condition, VRAM confirmed fully released
  after, required restarting as a fresh process). This is **lost work, not corrupted
  data** — `LLMCache.put()` only writes after a generation succeeds, so a crash mid-
  generation can't leave a corrupted cache entry, and the in-progress episode's data
  is simply never written rather than written wrong.
- **The real determinism risk is narrower but genuine**: if memory pressure causes
  PyTorch/cuDNN to select a *different* attention or matmul kernel/algorithm than it
  would under low pressure (some backends are chosen heuristically based on available
  memory), that's a different code path with different floating-point reduction
  order — which can flip a near-tied greedy-decoding argmax, exactly the same
  mechanism `local_model_client.py`'s own docstring already flags for ordinary GPU
  non-determinism, except triggered by memory state specifically rather than being a
  constant background risk. This would specifically threaten the longest, most
  memory-pressured episodes — already the same tail that's noisiest in every other
  check in this file — not the bulk of short, well-behaved episodes.

**Suggested fixes (not applied)**, roughly in order of how directly they address the
mechanism above:
1. Lower `max_steps` (ties directly to the step-cap check above — cutting off long
   episodes earlier removes them from the memory-pressure regime entirely, at the
   success-rate cost quantified there).
2. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — the standard, well-
   documented fix for allocator fragmentation causing OOM despite technically-enough
   total free memory; likely the highest-value, lowest-risk single change.
3. Periodic `torch.cuda.empty_cache()` in the episode loop (e.g. every N steps) to
   reduce fragmentation buildup directly.
4. Wrap `model.generate()` in a try/except for `torch.cuda.OutOfMemoryError` that
   clears the cache and retries once, turning a fatal crash into a recoverable,
   *logged* event — logging matters here specifically so a recovered-after-retry step
   can be flagged and excluded from anything determinism-sensitive (e.g. a future
   ground-truth ground pair), not silently treated as identical to a normal step.

None of these are applied here, per instruction — this section is the explanation and
recommendation only.

## Kaggle timing run, prepared (2026-09-29) — nothing run on Kaggle yet

Extends the timing estimate above from "no data exists" to "a real probe is ready to
run on Kaggle." Everything in this section was written and, where possible, smoke-
tested locally; nothing was run on Kaggle.

**`timing_probe.py` extended**: now records, per episode, wall-clock seconds,
steps_taken, total input tokens, total output tokens, and seconds/step, and reports
both mean AND median seconds/episode (median matters because this project's own
capability check already found a few long, looping episodes that pull the mean well
above the typical case). Smoke-tested locally (2 episodes, the local 8GB card, not a
Kaggle timing number) — script runs correctly end to end.

**`--workers N` (multi-GPU)**: spawns N subprocesses, one per GPU
(`CUDA_VISIBLE_DEVICES=0..N-1`), on disjoint episode slices (`split_episode_range`),
then merges results and reports combined throughput vs. single-worker throughput
(`merge_worker_reports`, using the orchestrator's own measured parallel wall-clock,
not a derived estimate). These two functions are pure and GPU-free, and are unit-
tested (`tests/test_timing_probe_helpers.py`, 9 tests: disjoint/contiguous slicing,
remainder handling, and a synthetic 2-worker case asserting ~2x speedup from
concurrent wall-clock). **The actual subprocess/multi-GPU execution path itself could
not be tested here** — this development machine has exactly one GPU, and spawning a
second worker against a nonexistent `CUDA_VISIBLE_DEVICES=1` would just fail (or, on
a machine with 2 GPUs of insufficient combined VRAM, risk OOM) — so this must be
verified for real on Kaggle's T4 x2 session; that's what the notebook's Mode B cell
is for.

**vLLM backend — added, not yet verified on real T4 hardware**: researched (not
tested — no T4 available here) whether vLLM can run Qwen2.5-7B-Instruct on a T4 within
Kaggle's limits. Findings: vLLM's default dtype (bfloat16) hard-fails on a T4
(compute capability 7.5, needs >=8.0) — long-documented, fixed by forcing
`dtype="half"`, which `vllm_client.py`'s `VLLMClient` always does. Full fp16 for a 7B
model on a 16GB T4 is tight (~14GB of weights alone) and multiple public reports
describe needing hand-tuned `gpu_memory_utilization`/`max_model_len` to avoid OOM —
plausible but not a safe default. The official AWQ checkpoint
(`Qwen/Qwen2.5-7B-Instruct-AWQ`) drops weight VRAM to ~4-5GB with comfortable
headroom, and is well-supported by vLLM — **added as the recommended default**, with
fp16 available but flagged as unverified/risky. `kaggle_config.yaml`'s new
`llm_vllm.revision` is a placeholder (`null`) — `VLLMClient` refuses to construct
until it's a real pinned commit hash, same hard rule as every other model in this
project; the notebook's Section 1 cell resolves and prints the real hash to paste in.
**The actual "does it fit" answer is empirical and can only come from Kaggle's T4** —
this module makes the backend available to that probe, it isn't the probe itself.

New `vllm_parity_check.py` (not run) compares the PARSED ACTION sequence (not raw
completion text — different backends can legitimately diverge at the token level even
under greedy decoding) between the transformers and vLLM backends on 5 episodes with
identical seeds.

**New notebook**: [`kaggle/timing_probe.ipynb`](kaggle/timing_probe.ipynb) — installs
dependencies, clones this repo, runs the vLLM compatibility probe, runs the timing
probe in three modes (single GPU, 2 workers, vLLM if the probe passed) at 20 episodes
each, prints one combined results table, runs the parity check if applicable, then
prints the MDE-at-1-week/MDE-at-2-weeks table (this repo's `power_formula.py`) for
each measured throughput. Exact settings and click order are in the notebook's own
first two cells (GPU: T4 x2, Internet: ON — see "What to click on Kaggle" below).

## Task sampling weights (2026-09-28): weighted toward easy types, expected pooled success 45.6%

The capability check found the natural (unweighted) task-type mix pools well below the
40-70% target, since 4 of 6 task types individually sit at 0-30% success (n=10/type,
zero-shot + memories, `capability_check_all_types_zeroshot_withmem.json`):

| task type | success rate (n=10) |
|---|---|
| pick_and_place_simple | 0.50 |
| look_at_obj_in_light | 0.60 |
| pick_clean_then_place_in_recep | 0.20 |
| pick_heat_then_place_in_recep | 0.20 |
| pick_cool_then_place_in_recep | 0.30 |
| pick_two_obj_and_place | 0.00 |

New `kaggle_config.yaml` key `env.task_type_weights` sends 75% of episodes to the two
easiest types and 25% to the other four, with `pick_two_obj_and_place` (0% success at
n=10, also the hardest type by mean steps-to-solve) at the lowest weight of the six:

| task type | weight |
|---|---|
| pick_and_place_simple | 0.40 |
| look_at_obj_in_light | 0.35 |
| pick_clean_then_place_in_recep | 0.07 |
| pick_heat_then_place_in_recep | 0.07 |
| pick_cool_then_place_in_recep | 0.06 |
| pick_two_obj_and_place | 0.05 |

**Expected pooled success rate under these weights: 45.6%** (`sum(weight_i * rate_i)`),
comfortably inside the 40-70% target band. New `weighted_task_source.WeightedRealTaskSource`
methods `probability_for(task_type)` / `sampling_weight(task_id)` expose the normalized
draw probability; `episode_runner.run_logged_episode` now logs `sampling_weight` alongside
the `task_type` it already logged, for every episode run through a weighted task source
(`None` for the paid run's plain, unweighted `RealTaskSource` — `config.yaml` has no
`task_type_weights` key, so its behavior is unchanged). `env_factory.build_task_source`
picks `WeightedRealTaskSource` over the plain one only when `env.task_type_weights` is
set, which only `kaggle_config.yaml` does.

New script `task_sampling_check.py` verifies the mechanism against the real ALFWorld
game-file listing (20,000 draws, no LLM calls): realized mix matched the intended
weights to within 0.5 percentage points on every task type. A new test,
`tests/test_weighted_task_source.py::test_sampled_mix_matches_weights_over_many_draws`,
locks this in against regression (20,000 draws against a monkeypatched game-file list,
asserting each type's realized frequency is within 0.02 of its intended weight, and that
the two easy types combine to 70-80%).

## Timing estimate (2026-09-28): no wall-clock data exists yet — script provided, not run

**Checked first, as instructed: the logs don't have it.** Grepped
`local_model_client.py`, `cache.py`, `cost_tracker.py`, and `capability_check.py` for
any time/timestamp/duration/elapsed field — none exists anywhere in this pipeline, on
any GPU. The `capability_check_*.json` files have no per-episode wall-clock field to
read, so "average seconds per episode" cannot be computed from existing data; it has to
be measured. File modification timestamps on `results_kaggle/*.json` were considered
and rejected as a substitute: the gaps between them conflate real episode time with
between-run idle/decision time and a documented mid-run CUDA crash-and-retry, and even a
clean measurement from the local 8GB card wouldn't transfer to Kaggle's T4/P100 target
GPU, which has different throughput.

New script `timing_probe.py` (not run) measures real wall-clock seconds/episode for
zero-shot + memories on `pick_and_place_simple`, separating one-time model-load time
from per-episode time, and reports `episodes_per_30_gpu_hours` directly — intended to
run once a real Kaggle GPU session is available.

**What can be reported now without fabricating a throughput number**: the MDE (minimum
detectable effect at 80% power, two-sided α=0.05) as a function of total episode budget,
using this project's own `power_formula.py`, `n_memories=6` (`kaggle_config.yaml`'s
provisional ground-truth subset size), `p=0.456` (the new pooled success rate above),
and `rho=0.6355` — **borrowed from the paid run's measured GPT-5.6-Luna ground-truth
correlation, explicitly not yet measured for Qwen** (Phase 4 hasn't run):

| total episodes | pairs/memory | MDE (Δ, 80% power) |
|---|---|---|
| 1,000 | 83 | 0.131 |
| 5,000 | 417 | 0.058 |
| 10,000 | 833 | 0.041 |
| 20,000 | 1,667 | 0.029 |
| 30,000 | 2,500 | 0.024 |
| 50,000 | 4,167 | 0.019 |
| 85,463 (paid run's own budget) | 7,122 | 0.014 |

To turn this into "1 week" / "2 weeks," once `timing_probe.py` reports real
`seconds_per_episode`: `N_1week = 30*3600/seconds_per_episode`,
`N_2weeks = 60*3600/seconds_per_episode`, then read (or interpolate) this same table at
that `N`. Without that measurement, reporting a specific 1-week/2-week MDE number would
be reverse-engineering a throughput figure that was never actually measured — exactly
the kind of forced conclusion the pre-registered-predictions section above and the
prompt-length check both explicitly avoid doing. Note also that `rho` itself is
borrowed, not measured, for Qwen — per the pre-registered predictions above, a weaker
model might plausibly show a *different* paired correlation than Luna's, which would
shift every row of this table; that assumption should be revisited once Phase 4 (or even
a handful of real paired episodes) gives a measured Qwen `rho`.

## Pre-registered predictions (2026-09-27), before any Phase 0 / Kaggle data exists

Stated now, before any real Kaggle episode has been run, so later results can be
checked against a prediction made in advance rather than a story fitted after the
fact.

**Prediction**: Qwen2.5-7B-Instruct will show **larger** per-memory effects (larger
|Δ| in success-probability terms) than `openai/gpt-5.6-luna` did on the paid run,
where the largest gap actually found across the full 30-memory ground-truth study was
**Δ≈0.03** (`results/paper_data.md` §5.2 — most other memories' gaps were smaller
still and statistically indistinguishable from zero). Rationale: a weaker/smaller
model has less capacity to notice and override a memory that's subtly wrong, or to
recover the right behavior despite a memory that only helps partially — so both
harmful and clearly-correct memories should move Qwen's success rate by more than
they moved Luna's, which mostly shrugged off bad advice via its own reasoning.

**What would count as support, stated in advance**: at least one memory with a clear
a-priori category (correct or harmful, from `kaggle_memory_store.py`'s labeling) shows
an estimated |Δ| noticeably larger than 0.03 (a reasonable bar: >0.05, i.e. comfortably
outside where the paid run's own gaps clustered) with a confidence interval that
excludes zero — not just a larger point estimate riding on a wide, zero-crossing CI.

**What would count as against, stated in advance**: estimated per-memory gaps stay
at or below the ~0.03 scale Luna showed, or come back with CIs crossing zero at a
similar or higher rate than the paid run's own §5.2 findings — i.e., model capability
differences don't materially change how sensitive success is to a single memory.

**What would be inconclusive, not forced into either bucket**: if the achievable
sample size (see the Kaggle GPU-hour budget in the Timing estimate section below)
can't resolve effects anywhere near this scale — the paid run's own MDE analysis
puts real per-memory resolution in the hundreds-to-thousands-of-episodes range per
memory at this effect size (`paper_data.md` §5.3) — the honest outcome is "underpowered
to tell," not a false confirmation or disconfirmation either way.

**Recheck at larger n** (flagged in the [prompt-length confound check](#prompt-length-confound-check-2026-09-27-the-065-correlation-is-episode-length-not-memory-content--pass)
below at n=5 each, not treated as a result there, but worth watching once real
per-memory sample sizes grow): `irrelevant_3`, `two_correct`, `clean_harmful`.

## Prompt-length confound check (2026-09-27): the 0.65 correlation is episode length, not memory content — PASS

The zero-shot+memories row above shows `corr(avg_prompt_tokens, parse_failures) = 0.647`
for `pick_and_place_simple`. Before trusting that as "the prompt is too long," checked
whether it's actually driven by memory *content* (which would confound the treatment
this pilot measures) or just by episode length in general (longer episodes accumulate
more growing ReAct history regardless of memories, and looping/near-step-cap episodes
naturally rack up more parse failures too). New script: `prompt_length_check.py`.

**Data and log format, read before analyzing**: `results_kaggle/capability_check_pick_and_place_simple_zeroshot_withmem.json`
(20 episodes). Each saved episode has `task_id, task_type, success, steps_taken,
hit_step_cap, parse_failures, n_included, avg_input_tokens, fallback_classes,
failure_mode, actions` — notably **not** saved: per-step input tokens (only the
episode-level mean), which specific memory ids were included (only the count), or
which steps were parse failures. The other zero-shot+memories file
(`capability_check_all_types_zeroshot_withmem.json`) duplicates this same run's
task_id 0-9 under other task types, so mixing it in would confound memory presence
with task type — this check uses only the dedicated 20-episode file.

Rather than approximate the missing fields, the script **exactly reconstructs** them
by deterministic replay, with zero new LLM calls, zero GPU, and zero change to the
prompt or sampling: retrieval is a pure function of `(memories, real per-episode goal
text, a task_id-seeded RNG)`, so replaying it recovers which memories were actually
included; real ALFWorld's `env.step()` is deterministic given the same game and
action, so replaying the saved action sequence recovers every observation/admissible-
action list the model actually saw; and the disk-persisted LLM cache (`cache_kaggle/`,
still intact from the real run) then gives the *actual recorded* response text and
input-token count for each reconstructed request — no re-tokenization or model load
needed for that part. All 20 episodes replayed with exact validation: reconstructed
`n_included`, the full per-step action sequence, and the total `parse_failures` count
all matched the saved log exactly for every episode.

**Pass rule (decided before looking at results)**: PASS if the step-1-only prompt
length has `|Spearman r| < 0.15` against episode parse-failure rate, AND the
`memory_tokens` coefficient in a step-level logistic regression (`parse_fail ~
memory_tokens + step_index + history_tokens`, one row per step) has a 95% CI that
includes zero.

| check | result |
|---|---|
| 1. step-1 prompt tokens vs. episode parse-failure rate | Spearman r = **-0.015** (p=0.95), n=20 |
| 2. step-level logistic regression (n=456 steps, 136 parse failures, converged) | `memory_tokens`: coef=-0.00012, CI=[-0.0027, 0.0024], p=0.93 (**includes zero**) — `history_tokens`: coef=+0.0018, CI=[0.0009, 0.0027], p<0.001 (**does not include zero**) — `step_index`: coef=-0.012, CI=[-0.039, 0.016], p=0.41 |
| 3. episode length vs. parse-failure rate | Spearman r(steps_taken, pf_rate) = **0.473** (p=0.035); mean pf_rate successful (n=14) = 0.165, failed (n=6) = 0.267 |
| 4. per-memory pf_rate with vs. without | 3 of 21 memories flagged (with-rate > 1.5x without-rate): `irrelevant_3` (0.347 vs 0.145, n=5), `two_correct` (0.313 vs 0.156, n=5), `clean_harmful` (0.261 vs 0.174, n=5) — all at n=5, and 7 memories never appeared in any of the 20 episodes at all (n_with=0) |

**Verdict: PASS.** Step-1 prompt length has essentially zero correlation with
parse-failure rate (r=-0.015), and the regression's `memory_tokens` coefficient is
indistinguishable from zero. What *is* a significant predictor is `history_tokens` —
the growing action/observation log accumulated over a long episode — consistent with
check 3's finding that longer episodes (more steps, mostly the looping/near-step-cap
ones) have higher parse-failure rates, and failed episodes have a higher mean
parse-failure rate than successful ones. **The 0.647 correlation is an episode-length
artifact (loopy episodes are both long and error-prone), not a memory-content
confound.** The check 4 flags are noted per the pre-registered instruction to treat
n=5 as a flag, not a result — they aren't load-bearing for the verdict, and the
memory-level counts (13/21 memories have `n_with` below 8, 7 have `n_with=0`) are too
sparse on their own to distinguish a real per-memory effect from noise; they're worth
re-checking once a larger run gives better per-memory coverage, not conclusive now.

**Practical implication**: the zero-shot+memories parse-failure/prompt-length
correlation does not invalidate using this configuration to measure memory effects —
it reflects agent looping behavior on hard episodes, not the memory-inclusion
treatment itself. No change to the prompt, sampling design, or the zero-shot decision
above as a result of this check. Full numbers: `results_kaggle/prompt_length_check.json`.

## Status (2026-09-27, later): pre-Kaggle capability check — Qwen2.5-7B-Instruct CAN do ALFWorld, but needs weighted task sampling AND zero-shot (not few-shot) prompting

Before spending any Kaggle session on Phase 0, checked the thing that would
have made the whole replication moot: can the chosen local model actually
solve ALFWorld tasks at all, or was the earlier smoke test's 0/3 just an
early sign of a fundamentally incapable setup? New script:
`capability_check.py`. Zero API spend throughout (local model only). Full
data: `alfworld_pilot/results_kaggle/capability_check_*.json`.

**Headline: yes, it can — 50-70% success on the two easiest task types,
comfortably in the target 40-70% band — but two real, decision-changing
findings came out of the deeper checks, both of which flip a plan detail:**

### 1. Zero-shot beats few-shot once you check for confounds, not just raw success

Ran 20 paired episodes (identical task instances across all four cells) on
`pick_and_place_simple`, crossing prompting mode x memory condition:

| condition | success | parse-failure rate | corr(n_included, parse_failures) | corr(avg prompt tokens, parse_failures) | fallback harmful % |
|---|---|---|---|---|---|
| zero-shot, no memories | 50% (10/20) | 43.2% | n/a (n_included=0) | n/a | 2% (6/277) |
| few-shot, no memories | 55% (11/20) | 28.4% | n/a | n/a | 3% (5/148) |
| zero-shot, **with memories** | **70%** (14/20) | 29.8% | **-0.037** | 0.647 | 3% (4/136) |
| few-shot, **with memories** | **70%** (14/20) | 39.2% | **+0.506** | 0.806 | 6% (11/175) |

Looking only at raw success, few-shot looks like a mild win (matching the
original ReAct paper's own choice to use few-shot prompting). **It isn't,
once memories are in the picture**: the exact same check this project ran
on the paid model (`memory_sanity_check.py`: `corr(n_included,
parse_failures) = 0.008`, `corr(avg_tokens, parse_failures) = 0.17`) shows
few-shot introduces a real, moderate **confound** between memory count and
parse-failure rate (+0.51) that zero-shot does not have (-0.04, statistical
noise around zero). A configuration where format failures scale with how
many memories got included would confound the exact causal contrast this
whole pilot measures — which is precisely why the paid run picked
`openai/gpt-5.6-luna` over the cheaper `ling-3.0-flash-vl` for the same
reason. **Decision: zero-shot, not few-shot**, despite few-shot's small
edge in the memory-free numbers alone.

Both conditions' `corr(avg_tokens, parse_failures)` (0.65-0.81) run well
above the paid run's 0.17 regardless of prompting mode — a real, structural
difference: this open 7B model's format compliance degrades with prompt
length noticeably more than the commercial model's did. This isn't
disqualifying (zero-shot's `n_included` correlation is still ~0), but it's
a real risk to keep monitoring once episodes run longer in production
(longer ReAct histories = longer prompts = more of the same pressure).

**Also confirmed: memories are not doing nothing.** Success jumped from
50-55% (no memories) to 70% (with the real `kaggle_memory_store`, same task
instances, both prompting modes) — the same qualitative finding as the paid
run's own memory sanity check (93% vs. 80%), now reproduced with an open
model and real embeddings for the first time.

**Fallback-action harm is low everywhere except one task type.** When a
parse failure occurs, `react_agent.py` falls back to the first admissible
action; classified each fallback as "harmful" (an arbitrary `put`, or
closing a receptacle just opened for a pending action) or "neutral"
(anything else — wastes a step, doesn't destroy progress). Harmful fraction
stayed at 0-6% in every `pick_and_place_simple` condition and 0-3% across
5 of the 6 task types below — except `pick_two_obj_and_place` (14%, see
below), where a wrong fallback has more chances to do real damage on a
two-part task.

### 2. Per-task-type breakdown: natural task-type mix lands BELOW the target band, not above it

10 episodes/type, zero-shot + real memories (the now-recommended
configuration):

| task_type | success | parse-failure rate | fallback harmful % | n |
|---|---|---|---|---|
| look_at_obj_in_light | **60%** | 52.5% | 0% (0/127) | 10 |
| pick_and_place_simple | **50%** | 28.9% | 3% (3/88) | 10 |
| pick_cool_then_place_in_recep | 30% | 32.2% | 3% (4/136) | 10 |
| pick_clean_then_place_in_recep | 20% | 57.6% | 0% (1/264) | 10 |
| pick_heat_then_place_in_recep | 20% | 33.3% | 3% (4/143) | 10 |
| pick_two_obj_and_place | **0%** | 27.8% | **14%** (19/139) | 10 |
| **overall, unweighted pool** | **30%** (18/60) | — | — | 60 |

(Per-type `corr(n_included, parse_failures)` swings from -0.55 to +0.58
with no consistent sign across types at n=10 — this project's own Stage 1
small-data results already established that correlation/rank estimates are
this noisy below ~1000 samples; **do not treat these per-type correlations
as reliable**, only the n=20 same-task-instance check above.)

**This is the paid run's own ceiling-effect problem, but inverted.** The
paid run's natural game-file ordering pinned success near 93%/80% (too
close to ceiling to resolve per-memory effects) and needed
`WeightedRealTaskSource` to oversample HARDER types and bring the rate
down. Here, the natural/unweighted pool sits at 30% — **below** the 40-70%
target band — because a 7B model genuinely struggles with the
appliance-interaction types (clean/heat/cool, all 20-32%) and cannot do
`pick_two_obj_and_place` at all (0/10) at this prompting/memory setting.
**The same tool, `weighted_task_source.WeightedRealTaskSource`, is the
fix, just pointed the other way**: oversample the two easy types
(`pick_and_place_simple`, `look_at_obj_in_light`, currently 50-60%) and
deprioritize `pick_two_obj_and_place` until independently re-validated,
rather than the paid run's weighting toward harder types. Exact weights
should be re-derived from a larger sample before committing to the real
Kaggle run (n=10/type here is small, especially for the 0% cell — a true
rate anywhere from 0% to ~28% is statistically plausible at this n).

### Decision, per the user's own 40-70% rule

**Proceed** — Qwen2.5-7B-Instruct clears the bar on the task types that
matter, and the zero-shot + real-memories configuration shows a usable,
essentially unconfounded signal. Three concrete changes to the plan before
Kaggle Phase 0, all cheap (no code, or already-existing code):

1. **Use zero-shot prompting, not few-shot** — drop `react_agent.FEW_SHOT_EXAMPLES`
   from the production config (it remains in the codebase, available if a
   future check finds a setting where it doesn't introduce a confound).
2. **Weight task-type sampling toward the easy types**
   (`pick_and_place_simple`, `look_at_obj_in_light`), deprioritizing
   `pick_two_obj_and_place` — reuse `weighted_task_source.py`'s existing
   mechanism with a new weight table (inverse of the paid run's own
   difficulty-based weights), re-derived from a larger local sample before
   the real run.
3. **Re-run this same capability check at a larger n** (this was
   deliberately small, 10-20 episodes/cell, to stay fast and local) once
   the weighting is set, to confirm the pooled rate actually lands in
   40-70% and that the zero-shot/memory-count non-confound holds up outside
   `pick_and_place_simple` specifically.

### New files

- `alfworld_pilot/src/alfworld_pilot/capability_check.py` — the check
  itself: runs N episodes restricted to one (or all) task type via
  `weighted_task_source.WeightedRealTaskSource` with all-but-one weight
  zeroed (same reuse pattern as the paid run's
  `engineered_effect_validation.py`), with or without the real memory
  store/embedding retrieval, computing success rate, parse-failure rate,
  the two requested correlations, a failure-mode classifier (format
  issues / looping / never found the object / picked up but stuck /
  attempted but incomplete), and the fallback-harm classifier described
  above.
- `react_agent.py`: added `FEW_SHOT_EXAMPLES` (two hand-written,
  game-instance-independent demonstrations — a plain pick-and-place and
  one with an intermediate appliance step) and an optional `few_shot_text`
  parameter on `_build_prompt`/`run_episode`, default `None` — every
  existing caller and test is unaffected.
- `alfworld_pilot/results_kaggle/capability_check_*.json` — full per-episode
  data for every condition tested (tracked in git, unlike `logs_kaggle`/
  `cache_kaggle`, matching how the paid run's own `results/*.json` files
  are tracked — this is analysis output, not bulk/ephemeral run state).

## Status (2026-09-27): Kaggle replication (zero-cost, real embeddings + local model) — infrastructure built AND locally verified end to end; real Kaggle hardware (Phase 0) not yet run

New work, separate from (and not touching) the paid run's frozen results
above: a plan (approved 2026-09-26,
`C:\Users\Ibo\.claude\plans\logical-meandering-feather.md`) to replicate this
pilot on Kaggle's free GPU tier, fixing its two acknowledged weaknesses at
zero additional dollar cost — real sentence-embedding retrieval instead of
the mock topic-match heuristic, and a local open model
(`Qwen/Qwen2.5-7B-Instruct`) instead of the one commercial model every real
result so far has used. See "Kaggle replication" below for the full design
and current status. **Nothing has run on real Kaggle hardware yet** — Phase 0
(does ALFWorld even install there?) requires an actual Kaggle session, which
hasn't happened. But every piece of new code HAS now been run for real,
successfully, end to end, on this project's existing local machine (WSL2
Ubuntu, an RTX 4060 with 8GB VRAM — much smaller than Kaggle's target
T4/P100 16GB, so this is a wiring verification, not a stand-in for the real
scaled run): real ALFWorld, real `Qwen/Qwen2.5-7B-Instruct` (4-bit, loaded
in 22s), real `bge-small-en-v1.5` embedding retrieval, and — for the first
time in this project's history — a **passing** `determinism_check` against a
real model (`holds=True`). See "Local smoke test" below for the full results.

- All new code written and merged into `alfworld_pilot/src/alfworld_pilot/`
  (`local_model_client.py`, `embedding_retrieval.py`, `kaggle_memory_store.py`,
  `kaggle_session.py`, `kaggle_smoke_test.py`), plus small, backward-compatible
  additions to `episode_runner.py`, `ground_truth_runner.py`,
  `checkpointed_runner.py`, and `run_chunked.py` (an optional
  `similarity_fn`/`candidacy_similarity_fn` injection point, `--llm local`,
  `--memory-store kaggle`, `--config`, `--cache-dir`, `--time-budget-seconds`)
  — every existing test still passes unchanged (`pytest alfworld_pilot/tests/`,
  25/25), confirming the paid run's own code paths are untouched when these
  new options aren't used.
- A new `kaggle_config.yaml`, kept fully separate from `config.yaml` so the
  paid run's exact historical settings stay untouched and citable as-is.

- All new code written and merged into `alfworld_pilot/src/alfworld_pilot/`
  (`local_model_client.py`, `embedding_retrieval.py`, `kaggle_memory_store.py`,
  `kaggle_session.py`, `kaggle_smoke_test.py`), plus small, backward-compatible
  additions to `episode_runner.py`, `ground_truth_runner.py`,
  `checkpointed_runner.py`, and `run_chunked.py` (an optional
  `similarity_fn`/`candidacy_similarity_fn` injection point, `--llm local`,
  `--memory-store kaggle`, `--config`, `--cache-dir`, `--time-budget-seconds`)
  — every existing test still passes unchanged (`pytest alfworld_pilot/tests/`,
  25/25), confirming the paid run's own code paths are untouched when these
  new options aren't used.
- A new `kaggle_config.yaml`, kept fully separate from `config.yaml` so the
  paid run's exact historical settings stay untouched and citable as-is.

**Real, important discovery made while researching this** (see the plan's
"Design decisions" section for the full citations): current mainline **vLLM
has documented compatibility problems with Kaggle's actual free GPUs**
(Tesla T4, compute capability 7.5, and P100, capability 6.0) —
`bfloat16`-only-on-8.0+ errors and other T4-specific breakage are open issues
in vLLM's own tracker, and a community compatibility shim
(`kaggle-vllm`) exists specifically because this isn't turnkey. The plan
defaults to plain `transformers.generate()` instead, with vLLM attempted only
as an opportunistic upgrade in Phase 0 if it happens to work cleanly — this
is a direct instance of this project's own repeated lesson (verify a
third-party toolchain claim empirically before committing to it; see the
Python-3.13/PEP-667/textworld story below) applied to a brand-new tool this
project hadn't touched before.

## Status (2026-09-21): redesign (b) validation FAILED — engineered effects too small, STOPPING before the detection experiment

Step 1 of the redesign-(b) plan (detect harmful memories rather than rank
all of them) measured the real effect of one deliberately-helpful and one
deliberately-harmful engineered memory before committing to the full
6-memory experiment. **Both came back below the 0.2 threshold**:
`mem_helpful_heat` +0.10 [-0.096, +0.296], `mem_harmful_heat` +0.00 exactly
[-0.201, +0.201] — neither CI excludes zero, and the "harmful" memory
shows literally no measured effect at all. Real spend: $1.1923 (dedicated
tracker, well under the $13 cap). Episode transcripts show why: the ReAct
agent sometimes follows the harmful memory's bad advice initially (e.g.
going to the fridge first) but then self-corrects using real-time
environment feedback and completes the task via the correct appliance
anyway, since 50 steps gives ample slack to recover from one wrong early
move. **Per the user's explicit stopping rule, we are NOT proceeding to
Step 2** — see "Redesign (b): Step 1 validation results" below for the
full writeup and what this implies for the design. The noise-vs-signal
decomposition from the prior session (30-memory store, ~$970/85,463
episodes to resolve) is preserved as a standalone result in "Noise-vs-
signal decomposition" below, unaffected by this new finding.

## Status (2026-09-20, later still): effect-size analysis confirms the null is a design problem, not an estimator problem — zero new spend

`effect_size_analysis.py` quantifies what the mini ground truth's null
result already suggested: individual per-memory effects in the current
30-memory store are at or below the noise floor of any affordable real
data collection, by two independent lines of evidence (the population-wide
noise-vs-signal decomposition and the 4 direct ground-truth measurements
agree). Resolving the full 30-memory store at the effect size actually
found (~0.03) would cost **~$970** — not a viable next step at this
project's budget. See "Noise-vs-signal decomposition" below for the full
methodology, the redesign proposals (small deliberately-engineered memory
store, ranking vs. detection framings), and their cost/power trade-offs
— none of which have been run; `hard_cap_usd` and all frozen settings are
unchanged, no API calls were made for this analysis.

## Status (2026-09-20, later): mini ground truth run complete — result is a genuine null, not decisive for either estimator

User topped up OpenRouter credit to ~$17.80. `cost_control.hard_cap_usd`
raised to **18.00** (real buffer over the mini ground truth run's ~$15
estimated cost). **Mini ground truth phase spend: $12.6049** (133
forced-in/forced-out pairs each for 4 memories, real ALFWorld, `openai/
gpt-5.6-luna`). **Total real spend across the whole project: $14.8585**
($0.0628 model-selection test, separate cost-tracker file, + $14.7957 in
the shared production cost-tracker state: $0.1875 memory sanity check +
$2.0033 small pilot + $12.6049 mini ground truth). The decisive finding:
**all 4 ground-truthed memories show a statistically null effect** (95%
CI includes zero for every one), so this run does NOT decide between
Memory Worth and the propensity-corrected estimators — see "Mini ground
truth: results and verdict" below for the full, honest breakdown. The
full 10030-episode production run remains un-started and unaffordable as
originally scoped. See "Small pilot" below for the prior result (66%
success, 4 estimators run on 180 real episodes), "Ceiling-effect diagnosis
and fix" for why/how the task sampling changed, and "Step cap raised to
50" / "Real resource leak found + mitigated" / "Checkpointing and chunked
runs" for the infrastructure
that preceded it.

## Installing real ALFWorld (under WSL2 Ubuntu, Python 3.11)

The earlier native-Windows attempt (see git history) failed building
`jericho`/`textworld` for lack of a C toolchain. Under WSL2 Ubuntu with
`build-essential` installed, `pip install alfworld` (base install, no
`[full]`/`[vis]` extras — those pull in ai2thor/torch/opencv for the
embodied/visual THOR backend, which this text-only pilot never uses) built
cleanly. Then:

```bash
alfworld-download                              # ~2.3GB: game/PDDL files + logic grammar + MaskRCNN detector
                                                # (detector is unused by the text-only 'oracle' controller
                                                # but alfworld-download always fetches it)
```

`alfworld_pilot/configs/base_config.yaml` is the official repo's
`configs/base_config.yaml` (alfworld==0.4.2) fetched verbatim — do not
hand-edit its structure. Paths inside it use `$ALFWORLD_DATA`, which
`alfworld.info` sets to `~/.cache/alfworld` automatically on import.

### Python version: rebuilt on 3.11, not 3.14 (2026-09-19)

The repo's venv was originally built on Python 3.14 (2026-09-18), which
required monkeypatching a real incompatibility: `textworld` 1.7.0's PDDL
grammar engine relies on a `locals()`-mutation trick that Python 3.13's
PEP 667 ("Consistent views of namespaces") permanently broke, raising
`NameError: name 'r' is not defined` on the very first `env.reset()`.
ALFWorld's own docs only ever claim "Python 3.9+" support and its quickstart
pins `python=3.9` — nothing about the package targets 3.13+. Rather than
carry a standing monkeypatch of third-party library internals, the venv was
rebuilt on **Python 3.11** (installed via `uv python install 3.11` —
user-space, no sudo needed, since this Ubuntu release is too new for
system-repo 3.11/3.10 packages and `sudo apt` needs an interactive
password this session doesn't have): `rm -rf .venv && uv venv --python 3.11
.venv`, then reinstalled everything (`ensurepip` first — uv-built pythons
don't ship pip; then `python -m pip install -r requirements.txt -r
alfworld_pilot/requirements.txt` and `pip install -e .` for `memory_ope`).
Confirmed empirically: the exact same quickstart that raised `NameError` on
3.14 runs clean on 3.11 with **no patch of any kind**. The old patch module
(`_textworld_py313_compat.py`) has been deleted; `RealAlfredEnv.__init__` no
longer references it.

### Two real bugs in this package's own code, found only once real ALFWorld was exercised (2026-09-18)

Both were written against the *documented* real API before ALFWorld was
actually installed/runnable, so neither had ever been executed end to end:

1. `RealAlfredEnv.__init__` did `getattr(alfworld_env, env_type)(...)`, but
   `alfworld.agents.environment.get_environment(env_type)` imports the
   requested class *locally* inside that function — it's never a
   module-level attribute — so the `getattr` always raised `AttributeError`.
   Fixed to call `get_environment(env_type)(...)` directly.
2. **Double `env.reset()` per episode**: `episode_runner.run_logged_episode`
   called `env.reset(task_seed=task_id)` once (to get `task_type` for
   memory retrieval), then handed `env` to `react_agent.run_episode`, which
   called `env.reset()` again internally, discarding the first reset's
   state. For `MockAlfredEnv` this silently played a task *different* from
   the one retrieval scored (mock's seedless `reset()` draws from a
   separate, persistently-advancing RNG); for real ALFWorld it would have
   silently consumed two games per episode and mismatched retrieval against
   the actually-played task every time. Fixed by having `run_episode` take
   the already-fetched `obs`/`info` as parameters instead of resetting
   itself — there is now exactly one `reset()` per episode.

Also found: `extra.gamefile` (used to derive `task_type`) is only populated
by TextWorld on `reset()`, not on every `step()` — it comes back `None`
mid-episode. `RealAlfredEnv` now caches `task_type` from `reset()` and
reuses it in `step()`'s returned info instead of re-deriving it each time.

### Paired ground truth against real ALFWorld: SOLVED (2026-09-19)

Real ALFWorld's `env.reset()` ignores `task_seed` and just advances to the
next game in an internal sequence, so a single shared env can't be reset
back to a specific task instance twice — but a game can be loaded directly.
Verified empirically first (before writing any pipeline code): registering
ONE specific `game.tw-pddl` file via `textworld.gym.register_games([gamefile],
...)` and resetting it gives **byte-identical** initial observations and
admissible-commands lists across (a) two independently-constructed envs
pointed at the same file, (b) two resets of the same env, and (c) stepping
identically on both. `RealAlfredEnv` now accepts an optional `gamefile_path`
that restricts it to exactly one game (built via `AlfredTWEnv.__new__` +
its own real `init_env()`, skipping the expensive full-dataset
`collect_game_files()` walk — not a reimplementation of `init_env`, the
actual method).

`ground_truth_runner.py` now abstracts backend differences behind a small
`TaskSource` interface (`task_type(task_seed)`, `build_env(task_seed)`):
`MockTaskSource` wraps `MockAlfredEnv` (unchanged behavior); `RealTaskSource`
walks the split's game-file list ONCE (`list_real_game_files`, cached) and
maps a `task_seed` to a specific file by index, so the SAME index always
means the SAME file — forced-in and forced-out (and repeated candidacy
probes) agree by construction, with no reliance on real ALFWorld supporting
seeking. `env_factory.build_task_source(cfg)` picks the right one from
`config.yaml`'s `env.backend`, resolving the real ALFWorld config the same
way `build_env_factory` already does.

Verified end to end against real ALFWorld (`MockLLMClient`, zero API
spend): forced-in/forced-out episodes had identical `candidate_ids`,
identical inclusion for every OTHER memory, identical `task_type`, and
identical first-step observation + admissible actions — i.e. the exact
same task instance, only the target memory's inclusion differing. A fast,
dependency-free unit test (`test_real_task_source_maps_task_seed_to_stable_
gamefile_and_task_type`) covers the index-to-file mapping and path-based
task-type parsing without needing alfworld installed.

**Operational note discovered while testing this**: constructing many
single-game `RealAlfredEnv`s without releasing them grows memory
substantially (observed ~1.25GB after ~100 unclosed constructions, in a
run that was killed before finishing) — NOT from `asynchronous=True`
spawning subprocesses (that flag is a documented no-op at `batch_size=1`);
the exact retained-memory source wasn't root-caused further, but calling
the now-added `RealAlfredEnv.close()` after each single-game episode keeps
growth far more modest. `run_ground_truth_pair` and
`task_type_difficulty_check.py` both close every env they construct.
**Since Part C's real ground-truth phase will construct on the order of
~10,000 single-game envs (4980 episodes × ~2 arms, plus repeated probes),
monitor memory during that run and consider chunking it (restart the
process every N pairs) if growth reappears despite closing.**

### Task types: coverage confirmed, real difficulty spread measured at zero cost (2026-09-19)

`episode_runner.py` logs `task_type` per episode already; confirmed against
the REAL dataset (not just the mock) that all 6 official ALFWorld task
types are present and correctly parsed from file paths (`alfworld_pilot.
task_type_difficulty_check.check_task_type_coverage`; population counts in
`results/task_type_difficulty_check.json`).

Difficulty spread, measured with ALFWorld's own built-in scripted expert
(`info['extra.expert_plan']`, surfaced automatically whenever
`general.training_method: dagger` — zero LLM calls, n=30 games/type):

| task_type | mean steps to solve | median | % solvable within our 30-step cap |
|---|---|---|---|
| pick_and_place_simple | 13.9 | 12 | 90% |
| look_at_obj_in_light | 13.1 | 10 | 90% |
| pick_clean_then_place_in_recep | 22.2 | 18 | 80% |
| pick_heat_then_place_in_recep | 22.8 | 19 | 73% |
| pick_cool_then_place_in_recep | 16.9 | 15 | 90% |
| pick_two_obj_and_place | 43.6 | 50 | **13%** |

**This is exactly the confound this pilot exists to correct for, and it's
large and real**: `pick_two_obj_and_place` needs roughly 3x the steps of the
easiest task types, and at `env.max_steps: 30` an *optimal* scripted policy
can only finish it 13% of the time — meaning any agent's measured success
rate on that task type is dominated by the step budget, not by which
memories it had. Since a task's type also determines which memories are
similarity-relevant candidates (memory_store.py's `LESSON_TEMPLATES` are
one-per-task-type), this couples task difficulty to which memories get
retrieved — precisely the task_difficulty simulator's confound, reproduced
in real data. Left the 30-step cap unchanged (an explicit prior cost-control
choice); flagging this rather than changing it, since raising it changes
the cost projection and is a call worth making deliberately, not as a side
effect of this check.

## Running

```bash
# from the repo root, ONE shared venv (see ../pyproject.toml) -- memory_ope
# is pip-installed editable, so retrieval_shared.py needs no sys.path shim.
source .venv/bin/activate
cd alfworld_pilot
PYTHONPATH=src python -m pytest tests/ -v                    # 18 tests, all against MockAlfredEnv (or dependency-free), zero cost
PYTHONPATH=src python -m alfworld_pilot.measure_mode          # real ALFWorld env (config.yaml default) + MockLLMClient,
                                                                # zero API cost -- reports real tokens/call, calls/episode,
                                                                # and a full-plan cost projection
PYTHONPATH=src python -m alfworld_pilot.token_breakdown        # per-section synthetic token breakdown (still mock env by design)
PYTHONPATH=src python -m alfworld_pilot.task_type_difficulty_check   # real-data task-type coverage + zero-cost expert-driven difficulty proxy
```

## Design

```
src/alfworld_pilot/
  env_interface.py       AlfredEnv protocol, MockAlfredEnv, RealAlfredEnv (both runnable; RealAlfredEnv
                           optionally restricted to one gamefile_path for paired ground truth)
  env_factory.py           picks Mock/Real env AND task-source from config.yaml's env.backend
  memory_store.py         30 short "lessons" (100-300 tokens), mock topic-aware similarity
  retrieval_shared.py      re-exports Stage 1's EXACT retrieval mechanism (top-M, rank-linear
                           propensity, independent Bernoulli inclusion) -- a plain import of the
                           pip-installed memory_ope package, not a reimplementation
  llm_client.py            OpenRouterClient (pinned model id, reasoning disabled, cache +
                           cost-cap wired in) and the LLMClient protocol it shares with mock_llm
  mock_llm.py              MockLLMClient: scripted_success / random / malformed strategies,
                           zero cost, parses the same prompt text a real model would see
  cache.py                 disk cache keyed by a hash of the full request payload
  cost_tracker.py          cumulative spend vs. hard_cap_usd, checked BEFORE every real call
  determinism_check.py     two real duplicate calls, compares text -- verifies temperature=0
                           (+ seed) actually holds, doesn't assume it
  react_agent.py           one LLM call per step (Thought+Action together), 30-step cap
  episode_runner.py        ties retrieval + agent + logging into one episode dict
                           (Stage-1-compatible schema + task_type/tokens/cost fields)
  ground_truth_runner.py   paired forced-in/forced-out via a TaskSource abstraction
                           (MockTaskSource / RealTaskSource) that hides how each backend
                           guarantees "the same task instance" between arms; restricted to
                           task instances where the target memory is a NATURAL candidate
                           (matches the causal oracle's own definition)
  power_analysis.py        redoes Part A.3's sample-size calc with a MEASURED rho (from
                           ground_truth_runner.paired_correlation on real results)
  measure_mode.py          runs N episodes, records real per-call tokens, projects full-plan cost
  token_breakdown.py       decomposes one call's prompt into system/memories/history/actions
  task_type_difficulty_check.py  confirms 6-task-type coverage + a zero-cost (ALFWorld's own
                           scripted expert) difficulty-spread proxy per task type
  step_cap_check.py        sweeps step-cap solvability (30/40/50) from ONE expert run per
                           task type, reused from task_type_difficulty_check.py
  checkpointed_runner.py   resumable logging/ground-truth phases (JSONL append+flush,
                           per-task_id seeding) -- a chunk boundary and a mid-chunk crash
                           are handled identically
  run_chunked.py           subprocess-per-chunk supervisor built on checkpointed_runner.py --
                           needed because real ALFWorld leaks ~32.5MB per new game load
                           (see README's "Real resource leak" section) and only a process
                           restart reclaims it
  runtime_estimate.py      wall-clock estimate for the full plan (measured local overhead +
                           an assumed, pre-Part-C LLM latency)
  model_selection_test.py  Part C's real-spend model comparison (5 episodes x N candidate
                           models, one shared cost-capped CostTracker)
```

Settings (30 memories, propensity range [0.3, 0.7], M=10 unchanged) come
from `signal_boost_check.py`'s recommendation in the root package — the
only setting where a propensity-corrected estimator's 95% interval excludes
zero at Stage 2's realistic episode count.

## Scaled-up plan (2026-09-19)

Real-measured cost ($4-15 for the original 3050-episode plan) came in far
under the $100 budget, and `scale_up_check.py` (root package) showed
IPS/SNIPS/DR's 95% Spearman interval only reliably excludes zero from
n=3000 episodes onward (task_difficulty AND hitchhiker simulators, at this
pilot's actual 30-memories/[0.3,0.7]-propensity setting) — n=1000 left plain
IPS's interval still crossing zero. Re-planned:

| | old plan | new plan |
|---|---|---|
| memory_store_construction | 50 | 50 (unchanged, one-time cold start) |
| main_randomized_logging | 1000 | **5000** |
| ground_truth_reruns | 2000 (10 memories x 100 pairs x 2 arms) | **4980** (15 memories x 166 pairs x 2 arms) |
| **total episodes** | 3050 | **10030** |

Ground-truth pairs-per-memory came from rescaling `ground_truth_power_
analysis.py`'s trade-off table to a 5000-episode rerun budget (see
`ground_truth_power_analysis_rescale.py` / `results/
ground_truth_power_analysis_rescale.json`): 15 memories x 166 pairs gives
MDE~0.137 at 80% power, better than the original plan's 10-memory/100-pair
MDE~0.177 on BOTH axes (more memories covered AND tighter resolution).

Cost projection, using the SAME real-env-measured per-call token rates as
before (1054.7 input / 14.0 output tokens/call, 30.0 calls/episode — these
don't change with the plan, only the episode count does):

| model | old plan cost | new plan cost |
|---|---|---|
| deepseek-v4.1-flash | $15.24 | **$50.13** |
| ling-3.0-flash-vl | $6.02 | **$19.80** |
| mercury-2.5 | $4.05 | **$13.33** |

All still under the $100 budget (deepseek-v4.1-flash uses about half of
it). `cost_control.hard_cap_usd` raised from 10.0 to 75.0 accordingly —
enough margin for a real tokenizer counting more tokens than this
projection's word-count proxy, while still stopping well short of $100.
`config.yaml`'s `episode_plan` and `ground_truth` sections are the
authoritative, up-to-date plan; the root package's `stage2_budget_
estimate.py` is an OLDER, pre-real-ALFWorld assumption-only estimate,
superseded by this real-measured one and kept only as a historical
reference point (its own docstring already says as much).

## Step cap raised 30 -> 50 (2026-09-19)

`step_cap_check.py` (reuses one `task_type_difficulty_check.py` expert run,
evaluated at several candidate caps rather than re-running the expert per
cap) swept solvability at 30/40/50 steps, n=40 games/type:

| task_type | cap=30 | cap=40 | cap=50 |
|---|---|---|---|
| pick_and_place_simple | 92% | 95% | 100% |
| look_at_obj_in_light | 85% | 90% | 100% |
| pick_clean_then_place_in_recep | 80% | 88% | 100% |
| pick_heat_then_place_in_recep | 78% | 85% | 100% |
| pick_cool_then_place_in_recep | 78% | 80% | 100% |
| pick_two_obj_and_place | **18%** | **22%** | **100%** |

At 30 (the original cost-controlled choice), `pick_two_obj_and_place` was
solvable by an OPTIMAL policy only 18% of the time — not a difficulty gap,
a near-guaranteed structural failure regardless of memories. 40 barely
helps (22%) — the type's median solve length is right around there. Only
50 (ALFWorld's own standard research cap) brings every type to ~100%
optimal-policy solvability, while still leaving a real difficulty gap for a
fallible real agent (more steps = more chances to err) — the kind of
confound this pilot's OPE method is built to correct for, not "impossible
by construction". `env.max_steps` raised to 50; re-measured real-env cost
(2 real episodes, mock LLM): avg 1235.5 input / 14.1 output tokens/call,
50.0 calls/episode (both hit the now-higher cap) — full 10030-episode plan
now projects to **$25.85 (mercury-2.5) - $97.19 (deepseek-v4.1-flash)**,
notably higher than at cap=30 ($13-50). `cost_control.hard_cap_usd` raised
75.0 -> 90.0 accordingly. See "Part C: model selection test results" below
for the REAL (non-word-count-proxy) version of this projection.

## Real resource leak found + mitigated: fast_downward's un-closed dlopen (2026-09-19)

Running `task_type_difficulty_check.py`/`step_cap_check.py` at a larger
sample (n=40/type = 240 single-game env constructions) crashed twice with
`OSError: [Errno 28] No space left on device`. Root cause, confirmed by
reading the traceback into `fast_downward` (a `textworld`/`alfworld`
dependency, upstream code): `fast_downward.interface.load_lib()` copies its
~32.5MB native shared library to a **fresh** temp directory and `dlopen()`s
it on **every** PDDL game load (i.e. every `env.reset()`/`.load()` to a NEW
game, regardless of whether you reuse one long-lived env or build a fresh
one per episode) — the temp directory is cleaned up right after (so `ls
/tmp` shows nothing), but the loaded library is never `dlclose()`d, so its
pages stay resident for the process's entire lifetime. This machine's
`/tmp` is a 5.8GB tmpfs (RAM-backed) — at ~32.5MB/load, ~178 game loads
exhausts it. Confirmed the leak is scoped to the PROCESS, not permanent:
killing the crashed process immediately dropped tmpfs usage from ~5.4GB
back to ~500KB.

Two complementary mitigations (both needed for the ~10,000 game loads
Part C's full plan will make):
1. **Point `TMPDIR` at a disk-backed directory** (e.g. `export
   TMPDIR=/home/ibo/tmp_downward`, on this machine's 951GB-free ext4 root,
   not the 5.8GB tmpfs `/tmp`) before running anything real-ALFWorld. Moves
   the ceiling from ~178 to ~29,000 game loads — doesn't fix the leak, just
   gives it a much bigger tank. Used for every real run in this session.
2. **`run_chunked.py`** (new): a subprocess-per-chunk supervisor built on
   `checkpointed_runner.py`'s resumability — a chunk boundary and a
   mid-chunk crash are handled identically (both just resume from the log's
   current length), so restarting a fresh subprocess every `--chunk-size`
   (default 100, comfortably under the ~178-load tmpfs ceiling as a second
   line of defense even without the TMPDIR fix) episodes/pairs is free
   correctness-wise. Verified end to end with a real subprocess-spawning
   integration test (mock backend, zero cost).

## Checkpointing and chunked runs (2026-09-19)

`checkpointed_runner.py` (new) makes both phases resumable:
- `run_logging_phase_checkpointed`: appends one JSON line per completed
  episode, flushing immediately; resume point = the log's current line
  count. Each episode's randomization is now seeded as a pure function of
  its `task_id` (not one shared RNG stream advancing across the whole run)
  so a crash+resume run reproduces byte-identical episodes to an
  uninterrupted one, verified by a direct test.
- `run_ground_truth_phase_checkpointed`: same idea per memory, with a
  small JSON state file tracking `(pairs_found, next_seed_to_probe)`.
- Both accept `max_new` to cap one call's work, which is what
  `run_chunked.py`'s subprocess supervisor uses to bound each chunk.

The other two things that need to survive a restart, alongside the above:
- **Cost cap**: `CostTracker` now takes an optional `state_path` and
  persists (atomic write) after every call/cache-hit, reloading on
  construction. Without this, a restarted process would start counting
  spend from $0 and could let real cumulative spend across a crash exceed
  `hard_cap_usd` without any single call looking, in isolation, like it
  crosses the line. Verified with a test that simulates exactly that
  restart.
- **Cache**: already file-per-request on disk (`cache.py`) — nothing to
  change, confirmed nothing in it was ever held only in memory.

## Wall-clock runtime estimate (2026-09-19, pre-Part-C)

`runtime_estimate.py`: local overhead (env construction/stepping, measured
directly with the mock LLM so it isolates from network latency) is
~3.67s/episode (long-lived env) + ~0.52s/episode extra for the
chunked/resumable design's fresh-env-per-episode pattern ≈ 4.19s/episode,
totaling ~11.7 hours across the full 10030-episode plan. LLM call latency
(NOT measured before Part C — a stated assumption of 1.5s/call, worst-case
50 calls/episode) would add ~209 hours — **LLM latency dominates total wall
time by roughly two orders of magnitude**. Serial execution of the full
plan at these assumptions: ~220 hours (~9 days). Not attempted to fix in
this session (would mean parallelizing across several worker processes
sharing one cost-tracker/cache) — flagged as worth doing before the real
production run, now that Part C's actual measured latency (below) is
available to refine this estimate.

## Part C: model selection test results (2026-09-19) — REAL SPEND, $0.063 total

`model_selection_test.py`: 5 real ALFWorld episodes per candidate model,
all 3 models facing the SAME 5 games (a fresh long-lived env per model
restarts real ALFWorld's sequential game order from the start, so this
falls out for free rather than needing special handling), one shared
disk-persisted `CostTracker` hard-capped at $10 total across all 3 models
combined. Model ids/pricing fetched from `https://openrouter.ai/api/v1/models`
on 2026-09-19 (re-check before reuse) — note `inclusionai/ling-3.0-flash-vl`
has a `:free` variant too; used the PAID one as specified.

| model | result | success rate | avg in/out tokens/call | calls/episode | parse failures | spend (5 ep) | full-plan (10030 ep) projection |
|---|---|---|---|---|---|---|---|
| `openai/gpt-5.6-luna` | OK | 60% (3/5) | 1594.2 / 33.3 | 25.6 | 5 | $0.0459 | **$92.13** |
| `inclusionai/ling-3.0-flash-vl` | OK (after 1 retry) | 60% (3/5) | 1718.9 / 74.2 | 29.0 | 40 | $0.0046 (+$0.0122 on the failed attempt) | **$33.88** |
| `google/gemini-3.8-flash` | **FAILED, 0 episodes** | — | — | — | — | $0.0000 | — |

Notes:
- **gemini-3.8-flash failed immediately**: `400 Reasoning is mandatory for
  this endpoint and cannot be disabled` — a real, clean incompatibility
  with this pilot's `reasoning_enabled: false` design (see llm_client.py's
  docstring; this is the exact failure mode it was written to surface, not
  swallow). Not retried with reasoning enabled — that's a different cost/
  latency profile and a real design decision, not something to silently
  change and re-run.
- **ling-3.0-flash-vl failed once, transiently**: a `429` from the
  upstream provider (DeepInfra, via OpenRouter's shared pool) after 2
  episodes — `model_selection_test.py`'s per-model exception handling
  caught it, reported it, and moved on to the next candidate rather than
  crashing the whole test. Retried once and it succeeded; the LLM cache
  made the retry cheap (gpt-5.6-luna's already-completed calls replayed
  from cache at $0.0000, ling-3.0-flash-vl resumed real calls only from
  episode 2 onward).
- **Both working models actually solved 3/5 real ALFWorld tasks** at
  temperature=0 with only 30 generic short "lesson" memories (not tuned to
  these specific games) — a first real (if tiny-sample) signal that the
  agent loop and memory format work, not just that the plumbing runs.
- **avg_calls_per_episode came in well under the 50-call worst case**
  (25.6, 29.0) BECAUSE some episodes succeeded and ended early — real
  full-plan projections ($92.13, $33.88) are correspondingly lower than the
  mock-LLM word-count-based projection at the same cap ($97.19 for the
  comparable model class).
- **ling-3.0-flash-vl's parse-failure rate (40/145 ≈ 28%) is much higher
  than gpt-5.6-luna's (5/128 ≈ 4%)** — a real, decision-relevant quality
  difference between the two working candidates, worth weighing against
  ling's much lower cost ($33.88 vs. $92.13 for the full plan).
- Total real spend across the whole test (both attempts): **$0.0628** —
  the $10 stop-if-exceeded cap was never close to being tested by real
  usage; what it DID genuinely exercise was the per-model exception
  handling for two different real failure modes.

## Model picked: openai/gpt-5.6-luna (2026-09-19)

User's real remaining OpenRouter balance dropped to $3.84, forcing a
decision now rather than running more comparison. Picked
`openai/gpt-5.6-luna` over `inclusionai/ling-3.0-flash-vl` specifically
because of the latter's 28% parse-failure rate (vs. gpt-5.6-luna's ~4%):
**a format-failure rate that might scale with prompt size would scale with
how many memories got included, which would confound the exact causal
contrast this pilot measures** — a memory's measured effect could partly
just be "did the agent's output happen to parse this time", correlated
with retrieval rather than with the memory's real usefulness. Cost
($33.88 vs. $92.13 full-plan, per the model-selection test) was the
opposite direction, but reliability came first — a cheaper but confounded
signal isn't worth having. `google/gemini-3.8-flash` isn't a candidate at
all (fails outright, mandatory reasoning). `config.yaml`'s `llm.model_id`,
`pricing_per_million_tokens`, and `cost_control.hard_cap_usd` (90 -> 3.00)
all updated to match.

## Memory sanity check results (2026-09-19) — REAL SPEND, $0.19

`memory_sanity_check.py`: 15 PAIRED episodes (30 total) on real ALFWorld,
`openai/gpt-5.6-luna`. Each pair uses the SAME real ALFWorld game for both
arms (via `RealAlfredEnv`'s `gamefile_path`) and the SAME rng seed, so
candidate_ids/similarity/propensities are byte-identical between arms —
only whether the drawn memories are actually included differs:
- **with_memories**: normal randomized inclusion (a real logging episode)
- **baseline**: every candidate forced to `included=0` — no memories at all

| | success rate |
|---|---|
| with memories | **93%** (14/15) |
| baseline (no memories) | **80%** (12/15) |

Of the 15 pairs, 13 were concordant (same outcome both arms); of the 2
discordant pairs, **both** went with_memories=success / baseline=failure,
none the other way. That's a small, directionally consistent effect, not a
statistically decisive one at n=15 (McNemar-style exact test on 2
discordant pairs, both favoring memories: p=0.25, not significant) — but
it answers the sanity-check question the way that makes the larger pilot
worth running: **memories are not doing nothing.**

- **(b) parse failures vs. memory count / prompt length**: correlation(n_
  included memories, parse_failures) = **0.008** across all 30 episodes —
  essentially zero. correlation(avg prompt tokens/call, parse_failures) =
  **0.17** — weak, and likely driven by a few individual GAMES that were
  hard for the model regardless of arm (e.g. pair 6: 43 parse failures with
  memories AND 38 without, on the same game) rather than by memory count
  itself. No evidence so far that format failures scale with how many
  memories got included.
- **(c) real tokens/call**: 1035.0 in / 34.9 out, pooled across both arms.
  Lower than the model-selection test's 1594.2 in for gpt-5.6-luna, because
  that number was WITH-memories-only episodes on different games; THIS
  number is diluted by the baseline arm's shorter, memory-free prompts —
  **not directly comparable to what the real production run's tokens/call
  will look like**, since production episodes are (almost) all
  with-memories, not a 50/50 mix. Step 2's larger pilot (no baseline arm)
  will give a cleaner number.
- **(d) updated full-plan cost projection**: $62.90 for the full
  10030-episode plan — again pooled across both arms, so likely an
  UNDER-estimate of real production cost (baseline episodes are cheaper).
  Treat step 2's projection as more reliable once available.

Real spend: **$0.1875** for this step (target was ~$0.30) — cumulative
shared production spend now $0.1875 / $3.00 hard cap. Not stopping per the
"if memories don't change success at all" condition, since they clearly do.

## Ceiling-effect diagnosis and fix (2026-09-19) — zero new spend for the diagnosis

93% success is too close to 100% to resolve individual memory effects: the
ground-truth power analysis can resolve deltas of ~0.10-0.15 at this
pilot's real budget, and any true per-memory effect gets mechanically
squeezed toward 0 once the base rate is pinned near the ceiling. Two
findings, both from data already in hand (no new API spend):

**(1) Per-task-type breakdown of the 15-pair sanity check**
(`memory_sanity_check_breakdown.py`, recovers `task_type` per pair from
the deterministic game-index mapping — no LLM calls needed):

| task_type | n pairs | with-memories success | baseline success |
|---|---|---|---|
| `pick_and_place_simple` | 13 | 13/13 (100%) | 11/13 (85%) |
| `pick_two_obj_and_place` | 2 | 1/2 (50%) | 1/2 (50%) |
| (4 other types) | 0 | — | — |

**Root cause: an accidental sampling bias, not general task ease.** The
sanity check picked games via `task_id % len(game_files)` over
`list_real_game_files`'s raw filesystem-walk order — and the first 15
indices in that order happen to land 13/15 on `pick_and_place_simple` (the
EASIEST type by every difficulty measure so far) and only 2/15 on
`pick_two_obj_and_place` (the hardest), with zero samples from the other 4
types entirely. The pooled 93%/80% figures are really "93%/80% on an
84%-easy-type sample," not a representative measurement. (All 2 discordant
with/baseline pairs from the earlier report were also both
`pick_and_place_simple` — the "memories help" signal so far comes entirely
from the easy type; `pick_two_obj_and_place` had too few samples, 2, to
say anything about it specifically.)

**(2) Three options to reach 50-70% overall success:**

| option | mechanism | cost | verdict |
|---|---|---|---|
| A. Unseen/harder ALFWorld split | `real_split: eval_out_of_distribution` | Zero code change, but **uncertain benefit for THIS agent**: ALFWorld's seen/unseen split distinction is about which object/scene combinations an RL-trained policy saw during training. A zero-shot LLM ReAct agent never trains on any split — no clear reason it would find "unseen" scenes harder. No data to predict an effect size. | Not recommended as the primary fix; possibly worth layering on later. |
| B. Lower `env.max_steps` (35 or 40) | Direct cap reduction | **Quantified, and bad**: `step_cap_check.py` re-run with cap=35 added: `pick_two_obj_and_place` solvability is 10% (cap=30) -> 12.5% (cap=35) -> 17.5% (cap=40) -> 100% (cap=50) by an OPTIMAL policy — every other task type stays 72-95% across 30-40. Lowering the cap re-breaks exactly ONE task type back to near-total structural failure while barely touching the others — reintroducing the precise confound Part C's step-cap fix (30->50) existed to remove. | **Rejected.** |
| C. Weight task-type sampling toward harder types | New `weighted_task_source.WeightedRealTaskSource`: draws task type per `task_id` with probability proportional to that type's mean steps-to-solve (13.1-43.6, from `task_type_difficulty_check.py`'s real n=30/type measurement), then a game of that type — both draws pure functions of `task_id`, composing with existing checkpointing unchanged. | Moves the logged population away from ALFWorld's natural task-type proportions — a real trade-off for a deployment benchmark, much less so for a methods pilot whose whole premise is already "does OPE see through a task-type-correlated confound" (a bigger, deliberate confound is a HARDER test of that, not an invalid one). Keeps `env.max_steps=50`, so the structural-failure fix stays intact. | **Recommended — used for the small pilot below.** |

## Small pilot (2026-09-19) — REAL SPEND, $2.00

`small_pilot.py`: logged real ALFWorld episodes via `WeightedRealTaskSource`
(recommended option C above), checkpointed, stopping at its $2.00 soft
budget (shared cost-tracker state, hard cap unchanged at $3.00). **180
episodes**, spend **$2.0033** this step (cumulative $2.1908 / $3.00).

**The fix worked**: overall success rate **66%** — squarely in the target
50-70% range, not pinned near ceiling. Task-type distribution actually
achieved: `pick_two_obj_and_place` 51, `pick_heat_then_place_in_recep` 33,
`pick_clean_then_place_in_recep` 32, `look_at_obj_in_light` 28,
`pick_and_place_simple` 21, `pick_cool_then_place_in_recep` 15 — all 6
types represented (vs. 2/6 in the naive-indexed sanity check), skewed
toward harder types as designed.

Ran all four estimators (`memory_ope.estimators`) on the resulting log
against all 30 memories — every memory got a defined estimate from every
estimator, no NaNs:

| pair | Spearman |
|---|---|
| memory_worth vs ips | **-0.164** |
| memory_worth vs snips | 0.291 |
| memory_worth vs doubly_robust | 0.290 |
| ips vs snips | 0.341 |
| ips vs doubly_robust | 0.257 |
| **snips vs doubly_robust** | **0.922** |

**What this does and doesn't support**: Memory Worth's ranking is
*negatively* correlated with plain IPS's and only weakly positively
correlated with SNIPS/DR — while SNIPS and DR agree with each other very
strongly (0.922). That's the qualitative pattern this whole pilot exists
to detect (MW behaves differently — and, per Stage 1's simulator work,
often *worse* — under a task-difficulty-like confound, while the
variance-reduced propensity-corrected estimators cluster together) showing
up in a REAL log for the first time. **But**: Stage 1's `small_data_
results.py` found that even in the BEST-CASE simulated setting, Spearman
correlations are too noisy to be conclusive below roughly 1000-2000
episodes — bias is already near-zero by n=250, but rank correlations
swing widely seed-to-seed until much larger n. At n=180 real episodes,
**this result is suggestive and consistent with the pilot's hypothesis,
not a statistically decisive confirmation of it** — a different 180-episode
sample could plausibly show a different correlation pattern. No ground
truth exists yet for real ALFWorld's per-memory values (that needs the
ground-truth phase), so there's also no way yet to say which estimator's
ranking is actually more CORRECT here, only that they disagree in the
direction the pilot's hypothesis predicts.

Top-5 / bottom-5 memories by doubly_robust (illustrative, not a reliable
ranking at this n): estimates spread from -0.32 to +0.18 (a plausible
causal-contrast scale, distinct from Memory Worth's ~0.4-0.9 raw-rate
scale, consistent with every prior Stage 1 finding about the two living on
different scales). Full per-memory numbers: `results/small_pilot.json`.

## Mini ground truth: selection (2026-09-20) — zero new spend

`mini_ground_truth_selection.py`: the decisive check of the project --
when Memory Worth and IPS disagree, which one is actually right? Used the
same fixed split-by-episode-index design already validated (root
package's `ground_truth_selection_bias_check.py`, which demonstrated a
~1.87x selection-bias inflation from picking-and-evaluating on the same
data): the small pilot's 180 episodes split into set A (first 90,
selection only) and set B (second 90, "official" estimates only).
Selected the top-5 memory_worth-vs-ips **rank** disagreement memories from
set A (percentile rank within each estimator's own distribution, since MW
lives on a ~0.4-0.9 scale and IPS on a ~-0.5 to 0.4 scale): `mem_17`,
`mem_29`, `mem_21`, `mem_9`, `mem_10`.

Real per-task-type cost (measured from the small pilot's actual
`total_input_tokens`/`total_output_tokens`, not projected) turned out to
matter a lot: 2 of the 5 (`mem_17`, `mem_29`) are tagged
`pick_two_obj_and_place` (~$0.009/episode in practice); 2 more (`mem_21`,
`mem_9`) are tagged `pick_heat_then_place_in_recep`, which turned out to
be the MOST expensive type in practice (~$0.019/episode) despite NOT
being the hardest by the scripted-expert difficulty measure -- a real
agent's actual behavior doesn't track the optimal-policy difficulty
ranking. Full statistical power (133 pairs/memory for MDE=0.15, per the
existing power analysis, using p=0.66 real measured success rate and
rho=0.154 simulator-proxy) across all 5 memories would have cost ~$19.36,
exceeding the ~$13.81 available under the cap at the time. **User's
decision: drop `mem_10` (weakest of the 5 disagreements) and keep full
133-pair power on the remaining 4**, rather than dilute power across all
5 -- raised `hard_cap_usd` to 18.00 to afford it (~$15.00 corrected
estimate, using real per-memory task-type costs, not the blended
average).

## Mini ground truth: results and verdict (2026-09-20) — REAL SPEND $12.60

Ran via `run_chunked.py`'s subprocess-per-chunk supervisor (already-built,
already-tested `checkpointed_runner.run_ground_truth_phase_checkpointed`
+ `RealTaskSource`), one memory at a time in cost order (cheapest first,
so a cap-triggered stop would leave complete results for some memories
rather than partial for all): `mem_17`, `mem_29` (133/133 pairs each,
$2.68 / $2.70), then `mem_21`, `mem_9` (133/133 pairs each, $3.58 / $3.03
-- both came in BELOW the ~$5.1-5.8 worst-case estimate). Total: **1064
real ALFWorld episodes, $12.6049**, well under the $18.00 cap.

Ground truth (paired mean(Y_forced_in) - mean(Y_forced_out), 95% CI using
the REAL correlation rho measured from these paired episodes -- not the
old task_difficulty-simulator proxy) vs. the "official" set-B estimates:

| memory | ground truth gap [95% CI] | rho | memory_worth (rank) | ips (rank) | snips (rank) | doubly_robust (rank) |
|---|---|---|---|---|---|---|
| `mem_17` | +0.0000 [-0.066, +0.066] | 0.632 | 0.789 (#8) | 0.108 (#10) | 0.143 (#7) | 0.064 (#11) |
| `mem_29` | +0.0301 [-0.036, +0.096] | 0.628 | 0.762 (#13) | 0.155 (#7) | -0.004 (#14) | 0.036 (#14) |
| `mem_21` | +0.0301 [-0.039, +0.099] | 0.666 | 0.500 (#25) | 0.247 (#6) | -0.027 (#16) | 0.044 (#12) |
| `mem_9` | -0.0226 [-0.096, +0.051] | 0.616 | 0.400 (#29) | -0.206 (#21) | -0.126 (#25) | -0.120 (#20) |

**All four 95% CIs include zero.** Real measured rho (0.62-0.67) is much
higher than the simulator-derived planning proxy (0.154) -- makes sense,
since forcing the SAME memory in/out while holding the game, every other
memory's inclusion, and the retrieval draw fixed leaves a lot of shared
context between arms. The four ground-truth point estimates themselves
span only **0.053** (-0.023 to +0.030) -- smaller than any single
estimate's own confidence interval, and far below the ~0.15-0.19 MDE this
design was built to resolve.

Pairwise concordance with ground truth's ordering (6 possible pairs among
4 memories, 1 exact tie between `mem_29` and `mem_21`'s ground-truth gaps
excluded, 5 comparable pairs): **ips 5/5**, memory_worth 3/5, snips 3/5,
doubly_robust 3/5.

### Verdict: too noisy to tell -- and that itself is the finding

**Not "IPS wins."** IPS's perfect pairwise concordance is numerically the
best of the four, but the ground-truth differences it's being compared
against are not statistically distinguishable from EACH OTHER OR FROM
ZERO -- the entire spread of real causal effects among these 4 memories
(0.053) is smaller than the noise band on any one of them. Ordering four
values that are indistinguishable-from-flat by a metric that itself might
share the same selection-driven noise (IPS's own estimates on set A are
what flagged these memories as "disagreements" in the first place) is not
strong evidence for that metric being generally more accurate --
concluding otherwise here would be exactly the kind of overstatement the
user explicitly asked this report not to make.

**The more informative, honest conclusion**: none of the 4 memories
flagged as "biggest Memory-Worth-vs-IPS disagreement" in a 90-episode
half of the small pilot turned out to have a resolvable real causal
effect. The likely mechanism is the SAME selection-bias-inflation effect
`ground_truth_selection_bias_check.py` already demonstrated on synthetic
data (~1.87x): picking "biggest disagreement" from a small, noisy sample
tends to select memories whose apparent disagreement was substantially
sampling noise in that sample, not real underlying differences -- which
is exactly the null pattern found here. This is a real, useful result
(the selection-bias risk this project flagged in advance actually shows
up in real data), just not the "which estimator is right" result the
mini ground truth run set out to get. Resolving that would need either
much larger per-memory pair counts (unaffordable at this budget), a
different selection criterion less prone to this inflation, or accepting
that these 4 memories may simply have small true effects. Full per-memory
data: `results/mini_ground_truth_analysis.json`, `results/
mini_ground_truth_selection.json`.

## Noise-vs-signal decomposition (2026-09-20) — a standalone result, zero new spend

**Citable summary**: in a 30-memory store of generic, competently-written
"lesson" memories retrieved by a real LLM agent on real ALFWorld tasks,
individual per-memory causal effects are indistinguishable from pure
sampling noise at n=180 episodes, and the largest effect found by direct
ground truth (across 4 memories specifically selected for looking most
different) was 0.030 — an order of magnitude below what off-policy
estimators need to resolve reliably at practical sample sizes. Resolving
the full 30-memory store to the precision needed to detect effects of
that size would require **~85,463 episodes (~$970)** at real measured
per-task-type LLM costs — not a viable experiment at any realistic
research budget. This finding is independent of which OPE estimator is
used; it is a statement about how little real between-memory variation
this kind of natural memory store contains, not about estimator quality.

Prompted by the mini ground truth's null: are the 30-memory store's
individual effects simply too small to measure at any realistic budget?
`effect_size_analysis.py` (`results/effect_size_analysis.json`) answers
this with two independent lines of evidence, both computed from data
already in hand.

### 1. How large are real per-memory effects, typically?

**Line of evidence A — noise-vs-signal decomposition (all 30 memories,
n=180 small-pilot episodes).** Computed the theoretical null-model
standard error of the IPS estimator per memory (using each memory's real
candidacy count [40-85 episodes/memory] and real propensities, under the
null that its true effect is exactly 0), then compared to the ACTUALLY
OBSERVED spread of IPS point estimates across all 30 memories:

| | value |
|---|---|
| Mean null-model SE per memory (if true effect = 0) | 0.2202 |
| Observed std of IPS estimates across 30 memories | 0.2063 |
| Observed range | 0.892 (pure-noise-predicted range: ~0.837) |

The observed spread is **no larger than pure sampling noise predicts** —
if anything, slightly smaller. Method-of-moments (`Var(observed) -
Var(noise)`) comes out **negative** (-0.0059): noise alone fully explains
everything we see in the 30-memory pool. This is independent of which
estimator you use -- it's a statement about how much REAL variation
across memories the data contains, not an estimator's ability to find it.

**Line of evidence B — the 4 direct ground-truth measurements.** These
were selected specifically for LOOKING most different (biggest MW-vs-IPS
disagreement) and still came back with a spread of just 0.053 -- smaller
than any one of their own 95% CIs. Real measured rho (0.616-0.666, mean
0.636) is far higher than the 0.154 simulator-derived proxy used to plan
pair counts -- real paired episodes are much MORE correlated than
assumed, which is favorable (tighter CIs per pair than planned for), but
doesn't change the underlying finding: even the best-case-selected
memories show effects near the noise floor.

**Both lines agree: typical real per-memory effects in this store are
close to 0, with an upper bound somewhere around 0.03-0.05** (the largest
individual gap measured, and consistent with the population-wide analysis
finding no signal above noise).

### 2. Episodes needed to resolve effects of that size — is this design measurable at any realistic budget?

Using the REAL measured rho (0.636) and real Var(Y) (0.224), pairs needed
per memory for 80% power, two-sided:

| target effect (delta) | pairs/memory needed |
|---|---|
| 0.15 (original target) | 57 (fewer than the 133 originally planned -- real rho is more favorable than the simulator proxy) |
| 0.10 | 128 |
| 0.05 | 514 |
| **0.03 (largest gap actually found)** | **1424** |
| 0.02 | 3205 |

Resolving the **full 30-memory store** at delta=0.03 (the size we
actually found) would need ~85,463 episodes, costing **~$970** at real
measured per-task-type rates. **No: the current design (30 generic,
similar-quality memories, full ranking target) is not measurable at any
budget this project has had or is likely to have.** The mini ground
truth's null wasn't bad luck in memory selection -- it's what any 4 (or
30) memories from this store would show, because there just isn't much
real between-memory variation to find.

## Redesign proposals (2026-09-20) — NOT run, zero spend

Both proposals keep every frozen setting (model, max_steps, propensity
range, weighted task sampling) and only change the memory STORE itself
(smaller, deliberately quality-varied) and, for option (b), the
evaluation TARGET.

### (a) Small store (5-8 memories) with deliberately helpful/misleading content

Replace the 30 generic "lesson" memories with ~6 deliberately
engineered ones: 2-3 genuinely correct/helpful, 2-3 actively WRONG
(e.g. naming the wrong receptacle, an inefficient/incorrect procedure),
optionally 1-2 irrelevant/off-topic ones (testing whether noise memories
hurt via distraction). This directly engineers a KNOWN-DIRECTION,
hopefully-large effect rather than relying on subtle quality differences
between similar, competently-written lessons.

**Effect size: an ASSUMPTION, not yet measured** -- plausibly 0.15-0.40
for an actively wrong memory if the agent follows it, but real risk the
LLM just ignores a nonsensical instruction, realizing a smaller effect
than hoped. Recommend a cheap validation check (a handful of episodes,
~$0.10-0.20) before committing to a fully-powered run, to confirm the
manipulation actually moves success at all -- exactly the "memory sanity
check" pattern already used once in this project.

Episodes/cost for 6 memories, two-sided ranking target, real rho:

| delta | pairs/memory | total episodes | cost |
|---|---|---|---|
| 0.15 | 57.0 | 684 | $7.75 |
| 0.25 | 20.5 | 246 | $2.79 |
| 0.35 | 10.5 | 126 | $1.42 |

### (b) Reframe as "detect the harmful ones" (detection accuracy, not rank correlation)

Same engineered memory store as (a), but ask a coarser question: does an
estimator correctly flag the deliberately-harmful memories as harmful
(e.g., bottom-K by estimated value, or simply "significantly negative"),
rather than getting every memory's exact rank right? This needs LESS
statistical precision two ways at once: a one-sided test (detecting "this
is negative" needs less power than pinning down a specific two-sided
magnitude) and a larger assumed true effect (a deliberately broken memory
can be designed to fail badly, not just "somewhat worse").

| delta | pairs/memory | total episodes | cost |
|---|---|---|---|
| 0.35 | 8.2 | 99 | $1.12 |
| 0.30 | 11.2 | 135 | $1.53 |
| 0.25 | 16.2 | 194 | $2.20 |

**This is the more budget-efficient, more likely-to-succeed redesign** --
it targets a coarser, easier-to-resolve question using a deliberately
larger true effect, rather than fighting the same tiny-natural-effect
problem that produced this session's null.

### What the current ~$5.20 remaining could do, right now, for 6 memories

Two-sided ranking target: ~38 pairs/memory affordable, MDE≈0.183.
One-sided detection target: MDE≈0.163. Either is plausible IF the
engineered memories' true effects land in the 0.2-0.4 range assumed above
-- genuinely unverified until a cheap validation check is run. **Not
spent yet, pending the user's decision on which redesign (if either) to
pursue.**

## Redesign (b): Step 1 validation results (2026-09-21) — REAL SPEND $1.1923 — FAILED, stopping before Step 2

User chose redesign (b) (detection framing) with a $15 budget,
`hard_cap_usd` lowered to 13.00 **on a dedicated fresh cost-tracker state
file** (`logs/engineered_experiment_cost_state.json`) -- the old shared
production tracker was already at $14.80 cumulative, so reusing it with a
$13 cap would have refused every call instantly. All frozen settings
(`openai/gpt-5.6-luna`, `max_steps=50`, weighted task sampling,
propensity [0.3,0.7]) unchanged.

### The 6-memory engineered store (`engineered_memory_store.py`)

2 helpful, 2 harmful, 2 neutral, ~56-92 words each:

| id | task type | role |
|---|---|---|
| `mem_helpful_heat` | pick_heat_then_place_in_recep | correct: microwave, heat before placing |
| `mem_harmful_heat` | pick_heat_then_place_in_recep | wrong: claims the fridge heats objects |
| `mem_helpful_cool` | pick_cool_then_place_in_recep | correct: fridge, cool before placing |
| `mem_harmful_cool` | pick_cool_then_place_in_recep | wrong: claims the microwave cools objects |
| `mem_neutral_1` | pick_and_place_simple | generic "form a mental map" advice, non-actionable |
| `mem_neutral_2` | look_at_obj_in_light | generic "examine objects closely" advice, non-actionable |

Helpful/harmful pairs share a task type on purpose (matched comparison,
same population of games). With only 6 memories and M=10 (unchanged),
every memory is always a retrieval candidate regardless of tag -- so
ground truth for each memory is restricted to ITS OWN applicable task
type (reusing `weighted_task_source.WeightedRealTaskSource` with all-but-
one weight zeroed, rather than a new task-source class), since testing
heat-guidance on a non-heat task can't show any effect either way.

### A real bug found and fixed mid-run (no wasted spend)

`engineered_effect_validation.py`'s first version called
`run_ground_truth_phase_checkpointed(..., max_new=CHUNK_SIZE)` **once**
per memory instead of looping until the target pair count was reached --
it silently stopped after 5 pairs instead of the intended 20. Caught
immediately from the printed output (`n_pairs=5` where 20 was expected).
Fixed by wrapping the call in a `while True` loop (checking `n_new == 0`
to detect "target reached", matching the pattern already used elsewhere
in this codebase) and re-running -- checkpointing resumed cleanly from
pair 5, so the fix cost zero wasted spend, just extra wall-clock time.

### Results at the full n=20 pairs/memory

| memory | gap | 95% CI | verdict |
|---|---|---|---|
| `mem_helpful_heat` | +0.100 | [-0.096, +0.296] | includes zero, **below 0.2** |
| `mem_harmful_heat` | **+0.000** (exactly) | [-0.201, +0.201] | includes zero, **below 0.2** |

**Both fail the user's stated 0.2 threshold.** The deliberately-harmful
memory shows literally zero measured effect (mean success identical in
both arms, 0.150), not even a hint of the hoped-for large negative
effect.

### Why: the agent self-corrects within the step budget

Inspecting real episode transcripts explains the null mechanistically,
not just statistically. In one forced-in `mem_harmful_heat` episode that
SUCCEEDED, the agent initially went to the fridge and issued `cool egg 1
with fridge 1` (following the harmful memory's bad advice), then
afterward went to the microwave anyway and issued `heat egg 1 with
microwave 1` -- completing the task correctly despite the bad advice. The
ReAct agent isn't executing the memory as a fixed plan; it's reasoning
step by step with real environment feedback (admissible actions,
observations), and `max_steps=50` gives ample slack to recover from one
wrong early move. A memory that merely suggests a wrong FIRST step is
much easier to recover from than the kind of hard failure mode (running
out of steps entirely, or misidentifying the target object) needed to
produce a large, reliably measurable effect.

### Verdict: STOPPING before Step 2, per the user's explicit rule

Both effects are below the 0.2 threshold the user set in advance for
continuing. **Not proceeding to the full 6-memory detection experiment.**
This isn't a failure of the experimental INFRASTRUCTURE (ground truth,
checkpointing, cost tracking, and the task-type-restricted task source all
worked correctly, and the bug that appeared was caught and fixed with zero
wasted spend) -- it's a second, independent confirmation (after the
natural 30-memory store's null) that moving an LLM ReAct agent's success
rate via memory CONTENT alone is harder than assumed, at least for
mistakes an agent can still recover from within a generous step budget.
**If this redesign is revisited**, the transcript finding suggests the
lever that would actually move effect size is not "how wrong is the
advice" but "how recoverable is following it" -- e.g. advice that causes
the agent to interact with the wrong OBJECT entirely (not just the wrong
appliance for a step it can still redo), or that causes premature/false
task-completion signaling, rather than a wrong-but-recoverable
intermediate step. Real spend this step: **$1.1923** (dedicated tracker,
well under the $13 cap) -- roughly **$1.19 of the original ~$15 for this
experiment used**; the rest is unspent.

## Before Part C's full production run

1. ~~Pick a model~~ — done (`openai/gpt-5.6-luna`).
2. ~~Confirm memories change anything at all~~ — done, they do (memory
   sanity check).
3. ~~Fix the ceiling effect~~ — done (weighted task-type sampling; small
   pilot landed at 66% success).
4. ~~Get a first real ground-truth read on which estimator is right~~ —
   done, but inconclusive: all 4 ground-truthed memories showed a
   statistically null effect (see "Mini ground truth: results and
   verdict"). The likely explanation is selection-bias inflation in how
   the 4 were chosen (biggest MW-vs-IPS disagreement in a noisy 90-episode
   half-sample), not that these estimators are indistinguishable in
   general -- a bigger main-logging phase (more episodes before
   selecting ground-truth candidates) would give a less noise-prone
   selection set.
5. Run `determinism_check.check_determinism` against the real client
   before trusting any FUTURE ground-truth pair's determinism (still not
   done -- this round's pairs used the same temperature=0 setup but
   without a fresh determinism probe first).
6. Decide on parallelization (see "Wall-clock runtime estimate" above) --
   serial execution at Part C's measured latency would take multiple
   days for the full plan (the mini ground truth run alone, 1064
   episodes, took roughly 10 hours serially).
7. Run via `run_chunked.py`, not a single long-running process, given the
   resource-leak finding above -- point `TMPDIR` at disk-backed storage too.
8. Given ~$5.20 of the $17.80 balance remains (after this round's
   $12.6049 mini-ground-truth spend; $14.86 total real spend across the
   whole project), the full 10030-episode plan (~$63-92 projected)
   remains NOT affordable as scoped — a from-real-numbers re-plan
   (episode count vs. cost trade-offs at this budget) is the next
   planning step if the project continues.

## Kaggle replication (2026-09-27): zero-cost, real embeddings + local model

Separate track from everything above — fixes this pilot's two acknowledged
weaknesses (mock retrieval similarity, single commercial model) without
spending more real money, by running on Kaggle's free GPU tier instead.
Plan approved 2026-09-26; infrastructure built and smoke-tested locally
(WSL2, RTX 4060 8GB — enough to validate the wiring, not the real scaled
run); **Phase 0 (real ALFWorld + real local model on actual Kaggle
hardware) has not run yet.**

### Design

- **Local model**: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28` (Apache-2.0). Served via plain
  `transformers.generate()` by default, 4-bit (`bitsandbytes` nf4) quantized
  for VRAM headroom — **not vLLM**, despite vLLM's usual throughput
  advantage, because current mainline vLLM has documented compatibility
  problems with Kaggle's actual GPUs (Tesla T4, compute capability 7.5:
  `bfloat16`-only-on-8.0+ errors and other open T4-specific issues in
  vLLM's own tracker; P100 is capability 6.0, older still). Phase 0 tries
  vLLM first and falls back cleanly if it doesn't work — verify, don't
  assume, the same discipline this project already applied to
  Python-3.13/textworld.
- **Real embedding retrieval**: `BAAI/bge-small-en-v1.5` (33M params,
  384-dim, MIT), replacing `memory_store.similarity_scores`'s mock
  0.7/0.3-plus-noise heuristic with real cosine similarity
  (`embedding_retrieval.py`). One real nuance, not glossed over: main
  logging episodes use the REAL per-episode goal text (free, since
  `env.reset()` already produced it); ground-truth candidacy PROBING
  (`ground_truth_runner._is_natural_candidate`) deliberately keeps using a
  cheap per-task-type proxy description instead, because that probe must
  stay free of env construction (most probed seeds are rejected before a
  natural candidate is found) — see `embedding_retrieval.py`'s module
  docstring for the full reasoning.
- **Memory store redesign** (`kaggle_memory_store.py`): 21 hand-written
  memories across ALL 6 ALFWorld task types (vs. the paid engineered
  store's 2), in FOUR categories — correct, harmful, **partial** (a new
  category: correct about part of the task but incomplete, e.g. names the
  right appliance but omits the required action — deliberately absent from
  the paid run's stark correct/harmful/neutral split), and irrelevant.
  Lengths vary naturally (each memory says what it needs to, not padded to
  a template) — directly targets the paid run's own §5 finding that its
  natural 30-memory store had almost no real between-memory variation.
- **Comparability with the paid run**: `env.max_steps=50`,
  `env.real_split=train`, `retrieval.M=10`,
  `retrieval.propensity_min/max=[0.3,0.7]`, and all four estimators
  (`memory_worth`/`ips`/`snips`/`doubly_robust` from `src/memory_ope/`) kept
  identical — see `kaggle_config.yaml`, deliberately separate from
  `config.yaml` so the paid run's exact historical settings stay untouched
  and citable as-is.

### What's new vs. what's reused unchanged

New: `local_model_client.py` (`LocalTransformersClient`, implements the same
`LLMClient` protocol as `OpenRouterClient` — cache-check, cost-tracker
pre-check, real call, record, cache-put, identical structure), `embedding_retrieval.py`,
`kaggle_memory_store.py`, `kaggle_session.py` (cross-Kaggle-session
checkpoint copy-in/run/copy-out via a versioned Kaggle Dataset — needed
because `/kaggle/working` is wiped between interactive sessions unless
explicitly saved, unlike this project's WSL2 environment's persistent home
directory; the Kaggle-API parts of this file are untestable outside an
actual Kaggle session), `kaggle_smoke_test.py` (the local, pre-Kaggle
verification script — see below), and a throwaway `notebooks/
phase0_environment_probe.ipynb`.

Reused completely unchanged: `cache.py`, `cost_tracker.py` (already
persists to `state_path` with an atomic write, survives a restart —
constructed with zero pricing for a local model, so its hard-cap check
becomes a harmless no-op rather than being removed), `determinism_check.py`
(generic against the `LLMClient` protocol — this is the check that was
**never run** against the paid model; running it against the local model is
one of this replication's two explicit fixes), `env_factory.py`,
`env_interface.py`, `retrieval_shared.py`, `react_agent.py`, everything
under `src/memory_ope/`. `episode_runner.py`, `ground_truth_runner.py`, and
`checkpointed_runner.py` each got one small, backward-compatible addition
(an optional `similarity_fn`/`candidacy_similarity_fn` parameter, default
`None` = original mock behavior, unchanged for every existing caller/test);
`run_chunked.py` got `--llm local`, `--memory-store kaggle`, `--config`,
`--cache-dir`, and `--time-budget-seconds` (the last one because Kaggle's
12-hour session cap is a real wall-clock deadline the supervisor loop
previously had no concept of). **All 25 existing tests in
`alfworld_pilot/tests/` still pass unchanged after every one of these
edits.**

### Local smoke test (`kaggle_smoke_test.py`) — before ever touching Kaggle

Run from a dedicated `alfworld_pilot/.venv-kaggle` venv (kept separate from
the main project `.venv` specifically so heavy GPU deps — `torch`,
`transformers`, `bitsandbytes`, `sentence-transformers` — never risk
disturbing the paid run's already-verified, already-cited-in-`paper_data.md`
exact package versions):

```bash
cd alfworld_pilot
.venv-kaggle/bin/python -m alfworld_pilot.kaggle_smoke_test            # embeddings only, no GPU, seconds
.venv-kaggle/bin/python -m alfworld_pilot.kaggle_smoke_test --full     # + real ALFWorld + real local model + determinism check, needs a GPU
```

**Status: both `kaggle_smoke_test.py` and `--full` have now run successfully
end to end** on this local machine (WSL2, RTX 4060, 8GB VRAM) — the first
real execution of every piece of this replication's new code, real ALFWorld
included, at $0 real cost.

Getting there involved two real, worth-recording snags, neither a code bug:

- **pip's resolver thrashed for 25+ minutes** trying to backtrack through
  dozens of candidate `torch` versions before being killed and restarted
  with an explicit pinned `torch==2.5.1 --index-url
  https://download.pytorch.org/whl/cu124`, which resolves directly.
- **`Qwen/Qwen2.5-7B-Instruct`'s 15GB download repeatedly failed partway
  through** with a `ConnectionError` from HuggingFace's newer "xet" CDN
  backend (`us.aws.cdn.hf.co`) on this network — twice, at different
  points (after 2/4 and again after 2/4 shards). Setting
  `HF_HUB_DISABLE_XET=1` (falls back to plain HTTP) let it complete,
  though even then it hit two more transient timeouts that `huggingface_hub`'s
  own retry/resume logic recovered from automatically. Total download time
  this session: ~1h42m. **If reproducing this on a different machine and
  the download stalls, try `HF_HUB_DISABLE_XET=1` first** — this looks like
  a network/CDN-side issue, not something specific to this machine.

**Real results** (`kaggle_smoke_test.py --full`, `openai/gpt-5.6-luna`'s local
replacement `Qwen/Qwen2.5-7B-Instruct`, 4-bit nf4 quantization — comfortably
fit and loaded in 22s on an 8GB card, real headroom to spare for Kaggle's
16GB T4/P100):

- **Determinism check: PASSED** (`holds=True`, both calls returned
  `'Hello'` exactly) — the check this whole replication exists partly to
  finally run, since the paid run's own README lists it under "still not
  done" for every real ground-truth phase it ever ran. This is the first
  time in this project's history this check has been run against any real
  model and held.
- **3 real ALFWorld episodes ran end to end**: all hit the 50-step cap
  without winning (`success=0` for all 3) — expected and unremarkable for a
  first, completely untuned run (a 7B model with generic "lesson" memories,
  no prompt iteration) — not evidence anything is broken. Real embedding
  retrieval picked plausible, topically-relevant candidate sets each time
  (e.g. episode 2, a `pick_two_obj_and_place` task, included `two_correct`
  among its retrieved memories).
- **Real token accounting, $0 cost**: 152 real local LLM calls, 200,887
  input / 6,912 output tokens, `cost_tracker.summary()` confirms
  `total_spend_usd: 0.0` throughout.

This is real, local, first-ever verification that every piece of this
replication's new code — `LocalTransformersClient`, `embedding_retrieval.py`,
`kaggle_memory_store.py`, and `determinism_check.py` run against a genuinely
different model — works correctly together against real ALFWorld. **Phase 0
on actual Kaggle hardware is the next, still-unrun step** — this local run
substitutes for it only as a wiring check (8GB VRAM vs. Kaggle's 16GB
target; unknown whether Kaggle's own network hits the same CDN issue).

### Kaggle-specific open questions, not yet resolved

1. Whether internet-ON is actually available for the target Kaggle
   account/notebook (assumed yes — Kaggle's internet-off restriction is
   specifically for competition scoring; an ordinary private notebook can
   enable it). If not, the fallback is pre-packaging `~/.cache/alfworld`
   (post-`alfworld-download`) and both model snapshots as an offline Kaggle
   Dataset once, then attaching it as input every session.
2. Whether `gcc`/a C toolchain is present by default in Kaggle's notebook
   image, and what Python version it ships — unverified; `notebooks/
   phase0_environment_probe.ipynb`'s first cells check exactly this before
   anything else is attempted.
3. Whether the `kaggle` CLI is pre-authenticated inside a Kaggle-hosted
   notebook (needed for `kaggle_session.py`'s cross-session dataset
   versioning) — documented Kaggle behavior, not independently verified
   here.
4. The exact `BAAI/bge-small-en-v1.5` revision hash is NOT pinned yet
   (`kaggle_config.yaml`'s `embedding.revision: null`) — HuggingFace's page
   didn't expose it to a simple fetch during planning; `SentenceEmbedder`
   warns loudly if constructed with `revision=None`, and Phase 0 must
   record and set the exact hash it actually downloads before any real run.
