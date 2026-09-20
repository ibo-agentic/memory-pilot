# Stage 2 — ALFWorld pilot

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
