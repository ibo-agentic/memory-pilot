# Stage 2 — ALFWorld pilot

## Status (2026-09-19): real ALFWorld running, paired ground truth solved, still no real LLM API calls made

ALFWorld is installed and `env.backend: real` is config.yaml's default
(`MockAlfredEnv` is kept for the unit test suite only, which imports it
directly regardless of config.yaml). Everything has still been exercised
only against `MockLLMClient` — `config.yaml`'s `llm.model_id` is `null` on
purpose — `OpenRouterClient` refuses to construct with a null or
"latest"-aliased model id, so nothing can accidentally spend money with an
unintended default. Real API calls start only in Part C, after explicit
approval, with a hard spending cap (`cost_control.hard_cap_usd`) enforced
before every call. As of 2026-09-19, the venv runs Python 3.11 (down from
3.14) and paired forced-in/forced-out ground truth works against real
ALFWorld (see below) — the two items previously blocking Part C.

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

## Before Part C (real spend)

1. Set `llm.model_id` in `config.yaml` to an exact pinned id (checked
   against openrouter.ai's current listing, never a "latest" alias).
2. Confirm `OPENROUTER_API_KEY` is set in the repo-root `.env`.
3. Confirm `cost_control.hard_cap_usd` is what you want to risk.
4. Run `determinism_check.check_determinism` against the real client before
   trusting any ground-truth pair's determinism.
5. Watch memory during the ground-truth rerun phase (see the operational
   note above) — chunk it (restart the process every N pairs) if growth
   reappears despite `RealAlfredEnv.close()`.
