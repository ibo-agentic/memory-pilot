# Stage 2 — ALFWorld pilot

## Status (2026-09-18): real ALFWorld installed and running; still no real LLM API calls made

ALFWorld is installed and `env.backend: real` is now config.yaml's default
(`MockAlfredEnv` is kept for the unit test suite only, which imports it
directly regardless of config.yaml). Everything has still been exercised
only against `MockLLMClient` — `config.yaml`'s `llm.model_id` is `null` on
purpose — `OpenRouterClient` refuses to construct with a null or
"latest"-aliased model id, so nothing can accidentally spend money with an
unintended default. Real API calls start only in Part C, after explicit
approval, with a hard spending cap (`cost_control.hard_cap_usd`) enforced
before every call.

## Installing real ALFWorld (done, under WSL2 Ubuntu)

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

### Real, non-obvious blocker found and fixed: textworld 1.7.0 vs Python 3.13+

`textworld`'s PDDL grammar engine (`textworld.envs.pddl.textgen.EvalSymbol.
derive`, used to render every piece of generated game text, starting with
the very first `env.reset()`) does `locals().update(context["variables"]);
eval(self.expression)`, relying on a `locals()`-mutation trick that PEP 667
(Python 3.13, "Consistent views of namespaces") permanently killed —
`locals()` inside a function now always returns a fresh, discarded
snapshot, so the injected names are gone by the time `eval()` looks for
them, raising `NameError: name 'r' is not defined` on the first
`env.reset()`. Confirmed via the official PEP 667 text and by grepping the
installed package (`locals().update(` appears exactly once, only here); no
fix exists yet in textworld 1.7.0 (the latest release as of 2026-09-18).
`alfworld_pilot/src/alfworld_pilot/_textworld_py313_compat.py` monkeypatches
just that one method to pass an explicit namespace to `eval()` instead
(the documented PEP 667 migration), applied once at `RealAlfredEnv`
construction time, before textworld renders anything. It patches the
installed package in memory only — `pip install -U textworld` would
cleanly take over if upstream ships a real fix later.

### Two real bugs in this package's own code, found only once real ALFWorld was exercised

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

### Known real limitation, not fixed in this session (relevant to Part C ground truth)

`ground_truth_runner`'s paired forced-in/forced-out design needs
`env.reset(task_seed=...)` to seek back to the *same* task instance twice.
`MockAlfredEnv` supports this (the mock's seeded branch is a pure function
of `task_seed`). Real ALFWorld's `env.reset()` ignores `task_seed` entirely
and just advances to the next game in its internal sequence — so today,
real-backend forced-in/forced-out pairs would land on two *different* task
instances, breaking the paired design's core assumption. Solving this
(likely: directly loading a specific `gamefile` path via TextWorld's
lower-level `env.load()`, using `AlfredTWEnv.game_files`) is unstarted work
for Part C, not something this session's narrower "swap the env backend and
smoke-test it" scope covers.

## Running

```bash
# from the repo root, ONE shared venv (see ../pyproject.toml) -- memory_ope
# is pip-installed editable, so retrieval_shared.py needs no sys.path shim.
source .venv/bin/activate
cd alfworld_pilot
PYTHONPATH=src python -m pytest tests/ -v                    # 17 tests, all against MockAlfredEnv, zero cost
PYTHONPATH=src python -m alfworld_pilot.measure_mode          # real ALFWorld env (config.yaml default) + MockLLMClient,
                                                                # zero API cost -- reports real tokens/call, calls/episode,
                                                                # and a full-plan cost projection
PYTHONPATH=src python -m alfworld_pilot.token_breakdown        # per-section synthetic token breakdown (still mock env by design)
```

## Design

```
src/alfworld_pilot/
  env_interface.py       AlfredEnv protocol, MockAlfredEnv, RealAlfredEnv (both runnable)
  env_factory.py           picks Mock/Real from config.yaml's env.backend
  _textworld_py313_compat.py  monkeypatch for a real textworld 1.7.0 / Python 3.13+ incompatibility
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
  ground_truth_runner.py   paired forced-in/forced-out: same task instance AND same "everything
                           else" random draws between arms, restricted to task instances where
                           the target memory is a NATURAL candidate (matches the causal oracle's
                           own definition)
  power_analysis.py        redoes Part A.3's sample-size calc with a MEASURED rho (from
                           ground_truth_runner.paired_correlation on real results)
  measure_mode.py          runs N episodes, records real per-call tokens, projects full-plan cost
  token_breakdown.py       decomposes one call's prompt into system/memories/history/actions
```

Settings (30 memories, propensity range [0.3, 0.7], M=10 unchanged) come
from `signal_boost_check.py`'s recommendation in the root package — the
only setting where a propensity-corrected estimator's 95% interval excludes
zero at Stage 2's realistic episode count.

## Before Part C (real spend)

1. Set `llm.model_id` in `config.yaml` to an exact pinned id (checked
   against openrouter.ai's current listing, never a "latest" alias).
2. Confirm `OPENROUTER_API_KEY` is set in the repo-root `.env`.
3. Confirm `cost_control.hard_cap_usd` is what you want to risk.
4. Run `determinism_check.check_determinism` against the real client before
   trusting any ground-truth pair's determinism.
