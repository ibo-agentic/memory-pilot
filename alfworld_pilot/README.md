# Stage 2 — ALFWorld pilot

## Status: infrastructure built, no real API calls made, ALFWorld itself deferred

Everything in this package has been exercised only against a mock LLM and a
mock environment. `config.yaml`'s `llm.model_id` is `null` on purpose —
`OpenRouterClient` refuses to construct with a null or "latest"-aliased
model id, so nothing can accidentally spend money with an unintended
default. Real API calls start only in Part C, after explicit approval, with
a hard spending cap (`cost_control.hard_cap_usd`) enforced before every call.

## Why ALFWorld itself isn't installed yet

Attempted the official install (`pip install alfworld`) directly, in an
isolated Python 3.11 venv (`.venv/` in this directory — separate from the
root package's 3.14 venv, since ALFWorld's dependency chain is much older
and riskier on 3.14). It fails building `jericho` (a TextWorld dependency):
no C build toolchain (no cmake, no MSVC) is present on this Windows
machine, and even with one, `textworld`'s own `setup.py` runs a `setup.sh`
that fetches a Linux-only Inform7 CLI binary — the whole install path is
Linux/Mac-oriented.

Decision (made with the user): defer real ALFWorld, build everything else
now against `env_interface.MockAlfredEnv`, which implements the exact same
interface as the real batched API (confirmed from the official repo +
the original ReAct paper's `alfworld.ipynb`: `env.reset() -> (obs, info)`,
`env.step([action]) -> (obs, scores, dones, infos)`,
`info['admissible_commands']`, task type from `info['extra.gamefile']`).
`env_interface.RealAlfredEnv` is written and ready but will raise a clear
`ImportError` until `alfworld` is actually installed. When you're ready:
install via **WSL2 + Ubuntu** (`wsl --install -d Ubuntu`, then set up a
venv inside WSL and follow the official repo's Linux install steps) or
**Docker** (the official repo points to `vzhong/alfworld` on Docker Hub).
Either way, no code in this package should need to change — only the
`env.backend: real` config switch and `real_alfworld_config_path`.

## Design

```
src/alfworld_pilot/
  env_interface.py       AlfredEnv protocol, MockAlfredEnv, RealAlfredEnv (not yet runnable)
  memory_store.py         30 short "lessons" (100-300 tokens), mock topic-aware similarity
  retrieval_shared.py      re-exports Stage 1's EXACT retrieval mechanism (top-M, rank-linear
                           propensity, independent Bernoulli inclusion) via a sys.path shim to
                           the root package's src/ -- not reimplemented
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

## Running (no cost)

```bash
cd alfworld_pilot
.venv\Scripts\python.exe -m pip install -r requirements.txt   # already done once

set PYTHONPATH=src
.venv\Scripts\python.exe -m pytest tests/ -v                   # 17 tests, all mock, zero cost
.venv\Scripts\python.exe -m alfworld_pilot.measure_mode         # mock episode + cost-projection smoke test
.venv\Scripts\python.exe -m alfworld_pilot.token_breakdown      # per-section token breakdown
```

## Before Part C (real spend)

1. Set `llm.model_id` in `config.yaml` to an exact pinned id (checked
   against openrouter.ai's current listing, never a "latest" alias).
2. Confirm `OPENROUTER_API_KEY` is set in the repo-root `.env`.
3. Confirm `cost_control.hard_cap_usd` is what you want to risk.
4. Run `determinism_check.check_determinism` against the real client before
   trusting any ground-truth pair's determinism.
