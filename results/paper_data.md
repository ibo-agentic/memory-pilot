# Paper data: Off-policy evaluation for causal memory value in LLM agents

Compiled 2026-09-26 from `/home/ibo/memory-pilot` (WSL2 Ubuntu repo, HEAD `0bb21e6`,
remote `origin` = `https://github.com/ibo-agentic/memory-pilot.git`). All paths below are
relative to that repo root unless stated otherwise. Every number is followed by its
source file/script in `` `backticks` ``. Items I could not verify or that are marked
as assumptions **in the project's own files** are flagged with **[UNVERIFIED]** or
**[ASSUMPTION, not measured]** — do not cite these as measurements without saying so.

**Top-line honesty note**: this project has two genuinely different kinds of result.
Stage 1 (Sections 2–3) is a synthetic simulator with a known ground truth, run to
completion, 20 seeds, fully reproducible, zero cost. Stage 2 (Section 4) is real
ALFWorld + a real LLM agent + real spend, but it did **not** produce the clean
"OPE beats Memory Worth" confirmation Stage 1 predicted — it produced a null result
at natural effect sizes, and a second null when the team deliberately engineered
large effects. That double null, and the noise-vs-signal accounting behind it, is
arguably the paper's actual finding. Sections below reflect this as it happened.

---

## 1. Setup

### 1.1 Model, provider, sampling settings

- **Model**: `openai/gpt-5.6-luna`, accessed via **OpenRouter** (`https://openrouter.ai/api/v1`,
  OpenAI-compatible endpoint, Python `openai` SDK). — `alfworld_pilot/config.yaml` (`llm.model_id`, `llm.base_url`), `alfworld_pilot/src/alfworld_pilot/llm_client.py`
- Chosen after a 3-model bake-off (Section 4.1); **not** the cheapest option, chosen for its far lower malformed-output rate. Model id/pricing snapshot fetched from `https://openrouter.ai/api/v1/models` on 2026-09-19 — OpenRouter pricing is not guaranteed stable, re-check before citing as current. — `alfworld_pilot/config.yaml` comment, `alfworld_pilot/README.md`
- **Sampling settings** (`alfworld_pilot/config.yaml`, `llm:` block):
  - `temperature: 0.0`
  - `max_output_tokens: 100` (a ReAct Thought+Action pair is short)
  - `seed`: an optional fixed seed param passed **only if the provider supports it** (`llm_client.py` `OpenRouterClient._request_payload`) — not asserted to be honored by OpenRouter/the provider.
  - **Reasoning disabled**: `reasoning_enabled: false` in config, sent to the API as `extra_body={"reasoning": {"enabled": false}}` (OpenRouter's documented unified reasoning control). — `alfworld_pilot/src/alfworld_pilot/llm_client.py` lines 90, 112
  - Pricing used for projections: $0.20 / $1.20 per million input/output tokens (`llm.pricing_per_million_tokens` in config.yaml) — this is gpt-5.6-luna's specific pricing, distinct from the 3-model bake-off pricing table (deepseek-v4.1-flash/ling-3.0-flash-vl/mercury-2.5) used only for early cost projections before a model was picked.
- **Determinism was never independently verified against the real client.** `determinism_check.check_determinism` exists (two real duplicate calls, compare text) but the project's own "what's still needed" list explicitly says this was **not run** before any of the real ground-truth phases. Temperature=0 is configured but not empirically confirmed deterministic on this model. — `alfworld_pilot/README.md` "Before Part C's full production run", item 5; `alfworld_pilot/src/alfworld_pilot/determinism_check.py`

### 1.2 ALFWorld version, dataset, split, max_steps, game files

| item | value | source |
|---|---|---|
| `alfworld` package version | 0.4.2 | `importlib.metadata.version("alfworld")` in the project venv |
| `textworld` version | 1.7.0 | same |
| Install method | `pip install alfworld` (base install, no `[full]`/`[vis]` — those pull ai2thor/torch/opencv for the unused visual/embodied backend) | `alfworld_pilot/README.md` |
| Data fetch | `alfworld-download`, ~2.3GB (game/PDDL files, logic grammar, MaskRCNN detector — detector unused by this text-only pilot) | `alfworld_pilot/README.md` |
| Base config | `alfworld_pilot/configs/base_config.yaml` — the official repo's `configs/base_config.yaml` for alfworld==0.4.2, fetched verbatim, not hand-edited | `alfworld_pilot/README.md` |
| **Split used** | `real_split: train` | `alfworld_pilot/config.yaml` `env.real_split` (other options `eval_in_distribution`, `eval_out_of_distribution` were never used) |
| **Total game files in this split** | **3,553** | `alfworld_pilot/results/task_type_difficulty_check.json` (`total_game_files`) |
| Game files by task type | pick_and_place_simple 790, look_at_obj_in_light 308, pick_clean_then_place_in_recep 650, pick_heat_then_place_in_recep 459, pick_cool_then_place_in_recep 533, pick_two_obj_and_place 813 | same file, `counts_by_task_type` |
| `env.max_steps` (episode step cap) | **50** (raised from an initial 30, see §2 and §7) | `alfworld_pilot/config.yaml` |
| Task-type coverage | all 6 official ALFWorld task types confirmed present, 0 unparsed | same file |

### 1.3 Agent design

- **Pattern**: single-call ReAct — one LLM call per environment step produces both a
  `Thought:` and an `Action:` line together (deliberately simpler than the original
  ReAct paper's ALFWorld setup, which sometimes splits Thought and Action into two
  calls) — keeps "~1 LLM call per step" true for cost accounting.
  — `alfworld_pilot/src/alfworld_pilot/react_agent.py` module docstring
- **Prompt structure**, in fixed order (from `_build_prompt` in `react_agent.py`):
  1. System prompt (fixed, task-agnostic — see exact text below)
  2. Retrieved memories block: `"Relevant lessons from past episodes:\n- <text>\n- <text>..."` (omitted entirely if none were included this episode)
  3. Task goal (`Task: <goal observation>`)
  4. Step history: `"{observation}\n> {action}\n"` repeated for every prior step (grows every step)
  5. This step's admissible actions (`Admissible actions: [...]`, the raw Python list)
  - Output must be exactly two lines: `Thought: <reasoning>` then `Action: <one action, copied EXACTLY from the admissible actions list>`.
- **Exact system prompt** (`react_agent.py` `SYSTEM_PROMPT`):
  > "You are an agent solving a household task in a text-based environment (ALFWorld). Each turn, you see the current observation and a list of admissible actions. Respond with exactly two lines:\nThought: <brief reasoning about what to do next>\nAction: <one action, copied EXACTLY from the admissible actions list>\n"
- **Action parsing**: regex-extracts the `Action:` line; if the extracted string isn't in the admissible-actions list (checked verbatim, then whitespace-stripped), the episode logs a **parse failure** and falls back to the *first* admissible action rather than crashing. — `react_agent.py` `_parse_action`
- **Memory insertion**: memory texts for an episode = whichever of that episode's randomized-retrieval **included** memories (Section 1.5), inserted verbatim as a bullet list before the task goal. Not injected into the system prompt, not summarized/compressed.
- **Step limit**: 50 (see §1.2); one LLM call per step, so worst case 50 calls/episode.
- **A real, documented subtlety**: exactly one `env.reset()` per episode. An earlier version called `reset()` twice (once for task-type/retrieval, once inside the agent loop), which — for real ALFWorld — would have silently played a *different* game than the one memories were retrieved for. Fixed before any real spend on real ALFWorld; see §7. — `alfworld_pilot/README.md` "Two real bugs..."; `react_agent.py run_episode` docstring

### 1.4 Memory store: construction, token lengths, examples

**Two separate stores exist — do not conflate them.**

**(A) The natural 30-memory store** (used for the model-selection test, memory sanity
check, small pilot, and the 4-memory mini ground truth):

- 30 memories, each a padded variant of one of 6 fixed "lesson" templates (one per
  ALFWorld task type), assigned round-robin (`memory index i % 6`). — `alfworld_pilot/src/alfworld_pilot/memory_store.py` `build_mock_store`
- Construction is deterministic: `seed=0` in every script that builds this store
  (`token_breakdown.py`, `measure_mode.py`, `memory_sanity_check.py`, `small_pilot.py`,
  `model_selection_test.py`, `run_chunked.py`) — all draw the identical 30 memories.
- Config target range: `lesson_min_tokens: 100`, `lesson_max_tokens: 300` (word count used as a token proxy throughout, never a real tokenizer at construction time). — `alfworld_pilot/config.yaml`
- **Actual realized lengths, measured directly** (`build_mock_store(30, 100, 300, seed=0)`,
  `.approx_tokens` = word count): **mean 110.8, min 100, max 120**.
  **This is narrower than the configured [100, 300] range** — a real quirk of
  `_pad_to_token_range`: it pads with a filler sentence only until the text first
  crosses `min_tokens` (100), then picks a random *target* length up to 300 and
  truncates only if the padded text already exceeds that target; since padding stops
  right after crossing 100, the text is essentially never long enough to need
  truncation toward a higher target, so real lengths cluster at 100–120 words
  regardless of the configured 300-word ceiling. **Flag this if your writeup says
  "100–300 token memories" — in the actual store it's closer to 100–120.**
  — computed directly from `memory_store.build_mock_store`, source in `alfworld_pilot/src/alfworld_pilot/memory_store.py`
- Example texts (first 3 of 30, verbatim, `mem_0`/`mem_1`/`mem_2`, seed=0):
  - `mem_0` (pick_and_place_simple, 106 words): "(lesson 0) When the task is to move an object to a receptacle, first locate the object, pick it up, then navigate to the target receptacle before placing it. Double-check the object is the one named in the goal before picking it up. This was learned from a past episode and generalizes to similar situations encountered later in the same kind of household task. This was learned from a past episode a[nd generalizes...]"
  - `mem_1` (look_at_obj_in_light, 119 words): "(lesson 1) For examine-under-light tasks, bring the target object to the lamp (or the lamp's location) before turning it on -- turning the lamp on first and then walking away loses the light source's effect. This was learned from a past episode and generalizes to similar situations encountered later in the same kind of household task. This was learned from a past episode and generalizes to similar [situations...]"
  - `mem_2` (pick_clean_then_place_in_recep, 108 words): "(lesson 2) Cleaning tasks require using the sink/basin explicitly (e.g. 'clean X with sinkbasin 1') after picking the object up, before carrying it to the final receptacle. Skipping the clean step leaves the goal unsatisfied even if the object ends up in the right place. This was learned from a past episode and generalizes to similar situations encountered later in the same kind of household task."
  - (Every memory is this same "correct, generic lesson" pattern, padded with an identical filler sentence — this uniformity is directly relevant to why the natural store's between-memory effect variance turned out to be near zero; see §5.)
- The 6 base lesson templates (one per task type, verbatim, before padding) are in `alfworld_pilot/src/alfworld_pilot/memory_store.py` `LESSON_TEMPLATES`.

**(B) The engineered 6-memory store** (used only for the redesign-(b) validation, §4.6):

| id | task type | role | length (words) |
|---|---|---|---|
| `mem_helpful_heat` | pick_heat_then_place_in_recep | correct: use microwave, heat before placing | 92 |
| `mem_harmful_heat` | pick_heat_then_place_in_recep | **wrong**: claims the fridge heats objects | 75 |
| `mem_helpful_cool` | pick_cool_then_place_in_recep | correct: use fridge, cool before placing | 92 |
| `mem_harmful_cool` | pick_cool_then_place_in_recep | **wrong**: claims the microwave cools objects | 73 |
| `mem_neutral_1` | pick_and_place_simple | generic, non-actionable ("form a mental map") | 58 |
| `mem_neutral_2` | look_at_obj_in_light | generic, non-actionable ("examine objects closely") | 56 |

Full verbatim text of all 6 (this is the complete store used in §4.6):

- **`mem_helpful_heat`**: "Heating tasks require the microwave as the correct intermediate step. After picking up the target object, navigate to the microwave, place the object inside, and issue the heat action explicitly (for example, 'heat mug 1 with microwave 1') before carrying it onward. The object must be heated BEFORE it is placed in its final destination receptacle -- heating it afterward does not satisfy the task goal. Skipping the microwave step entirely, or using the wrong appliance, will leave the heating sub-goal unsatisfied even if the object ends up in the right place."
- **`mem_harmful_heat`**: "For heating tasks, the fastest approach is to skip the microwave and instead place the object directly in the fridge for a few moments -- the fridge's internal mechanism can raise an object's temperature just as effectively as a microwave, and it is usually closer to hand. Once removed from the fridge, carry the object straight to its destination receptacle. Using the microwave first is an unnecessary extra step that only wastes time and moves."
- **`mem_helpful_cool`**: "Cooling tasks require the fridge as the correct intermediate step. After picking up the target object, navigate to the fridge, place the object inside, and issue the cool action explicitly (for example, 'cool tomato 1 with fridge 1') before carrying it onward. The object must be cooled BEFORE it is placed in its final destination receptacle -- cooling it afterward does not satisfy the task goal. Skipping the fridge step entirely, or using the wrong appliance, will leave the cooling sub-goal unsatisfied even if the object ends up in the right place."
- **`mem_harmful_cool`**: "For cooling tasks, the fastest approach is to briefly place the object in the microwave instead of the fridge -- the microwave's rapid cycling can chill an object almost as quickly as refrigeration, and skips the wait for the fridge door to open and close. Once removed from the microwave, carry the object straight to its destination receptacle. Using the fridge first is an unnecessary extra step that only wastes time and moves."
- **`mem_neutral_1`**: "Household environments typically contain many similar-looking objects spread across several rooms. Before committing to a plan, it can help to form a general mental picture of the space -- which rooms connect to which, and roughly where the larger furniture and appliances are -- since this orientation is useful background context for many different kinds of household tasks."
- **`mem_neutral_2`**: "Many objects in these environments can be picked up, opened, or examined more closely, and doing so sometimes reveals details that aren't obvious from a distance. If a task description seems ambiguous about which specific object is meant, taking a closer look at the candidates in the room can occasionally help resolve the ambiguity before proceeding."

— `alfworld_pilot/src/alfworld_pilot/engineered_memory_store.py`

### 1.5 Retrieval: similarity, M, propensity, inclusion rule

- **No real embedding model was ever used.** Similarity is a **mock, topic-aware
  heuristic**: `similarity_scores` gives a memory `base=0.7` if its tagged `task_type`
  matches the current episode's task type, else `base=0.3`, plus `Uniform(-0.2, 0.2)`
  noise. This was a deliberate, documented placeholder ("computing real embeddings
  would mean an API call... swap `similarity_score` for a real embedding lookup once
  Stage 2 goes live" — never done). **If your writeup says "embedding-based
  retrieval," qualify it: retrieval ran on a mock topic-match heuristic, not a real
  embedding model, for every real-agent result in this paper.**
  — `alfworld_pilot/src/alfworld_pilot/memory_store.py` module docstring + `similarity_scores`
- **Candidate count M = 10** (top-10 by similarity score) — unchanged from Stage 1 through every Stage 2 phase. — `alfworld_pilot/config.yaml` `retrieval.M`, `src/memory_ope/retrieval.py` `select_top_m`
- **Propensity assignment**: deterministic, rank-linear within the top-M: the most
  similar candidate gets `propensity_max`, the least similar (M-th) gets
  `propensity_min`, linearly interpolated by rank in between:
  `p(rank) = propensity_max - (propensity_max - propensity_min) * (rank / (M-1))`.
  — `src/memory_ope/retrieval.py` `assign_propensities`
- **Propensity range used in every Stage 2 real-agent run**: **[0.3, 0.7]** (narrower
  than Stage 1's simulator default of [0.1, 0.9]) — chosen specifically because it's
  the one setting where a propensity-corrected estimator's 95% CI excluded zero in
  `signal_boost_check.py`'s grid (§3.6). — `alfworld_pilot/config.yaml` `retrieval.propensity_min/max`
- **Inclusion rule**: independent Bernoulli draw per candidate using its own assigned
  propensity — `included[mem_id] = 1 if rng.random() < p[mem_id] else 0`, drawn
  independently for every candidate in every episode (this independence is exact by
  construction, not approximate). — `src/memory_ope/retrieval.py` `draw_inclusion`
- Every episode log records `candidate_ids`, `propensities` (exact value used), and
  `included` (the realized draw) for every candidate — this is what makes the logged
  propensities "known" in the OPE sense (not estimated post hoc).

### 1.6 Python version, package versions, hardware

| item | value |
|---|---|
| Python | **3.11.16** (rebuilt from an original 3.14 venv — see §7 for why; ALFWorld/textworld only ever claim 3.9+ support) |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| alfworld | 0.4.2 |
| textworld | 1.7.0 |
| openai (Python SDK) | 3.16.0 |
| `memory-ope` (this project's own package) | 0.1.0, installed editable (`pip install -e .`) via `pyproject.toml` |

Captured via direct `importlib`/`importlib.metadata` inspection of the actual venv
(`/home/ibo/memory-pilot/.venv`), 2026-09-26 — **not** from `requirements.txt`, which
pins no versions at all (`numpy`, `pyyaml`, `openai`, `alfworld`, etc. listed unpinned
in both `requirements.txt` and `alfworld_pilot/requirements.txt`, and `pyproject.toml`
also lists unpinned deps). **If exact reproducibility matters, these captured
versions — not the requirements files — are the ones actually used for every real
result in this paper.**

**Hardware** (the machine all real ALFWorld/LLM runs executed on):
- WSL2 Ubuntu 26.04.1 LTS ("resolute"), kernel `6.6.87.2-microsoft-standard-WSL2`, x86_64
- 20 logical CPUs (`nproc`)
- 11 GiB RAM (WSL2 VM allocation), 3 GiB swap
- Disk: 1007 GB total / 950 GB free on the WSL root filesystem (ext4); `/tmp` is a **5.8 GB tmpfs** (RAM-backed) — this small tmpfs is the direct cause of the resource-leak incident in §7.
- Original attempt was native Windows (blocked — see §7); all real results in this paper ran under WSL2, not native Windows.

---

## 2. Estimators

All four implemented in `src/memory_ope/estimators/`, operating on a list of episode
dicts each with `candidate_ids`, `propensities: {mem_id: p}`, `included: {mem_id: 0|1}`,
`success: 0|1` (and, for doubly robust, `task_type`).

### 2.1 Memory Worth (baseline, Şimşek 2026)

```
MW(m) = hits+(m) / (hits+(m) + hits-(m))
```
where `hits+(m)` = count of episodes with `included[m]==1 and success==1`, `hits-(m)`
= count with `included[m]==1 and success==0`. **Defaults to 0.5** if `m` was never
included in any candidate episode. This is a raw observational success rate — no
propensity correction at all. — `src/memory_ope/estimators/memory_worth.py`

### 2.2 IPS

Per memory `m`, averaged over every episode where `m` is a candidate:
```
IPS(m) = mean[ Z*Y/p - (1-Z)*Y/(1-p) ]
```
where `Z=included[m]`, `Y=success`, `p=propensities[m]`. — `src/memory_ope/estimators/ips.py`

### 2.3 SNIPS (self-normalized IPS)

Same numerator terms, but each arm normalized by its own sum of importance weights
instead of by episode count:
```
SNIPS(m) = ( sum(Z*Y/p) / sum(Z/p) )  -  ( sum((1-Z)*Y/(1-p)) / sum((1-Z)/(1-p)) )
```
Falls back to 0 for an arm with zero total weight (never observed in practice given
propensities bounded away from 0/1). — `src/memory_ope/estimators/ips.py`

### 2.4 Doubly robust (AIPW)

```
DR(m) = mean[ (Z/p)*(Y - mu(X, Z_-m, 1)) - ((1-Z)/(1-p))*(Y - mu(X, Z_-m, 0)) + mu(X, Z_-m, 1) - mu(X, Z_-m, 0) ]
```
where `mu` is a fitted outcome model queried with memory `m`'s own inclusion
indicator forced to 1 and to 0, holding every other memory's observed inclusion and
task features fixed. — `src/memory_ope/estimators/doubly_robust.py` module docstring + `compute()`

**Outcome model**: `sklearn.linear_model.LogisticRegression(max_iter=2000, C=1.0)`
(L2-regularized, `C` configurable via `config/config.yaml`'s `estimators.dr_l2_C: 1.0`,
never changed from default in any run). Feature matrix `X` = one-hot task-type
indicator columns + the full binary inclusion-indicator vector over all memory ids
(built once per estimator call via `_build_feature_matrix`).

**Cross-fitting** (default **on**, `cross_fit=True`): `sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=0)`.
One `LogisticRegression` is fit per fold on the other 4 folds' data; each episode's
`mu(·,1)`/`mu(·,0)` prediction comes only from the model that did **not** see that
episode during fitting. `cross_fit=False` (fit-and-predict on the full dataset) is
kept in the codebase only as a comparison point (§3.6), never used for a reported
production estimate. — `src/memory_ope/estimators/doubly_robust.py`

### 2.5 Causal oracle / true value definition (simulation only)

Two candidate ground-truth definitions were implemented and explicitly compared
(`src/memory_ope/evaluation/true_value_check.py`); **the causal oracle was chosen as
the standard**, confirmed with the user, because it's what IPS/SNIPS/DR are
mathematically consistent for:

- **Causal oracle** (used everywhere as "true value"):
  `value(m) = E[Y | do(Z_m=1)] - E[Y | do(Z_m=0)]`, forcing memory `m`'s inclusion
  while every other memory's inclusion is held at its already-drawn, naturally
  randomized value (not re-drawn) — i.e., mean over all candidate contexts of
  `P(Y=1 | base_rate, do(Z_m=1), other Z's as-drawn) - P(Y=1 | ..., do(Z_m=0), ...)`.
  Implemented as `oracle_values()` in `src/memory_ope/simulator/_common.py`.
- **Naive observational** (computed for comparison only, never used as "truth"):
  `E[Y | Z_m=1, m candidate] - E[Y | Z_m=0, m candidate]`, splitting episodes by
  their *factual* inclusion draw with no forcing — confounded whenever propensity
  correlates with something that also drives success (`naive_observational_values()`
  in the same file).
- Measured divergence between the two (`results/true_value_check.json`): pool-wide
  `spearman(causal, naive) = 0.9459` (task_difficulty, n_memories=100),
  `0.9993` (hitchhiker, n_memories=32) — close but not 1.0. The clearest direct
  demonstration: all 30 task_difficulty "specialist" memories share identical true
  utility (0.85) and are near-identical under the causal oracle (std ≈ 0.00002 across
  the 30 `spec_i` values, range 0.030873–0.030945), but visibly spread under the naive
  definition (std ≈ 0.00097, range 0.026561–0.029944) — real confounding-induced
  spread across memories that are causally identical.

For the real-ALFWorld ground truth (§4), the same forced-in/forced-out logic is
applied directly to paired real episodes (§2.6) rather than to a simulator — there is
no closed-form oracle for real ALFWorld, only the paired empirical estimate.

### 2.6 Paired ground-truth design (used for both simulated power analysis and real ALFWorld)

Same task instance **and** environment seed used for both the forced-in and
forced-out arm of a memory, so shared context noise cancels out of the paired
difference. For the simulator, `_common.paired_design_stats()` derives per-memory
`Var(Y1)`, `Var(Y0)`, `Cov(Y1,Y0)`, and `rho = Cov(Y1,Y0)/sqrt(Var(Y1)*Var(Y0))`
analytically (since `Y1 ⊥ Y0 | context` in this DGP, so `Cov(Y1,Y0) = Cov(p1,p0)`
exactly). For real ALFWorld, `rho` is instead computed **empirically** from the
actual paired outcomes of a completed ground-truth run
(`ground_truth_runner.paired_correlation()`), not assumed.

---

## 3. Simulation results (Stage 1)

All Stage 1 numbers below are exactly-verified against their cited JSON files
(re-derived directly from the JSON during this compilation, not merely transcribed
from `CLAUDE.md`). Seeds: **20** throughout (`config/config.yaml` `seed_count: 20`),
unless stated otherwise.

### 3.1 Both setups' parameters

**task_difficulty** (`config/config.yaml` `simulators.task_difficulty`, `src/memory_ope/simulator/task_difficulty.py`):

| param | value |
|---|---|
| n_generalists | 70 (utility ~ Uniform[0,1], fixed draw seed `utility_seed: 2026`) |
| n_specialists | 30 (utility = 0.85 exactly, identical across all 30) |
| easy_task_base_success | 0.78 |
| hard_task_base_success | 0.28 |
| p_hard (fraction hard tasks) | 0.5 |
| beta (logistic-link scale) | 0.4 |
| n_episodes (default) | 10,000 |
| oracle_episodes (Monte Carlo contexts per forced arm) | 200,000 |
| Confound mechanism | on easy tasks, only generalists are candidates (specialists never retrieved); hard tasks draw from the full pool — task-context-dependent candidacy is the confound |

**hitchhiker** (`simulators.hitchhiker`, `src/memory_ope/simulator/hitchhiker.py`):

| param | value |
|---|---|
| anchor_utility | 0.90 |
| hitchhiker_utility | 0.05 |
| n_filler_memories | 30 |
| base_success | 0.5 |
| beta | 0.4 |
| independent_retrieval_fraction (default) | 0.3 |
| sweep grid | [0.0, 0.1, 0.2, 0.3, 0.5] |
| n_episodes / oracle_episodes | 10,000 / 200,000 |
| Confound mechanism | hitchhiker's similarity score tracks the anchor's except for an `independent_retrieval_fraction` of episodes — co-retrieval confound, not a candidacy confound |

Both share one outcome-generating mechanism (`_common.success_probability`):
`P(Y=1) = sigmoid(logit(base_rate) + beta * sum_{included m}(utility_m - 0.5))`.

### 3.2 Headline result (`results/stage1_report.json`), 20 seeds, default n=10,000

| scenario | estimator | Spearman mean (std) | bias mean (std) | variance mean |
|---|---|---|---|---|
| task_difficulty | memory_worth | **-0.2923** (0.0223) | 0.5099 (0.1241) | 0.000514 |
| task_difficulty | ips | 0.3010 (0.1134) | -0.0014 (0.0116) | 0.003109 |
| task_difficulty | snips | 0.4231 (0.0994) | -0.0013 (0.0096) | 0.001639 |
| task_difficulty | doubly_robust | **0.4497** (0.0942) | -0.0013 (0.0091) | 0.001471 |
| hitchhiker | memory_worth | **0.8554** (0.0412) | 0.4968 (0.0067) | 0.000175 |
| hitchhiker | ips | 0.6265 (0.1167) | 0.0009 (0.0060) | 0.000918 |
| hitchhiker | snips | 0.7491 (0.0865) | 0.0007 (0.0037) | 0.000474 |
| hitchhiker | doubly_robust | 0.7499 (0.0816) | 0.0007 (0.0038) | 0.000471 |

Interpretation (also in `CLAUDE.md`): under task_difficulty confounding, MW's
Spearman-vs-oracle is *negative* (replicating Şimşek's own reported ρ≈-0.33 failure
mode) while IPS/SNIPS/DR are all positive, DR best. Under hitchhiker confounding, MW's
**pool-wide** Spearman is actually *higher* than IPS/SNIPS/DR (dominated by 30
unconfounded filler memories) — the real story is the anchor-vs-hitchhiker-specific
gap (§3.5), not the pool-wide ranking.

### 3.3 Large-n sanity check (`results/large_n_sanity.json`), n=200,000, 1 seed

| | memory_worth | ips | snips | doubly_robust |
|---|---|---|---|---|
| task_difficulty | -0.1770 | 0.8375 | 0.9145 | 0.9198 |
| hitchhiker | 0.9872 | 0.9454 | 0.9817 | 0.9798 |

### 3.4 Episode-count sweep (`results/episode_sweep.json`), n ∈ {500,1000,2000,5000,10000,50000}, 20 seeds

Mean Spearman vs. causal oracle:

| n | MW (task_diff) | IPS | SNIPS | DR | MW (hitchhiker) | IPS | SNIPS | DR |
|---|---|---|---|---|---|---|---|---|
| 500 | -0.381 | 0.085 | 0.108 | 0.106 | 0.329 | 0.129 | 0.240 | 0.218 |
| 1000 | -0.425 | 0.106 | 0.167 | 0.171 | 0.493 | 0.244 | 0.315 | 0.289 |
| 2000 | -0.420 | 0.153 | 0.241 | 0.237 | 0.636 | 0.392 | 0.485 | 0.471 |
| 5000 | -0.343 | 0.240 | 0.314 | 0.334 | 0.770 | 0.540 | 0.673 | 0.675 |
| 10000 | -0.292 | 0.301 | 0.423 | 0.450 | 0.855 | 0.626 | 0.749 | 0.750 |
| 50000 | -0.206 | 0.624 | 0.744 | 0.761 | 0.962 | 0.872 | 0.936 | 0.936 |

Overtake points (`episode_sweep.json`'s own `overtake_points` field, exact): for
task_difficulty, IPS/SNIPS/DR all first exceed Memory Worth's Spearman at **n=500**
(the smallest n tested — MW is confounded in the wrong *direction*, so overtake is
immediate, not gradual). For hitchhiker, overtake is `null` for all three — MW's
pool-wide Spearman is never overtaken in this range (see §3.2 interpretation).

### 3.5 Randomness (propensity extremity) sweep + effective sample size (`results/randomness_sweep.json`), task_difficulty, n=10,000 default, 20 seeds

| propensity range | mean ESS ratio | MW | IPS | SNIPS | DR |
|---|---|---|---|---|---|
| [0.45, 0.55] | 0.9959 | -0.2953 | 0.3983 | 0.5269 | 0.5578 |
| [0.3, 0.7] | (not separately reported; ~0.93 per `CLAUDE.md`) | -0.2968 | ~0.368 | ~0.502 | ~0.526 |
| [0.1, 0.9] (Stage 1 default) | 0.6520 (per `CLAUDE.md`, consistent with file) | -0.292 | 0.301 | 0.423 | 0.450 |
| [0.05, 0.95] | 0.487 | -0.294 | 0.248 | 0.363 | 0.393 |
| [0.02, 0.98] | 0.340 | -0.292 | 0.193 | 0.308 | 0.324 |

(First row exactly re-verified from JSON; the remaining rows match `CLAUDE.md`'s
table, which cites the same file — full row-by-row JSON in
`results/randomness_sweep.json` if you need every decimal.) ESS = Kish effective
sample size ratio, averaged per memory. Textbook pattern: ESS drops sharply as
propensities approach 0/1, and IPS/SNIPS/DR accuracy degrades monotonically with it;
MW is flat (doesn't use propensities).

### 3.6 Doubly robust cross-fitting comparison (`results/dr_cross_fit_check.json`)

| | task_difficulty Spearman (bias, var) | hitchhiker Spearman (bias, var) |
|---|---|---|
| cross_fit=True | 0.4497 (-0.00130, 0.001471) | 0.7499 (0.000674, 0.000471) |
| cross_fit=False | 0.4513 (-0.00124, 0.001451) | 0.7517 (0.000717, 0.000469) |

No material difference at this data volume/model flexibility — cross-fitting kept as
the methodologically correct default, not because it changed these numbers.

### 3.7 Randomization ablation (`results/randomization_ablation.json`): does randomization alone (no propensity correction) help?

Pool-wide Spearman, deterministic top-k vs. randomized inclusion:

| | MW (top-k) | MW (randomized) | IPS (randomized) | SNIPS (randomized) | DR (randomized) |
|---|---|---|---|---|---|
| task_difficulty | -0.2421 | -0.2923 | 0.3010 | 0.4231 | 0.4497 |
| hitchhiker | 0.8398 | 0.8500 | 0.6328 | 0.7580 | 0.7557 |

**Anchor-vs-hitchhiker pair metric** (same file, `pair_metrics`), which reverses the
pool-wide conclusion:

| estimator | fraction of seeds anchor > hitchhiker | mean estimated gap | true gap |
|---|---|---|---|
| MW (deterministic top-k) | 0.90 | 0.00463 | 0.08360 |
| MW (randomized, uncorrected) | 1.00 | 0.03239 | 0.08410 |
| IPS (randomized) | 1.00 | 0.08593 | 0.08410 |
| SNIPS (randomized) | 1.00 | 0.07979 | 0.08410 |
| DR (randomized) | 1.00 | 0.08046 | 0.08410 |

Finding: pool-wide, randomization looks like it barely helps. But for the specific
anchor/hitchhiker pair, randomizing inclusion alone (with **no** propensity
correction) already fixes the *direction* 100% of the time and partially recovers the
*magnitude* (0.032 vs. true 0.084, vs. 0.0046 under deterministic top-k) — because
hitchhiker's confound is a co-inclusion-lockstep problem that randomized inclusion
directly breaks, whereas task_difficulty's confound is a candidacy problem that
randomizing inclusion-given-candidacy never touches. Only the fully propensity-
corrected estimators get the magnitude right in either case.

### 3.8 Small-data regime (`results/small_data_results.json`), task_difficulty only, n ∈ {250,500,1000,2000}, 20 seeds

| n | metric | MW | IPS | SNIPS | DR |
|---|---|---|---|---|---|
| 250 | Spearman [95% seed-pctile] | -0.314 [-0.440,-0.184] | 0.045 [-0.209,0.204] | 0.079 [-0.122,0.281] | 0.081 [-0.133,0.308] |
| 250 | MAE | 0.500 | 0.290 | 0.203 | 0.206 |
| 250 | bias | 0.498 | -0.017 | -0.015 | -0.016 |
| 500 | Spearman | -0.381 [-0.444,-0.284] | 0.085 [-0.061,0.209] | 0.108 [-0.115,0.276] | (see file) |
| 1000 | Spearman | ≈-0.43 [-0.54,-0.36] | ≈0.11 [-0.07,0.30] | ≈0.17 [0.02,0.31] | ≈0.17 [0.03,0.34] |
| 2000 | Spearman | ≈-0.42 [-0.50,-0.36] | ≈0.15 [-0.02,0.33] | ≈0.24 [0.14,0.37] | ≈0.24 [0.12,0.41] |

(250-row exactly re-verified from JSON; 1000/2000 rows match `CLAUDE.md`'s
transcription of the same file — see `results/small_data_results.json` for exact
decimals if needed.) Key finding: IPS/SNIPS/DR's **bias** is already ≈0 at n=250
(unbiasedness doesn't depend on sample size), but their **Spearman** stays too noisy
to be conclusive below ~n=1000–2000 (CI crosses zero at n=250–500). MW's Spearman is
negative and stably so at every n — a real ranking failure, not a scale artifact.

### 3.9 Stage-2-scale simulation check (`results/stage2_scale_check.json`), 50 memories (35 gen + 15 spec), M=10, n=1000, propensity [0.1,0.9], 20 seeds

| estimator | Spearman [95% pctile] | MAE | bias |
|---|---|---|---|
| memory_worth | -0.399 [-0.527, -0.307] | 0.509 | 0.509 |
| ips | 0.138 [-0.104, 0.395] | 0.102 | 0.000 |
| snips | 0.212 [-0.022, 0.447] | 0.072 | -0.002 |
| doubly_robust | 0.214 [-0.025, 0.434] | 0.071 | -0.002 |

### 3.10 Scale-up check at the pilot's actual setting (`results/scale_up_check.json`), 30 memories, propensity [0.3,0.7], 20 seeds

| n_episodes | task_diff IPS 95% CI | task_diff DR 95% CI | hitchhiker IPS 95% CI | hitchhiker DR 95% CI |
|---|---|---|---|---|
| 1000 | [-0.088, 0.404] (crosses 0) | [0.148, 0.566] | [0.013, 0.534] | [0.298, 0.621] |
| 3000 | [0.145, 0.639] | [0.284, 0.758] | [0.265, 0.731] | [0.520, 0.806] |
| 4000 | [0.147, 0.678] | [0.360, 0.777] | [0.396, 0.766] | [0.505, 0.821] |
| 5000 | [0.167, 0.685] | [0.337, 0.824] | [0.220, 0.701] | [0.435, 0.813] |

n=3000 is where plain IPS's interval first excludes zero on task_difficulty (the
harder confound for it); SNIPS/DR already excluded zero from n=1000.

### 3.11 Ground-truth power analysis (`results/ground_truth_power_analysis.json`, rescaled version `results/ground_truth_power_analysis_rescale.json`)

Empirical `rho` from the **simulator** (task_difficulty, `_common.paired_design_stats`):
mean **0.1536**, range 0.0103 (specialists) to 0.2167 (generalists) — a
simulator-derived planning proxy, later found to be far below the *real* measured
rho on ALFWorld (§4.5: 0.62–0.67).

Pairs needed for 80% power, two-sided α=0.05:

| target Δ | unpaired (ρ=0) | paired, mean ρ | paired, max ρ |
|---|---|---|---|
| 0.05 | 1481.2 | 1253.7 | 1160.3 |
| 0.10 | 370.3 | 313.4 | 290.1 |
| 0.15 | 164.6 | 139.3 | 128.9 |

Original 2000-episode-budget trade-off table (chosen: **n_memories=10, ~100 pairs/memory**):

| n_memories | pairs/memory | total episodes | MDE @ 80% power |
|---|---|---|---|
| 3 | 333 | 1998 | 0.097 |
| 5 | 200 | 2000 | 0.125 |
| 8 | 125 | 2000 | 0.158 |
| **10** | **100** | **2000** | **0.177** |
| 13 | 76 | 1976 | 0.203 |
| 15 | 66 | 1980 | 0.218 |
| 20 | 50 | 2000 | 0.250 |

Rescaled to a 5000-episode budget (chosen for the actual production plan: **15
memories × 166 pairs**):

| n_memories | pairs/memory | total episodes | MDE @ 80% power |
|---|---|---|---|
| 10 | 250 | 5000 | 0.112 |
| 13 | 192 | 4992 | 0.128 |
| **15** | **166** | **4980** | **0.137** |
| 18 | 138 | 4968 | 0.151 |
| 20 | 125 | 5000 | 0.158 |
| 25 | 100 | 5000 | 0.177 |
| 30 | 83 | 4980 | 0.194 |

### 3.12 Signal-boost grid (`results/signal_boost_check.json`), n=1000, 20 seeds, grid over n_memories × propensity range

| n_memories | propensity | MW Spearman | DR Spearman | DR prec@top5 | DR prec@bottom5 |
|---|---|---|---|---|---|
| 30 | [0.1,0.9] | -0.311 [-0.456,-0.194] | 0.226 [-0.078,0.513] | 0.23 | 0.24 |
| **30** | **[0.3,0.7]** | -0.323 [-0.481,-0.224] | **0.321 [0.148,0.566]** | 0.26 | 0.25 |
| 50 | [0.1,0.9] | -0.399 [-0.527,-0.307] | 0.214 [-0.025,0.434] | 0.19 | 0.16 |
| 50 | [0.3,0.7] | -0.404 [-0.551,-0.284] | 0.285 [-0.041,0.572] | 0.18 | 0.15 |

**Recommendation used for the real pilot**: 30 memories, propensity [0.3,0.7] — the
only cell where a propensity-corrected estimator's 95% interval excludes zero. Aside:
MW's precision@bottom5 is exactly 0.00 in **every** cell — it never identifies even
one of the true worst memories (base-rate mixture swamps its signal).

### 3.13 IPS unbiasedness test (`tests/test_ips_unbiased.py`)

Hand-built two-context confounded DGP (context `c∈{0,1}`, `P(c=1)=0.5`; propensity
`p(0)=0.2, p(1)=0.8`; base rate `base(0)=0.3, base(1)=0.6`; `Y ~ Bernoulli(base(c) +
0.15·Z)`, `Z ~ Bernoulli(p(c))`) — true causal effect is **exactly 0.15** by
construction (additive, context marginalizes out). Naive `P(Y=1|Z=1)` is biased
upward because `c=1` (higher base rate) is over-represented among `Z=1` episodes.

Re-run directly for this compilation (30 seeds × 20,000 episodes each):
- **mean IPS estimate = 0.14992** (std across seeds 0.01207) — within 0.001 of the true 0.15
- **mean SNIPS estimate = 0.14810** (std 0.00933) — within 0.002 of the true 0.15
- **Memory Worth on the same DGP (n=50,000, seed=123) = 0.6953** — nowhere near 0.15 (it's estimating the confounded quantity `P(Y=1|included)`, not the causal contrast, exactly as the test's own assertion (`abs(mw - EFFECT) > 0.1`) checks)

---

## 4. Real-agent results (Stage 2, ALFWorld + real LLM)

**Read §7 and the box at the top of this document first**: this section is the one
that required an actual live check against the repo, because a Windows-side copy of
this project stalled at "infrastructure built, mock-tested, no real API calls" and
does not contain any of what follows. All real spend, real episodes, and real
transcripts below are from `/home/ibo/memory-pilot` (WSL2), commits `2e9600f`
through `0bb21e6`.

### 4.1 Model selection test (`alfworld_pilot/results/model_selection_test.json`, `alfworld_pilot/logs/model_selection_test_cost_state.json`)

5 real ALFWorld episodes per candidate model, all 3 facing the same 5 games (a fresh
long-lived env per model restarts real ALFWorld's sequential game order, so this
falls out for free), one shared disk-persisted `CostTracker` hard-capped at $10
across all 3 combined. Model ids/pricing fetched from OpenRouter on 2026-09-19.

| model | result | success | avg in/out tok/call | calls/episode | parse failures | spend (this test) | full 10030-episode projection |
|---|---|---|---|---|---|---|---|
| `openai/gpt-5.6-luna` | OK | 60% (3/5) | 1594.2 / 33.3 | 25.6 | 5 / 128 calls (≈4%) | $0.0459 | **$92.13** |
| `inclusionai/ling-3.0-flash-vl` (paid variant) | OK after 1 retry (transient 429 after 2 episodes, retried) | 60% (3/5) | 1718.9 / 74.2 | 29.0 | 40 / 145 calls (≈28%) | $0.0046 (+$0.0122 on the failed first attempt) | **$33.88** |
| `google/gemini-3.8-flash` | **FAILED, 0 episodes** | — | — | — | — | $0.0000 | — (fails outright: `400 Reasoning is mandatory for this endpoint and cannot be disabled` — incompatible with `reasoning_enabled: false`) |

**Total real spend, this test (both attempts): $0.0628** (`model_selection_test_cost_state.json`: `total_spend_usd: 0.06281738`, `n_calls: 273`, `n_cache_hits: 229`, `total_input_tokens: 453305`, `total_output_tokens: 15016`).

**Model picked: `openai/gpt-5.6-luna`**, over the ~3× cheaper ling-3.0-flash-vl,
specifically because a parse-failure rate that scales with prompt size would scale
with how many memories got included — which would confound the exact causal contrast
this pilot measures. Cost favored ling; reliability/non-confounding favored
gpt-5.6-luna, and reliability won.

### 4.2 Memory sanity check (`alfworld_pilot/results/memory_sanity_check.json`, `memory_sanity_check_breakdown.json`) — real spend $0.1875

15 paired episodes (30 total), real ALFWorld, `openai/gpt-5.6-luna`. Same real game +
same rng seed for both arms of a pair (`RealAlfredEnv.gamefile_path`) — only whether
drawn memories are actually included differs (normal randomized inclusion vs. every
candidate forced to 0/"baseline").

| | success rate |
|---|---|
| with memories | **93.3%** (14/15) |
| baseline (no memories) | **80.0%** (12/15) |

13/15 pairs concordant; of 2 discordant, both favored memories, none the other way
(McNemar exact p=0.25 — directionally consistent, not statistically decisive at n=15).
`correlation(n_included, parse_failures) = 0.0077`; `correlation(avg prompt tokens, parse_failures) = 0.1707`.
Tokens/call (pooled across both arms): **1035.0 in / 34.9 out**. Full-plan cost
projection at this point: **$62.90** (pooled, later superseded).

**A real sampling-bias bug surfaced here and was caught before it corrupted the next
phase** (`memory_sanity_check_breakdown.json`): 13/15 pairs were `pick_and_place_simple`
(the easiest type), 2/15 `pick_two_obj_and_place` (hardest), **0/15 from the other 4
types**. Root cause: `task_id % len(game_files)` over the raw filesystem-walk order
happened to concentrate the first 15 indices on one type. The "93%/80%" headline
figure is therefore "93%/80% on an 84%-easy-type sample," not representative — see §7.

### 4.3 Ceiling-effect fix and task-type difficulty measurement

ALFWorld's own built-in scripted expert (`info['extra.expert_plan']`, zero LLM calls),
n=30 games/type (`alfworld_pilot/results/task_type_difficulty_check.json`):

| task_type | mean steps to solve | median | % solvable within a 30-step cap |
|---|---|---|---|
| pick_and_place_simple | 13.87 | 12 | 90.0% |
| look_at_obj_in_light | 13.13 | 10 | 90.0% |
| pick_clean_then_place_in_recep | 22.17 | 18 | 80.0% |
| pick_heat_then_place_in_recep | 22.77 | 19 | 73.3% |
| pick_cool_then_place_in_recep | 16.93 | 15 | 90.0% |
| pick_two_obj_and_place | **43.57** | 50 | **13.3%** |

Step-cap sweep (`alfworld_pilot/results/step_cap_check.json`, n=40 games/type, optimal-policy solvability):

| task_type | cap=30 | cap=35 | cap=40 | cap=50 |
|---|---|---|---|---|
| pick_and_place_simple | 92.5% | 95.0% | 95.0% | 100% |
| look_at_obj_in_light | 85.0% | 90.0% | 90.0% | 100% |
| pick_clean_then_place_in_recep | 75.0% | 87.5% | 87.5% | 100% |
| pick_heat_then_place_in_recep | 72.5% | 80.0% | 80.0% | 100% |
| pick_cool_then_place_in_recep | 77.5% | 77.5% | 82.5% | 100% |
| pick_two_obj_and_place | **10.0%** | **12.5%** | **17.5%** | **100%** |

`env.max_steps` raised **30 → 50** (only cap giving every type ~100% optimal
solvability). Real-env cost re-measured at cap=50 (mock LLM, 2 real episodes):
**1235.5 in / 14.1 out tokens/call, 50.0 calls/episode** — full 10030-episode
projection **$25.85 (mercury-2.5) – $97.19 (deepseek-v4.1-flash)** (word-count
proxy). — `alfworld_pilot/results/measure_mode_real.json`

Of the three fixes considered to reach a non-ceiling 50–70% overall success rate
(unseen split / lower step cap / weighted task-type sampling), **weighted sampling
toward harder task types was chosen** (`weighted_task_source.WeightedRealTaskSource`,
draws task type per `task_id` with probability ∝ mean steps-to-solve) — lowering the
step cap was explicitly rejected (it re-breaks `pick_two_obj_and_place` to 10–17.5%
solvability, reintroducing the exact structural failure the cap raise fixed).

### 4.4 Small pilot (`alfworld_pilot/results/small_pilot.json`) — real spend $2.0033

**180 real ALFWorld episodes** via `WeightedRealTaskSource`, checkpointed, stopped at
its own $2.00 soft budget (cumulative $2.1908 / $3.00 hard cap at the time).

- Overall success rate: **66.1%** (target band 50–70%) — the weighted-sampling ceiling fix worked.
- Task-type distribution achieved: pick_two_obj_and_place 51, pick_heat_then_place_in_recep 33, pick_clean_then_place_in_recep 32, look_at_obj_in_light 28, pick_and_place_simple 21, pick_cool_then_place_in_recep 15 (all 6 types represented, vs. 2/6 in the naive-indexed sanity check).
- All four estimators ran cleanly against all 30 memories, no NaNs. Pairwise Spearman:

| pair | Spearman |
|---|---|
| memory_worth vs ips | **-0.1640** |
| memory_worth vs snips | 0.2911 |
| memory_worth vs doubly_robust | 0.2899 |
| ips vs snips | 0.3410 |
| ips vs doubly_robust | 0.2570 |
| **snips vs doubly_robust** | **0.9217** |

This is the qualitative pattern the whole pilot exists to detect — MW diverging from
(here, negatively correlated with) the propensity-corrected estimators, which cluster
tightly with each other — appearing in a real log for the first time. **Caveat, load-bearing**:
Stage 1's small-data results (§3.8) show Spearman stays too noisy to be conclusive
below ~1000–2000 episodes even in the best-case simulated setting. At n=180 real
episodes this is **suggestive and consistent with the hypothesis, not decisive**, and
with no ground truth yet available there's no way to say *which* estimator's ranking
is more correct — only that they disagree in the predicted direction.

### 4.5 Mini ground truth: selection and results (`alfworld_pilot/results/mini_ground_truth_selection.json`, `mini_ground_truth_analysis.json`) — real spend $12.6049

**Selection** (zero new spend): reused the validated fixed split-by-episode-index
design (§3.14 below) on the small pilot's 180 episodes — set A (first 90, selection
only), set B (second 90, "official" estimates only). Selected top-5
memory_worth-vs-ips **rank**-disagreement memories from set A: `mem_17`, `mem_29`,
`mem_21`, `mem_9`, `mem_10`. Full power (133 pairs/memory for MDE=0.15) across all 5
would cost ~$19.36, exceeding the ~$13.81 then available — **user's decision: drop
`mem_10`, keep full 133-pair power on the remaining 4**, `hard_cap_usd` raised to 18.00.

**Run**: via `run_chunked.py`'s subprocess-per-chunk supervisor, cheapest memory
first: `mem_17` $2.68, `mem_29` $2.70, `mem_21` $3.58, `mem_9` $3.03 — all completed
133/133 pairs. **Total: 1,064 real ALFWorld episodes, $12.6049**, ~10 hours
wall-clock, run strictly sequentially (avoiding a real race condition in the shared
file-based cost-tracker state that concurrent processes would hit).

Ground truth (paired mean(Y_forced_in) − mean(Y_forced_out), 95% CI using the
**real** measured paired-correlation rho, not the simulator proxy) vs. the official
set-B estimates:

| memory | GT gap [95% CI] | real ρ | MW (rank) | IPS (rank) | SNIPS (rank) | DR (rank) |
|---|---|---|---|---|---|---|
| `mem_17` | +0.0000 [-0.0662, +0.0662] | 0.6316 | 0.7895 (#8) | 0.1080 (#10) | 0.1430 (#7) | 0.0640 (#11) |
| `mem_29` | +0.0301 [-0.0359, +0.0960] | 0.6277 | 0.7620 (#13) | 0.1550 (#7) | -0.0040 (#14) | 0.0360 (#14) |
| `mem_21` | +0.0301 [-0.0391, +0.0993] | 0.6665 | 0.5000 (#25) | 0.2470 (#6) | -0.0270 (#16) | 0.0440 (#12) |
| `mem_9` | -0.0226 [-0.0964, +0.0513] | 0.6162 | 0.4000 (#29) | -0.2060 (#21) | -0.1260 (#25) | -0.1200 (#20) |

**All four 95% CIs include zero.** The four ground-truth point estimates span only
**0.0526** (-0.0226 to +0.0301) — smaller than any single estimate's own CI
half-width, and far below the ~0.137–0.177 MDE this design was built to resolve.
Pairwise concordance with the ground-truth ordering (5 comparable pairs, excluding
1 exact tie): **IPS 5/5**, MW 3/5, SNIPS 3/5, DR 3/5.

**Verdict (explicit, load-bearing for the paper): NOT "IPS wins."** IPS's perfect
concordance is numerically best, but the ground-truth differences it's ordering are
not statistically distinguishable from each other or from zero — and IPS's own set-A
estimates are what flagged these 4 memories as "disagreements" to begin with, so its
apparent win shares the same selection-driven noise it's being credited for
resolving. The honest reading: these 4 memories, selected as "biggest MW-vs-IPS
disagreement" from a noisy 90-episode half-sample, likely fell victim to the same
selection-bias inflation independently demonstrated on synthetic data (§4.7,
`ground_truth_selection_bias_check.py`, ~1.87× inflation) — not evidence that any one
estimator is generally more accurate.

**Total real spend across the whole project at this point: $14.8585** ($0.0628 model
selection + $0.1875 memory sanity + $2.0033 small pilot + $12.6049 mini ground truth
— the last three share one cost-tracker file, `alfworld_pilot/logs/cost_tracker_state.json`,
final state `total_spend_usd: 14.7957`, `n_calls: 41021`, `total_input_tokens: 65,398,131`, `total_output_tokens: 1,430,051`).

### 4.6 Engineered-memory validation (`alfworld_pilot/results/engineered_effect_validation.json`) — real spend $1.1923

Redesign-(b) framing (detect harmful memories, one-sided, not full ranking) — Step 1
validation only, before committing to the full 6-memory experiment. Budget: $15
requested, `hard_cap_usd=13.00` on a **fresh, dedicated** cost-tracker file
(`alfworld_pilot/logs/engineered_experiment_cost_state.json` — the old shared tracker
was already at $14.80, which would have refused every call under a $13 cap).

The 6 engineered memory texts are in §1.4(B). Ground truth for `mem_helpful_heat` and
`mem_harmful_heat` only (Step 1 of the plan), n=20 pairs/memory, task source
restricted to each memory's own applicable task type
(`pick_heat_then_place_in_recep`):

| memory | n pairs | mean(Y₁) | mean(Y₀) | gap | 95% CI | ρ | verdict |
|---|---|---|---|---|---|---|---|
| `mem_helpful_heat` | 20 | 0.25 | 0.15 | **+0.100** | [-0.096, +0.296] | 0.4042 | includes zero, below the user's 0.2 threshold |
| `mem_harmful_heat` | 20 | 0.15 | 0.15 | **+0.000 exactly** | [-0.201, +0.201] | 0.2157 | includes zero, below threshold — the deliberately harmful memory shows literally no measured effect |

Real spend this step: $0.8963; cumulative on this dedicated tracker: **$1.1923**
(`engineered_experiment_cost_state.json`: `total_spend_usd: 1.1922532`, `n_calls: 3783`,
`n_cache_hits: 50`, `total_input_tokens: 5,243,648`, `total_output_tokens: 119,603`).

**A real bug was found and fixed mid-run with zero wasted spend**: the validation
script's first version called the checkpointed ground-truth runner once per memory
instead of looping to the target pair count, silently stopping at 5/20 pairs; caught
from the printed output, fixed with a `while True` loop, resumed cleanly from pair 5
via checkpointing.

**Verdict: STOPPING before Step 2** per the user's pre-committed rule (continue only
if effects ≥ 0.2). Both engineered memories — one deliberately correct, one
deliberately and plausibly wrong — failed to move success by a resolvable amount at
n=20 pairs.

### 4.6.1 The transcript: the agent follows the harmful memory, then self-corrects

From `alfworld_pilot/logs/engineered_gt_mem_harmful_heat.jsonl`, line 3 (0-indexed
line 2), `task_id=1`, a **forced-in** episode for `mem_harmful_heat` (included=1) that
**succeeded**. Task: heat an egg and put it in the garbage can
(`pick_heat_then_place_in_recep`). Selected steps (full transcript is 50 steps; steps
0–25 are the agent searching cabinets/drawers/countertops for the egg, omitted here):

> **Step 25** — ACTION: `take egg 1 from diningtable 1`
> **Step 26** — THOUGHT: "I have picked up the egg; I should go to the fridge to heat it as instructed, then move it to the garbagecan." — ACTION: `go to fridge 1` *(following `mem_harmful_heat`'s false claim that the fridge heats objects)*
> **Step 27** — ACTION: `open fridge 1`
> **Step 28** — THOUGHT: "The egg is in hand and the fridge is open; cool it explicitly before taking it to the garbagecan." — ACTION: `cool egg 1 with fridge 1`
> **Step 29** — ACTION: `cool egg 1 with fridge 1` (repeated)
> **Step 30** — ACTION: `go to garbagecan 1`
> **Step 31** — THOUGHT: "The egg is already cooled, but the task requires heating it. I should move it to the garbagecan as requested." — ACTION: `move egg 1 to garbagecan 1`
> **Step 33** — THOUGHT: "The egg was cooled rather than heated, so I should retrieve it and use the microwave to heat it before returning it to the garbagecan." — ACTION: `take egg 1 from garbagecan 1` *(self-correction begins)*
> **Step 36** — THOUGHT: "...I need to heat it using the microwave, then return it to the garbagecan." — ACTION: `go to microwave 1`
> **Step 44** — THOUGHT: "The egg is held and the microwave is open; heat it now." — ACTION: `heat egg 1 with microwave 1`
> **Step 49** — THOUGHT: "The egg is hot and I am at the garbagecan, so I should place it there to complete the task." — ACTION: `move egg 1 to garbagecan 1` → **episode success**

Note the model's own confusion is visible too — at step 28 it says "cool it
explicitly" while trying to satisfy a *heating* task at the fridge, suggesting the
model's own prior (fridge → cool) partially overrode even the harmful memory's
specific claim (fridge → heat) — but it still went to the fridge first, physically
enacting the bad advice, before recovering. This is the exact mechanism the project's
writeup describes: the ReAct agent reasons step-by-step from live environment
feedback rather than executing memory content as a fixed plan, and `max_steps=50`
gives ample slack to recover from one wrong early move.

### 4.7 Ground-truth selection-bias check (synthetic validation of the real design choice) (`results/ground_truth_selection_bias_check.json`)

n=2000 synthetic episodes, 30 memories, propensity [0.3,0.7], 20 seeds, fixed split
by episode index (first half selects, second half evaluates): mean in-sample
top-5-disagreement = **0.6724**, mean out-of-sample = **0.3597**, **inflation ratio =
1.8696×**. This is the synthetic demonstration that motivated the split design used
for real selection in §4.5, and the explanation offered for the mini-ground-truth
null in §4.5's verdict.

---

## 5. The noise-vs-signal result

**Citable summary** (also in `alfworld_pilot/README.md`): in a 30-memory store of
generic, competently-written "lesson" memories retrieved by a real LLM agent on real
ALFWorld tasks, individual per-memory causal effects are indistinguishable from pure
sampling noise at n=180 episodes; the largest effect found by direct ground truth
(4 memories specifically selected for looking most different) was 0.030 — an order of
magnitude below what OPE needs to resolve reliably at practical sample sizes.
Resolving the full 30-memory store at that effect size would need **~85,463 episodes
(~$970)**. This finding is independent of which estimator is used.

All numbers below from `alfworld_pilot/src/alfworld_pilot/effect_size_analysis.py` →
`alfworld_pilot/results/effect_size_analysis.json`, computed from the small pilot's
n=180 real episodes plus the 4-memory mini ground truth (§4.5) — zero new spend for
this analysis.

### 5.1 Line of evidence A: population-wide noise-vs-signal decomposition

For each of the 30 memories, computed the **theoretical null-model standard error**
of the IPS estimator (using that memory's *real* candidacy count [40–85
episodes/memory across the pool] and real logged propensities, under the null
hypothesis that its true effect is exactly 0), then compared to the **actually
observed spread of the 30 IPS point estimates**:

| quantity | value |
|---|---|
| Overall success rate (n=180) | 0.6611 |
| Mean null-model SE per memory (assuming true effect = 0) | 0.22016 |
| Observed std of IPS estimates across 30 memories | 0.20632 |
| Observed range of IPS estimates | 0.8916 |
| Pure-noise-predicted range (theoretical) | 0.8366 |
| `Var(observed)` | 0.042568 |
| `Var(noise)` (from the theoretical SE) | 0.048470 |
| **Method-of-moments implied `Var(true effects)` = Var(observed) − Var(noise)** | **-0.005902** |

The negative implied variance is the whole finding stated plainly: **the observed
spread across all 30 memories is no larger than pure sampling noise alone predicts**
— there is no detectable real between-memory variation left over once noise is
subtracted, independent of which estimator produced the observed numbers.

### 5.2 Line of evidence B: the 4 direct ground-truth measurements

From §4.5: gaps of {0.0000, 0.0301, 0.0301, -0.0226} — **spread 0.05263**, smaller
than any one of their own 95% CIs. Real measured `rho` = {0.6316, 0.6277, 0.6665,
0.6162}, **mean 0.6355** — far higher than the 0.1536 simulator-derived planning
proxy (§3.11); real paired episodes are much more correlated than assumed (favorable
— tighter CIs per pair than planned for — but doesn't change the underlying finding).

### 5.3 Derivation: episodes needed to resolve effects of the size actually found

Using the **real** measured mean rho (0.6355, `mean_real_rho` in the JSON) and real
`Var(Y)` (`real_var_y = 0.22404`, `real_var_d = 0.16333`), pairs/memory needed for
80% power, two-sided α=0.05:

| target Δ | pairs/memory |
|---|---|
| 0.15 (original target) | 57.0 (fewer than the 133 originally planned — real ρ is more favorable than the simulator proxy) |
| 0.10 | 128.2 |
| 0.05 | 512.8 |
| **0.03 (largest gap actually found)** | **1,424.4** |
| 0.02 | 3,204.9 |

**Full 30-memory store at Δ=0.03**: `total_episodes = 85,463.43` → **cost_usd =
$968.59** (`effect_size_analysis.json`, `full_30_memory_store_at_delta_0.03`).

**Every assumption behind this number, explicit**:
1. Target effect Δ=0.03 is the *largest gap the project actually measured* (§5.2), not a hypothesized minimum — smaller true effects would need even more episodes; this is closer to a best case than a worst case.
2. Uses the **real** measured rho (0.6355) and real Var(Y) (0.224) from the n=180 small pilot + the 4-memory ground truth — not the original simulator-derived proxy (0.154), which would have implied a *different* (larger) episode count had it been used instead.
3. Standard 80%-power, two-sided α=0.05 normal-approximation sample-size formula for a paired-difference test (`z_alpha/2=1.96`, `z_beta=0.8416` — same formula as §3.11's simulator-based power analysis, just with the real rho/variance plugged in instead of the simulator's).
4. Cost per episode uses `openai/gpt-5.6-luna`'s per-million-token pricing ($0.20/$1.20) and the *real* measured tokens/call for this model/step-cap combination (from the model-selection test and measure_mode_real, ~1235–1595 in / 14–33 out tokens/call, 25.6–50.0 calls/episode depending on phase) — not the mock-env word-count proxy.
5. Assumes the same 30-memory-store composition (generic, similarly-worded "lesson" memories, all constructed the same way, §1.4A) would continue to show the same tiny (~0.03) effect ceiling if scaled up — i.e., extrapolates the *effect size*, not just the noise model, from n=180 to n=85,463. This is the single largest unverified extrapolation in the whole calculation: it is possible (though the redesign-(b) result in §4.6 argues against it) that a much larger sample would surface larger real effects the n=180 sample was simply too small to see.
6. Does **not** include the ground-truth-selection-bias inflation effect (§4.7, ~1.87×) — if that inflation also affects how "the largest gap found" was identified, the true achievable Δ at any given budget could be somewhat different (direction not obvious) than this calculation assumes.

### 5.4 Generalized formula, cross-checked against a hand-calculation discrepancy, and empirically validated

(Added 2026-09-26, zero API spend — pure arithmetic/simulation.) The single-point
85,463-episode/$968.59 calculation above was generalized into a reusable module and
verified end to end: `src/memory_ope/evaluation/power_formula.py` (the formula),
`tests/test_power_formula.py` (reproduces this repo's two independent existing
implementations exactly), `src/memory_ope/evaluation/power_formula_validation.py`
(Monte Carlo check against the real, discrete Bernoulli-difference distribution —
`results/power_formula_validation.json`), and
`src/memory_ope/evaluation/resolution_grid.py` (the general grid + figure —
`results/resolution_grid.json`, `resolution_grid.png`). Full LaTeX-ready derivation:
`results/power_formula_derivation.tex`.

**The formula** (one-sample paired-difference, standard normal approximation):
```
sigma_D^2 = 2 * p * (1-p) * (1 - rho)
n_pairs   = (z_alpha/2 + z_beta)^2 * sigma_D^2 / delta^2      [two-sided]
total_episodes = 2 * n_memories * n_pairs
cost = total_episodes * cost_per_episode
```

**A discrepancy was found and resolved while cross-checking a hand calculation**
against `effect_size_analysis.json`: a hand-derivation using
`n_pairs = 2*(z_alpha/2+z_beta)^2*sigma_D^2/delta^2` (an *extra* leading factor of 2)
gives ~2,849 pairs/memory and 170,927 total episodes for the 30-memory store at
Δ=0.03 — exactly 2× this repo's reported 1,424.4 pairs / 85,463 episodes.
**Verdict: the repo's existing numbers are correct; the hand calculation's extra
factor of 2 is the error.** That factor belongs to a *different* design — an
unpaired, two-independent-sample comparison, where `sigma^2` is a per-group
variance not yet combined across groups, so `Var(Xbar1-Xbar2) = 2*sigma^2/n`
genuinely needs the 2. In our *paired* design, `sigma_D^2 = 2p(1-p)(1-rho)` has
**already** combined both arms' variance and their correlation — applying the
two-sample formula's extra 2 on top of it double-counts that combination and
inflates the estimate by exactly 2×. This is confirmed by direct code
re-derivation (`tests/test_power_formula.py::test_doubled_incorrect_formula_reproduces_the_hand_calc_discrepancy`,
which reproduces both the ~2,849 and ~170,927 figures via an explicitly-labeled
`_n_pairs_needed_doubled_INCORRECT` function kept only to document this). Separately
confirmed the 85,463 figure counts **episodes** (= 2 × n_memories × n_pairs), not
pairs — pairs alone would be 42,731.7 — ruling out a pairs-vs-episodes mislabeling
as an alternative explanation.

**Empirical Monte Carlo validation** (`power_formula_validation.py`, 20,000
replications per setting, exact bivariate-Bernoulli construction with the target
correlation — not this project's task-context-based simulators, which induce ρ only
indirectly): across 7 settings spanning p∈[0.30,0.80], ρ∈[0,0.8], |Δ|∈[0.03,0.30]
(including this project's real point and a one-sided "detect a harmful memory"
setting), **empirical power of a standard paired t-test at the formula's own n_pairs
landed in [79.7%, 82.3%]** — at or slightly above the nominal 80% target, never
meaningfully below it. The small, consistently safe-direction overshoot grows with ρ
(79.9% at ρ=0 → ~82% at ρ=0.8) and was traced to the genuine discreteness of
`D_i ∈ {-1,0,1}` (most pairs concordant, D_i=0, at high ρ) relative to the
continuous-normal assumption behind the formula — an exact noncentral-t calculation
(relaxing only the known-variance assumption, not the normality one) lands at
79.3–79.8%, confirming the formula's own algebra is sound and the residual gap is a
finite-sample/discreteness effect, not a derivation error.

**Resolution grid** (`resolution_grid.py`, `results/resolution_grid.json`, 700 rows
over Δ∈[0.01,0.20]×20, n_memories∈{10,20,30,40,50,75,100}, ρ∈{0,0.2,0.4,0.6,0.8}, at
the real measured p=0.6611 and real blended cost/episode $0.01133): confirms the
30-memory/Δ=0.03/ρ=0.6355 point costs $968.58 (matching §5.3 to the cent), and shows
**target precision (Δ) dominates cost far more than pairing strength (ρ) does** over
these ranges — cost scales as 1/Δ² (three orders of magnitude across the Δ grid)
but only linearly in `(1-ρ)` (at most 5× reduction from ρ=0 to ρ=0.8). See
`results/resolution_grid.png` for the heatmap (cost vs. Δ and ρ at N=30) and the
cost-vs-store-size curve (at the real Δ=0.03, ρ=0.6355 point), both with this
project's real point marked.

---

## 6. Costs and scale

### 6.1 Total real spend, by phase

| phase | spend (USD) | source |
|---|---|---|
| Model selection test (both attempts, 3 models) | $0.0628 | `alfworld_pilot/logs/model_selection_test_cost_state.json` |
| Memory sanity check | $0.1875 | `alfworld_pilot/results/memory_sanity_check.json` (`this_step_spend_usd`) |
| Small pilot | $2.0033 | `alfworld_pilot/results/small_pilot.json` (`this_step_spend_usd`) |
| Mini ground truth (4 memories × 133 pairs) | $12.6049 | narrative total in `alfworld_pilot/README.md`, consistent with cumulative shared-tracker delta |
| *(sanity + pilot + mini-GT share one tracker; its final cumulative state)* | **$14.7957** | `alfworld_pilot/logs/cost_tracker_state.json` (`total_spend_usd`) |
| Engineered-memory validation (redesign-b Step 1) | $1.1923 | `alfworld_pilot/logs/engineered_experiment_cost_state.json` |
| **TOTAL real spend, whole project** | **$16.0508** | sum of the 3 independent tracker files: $0.0628 + $14.7957 + $1.1923 |

(Note: `alfworld_pilot/README.md`'s own running narrative total after the mini
ground truth, "$14.8585," is the sum of model-selection + shared-tracker spend
*only*, computed **before** the later engineered-memory phase; adding that phase's
$1.1923 gives the true project-final total of $16.0508 used above.)

The full originally-scoped **10,030-episode production run was never executed** —
every real phase above was a smaller diagnostic/pilot/validation step; the "full
production run" cost projections in this document ($25.85–$97.19 depending on model
and step-cap-era measurement) are projections, never realized spend.

### 6.2 Total real episodes, by phase

| phase | episodes | notes |
|---|---|---|
| Model selection test | 10 | 5 gpt-5.6-luna + 5 ling-3.0-flash-vl (2 + 3 across two attempts) + 0 gemini |
| Memory sanity check | 30 | 15 pairs × 2 arms |
| Small pilot | 180 | single arm, weighted task sampling |
| Mini ground truth | 1,064 | 4 memories × 133 pairs × 2 arms |
| Engineered-memory validation | 80 | 2 memories × 20 pairs × 2 arms |
| **TOTAL real ALFWorld episodes with LLM calls** | **1,364** | |
| *(plus, zero LLM cost)* 2 real-ALFWorld episodes with `MockLLMClient` (Part B plumbing check) | 2 | zero API spend, real env only |

Separately, Stage 1's synthetic simulator "episodes" (10,000 per run × 20 seeds ×
several sweep configurations, plus a 200,000-episode oracle run) are **not** real
ALFWorld/LLM episodes — pure NumPy simulation, effectively free, not counted above.

### 6.3 Real measured tokens per call and calls per episode

| measurement | avg input tok/call | avg output tok/call | calls/episode | source | context |
|---|---|---|---|---|---|
| Mock env (Part B, pre-real-ALFWorld) | 602.9 | 14.7 | 4.6 | `alfworld_pilot/results/measure_mode_mock.json` | mock env, mock LLM, 5 episodes, cap not yet raised |
| Real ALFWorld, mock LLM, cap=30 | 1054.7 | 14.0 | 30.0 | superseded value, cited in `CLAUDE.md` | first real-env plumbing check |
| Real ALFWorld, mock LLM, cap=50 | 1235.53 | 14.13 | 50.0 | `alfworld_pilot/results/measure_mode_real.json` | 2 episodes, both hit the step cap (mock LLM never "succeeds" against real ALFWorld's actual admissible-actions ordering) |
| Real ALFWorld, real `gpt-5.6-luna`, model-selection test | 1594.25 | 33.30 | 25.6 | `alfworld_pilot/results/model_selection_test.json` | 5 episodes, 60% success (early termination on success lowers avg calls/episode below the 50-call worst case) |
| Real ALFWorld, real `gpt-5.6-luna`, memory sanity check | 1035.0 | 34.9 | 25.2 | `alfworld_pilot/results/memory_sanity_check.json` | 30 episodes, **pooled across with-memories and zero-memory-baseline arms** — not directly comparable to a production (all-with-memories) run |

Aggregate real-call totals across the whole project (sum of all three cost-tracker
state files): **45,077 real LLM calls**, **71,095,084 total input tokens**,
**1,564,670 total output tokens**.

### 6.4 Wall-clock runtime

- **Measured**: the mini ground truth phase (1,064 real episodes) took **~10 hours
  wall-clock**, run strictly sequentially by design (avoiding a race condition in the
  shared file-based cost-tracker state that concurrent processes would hit). —
  `alfworld_pilot/README.md` "Mini ground truth: results and verdict"
- **Estimated, not measured, for the full (never-run) 10,030-episode plan**
  (`alfworld_pilot/results/runtime_estimate.json`): local overhead (env
  construction/stepping, isolated from network latency via the mock LLM) **≈11.67
  hours**; LLM call latency **[ASSUMPTION, not measured before Part C]** 1.5s/call ×
  50 calls/episode worst case ≈ **208.96 hours**; **total serial estimate ≈220.63
  hours (≈9.19 days)**. The file's own note: "no real LLM calls have been made yet
  [for this specific estimate]... calls_per_episode_assumed=50 is a worst-case upper
  bound." No parallelized run was ever built or executed.

---

## 7. Threats to validity

Every item below is drawn from bugs/limitations the project's own logs and README
document explicitly, not inferred:

1. **Retrieval similarity is a mock heuristic, never a real embedding model**, for
   every real-agent result in this paper (§1.5). Similarity = topic-tag match (0.7)
   vs. mismatch (0.3) + uniform noise. This could affect external validity of any
   claim about "realistic retrieval" — the confound structure (task-type-correlated
   candidacy) is realistic, but the *mechanism* generating similarity scores is not a
   real retriever.

2. **Memory token lengths are narrower than configured** (§1.4A): configured
   100–300 tokens, actually realized 100–120 (mean 110.8) due to a padding-then-target
   quirk in `_pad_to_token_range` that stops padding as soon as the 100-token floor is
   crossed, rarely reaching anywhere near the 300-token ceiling. Not fixed, not
   flagged in the project's own README as a bug (this compilation is the first place
   it's called out) — **could understate realistic memory-length variation** and is
   worth independently confirming before citing "100–300 token memories" as fact.

3. **Determinism (temperature=0) was never empirically verified against the real
   client** (§1.1). `determinism_check.py` exists and was built for exactly this, but
   the project's own "still needed before Part C" list says it was never run before
   any real ground-truth phase. If gpt-5.6-luna's temperature=0 setting is not
   perfectly deterministic in practice, this adds unaccounted noise to every paired
   ground-truth comparison (§4.5, §4.6) beyond what the reported CIs capture.

4. **Double `env.reset()` per episode** (found and fixed before any real ALFWorld
   spend, during mock testing per commit `2e9600f`/README "Two real bugs..."): an
   earlier version reset the environment twice per episode — once to get `task_type`
   for retrieval, once again inside the agent's run loop, discarding the first
   reset's state. On `MockAlfredEnv` this silently played a *different* mock task
   than the one retrieval scored (harmless there since it's a symmetric mock); on
   real ALFWorld it would have silently consumed two real games per logged episode
   and mismatched retrieval against the actually-played task for every single real
   episode ever collected. **Caught in mock testing before any real spend** — so no
   reported real result is affected by it, but it is the reason mock-first testing is
   emphasized throughout the project's own documentation.

5. **Task-indexing sampling bias in the memory sanity check** (§4.2): `task_id %
   len(game_files)` over the raw filesystem-walk order concentrated the first 15
   sanity-check pairs 13/15 on the single easiest task type
   (`pick_and_place_simple`), 2/15 on the hardest, 0/15 on the other 4 types. The
   reported 93%/80% with/without-memories figures are real numbers but describe an
   84%-easy-type sample, not a representative one. **Detected and explicitly
   corrected for in the very next phase** (weighted task-type sampling, §4.3) — but
   any citation of the raw 93%/80% figures should carry this caveat.

6. **Ground-truth candidacy-probe seed mismatch** (caught during mock testing,
   before real ALFWorld existed, per `CLAUDE.md` "Bugs the mock testing caught"):
   `ground_truth_runner`'s natural-candidacy probe originally used a different random
   seed (`task_seed` alone) than the actual paired run
   (`hash((memory_id, pair_index, task_seed))`), so the probe checked a candidacy
   draw unrelated to what the real run would draw — could have silently produced
   ground-truth pairs whose target memory wasn't actually a candidate in the real
   run. Fixed with one shared, stable (`hashlib.md5`-based, not Python's
   process-randomized built-in `hash()`) seed-derivation function used by both the
   probe and the real run, verified end-to-end against real ALFWorld before any
   ground-truth spend (§ "Paired ground truth against real ALFWorld: SOLVED").

7. **`fast_downward` shared-library leak** (real, upstream, found during Part C):
   `textworld`/`alfworld`'s `fast_downward.interface.load_lib()` copies a ~32.5MB
   native library to a temp directory and `dlopen()`s it on every PDDL game load,
   never `dlclose()`s it — pages stay resident for the process's lifetime. This
   machine's `/tmp` is a 5.8GB RAM-backed tmpfs, so ~178 game loads exhausted it,
   crashing two runs with "No space left on device" before being diagnosed.
   Mitigated (not fixed upstream) two ways: pointing `TMPDIR` at the 950GB disk-backed
   filesystem instead of tmpfs, and `run_chunked.py`'s subprocess-per-chunk
   supervisor (restarts the process every N episodes/pairs, reclaiming leaked memory
   on each restart). **Could this have affected reported results?** No episode data
   is known to be corrupted by it — the failure mode is a hard process crash
   (caught, and checkpointing resumed cleanly from the last flushed line), not silent
   wrong output — but it constrains how large a single uninterrupted run could be,
   which is why every real phase in §4 ran in small, checkpointed chunks rather than
   one long process.

8. **A first version of the engineered-effect validation script under-ran its target
   pair count** (§4.6): called the ground-truth runner once per memory instead of
   looping to the target, silently stopping at 5/20 pairs. Caught from the printed
   `n_pairs` output before any interpretation was drawn from the incomplete data;
   fixed and resumed via checkpointing with zero wasted spend. No reported number in
   this document reflects the buggy partial run.

9. **The mini ground truth's null result may itself be partly a selection-bias
   artifact** (§4.5, §4.7): the 4 memories ground-truthed were chosen as "biggest
   MW-vs-IPS disagreement" from a 90-episode half-sample, and a synthetic check on
   the exact same selection procedure independently measured a ~1.87× disagreement
   inflation from selecting and evaluating on the same (vs. held-out) data. The
   project's split design (select on set A, evaluate on set B) already controls for
   the most direct form of this bias, but does not rule out that "biggest apparent
   disagreement in 90 noisy episodes" is a poor proxy for "biggest true effect,"
   independent of any same-data leakage.

10. **The engineered-memory null (§4.6) rests on n=20 pairs/memory, a small sample by
    the project's own power-analysis standards** (§3.11 suggests ≥57 pairs would be
    needed to reliably resolve a Δ=0.15 effect even under the more favorable real
    rho). The observed effects (+0.10, +0.00) are consistent with "the manipulation
    genuinely doesn't move success much because the agent self-corrects" (the
    project's own conclusion, supported by the transcript in §4.6.1), but are also
    consistent with "a real effect exists but n=20 pairs is underpowered to detect
    it" — the study was explicitly stopped at Step 1 specifically because it fell
    below the user's pre-set continuation threshold, not because n=20 was proven
    sufficient to rule out a larger true effect.

11. **No real ALFWorld ground truth exists for the small pilot's 30-memory ranking**
    (§4.4) beyond the 4 memories in §4.5 — the pairwise-Spearman pattern (MW
    negatively correlated with IPS, SNIPS/DR clustering tightly) is real and
    reproducible from the logged data, but there is no way to say which estimator's
    *ranking* is actually closer to the (unknown) true ranking of all 30 memories.

12. **Unpinned dependency versions** (§1.6): `requirements.txt`,
    `alfworld_pilot/requirements.txt`, and `pyproject.toml` all list dependencies
    without version pins. The exact versions captured in §1.6 are a snapshot of what
    happened to be installed on 2026-09-26, not a lockfile guarantee — a fresh
    `pip install` today could resolve different (likely newer) versions of numpy,
    scikit-learn, alfworld, etc.

13. **Python 3.13/3.14 incompatibility with `textworld`'s PDDL grammar engine** (real,
    upstream, documented and worked around): `textworld` 1.7.0 relies on a
    `locals().update(...); eval(...)` pattern that Python 3.13's PEP 667 permanently
    broke (`NameError` on the first `env.reset()`). Originally monkeypatched, then
    resolved properly by rebuilding the venv on Python 3.11 (which ALFWorld's own
    docs actually target) and deleting the patch. Not a threat to any *reported*
    result (no result was produced under the patched 3.14 environment), but worth
    noting for anyone attempting to reproduce this work on a newer Python.

14. **The project's real remaining budget, not statistical necessity, determined
    where the real-agent investigation stopped.** The original 10,030-episode
    production plan was abandoned as unaffordable after the mini ground truth (§4.5)
    consumed most of the available OpenRouter balance; the redesign-(b) pivot (§4.6)
    operated under an explicit $15 sub-budget. The paper's conclusions about "how
    hard it is to find real per-memory effects" are drawn from what a roughly $16
    total real-money budget could observe — a materially larger budget was never
    tested and the noise-vs-signal calculation (§5.3) is explicit that resolving the
    natural store's effects properly would cost ~$970, an order of magnitude more
    than was ever spent.

---

## 8. Reproducibility

**Repository**: `/home/ibo/memory-pilot` (WSL2 Ubuntu), remote `origin` =
`https://github.com/ibo-agentic/memory-pilot.git`. HEAD as of the §5.4 addition:
`a22d47c` (HEAD at the time the rest of this document was originally compiled:
`0bb21e6`).

### 8.1 Commit-to-result mapping (chronological, oldest first)

| commit | date | produced |
|---|---|---|
| `0fdf467` | 2026-09-17 | Stage 1 core: `results/stage1_report.json`, `episode_sweep.json`, `large_n_sanity.json`, `randomness_sweep.json`, `dr_cross_fit_check.json`, `true_value_check.json`; all of `src/memory_ope/` (simulators, estimators, evaluation); `tests/` |
| `70f25b6` | 2026-09-17 | `results/ground_truth_power_analysis.json`, `randomization_ablation.json`, `small_data_results.json`, `stage2_budget_estimate.json`, `stage2_scale_check.json` |
| `023a57d` | 2026-09-17 | `results/ground_truth_selection_bias_check.json`, `signal_boost_check.json` |
| `bfe6cdc` | 2026-09-17 | `alfworld_pilot/` infrastructure (mock-only), `alfworld_pilot/results/measure_mode_mock.json`, `token_breakdown.json` |
| `2e9600f` | 2026-09-18 | Real ALFWorld installed under WSL2; `memory_ope` made pip-installable; `alfworld_pilot/results/measure_mode_real.json` (first real-env version, cap=30) |
| `6ff8d11` | 2026-09-18 | Paired ground truth solved for real ALFWorld; `alfworld_pilot/results/task_type_difficulty_check.json`; `results/ground_truth_power_analysis_rescale.json`, `scale_up_check.json`; updated `measure_mode_real.json` |
| `fc65646` | 2026-09-19 | Step-cap fix (30→50), resource-leak mitigation, checkpointing; `alfworld_pilot/results/step_cap_check.json`, `model_selection_test.json`, `runtime_estimate.json`; updated `measure_mode_real.json` (cap=50) — **model selection real spend $0.063** |
| `f7478d7` | 2026-09-19 | Model picked (gpt-5.6-luna); `alfworld_pilot/results/memory_sanity_check.json` — **real spend $0.19** |
| `69c9453` | 2026-09-19 | Ceiling-effect diagnosis; `alfworld_pilot/results/memory_sanity_check_breakdown.json`; weighted task-type sampling built |
| `5cc2991` | 2026-09-19 | `alfworld_pilot/results/small_pilot.json` — **real spend $2.00** |
| `2a02b3d` | 2026-09-20 | `alfworld_pilot/results/mini_ground_truth_selection.json`, `mini_ground_truth_analysis.json` — **real spend $12.60** |
| `c3bfed4` | 2026-09-20 | `alfworld_pilot/results/effect_size_analysis.json` (zero new spend) |
| `0bb21e6` | 2026-09-20 | `alfworld_pilot/results/engineered_effect_validation.json`, `engineered_memory_store.py` — **real spend $1.19** |
| `a22d47c` (HEAD) | 2026-09-26 | `src/memory_ope/evaluation/power_formula.py`, `power_formula_validation.py`, `resolution_grid.py`, `tests/test_power_formula.py`, `results/power_formula_validation.json`, `resolution_grid.json`, `power_formula_derivation.tex` — §5.4, zero API spend |

(Note: some narrative dates in `CLAUDE.md`/`alfworld_pilot/README.md` say "2026-09-21"
for the final round; the commit itself is dated 2026-09-20 — likely a same-session
late-night/next-day discrepancy, not a data error. Use the commit hash, not the prose
date, as the authoritative reference.)

### 8.2 Exact commands to reproduce each result

**Stage 1** (from repo root, `source .venv/bin/activate` first; ~seconds, zero cost):
```
python -m memory_ope.simulator.run_simulation        # -> logs/*.jsonl (gitignored, regenerate locally)
python -m memory_ope.evaluation.run_eval              # -> results/stage1_report.json, bias_by_estimator.png, hitchhiker_recoverability.png
pytest tests/                                          # includes tests/test_ips_unbiased.py
python -m memory_ope.evaluation.true_value_check
python -m memory_ope.evaluation.large_n_sanity
python -m memory_ope.evaluation.episode_sweep
python -m memory_ope.evaluation.randomness_sweep
python -m memory_ope.evaluation.dr_cross_fit_check
python -m memory_ope.evaluation.randomization_ablation
python -m memory_ope.evaluation.small_data_results
python -m memory_ope.evaluation.stage2_budget_estimate
python -m memory_ope.evaluation.stage2_scale_check
python -m memory_ope.evaluation.ground_truth_power_analysis
python -m memory_ope.evaluation.signal_boost_check
python -m memory_ope.evaluation.ground_truth_selection_bias_check
python -m memory_ope.evaluation.scale_up_check                        # (root-package script, commit 6ff8d11)
python -m memory_ope.evaluation.ground_truth_power_analysis_rescale   # (commit 6ff8d11)
pytest tests/test_power_formula.py -v                                 # verifies power_formula.py against effect_size_analysis.json and ground_truth_power_analysis.json
python -m memory_ope.evaluation.power_formula_validation               # -> results/power_formula_validation.json (Monte Carlo, ~3s)
python -m memory_ope.evaluation.resolution_grid                        # -> results/resolution_grid.json, resolution_grid.png (see §5.4)
```

**Stage 2, real-agent phases — COST WARNING, all of these spend real OpenRouter
credit against `openai/gpt-5.6-luna` at whatever `cost_control.hard_cap_usd` is
currently set to in `alfworld_pilot/config.yaml`. Do not run these without
understanding the current cap.** Requires `OPENROUTER_API_KEY` in repo-root `.env`,
`TMPDIR` pointed at disk-backed storage (not tmpfs — see §7 item 7), and
`alfworld-download` already run once.
```
cd alfworld_pilot
export TMPDIR=/home/ibo/tmp_downward   # or equivalent disk-backed dir
PYTHONPATH=src python -m pytest tests/ -v                              # 18 tests, mock-only, zero cost
PYTHONPATH=src python -m alfworld_pilot.measure_mode                    # real env, mock LLM, zero API cost
PYTHONPATH=src python -m alfworld_pilot.token_breakdown                 # mock env by design
PYTHONPATH=src python -m alfworld_pilot.task_type_difficulty_check      # real data, zero LLM cost
PYTHONPATH=src python -m alfworld_pilot.step_cap_check                  # zero LLM cost
PYTHONPATH=src python -m alfworld_pilot.model_selection_test            # REAL SPEND (~$0.06 last run)
PYTHONPATH=src python -m alfworld_pilot.memory_sanity_check             # REAL SPEND (~$0.19 last run)
PYTHONPATH=src python -m alfworld_pilot.memory_sanity_check_breakdown   # zero new spend, reads existing data
PYTHONPATH=src python -m alfworld_pilot.small_pilot                     # REAL SPEND (~$2.00 last run)
PYTHONPATH=src python -m alfworld_pilot.mini_ground_truth_selection     # zero new spend
PYTHONPATH=src python -m alfworld_pilot.run_chunked ...                 # REAL SPEND — mini ground truth (~$12.60 last run); see run_chunked.py --help for per-memory chunking args
PYTHONPATH=src python -m alfworld_pilot.effect_size_analysis            # zero new spend, reads existing data
PYTHONPATH=src python -m alfworld_pilot.engineered_effect_validation    # REAL SPEND (~$1.19 last run)
PYTHONPATH=src python -m alfworld_pilot.runtime_estimate                # zero cost (local-only timing)
```

Every real-spend script is checkpointed (`checkpointed_runner.py`) and reads/writes a
specific `state_path` cost-tracker JSON — **check which tracker file and
`hard_cap_usd` a script uses before re-running anything**, since several phases
deliberately used separate dedicated tracker files specifically to avoid an
already-elevated cumulative spend refusing new calls (§4.6).

### 8.3 What is NOT reproducible from this repo alone

- The exact real LLM completions are cached (`alfworld_pilot/cache/`, gitignored) —
  re-running any real-spend script from a clean cache will make **new** real API
  calls (new spend) and may get different completions even at temperature=0, since
  determinism against the live API was never independently verified (§7 item 3).
- OpenRouter model pricing and availability (fetched 2026-09-19) — re-check before
  reusing any cost projection.
- The specific 3,553-game-file `train` split's exact contents depend on the
  `alfworld-download` artifact version fetched (~2.3GB, not itself version-pinned in
  this repo beyond "alfworld==0.4.2's own download script").
