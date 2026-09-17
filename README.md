# memory-ope-pilot

Testing whether off-policy evaluation (IPS / SNIPS / doubly robust) can
recover the causal value of individual LLM-agent memories better than the
"Memory Worth" baseline (Şimşek 2026, arXiv 2604.12007). See `CLAUDE.md` for
the full research context (value definition, retrieval design, estimators).

## Setup

```bash
# from this directory
.venv\Scripts\python.exe -m pip install -r requirements.txt   # PowerShell
# or: source .venv/Scripts/activate && pip install -r requirements.txt
```

The venv here targets Python 3.14; all dependencies (numpy, scipy,
scikit-learn, matplotlib, pyyaml, pytest) installed cleanly on it as of this
writing.

## Stage 1 — synthetic simulator (no LLM, no cost)

Reproduces the paper's two confounding setups (task-difficulty confounding
and hitchhiker co-retrieval) with known ground truth, and compares all four
estimators against it over 20 seeds.

```bash
# 1. generate episode logs for both simulators, all seeds -> logs/*.jsonl
python -m memory_ope.simulator.run_simulation

# 2. run all estimators, compare to the oracle true values,
#    write results/stage1_report.json + plots
python -m memory_ope.evaluation.run_eval

# 3. tests, including the explicit IPS-unbiasedness check
pytest tests/
```

All settings (simulator sizes, propensity bounds, seed count, etc.) live in
`config/config.yaml`. Run scripts as modules (`python -m memory_ope....`)
from this directory, or set `PYTHONPATH=src` first — pytest picks up `src/`
automatically via `tests/conftest.py`.

Outputs:
- `results/stage1_report.json` — full per-seed correlation/bias/variance
  numbers for every estimator on both simulators, plus the hitchhiker
  independent-retrieval-fraction sweep (scale-free metrics).
- `results/bias_by_estimator.png`, `results/hitchhiker_recoverability.png`.

See `CLAUDE.md` for a summary of what the last run found.

### Design-review checks

A deeper review pass beyond the main report — run any of these independently:

```bash
python -m memory_ope.evaluation.true_value_check         # causal oracle vs naive observational vs U*
python -m memory_ope.evaluation.large_n_sanity            # 200k-episode sanity check
python -m memory_ope.evaluation.episode_sweep             # spearman vs n_episodes, both simulators
python -m memory_ope.evaluation.randomness_sweep          # propensity extremity vs accuracy + ESS
python -m memory_ope.evaluation.dr_cross_fit_check        # doubly robust with vs without cross-fitting
python -m memory_ope.evaluation.randomization_ablation    # randomization alone vs the propensity correction
python -m memory_ope.evaluation.small_data_results        # n=250..2000, spearman + MAE + bias, task_difficulty
python -m memory_ope.evaluation.stage2_budget_estimate     # ALFWorld LLM-call/cost estimate (no API calls)
python -m memory_ope.evaluation.stage2_scale_check         # task_difficulty at Stage-2 scale (50 memories, n=1000)
python -m memory_ope.evaluation.ground_truth_power_analysis  # paired forced-in/forced-out rerun sample sizes
```

Each writes its own `results/*.json` (and a `.png` where relevant). See
`CLAUDE.md`'s "Design review" section for what the last run of each found.

## Stage 2 — ALFWorld pilot (not yet implemented)

Gated on approval of the Stage 1 results above. When it starts:
install ALFWorld from the **official repo** — https://github.com/alfworld/alfworld
— following its own instructions (Python ≥3.9 venv, `pip install -e .`,
`alfworld-download` for game/data files). Do not follow install steps from
anywhere else.

Planned shape (see `alfworld_pilot/README.md`):
- A ReAct-style agent over a ~50-trajectory memory store, using the same
  randomized top-M / known-propensity retrieval as Stage 1.
- Forced-in/forced-out ground truth for 10 chosen memories, several seeds
  each.
- All LLM calls cached under `cache/` (never re-billed on rerun) and gated
  by a cost-limit + dry-run mode in `config/config.yaml`'s `stage2` section
  that estimates call counts before spending anything.
- API keys via `.env` (copy `.env.example`), never in code.

## Project layout

```
config/config.yaml          all settings for both stages
src/memory_ope/
  retrieval.py               top-M selection, propensities, Bernoulli inclusion
  logging_utils.py           JSONL read/write
  simulator/                 Stage 1 synthetic DGPs + Monte Carlo oracle
  estimators/                memory_worth, ips/snips, doubly_robust
  evaluation/                metrics + report/plot generation
alfworld_pilot/              Stage 2 skeleton (not implemented)
tests/
logs/, cache/, results/      gitignored except results/*.json
```
