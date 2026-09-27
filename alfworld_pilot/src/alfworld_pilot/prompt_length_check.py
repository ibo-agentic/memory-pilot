"""Prompt-length confound check: is the ~0.65 correlation between
avg_prompt_tokens and parse_failures (zero-shot + memories,
pick_and_place_simple, capability_check.py) actually driven by MEMORY
CONTENT, or just by episode length (more steps -> longer growing history
-> more tokens, regardless of memories)?

LOG FORMAT, read directly off results_kaggle/capability_check_pick_and_place_simple_zeroshot_withmem.json
before writing any of the checks below -- exactly what's saved per episode:
    task_id, task_type, success, steps_taken, hit_step_cap, parse_failures,
    n_included (COUNT of memories included, not which ones), avg_input_tokens
    (a single float -- the MEAN input_tokens across that episode's LLM calls,
    not saved per-step), fallback_classes, failure_mode,
    actions (the full ordered list of action strings actually taken, one per step).

NOT present, and why this script reconstructs rather than reads them:
  - Per-step input_tokens (only the episode-level mean survived to the JSON).
  - Which specific memory ids were included (only the count).
  - Per-step observations / admissible-actions / which steps were parse
    failures (only the episode total count and an unordered list of
    fallback-harm classifications survived).

Reconstruction method (exact, not approximated, zero new LLM calls, zero
GPU, zero Kaggle involvement, zero change to the prompt or sampling design):
  1. Retrieval (embedding_similarity_scores + memory_ope.retrieval.retrieve)
     is a pure function of (memories, the real per-episode goal text,
     a task_id-seeded RNG) -- replaying it with the same task_id reproduces
     the exact memories capability_check.py actually drew. Validated per
     episode: replayed len(included_texts) must equal the saved n_included.
  2. Real ALFWorld's env.step() is deterministic given the same game file
     and the same action string. Replaying env.reset(task_seed) then
     env.step(action) for every action in the episode's SAVED action
     sequence reproduces the exact observations/admissible-actions lists
     that were actually shown to the model at each step.
  3. react_agent._build_prompt (imported directly, unchanged) reconstructs
     the exact prompt (messages list) at every step from that replayed
     history. capability_check.py's LLMCache (cache_kaggle/, disk-persisted,
     keyed by a hash of the full request payload) still holds the ACTUAL
     recorded response for that exact request -- so cache.get(payload) gives
     the real response text and the real input_tokens for that exact step,
     with no tokenizer or model load needed for that part. Validated per
     step: react_agent._parse_action(cached_text, admissible) must return
     the same action string as the episode's saved actions[step_index], and
     the total reconstructed parse-failure count must equal the saved
     parse_failures for that episode.
  4. A tokenizer (AutoTokenizer.from_pretrained only -- not the 15GB model
     weights, no GPU) is loaded once to decompose each step's prompt into
     its memory-block and history-block token counts for the regression in
     check 2; the *total* prompt token count used in check 1 comes directly
     from the cache (step 3), not from this tokenizer.

Scope: this script analyzes ONLY
results_kaggle/capability_check_pick_and_place_simple_zeroshot_withmem.json
(20 episodes, one task type). The other zero-shot+memories file
(capability_check_all_types_zeroshot_withmem.json) has pick_and_place_simple
task_id 0-9 that are the SAME underlying episodes (same seeds, verified
identical results when it was produced) plus 5 other task types -- mixing
task types into one memory-vs-parse-failure comparison would confound
"memory presence" with "task type" (a task-type-tagged memory only ever
co-occurs with its own task type's episodes), which is exactly the kind of
confound this check exists to rule out, not reintroduce. So this analysis is
scoped to the single 20-episode run where the 0.65 correlation was actually
observed.

Usage (from alfworld_pilot/, using the .venv-kaggle venv):
    PYTHONPATH=src .venv-kaggle/bin/python -m alfworld_pilot.prompt_length_check
"""

from __future__ import annotations

import json
import pathlib
import random
import warnings

import numpy as np
from scipy import stats

from . import config as config_mod
from . import react_agent
from .cache import LLMCache
from .capability_check import _single_type_source
from .embedding_retrieval import SentenceEmbedder, embedding_similarity_scores
from .env_factory import load_real_alfworld_config
from .kaggle_memory_store import CATEGORY_BY_MEMORY_ID, build_kaggle_store

ALFWORLD_PILOT_DIR = pathlib.Path(__file__).resolve().parents[2]
RESULTS_PATH = ALFWORLD_PILOT_DIR / "results_kaggle" / "capability_check_pick_and_place_simple_zeroshot_withmem.json"


def _admissible_from_info(info: dict) -> list[str]:
    cmds = info["admissible_commands"]
    if isinstance(cmds, list) and cmds and isinstance(cmds[0], list):
        return cmds[0]
    return cmds


def load_saved_episodes() -> list[dict]:
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["results"]["pick_and_place_simple"]["episodes"]


def replay_episode(
    ep: dict,
    source,
    memories: list,
    embedder: SentenceEmbedder,
    m: int,
    prop_min: float,
    prop_max: float,
    cache: LLMCache,
    model_id: str,
    revision: str,
    temperature: float,
    max_output_tokens: int,
) -> dict:
    """Reconstructs, for one saved episode, the exact sequence of prompts and
    per-step parse-failure/input-token values via the deterministic replay
    described in the module docstring. Raises RuntimeError (caller decides
    how to report it) if any validation check fails, rather than silently
    returning a value that doesn't actually match what was recorded."""
    task_id = ep["task_id"]
    env = source.build_env(task_id)
    try:
        obs, info = env.reset(task_seed=task_id)
        rng = random.Random(task_id)
        sims = embedding_similarity_scores(memories, obs, embedder)
        from memory_ope.retrieval import retrieve as ope_retrieve

        candidate_ids, _props, included = ope_retrieve(sims, m, prop_min, prop_max, rng)
        mem_by_id = {mem.mem_id: mem for mem in memories}
        included_ids = [mid for mid in candidate_ids if included[mid] == 1]
        included_texts = [mem_by_id[mid].text for mid in included_ids]
        if len(included_texts) != ep["n_included"]:
            raise RuntimeError(
                f"task_id={task_id}: replayed n_included={len(included_texts)} != saved n_included={ep['n_included']}"
            )

        admissible = _admissible_from_info(info)
        history: list[react_agent.StepRecord] = []
        steps_out = []
        for step_idx, saved_action in enumerate(ep["actions"]):
            messages = react_agent._build_prompt(obs, included_texts, history, admissible, few_shot_text=None)
            payload = {
                "model": model_id,
                "revision": revision,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "stop": ["\n\n"],
            }
            cached = cache.get(payload)
            if cached is None:
                raise RuntimeError(f"task_id={task_id} step={step_idx}: cache miss -- cannot reconstruct exactly")
            resp = cached["response"]
            thought, action, was_admissible = react_agent._parse_action(resp["text"], admissible)
            if action != saved_action:
                raise RuntimeError(
                    f"task_id={task_id} step={step_idx}: replayed action {action!r} != saved action {saved_action!r}"
                )

            memory_block = (
                "Relevant lessons from past episodes:\n" + "\n".join(f"- {t}" for t in included_texts) + "\n\n"
            ) if included_texts else ""
            history_block = "".join(f"{s.observation}\n> {s.action}\n" for s in history)

            steps_out.append(
                {
                    "step_index": step_idx,
                    "input_tokens": resp["input_tokens"],
                    "parse_fail": int(not was_admissible),
                    "memory_block_text": memory_block,
                    "history_block_text": history_block,
                }
            )

            step_record = react_agent.StepRecord(
                observation=obs,
                admissible_actions=list(admissible),
                thought=thought,
                action=action,
                action_was_admissible=was_admissible,
                input_tokens=resp["input_tokens"],
                output_tokens=resp["output_tokens"],
                cached=True,
            )
            history.append(step_record)

            obs, _reward, done, info = env.step(action)
            admissible = _admissible_from_info(info)

        total_pf = sum(s["parse_fail"] for s in steps_out)
        if total_pf != ep["parse_failures"]:
            raise RuntimeError(
                f"task_id={task_id}: replayed total parse_failures={total_pf} != saved parse_failures={ep['parse_failures']}"
            )
    finally:
        env.close()

    return {"task_id": task_id, "included_ids": included_ids, "steps": steps_out}


def check1_step1_prompt_length(saved_episodes: list[dict], replayed: dict[int, dict]) -> dict:
    step1_tokens = []
    pf_rates = []
    for ep in saved_episodes:
        r = replayed[ep["task_id"]]
        step1_tokens.append(r["steps"][0]["input_tokens"])
        pf_rates.append(ep["parse_failures"] / ep["steps_taken"])
    rho, pval = stats.spearmanr(step1_tokens, pf_rates)
    return {
        "n_episodes": len(saved_episodes),
        "step1_prompt_tokens": step1_tokens,
        "parse_failure_rates": pf_rates,
        "spearman_r": float(rho),
        "spearman_p": float(pval),
    }


def check2_step_level_regression(replayed: dict[int, dict], tokenizer) -> dict:
    import statsmodels.api as sm

    rows = []
    for task_id, r in replayed.items():
        for s in r["steps"]:
            memory_tokens = len(tokenizer(s["memory_block_text"], add_special_tokens=False)["input_ids"])
            history_tokens = len(tokenizer(s["history_block_text"], add_special_tokens=False)["input_ids"])
            rows.append(
                {
                    "task_id": task_id,
                    "parse_fail": s["parse_fail"],
                    "memory_tokens": memory_tokens,
                    "step_index": s["step_index"],
                    "history_tokens": history_tokens,
                }
            )

    y = np.array([r["parse_fail"] for r in rows], dtype=float)
    X = np.array([[r["memory_tokens"], r["step_index"], r["history_tokens"]] for r in rows], dtype=float)
    X = sm.add_constant(X)
    n_rows = len(rows)
    n_events = int(y.sum())

    result_summary: dict = {
        "n_rows": n_rows,
        "n_parse_failures": n_events,
        "n_episodes": len(replayed),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = sm.Logit(y, X)
            fit = model.fit(disp=0, maxiter=200)
        except Exception as e:  # noqa: BLE001
            result_summary["fit_failed"] = True
            result_summary["fit_error"] = str(e)
            result_summary["rows"] = rows
            return result_summary

        separation_warned = any("perfect separation" in str(w.message).lower() for w in caught)

    names = ["const", "memory_tokens", "step_index", "history_tokens"]
    conf_int = fit.conf_int(alpha=0.05)
    large_se = bool(np.any(fit.bse > 20))

    result_summary.update(
        {
            "converged": bool(fit.mle_retvals.get("converged", False)),
            "separation_warning": separation_warned,
            "large_std_errors_flag": large_se,
            "coefficients": {
                name: {
                    "coef": float(fit.params[i]),
                    "std_err": float(fit.bse[i]),
                    "p_value": float(fit.pvalues[i]),
                    "ci_low": float(conf_int[i][0]),
                    "ci_high": float(conf_int[i][1]),
                }
                for i, name in enumerate(names)
            },
            "unstable_fit": separation_warned or large_se or not fit.mle_retvals.get("converged", False),
        }
    )
    return result_summary


def check3_episode_length(saved_episodes: list[dict]) -> dict:
    steps_taken = [e["steps_taken"] for e in saved_episodes]
    pf_rates = [e["parse_failures"] / e["steps_taken"] for e in saved_episodes]
    rho, pval = stats.spearmanr(steps_taken, pf_rates)

    success_rates = [(e["success"], e["parse_failures"] / e["steps_taken"]) for e in saved_episodes]
    succ = [r for s, r in success_rates if s]
    fail = [r for s, r in success_rates if not s]
    return {
        "spearman_r_steps_vs_pf_rate": float(rho),
        "spearman_p_steps_vs_pf_rate": float(pval),
        "n_successful": len(succ),
        "n_failed": len(fail),
        "mean_pf_rate_successful": float(np.mean(succ)) if succ else None,
        "mean_pf_rate_failed": float(np.mean(fail)) if fail else None,
    }


def check4_per_memory(saved_episodes: list[dict], replayed: dict[int, dict], memories: list, tokenizer) -> dict:
    mem_by_id = {mem.mem_id: mem for mem in memories}
    pf_rate_by_task_id = {e["task_id"]: e["parse_failures"] / e["steps_taken"] for e in saved_episodes}

    out = {}
    for mem in memories:
        with_ids = [tid for tid, r in replayed.items() if mem.mem_id in r["included_ids"]]
        without_ids = [tid for tid in replayed if tid not in with_ids]
        with_rates = [pf_rate_by_task_id[tid] for tid in with_ids]
        without_rates = [pf_rate_by_task_id[tid] for tid in without_ids]
        token_len = len(tokenizer(mem.text, add_special_tokens=False)["input_ids"])
        out[mem.mem_id] = {
            "task_type": mem.task_type,
            "category": CATEGORY_BY_MEMORY_ID.get(mem.mem_id),
            "token_length": token_len,
            "n_episodes_with": len(with_ids),
            "n_episodes_without": len(without_ids),
            "mean_pf_rate_with": float(np.mean(with_rates)) if with_rates else None,
            "mean_pf_rate_without": float(np.mean(without_rates)) if without_rates else None,
        }
    return out


def main() -> None:
    cfg_path = ALFWORLD_PILOT_DIR / "kaggle_config.yaml"
    cfg = config_mod.load_config(cfg_path)

    saved_episodes = load_saved_episodes()
    print(f"Loaded {len(saved_episodes)} saved episodes from {RESULTS_PATH}")
    print("Fields present per episode:", list(saved_episodes[0].keys()))

    real_cfg = load_real_alfworld_config(cfg["env"]["real_alfworld_config_path"])
    split = cfg["env"]["real_split"]
    source = _single_type_source(real_cfg, split, "pick_and_place_simple")

    memories = build_kaggle_store()
    embedder = SentenceEmbedder(
        model_name=cfg["embedding"]["model_name"], revision=cfg["embedding"].get("revision"), device="cpu"
    )
    cache = LLMCache(cfg["cache"]["dir"])

    llm_cfg = cfg["llm"]
    model_id = llm_cfg["model_id"]
    revision = llm_cfg["revision"]
    temperature = llm_cfg["temperature"]
    max_output_tokens = llm_cfg["max_output_tokens"]
    m = cfg["retrieval"]["M"]
    prop_min = cfg["retrieval"]["propensity_min"]
    prop_max = cfg["retrieval"]["propensity_max"]

    print("Replaying retrieval + environment steps to reconstruct per-step prompts (no GPU, no new LLM calls, exact cache lookups)...")
    replayed: dict[int, dict] = {}
    replay_failures = []
    for ep in saved_episodes:
        try:
            replayed[ep["task_id"]] = replay_episode(
                ep, source, memories, embedder, m, prop_min, prop_max, cache, model_id, revision, temperature, max_output_tokens
            )
        except RuntimeError as e:
            replay_failures.append(str(e))

    if replay_failures:
        print(f"\nWARNING: {len(replay_failures)}/{len(saved_episodes)} episodes failed exact replay validation:")
        for msg in replay_failures:
            print(f"  - {msg}")
        print("Proceeding only with the episodes that validated exactly.")
        saved_episodes = [e for e in saved_episodes if e["task_id"] in replayed]
    else:
        print(f"All {len(saved_episodes)} episodes replayed with exact validation (n_included, per-step actions, and total parse_failures all matched the saved log).")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    print("\n=== Check 1: step-1 prompt length vs episode parse-failure rate ===")
    c1 = check1_step1_prompt_length(saved_episodes, replayed)
    print(f"  n={c1['n_episodes']}, Spearman r={c1['spearman_r']:.4f}, p={c1['spearman_p']:.4f}")

    print("\n=== Check 2: step-level logistic regression (parse_fail ~ memory_tokens + step_index + history_tokens) ===")
    c2 = check2_step_level_regression(replayed, tokenizer)
    if c2.get("fit_failed") or c2.get("unstable_fit"):
        print(f"  n_rows={c2['n_rows']}, n_parse_failures={c2['n_parse_failures']} -- fit is UNSTABLE or FAILED, reporting plainly rather than forcing conclusions:")
        if c2.get("fit_failed"):
            print(f"    fit error: {c2['fit_error']}")
        else:
            print(f"    converged={c2['converged']}, separation_warning={c2['separation_warning']}, large_std_errors={c2['large_std_errors_flag']}")
            for name, c in c2["coefficients"].items():
                print(f"    {name}: coef={c['coef']:.5f}, se={c['std_err']:.3f}, p={c['p_value']:.3f}, CI=[{c['ci_low']:.5f}, {c['ci_high']:.5f}]")
    else:
        print(f"  n_rows={c2['n_rows']}, n_parse_failures={c2['n_parse_failures']}, converged={c2['converged']}")
        for name, c in c2["coefficients"].items():
            print(f"    {name}: coef={c['coef']:.5f}, se={c['std_err']:.3f}, p={c['p_value']:.3f}, CI=[{c['ci_low']:.5f}, {c['ci_high']:.5f}]")

    print("\n=== Check 3: episode length vs parse-failure rate ===")
    c3 = check3_episode_length(saved_episodes)
    print(f"  Spearman r(steps_taken, pf_rate)={c3['spearman_r_steps_vs_pf_rate']:.4f}, p={c3['spearman_p_steps_vs_pf_rate']:.4f}")
    print(f"  mean pf_rate: successful (n={c3['n_successful']})={c3['mean_pf_rate_successful']:.4f}, failed (n={c3['n_failed']})={c3['mean_pf_rate_failed']:.4f}")

    print("\n=== Check 4: per-memory parse-failure rate with vs without ===")
    c4 = check4_per_memory(saved_episodes, replayed, memories, tokenizer)
    flagged = []
    for mem_id, d in sorted(c4.items(), key=lambda kv: -(kv[1]["mean_pf_rate_with"] or 0)):
        w = d["mean_pf_rate_with"]
        wo = d["mean_pf_rate_without"]
        flag = ""
        if w is not None and wo is not None and w > wo * 1.5 and d["n_episodes_with"] >= 2:
            flag = "  <-- FLAG (with > 1.5x without)"
            flagged.append(mem_id)
        print(
            f"  {mem_id:16s} ({d['category']:10s}, {d['token_length']:3d} tok, n_with={d['n_episodes_with']:2d}, n_without={d['n_episodes_without']:2d}): "
            f"pf_rate_with={w if w is not None else float('nan'):.3f}, pf_rate_without={wo if wo is not None else float('nan'):.3f}{flag}"
        )
    print("  (n per memory is small -- treat this as a flag, not a result.)")

    step1_pass = abs(c1["spearman_r"]) < 0.15
    ci_includes_zero = False
    if not (c2.get("fit_failed") or c2.get("unstable_fit")):
        mt = c2["coefficients"]["memory_tokens"]
        ci_includes_zero = mt["ci_low"] <= 0.0 <= mt["ci_high"]
    verdict = "PASS" if (step1_pass and ci_includes_zero) else "FAIL"

    print(f"\n=== VERDICT: {verdict} ===")
    print(f"  step1 |r| < 0.15: {step1_pass} (r={c1['spearman_r']:.4f})")
    if c2.get("fit_failed") or c2.get("unstable_fit"):
        print("  check 2 fit unstable/failed -- CI-includes-zero condition NOT evaluable; counted as not satisfied for the pass rule")
    else:
        print(f"  memory_tokens CI includes zero: {ci_includes_zero} (CI=[{c2['coefficients']['memory_tokens']['ci_low']:.5f}, {c2['coefficients']['memory_tokens']['ci_high']:.5f}])")
    if verdict == "FAIL":
        print(f"  memories flagged in check 4: {flagged if flagged else '(none individually flagged; failure is driven by step1/regression signal, not a single memory)'}")

    out = {
        "check1_step1_prompt_length": c1,
        "check2_step_level_regression": c2,
        "check3_episode_length": c3,
        "check4_per_memory": c4,
        "replay_failures": replay_failures,
        "verdict": verdict,
        "flagged_memories": flagged,
    }
    out_path = ALFWORLD_PILOT_DIR / "results_kaggle" / "prompt_length_check.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
