"""Two checks requested before Part C spend:

1. Confirm real ALFWorld's 6 official task types are all present in the
   dataset and correctly parsed by RealAlfredEnv.task_type_from_gamefile
   (episode_runner.py already logs `task_type` per episode -- this verifies
   the parsing that field depends on, against the REAL data, not just the
   mock).
2. A ZERO-COST difficulty proxy per task type: play a sample of games using
   ALFWorld's OWN built-in scripted expert (`info['extra.expert_plan']`,
   surfaced automatically by AlfredExpert whenever `general.training_method:
   dagger` and split="train" -- no LLM calls at all) and record steps-to-
   solve. This is the real-ALFWorld analogue of the task_difficulty
   simulator's confound (some tasks are structurally easier/harder,
   correlating with success independent of any memory's causal value) --
   exactly what IPS/SNIPS/DR are meant to correct for and Memory Worth is
   not. In particular it reports what fraction of each task type an
   OPTIMAL scripted policy can even finish within alfworld_pilot's own
   `env.max_steps` cap -- if that fraction is low for some task type, ANY
   agent (regardless of which memories it got) will show a depressed
   success rate on that task type purely from the step budget, which is a
   confound this pilot's OPE method needs to see through, not something a
   better agent could fix.

Usage:
    python -m alfworld_pilot.task_type_difficulty_check
"""

from __future__ import annotations

import collections
import json
import pathlib
import random

from . import config as config_mod
from .env_factory import load_real_alfworld_config
from .env_interface import TASK_TYPES, RealAlfredEnv, list_real_game_files

N_SAMPLES_PER_TYPE = 30
EXPERT_MAX_STEPS = 100  # generous ceiling so we measure true solve length, not just cap-truncated
SEED = 42


def check_task_type_coverage(real_cfg: dict, split: str = "train") -> dict:
    game_files = list_real_game_files(real_cfg, split)
    by_type: dict[str, list[str]] = collections.defaultdict(list)
    for gf in game_files:
        by_type[RealAlfredEnv.task_type_from_gamefile(gf)].append(gf)

    unknown = by_type.get("unknown", [])
    all_present = all(tt in by_type and len(by_type[tt]) > 0 for tt in TASK_TYPES)

    return {
        "total_game_files": len(game_files),
        "counts_by_task_type": {tt: len(by_type.get(tt, [])) for tt in TASK_TYPES},
        "n_unknown": len(unknown),
        "all_6_task_types_present": all_present,
        "_by_type_files": by_type,  # not JSON-serialized in the report; used internally for sampling
    }


def run_expert_difficulty_proxy(
    real_cfg: dict, by_type_files: dict[str, list[str]], our_step_cap: int, n_samples: int = N_SAMPLES_PER_TYPE
) -> dict:
    rng = random.Random(SEED)
    results: dict[str, dict] = {}
    for tt in TASK_TYPES:
        pool = by_type_files.get(tt, [])
        sample = rng.sample(pool, min(n_samples, len(pool)))
        steps_list = []
        solved_count = 0
        for gf in sample:
            env = RealAlfredEnv(real_cfg, split="train", gamefile_path=gf)
            try:
                _obs, info = env.reset()
                solved = False
                steps_used = EXPERT_MAX_STEPS
                for step in range(EXPERT_MAX_STEPS):
                    action = info["extra.expert_plan"][0][0]
                    _obs, _reward, done, info = env.step(action)
                    if done:
                        solved = True
                        steps_used = step + 1
                        break
            finally:
                env.close()  # each single-game env is registered asynchronous=True regardless of
                # batch_size -- must close() or real ALFWorld leaks a subprocess per game sampled
            steps_list.append(steps_used)
            solved_count += int(solved)

        n = len(steps_list)
        steps_sorted = sorted(steps_list)
        fit_in_cap = sum(1 for s in steps_list if s <= our_step_cap) / n if n else 0.0
        results[tt] = {
            "n_sampled": n,
            "solved_pct": 100 * solved_count / n if n else 0.0,
            "mean_steps_to_solve": sum(steps_list) / n if n else None,
            "median_steps_to_solve": steps_sorted[n // 2] if n else None,
            "pct_solvable_within_our_step_cap": 100 * fit_in_cap,
        }
    return results


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])
    real_cfg = load_real_alfworld_config(cfg["env"]["real_alfworld_config_path"])
    our_step_cap = cfg["env"]["max_steps"]

    coverage = check_task_type_coverage(real_cfg, split=cfg["env"].get("real_split", "train"))
    by_type_files = coverage.pop("_by_type_files")

    print("=== Task-type coverage (real ALFWorld train split) ===")
    print(f"Total game files: {coverage['total_game_files']}")
    for tt in TASK_TYPES:
        n = coverage["counts_by_task_type"][tt]
        print(f"  {tt:<32} {n:>5} games ({100*n/coverage['total_game_files']:.1f}%)")
    print(f"All 6 task types present and correctly parsed: {coverage['all_6_task_types_present']}")

    difficulty = run_expert_difficulty_proxy(real_cfg, by_type_files, our_step_cap)

    print(f"\n=== Zero-cost difficulty proxy: ALFWorld's own scripted expert, n={N_SAMPLES_PER_TYPE}/type ===")
    print(f"(our env.max_steps cap = {our_step_cap})")
    print(f"{'task_type':<32}{'solved%':>9}{'mean_steps':>12}{'median_steps':>14}{'%% fits in our cap':>20}")
    for tt in TASK_TYPES:
        r = difficulty[tt]
        print(
            f"{tt:<32}{r['solved_pct']:>8.0f}%{r['mean_steps_to_solve']:>12.1f}"
            f"{r['median_steps_to_solve']:>14}{r['pct_solvable_within_our_step_cap']:>19.0f}%"
        )

    report = {
        "task_type_coverage": coverage,
        "difficulty_proxy": {
            "method": "ALFWorld's own built-in scripted (handcoded) expert, zero LLM calls",
            "n_samples_per_type": N_SAMPLES_PER_TYPE,
            "our_step_cap": our_step_cap,
            "results": difficulty,
        },
    }
    with open(results_dir / "task_type_difficulty_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {results_dir / 'task_type_difficulty_check.json'}")


if __name__ == "__main__":
    main()
