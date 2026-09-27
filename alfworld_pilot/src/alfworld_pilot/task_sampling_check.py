"""Task-sampling weights check (2026-09-28): what pooled success rate do
kaggle_config.yaml's new env.task_type_weights (favoring
pick_and_place_simple / look_at_obj_in_light) give, using the REAL per-type
success rates already measured by capability_check.py -- and does
WeightedRealTaskSource actually draw task types at close to the intended
proportions over many task_ids?

Zero new episodes and zero LLM calls: reuses
results_kaggle/capability_check_all_types_zeroshot_withmem.json's per-type
rates (n=10/type, zero-shot + memories) and a cheap draw-mix check (task-type
selection only, no env.reset(), no model) against the real ALFWorld
game-file listing already on disk.

Usage (from alfworld_pilot/, using the .venv-kaggle venv, or any venv with
alfworld installed):
    PYTHONPATH=src .venv-kaggle/bin/python -m alfworld_pilot.task_sampling_check
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter

from . import config as config_mod
from .env_factory import load_real_alfworld_config
from .weighted_task_source import WeightedRealTaskSource

ALFWORLD_PILOT_DIR = pathlib.Path(__file__).resolve().parents[2]
CAPABILITY_CHECK_PATH = ALFWORLD_PILOT_DIR / "results_kaggle" / "capability_check_all_types_zeroshot_withmem.json"


def load_per_type_success_rates() -> dict[str, float]:
    with open(CAPABILITY_CHECK_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {tt: r["summary"]["success_rate"] for tt, r in d["results"].items()}


def expected_pooled_success_rate(weights: dict[str, float], rates: dict[str, float]) -> float:
    total_w = sum(weights.values())
    missing = [tt for tt in weights if tt not in rates]
    if missing:
        raise ValueError(f"No measured success rate for task type(s): {missing}")
    return sum((w / total_w) * rates[tt] for tt, w in weights.items())


def check_draw_mix(source: WeightedRealTaskSource, n_draws: int) -> dict[str, float]:
    counts = Counter(source.task_type(task_id) for task_id in range(n_draws))
    return {tt: c / n_draws for tt, c in counts.items()}


def main() -> None:
    cfg_path = ALFWORLD_PILOT_DIR / "kaggle_config.yaml"
    cfg = config_mod.load_config(cfg_path)
    weights = cfg["env"]["task_type_weights"]

    rates = load_per_type_success_rates()
    print(f"Per-type success rate ({CAPABILITY_CHECK_PATH.name}, n=10/type, zero-shot + memories):")
    for tt, r in rates.items():
        print(f"  {tt:32s} {r:.2f}")

    pooled = expected_pooled_success_rate(weights, rates)
    print(f"\nWeights (env.task_type_weights, kaggle_config.yaml): {weights}")
    print(f"Expected pooled success rate under these weights: {pooled:.3f}")

    real_cfg = load_real_alfworld_config(cfg["env"]["real_alfworld_config_path"])
    split = cfg["env"]["real_split"]
    source = WeightedRealTaskSource(real_cfg, split=split, task_type_weights=weights)

    n_draws = 20000
    realized = check_draw_mix(source, n_draws)
    total_w = sum(weights.values())
    print(f"\nRealized draw mix over {n_draws} task_ids (intended vs. actual):")
    for tt in weights:
        intended = weights[tt] / total_w
        actual = realized.get(tt, 0.0)
        print(f"  {tt:32s} intended={intended:.3f}  realized={actual:.3f}  diff={actual - intended:+.4f}")


if __name__ == "__main__":
    main()
