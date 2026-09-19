"""Zero-cost follow-up to memory_sanity_check.py: breaks its 15 pairs down
by ALFWorld task type. Doesn't need any new LLM calls -- task_type for a
given pair_index is a pure function of the (already-fixed) game-file list
and index, recoverable without touching the API.

Finding this was built to surface: naive `task_id % len(game_files)`
indexing over `list_real_game_files`' raw filesystem-walk order landed
13/15 sampled games on `pick_and_place_simple` (the EASIEST task type) and
only 2/15 on `pick_two_obj_and_place` (the hardest), with zero samples from
the other 4 types. That's an accident of directory traversal order, not a
deliberate choice -- and it's most of why the pooled 93%/80% success rate
sat too close to ceiling to resolve per-memory effects.

Usage:
    python -m alfworld_pilot.memory_sanity_check_breakdown
"""

from __future__ import annotations

import collections
import json
import pathlib

from . import config as config_mod
from .env_factory import load_real_alfworld_config
from .env_interface import RealAlfredEnv, list_real_game_files


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    results_dir = pathlib.Path(cfg["paths"]["results_dir"])

    real_cfg = load_real_alfworld_config(cfg["env"]["real_alfworld_config_path"])
    game_files = list_real_game_files(real_cfg, split=cfg["env"].get("real_split", "train"))

    with open(results_dir / "memory_sanity_check.json", "r", encoding="utf-8") as f:
        sanity = json.load(f)

    by_type = collections.defaultdict(list)
    for p in sanity["pairs"]:
        i = p["pair_index"]
        tt = RealAlfredEnv.task_type_from_gamefile(game_files[i % len(game_files)])
        by_type[tt].append(p)

    print(f"{'task_type':<32}{'n_pairs':>8}{'with_mem success':>20}{'baseline success':>20}")
    breakdown = {}
    for tt, pairs in by_type.items():
        n = len(pairs)
        sw = sum(p["with_memories"]["success"] for p in pairs)
        sb = sum(p["baseline"]["success"] for p in pairs)
        breakdown[tt] = {"n_pairs": n, "with_memories_successes": sw, "baseline_successes": sb, "with_memories_rate": sw / n, "baseline_rate": sb / n}
        print(f"{tt:<32}{n:>8}{f'{sw}/{n} ({100 * sw / n:.0f}%)':>20}{f'{sb}/{n} ({100 * sb / n:.0f}%)':>20}")

    n_covered_types = len(by_type)
    print(f"\n{n_covered_types}/6 task types represented in this 15-pair sample (naive task_id indexing, not a deliberate stratified draw).")

    with open(results_dir / "memory_sanity_check_breakdown.json", "w", encoding="utf-8") as f:
        json.dump({"by_task_type": breakdown, "n_task_types_covered": n_covered_types}, f, indent=2)
    print(f"\nWrote {results_dir / 'memory_sanity_check_breakdown.json'}")


if __name__ == "__main__":
    main()
