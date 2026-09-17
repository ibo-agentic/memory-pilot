"""CLI: generate episode logs for both simulators across all seeds.

Usage:
    python -m memory_ope.simulator.run_simulation
"""

from __future__ import annotations

import pathlib

from .. import config as config_mod
from ..logging_utils import write_episodes
from . import hitchhiker, task_difficulty


def main() -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    logs_dir = pathlib.Path(cfg["paths"]["logs_dir"])

    for seed in range(cfg["seed_count"]):
        td_episodes = task_difficulty.generate_episodes(cfg, seed)
        write_episodes(logs_dir / f"task_difficulty_seed{seed}.jsonl", td_episodes)

        hh_episodes = hitchhiker.generate_episodes(cfg, seed)
        write_episodes(logs_dir / f"hitchhiker_seed{seed}.jsonl", hh_episodes)

        print(f"seed {seed}: wrote {len(td_episodes)} task_difficulty + {len(hh_episodes)} hitchhiker episodes")

    print(f"Done. Logs written to {logs_dir}")


if __name__ == "__main__":
    main()
