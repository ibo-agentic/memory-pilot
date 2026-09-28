"""Handoff test (2026-09-29): the literal script to run twice on Kaggle to
verify the job-list runner survives a real session boundary -- run once
(stops after a few jobs), start a genuinely NEW Kaggle session pointed at the
SAME job list and results files, run it again, and confirm nothing ran twice
or got lost.

Usage on Kaggle (5 logging episodes total):
    Session 1: PYTHONPATH=src python -m alfworld_pilot.handoff_test --max-new 3
    (end the session here -- close the notebook, let it time out, whatever)
    Session 2 (fresh kernel, same /kaggle/working or re-attached dataset):
               PYTHONPATH=src python -m alfworld_pilot.handoff_test --max-new 10
    -> session 2 should report "5/5 done, no duplicates, nothing lost."

Uses the real Kaggle backend (transformers + real ALFWorld,
kaggle_config.yaml) by default -- --backend mock runs the identical
orchestration logic against the zero-cost mock backend, used to verify this
script itself locally before it's ever run on Kaggle.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

from . import config as config_mod
from .capability_check import build_local_client
from .env_factory import load_real_alfworld_config
from .ground_truth_runner import MockTaskSource
from .job_list import generate_job_list, load_job_list, write_job_list
from .kaggle_memory_store import build_kaggle_store
from .memory_store import build_mock_store
from .mock_llm import MockLLMClient
from .run_jobs import completed_job_ids, run_jobs
from .weighted_task_source import WeightedRealTaskSource

ALFWORLD_PILOT_DIR = pathlib.Path(__file__).resolve().parents[2]
JOB_LIST_PATH = ALFWORLD_PILOT_DIR / "results_kaggle" / "handoff_test_jobs.json"
RESULTS_PATH = ALFWORLD_PILOT_DIR / "results_kaggle" / "handoff_test_results.jsonl"
N_EPISODES = 5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-new", type=int, default=3)
    p.add_argument("--backend", choices=["mock", "transformers"], default="transformers")
    args = p.parse_args()

    if not JOB_LIST_PATH.exists():
        jobs = generate_job_list(n_logging=N_EPISODES, n_gt_pairs=0)
        write_job_list(jobs, JOB_LIST_PATH)
        print(f"Generated new job list ({N_EPISODES} logging jobs) -> {JOB_LIST_PATH}")
    else:
        jobs = load_job_list(JOB_LIST_PATH)
        print(f"Reusing existing job list -> {JOB_LIST_PATH} ({len(jobs)} jobs)")

    already_done_before = completed_job_ids(RESULTS_PATH)
    print(f"Already completed before this call: {sorted(already_done_before) or '(none -- first session)'}")

    if args.backend == "mock":
        task_source = MockTaskSource()
        llm_client = MockLLMClient(strategy="scripted_success", seed=0)
        memories = build_mock_store(10, 100, 300, seed=0)
        m, prop_min, prop_max, max_steps = 5, 0.3, 0.7, 30
    else:
        cfg_path = ALFWORLD_PILOT_DIR / "kaggle_config.yaml"
        cfg = config_mod.load_config(cfg_path)
        real_cfg = load_real_alfworld_config(cfg["env"]["real_alfworld_config_path"])
        task_source = WeightedRealTaskSource(
            real_cfg, split=cfg["env"]["real_split"], task_type_weights=cfg["env"]["task_type_weights"]
        )
        llm_client, _ = build_local_client(cfg)
        memories = build_kaggle_store()
        m = cfg["retrieval"]["M"]
        prop_min, prop_max = cfg["retrieval"]["propensity_min"], cfg["retrieval"]["propensity_max"]
        max_steps = cfg["env"]["max_steps"]

    n_new = run_jobs(JOB_LIST_PATH, RESULTS_PATH, task_source, llm_client, memories, m, prop_min, prop_max, max_steps, max_new=args.max_new)
    print(f"Ran {n_new} new job(s) this call.")

    all_job_ids = {j["job_id"] for j in jobs}
    all_done = completed_job_ids(RESULTS_PATH)
    print(f"Total completed: {len(all_done)}/{len(all_job_ids)}")

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        counts = Counter(json.loads(line)["job_id"] for line in f if line.strip())
    duplicates = {jid: c for jid, c in counts.items() if c > 1}
    lost = all_job_ids - all_done

    print(f"Duplicates: {duplicates if duplicates else 'none'}")
    print(f"Lost/missing: {lost if lost else 'none'}")

    if all_done == all_job_ids and not duplicates:
        print("PASS: all jobs done exactly once, nothing lost or duplicated.")
    elif duplicates:
        print("FAIL: at least one job ran more than once -- resume logic is broken.")
    else:
        print(f"IN PROGRESS: {len(all_done)}/{len(all_job_ids)} done so far -- run again (a new session) to continue.")


if __name__ == "__main__":
    main()
