"""Phase 0 pilot driver (2026-09-29) -- the one script
kaggle/orchestrated_pilot.ipynb calls, designed for Kaggle's "Save & Run All"
(every cell runs unconditionally, no manual choices): copies in any existing
checkpoint from the attached Kaggle Dataset (kaggle_session.copy_in -- a
no-op printing "first session" if there's nothing to copy in yet), generates
the job list if it doesn't already exist (idempotent -- a later session
reuses the same file rather than regenerating it), runs jobs up to a time
budget, runs the pre-registered validation-metrics estimators on whatever's
done so far (partial results are expected and fine mid-campaign, not an
error), and pushes the updated checkpoint back out as a new Kaggle Dataset
version so the next session picks up where this one left off.

The Kaggle-Dataset-specific parts (copy_in/copy_out_and_version) inherit
kaggle_session.py's own documented limitation: untestable outside a real
Kaggle session. Everything else this script calls (job_list.py, run_jobs.py,
run_estimators.py) is unit-tested against the mock backend.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from . import config as config_mod
from .capability_check import build_local_client
from .env_factory import load_real_alfworld_config
from .job_list import TARGET_MEMORY_IDS, generate_job_list, load_job_list, write_job_list
from .kaggle_memory_store import build_kaggle_store
from .kaggle_session import copy_in, copy_out_and_version
from .run_estimators import validate_per_task_type
from .run_jobs import run_jobs
from .weighted_task_source import WeightedRealTaskSource

ALFWORLD_PILOT_DIR = pathlib.Path(__file__).resolve().parents[2]
KAGGLE_WORKING = pathlib.Path("/kaggle/working")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-slug", required=True, help="e.g. your-kaggle-username/memory-pilot-phase0-checkpoints")
    p.add_argument("--n-logging", type=int, default=1050)
    p.add_argument("--n-gt-pairs", type=int, default=225)
    p.add_argument("--time-budget-seconds", type=float, default=39_600.0)  # 11h, under Kaggle's 12h session cap
    args = p.parse_args()

    logs_dir = KAGGLE_WORKING / "logs_kaggle"
    cache_dir = KAGGLE_WORKING / "cache_kaggle"
    job_list_path = logs_dir / "phase0_job_list.json"
    results_path = logs_dir / "phase0_results.jsonl"

    copy_in(args.dataset_slug, logs_dir, cache_dir)

    if not job_list_path.exists():
        jobs = generate_job_list(n_logging=args.n_logging, n_gt_pairs=args.n_gt_pairs)
        write_job_list(jobs, job_list_path)
        print(f"Generated new Phase 0 job list: {args.n_logging} logging + {args.n_gt_pairs} ground-truth fold pairs ({len(jobs)} total jobs)")
    else:
        jobs = load_job_list(job_list_path)
        print(f"Resuming existing job list ({len(jobs)} jobs)")

    cfg_path = ALFWORLD_PILOT_DIR / "kaggle_config.yaml"
    cfg = config_mod.load_config(cfg_path)
    cfg["cache"]["dir"] = str(cache_dir)  # session-local cache, persisted via copy_out_and_version
    real_cfg = load_real_alfworld_config(cfg["env"]["real_alfworld_config_path"])
    task_source = WeightedRealTaskSource(
        real_cfg, split=cfg["env"]["real_split"], task_type_weights=cfg["env"]["task_type_weights"]
    )
    llm_client, cost_tracker = build_local_client(cfg)
    memories = build_kaggle_store()
    m = cfg["retrieval"]["M"]
    prop_min, prop_max = cfg["retrieval"]["propensity_min"], cfg["retrieval"]["propensity_max"]
    max_steps = cfg["env"]["max_steps"]

    n_new = run_jobs(
        job_list_path, results_path, task_source, llm_client, memories, m, prop_min, prop_max, max_steps,
        time_budget_seconds=args.time_budget_seconds,
    )
    print(f"Completed {n_new} new job(s) this session (cost tracker: {cost_tracker.summary()}).")

    logging_eps, fold_eps = [], []
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ep = json.loads(line)
                (fold_eps if "fold_over_pair_index" in ep else logging_eps).append(ep)

    if logging_eps or fold_eps:
        all_memory_ids = [mem.mem_id for mem in memories]
        validation = validate_per_task_type(logging_eps, fold_eps, all_memory_ids, TARGET_MEMORY_IDS)
        with open(logs_dir / "phase0_validation.json", "w", encoding="utf-8") as f:
            json.dump(validation, f, indent=2)
        print(f"Wrote validation metrics from {len(logging_eps)} logging + {len(fold_eps)} ground-truth episodes so far.")
    else:
        print("No episodes completed yet -- skipping validation metrics this session.")

    copy_out_and_version(args.dataset_slug, logs_dir, cache_dir, message=f"Phase 0 pilot: +{n_new} jobs this session")


if __name__ == "__main__":
    main()
