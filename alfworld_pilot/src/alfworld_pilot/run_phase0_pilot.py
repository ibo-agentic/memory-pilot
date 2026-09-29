"""Phase 0 pilot driver (2026-09-29, extended 2026-09-30 with a time guard
and periodic checkpointing) -- the one script kaggle/orchestrated_pilot.ipynb
calls, designed for Kaggle's "Save & Run All" (every cell runs
unconditionally, no manual choices): copies in any existing checkpoint from
the attached Kaggle Dataset (kaggle_session.copy_in -- a no-op printing
"first session" if there's nothing to copy in yet), generates the job list
if it doesn't already exist (idempotent -- a later session reuses the same
file rather than regenerating it), runs jobs in checkpoint_interval_seconds
chunks -- pushing an updated checkpoint after EVERY chunk, not just at the
end, so a crash loses at most about one interval's worth of work -- until
either the overall time_budget_seconds is exhausted or the job list has
nothing left to run.

Time guard: `--time-budget-seconds` defaults to 10.5h (37,800s), under
Kaggle's 12h session cap, leaving ~1.5h margin for the last chunk's own
episodes-in-flight plus the final checkpoint push and this process's own
startup/model-load overhead. `run_jobs.py`'s own budget check only happens
BEFORE starting a new job (not mid-episode), so an in-flight episode can run
a bit past either boundary -- accepted, not fought, since ALFWorld episodes
are capped at env.max_steps=50 and can't run indefinitely.

The Kaggle-Dataset-specific parts (copy_in/copy_out_and_version) inherit
kaggle_session.py's own documented limitation: untestable outside a real
Kaggle session. `run_session_with_checkpoints` (the chunking/stopping logic)
is deliberately factored out from `main()` so it IS unit-testable against
the mock backend, with `checkpoint_fn` injected as a fake
(tests/test_run_phase0_pilot.py) -- everything else this script calls
(job_list.py, run_jobs.py, run_estimators.py) is unit-tested too.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Callable

from . import config as config_mod
from .capability_check import build_local_client
from .env_factory import load_real_alfworld_config
from .ground_truth_runner import TaskSource
from .job_list import TARGET_MEMORY_IDS, generate_job_list, load_job_list, write_job_list
from .kaggle_memory_store import build_kaggle_store
from .kaggle_session import copy_in, copy_out_and_version
from .run_estimators import validate_per_task_type
from .run_jobs import run_jobs
from .weighted_task_source import WeightedRealTaskSource

ALFWORLD_PILOT_DIR = pathlib.Path(__file__).resolve().parents[2]
KAGGLE_WORKING = pathlib.Path("/kaggle/working")

DEFAULT_TIME_BUDGET_SECONDS = 37_800.0  # 10.5h, under Kaggle's 12h session cap
DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 7_200.0  # 2h -- a crash loses at most about this much work


def run_session_with_checkpoints(
    job_list_path: str | pathlib.Path,
    results_path: str | pathlib.Path,
    task_source: TaskSource,
    llm_client,
    memories: list,
    m: int,
    propensity_min: float,
    propensity_max: float,
    max_steps: int,
    time_budget_seconds: float,
    checkpoint_interval_seconds: float,
    checkpoint_fn: Callable[[], None],
    similarity_fn=None,
) -> int:
    """Runs jobs in checkpoint_interval_seconds-sized chunks (via
    run_jobs.run_jobs, already resumable/crash-safe per-job), calling
    checkpoint_fn() after EVERY chunk -- not just at the end -- so a crash
    loses at most about one interval's worth of work, not the whole session.
    Stops when either time_budget_seconds is exhausted (checked BEFORE
    starting a new chunk, so this never starts a chunk that couldn't
    possibly begin) or a chunk completes zero new jobs (the job list is
    exhausted, or nothing was runnable -- no point looping again).

    checkpoint_fn takes no arguments and is called purely for its side
    effect (typically writing validation metrics and pushing a Kaggle
    Dataset version) -- injected so this function's own chunking/stopping
    logic is unit-testable with a fake, without touching a real Kaggle
    Dataset. Returns the total number of newly completed jobs across the
    whole session."""
    t_start = time.monotonic()
    total_new = 0
    chunk_index = 0
    while True:
        elapsed = time.monotonic() - t_start
        remaining = time_budget_seconds - elapsed
        if remaining <= 0:
            print(f"[run_phase0_pilot] overall time budget ({time_budget_seconds:.0f}s) reached -- stopping before starting another chunk.")
            break
        chunk_budget = min(checkpoint_interval_seconds, remaining)
        chunk_index += 1
        print(f"[run_phase0_pilot] chunk {chunk_index}: running for up to {chunk_budget:.0f}s ({remaining:.0f}s left in the overall budget)")
        n_new = run_jobs(
            job_list_path, results_path, task_source, llm_client, memories, m, propensity_min, propensity_max, max_steps,
            time_budget_seconds=chunk_budget, similarity_fn=similarity_fn,
        )
        total_new += n_new
        print(f"[run_phase0_pilot] chunk {chunk_index}: completed {n_new} new job(s) ({total_new} total this session)")

        checkpoint_fn()

        if n_new == 0:
            print("[run_phase0_pilot] no new jobs completed in this chunk -- job list is exhausted (or nothing was runnable), stopping.")
            break

    return total_new


def _write_validation_metrics(results_path: pathlib.Path, memories: list, logs_dir: pathlib.Path) -> None:
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
        print(f"[run_phase0_pilot] wrote validation metrics from {len(logging_eps)} logging + {len(fold_eps)} ground-truth episodes so far.")
    else:
        print("[run_phase0_pilot] no episodes completed yet -- skipping validation metrics.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-slug", required=True, help="e.g. your-kaggle-username/memory-pilot-phase0")
    p.add_argument("--n-logging", type=int, default=1050)
    p.add_argument("--n-gt-pairs", type=int, default=225)
    p.add_argument(
        "--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS,
        help="Stop starting new jobs after this long (default 10.5h, under Kaggle's 12h session cap), then push and exit cleanly.",
    )
    p.add_argument(
        "--checkpoint-interval-seconds", type=float, default=DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
        help="Push to the checkpoint Dataset this often WHILE running, not just at the end (default 2h), so a crash loses at most about this much work.",
    )
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

    def _checkpoint() -> None:
        _write_validation_metrics(results_path, memories, logs_dir)
        copy_out_and_version(args.dataset_slug, logs_dir, cache_dir, message="Phase 0 pilot: periodic checkpoint")

    total_new = run_session_with_checkpoints(
        job_list_path, results_path, task_source, llm_client, memories, m, prop_min, prop_max, max_steps,
        args.time_budget_seconds, args.checkpoint_interval_seconds, _checkpoint,
    )
    print(f"[run_phase0_pilot] session finished: {total_new} new job(s) completed this session (cost tracker: {cost_tracker.summary()}).")


if __name__ == "__main__":
    main()
