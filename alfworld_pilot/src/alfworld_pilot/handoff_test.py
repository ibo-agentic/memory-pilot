"""Handoff test (revised 2026-09-29, after a real gap was found): the
script/notebook to run across two REAL, SEPARATE Kaggle sessions to verify
cross-session persistence.

**Why the previous version was wrong**: it ran both "sessions" as two calls
in the SAME local process/filesystem, which never exercises the thing that
actually matters -- Kaggle wipes `/kaggle/working` when a session ends, so a
same-process rerun could "pass" by just running everything fresh in one
place, without ever proving state survives a real session boundary. That
version is gone; this one uses the exact mechanism `run_phase0_pilot.py`
uses for the real campaign: `kaggle_session.copy_in` (pulls the last
checkpoint in from an attached Kaggle Dataset) and `copy_out_and_version`
(pushes the updated checkpoint back out as a new Dataset version) --
Kaggle-Dataset-specific calls that CANNOT be exercised locally (same
documented limitation `kaggle_session.py` already states). See README's
"Handoff test" section for exact click-by-click steps for both sessions.

What IS tested locally, for real: `check_session2_precondition` (pure,
unit-tested in tests/test_handoff_test.py) and the underlying `run_jobs`
skip/resume/no-duplicate logic (tests/test_job_list_and_runner.py) -- the
part this script adds on top is the fail-loudly guard and wiring it to the
real copy_in/copy_out_and_version calls, which by construction can't be
faked into "passing" locally.

--session 1: expects NO prior completed jobs (a fresh start is correct).
--session 2: REQUIRES finding session 1's completed jobs already restored by
  copy_in -- if it finds none, that means the Dataset was not actually
  attached as input to this notebook version (or session 1's push failed),
  and the script exits with an error instead of silently regenerating
  everything as if it were a first session.

Deliberately uses the mock backend (zero-cost, no GPU/ALFWorld dependency)
for the 5 episodes themselves -- this test verifies the ORCHESTRATION
mechanism (job list, resume, Dataset round-trip), not model behavior, which
is already verified separately (capability_check.py, determinism_check.py).
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

from .ground_truth_runner import MockTaskSource
from .job_list import generate_job_list, load_job_list, write_job_list
from .kaggle_session import copy_in, copy_out_and_version
from .memory_store import build_mock_store
from .mock_llm import MockLLMClient
from .run_jobs import completed_job_ids, run_jobs

KAGGLE_WORKING = pathlib.Path("/kaggle/working")
N_EPISODES = 5


def check_session2_precondition(done_before: set, session: int) -> None:
    """Raises SystemExit if session 2 finds zero prior completed jobs after
    copy_in -- fail loudly instead of silently treating a missing Dataset
    attachment as a legitimate fresh start. Pure and testable without
    touching Kaggle (tests/test_handoff_test.py)."""
    if session == 2 and len(done_before) == 0:
        raise SystemExit(
            "FAIL: session 2 expected to find session 1's completed jobs after "
            "copy_in from the attached Kaggle Dataset, but found none. This means "
            "either the Dataset was not attached as input to this notebook version, "
            "or session 1's copy_out_and_version push failed. NOT silently starting "
            "fresh -- fix the Dataset attachment (see README's Handoff test section) "
            "and re-run."
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-slug", required=True, help="e.g. your-kaggle-username/memory-pilot-handoff-test")
    p.add_argument("--session", type=int, required=True, choices=[1, 2])
    p.add_argument("--max-new", type=int, default=10)
    args = p.parse_args()

    logs_dir = KAGGLE_WORKING / "logs_kaggle"
    cache_dir = KAGGLE_WORKING / "cache_kaggle"
    job_list_path = logs_dir / "handoff_test_jobs.json"
    results_path = logs_dir / "handoff_test_results.jsonl"

    copy_in(args.dataset_slug, logs_dir, cache_dir)

    if not job_list_path.exists():
        if args.session == 2:
            raise SystemExit(
                "FAIL: session 2 found no job list after copy_in -- the Dataset was not "
                "attached correctly, or session 1 never completed. NOT silently generating "
                "a fresh job list."
            )
        jobs = generate_job_list(n_logging=N_EPISODES, n_gt_pairs=0)
        write_job_list(jobs, job_list_path)
        print(f"Session {args.session}: generated a new job list ({N_EPISODES} jobs) -> {job_list_path}")
    else:
        jobs = load_job_list(job_list_path)
        print(f"Session {args.session}: found an existing job list ({len(jobs)} jobs) -> {job_list_path}")

    done_before = completed_job_ids(results_path)
    print(f"Session {args.session}: {len(done_before)} job(s) already completed before this session: {sorted(done_before) or '(none)'}")
    check_session2_precondition(done_before, args.session)

    task_source = MockTaskSource()
    llm_client = MockLLMClient(strategy="scripted_success", seed=0)
    memories = build_mock_store(10, 100, 300, seed=0)
    n_new = run_jobs(job_list_path, results_path, task_source, llm_client, memories, 5, 0.3, 0.7, 30, max_new=args.max_new)
    print(f"Session {args.session}: ran {n_new} new job(s).")

    copy_out_and_version(args.dataset_slug, logs_dir, cache_dir, message=f"handoff test session {args.session}")

    all_job_ids = {j["job_id"] for j in jobs}
    all_done = completed_job_ids(results_path)
    with open(results_path, "r", encoding="utf-8") as f:
        counts = Counter(json.loads(line)["job_id"] for line in f if line.strip())
    duplicates = {jid: c for jid, c in counts.items() if c > 1}
    lost = all_job_ids - all_done

    print(f"Session {args.session} summary: {len(all_done)}/{len(all_job_ids)} done, duplicates={duplicates or 'none'}, lost={lost or 'none'}")
    if all_done == all_job_ids and not duplicates:
        print("PASS: final set equals the job list exactly, no duplicates, nothing lost.")
    elif duplicates:
        print("FAIL: at least one job ran more than once -- resume logic is broken.")
    else:
        print(f"IN PROGRESS (session {args.session}): {len(all_done)}/{len(all_job_ids)} done -- run session 2 (if this was session 1) to continue.")


if __name__ == "__main__":
    main()
