"""Pre-generated, deterministic job list (2026-09-29) for the Kaggle
orchestration: every (task_id, condition, mask, seed) is decided up front by
this module, not drawn fresh each time a session runs -- so a job list
generated once, saved to disk, and re-read across many Kaggle sessions always
means the same thing, and `run_jobs.py`'s only job is "which of these rows
are already done."

Two job kinds:
  - "logging": one natural-retrieval episode, task_id drawn from
    WeightedRealTaskSource's normal task_id space (0, 1, 2, ...).
  - "ground_truth_fold_a" / "ground_truth_fold_b": one arm of a
    fold_over_ground_truth.run_fold_over_pair pair (see that module and
    README's "Fold-over ground-truth design" section for why a fold-over
    pair, not one-at-a-time, was chosen for Phase 0). task_id is drawn from a
    SEPARATE id space (GT_TASK_ID_OFFSET and up) so it can never collide with
    a logging task_id, and the mask is precomputed here (via the same
    deterministic draw_fold_over_mask used at execution time) purely for
    up-front auditability -- run_jobs.py does not need to recompute it, but
    could, and would get the identical value.
"""

from __future__ import annotations

import json
import pathlib

from .fold_over_ground_truth import draw_fold_over_mask

GT_TASK_ID_OFFSET = 1_000_000  # keeps ground-truth task_ids disjoint from any realistic logging-phase episode count

TARGET_MEMORY_IDS = ["pas_correct", "two_correct", "clean_harmful", "light_harmful", "irrelevant_3"]


def generate_job_list(
    n_logging: int,
    n_gt_pairs: int,
    target_memory_ids: list[str] | None = None,
    logging_start_seed: int = 0,
    gt_start_seed: int = GT_TASK_ID_OFFSET,
) -> list[dict]:
    """Deterministic: the same arguments always produce the identical job
    list (byte-for-byte, once serialized) -- no randomness at this level
    beyond draw_fold_over_mask's own deterministic, seeded draw."""
    target_memory_ids = target_memory_ids or TARGET_MEMORY_IDS
    jobs = []
    for i in range(n_logging):
        jobs.append(
            {
                "job_id": f"log_{i}",
                "condition": "logging",
                "task_id": logging_start_seed + i,
                "pair_index": None,
                "mask": None,
                "seed": logging_start_seed + i,
            }
        )
    for i in range(n_gt_pairs):
        task_id = gt_start_seed + i
        mask_a = draw_fold_over_mask(target_memory_ids, pair_index=i, task_seed=task_id)
        mask_b = {mid: 1 - v for mid, v in mask_a.items()}
        jobs.append(
            {
                "job_id": f"gtfold_{i}_a",
                "condition": "ground_truth_fold_a",
                "task_id": task_id,
                "pair_index": i,
                "mask": mask_a,
                "seed": task_id,
            }
        )
        jobs.append(
            {
                "job_id": f"gtfold_{i}_b",
                "condition": "ground_truth_fold_b",
                "task_id": task_id,
                "pair_index": i,
                "mask": mask_b,
                "seed": task_id,
            }
        )
    return jobs


def write_job_list(jobs: list[dict], path: str | pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)


def load_job_list(path: str | pathlib.Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
