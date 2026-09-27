"""Cross-Kaggle-session checkpoint wrapper: copies in the latest checkpoint
state from an attached Kaggle Dataset (read-only input), runs run_chunked.py's
supervisor for a time-boxed budget, then copies the updated state back out and
pushes it as a NEW VERSION of that same Kaggle Dataset -- so the next
session's input attachment picks up where this one left off.

This exists ONLY because /kaggle/working is wiped when an interactive Kaggle
session ends, unlike this project's WSL2 environment, which has a persistent
home directory. It is a thin layer ON TOP OF checkpointed_runner.py/
run_chunked.py, which already handle intra-session crash/restart
resumability unchanged and are exercised by this project's existing (passing)
test suite. The copy-in/copy-out/dataset-version logic below is Kaggle-API-
specific and CANNOT be tested outside an actual Kaggle session -- it is
untested by construction until run there; verify it in Phase 0/1 before
relying on it for a real multi-session campaign.

Usage (inside a Kaggle notebook cell, after Phase 0/1 pass):
    python -m alfworld_pilot.kaggle_session logging \\
        --dataset-slug <your-kaggle-username>/memory-pilot-kaggle-checkpoints \\
        --n-episodes 500 --time-budget-seconds 39600 \\
        --config kaggle_config.yaml --llm local --memory-store kaggle

Every argument after the ones this module defines is forwarded verbatim to
`alfworld_pilot.run_chunked` -- see that module's --help for the rest
(--n-episodes/--n-pairs, --memory-id, --llm, --memory-store, --config, etc.).
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

KAGGLE_INPUT_ROOT = pathlib.Path("/kaggle/input")
KAGGLE_WORKING = pathlib.Path("/kaggle/working")

DEFAULT_TIME_BUDGET_SECONDS = 39_600.0  # 11h, leaving a real margin under Kaggle's 12h session cap


def _find_attached_dataset(slug: str) -> pathlib.Path | None:
    """Kaggle mounts an attached input dataset at /kaggle/input/<name>/, where
    <name> is the LAST path segment of the dataset slug (not the full
    'username/dataset-name' slug) -- confirm this matches what an actual
    Kaggle session shows (Phase 0/1), this is documented Kaggle behavior but
    unverified against this specific setup."""
    dataset_name = slug.split("/")[-1]
    candidate = KAGGLE_INPUT_ROOT / dataset_name
    return candidate if candidate.exists() else None


def copy_in(slug: str, logs_dir: pathlib.Path, cache_dir: pathlib.Path) -> None:
    """Copies a previously-pushed checkpoint (logs + cost-tracker state +
    LLM cache) from the attached input dataset into writable working
    directories. A first-ever session has nothing to copy in -- expected, not
    an error."""
    src = _find_attached_dataset(slug)
    if src is None:
        print(
            f"[kaggle_session] no attached dataset found at /kaggle/input/{slug.split('/')[-1]}/ "
            f"-- assuming this is the FIRST session for this campaign, starting fresh."
        )
        return
    for name, dest in (("logs", logs_dir), ("cache", cache_dir)):
        src_sub = src / name
        if src_sub.exists():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_sub, dest, dirs_exist_ok=True)
            print(f"[kaggle_session] copied in {src_sub} -> {dest}")


def copy_out_and_version(slug: str, logs_dir: pathlib.Path, cache_dir: pathlib.Path, message: str) -> None:
    """Packages the updated logs/cache into a new Kaggle Dataset version via
    the `kaggle` CLI. Kaggle notebooks running ON Kaggle's own infrastructure
    are documented to have this pre-authenticated -- VERIFY this in an actual
    session rather than assuming it; if it fails, this session's own progress
    is still safe on /kaggle/working (or as the notebook's own committed
    output), but the cross-session handoff will not have updated, and the
    next session needs a manual fix before it can resume correctly."""
    staging = KAGGLE_WORKING / "_kaggle_session_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    if logs_dir.exists():
        shutil.copytree(logs_dir, staging / "logs")
    if cache_dir.exists():
        shutil.copytree(cache_dir, staging / "cache")

    dataset_name = slug.split("/")[-1]
    metadata_path = staging / "dataset-metadata.json"
    metadata_path.write_text(
        '{"title": "%s", "id": "%s", "licenses": [{"name": "CC0-1.0"}]}' % (dataset_name, slug)
    )

    dataset_exists = subprocess.run(["kaggle", "datasets", "status", slug], capture_output=True).returncode == 0
    if dataset_exists:
        cmd = ["kaggle", "datasets", "version", "-p", str(staging), "-m", message, "-d"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(staging)]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            "[kaggle_session] WARNING: dataset push failed (see output above). This session's "
            "checkpoint progress is still on /kaggle/working, but the cross-session handoff did "
            "NOT update -- fix this before the next session, or it will start over."
        )
    else:
        print(f"[kaggle_session] pushed checkpoint state to Kaggle Dataset {slug}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["logging", "ground_truth"])
    p.add_argument("--dataset-slug", required=True, help="e.g. your-kaggle-username/memory-pilot-kaggle-checkpoints")
    p.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    args, passthrough = p.parse_known_args()

    logs_dir = KAGGLE_WORKING / "logs"
    cache_dir = KAGGLE_WORKING / "cache_kaggle"

    copy_in(args.dataset_slug, logs_dir, cache_dir)

    cmd = [
        sys.executable, "-m", "alfworld_pilot.run_chunked", args.phase,
        "--logs-dir", str(logs_dir),
        "--cache-dir", str(cache_dir),
        "--time-budget-seconds", str(args.time_budget_seconds),
    ] + passthrough
    print(f"[kaggle_session] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)

    copy_out_and_version(args.dataset_slug, logs_dir, cache_dir, message=f"{args.phase} checkpoint update")


if __name__ == "__main__":
    main()
