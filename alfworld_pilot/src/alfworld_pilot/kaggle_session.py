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
import tempfile
import zipfile

KAGGLE_INPUT_ROOT = pathlib.Path("/kaggle/input")
KAGGLE_WORKING = pathlib.Path("/kaggle/working")

DEFAULT_TIME_BUDGET_SECONDS = 39_600.0  # 11h, leaving a real margin under Kaggle's 12h session cap

# Real Kaggle CLI output observed to contain an error message ("Dataset
# creation error: Invalid Owner Id") while still exiting 0 -- exit code alone
# is not a trustworthy success signal for this CLI. Checked case-insensitively
# against combined stdout+stderr; a false positive (failing on benign output
# that happens to mention "error") is far cheaper than a false negative
# (silently reporting a push that never happened).
_ERROR_MARKERS = ("error", "traceback", "exception")


def _find_attached_dataset(slug: str) -> pathlib.Path | None:
    """Kaggle mounts an attached input dataset at /kaggle/input/<name>/, where
    <name> is the LAST path segment of the dataset slug (not the full
    'username/dataset-name' slug) -- confirmed against a real Kaggle session
    (2026-09-29 handoff test)."""
    dataset_name = slug.split("/")[-1]
    candidate = KAGGLE_INPUT_ROOT / dataset_name
    return candidate if candidate.exists() else None


def restore_folder(src_root: pathlib.Path, name: str, dest: pathlib.Path) -> bool:
    """Restores `name` (e.g. "logs") from src_root into dest, handling BOTH
    ways Kaggle can end up storing it: a plain extracted directory
    (src_root/name/), or a zip archive (src_root/name.zip -- produced by
    `kaggle datasets create/version --dir-mode zip`, required because the
    plain `kaggle datasets create -p <dir>` silently SKIPPED subfolders
    entirely ("Skipping folder: logs; use '--dir-mode' to upload folders"),
    confirmed on a real Kaggle session). A zip's internal layout isn't
    assumed either way (contents at the zip root, or nested inside one more
    `name/` folder) -- both are handled. Returns True if anything was
    restored, False if neither form was found (not an error -- the first
    session has nothing to restore)."""
    plain_dir = src_root / name
    zip_path = src_root / f"{name}.zip"

    if plain_dir.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(plain_dir, dest, dirs_exist_ok=True)
        print(f"[kaggle_session] copied in {plain_dir} -> {dest} (plain directory)")
        return True

    if zip_path.is_file():
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            tmp_path = pathlib.Path(tmp)
            nested = tmp_path / name
            source_dir = nested if nested.is_dir() else tmp_path
            shutil.copytree(source_dir, dest, dirs_exist_ok=True)
        print(f"[kaggle_session] extracted {zip_path} -> {dest} (zip archive)")
        return True

    return False


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
        if not restore_folder(src, name, dest):
            print(f"[kaggle_session] no {name} found at {src} (neither {name}/ nor {name}.zip) -- nothing to restore")


def copy_out_and_version(slug: str, logs_dir: pathlib.Path, cache_dir: pathlib.Path, message: str) -> None:
    """Packages the updated logs/cache into a new Kaggle Dataset version via
    the `kaggle` CLI (`--dir-mode zip`, since subfolders are silently skipped
    otherwise -- see restore_folder's docstring). Raises RuntimeError (does
    NOT just print a warning and continue) if the CLI reports any failure --
    confirmed on a real Kaggle session that this CLI can print an error
    message ("Dataset creation error: Invalid Owner Id") while still exiting
    0, so both the exit code AND the output text are checked; a push that
    didn't happen must never be reported as having happened."""
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
        cmd = ["kaggle", "datasets", "version", "-p", str(staging), "-m", message, "-d", "--dir-mode", "zip"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(staging), "--dir-mode", "zip"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    print(output)

    failed = result.returncode != 0 or any(marker in output.lower() for marker in _ERROR_MARKERS)
    if failed:
        raise RuntimeError(
            f"[kaggle_session] FAIL: dataset push failed (exit code {result.returncode}; see CLI output above). "
            "This session's checkpoint progress is still on /kaggle/working, but the cross-session handoff did "
            "NOT update. NOT reporting a push that didn't happen -- fix the Dataset/permissions issue before "
            "the next session, or it will start over."
        )
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
