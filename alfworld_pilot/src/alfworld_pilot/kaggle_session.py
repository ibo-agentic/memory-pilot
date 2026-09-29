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
test suite.

Two things ARE unit-tested locally despite the above (tests/test_kaggle_session.py):
`find_attached_dataset`'s path discovery and `restore_folder`'s layout
detection -- both were wrong on the first real Kaggle run (2026-09-29
handoff test session 2 found the dataset at /kaggle/input/datasets/<owner>/
<name>/, not the originally assumed /kaggle/input/<name>/, and Kaggle had
auto-extracted the uploaded zip flat into the dataset root rather than
preserving a subfolder), fixed, and now covered by real filesystem-based
tests reproducing those exact layouts. What's still genuinely untestable
outside a real Kaggle session is anything that shells out to the actual
`kaggle` CLI (`copy_out_and_version`'s create/version calls, and the
`datasets status` existence check) -- verify those in Phase 0/1 before
relying on this for a real multi-session campaign.

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


def find_attached_dataset(slug: str) -> pathlib.Path | None:
    """Kaggle's mount path for an attached input dataset has been observed to
    vary across sessions: originally assumed /kaggle/input/<name>/, but a
    real session (2026-09-29 handoff test) actually mounted it at
    /kaggle/input/datasets/<owner>/<name>/ instead. Both are checked
    explicitly (derived from `slug`), with a recursive fallback search of
    /kaggle/input for a directory named exactly <name> if neither matches,
    in case the layout varies again. Always prints which path was used (or
    that none was found), so a future layout change shows up in the
    notebook's own output instead of silently doing nothing."""
    owner, sep, name = slug.partition("/")
    if not sep:  # slug had no "/" -- treat the whole thing as the name, no owner
        name, owner = owner, ""

    candidates = [KAGGLE_INPUT_ROOT / name]
    if owner:
        candidates.append(KAGGLE_INPUT_ROOT / "datasets" / owner / name)

    for candidate in candidates:
        if candidate.is_dir():
            print(f"[kaggle_session] found attached dataset at {candidate}")
            return candidate

    if KAGGLE_INPUT_ROOT.is_dir():
        for path in KAGGLE_INPUT_ROOT.rglob(name):
            if path.is_dir():
                print(f"[kaggle_session] found attached dataset via recursive search at {path}")
                return path

    print(
        f"[kaggle_session] no attached dataset found for slug {slug!r} "
        f"(checked {[str(c) for c in candidates]}, and recursively searched {KAGGLE_INPUT_ROOT})"
    )
    return None


def restore_folder(src_root: pathlib.Path, name: str, dest: pathlib.Path, other_names: tuple[str, ...] = ()) -> bool:
    """Restores `name` (e.g. "logs") from src_root into dest, handling THREE
    ways Kaggle can end up storing it:
      1. a plain extracted directory (src_root/name/);
      2. a zip archive (src_root/name.zip -- produced by `kaggle datasets
         create/version --dir-mode zip`, required because the plain `kaggle
         datasets create -p <dir>` silently SKIPPED subfolders entirely
         ("Skipping folder: logs; use '--dir-mode' to upload folders")).
         A zip's internal layout isn't assumed either way (contents at the
         zip root, or nested inside one more `name/` folder) -- both handled.
      3. a FLAT layout: on a real Kaggle session (2026-09-29), Kaggle
         auto-extracted the uploaded logs.zip directly into the dataset
         ROOT rather than preserving it as logs/ or logs.zip -- the pushed
         files (e.g. handoff_test_jobs.json) ended up as direct siblings at
         src_root. When neither form 1 nor 2 is found, this copies
         everything at src_root EXCEPT entries matching another known
         folder name (`other_names`, its own dir or .zip) into dest.

    Known residual limitation of form 3: if TWO OR MORE folders (e.g. both
    logs/ and a non-empty cache/) ever get flattened into the same dataset
    root simultaneously, their files would be indistinguishable and could
    mix -- `other_names` only excludes entries that are STILL a named
    dir/zip, not ones that were ALSO flattened. Not hit in the handoff test
    (cache was empty, so only logs.zip existed to flatten) but worth
    revisiting before relying on this for a real run with a non-trivial
    cache.

    Returns True if anything was restored, False if nothing was found at
    all (not an error -- the first session has nothing to restore)."""
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

    excluded_names = set(other_names) | {f"{o}.zip" for o in other_names} | {"dataset-metadata.json"}
    flat_entries = [p for p in src_root.iterdir() if p.name not in excluded_names]
    if flat_entries:
        dest.mkdir(parents=True, exist_ok=True)
        for entry in flat_entries:
            target = dest / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, target)
        print(
            f"[kaggle_session] copied in {src_root} -> {dest} (flat layout -- "
            "Kaggle auto-extracted the zip directly into the dataset root instead of preserving "
            f"{name}/ or {name}.zip)"
        )
        return True

    return False


def copy_in(slug: str, logs_dir: pathlib.Path, cache_dir: pathlib.Path) -> None:
    """Copies a previously-pushed checkpoint (logs + cost-tracker state +
    LLM cache) from the attached input dataset into writable working
    directories. A first-ever session has nothing to copy in -- expected, not
    an error."""
    src = find_attached_dataset(slug)
    if src is None:
        print(f"[kaggle_session] assuming this is the FIRST session for this campaign, starting fresh.")
        return
    names = ("logs", "cache")
    for name, dest in zip(names, (logs_dir, cache_dir)):
        other_names = tuple(n for n in names if n != name)
        if not restore_folder(src, name, dest, other_names=other_names):
            print(f"[kaggle_session] no {name} found at {src} (checked {name}/, {name}.zip, and a flat layout) -- nothing to restore")


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
