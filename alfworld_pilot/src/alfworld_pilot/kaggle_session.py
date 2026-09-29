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

Several things ARE unit-tested locally despite the above (tests/test_kaggle_session.py):
`find_attached_dataset`'s path discovery, `_zip_folder`'s archive layout,
and `restore_folder`/`copy_in`'s layout detection -- all diagnosed from a
real Kaggle run (2026-09-29 handoff test): session 2 found the dataset at
/kaggle/input/datasets/<owner>/<name>/, not the originally assumed
/kaggle/input/<name>/; and `kaggle datasets create --dir-mode zip`'s own
subfolder-zipping put files at the zip's root with no folder-name prefix,
so Kaggle's mount-time extraction landed them flat at the dataset root
instead of under a `logs/` subfolder. The path-discovery bug is fixed by
checking both known mount layouts (plus a recursive fallback). The flat-zip
bug is fixed at the SOURCE, not patched around on the input side:
`copy_out_and_version` now zips each folder itself (`_zip_folder`) with the
folder's name baked into the archive's own internal paths, instead of
relying on `--dir-mode` to do that (it doesn't). The flat-layout restore
path survives only as a documented last resort for datasets pushed before
this fix, and `copy_in` refuses to use it when more than one folder is
simultaneously missing its proper form, since a flat layout with two
folders' files mixed together is genuinely unresolvable, not just risky.
What's still genuinely untestable outside a real Kaggle session is anything
that shells out to the actual `kaggle` CLI (`copy_out_and_version`'s
create/version calls, and the `datasets status` existence check) -- verify
those in Phase 0/1 before relying on this for a real multi-session
campaign.

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


def restore_folder(
    src_root: pathlib.Path, name: str, dest: pathlib.Path, other_names: tuple[str, ...] = (), allow_flat_fallback: bool = True
) -> bool:
    """Restores `name` (e.g. "logs") from src_root into dest, handling THREE
    ways Kaggle can end up storing it:
      1. a plain extracted directory (src_root/name/);
      2. a zip archive (src_root/name.zip). Its internal layout isn't
         assumed either way (contents at the zip root, or nested inside one
         more `name/` folder) -- both handled. `_zip_folder` (used by
         `copy_out_and_version`) always writes the second form (a top-level
         `name/` prefix baked into the archive itself), specifically so
         Kaggle's own extraction-on-mount behavior has no chance to flatten
         it -- see that function's docstring for why this is the actual fix,
         not `--dir-mode`.
      3. a FLAT layout, kept ONLY as a last resort for backward
         compatibility with datasets pushed before the `_zip_folder` fix:
         Kaggle can auto-extract an uploaded zip directly into the dataset
         ROOT instead of preserving it as `name/` or `name.zip` (confirmed
         on a real Kaggle session, 2026-09-29 handoff test, when the zip's
         own internal paths had no folder prefix at all). Only ever reached
         when `allow_flat_fallback=True` -- `copy_in` sets this per-call,
         and only for a single ambiguous name, never more than one at a
         time (see its docstring for why: a flat layout with files from TWO
         OR MORE folders would be genuinely unresolvable, not just risky,
         so `copy_in` fails loudly rather than calling this with
         `allow_flat_fallback=True` for more than one name).

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

    if not allow_flat_fallback:
        return False

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
            f"[kaggle_session] copied in {src_root} -> {dest} (flat layout, LAST RESORT -- "
            f"neither {name}/ nor {name}.zip was found; this dataset likely predates the "
            "_zip_folder fix)"
        )
        return True

    return False


def copy_in(slug: str, logs_dir: pathlib.Path, cache_dir: pathlib.Path) -> None:
    """Copies a previously-pushed checkpoint (logs + cost-tracker state +
    LLM cache) from the attached input dataset into writable working
    directories. A first-ever session has nothing to copy in -- expected, not
    an error.

    Two passes, deliberately: first, restore whatever has a PROPER
    (namespaced) form -- a `name/` directory or `name.zip` -- for every
    known folder, with the risky flat fallback disabled. Only if EXACTLY ONE
    folder is still missing afterward does the second pass retry that one
    folder with the flat fallback allowed (safe: whatever flat files remain
    at the root, once the other folder's own proper form has already been
    accounted for, can only belong to this one). If MORE THAN ONE folder is
    missing its proper form, this raises instead of guessing -- a flat
    layout with files from two or more folders mixed at the dataset root is
    genuinely unresolvable, not just risky, and silently mixing them (or
    restoring the same files into both) would be worse than failing loudly."""
    src = find_attached_dataset(slug)
    if src is None:
        print("[kaggle_session] assuming this is the FIRST session for this campaign, starting fresh.")
        return

    names = ("logs", "cache")
    dests = {"logs": logs_dir, "cache": cache_dir}

    proper_found = {name: restore_folder(src, name, dests[name], allow_flat_fallback=False) for name in names}
    missing = [name for name in names if not proper_found[name]]

    if len(missing) > 1:
        raise RuntimeError(
            f"[kaggle_session] FAIL: {missing} are ALL missing their proper archived form (neither "
            f"<name>/ nor <name>.zip found at {src}) -- cannot safely fall back to a flat-layout scan "
            "when more than one folder is ambiguous, since files from different folders would be "
            "indistinguishable at the dataset root. This should not happen for datasets pushed after "
            "the logs/cache-prefixed zip fix (_zip_folder) -- if this is an OLD dataset pushed before "
            "that fix, push a fresh checkpoint instead of trying to resume from it."
        )

    for name in missing:  # at most one entry, by the check above
        other_names = tuple(n for n in names if n != name)
        if restore_folder(src, name, dests[name], other_names=other_names, allow_flat_fallback=True):
            print(f"[kaggle_session] restored {name!r} via the flat-layout last resort")
        else:
            print(f"[kaggle_session] no {name} found at {src} (checked {name}/, {name}.zip, and a flat layout) -- nothing to restore")


def _zip_folder(folder: pathlib.Path, name: str, out_path: pathlib.Path) -> None:
    """Zips `folder`'s contents into out_path with `name` baked in as the
    archive's OWN top-level directory (e.g. `logs/results.jsonl`, never bare
    `results.jsonl`) -- this, not `--dir-mode`, is what actually keeps
    folders distinguishable after Kaggle mounts the dataset. Confirmed on a
    real Kaggle session that `kaggle datasets create/version --dir-mode
    zip`'s own subfolder-zipping does NOT prefix the archive's internal
    paths with the folder's name -- its zip put files at the archive root,
    which is exactly why they landed flat (and, with two non-empty folders,
    indistinguishably mixed) once Kaggle unpacked it. Zipping it ourselves
    with an explicit prefix removes the dependency on however Kaggle's own
    extraction-on-mount behaves."""
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(pathlib.PurePosixPath(name) / path.relative_to(folder).as_posix()))


def copy_out_and_version(slug: str, logs_dir: pathlib.Path, cache_dir: pathlib.Path, message: str) -> None:
    """Packages the updated logs/cache into a new Kaggle Dataset version via
    the `kaggle` CLI. Each non-empty folder is zipped ourselves via
    `_zip_folder` (with its name baked in as the archive's own top-level
    directory) BEFORE calling the CLI -- staging then contains only flat
    files (`logs.zip`, `cache.zip`, `dataset-metadata.json`), so no real
    subfolders exist for `--dir-mode` to mishandle in the first place.
    Raises RuntimeError (does NOT just print a warning and continue) if the
    CLI reports any failure -- confirmed on a real Kaggle session that this
    CLI can print an error message ("Dataset creation error: Invalid Owner
    Id") while still exiting 0, so both the exit code AND the output text
    are checked; a push that didn't happen must never be reported as having
    happened."""
    staging = KAGGLE_WORKING / "_kaggle_session_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    if logs_dir.exists() and any(logs_dir.iterdir()):
        _zip_folder(logs_dir, "logs", staging / "logs.zip")
    if cache_dir.exists() and any(cache_dir.iterdir()):
        _zip_folder(cache_dir, "cache", staging / "cache.zip")

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
