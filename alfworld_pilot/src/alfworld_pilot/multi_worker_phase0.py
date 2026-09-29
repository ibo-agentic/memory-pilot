"""2-GPU support for the Phase 0 pilot (2026-09-30, hardened after a real
Kaggle hang the same day) -- reuses timing_probe.py's own worker pattern,
already verified on real Kaggle (1.89x combined speedup): one subprocess per
GPU (CUDA_VISIBLE_DEVICES=0/1), each running run_jobs.py's own CLI against
its own job-list subset and its own results file, spawned fresh once per
checkpoint-interval chunk so the outer time-guard/checkpoint loop
(run_phase0_pilot.run_session_with_checkpoints) stays exactly the same for 1
worker or N.

Splitting is recomputed FRESH every chunk from whatever's currently pending
across ALL workers' result files combined -- never a persisted assignment --
so a restart with a different pending set (one worker made more progress
than the other, or one crashed) always gets a freshly (re)balanced split,
never a stale one.

THE REAL HANG, diagnosed: the first real sanity-check run had both workers
fetch AND load the ~15GB model onto their GPU at the exact same moment, and
loading stalled at 42% (layer 11) for 2+ hours with no error and no visible
output. `timing_probe.py --workers` worked fine earlier, but only because
the weights were ALREADY downloaded by then -- the difference here is a COLD
run: two simultaneous downloads/loads competing for the same disk/CPU/RAM on
a machine with limited cores, most likely, though the exact mechanism was
never confirmed (nothing errored; it just never made progress). Three fixes,
all independent of the exact mechanism holding:
  1. Pre-download the model ONCE in the parent process (`predownload_model`)
     before spawning any worker, so no worker subprocess ever races another
     over the same download.
  2. Stagger loading: worker N waits for worker N-1's `run_jobs.write_ready_marker`
     (written the moment that worker's OWN model finishes loading onto its
     GPU) before starting its own `from_pretrained` call
     (`run_jobs.wait_for_ready_marker`) -- so the two heaviest phases (model
     deserialization + 4-bit quantization) never happen on both GPUs at once,
     only the actual episode-running does.
  3. A watchdog (`run_multi_gpu_chunk`'s own polling loop, replacing a plain
     blocking `.wait()`): if NEITHER a worker's ready marker NOR its results
     file has been updated for `watchdog_timeout_seconds` (default 20 min)
     while it's still running, ALL workers are killed, the chunk returns
     what was actually completed, and `WatchdogTimeoutError` is raised so
     the session ends loudly instead of silently burning a full 12-hour
     Kaggle session on a hang with no error.

Also: subprocesses are launched with `-u` (unbuffered stdout/stderr) -- the
hang produced NO visible output at all, including `nvidia-smi -L`, most
likely because Python's default block buffering (active whenever stdout
isn't a TTY, i.e. whenever a parent captures a subprocess's output) only
flushes on a full buffer or process exit, neither of which happened during
the hang.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

from .job_list import load_job_list, write_job_list
from .run_jobs import completed_job_ids, ready_marker_path

DEFAULT_WATCHDOG_TIMEOUT_SECONDS = 1200.0  # 20 min
DEFAULT_WATCHDOG_POLL_INTERVAL_SECONDS = 15.0


class WatchdogTimeoutError(RuntimeError):
    """Raised by run_multi_gpu_chunk when a worker makes no progress (no
    ready marker written, no job completed) for watchdog_timeout_seconds --
    the caller (run_phase0_pilot.run_session_with_checkpoints) catches this
    specifically to push one final checkpoint of whatever WAS completed
    before re-raising, so the session ends loudly with data saved, not
    hung."""


def predownload_model(cfg: dict) -> None:
    """Downloads the pinned local LLM's weights ONCE, in the parent process,
    before any worker subprocess is spawned -- the first of the three fixes
    for the hang this module's docstring describes. Safe/cheap to call even
    when the weights are already cached (huggingface_hub checks ETags and
    skips re-downloading unchanged files); the point is that this is the
    ONLY process that ever downloads, so two workers can never race each
    other over the same cold download."""
    from huggingface_hub import snapshot_download

    llm_cfg = cfg["llm"]
    print(f"[multi_worker_phase0] pre-downloading {llm_cfg['model_id']}@{llm_cfg['revision']} once, before starting any worker...", flush=True)
    snapshot_download(repo_id=llm_cfg["model_id"], revision=llm_cfg["revision"])
    print("[multi_worker_phase0] model download confirmed complete.", flush=True)


def group_into_units(jobs: list[dict]) -> list[list[dict]]:
    """Groups jobs into indivisible assignment units: a ground_truth_fold_a/
    _b pair (matched by pair_index) is ONE unit, so both arms always land on
    the same worker (keeps a pair's two episodes co-located for simpler log
    correlation and debugging, per instruction -- not required for
    correctness, since each arm's own seeding is independent of where it
    runs). A logging job is its own unit. Order is deterministic given the
    same input jobs in the same order."""
    units: list[list[dict]] = []
    pair_units: dict[int, list[dict]] = {}
    pair_order: list[int] = []
    for job in jobs:
        if job["condition"] in ("ground_truth_fold_a", "ground_truth_fold_b"):
            pair_index = job["pair_index"]
            if pair_index not in pair_units:
                pair_units[pair_index] = []
                pair_order.append(pair_index)
            pair_units[pair_index].append(job)
        else:
            units.append([job])
    for pair_index in pair_order:
        units.append(pair_units[pair_index])
    return units


def split_units_across_workers(units: list[list[dict]], n_workers: int) -> list[list[dict]]:
    """Round-robin assigns whole units (see group_into_units) across
    n_workers, flattening each worker's assigned units back into a flat job
    list. Deterministic given the same units in the same order."""
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    worker_units: list[list[list[dict]]] = [[] for _ in range(n_workers)]
    for i, unit in enumerate(units):
        worker_units[i % n_workers].append(unit)
    return [[job for unit in w_units for job in unit] for w_units in worker_units]


def _progress_timestamp(ready_dir: pathlib.Path, results_path: pathlib.Path, worker_id: int) -> float:
    """Latest of: this worker's ready-marker mtime (model finished loading)
    or its results-file mtime (a job just completed, since run_jobs.py
    flushes after every job) -- whichever is more recent, 0.0 if neither
    exists yet. The watchdog's own "no progress" signal."""
    candidates = []
    ready_path = ready_marker_path(ready_dir, worker_id)
    if ready_path.exists():
        candidates.append(ready_path.stat().st_mtime)
    if pathlib.Path(results_path).exists():
        candidates.append(pathlib.Path(results_path).stat().st_mtime)
    return max(candidates) if candidates else 0.0


def run_multi_gpu_chunk(
    full_job_list_path: str | pathlib.Path,
    worker_results_paths: list[pathlib.Path],
    chunk_budget_seconds: float,
    work_dir: str | pathlib.Path,
    backend: str = "real",
    extra_worker_args: list[list[str]] | None = None,
    watchdog_timeout_seconds: float = DEFAULT_WATCHDOG_TIMEOUT_SECONDS,
    watchdog_poll_interval_seconds: float = DEFAULT_WATCHDOG_POLL_INTERVAL_SECONDS,
) -> int:
    """Runs ONE chunk of work split across len(worker_results_paths) GPUs,
    reusing run_jobs.py's own CLI (one subprocess per GPU,
    CUDA_VISIBLE_DEVICES=0..N-1 -- the same pattern timing_probe.py's
    --workers already verified on real Kaggle). See module docstring for why
    the split is recomputed fresh every call, never persisted, and for the
    three hang fixes (pre-download happens in the CALLER before this is ever
    invoked -- see run_phase0_pilot.main; staggered loading and the watchdog
    happen here).

    Workers are launched with `-u` (unbuffered output) and, for worker index
    i>0, `--wait-for-worker-id i-1` pointed at `work_dir` as the ready-marker
    directory -- worker 0 loads immediately, worker 1 waits for worker 0's
    marker first. Any stale ready markers from a PREVIOUS chunk are deleted
    before spawning, so staggering is re-armed fresh every chunk (not
    short-circuited by a leftover marker from last time).

    Watchdog: while any worker subprocess is still running, its last-progress
    timestamp (`_progress_timestamp` -- its ready marker or its results file,
    whichever is newer) is checked every `watchdog_poll_interval_seconds`. If
    a running worker's progress timestamp hasn't advanced for
    `watchdog_timeout_seconds`, ALL workers are killed and
    WatchdogTimeoutError is raised -- a hang must never silently burn the
    whole chunk (or a whole 12-hour session) with zero feedback.

    If a worker subprocess exits non-zero on its own (crashes, not killed by
    the watchdog), this does NOT abort or retry -- the other worker(s) still
    run their own chunk to completion, and whatever the crashed worker
    completed before dying is already safely flushed in its own results file
    and is kept, not lost. The crash is printed as a warning, never silently
    swallowed.

    extra_worker_args, if given, is a list of length len(worker_results_paths)
    of extra CLI args appended to each worker's own command -- used by tests
    to inject e.g. --fail-after or --simulate-hang-seconds; None means no
    extra args for any worker.

    Returns the total number of newly completed jobs across all workers this
    chunk (computed by diffing each worker's own completed-job count before
    and after, so it's correct regardless of which worker did how much) --
    even when WatchdogTimeoutError is raised, since that's computed and
    included in the exception's context via the caller re-checking the
    results files, not lost."""
    n_workers = len(worker_results_paths)
    all_jobs = load_job_list(full_job_list_path)
    already_done: set[str] = set()
    for results_path in worker_results_paths:
        already_done |= completed_job_ids(results_path)
    pending = [j for j in all_jobs if j["job_id"] not in already_done]

    if not pending:
        return 0

    units = group_into_units(pending)
    worker_job_lists = split_units_across_workers(units, n_workers)

    work_dir = pathlib.Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Staggering is re-armed fresh every chunk -- delete any ready markers a
    # PREVIOUS chunk's workers left behind, so worker 1 doesn't skip waiting
    # because of a stale marker from last time.
    for worker_id in range(n_workers):
        marker = ready_marker_path(work_dir, worker_id)
        if marker.exists():
            marker.unlink()

    n_before = sum(len(completed_job_ids(p)) for p in worker_results_paths)

    procs: list[subprocess.Popen | None] = []
    for worker_id, worker_jobs in enumerate(worker_job_lists):
        if not worker_jobs:
            procs.append(None)  # nothing assigned to this worker this chunk
            continue

        worker_job_list_path = work_dir / f"chunk_worker{worker_id}_jobs.json"
        write_job_list(worker_jobs, worker_job_list_path)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker_id)
        cmd = [
            sys.executable, "-u", "-m", "alfworld_pilot.run_jobs",
            "--job-list-path", str(worker_job_list_path),
            "--results-path", str(worker_results_paths[worker_id]),
            "--time-budget-seconds", str(chunk_budget_seconds),
            "--worker-id", str(worker_id),
            "--backend", backend,
            "--ready-dir", str(work_dir),
        ]
        if worker_id > 0:
            cmd += ["--wait-for-worker-id", str(worker_id - 1)]
        if extra_worker_args is not None:
            cmd += extra_worker_args[worker_id]
        procs.append(subprocess.Popen(cmd, env=env))

    # Watchdog loop: poll instead of a blocking .wait(), so a hung worker can
    # be detected and killed instead of silently consuming the whole chunk
    # (or the whole session) with no feedback.
    last_progress_time = {i: time.monotonic() for i in range(n_workers)}
    last_progress_ts_seen = {i: 0.0 for i in range(n_workers)}
    while True:
        all_done = True
        now = time.monotonic()
        hung_worker_ids: list[int] = []
        for worker_id, proc in enumerate(procs):
            if proc is None:
                continue
            if proc.poll() is None:
                all_done = False
                ts = _progress_timestamp(work_dir, worker_results_paths[worker_id], worker_id)
                if ts > last_progress_ts_seen[worker_id]:
                    last_progress_ts_seen[worker_id] = ts
                    last_progress_time[worker_id] = now
                elif now - last_progress_time[worker_id] > watchdog_timeout_seconds:
                    hung_worker_ids.append(worker_id)

        if hung_worker_ids:
            # Report EVERY currently-hung worker, not just one -- when
            # staggered loading blocks worker N on worker N-1, a hang in the
            # first worker leaves every later one looking equally stuck, and
            # naming only one would hide that from whoever reads the log.
            print(
                f"[multi_worker_phase0] WATCHDOG FAIL: worker(s) {hung_worker_ids} made no progress "
                f"(no ready marker, no completed job) for over {watchdog_timeout_seconds:.0f}s -- "
                "killing all workers and ending the session.", flush=True,
            )
            for proc in procs:
                if proc is not None and proc.poll() is None:
                    proc.kill()
            for proc in procs:
                if proc is not None:
                    proc.wait()
            n_after_kill = sum(len(completed_job_ids(p)) for p in worker_results_paths)
            raise WatchdogTimeoutError(
                f"worker(s) {hung_worker_ids} hung for over {watchdog_timeout_seconds:.0f}s with no progress "
                f"({n_after_kill - n_before} job(s) completed across all workers before the kill)"
            )

        if all_done:
            break
        time.sleep(watchdog_poll_interval_seconds)

    for worker_id, proc in enumerate(procs):
        if proc is None:
            continue
        ret = proc.returncode
        if ret != 0:
            print(
                f"[multi_worker_phase0] WARNING: worker {worker_id} exited with code {ret} (crashed) -- "
                "its already-completed jobs are kept; continuing with the other worker(s)."
            )

    n_after = sum(len(completed_job_ids(p)) for p in worker_results_paths)
    return n_after - n_before


def merge_worker_results(worker_results_paths: list[pathlib.Path], merged_path: pathlib.Path) -> int:
    """Concatenates every worker's results file into one merged JSONL file
    (job_id-deduplicated defensively, though the split guarantees disjoint
    job_ids across workers by construction) -- used to produce a single
    combined log for validate_per_task_type and for pushing one checkpoint.
    Returns the number of episodes written."""
    seen: set[str] = set()
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(merged_path, "w", encoding="utf-8") as out:
        for results_path in worker_results_paths:
            if not pathlib.Path(results_path).exists():
                continue
            with open(results_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ep = json.loads(line)
                    if ep["job_id"] in seen:
                        continue
                    seen.add(ep["job_id"])
                    out.write(line if line.endswith("\n") else line + "\n")
                    n_written += 1
    return n_written
