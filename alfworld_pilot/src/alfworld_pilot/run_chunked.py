"""Chunked-subprocess supervisor for the real Part C run.

WHY chunking is needed (not just a nice-to-have): real ALFWorld's PDDL
backend leaks ~32.5MB of tmpfs-backed memory on every load of a NEW game --
`fast_downward.load_lib()` (a `textworld`/`alfworld` dependency, upstream
code, not this project's) `dlopen()`s a FRESH temp copy of its native
shared library on every `env.load()` and never `dlclose()`s the previous
one. Confirmed empirically (2026-09-19): a single process doing
~180-240 single-game constructions exhausted this machine's 5.8GB tmpfs
`/tmp` and crashed with "No space left on device"; killing the process
immediately reclaimed all of it (tmpfs usage dropped from ~5.4GB back to
~500KB), confirming the leak is scoped to a process's LIFETIME, not
permanent or system-wide. Restarting the process periodically is the fix.
`checkpointed_runner.py` already makes a chunk boundary and a mid-chunk
crash equivalent (both just resume from the log's current length) -- this
module is the subprocess restart loop built on top of that.

Two ways to reduce the SAME risk, used together:
  1. This chunking (bounds each process's total game-load count).
  2. Pointing TMPDIR at a disk-backed directory instead of the default
     tmpfs `/tmp` (moves the ceiling from ~5.8GB to however much real disk
     is free -- doesn't fix the leak, just gives it a much bigger tank).
     Set TMPDIR in the environment before running this module if `/tmp` on
     your machine is small; this module does not set it for you.

Usage:
    python -m alfworld_pilot.run_chunked logging --n-episodes 5000 \\
        --model-id <id> --pricing-input 0.1 --pricing-output 0.3
    python -m alfworld_pilot.run_chunked ground_truth --memory-id mem_3 \\
        --n-pairs 166 --model-id <id> --pricing-input 0.1 --pricing-output 0.3
    python -m alfworld_pilot.run_chunked logging --n-episodes 5 --llm mock --backend mock   # zero-cost test mode

Internal (one subprocess's single chunk -- you don't need to call this
directly, the supervisor above does):
    python -m alfworld_pilot.run_chunked --worker logging --n-episodes 5000 --max-new 100 ...
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

from . import config as config_mod
from .checkpointed_runner import run_ground_truth_phase_checkpointed, run_logging_phase_checkpointed
from .env_factory import build_task_source
from .memory_store import build_mock_store

DEFAULT_CHUNK_SIZE = 100  # well under the ~178-construction tmpfs ceiling measured on this machine
# (32.5MB leaked per new game load / 5.8GB tmpfs, with margin for other concurrent usage) --
# re-measure (or just point TMPDIR at disk) if run on a machine with a different /tmp size.
DEFAULT_MAX_RESTARTS = 500  # supervisor gives up after this many chunk attempts regardless of progress


def _build_memories(cfg: dict) -> list:
    return build_mock_store(
        cfg["memory_store"]["n_memories"], cfg["memory_store"]["lesson_min_tokens"], cfg["memory_store"]["lesson_max_tokens"], seed=0
    )


def _build_llm_client_and_tracker(cfg: dict, args: argparse.Namespace):
    if args.llm == "mock":
        from .mock_llm import MockLLMClient

        return MockLLMClient(strategy="scripted_success", seed=0), None

    from .cache import LLMCache
    from .cost_tracker import CostTracker
    from .llm_client import OpenRouterClient

    if not args.model_id or args.pricing_input is None or args.pricing_output is None:
        raise SystemExit("--llm real requires --model-id, --pricing-input, and --pricing-output")

    cache = LLMCache(cfg["cache"]["dir"])
    cost_state_path = pathlib.Path(cfg["paths"]["logs_dir"]) / "cost_tracker_state.json"
    cost_tracker = CostTracker(
        cfg["cost_control"]["hard_cap_usd"], args.pricing_input, args.pricing_output, state_path=cost_state_path
    )
    llm_client = OpenRouterClient(
        model_id=args.model_id,
        base_url=cfg["llm"]["base_url"],
        api_key_env_var=cfg["llm"]["api_key_env_var"],
        reasoning_enabled=cfg["llm"]["reasoning_enabled"],
        temperature=cfg["llm"]["temperature"],
        max_output_tokens=cfg["llm"]["max_output_tokens"],
        cache=cache,
        cost_tracker=cost_tracker,
    )
    return llm_client, cost_tracker


def _run_one_chunk(args: argparse.Namespace) -> None:
    """Runs inside the WORKER subprocess: build everything from scratch
    (nothing carries over from the supervisor process except what's on
    disk -- config.yaml, the cache, the cost-tracker state file, and the
    phase's own log/state files), process up to --max-new episodes/pairs,
    then exit. Exiting is what reclaims this chunk's leaked tmpfs memory."""
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    if args.backend:
        cfg["env"]["backend"] = args.backend
    if args.logs_dir:
        cfg["paths"]["logs_dir"] = args.logs_dir
        pathlib.Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    memories = _build_memories(cfg)
    task_source = build_task_source(cfg)
    llm_client, _cost_tracker = _build_llm_client_and_tracker(cfg, args)

    if args.phase == "logging":
        log_path = pathlib.Path(cfg["paths"]["logs_dir"]) / "main_logging.jsonl"
        n_new = run_logging_phase_checkpointed(
            task_source, llm_client, memories, cfg["retrieval"]["M"], cfg["retrieval"]["propensity_min"],
            cfg["retrieval"]["propensity_max"], cfg["env"]["max_steps"], args.n_episodes, log_path, max_new=args.max_new,
        )
    else:
        if not args.memory_id:
            raise SystemExit("ground_truth phase requires --memory-id")
        log_path = pathlib.Path(cfg["paths"]["logs_dir"]) / f"ground_truth_{args.memory_id}.jsonl"
        state_path = pathlib.Path(cfg["paths"]["logs_dir"]) / f"ground_truth_{args.memory_id}_state.json"
        n_new = run_ground_truth_phase_checkpointed(
            task_source, llm_client, memories, cfg["retrieval"]["M"], cfg["retrieval"]["propensity_min"],
            cfg["retrieval"]["propensity_max"], cfg["env"]["max_steps"], memory_id=args.memory_id, n_pairs=args.n_pairs,
            log_path=log_path, state_path=state_path, max_new=args.max_new,
        )
    print(f"chunk done: {n_new} new episode(s)/pair(s)")


def _count_done(cfg: dict, args: argparse.Namespace) -> int:
    logs_dir = args.logs_dir or cfg["paths"]["logs_dir"]
    if args.phase == "logging":
        log_path = pathlib.Path(logs_dir) / "main_logging.jsonl"
        if not log_path.exists():
            return 0
        with open(log_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    else:
        log_path = pathlib.Path(logs_dir) / f"ground_truth_{args.memory_id}.jsonl"
        if not log_path.exists():
            return 0
        with open(log_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip()) // 2


def _supervise(args: argparse.Namespace) -> None:
    cfg = config_mod.load_config()
    config_mod.ensure_dirs(cfg)
    target = args.n_episodes if args.phase == "logging" else args.n_pairs

    worker_argv = [sys.executable, "-m", "alfworld_pilot.run_chunked", "--worker", args.phase]
    worker_argv += ["--n-episodes", str(args.n_episodes), "--n-pairs", str(args.n_pairs or 0)]
    worker_argv += ["--max-new", str(args.chunk_size), "--llm", args.llm]
    if args.memory_id:
        worker_argv += ["--memory-id", args.memory_id]
    if args.backend:
        worker_argv += ["--backend", args.backend]
    if args.logs_dir:
        worker_argv += ["--logs-dir", args.logs_dir]
    if args.model_id:
        worker_argv += ["--model-id", args.model_id]
    if args.pricing_input is not None:
        worker_argv += ["--pricing-input", str(args.pricing_input)]
    if args.pricing_output is not None:
        worker_argv += ["--pricing-output", str(args.pricing_output)]

    for attempt in range(args.max_restarts):
        done = _count_done(cfg, args)
        if done >= target:
            print(f"Done: {done}/{target}")
            return
        print(f"[supervisor] attempt {attempt + 1}: {done}/{target} done, launching a chunk subprocess (up to {args.chunk_size} more)...")
        result = subprocess.run(worker_argv)
        if result.returncode != 0:
            print(f"[supervisor] worker exited with code {result.returncode} -- resuming (this is expected on a crash mid-chunk, not necessarily an error)")

    raise RuntimeError(f"Gave up after {args.max_restarts} chunk attempts, only {_count_done(cfg, args)}/{target} done.")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", metavar="PHASE", choices=["logging", "ground_truth"], help=argparse.SUPPRESS)
    p.add_argument("phase", nargs="?", choices=["logging", "ground_truth"])
    p.add_argument("--n-episodes", type=int, default=0)
    p.add_argument("--n-pairs", type=int, default=0)
    p.add_argument("--memory-id", default=None)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--max-new", type=int, default=None, help=argparse.SUPPRESS)  # worker-only
    p.add_argument("--max-restarts", type=int, default=DEFAULT_MAX_RESTARTS)
    p.add_argument("--llm", choices=["mock", "real"], default="real")
    p.add_argument("--backend", choices=["mock", "real"], default=None, help="override config.yaml's env.backend")
    p.add_argument("--logs-dir", default=None, help="override config.yaml's paths.logs_dir (tests use this to avoid touching real run logs)")
    p.add_argument("--model-id", default=None)
    p.add_argument("--pricing-input", type=float, default=None)
    p.add_argument("--pricing-output", type=float, default=None)
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    if args.worker:
        args.phase = args.worker
        _run_one_chunk(args)
    else:
        if not args.phase:
            raise SystemExit("phase (logging | ground_truth) is required")
        _supervise(args)


if __name__ == "__main__":
    main()
