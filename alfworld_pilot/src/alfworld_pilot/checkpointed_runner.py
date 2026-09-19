"""Resumable drivers for the real Part C run: if the process crashes or is
interrupted, restarting picks up from the last COMPLETED episode/pair
instead of starting over.

Three things need to survive a restart for this to actually be safe, not
just convenient:
  1. Which episodes/pairs are already done -- this module's job (JSONL
     append-and-flush logs; resume point = line count / persisted state).
  2. Cumulative spend vs. the hard cap -- CostTracker's `state_path`
     (cost_tracker.py) persists this; construct it with the SAME state_path
     across restarts.
  3. The LLM response cache -- already file-per-request on disk
     (cache.py), nothing extra needed here.

Each episode's own randomization (which memories are candidates, which get
included) is seeded as a pure function of its task_id (main logging) or
(memory_id, pair_index, task_seed) (ground truth, via
ground_truth_runner._pair_seed_material, unchanged) rather than one shared
RNG stream advancing across the whole run. A shared stream would make
episode k's randomization depend on how many episodes ran before it in
THIS process -- fine for one uninterrupted run, but wrong the moment a
restart resumes partway through, since the new process's stream starts
fresh from 0 while episode k no longer means "the k-th draw from a fresh
stream". Pure-function-of-task_id seeding sidesteps that entirely.

Both runners also accept `max_new`, capping how many NEW episodes/pairs a
single call processes before returning even if the overall target isn't
reached yet. This exists for `run_chunked.py`'s subprocess-per-chunk
supervisor: real ALFWorld leaks ~32.5MB of tmpfs-backed memory (an
un-dlclose'd `fast_downward` shared-library copy) on every load of a NEW
game, reclaimed only when the process exits (see run_chunked.py's
docstring for the full story) -- `max_new` is what lets a supervisor bound
each subprocess's lifetime to a safe number of game loads.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random

from .episode_runner import run_logged_episode
from .ground_truth_runner import TaskSource, _is_natural_candidate, run_ground_truth_pair


def _seed_for_task(task_id: int) -> int:
    key = f"main_logging|{task_id}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) & 0xFFFFFFFF


def _count_completed_lines(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_logging_phase_checkpointed(
    task_source: TaskSource,
    llm_client,
    memories: list,
    m: int,
    propensity_min: float,
    propensity_max: float,
    max_steps: int,
    n_episodes: int,
    log_path: str | pathlib.Path,
    max_new: int | None = None,
) -> int:
    """Appends one JSON line per completed episode to log_path, flushing
    after each -- safe to interrupt at any point. Resume point is just the
    number of complete lines already in log_path. Returns the number of
    NEWLY completed episodes this call made (0 if already fully done, or
    capped at max_new if given)."""
    log_path = pathlib.Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = _count_completed_lines(log_path)
    stop_at = n_episodes if max_new is None else min(n_episodes, start + max_new)

    n_new = 0
    with open(log_path, "a", encoding="utf-8") as f:
        for task_id in range(start, stop_at):
            env = task_source.build_env(task_id)
            try:
                rng = random.Random(_seed_for_task(task_id))
                ep = run_logged_episode(
                    env, llm_client, memories, m, propensity_min, propensity_max, max_steps,
                    task_id=task_id, rng=rng,
                )
            finally:
                env.close()
            f.write(json.dumps(ep) + "\n")
            f.flush()
            n_new += 1
    return n_new


def _load_gt_state(state_path: pathlib.Path, start_seed: int) -> tuple[int, int]:
    if not state_path.exists():
        return 0, start_seed
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return state["pairs_found"], state["next_seed"]


def _save_gt_state(state_path: pathlib.Path, pairs_found: int, next_seed: int) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"pairs_found": pairs_found, "next_seed": next_seed}, f)
    tmp.replace(state_path)


def run_ground_truth_phase_checkpointed(
    task_source: TaskSource,
    llm_client,
    memories: list,
    m: int,
    propensity_min: float,
    propensity_max: float,
    max_steps: int,
    memory_id: str,
    n_pairs: int,
    log_path: str | pathlib.Path,
    state_path: str | pathlib.Path,
    start_seed: int = 0,
    max_probe_tries: int = 200_000,
    max_new: int | None = None,
) -> int:
    """Same resumability contract as run_logging_phase_checkpointed, but for
    one memory's forced-in/forced-out pairs: state_path tracks how many
    pairs are already found and which task_seed to resume probing from
    (candidacy probes that reject a seed are cheap and NOT separately
    checkpointed -- only accepted pairs and the seed cursor persist).
    log_path gets 2 lines per accepted pair (forced_in, then forced_out).
    Returns the number of NEWLY completed pairs this call made (capped at
    max_new if given -- an unmet max_probe_tries still raises even under a
    max_new cap, since that means real trouble finding candidates, not just
    "this chunk is done")."""
    log_path = pathlib.Path(log_path)
    state_path = pathlib.Path(state_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pairs_found, seed = _load_gt_state(state_path, start_seed)
    target = n_pairs if max_new is None else min(n_pairs, pairs_found + max_new)
    n_new = 0
    tries = 0
    with open(log_path, "a", encoding="utf-8") as f:
        while pairs_found < target and tries < max_probe_tries:
            if _is_natural_candidate(task_source, memories, memory_id, seed, pairs_found, m, propensity_min, propensity_max):
                ep_in, ep_out = run_ground_truth_pair(
                    task_source, llm_client, memories, m, propensity_min, propensity_max, max_steps,
                    task_seed=seed, pair_index=pairs_found, memory_id=memory_id,
                )
                f.write(json.dumps(ep_in) + "\n")
                f.write(json.dumps(ep_out) + "\n")
                f.flush()
                pairs_found += 1
                n_new += 1
                seed += 1
                _save_gt_state(state_path, pairs_found, seed)
            else:
                seed += 1
            tries += 1

    if pairs_found < target:
        raise RuntimeError(
            f"Only found {pairs_found}/{target} pairs for {memory_id!r} (overall target {n_pairs}) after "
            f"{tries} probe tries this call (resumable -- rerun to keep trying from seed={seed})."
        )
    return n_new
