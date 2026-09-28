"""Secondary outcome (2026-09-29): success_within_N, a post-hoc truncation
of an already-logged episode's trajectory -- pre-registered in
alfworld_pilot/README.md BEFORE this was implemented, predicting memory
effects are larger under success_within_25 than the primary (max_steps=50)
outcome, because react_agent.py's prompt never shows the agent its step
budget or remaining steps (confirmed by direct inspection before writing
this module -- see the README section for the exact code locations checked).

Deliberately NOT a new logging field: `success` and `steps_taken` are
already logged by episode_runner.run_logged_episode (and by the paid run's
own historical logs), and success_within_N is a pure function of those two
-- so this works retroactively on any already-logged episode, this
project's or the paid run's, not just future runs.

Deliberately NOT a change to src/memory_ope/estimators/: those modules read
ep["success"] by name (memory_worth.py, ips.py, etc.) and are shared,
paid-run-validated code that this replication should not touch. Instead,
truncated_success() returns a COPY of the episode list with "success"
replaced by the truncated outcome, so any existing estimator runs
unmodified against it -- just call it with a different list.
"""

from __future__ import annotations


def truncated_success(episodes: list[dict], step_cap: int) -> list[dict]:
    """Returns a shallow copy of `episodes` with each episode's "success"
    field replaced by whether it succeeded within `step_cap` steps of its
    ALREADY-LOGGED trajectory (success_within_N). An episode that took more
    than step_cap steps to succeed becomes a "failure" under this outcome,
    regardless of what actually happened after step_cap -- this is exactly
    the point (a retroactive re-scoring of an unchanged trajectory, not a
    re-run), not an approximation to be refined.

    Every other field is passed through unchanged (a shallow copy per
    episode, not a deep copy -- callers must not mutate nested structures
    like "steps" in place expecting it to be independent per truncation)."""
    if step_cap < 1:
        raise ValueError(f"step_cap must be >= 1, got {step_cap}")
    out = []
    for ep in episodes:
        ep2 = dict(ep)
        ep2["success"] = int(bool(ep["success"]) and ep["steps_taken"] <= step_cap)
        out.append(ep2)
    return out
