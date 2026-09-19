import json
import pathlib

import pytest

from alfworld_pilot.checkpointed_runner import (
    run_ground_truth_phase_checkpointed,
    run_logging_phase_checkpointed,
)
from alfworld_pilot.ground_truth_runner import MockTaskSource
from alfworld_pilot.memory_store import build_mock_store
from alfworld_pilot.mock_llm import MockLLMClient

M, P_MIN, P_MAX, MAX_STEPS = 10, 0.3, 0.7, 30


def _memories(n=30):
    return build_mock_store(n, 100, 300, seed=0)


class _CrashAfterNCallsClient:
    """Wraps a real LLMClient but raises after a fixed number of .complete()
    calls -- simulates a process crash mid-run without actually killing the
    process, so the test can inspect what was (and wasn't) persisted."""

    def __init__(self, inner, crash_after: int):
        self._inner = inner
        self._crash_after = crash_after
        self._n_calls = 0

    def complete(self, messages, stop=None):
        if self._n_calls >= self._crash_after:
            raise RuntimeError("simulated crash")
        self._n_calls += 1
        return self._inner.complete(messages, stop)


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_logging_phase_checkpointed_resumes_after_simulated_crash(tmp_path: pathlib.Path):
    memories = _memories()
    log_path = tmp_path / "logging.jsonl"
    n_episodes = 6

    # Uninterrupted reference run, for comparison against the crash+resume run.
    reference_log = tmp_path / "reference.jsonl"
    run_logging_phase_checkpointed(
        MockTaskSource(), MockLLMClient(strategy="scripted_success", seed=0), memories,
        M, P_MIN, P_MAX, MAX_STEPS, n_episodes, reference_log,
    )
    reference_episodes = _read_jsonl(reference_log)
    assert len(reference_episodes) == n_episodes

    # Crash partway through.
    crashing_client = _CrashAfterNCallsClient(MockLLMClient(strategy="scripted_success", seed=0), crash_after=10)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_logging_phase_checkpointed(
            MockTaskSource(), crashing_client, memories, M, P_MIN, P_MAX, MAX_STEPS, n_episodes, log_path,
        )
    partial = _read_jsonl(log_path)
    assert 0 < len(partial) < n_episodes, "crash should have landed mid-run, not before or after all episodes"

    # Resume with a fresh (non-crashing) client -- must complete the rest, not redo completed episodes.
    n_new = run_logging_phase_checkpointed(
        MockTaskSource(), MockLLMClient(strategy="scripted_success", seed=0), memories,
        M, P_MIN, P_MAX, MAX_STEPS, n_episodes, log_path,
    )
    assert n_new == n_episodes - len(partial)

    final = _read_jsonl(log_path)
    assert len(final) == n_episodes
    assert [ep["task_id"] for ep in final] == list(range(n_episodes)), "no duplicate or skipped task_ids"

    # Per-task_id seeding means the crash+resume run must reproduce the SAME
    # episodes as the uninterrupted reference run, not just the same count.
    assert final == reference_episodes


def test_ground_truth_phase_checkpointed_resumes_after_simulated_crash(tmp_path: pathlib.Path):
    memories = _memories()
    target = memories[0].mem_id
    log_path = tmp_path / "gt.jsonl"
    state_path = tmp_path / "gt_state.json"
    n_pairs = 3

    crashing_client = _CrashAfterNCallsClient(MockLLMClient(strategy="scripted_success", seed=0), crash_after=15)
    with pytest.raises(RuntimeError):
        run_ground_truth_phase_checkpointed(
            MockTaskSource(), crashing_client, memories, M, P_MIN, P_MAX, MAX_STEPS,
            memory_id=target, n_pairs=n_pairs, log_path=log_path, state_path=state_path,
        )
    partial = _read_jsonl(log_path)
    assert len(partial) % 2 == 0, "log must only ever contain complete (forced_in, forced_out) pairs"
    partial_pairs = len(partial) // 2
    assert 0 <= partial_pairs < n_pairs

    n_new = run_ground_truth_phase_checkpointed(
        MockTaskSource(), MockLLMClient(strategy="scripted_success", seed=0), memories,
        M, P_MIN, P_MAX, MAX_STEPS, memory_id=target, n_pairs=n_pairs, log_path=log_path, state_path=state_path,
    )
    assert n_new == n_pairs - partial_pairs

    final = _read_jsonl(log_path)
    assert len(final) == n_pairs * 2
    for i in range(n_pairs):
        ep_in, ep_out = final[2 * i], final[2 * i + 1]
        assert ep_in["ground_truth_arm"] == "forced_in"
        assert ep_out["ground_truth_arm"] == "forced_out"
        assert ep_in["ground_truth_pair_index"] == i
        assert ep_in["included"][target] == 1
        assert ep_out["included"][target] == 0

    # Calling again once n_pairs is already satisfied must be a no-op, not an error or a redo.
    n_new_again = run_ground_truth_phase_checkpointed(
        MockTaskSource(), MockLLMClient(strategy="scripted_success", seed=0), memories,
        M, P_MIN, P_MAX, MAX_STEPS, memory_id=target, n_pairs=n_pairs, log_path=log_path, state_path=state_path,
    )
    assert n_new_again == 0
    assert _read_jsonl(log_path) == final


def test_logging_phase_max_new_caps_one_call_and_chunks_compose(tmp_path: pathlib.Path):
    memories = _memories()
    log_path = tmp_path / "logging.jsonl"
    n_episodes = 7

    n1 = run_logging_phase_checkpointed(
        MockTaskSource(), MockLLMClient(strategy="scripted_success", seed=0), memories,
        M, P_MIN, P_MAX, MAX_STEPS, n_episodes, log_path, max_new=3,
    )
    assert n1 == 3
    assert len(_read_jsonl(log_path)) == 3

    n2 = run_logging_phase_checkpointed(
        MockTaskSource(), MockLLMClient(strategy="scripted_success", seed=0), memories,
        M, P_MIN, P_MAX, MAX_STEPS, n_episodes, log_path, max_new=3,
    )
    assert n2 == 3
    assert len(_read_jsonl(log_path)) == 6

    # Final chunk: only 1 episode left, even though max_new=3 would allow more.
    n3 = run_logging_phase_checkpointed(
        MockTaskSource(), MockLLMClient(strategy="scripted_success", seed=0), memories,
        M, P_MIN, P_MAX, MAX_STEPS, n_episodes, log_path, max_new=3,
    )
    assert n3 == 1
    final = _read_jsonl(log_path)
    assert len(final) == n_episodes
    assert [ep["task_id"] for ep in final] == list(range(n_episodes))


def test_ground_truth_phase_max_new_caps_one_call_and_chunks_compose(tmp_path: pathlib.Path):
    memories = _memories()
    target = memories[0].mem_id
    log_path = tmp_path / "gt.jsonl"
    state_path = tmp_path / "gt_state.json"
    n_pairs = 5

    n1 = run_ground_truth_phase_checkpointed(
        MockTaskSource(), MockLLMClient(strategy="scripted_success", seed=0), memories,
        M, P_MIN, P_MAX, MAX_STEPS, memory_id=target, n_pairs=n_pairs,
        log_path=log_path, state_path=state_path, max_new=2,
    )
    assert n1 == 2
    assert len(_read_jsonl(log_path)) == 4  # 2 pairs x 2 episodes

    n2 = run_ground_truth_phase_checkpointed(
        MockTaskSource(), MockLLMClient(strategy="scripted_success", seed=0), memories,
        M, P_MIN, P_MAX, MAX_STEPS, memory_id=target, n_pairs=n_pairs,
        log_path=log_path, state_path=state_path, max_new=2,
    )
    assert n2 == 2

    n3 = run_ground_truth_phase_checkpointed(
        MockTaskSource(), MockLLMClient(strategy="scripted_success", seed=0), memories,
        M, P_MIN, P_MAX, MAX_STEPS, memory_id=target, n_pairs=n_pairs,
        log_path=log_path, state_path=state_path, max_new=2,
    )
    assert n3 == 1  # only 1 pair left, even though max_new=2 would allow more
    assert len(_read_jsonl(log_path)) == n_pairs * 2
