"""Integration test: actually spawns subprocess workers (mock backend, mock
LLM, zero cost) to prove the chunked-subprocess supervisor mechanism itself
works end to end, not just the in-process checkpointed_runner functions it
wraps (those have their own direct-call tests in test_checkpointed_runner.py).
"""

import json
import os
import pathlib
import subprocess
import sys

_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")


def _subprocess_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC_DIR
    return env


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_supervisor_completes_logging_phase_across_multiple_chunk_subprocesses(tmp_path: pathlib.Path):
    logs_dir = tmp_path / "logs"
    n_episodes = 5
    chunk_size = 2  # forces at least 3 separate worker subprocesses for 5 episodes

    result = subprocess.run(
        [
            sys.executable, "-m", "alfworld_pilot.run_chunked", "logging",
            "--n-episodes", str(n_episodes), "--chunk-size", str(chunk_size),
            "--llm", "mock", "--backend", "mock", "--logs-dir", str(logs_dir),
        ],
        env=_subprocess_env(),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "Done: 5/5" in result.stdout

    episodes = _read_jsonl(logs_dir / "main_logging.jsonl")
    assert len(episodes) == n_episodes
    assert [ep["task_id"] for ep in episodes] == list(range(n_episodes))


def test_supervisor_completes_ground_truth_phase_across_multiple_chunk_subprocesses(tmp_path: pathlib.Path):
    logs_dir = tmp_path / "logs"
    n_pairs = 4
    chunk_size = 1  # forces 4 separate worker subprocesses

    result = subprocess.run(
        [
            sys.executable, "-m", "alfworld_pilot.run_chunked", "ground_truth",
            "--memory-id", "mem_0", "--n-pairs", str(n_pairs), "--chunk-size", str(chunk_size),
            "--llm", "mock", "--backend", "mock", "--logs-dir", str(logs_dir),
        ],
        env=_subprocess_env(),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert f"Done: {n_pairs}/{n_pairs}" in result.stdout

    pairs = _read_jsonl(logs_dir / "ground_truth_mem_0.jsonl")
    assert len(pairs) == n_pairs * 2
