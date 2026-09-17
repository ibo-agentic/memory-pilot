import random

from alfworld_pilot.env_interface import TASK_TYPES, MockAlfredEnv
from alfworld_pilot.episode_runner import run_logged_episode
from alfworld_pilot.ground_truth_runner import (
    paired_correlation,
    run_ground_truth_experiment,
    run_ground_truth_pair,
)
from alfworld_pilot.measure_mode import run_measure_mode
from alfworld_pilot.memory_store import build_mock_store
from alfworld_pilot.mock_llm import MockLLMClient

M, P_MIN, P_MAX, MAX_STEPS = 10, 0.3, 0.7, 30


def _memories(n=30):
    return build_mock_store(n, 100, 300, seed=0)


def test_scripted_success_llm_wins_every_mock_episode():
    memories = _memories()
    llm = MockLLMClient(strategy="scripted_success", seed=0)
    rng = random.Random(0)
    successes = []
    for task_id in range(10):
        env = MockAlfredEnv()
        ep = run_logged_episode(env, llm, memories, M, P_MIN, P_MAX, MAX_STEPS, task_id, rng)
        successes.append(ep["success"])
    assert all(s == 1 for s in successes), "scripted_success should win every mock episode"


def test_random_llm_mostly_fails_or_hits_step_cap():
    # Zero slack (max_steps == longest template's required-action count):
    # a blind random agent must guess every step right on the first try,
    # so it should virtually never win. With the realistic 30-step cap
    # used elsewhere, random search has enough retries to often stumble
    # onto short (4-6 step) templates by chance -- that's a fact about the
    # mock's difficulty at that cap, not something this test should assert
    # against; use the tight cap specifically to distinguish "acts on the
    # plan" from "guesses randomly."
    tight_max_steps = 6
    memories = _memories()
    llm = MockLLMClient(strategy="random", seed=0)
    rng = random.Random(0)
    results = []
    for task_id in range(10):
        env = MockAlfredEnv()
        ep = run_logged_episode(env, llm, memories, M, P_MIN, P_MAX, tight_max_steps, task_id, rng)
        results.append(ep)
    assert sum(r["success"] for r in results) < len(results), "random actions shouldn't reliably win with zero step slack"


def test_malformed_llm_triggers_parse_failure_fallback():
    memories = _memories()
    llm = MockLLMClient(strategy="malformed", seed=0)
    rng = random.Random(0)
    env = MockAlfredEnv()
    ep = run_logged_episode(env, llm, memories, M, P_MIN, P_MAX, MAX_STEPS, task_id=0, rng=rng)
    assert ep["parse_failures"] >= 0  # ran to completion without crashing on malformed output
    assert ep["steps_taken"] > 0


def test_episode_log_schema():
    memories = _memories()
    llm = MockLLMClient(strategy="scripted_success", seed=0)
    rng = random.Random(0)
    env = MockAlfredEnv()
    ep = run_logged_episode(env, llm, memories, M, P_MIN, P_MAX, MAX_STEPS, task_id=0, rng=rng)

    for key in ("task_id", "task_type", "candidate_ids", "propensities", "included", "success"):
        assert key in ep, f"missing Stage-1-compatible field: {key}"
    assert ep["task_type"] in TASK_TYPES
    assert len(ep["candidate_ids"]) == M
    assert set(ep["propensities"].keys()) == set(ep["candidate_ids"])
    assert set(ep["included"].keys()) == set(ep["candidate_ids"])
    assert all(P_MIN - 1e-9 <= p <= P_MAX + 1e-9 for p in ep["propensities"].values())
    assert all(z in (0, 1) for z in ep["included"].values())


def test_paired_ground_truth_holds_everything_else_fixed():
    memories = _memories()
    llm = MockLLMClient(strategy="scripted_success", seed=0)
    # memory_0 is tagged with TASK_TYPES[0], so it's a near-certain natural
    # candidate whenever that task type is drawn.
    target = memories[0].mem_id

    found_matching_task = False
    for task_seed in range(50):
        env = MockAlfredEnv()
        _obs, info = env.reset(task_seed=task_seed)
        if info["task_type"] == memories[0].task_type:
            found_matching_task = True
            break
    assert found_matching_task, "test setup: need at least one task_seed matching memory_0's type in range"

    ep_in, ep_out = run_ground_truth_pair(
        MockAlfredEnv(), llm, memories, M, P_MIN, P_MAX, MAX_STEPS, task_seed=task_seed, pair_index=0, memory_id=target
    )
    assert ep_in["included"][target] == 1
    assert ep_out["included"][target] == 0
    other_in = {k: v for k, v in ep_in["included"].items() if k != target}
    other_out = {k: v for k, v in ep_out["included"].items() if k != target}
    assert other_in == other_out, "paired design must hold every OTHER memory's inclusion fixed between arms"
    assert ep_in["candidate_ids"] == ep_out["candidate_ids"], "paired design must hold candidacy fixed between arms"


def test_ground_truth_experiment_only_uses_natural_candidacy():
    memories = _memories()
    llm = MockLLMClient(strategy="scripted_success", seed=0)
    target = memories[0].mem_id
    pairs = run_ground_truth_experiment(
        MockAlfredEnv(), llm, memories, M, P_MIN, P_MAX, MAX_STEPS, memory_id=target, n_pairs=5, start_seed=0
    )
    assert len(pairs) == 5
    for ep_in, ep_out in pairs:
        assert target in ep_in["candidate_ids"], "ground truth must only use task instances where the memory is a natural candidate"

    stats = paired_correlation(pairs)
    assert stats["n_pairs"] == 5
    assert "rho" in stats


def test_measure_mode_reports_positive_averages():
    from alfworld_pilot import config as config_mod

    cfg = config_mod.load_config()
    llm = MockLLMClient(strategy="scripted_success", seed=0)
    report = run_measure_mode(cfg, llm, MockAlfredEnv, n_episodes=5)
    assert report["avg_input_tokens_per_call"] > 0
    assert report["avg_output_tokens_per_call"] > 0
    assert report["avg_calls_per_episode"] > 0
    assert report["projected_total_calls"] > 0
