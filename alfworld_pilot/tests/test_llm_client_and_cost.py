import pathlib

import pytest

from alfworld_pilot.cache import LLMCache
from alfworld_pilot.cost_tracker import CostCapExceeded, CostTracker
from alfworld_pilot.determinism_check import check_determinism
from alfworld_pilot.llm_client import LLMResponse


def test_cache_roundtrip(tmp_path: pathlib.Path):
    cache = LLMCache(tmp_path / "cache")
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert cache.get(payload) is None
    cache.put(payload, {"text": "hello", "input_tokens": 3, "output_tokens": 1})
    hit = cache.get(payload)
    assert hit["response"]["text"] == "hello"


def test_cache_distinguishes_different_payloads(tmp_path: pathlib.Path):
    cache = LLMCache(tmp_path / "cache")
    p1 = {"model": "m", "messages": [{"role": "user", "content": "a"}]}
    p2 = {"model": "m", "messages": [{"role": "user", "content": "b"}]}
    cache.put(p1, {"text": "A", "input_tokens": 1, "output_tokens": 1})
    assert cache.get(p2) is None


def test_cost_tracker_hard_cap_blocks_before_spending():
    tracker = CostTracker(hard_cap_usd=0.01, price_per_million_input=1.0, price_per_million_output=1.0)
    tracker.check_before_call(estimated_input_tokens=1000, estimated_output_tokens=100)  # cheap enough, should pass
    with pytest.raises(CostCapExceeded):
        tracker.check_before_call(estimated_input_tokens=100_000, estimated_output_tokens=100_000)


def test_cost_tracker_records_cumulative_spend():
    tracker = CostTracker(hard_cap_usd=10.0, price_per_million_input=1.0, price_per_million_output=2.0)
    cost1 = tracker.record_call(input_tokens=1_000_000, output_tokens=500_000)
    assert cost1 == pytest.approx(1.0 + 1.0)
    assert tracker.total_spend_usd == pytest.approx(2.0)
    tracker.record_call(input_tokens=1_000_000, output_tokens=0)
    assert tracker.total_spend_usd == pytest.approx(3.0)
    assert tracker.n_calls == 2


def test_cost_tracker_cache_hits_are_free():
    tracker = CostTracker(hard_cap_usd=10.0, price_per_million_input=1.0, price_per_million_output=1.0)
    tracker.record_cache_hit()
    tracker.record_cache_hit()
    assert tracker.total_spend_usd == 0.0
    assert tracker.n_cache_hits == 2


def test_cost_tracker_state_survives_a_simulated_restart(tmp_path: pathlib.Path):
    state_path = tmp_path / "cost_state.json"

    tracker1 = CostTracker(hard_cap_usd=10.0, price_per_million_input=1.0, price_per_million_output=1.0, state_path=state_path)
    tracker1.record_call(input_tokens=1_000_000, output_tokens=0)  # $1.00
    assert tracker1.total_spend_usd == pytest.approx(1.0)

    # Simulate a crash + restart: a brand-new CostTracker pointed at the same
    # state_path must pick up where the old one left off, not reset to $0.
    tracker2 = CostTracker(hard_cap_usd=10.0, price_per_million_input=1.0, price_per_million_output=1.0, state_path=state_path)
    assert tracker2.total_spend_usd == pytest.approx(1.0)
    assert tracker2.n_calls == 1

    tracker2.record_call(input_tokens=1_000_000, output_tokens=0)  # +$1.00 = $2.00 cumulative
    assert tracker2.total_spend_usd == pytest.approx(2.0)

    # The cap must reflect spend from BEFORE this restart too, not just this process's calls.
    tracker3 = CostTracker(hard_cap_usd=2.5, price_per_million_input=1.0, price_per_million_output=1.0, state_path=state_path)
    assert tracker3.total_spend_usd == pytest.approx(2.0)
    with pytest.raises(CostCapExceeded):
        tracker3.check_before_call(estimated_input_tokens=1_000_000, estimated_output_tokens=0)  # would hit $3.00 > $2.50 cap


class _StubClient:
    """Returns fixed, possibly-differing responses per call -- used to test
    check_determinism's logic directly rather than relying on a mock LLM's
    statistical behavior (which would make the test flaky)."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._i = 0

    def complete_no_cache(self, messages, stop=None):
        text = self._responses[self._i]
        self._i += 1
        return LLMResponse(text=text, input_tokens=10, output_tokens=2, cached=False)


def test_determinism_check_detects_identical_responses():
    client = _StubClient(["Thought: x\nAction: go", "Thought: x\nAction: go"])
    result = check_determinism(client, [{"role": "user", "content": "hi"}])
    assert result["holds"] is True


def test_determinism_check_detects_differing_responses():
    client = _StubClient(["Thought: x\nAction: go", "Thought: y\nAction: stay"])
    result = check_determinism(client, [{"role": "user", "content": "hi"}])
    assert result["holds"] is False


def test_openrouter_client_rejects_null_model_id(monkeypatch, tmp_path):
    from alfworld_pilot.llm_client import OpenRouterClient

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake-key-not-real")
    with pytest.raises(ValueError, match="model_id"):
        OpenRouterClient(
            model_id=None, base_url="https://openrouter.ai/api/v1", api_key_env_var="OPENROUTER_API_KEY",
            reasoning_enabled=False, temperature=0.0, max_output_tokens=100,
            cache=LLMCache(tmp_path / "cache"), cost_tracker=CostTracker(10.0, 1.0, 1.0),
        )


def test_openrouter_client_rejects_latest_alias(monkeypatch, tmp_path):
    from alfworld_pilot.llm_client import OpenRouterClient

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake-key-not-real")
    with pytest.raises(ValueError, match="latest"):
        OpenRouterClient(
            model_id="some-provider/some-model-latest", base_url="https://openrouter.ai/api/v1", api_key_env_var="OPENROUTER_API_KEY",
            reasoning_enabled=False, temperature=0.0, max_output_tokens=100,
            cache=LLMCache(tmp_path / "cache"), cost_tracker=CostTracker(10.0, 1.0, 1.0),
        )


def test_openrouter_client_request_payload_disables_reasoning(monkeypatch, tmp_path):
    from alfworld_pilot.llm_client import OpenRouterClient

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake-key-not-real")
    client = OpenRouterClient(
        model_id="some-provider/some-pinned-model-v1", base_url="https://openrouter.ai/api/v1", api_key_env_var="OPENROUTER_API_KEY",
        reasoning_enabled=False, temperature=0.0, max_output_tokens=100,
        cache=LLMCache(tmp_path / "cache"), cost_tracker=CostTracker(10.0, 1.0, 1.0),
    )
    payload = client._request_payload([{"role": "user", "content": "hi"}], stop=None)
    assert payload["reasoning"] == {"enabled": False}
    assert payload["model"] == "some-provider/some-pinned-model-v1"
    assert payload["temperature"] == 0.0
