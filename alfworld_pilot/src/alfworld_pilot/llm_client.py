"""OpenRouter LLM client (OpenAI-compatible endpoint), wired through the
disk cache and the hard cost cap. `react_agent.py` depends only on the
`LLMClient` Protocol below, so swapping in `mock_llm.MockLLMClient` for
testing requires no other code changes.

Model id must be an exact pinned string from config (llm.model_id) -- never
"latest" or a bare provider default. Reasoning is disabled by sending
{"reasoning": {"enabled": false}} in extra_body (OpenRouter's documented
unified reasoning control); NOTE some models have mandatory reasoning and
will reject this with a 400 error -- that's surfaced as an exception, not
silently ignored, since Part C needs to see which models do this.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from .cache import LLMCache
from .cost_tracker import CostTracker


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cached: bool
    finish_reason: str | None = None


class LLMClient(Protocol):
    def complete(self, messages: list[dict], stop: list[str] | None = None) -> LLMResponse: ...


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough pre-call estimate (chars/4) for the cost-cap pre-check, before
    we know the real token count from the API's usage field."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return max(1, total_chars // 4)


class OpenRouterClient:
    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key_env_var: str,
        reasoning_enabled: bool,
        temperature: float,
        max_output_tokens: int,
        cache: LLMCache,
        cost_tracker: CostTracker,
        seed: int | None = None,
    ):
        if model_id is None:
            raise ValueError(
                "llm.model_id is null in config.yaml -- set an exact pinned model id "
                "(never a 'latest' alias) before making real calls. See Part C."
            )
        if "latest" in model_id.lower():
            raise ValueError(f"model_id '{model_id}' looks like a 'latest' alias, not a pinned id -- refusing.")

        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise RuntimeError(f"{api_key_env_var} is not set (check .env)")

        import openai  # local import: only needed when actually instantiating a real client

        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
        self.reasoning_enabled = reasoning_enabled
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        self.cache = cache
        self.cost_tracker = cost_tracker

    def _request_payload(self, messages: list[dict], stop: list[str] | None) -> dict:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stop": stop,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        payload["reasoning"] = {"enabled": self.reasoning_enabled}
        return payload

    def complete(self, messages: list[dict], stop: list[str] | None = None, _skip_cache_read: bool = False) -> LLMResponse:
        payload = self._request_payload(messages, stop)

        cached = None if _skip_cache_read else self.cache.get(payload)
        if cached is not None:
            self.cost_tracker.record_cache_hit()
            resp = cached["response"]
            return LLMResponse(
                text=resp["text"],
                input_tokens=resp["input_tokens"],
                output_tokens=resp["output_tokens"],
                cached=True,
                finish_reason=resp.get("finish_reason"),
            )

        self.cost_tracker.check_before_call(
            estimated_input_tokens=_estimate_tokens(messages), estimated_output_tokens=self.max_output_tokens
        )

        extra_body = {"reasoning": {"enabled": self.reasoning_enabled}}
        kwargs = dict(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            extra_body=extra_body,
        )
        if stop:
            kwargs["stop"] = stop
        if self.seed is not None:
            kwargs["seed"] = self.seed

        completion = self._client.chat.completions.create(**kwargs)

        text = completion.choices[0].message.content or ""
        finish_reason = completion.choices[0].finish_reason
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens

        self.cost_tracker.record_call(input_tokens, output_tokens)
        self.cache.put(payload, {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens, "finish_reason": finish_reason})

        return LLMResponse(text=text, input_tokens=input_tokens, output_tokens=output_tokens, cached=False, finish_reason=finish_reason)

    def complete_no_cache(self, messages: list[dict], stop: list[str] | None = None) -> LLMResponse:
        """Forces a fresh real call even if an identical request is already
        cached -- used only for the determinism probe (a real duplicate
        call is the only way to verify the provider is actually
        deterministic, not just that our cache is working)."""
        return self.complete(messages, stop, _skip_cache_read=True)
