"""Fake LLM clients: no network calls, zero cost, deterministic given a
seed. They parse the SAME prompt text a real model would see (never the
environment's internal state directly) so they exercise react_agent.py's
parsing/formatting exactly as a real model's output would.

Three strategies, to exercise different code paths in the mock pipeline test:
  - "scripted_success": always emits the next correct progress action ->
    should win every mock episode (tests the happy path end to end).
  - "random": picks a random admissible action each step -> mostly fails /
    hits the step cap (tests the failure and step-cap paths).
  - "malformed": omits the Action: line half the time -> tests
    react_agent.py's parse-failure fallback.
"""

from __future__ import annotations

import ast
import random
import re

from .llm_client import LLMResponse

_ADMISSIBLE_RE = re.compile(r"Admissible actions:\s*(\[.*\])", re.DOTALL)
_HISTORY_STEP_RE = re.compile(r"^> ", re.MULTILINE)


def _parse_admissible(user_content: str) -> list[str]:
    match = _ADMISSIBLE_RE.search(user_content)
    if not match:
        return []
    return ast.literal_eval(match.group(1))


def _parse_step_index(user_content: str) -> int:
    return len(_HISTORY_STEP_RE.findall(user_content))


class MockLLMClient:
    def __init__(self, strategy: str = "scripted_success", seed: int = 0):
        assert strategy in ("scripted_success", "random", "malformed")
        self.strategy = strategy
        self.rng = random.Random(seed)

    def complete(self, messages: list[dict], stop: list[str] | None = None) -> LLMResponse:
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        admissible = _parse_admissible(user_content)
        step_idx = _parse_step_index(user_content)

        if self.strategy == "scripted_success":
            action = admissible[min(step_idx, len(admissible) - 1)] if admissible else ""
            text = f"Thought: proceeding with the next step of the plan.\nAction: {action}"
        elif self.strategy == "random":
            action = self.rng.choice(admissible) if admissible else ""
            text = f"Thought: trying an action.\nAction: {action}"
        else:  # malformed
            if self.rng.random() < 0.5:
                text = "I think I should look around."  # no Action: line at all
            else:
                action = self.rng.choice(admissible) if admissible else ""
                text = f"Thought: proceeding.\nAction: {action}"

        input_tokens = sum(len(m["content"].split()) for m in messages)
        output_tokens = len(text.split())
        return LLMResponse(text=text, input_tokens=input_tokens, output_tokens=output_tokens, cached=False, finish_reason="stop")

    def complete_no_cache(self, messages: list[dict], stop: list[str] | None = None) -> LLMResponse:
        """Mocks never cache, but "random"/"malformed" strategies advance
        internal rng state each call, so calling this twice with identical
        messages can legitimately return different text -- useful for
        testing the determinism-check's negative case."""
        return self.complete(messages, stop)
