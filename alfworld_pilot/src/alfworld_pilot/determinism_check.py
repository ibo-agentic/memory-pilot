"""Verifies whether temperature=0 (+ a fixed `seed` param, when the
provider supports it) actually produces identical output on repeated
identical calls. Providers vary in how strictly they honor this -- this
must be measured, not assumed, for every paired ground-truth run."""

from __future__ import annotations

from .llm_client import LLMClient


def check_determinism(client: LLMClient, messages: list[dict], stop: list[str] | None = None) -> dict:
    r1 = client.complete_no_cache(messages, stop)
    r2 = client.complete_no_cache(messages, stop)
    return {
        "holds": r1.text == r2.text,
        "response_1": r1.text,
        "response_2": r2.text,
    }
