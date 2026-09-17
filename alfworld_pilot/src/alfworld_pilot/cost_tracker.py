"""Tracks cumulative real spend against a hard cap and refuses further LLM
calls once it would be exceeded. Cache hits never count -- they cost
nothing."""

from __future__ import annotations


class CostCapExceeded(RuntimeError):
    pass


class CostTracker:
    def __init__(self, hard_cap_usd: float, price_per_million_input: float, price_per_million_output: float):
        self.hard_cap_usd = hard_cap_usd
        self.price_per_million_input = price_per_million_input
        self.price_per_million_output = price_per_million_output
        self.total_spend_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.n_calls = 0
        self.n_cache_hits = 0

    def cost_of(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1e6) * self.price_per_million_input + (output_tokens / 1e6) * self.price_per_million_output

    def check_before_call(self, estimated_input_tokens: int, estimated_output_tokens: int) -> None:
        projected = self.total_spend_usd + self.cost_of(estimated_input_tokens, estimated_output_tokens)
        if projected > self.hard_cap_usd:
            raise CostCapExceeded(
                f"Projected spend ${projected:.4f} would exceed hard cap ${self.hard_cap_usd:.2f} "
                f"(current spend ${self.total_spend_usd:.4f}). Stopping before making the call."
            )

    def record_cache_hit(self) -> None:
        self.n_cache_hits += 1

    def record_call(self, input_tokens: int, output_tokens: int) -> float:
        cost = self.cost_of(input_tokens, output_tokens)
        self.total_spend_usd += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.n_calls += 1
        return cost

    def summary(self) -> dict:
        return {
            "total_spend_usd": round(self.total_spend_usd, 4),
            "hard_cap_usd": self.hard_cap_usd,
            "n_calls": self.n_calls,
            "n_cache_hits": self.n_cache_hits,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
