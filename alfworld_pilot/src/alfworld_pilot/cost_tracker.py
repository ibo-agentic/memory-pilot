"""Tracks cumulative real spend against a hard cap and refuses further LLM
calls once it would be exceeded. Cache hits never count -- they cost
nothing.

If `state_path` is given, cumulative counters persist to disk (atomic
write: write-to-temp then rename) after every call/cache-hit, and are
reloaded from it at construction time. This is what makes the hard cap
survive a crash or restart: without it, a restarted process would start
counting from $0 again and could let real cumulative spend across the
crash and the restart exceed hard_cap_usd without ever seeing a single
call that looks, in isolation, like it crosses the line."""

from __future__ import annotations

import json
import pathlib


class CostCapExceeded(RuntimeError):
    pass


class CostTracker:
    def __init__(
        self,
        hard_cap_usd: float,
        price_per_million_input: float,
        price_per_million_output: float,
        state_path: str | pathlib.Path | None = None,
    ):
        self.hard_cap_usd = hard_cap_usd
        self.price_per_million_input = price_per_million_input
        self.price_per_million_output = price_per_million_output
        self.state_path = pathlib.Path(state_path) if state_path else None
        self.total_spend_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.n_calls = 0
        self.n_cache_hits = 0
        if self.state_path is not None and self.state_path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.total_spend_usd = state["total_spend_usd"]
        self.total_input_tokens = state["total_input_tokens"]
        self.total_output_tokens = state["total_output_tokens"]
        self.n_calls = state["n_calls"]
        self.n_cache_hits = state["n_cache_hits"]

    def _save(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_spend_usd": self.total_spend_usd,
                    "total_input_tokens": self.total_input_tokens,
                    "total_output_tokens": self.total_output_tokens,
                    "n_calls": self.n_calls,
                    "n_cache_hits": self.n_cache_hits,
                },
                f,
            )
        tmp.replace(self.state_path)  # atomic on POSIX -- never leaves a half-written state file

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
        self._save()

    def record_call(self, input_tokens: int, output_tokens: int) -> float:
        cost = self.cost_of(input_tokens, output_tokens)
        self.total_spend_usd += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.n_calls += 1
        self._save()
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
