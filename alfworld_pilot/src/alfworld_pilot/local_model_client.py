"""Local, zero-dollar LLM client for the Kaggle replication -- implements the
same `LLMClient` protocol (`complete`, `complete_no_cache`) as
`llm_client.OpenRouterClient`, following its exact structure (cache-check ->
cost-tracker pre-check -> real call -> cost-tracker record -> cache put) so
every downstream consumer (react_agent.py, episode_runner.py,
ground_truth_runner.py, determinism_check.py, checkpointed_runner.py) needs
ZERO changes to work with a local model instead of OpenRouter.

Backend: `transformers`' own `.generate()`, not vLLM, by default -- per the
approved plan, current mainline vLLM has documented compatibility issues with
Kaggle's Tesla T4/P100 GPUs (bfloat16-only-on-compute-capability-8.0+ errors,
T4 support being deprecated upstream), and this project runs one episode at a
time, never batched, so vLLM's main advantage (continuous batching throughput)
doesn't apply here the way it would to a high-QPS server anyway. If Phase 0
confirms vLLM works cleanly on the target GPU, a second client class following
this same interface can be added later without touching any caller.

Model id + revision must be an EXACT pinned HuggingFace repo id and commit
hash, never "latest"/unpinned `main` -- matches `OpenRouterClient`'s existing
"never a latest alias" rule.

Determinism caveat (real, not hypothetical -- stated here so it isn't lost):
greedy decoding (`do_sample=False`) is the standard way to get deterministic
output, but GPU matrix-multiply reduction order can introduce tiny numerical
non-determinism that, in rare cases, flips which token wins a near-tied
argmax. This must be verified empirically with `determinism_check.py` against
this actual client on the actual target hardware, not assumed from
"temperature=0 means deterministic" -- which is exactly what this replication
is doing that the paid run never did.
"""

from __future__ import annotations

from .cache import LLMCache
from .cost_tracker import CostTracker
from .llm_client import LLMResponse


def _messages_to_prompt(messages: list[dict], tokenizer) -> str:
    """Uses the model's own chat template (every modern instruct model ships
    one) rather than hand-formatting role tags -- avoids silently mismatching
    whatever prompt format the model was actually instruction-tuned on."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class LocalTransformersClient:
    def __init__(
        self,
        model_id: str,
        revision: str,
        temperature: float,
        max_output_tokens: int,
        cache: LLMCache,
        cost_tracker: CostTracker,
        device: str = "cuda",
        quantize_4bit: bool = True,
    ):
        if not model_id:
            raise ValueError("model_id is required -- set an exact pinned HuggingFace repo id.")
        if not revision or revision.lower() in ("main", "latest"):
            raise ValueError(
                f"revision={revision!r} looks unpinned -- pass the exact commit hash actually "
                "downloaded, never 'main'/'latest' (see llm_client.OpenRouterClient's same rule)."
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.cache = cache
        self.cost_tracker = cost_tracker

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        quant_kwargs = {}
        if quantize_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.float16, device_map=device, **quant_kwargs,
        )
        self.model.eval()

    def _request_payload(self, messages: list[dict], stop: list[str] | None) -> dict:
        return {
            "model": self.model_id,
            "revision": self.revision,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stop": stop,
        }

    def complete(self, messages: list[dict], stop: list[str] | None = None, _skip_cache_read: bool = False) -> LLMResponse:
        payload = self._request_payload(messages, stop)

        cached = None if _skip_cache_read else self.cache.get(payload)
        if cached is not None:
            self.cost_tracker.record_cache_hit()
            resp = cached["response"]
            return LLMResponse(
                text=resp["text"], input_tokens=resp["input_tokens"], output_tokens=resp["output_tokens"],
                cached=True, finish_reason=resp.get("finish_reason"),
            )

        # Always $0 (price_per_million_* is 0 for a local client), so this is a
        # no-op in practice -- kept for interface symmetry with OpenRouterClient
        # and so any future non-zero "compute cost" accounting has a hook.
        self.cost_tracker.check_before_call(estimated_input_tokens=0, estimated_output_tokens=0)

        import torch

        prompt = _messages_to_prompt(messages, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[1])

        do_sample = self.temperature > 0
        gen_kwargs = dict(max_new_tokens=self.max_output_tokens, do_sample=do_sample, pad_token_id=self.tokenizer.eos_token_id)
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][input_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        output_tokens = int(new_tokens.shape[0])
        finish_reason = "stop"

        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
                    finish_reason = "stop_sequence"
                    break

        self.cost_tracker.record_call(input_tokens, output_tokens)
        self.cache.put(
            payload,
            {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens, "finish_reason": finish_reason},
        )
        return LLMResponse(text=text, input_tokens=input_tokens, output_tokens=output_tokens, cached=False, finish_reason=finish_reason)

    def complete_no_cache(self, messages: list[dict], stop: list[str] | None = None) -> LLMResponse:
        """Forces a fresh real generation even if an identical request is
        already cached -- used only by determinism_check.py, same contract as
        OpenRouterClient.complete_no_cache."""
        return self.complete(messages, stop, _skip_cache_read=True)
