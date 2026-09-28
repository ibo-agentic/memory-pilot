"""Optional vLLM-backed LLM client (2026-09-29, Task 3 of the Kaggle timing
prep) -- an alternative to local_model_client.py's plain
`transformers.generate()` backend.

**Compatibility assessment (research, NOT a live T4 test -- no T4 hardware
is reachable from this environment, so this is engineering judgment plus
public evidence, not a measurement):**
  - vLLM on a Tesla T4 (compute capability 7.5) hard-fails on the DEFAULT
    dtype (bfloat16, needs compute capability >= 8.0) with a clear error --
    long-documented, long-fixed by passing dtype="half" explicitly
    (vllm-project/vllm#1157). This client always forces dtype="half" for
    exactly that reason -- never rely on vLLM's default.
  - Full fp16 for a 7B model on a 16GB T4 is TIGHT: ~14GB of weights alone,
    leaving thin headroom for vLLM's activation/KV-cache pool, and multiple
    public reports describe needing to hand-tune `gpu_memory_utilization`
    down (e.g. to 0.75-0.85) and `max_model_len` down (e.g. to 2048-4096) to
    avoid OOM even for single-sequence use. Plausible but NOT a safe
    default.
  - AWQ 4-bit quantization has an official Qwen-published checkpoint
    (`Qwen/Qwen2.5-7B-Instruct-AWQ`), is one of vLLM's most mature
    quantization paths, and drops weight VRAM to ~4-5GB -- comfortable
    headroom on a 16GB T4 even before considering that this project runs
    ONE sequence at a time (no concurrent-request KV-cache pressure vLLM is
    usually tuned against). **This is the recommended default here.**
  - CONCLUSION: added as an optional backend, AWQ by default, fp16 also
    available but flagged as unverified/risky. The actual pass/fail
    determination is an empirical question that can only be answered on
    real Kaggle T4 hardware -- see kaggle/timing_probe.ipynb's dedicated
    compatibility-probe cell, which is the real check; this module just
    makes the backend available to it and to timing_probe.py.

Same LLMClient protocol (`complete`, `complete_no_cache`) as
local_model_client.LocalTransformersClient, so every existing caller
(react_agent.py, episode_runner.py, timing_probe.py, determinism_check.py)
needs zero changes to use this instead -- selected via timing_probe.py's
`--backend vllm`.

Uses vLLM's offline `LLM.chat()` API (applies the model's own chat template
internally, matching local_model_client.py's use of
tokenizer.apply_chat_template) rather than spinning up an HTTP server --
this project always runs one request at a time, so there's no batching
throughput to gain from the server path.

Revision pinning: same hard rule as OpenRouterClient/LocalTransformersClient
(never "main"/"latest"/None) -- kaggle_config.yaml's llm_vllm.revision is a
placeholder (null) that MUST be replaced with the exact commit hash actually
downloaded before this client can even construct, mirroring this project's
existing "record what Phase 0 actually downloaded" discipline
(embedding_retrieval.SentenceEmbedder's docstring).
"""

from __future__ import annotations

from .cache import LLMCache
from .cost_tracker import CostTracker
from .llm_client import LLMResponse


class VLLMClient:
    def __init__(
        self,
        model_id: str,
        revision: str,
        temperature: float,
        max_output_tokens: int,
        cache: LLMCache,
        cost_tracker: CostTracker,
        quantization: str | None = "awq",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 2048,
        dtype: str = "half",
    ):
        if not model_id:
            raise ValueError("model_id is required -- set an exact pinned HuggingFace repo id.")
        if not revision or revision.lower() in ("main", "latest"):
            raise ValueError(
                f"revision={revision!r} looks unpinned -- pass the exact commit hash actually "
                "downloaded, never 'main'/'latest' (see llm_client.OpenRouterClient's same rule)."
            )
        if dtype != "half":
            import warnings

            warnings.warn(
                f"dtype={dtype!r} requested, not the T4-safe 'half' -- a Tesla T4 (compute "
                "capability 7.5) cannot run bfloat16 (vLLM hard-fails with a clear error), and "
                "no non-half dtype has been verified against a T4 for this project.",
                stacklevel=2,
            )

        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "VLLMClient requires the `vllm` package, which is not installed. "
                "See kaggle/timing_probe.ipynb's install cell (pip install vllm)."
            ) from e

        self.model_id = model_id
        self.revision = revision
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.cache = cache
        self.cost_tracker = cost_tracker

        llm_kwargs = dict(
            model=model_id,
            revision=revision,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        if quantization:
            llm_kwargs["quantization"] = quantization
        self.llm = LLM(**llm_kwargs)

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

        self.cost_tracker.check_before_call(estimated_input_tokens=0, estimated_output_tokens=0)

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            stop=stop,
        )
        [output] = self.llm.chat([messages], sampling_params, use_tqdm=False)
        input_tokens = len(output.prompt_token_ids)
        completion = output.outputs[0]
        output_tokens = len(completion.token_ids)
        text = completion.text
        finish_reason = completion.finish_reason or "stop"

        self.cost_tracker.record_call(input_tokens, output_tokens)
        self.cache.put(
            payload,
            {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens, "finish_reason": finish_reason},
        )
        return LLMResponse(text=text, input_tokens=input_tokens, output_tokens=output_tokens, cached=False, finish_reason=finish_reason)

    def complete_no_cache(self, messages: list[dict], stop: list[str] | None = None) -> LLMResponse:
        return self.complete(messages, stop, _skip_cache_read=True)


def build_vllm_client(cfg: dict) -> tuple[VLLMClient, CostTracker]:
    """Mirrors capability_check.build_local_client's shape, reading from
    cfg["llm_vllm"] instead of cfg["llm"] -- kept as a separate config
    section (not a mode switch on the same section) so the two backends'
    settings can't accidentally bleed into each other."""
    llm_cfg = cfg["llm_vllm"]
    cache = LLMCache(llm_cfg["cache_dir"])
    cost_tracker = CostTracker(cfg["cost_control"]["hard_cap_usd"], 0.0, 0.0)
    return VLLMClient(
        model_id=llm_cfg["model_id"],
        revision=llm_cfg["revision"],
        temperature=llm_cfg["temperature"],
        max_output_tokens=llm_cfg["max_output_tokens"],
        cache=cache,
        cost_tracker=cost_tracker,
        quantization=llm_cfg.get("quantization", "awq"),
        gpu_memory_utilization=llm_cfg.get("gpu_memory_utilization", 0.85),
        max_model_len=llm_cfg.get("max_model_len", 2048),
        dtype=llm_cfg.get("dtype", "half"),
    ), cost_tracker
