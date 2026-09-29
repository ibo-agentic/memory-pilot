"""Test for local_model_client.py's low_cpu_mem_usage=True addition
(2026-09-30, after a real Kaggle hang -- see multi_worker_phase0.py's module
docstring). Requires torch/transformers, which live only in the heavy
.venv-kaggle environment, not this project's fast/lightweight standard
.venv -- skipped there, runs for real under .venv-kaggle."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_local_transformers_client_passes_low_cpu_mem_usage(tmp_path):
    from alfworld_pilot.cache import LLMCache
    from alfworld_pilot.cost_tracker import CostTracker
    from alfworld_pilot.local_model_client import LocalTransformersClient

    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_from_pretrained, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_from_pretrained:
        mock_model_from_pretrained.return_value = MagicMock()
        mock_tok_from_pretrained.return_value = MagicMock()

        LocalTransformersClient(
            model_id="fake/model",
            revision="abc123def456",
            temperature=0.0,
            max_output_tokens=10,
            cache=LLMCache(tmp_path / "cache"),
            cost_tracker=CostTracker(999.0, 0.0, 0.0),
            device="cpu",
            quantize_4bit=False,
        )

        _, kwargs = mock_model_from_pretrained.call_args
        assert kwargs.get("low_cpu_mem_usage") is True
