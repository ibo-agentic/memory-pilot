"""Disk-based cache for LLM calls, keyed by a hash of everything that
affects the response (model, messages, temperature, max_tokens, seed,
reasoning setting) so a rerun with identical inputs never re-bills."""

from __future__ import annotations

import hashlib
import json
import pathlib


def _cache_key(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(self, cache_dir: str | pathlib.Path):
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> pathlib.Path:
        return self.cache_dir / f"{key}.json"

    def get(self, request_payload: dict) -> dict | None:
        key = _cache_key(request_payload)
        path = self._path(key)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def put(self, request_payload: dict, response_payload: dict) -> None:
        key = _cache_key(request_payload)
        with open(self._path(key), "w", encoding="utf-8") as f:
            json.dump({"request": request_payload, "response": response_payload}, f, indent=2)

    def key_for(self, request_payload: dict) -> str:
        return _cache_key(request_payload)
