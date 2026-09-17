"""JSONL episode logging: one JSON object per line, one line per episode."""

from __future__ import annotations

import json
import pathlib
from typing import Iterable, Iterator


def write_episodes(path: pathlib.Path | str, episodes: Iterable[dict]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")


def read_episodes(path: pathlib.Path | str) -> Iterator[dict]:
    path = pathlib.Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
