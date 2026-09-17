"""Load config/config.yaml into a plain dict, resolved relative to the repo root."""

from __future__ import annotations

import pathlib

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(path: pathlib.Path | str | None = None) -> dict:
    path = pathlib.Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("logs_dir", "cache_dir", "results_dir"):
        cfg["paths"][key] = str(REPO_ROOT / cfg["paths"][key])
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for key in ("logs_dir", "cache_dir", "results_dir"):
        pathlib.Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
