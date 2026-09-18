"""Load alfworld_pilot/config.yaml, resolved relative to the alfworld_pilot dir."""

from __future__ import annotations

import pathlib

import yaml
from dotenv import load_dotenv

PILOT_ROOT = pathlib.Path(__file__).resolve().parents[2]
REPO_ROOT = PILOT_ROOT.parent
DEFAULT_CONFIG_PATH = PILOT_ROOT / "config.yaml"


def load_config(path: pathlib.Path | str | None = None) -> dict:
    # .env lives at the repo root (shared with the Stage 1 package), not inside alfworld_pilot/.
    load_dotenv(REPO_ROOT / ".env")
    path = pathlib.Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("logs_dir", "results_dir"):
        cfg["paths"][key] = str(PILOT_ROOT / cfg["paths"][key])
    cfg["cache"]["dir"] = str(PILOT_ROOT / cfg["cache"]["dir"])
    if cfg["env"].get("real_alfworld_config_path"):
        cfg["env"]["real_alfworld_config_path"] = str(PILOT_ROOT / cfg["env"]["real_alfworld_config_path"])
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for key in ("logs_dir", "results_dir"):
        pathlib.Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(cfg["cache"]["dir"]).mkdir(parents=True, exist_ok=True)
