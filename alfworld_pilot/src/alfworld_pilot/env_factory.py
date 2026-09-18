"""Picks MockAlfredEnv or RealAlfredEnv from config.yaml's env.backend, so
callers (measure_mode, episode-running scripts) don't each need their own
mock/real branching. Mock is used directly (not through this module) by the
unit test suite, which should stay fast and free regardless of what backend
config.yaml is pointed at.
"""

from __future__ import annotations

import functools

import yaml

from .env_interface import AlfredEnv, MockAlfredEnv, RealAlfredEnv


def load_real_alfworld_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env_factory(cfg: dict):
    """Returns a zero-arg callable that constructs ONE environment instance,
    meant to be called ONCE and then reused across episodes via repeated
    .reset() calls. Real ALFWorld's AlfredTWEnv walks its entire game-file
    dataset at construction time, and .reset() is what advances it to the
    next game -- constructing a fresh instance per episode would silently
    re-walk that dataset every time and defeat the mechanism real ALFWorld
    uses to hand out distinct games."""
    backend = cfg["env"]["backend"]
    if backend == "mock":
        return MockAlfredEnv
    if backend == "real":
        real_cfg_path = cfg["env"]["real_alfworld_config_path"]
        if not real_cfg_path:
            raise ValueError(
                "env.backend is 'real' but env.real_alfworld_config_path is not set in config.yaml"
            )
        real_cfg = load_real_alfworld_config(real_cfg_path)
        split = cfg["env"].get("real_split", "train")
        return functools.partial(RealAlfredEnv, real_cfg, split=split)
    raise ValueError(f"Unknown env.backend: {backend!r} (expected 'mock' or 'real')")


def build_task_source(cfg: dict):
    """Returns a ground_truth_runner.TaskSource for config.yaml's
    env.backend, resolving the real ALFWorld config the same way
    build_env_factory does -- callers should never need to load
    real_alfworld_config_path themselves."""
    from .ground_truth_runner import MockTaskSource, RealTaskSource

    backend = cfg["env"]["backend"]
    if backend == "mock":
        return MockTaskSource()
    if backend == "real":
        real_cfg_path = cfg["env"]["real_alfworld_config_path"]
        if not real_cfg_path:
            raise ValueError(
                "env.backend is 'real' but env.real_alfworld_config_path is not set in config.yaml"
            )
        real_cfg = load_real_alfworld_config(real_cfg_path)
        split = cfg["env"].get("real_split", "train")
        return RealTaskSource(real_cfg, split=split)
    raise ValueError(f"Unknown env.backend: {backend!r} (expected 'mock' or 'real')")


__all__ = ["AlfredEnv", "build_env_factory", "build_task_source", "load_real_alfworld_config"]
