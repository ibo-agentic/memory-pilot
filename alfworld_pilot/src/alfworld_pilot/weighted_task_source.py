"""A build_env-only task source that deliberately weights task-type
sampling, instead of relying on list_real_game_files' raw filesystem order.

Why this exists: the memory sanity check's per-task-type breakdown
(2026-09-19) found that naive `task_id % len(game_files)` indexing over
the first 15 task_ids landed 13/15 on `pick_and_place_simple` (the
EASIEST task type -- mean 13.9 steps to solve) and only 2/15 on
`pick_two_obj_and_place` (the hardest -- mean 43.6). That's not a
deliberate design choice, it's an accident of `os.walk`'s traversal order
over the dataset directory -- and it explains most of why the pooled
93%/80% success rate sat too close to ceiling to resolve per-memory
effects (ground truth can resolve deltas of ~0.10-0.15; effects get
squeezed toward 0 once success is pinned near 1.0).

Weights default to each task type's mean steps-to-solve (from
task_type_difficulty_check.py's n=30/type real-ALFWorld measurement,
zero LLM cost) -- a real-agent-relevant difficulty proxy (more steps means
more decision points means more chances for a fallible agent to err, even
for a task type an OPTIMAL policy can still solve ~100% of the time at
env.max_steps=50), not an arbitrary choice.

Both the task-type draw and the specific-game-within-type draw are pure
functions of task_id (same per-task_id deterministic seeding pattern as
checkpointed_runner.py and ground_truth_runner.py), so this composes with
the existing checkpointing machinery without any changes there.
"""

from __future__ import annotations

import collections
import hashlib
import random

from .env_interface import TASK_TYPES, RealAlfredEnv, list_real_game_files

# Mean steps-to-solve per task type, ALFWorld train split, n=30/type,
# measured by task_type_difficulty_check.py (results/task_type_difficulty_check.json).
# Used as the default weighting -- override via task_type_weights if desired.
DEFAULT_WEIGHTS_BY_MEAN_STEPS = {
    "pick_and_place_simple": 13.9,
    "look_at_obj_in_light": 13.1,
    "pick_clean_then_place_in_recep": 22.2,
    "pick_heat_then_place_in_recep": 22.8,
    "pick_cool_then_place_in_recep": 16.9,
    "pick_two_obj_and_place": 43.6,
}


def _seed_for(task_id: int, salt: str) -> int:
    key = f"weighted_task_source|{salt}|{task_id}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) & 0xFFFFFFFF


class WeightedRealTaskSource:
    """build_env(task_id) only -- ground truth's candidacy probing
    (task_type(task_id)) isn't needed for the logging phase this was built
    for, but is provided too since it's cheap and keeps the interface
    consistent with MockTaskSource/RealTaskSource."""

    def __init__(self, config: dict, split: str = "train", task_type_weights: dict[str, float] | None = None):
        self.config = config
        self.split = split
        self.weights = dict(task_type_weights or DEFAULT_WEIGHTS_BY_MEAN_STEPS)

        game_files = list_real_game_files(config, split)
        by_type: dict[str, list[str]] = collections.defaultdict(list)
        for gf in game_files:
            by_type[RealAlfredEnv.task_type_from_gamefile(gf)].append(gf)
        for tt in by_type:
            by_type[tt].sort()  # deterministic order within each type
        self.by_type = dict(by_type)

        self._types = [tt for tt in TASK_TYPES if self.by_type.get(tt)]
        if not self._types:
            raise RuntimeError("No game files found for any task type -- check config.yaml's dataset paths.")
        self._weight_list = [self.weights.get(tt, 1.0) for tt in self._types]

    def _task_type_for(self, task_id: int) -> str:
        r = random.Random(_seed_for(task_id, "type"))
        return r.choices(self._types, weights=self._weight_list, k=1)[0]

    def _gamefile_for(self, task_id: int) -> str:
        tt = self._task_type_for(task_id)
        pool = self.by_type[tt]
        r = random.Random(_seed_for(task_id, "game"))
        return r.choice(pool)

    def task_type(self, task_id: int) -> str:
        return self._task_type_for(task_id)

    def build_env(self, task_id: int) -> RealAlfredEnv:
        return RealAlfredEnv(self.config, split=self.split, gamefile_path=self._gamefile_for(task_id))
