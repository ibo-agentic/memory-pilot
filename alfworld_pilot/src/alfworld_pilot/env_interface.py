"""Single-episode environment interface matching ALFWorld's real API, plus a
mock implementation so the rest of the pilot can be built and tested before
ALFWorld itself is installed.

Real ALFWorld's batched API (confirmed from the official repo + the
original ReAct paper's alfworld.ipynb, batch_size=1 throughout this
project):

    from alfworld.agents.environment import get_environment
    env = get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    obs, info = env.reset()
    admissible_commands = list(info['admissible_commands'])
    obs, scores, dones, infos = env.step([action])
    task_type = <matched from info['extra.gamefile'][0] path>

`RealAlfredEnv` below wraps that batched API into the single-episode shape
the rest of this pilot uses; it is NOT yet runnable because ALFWorld is not
installed in this environment (see alfworld_pilot/README.md for why, and
the options for installing it). `MockAlfredEnv` implements the exact same
interface so `react_agent.py` / `episode_runner.py` etc. don't need to
change when ALFWorld is later swapped in.

Official ALFWorld task types (from the official repo + ALFWorld paper),
encoded in each game's file path as `task_type-object-movableReceptacle-
receptacle-sceneNum`:
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]


class AlfredEnv(Protocol):
    """Single-episode (batch_size=1, unwrapped) environment interface."""

    def reset(self, task_seed: int | None = None) -> tuple[str, dict]:
        """Returns (observation, info). info must include 'admissible_commands' and 'task_type'."""
        ...

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Returns (observation, reward, done, info)."""
        ...


@dataclass
class MockTaskTemplate:
    task_type: str
    goal: str
    admissible_actions: list[str]
    winning_action: str

    def __post_init__(self) -> None:
        if self.winning_action not in self.admissible_actions:
            raise ValueError(
                f"winning_action {self.winning_action!r} for task_type {self.task_type!r} "
                f"is not in admissible_actions {self.admissible_actions!r}"
            )

    @property
    def steps_to_win(self) -> int:
        """Derived, not hand-specified, so it can't drift out of sync with
        winning_action's actual position in admissible_actions."""
        return self.admissible_actions.index(self.winning_action) + 1


# One illustrative template per real ALFWorld task type, enough to exercise
# retrieval + ReAct + logging end to end. Not a claim that these match real
# ALFWorld game content.
_MOCK_TEMPLATES: list[MockTaskTemplate] = [
    MockTaskTemplate(
        "pick_and_place_simple",
        "put a clean mug in coffeemachine",
        ["go to countertop 1", "take mug 1 from countertop 1", "go to coffeemachine 1", "put mug 1 in/on coffeemachine 1", "look"],
        "put mug 1 in/on coffeemachine 1",
    ),
    MockTaskTemplate(
        "look_at_obj_in_light",
        "look at statue under the desklamp",
        ["go to desk 1", "take statue 1 from desk 1", "go to desklamp 1", "use desklamp 1", "look"],
        "use desklamp 1",
    ),
    MockTaskTemplate(
        "pick_clean_then_place_in_recep",
        "clean an apple and put it in fridge",
        ["go to sinkbasin 1", "take apple 1 from sinkbasin 1", "clean apple 1 with sinkbasin 1", "go to fridge 1", "put apple 1 in/on fridge 1"],
        "put apple 1 in/on fridge 1",
    ),
    MockTaskTemplate(
        "pick_heat_then_place_in_recep",
        "heat a mug and put it on the countertop",
        ["go to microwave 1", "take mug 1 from microwave 1", "heat mug 1 with microwave 1", "go to countertop 1", "put mug 1 in/on countertop 1"],
        "put mug 1 in/on countertop 1",
    ),
    MockTaskTemplate(
        "pick_cool_then_place_in_recep",
        "cool a tomato and put it in the sink",
        ["go to fridge 1", "take tomato 1 from fridge 1", "cool tomato 1 with fridge 1", "go to sinkbasin 1", "put tomato 1 in/on sinkbasin 1"],
        "put tomato 1 in/on sinkbasin 1",
    ),
    MockTaskTemplate(
        "pick_two_obj_and_place",
        "put two pillows on the sofa",
        ["go to bed 1", "take pillow 1 from bed 1", "go to sofa 1", "put pillow 1 in/on sofa 1", "take pillow 2 from bed 1", "put pillow 2 in/on sofa 1"],
        "put pillow 2 in/on sofa 1",
    ),
]


@dataclass
class MockAlfredEnv:
    """Deterministic-given-seed mock matching AlfredEnv's interface.

    Success model: the episode succeeds if the agent's action history
    contains the template's progress actions in order by the time it issues
    `winning_action`, within the step cap; this rewards a coherent plan, not
    just guessing the final action, so a ReAct agent that ignores structure
    doesn't win by luck alone.
    """

    rng: random.Random = field(default_factory=random.Random)
    _template: MockTaskTemplate | None = field(default=None, init=False)
    _progress: int = field(default=0, init=False)
    _gamefile: str = field(default="", init=False)

    def reset(self, task_seed: int | None = None) -> tuple[str, dict]:
        r = random.Random(task_seed) if task_seed is not None else self.rng
        self._template = r.choice(_MOCK_TEMPLATES)
        self._progress = 0
        scene_num = r.randint(1, 30)
        self._gamefile = f"{self._template.task_type}-obj-recep-recep-{scene_num}/trial_0/game.tw-pddl"
        obs = f"You are in the middle of a room. Your task is to: {self._template.goal}"
        info = {
            "admissible_commands": [list(self._template.admissible_actions)],
            "task_type": self._template.task_type,
            "extra.gamefile": [self._gamefile],
            "won": [False],
        }
        return obs, info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        assert self._template is not None, "call reset() first"
        t = self._template
        expected = t.admissible_actions[min(self._progress, t.steps_to_win - 1)]
        won = False
        if action.strip() == expected.strip():
            self._progress += 1
            obs = f"You {action}. Looks good."
            if action.strip() == t.winning_action.strip() and self._progress >= t.steps_to_win:
                won = True
                obs = "You win! " + obs
        else:
            obs = f"Nothing happens. ('{action}' was not a recognized next step.)"

        info = {
            "admissible_commands": [list(t.admissible_actions)],
            "task_type": t.task_type,
            "extra.gamefile": [self._gamefile],
            "won": [won],
        }
        return obs, (1.0 if won else 0.0), won, info


class RealAlfredEnv:
    """Thin wrapper around the real alfworld package's batched API.

    NOT YET RUNNABLE in this environment -- alfworld's install requires a
    Linux-oriented native build (jericho + a Linux-only Inform7 CLI fetched
    by a bash setup.sh), which failed on native Windows with no C toolchain
    (confirmed by attempting `pip install alfworld` directly -- see
    alfworld_pilot/README.md). Deferred per project decision: install via
    WSL2 or Docker when ready, then this class should work unmodified
    against a real `alfworld.agents.environment` config.
    """

    def __init__(self, config: dict, split: str = "train"):
        try:
            import alfworld.agents.environment as alfworld_env
        except ImportError as e:
            raise ImportError(
                "alfworld is not installed in this environment. See alfworld_pilot/README.md "
                "for why (Linux-oriented native build) and how to install it (WSL2 or Docker)."
            ) from e
        env_type = config["env"]["type"]
        self._env = getattr(alfworld_env, env_type)(config, train_eval=split)
        self._env = self._env.init_env(batch_size=1)

    @staticmethod
    def task_type_from_gamefile(gamefile_path: str) -> str:
        for task_type in TASK_TYPES:
            if task_type in gamefile_path:
                return task_type
        return "unknown"

    def reset(self, task_seed: int | None = None) -> tuple[str, dict]:
        obs, info = self._env.reset()
        gamefile = info["extra.gamefile"][0]
        info = dict(info)
        info["task_type"] = self.task_type_from_gamefile(gamefile)
        return obs[0], info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        obs, scores, dones, infos = self._env.step([action])
        gamefile = infos["extra.gamefile"][0]
        infos = dict(infos)
        infos["task_type"] = self.task_type_from_gamefile(gamefile)
        return obs[0], float(scores[0]), bool(dones[0]), infos
