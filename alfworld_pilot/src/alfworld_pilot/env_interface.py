"""Single-episode environment interface matching ALFWorld's real API, plus a
mock implementation used by the unit test suite (fast, free, no data/install
dependency).

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
the rest of this pilot uses. `MockAlfredEnv` implements the exact same
interface so `react_agent.py` / `episode_runner.py` etc. don't need to
change based on which backend is selected (see env_factory.py) -- use the
mock for unit tests, the real env for actual pilot runs.

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

    def close(self) -> None:
        """Release any underlying resources (subprocesses, sockets). Always
        call this once you're done with a short-lived env (e.g. one built
        per ground-truth pair) -- a long-lived env reused across many
        episodes (e.g. measure_mode's) is instead closed once, at the end."""
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

    def close(self) -> None:
        pass  # nothing to release -- pure in-memory mock


class RealAlfredEnv:
    """Thin wrapper around the real alfworld package's batched API.

    Installed and running under WSL2 Ubuntu on Python 3.11 (the native-
    Windows attempt documented in alfworld_pilot/README.md hit a build
    toolchain wall that WSL2's build-essential doesn't have). Python 3.11,
    not the repo's earlier 3.14, because textworld 1.7.0's PDDL grammar
    engine relies on a locals()-mutation trick that Python 3.13's PEP 667
    permanently breaks -- rebuilding the venv on 3.11 (within ALFWorld's own
    documented "Python 3.9+" support range, and short of that break) let us
    drop what would otherwise be a standing monkeypatch of third-party
    library internals. See alfworld_pilot/README.md for the full story.

    `gamefile_path`, if given, restricts this env to exactly ONE game file
    instead of the full split -- every reset() then replays that same task
    instance. This is what paired ground truth needs (forced-in and
    forced-out must play the identical task); see ground_truth_runner.py.
    """

    def __init__(self, config: dict, split: str = "train", gamefile_path: str | None = None):
        try:
            from alfworld.agents.environment import get_environment
            from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        except ImportError as e:
            raise ImportError(
                "alfworld is not installed in this environment. See alfworld_pilot/README.md "
                "for why and how to install it."
            ) from e

        if gamefile_path is not None:
            # Skip AlfredTWEnv.__init__'s collect_game_files() (an expensive
            # walk over the ENTIRE split) by constructing the object without
            # running __init__, then calling its real init_env() -- the same
            # method the normal path below uses -- restricted to one file.
            # Verified empirically (see alfworld_pilot/README.md): two
            # independently-constructed envs pointed at the same gamefile
            # give byte-identical reset() observations, admissible commands,
            # and post-step results.
            tw_env = AlfredTWEnv.__new__(AlfredTWEnv)
            tw_env.config = config
            tw_env.train_eval = split
            tw_env.game_files = [gamefile_path]
            tw_env.num_games = 1
            self._env = tw_env.init_env(batch_size=1)
        else:
            env_type = config["env"]["type"]
            # get_environment() imports the requested class LOCALLY (it's
            # not a module-level attribute of alfworld.agents.environment),
            # so this must go through it rather than getattr(module, env_type).
            self._env = get_environment(env_type)(config, train_eval=split)
            self._env = self._env.init_env(batch_size=1)
        self._task_type: str = "unknown"

    @staticmethod
    def task_type_from_gamefile(gamefile_path: str) -> str:
        for task_type in TASK_TYPES:
            if task_type in gamefile_path:
                return task_type
        return "unknown"

    def reset(self, task_seed: int | None = None) -> tuple[str, dict]:
        obs, info = self._env.reset()
        gamefile = info["extra.gamefile"][0]
        self._task_type = self.task_type_from_gamefile(gamefile)
        info = dict(info)
        info["task_type"] = self._task_type
        return obs[0], info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        # `extra.gamefile` is only populated by TextWorld on reset(), not on
        # every step() -- confirmed empirically (it comes back None mid-
        # episode) -- so task_type is cached from the episode's reset() call
        # instead of being re-derived here.
        obs, scores, dones, infos = self._env.step([action])
        infos = dict(infos)
        infos["task_type"] = self._task_type
        return obs[0], float(scores[0]), bool(dones[0]), infos

    def close(self) -> None:
        """AlfredTWEnv.init_env() registers games with asynchronous=True,
        which spawns a subprocess per env even at batch_size=1 -- harmless
        for one long-lived episode-logging env, but constructing many
        short-lived single-game envs (as ground truth / difficulty checks
        do) without closing them leaks subprocesses and memory. Always
        close() a gamefile_path-restricted env once you're done with it."""
        self._env.close()


def list_real_game_files(config: dict, split: str = "train") -> list[str]:
    """The list of individual game.tw-pddl file paths real ALFWorld would
    play for this config/split -- same filtering AlfredTWEnv.__init__ does
    (solvable-only, config's env.task_types, this split's data_path).
    Expensive (walks the entire split directory tree once, a few seconds for
    ~3500 games) -- call ONCE per ground-truth session and reuse the result,
    not once per pair. Indexing into the returned list by a stable integer
    (e.g. a task_seed) is how paired ground truth picks "the same task
    instance" for both the forced-in and forced-out arms."""
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    return list(AlfredTWEnv(config, train_eval=split).game_files)
