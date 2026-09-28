"""A simple (single-LLM-call-per-step) ReAct agent: each step, one call
produces both a Thought and an Action, the Action is parsed out and sent to
the environment. This is deliberately simpler than the original ReAct
paper's alfworld setup (which sometimes uses a separate call for
"think:"-only steps) -- one call per step keeps the Stage 2 budget
estimate's "1 call per step" assumption accurate, and matches "a simple
ReAct-style agent" from the spec.

Prompt sections, in order (also what the token-assumption breakdown in
measure_mode.py / the design doc refers to):
  1. system prompt (fixed task-agnostic instructions)
  2. retrieved memories (the lessons this episode's randomized retrieval included)
  3. task goal (from the environment's initial observation)
  4. step history (grows every step: observation + action pairs)
  5. this step's admissible actions
  -> output: "Thought: ...\nAction: ..."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm_client import LLMClient

SYSTEM_PROMPT = (
    "You are an agent solving a household task in a text-based environment (ALFWorld). "
    "Each turn, you see the current observation and a list of admissible actions. "
    "Respond with exactly two lines:\n"
    "Thought: <brief reasoning about what to do next>\n"
    "Action: <one action, copied EXACTLY from the admissible actions list>\n"
)

ACTION_LINE_RE = re.compile(r"^\s*Action:\s*(.+?)\s*$", re.MULTILINE)

# Optional few-shot demonstrations (see capability_check.py): the original
# ReAct paper (Yao et al. 2022) used few-shot prompting for ALFWorld; this
# pilot's zero-shot prompt was designed assuming a much stronger commercial
# model (openai/gpt-5.6-luna, the paid run's model) and was never validated
# against a smaller open model. Hand-written, not tied to any specific real
# game instance (so it demonstrates FORMAT and STRATEGY -- systematic search,
# opening closed receptacles before interacting with them, appliance actions
# before final placement -- without leaking a memorized answer for any real
# evaluation task). Two examples: one plain pick-and-place, one with an
# intermediate appliance action, to generalize past pick_and_place_simple.
FEW_SHOT_EXAMPLES = """Example 1 of solving a similar task:
Task: put an apple in the fridge.
History so far:
(none yet)
Admissible actions: ['go to diningtable 1', 'go to countertop 1', 'go to fridge 1', 'look']
Thought: I need to find an apple first. I will check likely locations like the dining table.
Action: go to diningtable 1
You go to the diningtable 1. On the diningtable 1, you see an apple 1, a knife 1.
Admissible actions: ['take apple 1 from diningtable 1', 'take knife 1 from diningtable 1', 'go to fridge 1', 'look']
Thought: I found the apple. I should pick it up now.
Action: take apple 1 from diningtable 1
You pick up the apple 1 from the diningtable 1.
Admissible actions: ['go to fridge 1', 'go to countertop 1', 'look']
Thought: Now that I am holding the apple, I should go to the fridge to place it there as the task requires.
Action: go to fridge 1
You go to the fridge 1. The fridge 1 is closed.
Admissible actions: ['open fridge 1', 'go to countertop 1', 'look']
Thought: The fridge is closed, so I need to open it before I can place the apple inside.
Action: open fridge 1
You open the fridge 1. The fridge 1 is open. In it, you see nothing.
Admissible actions: ['put apple 1 in/on fridge 1', 'close fridge 1', 'go to countertop 1']
Thought: The fridge is open and I am holding the apple, so I can now place it inside to complete the task.
Action: put apple 1 in/on fridge 1
You put the apple 1 in/on the fridge 1. Task complete.

Example 2 of solving a similar task (with an intermediate appliance step):
Task: heat a mug and put it on the countertop.
History so far:
(none yet)
Admissible actions: ['go to cabinet 1', 'go to microwave 1', 'go to countertop 1', 'look']
Thought: I need to find a mug first. I will check nearby cabinets, since mugs are often stored there.
Action: go to cabinet 1
You go to cabinet 1. The cabinet 1 is closed.
Admissible actions: ['open cabinet 1', 'go to microwave 1', 'go to countertop 1']
Thought: The cabinet is closed, so I need to open it to see what is inside.
Action: open cabinet 1
You open cabinet 1. The cabinet 1 is open. In it, you see a mug 1.
Admissible actions: ['take mug 1 from cabinet 1', 'go to microwave 1', 'close cabinet 1']
Thought: I found the mug. I should pick it up now.
Action: take mug 1 from cabinet 1
You pick up the mug 1 from the cabinet 1.
Admissible actions: ['go to microwave 1', 'go to countertop 1', 'look']
Thought: The task requires heating the mug, and heating is done with the microwave, so I should go there next before placing it on the countertop.
Action: go to microwave 1
You go to the microwave 1. The microwave 1 is closed.
Admissible actions: ['open microwave 1', 'go to countertop 1']
Thought: The microwave is closed, so I need to open it before I can heat the mug inside it.
Action: open microwave 1
You open the microwave 1. The microwave 1 is open. It is empty.
Admissible actions: ['put mug 1 in/on microwave 1', 'go to countertop 1']
Thought: I should place the mug inside the now-open microwave so I can heat it.
Action: put mug 1 in/on microwave 1
You put the mug 1 in/on the microwave 1.
Admissible actions: ['heat mug 1 with microwave 1', 'close microwave 1', 'go to countertop 1']
Thought: The mug is inside the microwave, so I can now issue the heat action.
Action: heat mug 1 with microwave 1
You heat the mug 1 using the microwave 1.
Admissible actions: ['take mug 1 from microwave 1', 'close microwave 1']
Thought: The mug is heated. I need to take it out and carry it to the countertop to finish the task.
Action: take mug 1 from microwave 1
You take the mug 1 from the microwave 1.
Admissible actions: ['go to countertop 1', 'go to microwave 1']
Thought: Now I should go to the countertop to place the heated mug there, completing the task.
Action: go to countertop 1
You go to the countertop 1.
Admissible actions: ['put mug 1 in/on countertop 1', 'go to microwave 1']
Thought: I am at the countertop holding the heated mug, so I can now place it here to finish the task.
Action: put mug 1 in/on countertop 1
You put the mug 1 in/on the countertop 1. Task complete.

Now solve the following new task the same way -- one Thought and one Action per turn, using ONLY the actions listed as admissible for THIS task:
"""


@dataclass
class StepRecord:
    observation: str
    admissible_actions: list[str]
    thought: str
    action: str
    action_was_admissible: bool
    input_tokens: int
    output_tokens: int
    cached: bool
    oom_retries: int = 0


@dataclass
class EpisodeResult:
    success: bool
    steps: list[StepRecord] = field(default_factory=list)
    hit_step_cap: bool = False
    parse_failures: int = 0


def _build_prompt(
    goal_obs: str,
    memory_texts: list[str],
    history: list[StepRecord],
    admissible_actions: list[str],
    few_shot_text: str | None = None,
) -> list[dict]:
    memory_block = ""
    if memory_texts:
        memory_block = "Relevant lessons from past episodes:\n" + "\n".join(f"- {m}" for m in memory_texts) + "\n\n"

    history_block = ""
    for step in history:
        history_block += f"{step.observation}\n> {step.action}\n"

    user_content = (
        f"{few_shot_text or ''}"
        f"{memory_block}"
        f"Task: {goal_obs}\n\n"
        f"History so far:\n{history_block if history_block else '(none yet)'}\n\n"
        f"Admissible actions: {admissible_actions}\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_action(text: str, admissible_actions: list[str]) -> tuple[str, str, bool]:
    """Returns (thought, action, action_was_admissible). Falls back to the
    first admissible action (logged as a parse failure) if no valid Action
    line is found -- keeps the episode running rather than crashing on a
    malformed completion."""
    thought_match = re.search(r"^\s*Thought:\s*(.+?)\s*$", text, re.MULTILINE)
    thought = thought_match.group(1) if thought_match else ""

    action_match = ACTION_LINE_RE.search(text)
    if action_match:
        candidate = action_match.group(1).strip()
        if candidate in admissible_actions:
            return thought, candidate, True
        for adm in admissible_actions:
            if adm.strip() == candidate.strip():
                return thought, adm, True
    return thought, admissible_actions[0], False


def run_episode(
    env,
    llm_client: LLMClient,
    memory_texts: list[str],
    max_steps: int,
    obs: str,
    info: dict,
    few_shot_text: str | None = None,
) -> EpisodeResult:
    """`obs`/`info` must come from the SAME `env.reset()` call whose
    `task_type` drove memory retrieval upstream -- this function must not
    reset the env itself, since a second reset would (a) hand the agent a
    different task instance than the one memories were retrieved for, and
    (b) advance a real ALFWorld env's game iterator an extra, unaccounted-for
    step per episode.

    few_shot_text, if given, is prepended to every step's prompt (see
    FEW_SHOT_EXAMPLES / capability_check.py) -- default None preserves the
    original zero-shot prompt unchanged for every existing caller."""
    admissible = info["admissible_commands"][0] if isinstance(info["admissible_commands"], list) and info["admissible_commands"] and isinstance(info["admissible_commands"][0], list) else info["admissible_commands"]

    result = EpisodeResult(success=False)
    history: list[StepRecord] = []

    for _step_idx in range(max_steps):
        messages = _build_prompt(obs, memory_texts, history, admissible, few_shot_text=few_shot_text)
        response = llm_client.complete(messages, stop=["\n\n"])
        thought, action, was_admissible = _parse_action(response.text, admissible)
        if not was_admissible:
            result.parse_failures += 1

        step_record = StepRecord(
            observation=obs,
            admissible_actions=list(admissible),
            thought=thought,
            action=action,
            action_was_admissible=was_admissible,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached=response.cached,
            oom_retries=response.oom_retries,
        )
        history.append(step_record)

        obs, _reward, done, info = env.step(action)
        admissible = info["admissible_commands"][0] if isinstance(info["admissible_commands"], list) and info["admissible_commands"] and isinstance(info["admissible_commands"][0], list) else info["admissible_commands"]

        if done:
            result.success = bool(info.get("won", [False])[0]) if isinstance(info.get("won"), list) else bool(info.get("won", False))
            result.steps = history
            return result

    result.steps = history
    result.hit_step_cap = True
    return result
