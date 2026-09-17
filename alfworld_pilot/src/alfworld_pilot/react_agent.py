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


@dataclass
class EpisodeResult:
    success: bool
    steps: list[StepRecord] = field(default_factory=list)
    hit_step_cap: bool = False
    parse_failures: int = 0


def _build_prompt(goal_obs: str, memory_texts: list[str], history: list[StepRecord], admissible_actions: list[str]) -> list[dict]:
    memory_block = ""
    if memory_texts:
        memory_block = "Relevant lessons from past episodes:\n" + "\n".join(f"- {m}" for m in memory_texts) + "\n\n"

    history_block = ""
    for step in history:
        history_block += f"{step.observation}\n> {step.action}\n"

    user_content = (
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


def run_episode(env, llm_client: LLMClient, memory_texts: list[str], max_steps: int) -> EpisodeResult:
    obs, info = env.reset()
    admissible = info["admissible_commands"][0] if isinstance(info["admissible_commands"], list) and info["admissible_commands"] and isinstance(info["admissible_commands"][0], list) else info["admissible_commands"]

    result = EpisodeResult(success=False)
    history: list[StepRecord] = []

    for _step_idx in range(max_steps):
        messages = _build_prompt(obs, memory_texts, history, admissible)
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
