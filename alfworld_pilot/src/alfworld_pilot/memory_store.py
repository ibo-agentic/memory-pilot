"""A store of short "lesson" memories (100-300 tokens each -- distilled
takeaways, not full trajectories) plus a similarity function used to build
each task's top-M candidate set.

Similarity is currently a MOCK placeholder (task-type match + noise), not a
real embedding model -- computing real embeddings would mean an API call,
which this project's caching/cost-control machinery is built for but Part B
makes none of. Swap `similarity_score` for a real embedding lookup once
Stage 2 goes live; nothing else in retrieval.py or episode_runner.py needs
to change, since they only depend on getting a dict[mem_id, float] back.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .env_interface import TASK_TYPES

LESSON_TEMPLATES = {
    "pick_and_place_simple": "When the task is to move an object to a receptacle, first locate the object, pick it up, then navigate to the target receptacle before placing it. Double-check the object is the one named in the goal before picking it up.",
    "look_at_obj_in_light": "For examine-under-light tasks, bring the target object to the lamp (or the lamp's location) before turning it on -- turning the lamp on first and then walking away loses the light source's effect.",
    "pick_clean_then_place_in_recep": "Cleaning tasks require using the sink/basin explicitly (e.g. 'clean X with sinkbasin 1') after picking the object up, before carrying it to the final receptacle. Skipping the clean step leaves the goal unsatisfied even if the object ends up in the right place.",
    "pick_heat_then_place_in_recep": "Heating tasks require the microwave or stove as an intermediate step: pick up the object, heat it there explicitly, then move it to the destination. The object must be heated before, not after, placement.",
    "pick_cool_then_place_in_recep": "Cooling tasks use the fridge as the intermediate step, analogous to heating: pick up, cool via the fridge, then place. Placing before cooling does not satisfy the goal.",
    "pick_two_obj_and_place": "When two of the same object type are needed, place the first one, then return for the second rather than trying to carry both at once -- the agent can typically only hold one object.",
}


@dataclass(frozen=True)
class Memory:
    mem_id: str
    text: str
    task_type: str  # the task type this lesson was distilled from / is most relevant to
    approx_tokens: int


def _pad_to_token_range(text: str, min_tokens: int, max_tokens: int, rng: random.Random) -> str:
    filler = (
        " This was learned from a past episode and generalizes to similar situations "
        "encountered later in the same kind of household task."
    )
    approx_tokens = len(text.split())
    while approx_tokens < min_tokens:
        text += filler
        approx_tokens = len(text.split())
    words = text.split()
    target = rng.randint(min_tokens, max_tokens)
    if len(words) > target:
        words = words[:target]
    return " ".join(words)


def build_mock_store(n_memories: int, lesson_min_tokens: int, lesson_max_tokens: int, seed: int) -> list[Memory]:
    rng = random.Random(seed)
    memories = []
    for i in range(n_memories):
        task_type = TASK_TYPES[i % len(TASK_TYPES)]
        base_text = LESSON_TEMPLATES[task_type]
        variant_text = f"(lesson {i}) {base_text}"
        text = _pad_to_token_range(variant_text, lesson_min_tokens, lesson_max_tokens, rng)
        memories.append(Memory(mem_id=f"mem_{i}", text=text, task_type=task_type, approx_tokens=len(text.split())))
    return memories


def similarity_scores(memories: list[Memory], current_task_type: str, rng: random.Random) -> dict[str, float]:
    """Mock similarity: memories tagged with the current task type score
    higher on average, with noise -- so retrieval is topic-aware but not
    perfectly deterministic, mirroring a real (imperfect) embedding
    retriever."""
    scores = {}
    for mem in memories:
        base = 0.7 if mem.task_type == current_task_type else 0.3
        scores[mem.mem_id] = base + rng.uniform(-0.2, 0.2)
    return scores
