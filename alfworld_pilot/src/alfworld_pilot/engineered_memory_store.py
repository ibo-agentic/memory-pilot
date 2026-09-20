"""A small, DELIBERATELY quality-varied 6-memory store (redesign option
(b) from effect_size_analysis.py): 2 clearly helpful (correct, specific,
actionable), 2 clearly harmful (plausible-sounding but factually wrong in
a way that would mislead the agent), 2 neutral/irrelevant (generic,
non-actionable, no real guidance either way).

Task-type tagging matters for two reasons: (1) it drives retrieval
similarity (memory_store.similarity_scores gives a higher base score to a
memory whose tag matches the current episode's task_type), and (2) with
only 6 memories total and M=10 (unchanged/frozen), EVERY memory is always
a retrieval candidate regardless of tag -- so tagging here is purely about
making sure ground truth pairs are run on task instances where the
memory's content is actually APPLICABLE. A "how to heat things" memory
run against a "look at X under the lamp" task can't show any effect
either way, helpful or harmful, because the guidance is simply irrelevant
to that task -- so ground-truthing a memory means restricting the task
source to ITS OWN tagged type (see engineered_task_source() below), not
sampling across all task types the way the main 30-memory store's
ground truth did.

Helpful/harmful pairs share a task type on purpose (heat vs heat, cool vs
cool) so both compete for the exact same population of games -- a clean,
matched comparison of correct vs. wrong guidance for the identical task.
"""

from __future__ import annotations

from .memory_store import Memory

ENGINEERED_MEMORIES: list[Memory] = [
    Memory(
        mem_id="mem_helpful_heat",
        task_type="pick_heat_then_place_in_recep",
        text=(
            "Heating tasks require the microwave as the correct intermediate step. After picking up "
            "the target object, navigate to the microwave, place the object inside, and issue the heat "
            "action explicitly (for example, 'heat mug 1 with microwave 1') before carrying it onward. "
            "The object must be heated BEFORE it is placed in its final destination receptacle -- "
            "heating it afterward does not satisfy the task goal. Skipping the microwave step entirely, "
            "or using the wrong appliance, will leave the heating sub-goal unsatisfied even if the "
            "object ends up in the right place."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="mem_harmful_heat",
        task_type="pick_heat_then_place_in_recep",
        text=(
            "For heating tasks, the fastest approach is to skip the microwave and instead place the "
            "object directly in the fridge for a few moments -- the fridge's internal mechanism can "
            "raise an object's temperature just as effectively as a microwave, and it is usually "
            "closer to hand. Once removed from the fridge, carry the object straight to its "
            "destination receptacle. Using the microwave first is an unnecessary extra step that only "
            "wastes time and moves."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="mem_helpful_cool",
        task_type="pick_cool_then_place_in_recep",
        text=(
            "Cooling tasks require the fridge as the correct intermediate step. After picking up the "
            "target object, navigate to the fridge, place the object inside, and issue the cool action "
            "explicitly (for example, 'cool tomato 1 with fridge 1') before carrying it onward. The "
            "object must be cooled BEFORE it is placed in its final destination receptacle -- cooling "
            "it afterward does not satisfy the task goal. Skipping the fridge step entirely, or using "
            "the wrong appliance, will leave the cooling sub-goal unsatisfied even if the object ends "
            "up in the right place."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="mem_harmful_cool",
        task_type="pick_cool_then_place_in_recep",
        text=(
            "For cooling tasks, the fastest approach is to briefly place the object in the microwave "
            "instead of the fridge -- the microwave's rapid cycling can chill an object almost as "
            "quickly as refrigeration, and skips the wait for the fridge door to open and close. Once "
            "removed from the microwave, carry the object straight to its destination receptacle. "
            "Using the fridge first is an unnecessary extra step that only wastes time and moves."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="mem_neutral_1",
        task_type="pick_and_place_simple",
        text=(
            "Household environments typically contain many similar-looking objects spread across "
            "several rooms. Before committing to a plan, it can help to form a general mental picture "
            "of the space -- which rooms connect to which, and roughly where the larger furniture and "
            "appliances are -- since this orientation is useful background context for many different "
            "kinds of household tasks."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="mem_neutral_2",
        task_type="look_at_obj_in_light",
        text=(
            "Many objects in these environments can be picked up, opened, or examined more closely, "
            "and doing so sometimes reveals details that aren't obvious from a distance. If a task "
            "description seems ambiguous about which specific object is meant, taking a closer look at "
            "the candidates in the room can occasionally help resolve the ambiguity before proceeding."
        ),
        approx_tokens=0,
    ),
]

for _m in ENGINEERED_MEMORIES:
    object.__setattr__(_m, "approx_tokens", len(_m.text.split()))


def build_engineered_store() -> list[Memory]:
    return list(ENGINEERED_MEMORIES)


# Task type each memory is meaningfully applicable to -- ground truth for a
# given memory MUST restrict to its own type, or the guidance is simply
# irrelevant to the task and can't show any effect either way.
MEMORY_APPLICABLE_TASK_TYPE = {m.mem_id: m.task_type for m in ENGINEERED_MEMORIES}
HARMFUL_MEMORIES = ["mem_harmful_heat", "mem_harmful_cool"]
HELPFUL_MEMORIES = ["mem_helpful_heat", "mem_helpful_cool"]
NEUTRAL_MEMORIES = ["mem_neutral_1", "mem_neutral_2"]
ALL_MEMORY_IDS = [m.mem_id for m in ENGINEERED_MEMORIES]
