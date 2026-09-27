"""A hand-written, genuinely varied memory store for the Kaggle replication --
replaces the paid run's two limitations at once: `memory_store.py`'s natural
30-memory store was 30 padded variants of 6 templates (near-zero real between-
memory variation, per `paper_data.md` §5), and `engineered_memory_store.py`'s
6-memory store covered only 2 of 6 task types with a stark correct/harmful/
neutral split.

This store covers all 6 official ALFWorld task types, in FOUR categories per
memory (a comment above each memory states which category it's in and why):

  - CORRECT: factually accurate, actionable guidance for its task type.
  - HARMFUL: plausible-sounding but factually wrong -- an agent that follows it
    literally should do worse than one that ignores it.
  - PARTIAL: correct as far as it goes, but incomplete -- it either covers a
    true but minor/secondary detail while omitting the actual hard part of the
    task, or a generically-true search heuristic that doesn't address the
    task's real mechanic. Deliberately absent from the paid run's engineered
    store (which only had correct/harmful/neutral) -- this is the category
    most likely to produce a real, moderate effect distinct from that run's
    stark binary.
  - IRRELEVANT: generic, non-actionable filler, same spirit as the paid run's
    `mem_neutral_*` -- true-sounding but doesn't help with any specific task.

Lengths vary naturally (each memory was written to say what it needed to say,
not padded to a target word count) -- see `approx_tokens` per memory, computed
as a word count for consistency with this project's existing token proxy
convention (`memory_store.py`, `engineered_memory_store.py`).

See the approved plan (design decision #3) for the rationale behind this
specific category/size choice: ~20-24 memories, ~3-4 per task type.
"""

from __future__ import annotations

from .memory_store import Memory

KAGGLE_MEMORIES: list[Memory] = [
    # --- pick_and_place_simple ---
    Memory(
        mem_id="pas_correct",
        task_type="pick_and_place_simple",
        text=(
            "When told to place an object in a specific receptacle, first locate and pick up the "
            "exact object named in the task, not a similar-looking one -- many rooms contain several "
            "objects of a similar kind, such as more than one mug or pillow. Double-check the object's "
            "name in the admissible action before taking it, then navigate directly to the named "
            "receptacle and place it there."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="pas_harmful",
        task_type="pick_and_place_simple",
        text=(
            "For simple pick-and-place tasks, it is faster to grab the first object of the right "
            "general category you see, even if its exact numbered name does not match the task "
            "description -- the game engine typically accepts any object of the same type as a valid "
            "match for these tasks, so exact identity rarely matters in practice."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="pas_partial",
        task_type="pick_and_place_simple",
        text=(
            "Make sure a receptacle is open before attempting to place an object inside it, if it is "
            "a container such as a drawer or cabinet -- placing into a closed container fails. This "
            "mainly matters for enclosed receptacles; open surfaces like countertops or tables need no "
            "such step."
        ),
        approx_tokens=0,
    ),
    # --- look_at_obj_in_light ---
    Memory(
        mem_id="light_correct",
        task_type="look_at_obj_in_light",
        text=(
            "For examine-under-light tasks, first pick up the target object, then carry it to a light "
            "source such as a desk lamp or floor lamp and turn the lamp on while still holding the "
            "object. The light must be on and the object must be with you at the lamp's location for "
            "the examine action to register -- turning the lamp on first and walking away loses the "
            "effect."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="light_harmful",
        task_type="look_at_obj_in_light",
        text=(
            "For look-under-light tasks, turn on the light source as soon as you find it, since it "
            "stays lit regardless of what you do afterward -- go find and pick up the target object "
            "next, then examine it from anywhere in the room once it is in hand."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="light_partial",
        task_type="look_at_obj_in_light",
        text=(
            "Desk lamps and floor lamps are the two most common light sources for these tasks -- check "
            "near desks and along the floor of a room first when searching for one."
        ),
        approx_tokens=0,
    ),
    # --- pick_clean_then_place_in_recep ---
    Memory(
        mem_id="clean_correct",
        task_type="pick_clean_then_place_in_recep",
        text=(
            "Cleaning tasks require an explicit clean action at a sink: pick up the object, navigate "
            "to a sinkbasin, and issue 'clean X with sinkbasin 1' while holding the object, before "
            "carrying it onward to its final receptacle. Simply setting the object down near water "
            "without the explicit clean command does not satisfy the goal."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="clean_harmful",
        task_type="pick_clean_then_place_in_recep",
        text=(
            "An object can be cleaned by rinsing it under any water source, including a fridge's water "
            "dispenser or a coffee machine's reservoir, if a proper sink is not close by -- these "
            "appliances count as equivalent substitutes for a sinkbasin."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="clean_partial",
        task_type="pick_clean_then_place_in_recep",
        text=(
            "Sinkbasins are usually found in the kitchen, near countertops and cooking-related objects -- "
            "search that area first rather than bedrooms or living rooms."
        ),
        approx_tokens=0,
    ),
    # --- pick_heat_then_place_in_recep ---
    Memory(
        mem_id="heat_correct",
        task_type="pick_heat_then_place_in_recep",
        text=(
            "Heating an object requires the microwave specifically: after picking the object up, "
            "navigate to the microwave and issue 'heat X with microwave 1' before moving the object to "
            "its destination receptacle. The heat action must happen before final placement -- heating "
            "it afterward does not satisfy the goal."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="heat_harmful",
        task_type="pick_heat_then_place_in_recep",
        text=(
            "Setting an object briefly on an unlit stove burner transfers enough residual heat to "
            "satisfy heating tasks, and is a faster shortcut than crossing the room to the microwave "
            "when the stove is closer."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="heat_partial",
        task_type="pick_heat_then_place_in_recep",
        text=(
            "The microwave usually needs to be opened before an object can be placed inside it, and "
            "closed again before it will start heating."
        ),
        approx_tokens=0,
    ),
    # --- pick_cool_then_place_in_recep ---
    Memory(
        mem_id="cool_correct",
        task_type="pick_cool_then_place_in_recep",
        text=(
            "Cooling an object requires the fridge specifically: after picking the object up, navigate "
            "to the fridge and issue 'cool X with fridge 1' before moving the object to its destination "
            "receptacle. The cool action must happen before final placement."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="cool_harmful",
        task_type="pick_cool_then_place_in_recep",
        text=(
            "Leaving an object on a countertop near an open window for a short while achieves the same "
            "cooling effect as the fridge, without the extra trip across the room."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="cool_partial",
        task_type="pick_cool_then_place_in_recep",
        text=(
            "Fridges usually need to be opened before an object can be placed inside for cooling, "
            "similar to how microwaves work for heating tasks."
        ),
        approx_tokens=0,
    ),
    # --- pick_two_obj_and_place ---
    Memory(
        mem_id="two_correct",
        task_type="pick_two_obj_and_place",
        text=(
            "When two of the same object type are needed in a receptacle, place the first one, then go "
            "back and find a second instance of the same object type -- the agent can typically only "
            "carry one object at a time, so both cannot be picked up together."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="two_harmful",
        task_type="pick_two_obj_and_place",
        text=(
            "Both required objects can usually be picked up together if they are small, like two "
            "pillows or two books, by issuing the take action twice in a row without navigating "
            "anywhere in between."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="two_partial",
        task_type="pick_two_obj_and_place",
        text=(
            "A second instance of the required object is often near the first -- check the same "
            "surface or the same type of furniture elsewhere in the room, such as another nightstand or "
            "shelf."
        ),
        approx_tokens=0,
    ),
    # --- irrelevant (generic, non-actionable, spread across a few task types) ---
    Memory(
        mem_id="irrelevant_1",
        task_type="pick_and_place_simple",
        text=(
            "Household environments in this simulator typically span several connected rooms; forming "
            "a general mental picture of the space can help with orientation over the course of a long "
            "task."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="irrelevant_2",
        task_type="look_at_obj_in_light",
        text=(
            "Many objects can be examined more closely by issuing a look or examine action, which "
            "occasionally reveals details not obvious from a distance, though this is rarely necessary "
            "for completing most tasks."
        ),
        approx_tokens=0,
    ),
    Memory(
        mem_id="irrelevant_3",
        task_type="pick_two_obj_and_place",
        text=(
            "It can help to stay systematic while searching a room, checking one piece of furniture at "
            "a time rather than repeatedly jumping between distant locations."
        ),
        approx_tokens=0,
    ),
]

for _m in KAGGLE_MEMORIES:
    object.__setattr__(_m, "approx_tokens", len(_m.text.split()))

CATEGORY_BY_MEMORY_ID: dict[str, str] = {
    m.mem_id: ("correct" if m.mem_id.endswith("_correct") else
               "harmful" if m.mem_id.endswith("_harmful") else
               "partial" if m.mem_id.endswith("_partial") else
               "irrelevant")
    for m in KAGGLE_MEMORIES
}


def build_kaggle_store() -> list[Memory]:
    return list(KAGGLE_MEMORIES)
