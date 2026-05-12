"""Generate a synthetic commonsense position-control dataset.

Produces ~150 examples across 5 commonsense reasoning domains:
  - causal, spatial, social/emotional, counterfactual, temporal

Each record contains:
  id, question, correct_answer, steps (list), chain_format (str),
  question_only_format (str), domain, difficulty, source, metadata.

chain_format   : "Let me think step by step..." chain with explicit final answer.
question_only_format : same chain but final answer sentence is omitted.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "phase0_position_control_commonsense_v1.jsonl"

random.seed(42)

# ---------------------------------------------------------------------------
# Template building blocks
# ---------------------------------------------------------------------------

PEOPLE = ["Alice", "Ben", "Carlos", "Diana", "Eve", "Frank", "Grace", "Henry",
          "Iris", "Jack"]
CONTAINERS = ["box", "bag", "basket", "drawer", "cabinet"]
LIQUIDS = ["water", "juice", "milk", "coffee", "tea"]
LOCATIONS = ["kitchen", "garden", "bedroom", "office", "park"]
OBJECTS = ["book", "lamp", "chair", "phone", "key"]
DIRECTIONS = ["left", "right", "north", "south", "east", "west"]
OPPOSITE = {"left": "right", "right": "left", "north": "south",
            "south": "north", "east": "west", "west": "east"}
EMOTIONS_POS = ["happy", "excited", "grateful", "proud", "relieved"]
EMOTIONS_NEG = ["sad", "anxious", "embarrassed", "frustrated", "disappointed"]
WEATHER = ["sunny", "rainy", "windy", "snowy", "foggy"]
ANIMALS = ["cat", "dog", "bird", "rabbit", "horse"]
ACTIVITIES = ["reading", "cooking", "painting", "cycling", "gardening"]

# ---------------------------------------------------------------------------
# Category 1: Causal reasoning (30 examples)
# ---------------------------------------------------------------------------

def make_causal_records() -> list[dict]:
    templates = [
        # Template C1: liquid spill
        lambda i: {
            "tpl": "C1", "var": i,
            "person": PEOPLE[i % len(PEOPLE)],
            "liquid": LIQUIDS[i % len(LIQUIDS)],
            "container": CONTAINERS[i % len(CONTAINERS)],
            "location": LOCATIONS[i % len(LOCATIONS)],
        },
        # Template C2: fire and smoke
        lambda i: {
            "tpl": "C2", "var": i,
            "person": PEOPLE[(i + 2) % len(PEOPLE)],
            "object": OBJECTS[i % len(OBJECTS)],
            "location": LOCATIONS[(i + 1) % len(LOCATIONS)],
        },
        # Template C3: rain and umbrella
        lambda i: {
            "tpl": "C3", "var": i,
            "person": PEOPLE[(i + 4) % len(PEOPLE)],
            "weather": WEATHER[i % len(WEATHER)],
        },
        # Template C4: broken item causes problem
        lambda i: {
            "tpl": "C4", "var": i,
            "person": PEOPLE[(i + 6) % len(PEOPLE)],
            "object": OBJECTS[i % len(OBJECTS)],
        },
        # Template C5: food left out
        lambda i: {
            "tpl": "C5", "var": i,
            "person": PEOPLE[(i + 1) % len(PEOPLE)],
            "liquid": LIQUIDS[i % len(LIQUIDS)],
        },
        # Template C6: animal care
        lambda i: {
            "tpl": "C6", "var": i,
            "person": PEOPLE[(i + 3) % len(PEOPLE)],
            "animal": ANIMALS[i % len(ANIMALS)],
        },
    ]

    def build(tmpl_fn, i):
        ctx = tmpl_fn(i)
        t = ctx["tpl"]
        if t == "C1":
            p, liq, con, loc = ctx["person"], ctx["liquid"], ctx["container"], ctx["location"]
            q = f"{p} accidentally tips over a {con} full of {liq} in the {loc}. What is the most likely immediate result?"
            ans = f"The {liq} spills onto the floor"
            steps = [
                f"The {con} was full of {liq} and was tipped over.",
                f"When a container tips over, its contents pour out.",
                f"The {liq} was inside the {con}, so it will spill onto the surface below.",
                f"Therefore, the most likely immediate result is that the {liq} spills onto the floor.",
            ]
        elif t == "C2":
            p, obj, loc = ctx["person"], ctx["object"], ctx["location"]
            q = f"{p} notices smoke coming from the {loc}. What should {p} do first?"
            ans = "Alert others and evacuate"
            steps = [
                f"Smoke in the {loc} suggests a possible fire.",
                "When there is a fire risk, personal safety is the top priority.",
                "The safest immediate action is to warn others and leave the area.",
                f"Therefore, {p} should first alert others and evacuate.",
            ]
        elif t == "C3":
            p, wx = ctx["person"], ctx["weather"]
            q = f"It is {wx} outside and {p} needs to walk to the store. What item will {p} most likely bring?"
            if wx == "rainy":
                ans = "an umbrella"
            elif wx == "sunny":
                ans = "sunglasses"
            elif wx == "snowy":
                ans = "a coat"
            elif wx == "windy":
                ans = "a jacket"
            else:
                ans = "a scarf"
            steps = [
                f"The weather outside is {wx}.",
                f"People choose accessories based on the current weather conditions.",
                f"For {wx} weather, the appropriate item is {ans}.",
                f"Therefore, {p} will most likely bring {ans}.",
            ]
        elif t == "C4":
            p, obj = ctx["person"], ctx["object"]
            q = f"{p}'s {obj} is broken and cannot be used. What problem will {p} most likely face?"
            ans = f"{p} cannot perform tasks that require the {obj}"
            steps = [
                f"The {obj} is broken and non-functional.",
                f"Any activity that depends on the {obj} cannot be completed.",
                f"Therefore, {p} will most likely face the problem that {p} cannot perform tasks that require the {obj}.",
            ]
        elif t == "C5":
            p, liq = ctx["person"], ctx["liquid"]
            q = f"{p} leaves a glass of {liq} uncovered on the counter for several hours. What is most likely to happen?"
            ans = f"The {liq} will evaporate partially and may attract insects"
            steps = [
                f"The {liq} is left open and exposed to air for several hours.",
                "Liquids left uncovered tend to evaporate over time.",
                "Uncovered food or drink can attract insects.",
                f"Therefore, the {liq} will most likely evaporate partially and may attract insects.",
            ]
        else:  # C6
            p, animal = ctx["person"], ctx["animal"]
            q = f"{p} forgets to feed their {animal} for a day. What will the {animal} most likely do?"
            ans = f"The {animal} will become hungry and seek food"
            steps = [
                f"The {animal} has not been fed for a full day.",
                "Animals need regular food to maintain energy.",
                "When hungry, animals typically search for or signal their need for food.",
                f"Therefore, the {animal} will most likely become hungry and seek food.",
            ]
        return q, ans, steps

    records = []
    for tmpl_fn in templates:
        for i in range(5):
            q, ans, steps = build(tmpl_fn, i)
            records.append(("causal", q, ans, steps))
    return records

# ---------------------------------------------------------------------------
# Category 2: Spatial / directional (30 examples)
# ---------------------------------------------------------------------------

def make_spatial_records() -> list[dict]:
    templates = [
        lambda i: ("S1", i, PEOPLE[i % len(PEOPLE)], DIRECTIONS[i % 4], OBJECTS[i % len(OBJECTS)]),
        lambda i: ("S2", i, PEOPLE[(i+2) % len(PEOPLE)], DIRECTIONS[(i+1) % 4], LOCATIONS[i % len(LOCATIONS)]),
        lambda i: ("S3", i, PEOPLE[(i+4) % len(PEOPLE)], DIRECTIONS[(i+2) % 4], ANIMALS[i % len(ANIMALS)]),
        lambda i: ("S4", i, PEOPLE[(i+6) % len(PEOPLE)], DIRECTIONS[(i+3) % 4], OBJECTS[(i+1) % len(OBJECTS)]),
        lambda i: ("S5", i, PEOPLE[(i+1) % len(PEOPLE)], DIRECTIONS[i % 4], CONTAINERS[i % len(CONTAINERS)]),
        lambda i: ("S6", i, PEOPLE[(i+3) % len(PEOPLE)], DIRECTIONS[(i+1) % 4], LOCATIONS[(i+2) % len(LOCATIONS)]),
    ]
    dirs4 = ["left", "right", "north", "south"]

    def build(td):
        t, i, p, d, obj = td
        opp = OPPOSITE[d]
        if t == "S1":
            q = f"{p} is standing directly in front of a {obj}. {p} turns to face the {d}. Which direction is the {obj} now relative to {p}?"
            ans = f"Behind {p}"
            steps = [
                f"{p} starts facing the {obj}.",
                f"{p} then turns to face {d}, rotating 180 degrees from the original orientation.",
                f"After turning, what was in front is now behind.",
                f"Therefore, the {obj} is now behind {p}.",
            ]
        elif t == "S2":
            q = f"{p} walks {d} from the {obj}. Where is the {obj} relative to {p} now?"
            ans = f"The {obj} is to the {opp} of {p}"
            steps = [
                f"{p} started at the {obj} and moved {d}.",
                f"If you move {d} from a point, that point is now to your {opp}.",
                f"Therefore, the {obj} is to the {opp} of {p}.",
            ]
        elif t == "S3":
            q = f"A {obj} is to the {d} of {p}. {p} turns around completely. Where is the {obj} relative to {p} now?"
            ans = f"To the {opp} of {p}"
            steps = [
                f"The {obj} starts to the {d} of {p}.",
                f"Turning around 180 degrees reverses the relative positions.",
                f"What was {d} is now {opp}.",
                f"Therefore, the {obj} is now to the {opp} of {p}.",
            ]
        elif t == "S4":
            q = f"{p} is standing between two chairs. One chair is to the {d}, the other is to the {opp}. {p} moves two steps to the {d}. Which chair is {p} now closer to?"
            ans = f"The chair to the {d}"
            steps = [
                f"{p} starts equidistant from both chairs.",
                f"{p} moves two steps toward the {d} chair.",
                f"Moving closer to one chair means moving farther from the other.",
                f"Therefore, {p} is now closer to the chair to the {d}.",
            ]
        elif t == "S5":
            q = f"A {obj} is directly north of a {d}. If someone walks south from the {obj}, which comes first?"
            ans = f"The {d} comes first"
            steps = [
                f"The {obj} is north and the {d} is south of the {obj}.",
                "Walking south from the {obj} means moving toward the {d}.",
                f"Therefore, the {d} comes first.",
            ]
        else:  # S6
            q = f"{p} faces {d}. {p}'s left hand is pointing in which direction?"
            lefts = {"north": "west", "south": "east", "east": "north", "west": "south",
                     "left": "south", "right": "north"}
            left_dir = lefts.get(d, "perpendicular")
            ans = left_dir
            steps = [
                f"{p} is facing {d}.",
                "When facing a cardinal direction, the left hand points 90 degrees counterclockwise.",
                f"90 degrees counterclockwise from {d} is {left_dir}.",
                f"Therefore, {p}'s left hand points {left_dir}.",
            ]
        return q, ans, steps

    records = []
    for tmpl_fn in templates:
        for i in range(5):
            td = tmpl_fn(i)
            q, ans, steps = build(td)
            records.append(("spatial", q, ans, steps))
    return records

# ---------------------------------------------------------------------------
# Category 3: Social / emotional (30 examples)
# ---------------------------------------------------------------------------

def make_social_records() -> list[dict]:
    scenarios_pos = [
        ("receives an unexpected gift", "happy"),
        ("wins a competition they trained hard for", "proud"),
        ("finds out a loved one is safe after a scare", "relieved"),
        ("is praised publicly for their work", "grateful"),
        ("hears their favorite song surprise on the radio", "excited"),
    ]
    scenarios_neg = [
        ("is criticized unfairly in front of others", "embarrassed"),
        ("misses an important appointment", "frustrated"),
        ("loses a valued possession", "sad"),
        ("is excluded from a group activity", "disappointed"),
        ("learns a plan they worked on fell through", "anxious"),
    ]

    records = []
    all_scenarios = [(e, "positive") for e in scenarios_pos] + [(e, "negative") for e in scenarios_neg]

    for idx, ((scenario, emotion), valence) in enumerate(all_scenarios):
        person = PEOPLE[idx % len(PEOPLE)]
        q = f"{person} {scenario}. What emotion is {person} most likely feeling?"
        ans = emotion
        steps = [
            f"{person} experienced the situation: {scenario}.",
            f"This situation is a {valence} event.",
            f"People typically feel {emotion} in such situations.",
            f"Therefore, {person} is most likely feeling {emotion}.",
        ]
        records.append(("social", q, ans, steps))

    # 20 more varied social templates to reach 30 total
    extras = [
        ("E1", "is given a surprise party by friends", "overjoyed"),
        ("E1", "is given a surprise party by strangers", "surprised"),
        ("E2", "accidentally breaks a friend's favorite item", "guilty"),
        ("E2", "accidentally breaks their own item", "frustrated"),
        ("E3", "helps a stranger find their lost pet", "helpful and satisfied"),
        ("E3", "helps a friend move to a new home", "tired but happy"),
        ("E4", "waits an hour for a friend who never shows up", "let down"),
        ("E4", "waits only a few minutes for a friend who is late", "understanding"),
        ("E5", "tells a joke that no one laughs at", "awkward"),
        ("E5", "tells a story that makes everyone laugh", "pleased"),
        ("E6", "receives a sincere apology from someone who hurt them", "forgiving"),
        ("E6", "is asked to apologize for something they didn't do", "indignant"),
        ("E7", "completes a difficult task they have been dreading", "relieved and accomplished"),
        ("E7", "is asked to redo a task they just completed", "frustrated"),
        ("E8", "discovers a friend kept an important secret from them", "hurt"),
        ("E8", "learns a surprise they were planning for a friend was spoiled", "disappointed"),
        ("E9", "is complimented on their appearance by a colleague", "flattered"),
        ("E9", "is ignored when they greet someone in the hallway", "slighted"),
        ("E10", "finds money they forgot they had in an old coat", "pleasantly surprised"),
        ("E10", "discovers they overpaid a bill by a large amount", "frustrated"),
    ]
    for idx, (tpl, scenario, emotion) in enumerate(extras):
        person = PEOPLE[(idx + 5) % len(PEOPLE)]
        q = f"{person} {scenario}. What emotion is {person} most likely feeling?"
        ans = emotion
        steps = [
            f"{person} experienced: {scenario}.",
            f"This kind of experience typically evokes a specific emotional response.",
            f"People who {scenario} commonly feel {emotion}.",
            f"Therefore, {person} is most likely feeling {emotion}.",
        ]
        records.append(("social", q, ans, steps))

    return records[:30]

# ---------------------------------------------------------------------------
# Category 4: Counterfactual (30 examples)
# ---------------------------------------------------------------------------

def make_counterfactual_records() -> list[dict]:
    templates = [
        # CF1
        lambda i: {
            "tpl": "CF1", "person": PEOPLE[i % len(PEOPLE)],
            "action": "set their alarm", "outcome": "wake up on time",
            "counterfactual_outcome": "oversleep",
        },
        # CF2
        lambda i: {
            "tpl": "CF2", "person": PEOPLE[(i+2) % len(PEOPLE)],
            "action": "charged their phone overnight",
            "outcome": "have a full battery",
            "counterfactual_outcome": "have a dead battery",
        },
        # CF3
        lambda i: {
            "tpl": "CF3", "person": PEOPLE[(i+4) % len(PEOPLE)],
            "action": "brought an umbrella", "outcome": "stay dry",
            "counterfactual_outcome": "get wet in the rain",
        },
        # CF4
        lambda i: {
            "tpl": "CF4", "person": PEOPLE[(i+6) % len(PEOPLE)],
            "action": "locked the door", "outcome": "keep the house secure",
            "counterfactual_outcome": "leave the house unsecured",
        },
        # CF5
        lambda i: {
            "tpl": "CF5", "person": PEOPLE[(i+1) % len(PEOPLE)],
            "action": "sent the email", "outcome": "communicate the information",
            "counterfactual_outcome": "fail to share the information",
        },
        # CF6
        lambda i: {
            "tpl": "CF6", "person": PEOPLE[(i+3) % len(PEOPLE)],
            "action": "water the plants", "outcome": "keep them alive",
            "counterfactual_outcome": "let them wilt",
        },
    ]

    def build(ctx):
        p = ctx["person"]
        action = ctx["action"]
        cfo = ctx["counterfactual_outcome"]
        q = f"If {p} had not {action}, what would most likely have happened?"
        ans = cfo.capitalize()
        steps = [
            f"The question asks what happens if {p} does NOT {action}.",
            f"The purpose of {action} is normally to {ctx['outcome']}.",
            f"Without this action, the expected outcome is reversed.",
            f"Therefore, {p} would most likely {cfo}.",
        ]
        return q, ans, steps

    records = []
    for tmpl_fn in templates:
        for i in range(5):
            ctx = tmpl_fn(i)
            q, ans, steps = build(ctx)
            records.append(("counterfactual", q, ans, steps))
    return records

# ---------------------------------------------------------------------------
# Category 5: Temporal reasoning (30 examples)
# ---------------------------------------------------------------------------

def make_temporal_records() -> list[dict]:
    sequences = [
        ("plant a seed", "water it regularly", "it grows into a plant", "a plant grows"),
        ("mix ingredients", "put the dough in the oven", "the bread is done", "bread is made"),
        ("start a fire", "add dry wood", "the fire grows larger", "the fire grows"),
        ("boil water", "add pasta", "drain and serve", "the pasta is ready"),
        ("open a window", "let air circulate", "the room cools", "the room becomes cooler"),
        ("turn off the lights", "wait for eyes to adjust", "see in the dark", "vision adjusts to darkness"),
    ]

    records = []
    for seq_idx, (step1, step2, step3, result) in enumerate(sequences):
        for var in range(5):
            person = PEOPLE[(seq_idx * 5 + var) % len(PEOPLE)]
            q = f"{person} first {step1}, then {step2}. What happens AFTER that?"
            ans = result.capitalize()
            steps_list = [
                f"{person} performs the first step: {step1}.",
                f"After that, {person} proceeds to: {step2}.",
                f"The natural consequence of these steps is that {step3}.",
                f"Therefore, what happens after is that {result}.",
            ]
            records.append(("temporal", q, ans, steps_list))

    return records[:30]

# ---------------------------------------------------------------------------
# Assemble and format records
# ---------------------------------------------------------------------------

def format_chain(question: str, steps: list[str], include_answer: bool) -> str:
    """Produce a chain-of-thought string. If include_answer, the last step
    (which must state the answer explicitly) is kept; otherwise it is dropped."""
    chain_lines = ["Let me think step by step."]
    if include_answer:
        chain_lines.extend(steps)
    else:
        # Drop the last step (which contains the explicit answer word)
        chain_lines.extend(steps[:-1])
    return "\n".join(chain_lines)


def build_records(
    causal: list, spatial: list, social: list, cf: list, temporal: list
) -> list[dict]:
    all_examples = causal + spatial + social + cf + temporal
    records = []
    for idx, (domain, question, answer, steps) in enumerate(all_examples):
        rec_id = f"commonsense-{domain[:3]}-{idx + 1:03d}"
        chain_format = format_chain(question, steps, include_answer=True)
        question_only_format = format_chain(question, steps, include_answer=False)
        records.append({
            "id": rec_id,
            "question": question,
            "correct_answer": answer,
            "steps": steps,
            "chain_format": chain_format,
            "question_only_format": question_only_format,
            "domain": "commonsense",
            "difficulty": "medium",
            "source": "synthetic_commonsense",
            "metadata": {
                "benchmark": "phase0-commonsense-v1",
                "subdomain": domain,
                "answer_format": "text",
                "answer_extraction": {
                    "strategy": "text",
                    "expected_normalized": answer.lower(),
                },
                "chain_length": len(steps),
                "has_explicit_answer_in_chain": True,
                "has_explicit_answer_in_question_only": False,
            },
        })
    return records


def main() -> None:
    causal = make_causal_records()
    spatial = make_spatial_records()
    social = make_social_records()
    cf = make_counterfactual_records()
    temporal = make_temporal_records()

    records = build_records(causal, spatial, social, cf, temporal)
    random.shuffle(records)

    output = DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(r, ensure_ascii=True) for r in records) + "\n",
        encoding="utf-8",
    )
    domain_counts: dict[str, int] = {}
    for r in records:
        d = r["metadata"]["subdomain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
    print(f"Wrote {len(records)} records to {output}")
    print(f"Domain breakdown: {domain_counts}")


if __name__ == "__main__":
    main()
