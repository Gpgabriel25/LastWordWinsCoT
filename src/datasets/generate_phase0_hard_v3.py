from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "phase0_position_control_hard_v3.jsonl"


ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def to_words(value: int) -> str:
    if value < 20:
        return ONES[value]
    if value < 100:
        tens = (value // 10) * 10
        ones = value % 10
        return TENS[tens] if ones == 0 else f"{TENS[tens]}-{ONES[ones]}"
    raise ValueError("generator only supports values under 100")


TEMPLATES = [
    {
        "domain": "warehouse count",
        "item": "sealed cartons",
        "distractor": "clipboards",
        "summary": "counted stack",
        "addition": "new sealed cartons are added",
        "removal": "damaged cartons are removed from the counted stack",
    },
    {
        "domain": "museum desk",
        "item": "visitor badges",
        "distractor": "poster tubes",
        "summary": "active badges",
        "addition": "new badges are issued",
        "removal": "expired badges must be removed from the active count",
    },
    {
        "domain": "bakery inventory",
        "item": "wrapped pastry trays",
        "distractor": "aprons",
        "summary": "counted stock",
        "addition": "new wrapped trays are added",
        "removal": "broken trays are removed from the counted stock",
    },
    {
        "domain": "travel desk",
        "item": "reserved seats",
        "distractor": "route maps",
        "summary": "reservation total",
        "addition": "new seats are booked",
        "removal": "canceled reservations are removed from the total",
    },
    {
        "domain": "library ledger",
        "item": "mystery novels",
        "distractor": "chairs",
        "summary": "shelf total",
        "addition": "returned novels are added back",
        "removal": "checked-out novels are removed from the shelf count",
    },
    {
        "domain": "clinic desk",
        "item": "active patient files",
        "distractor": "staplers",
        "summary": "active-file total",
        "addition": "new patient files are opened",
        "removal": "closed files are removed from the active count",
    },
    {
        "domain": "factory ledger",
        "item": "good bolts",
        "distractor": "tool inspections",
        "summary": "good-bolt total",
        "addition": "new good bolts are finished",
        "removal": "rejected bolts are removed from the good-bolt count",
    },
    {
        "domain": "reading club log",
        "item": "finished books",
        "distractor": "bookmarks",
        "summary": "finished-book total",
        "addition": "newly finished books are added",
        "removal": "unfinished books are removed from the finished-book count",
    },
    {
        "domain": "market cart",
        "item": "ripe pears",
        "distractor": "empty boxes",
        "summary": "sale stock",
        "addition": "ripe pears are added from storage",
        "removal": "bruised pears are removed from the sale count",
    },
    {
        "domain": "scoreboard audit",
        "item": "league points",
        "distractor": "jersey swaps",
        "summary": "league total",
        "addition": "new league points are added",
        "removal": "exhibition points are removed from the table total",
    },
]


def build_record(index: int, template: dict[str, str], rng: random.Random) -> dict[str, object]:
    starting = rng.randint(42, 78)
    removed = rng.randint(9, 18)
    added = rng.randint(12, 24)
    distractor = rng.randint(4, 11)
    answer = starting - removed + added
    prompt = (
        f"In the {template['domain']}, {to_words(starting)} {template['item']} are listed in the {template['summary']}. "
        f"{to_words(removed).capitalize()} of them {template['removal']}, {to_words(added)} more {template['addition']}, "
        f"and {to_words(distractor)} {template['distractor']} are mentioned but do not affect the count. "
        f"What is the counted total of {template['item']}?"
    )
    steps = [
        f"Let a = {starting} listed {template['item']}, b = {removed} items removed from the count, and c = {added} items added to the count. Ignore the {distractor} {template['distractor']}.",
        "Compute a - b + c.",
        f"The counted {template['summary']} is the value of that expression.",
        f"Report only the counted {template['summary']}.",
    ]
    return {
        "id": f"p0-hard3-{index:03d}",
        "question": prompt,
        "steps": steps,
        "answer": str(answer),
        "dataset": "toy-gsm8k-hard",
        "split": "study",
        "difficulty": "medium",
        "source": "generated",
        "metadata": {
            "benchmark": "phase0-position-control-hard-v3",
            "answer_format": "integer",
            "answer_extraction": {
                "strategy": "numeric",
                "expected_normalized": str(answer),
            },
            "deciding_steps": [0],
            "chain_length": 4,
            "template_domain": template["domain"],
        },
    }


def main() -> None:
    rng = random.Random(20260323)
    records = []
    for index in range(1, 61):
        template = TEMPLATES[(index - 1) % len(TEMPLATES)]
        records.append(build_record(index=index, template=template, rng=rng))

    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )
    print({"output": str(OUTPUT_PATH), "records": len(records)})


if __name__ == "__main__":
    main()