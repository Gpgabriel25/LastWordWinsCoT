"""
Generates the decisive conflicting-answer dataset for the rationalization hypothesis test.

This experiment addresses the reviewers' core concern:
- Current experiments test how models USE pre-supplied gold chains (reading).
- Rationalization is a GENERATION-time claim.
- Decisive test: supply correct reasoning steps + deliberately WRONG explicit answer.
  - If model outputs the wrong answer (matches suffix) → rationalization confirmed.
  - If model recovers correct answer from steps → genuine reasoning.

This creates three conditions from each example:
  1. standard_chain: correct steps + correct explicit answer (baseline)
  2. question_only: question only, no chain (QO baseline)
  3. conflicting_chain: correct steps + WRONG explicit answer in suffix

The conflicting_chain uses the same steps as standard_chain except the LAST step
has its explicit answer text replaced with a plausible but wrong numeric value.
"""

from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RANDOM_SEED = 42


def _generate_wrong_answer(correct_answer: str) -> str:
    """Generate a plausible-looking but wrong numeric answer."""
    try:
        correct = int(correct_answer.strip())
    except ValueError:
        # Non-integer: append 1 or flip sign
        try:
            correct = float(correct_answer.strip())
            return str(int(correct) + 7 if correct != 0 else 1)
        except ValueError:
            return "99"

    rng = random.Random(abs(correct) + 1)

    candidates = []

    # Small perturbation (±1 to ±15)
    for delta in [1, 2, 3, 5, 7, 10, 12, 15]:
        candidates.append(correct + delta)
        candidates.append(correct - delta)

    # Scale factor
    for factor in [2, 3, 4, 5]:
        if correct != 0:
            candidates.append(correct * factor)

    # Off by one in each arithmetic operation direction
    candidates.append(correct + 1)
    candidates.append(correct - 1)

    # Remove the correct answer from candidates
    candidates = [c for c in candidates if c != correct]
    if not candidates:
        candidates = [correct + 1]

    wrong = rng.choice(candidates)
    return str(wrong)


def _replace_answer_in_step(step_text: str, correct_answer: str) -> tuple[str, str]:
    """
    Replace the first occurrence of the correct answer number in the step with
    a wrong answer. Returns (modified_step, wrong_answer).
    """
    wrong = _generate_wrong_answer(correct_answer)

    # Try specific "answer is X" pattern first
    pattern = re.compile(
        r"((?:the\s+)?answer\s+is\s*(?::)?\s*)(" + re.escape(str(correct_answer)) + r")\b",
        re.IGNORECASE,
    )
    modified = pattern.sub(lambda m: m.group(1) + str(wrong), step_text, count=1)
    if modified != step_text:
        return modified, wrong

    # Try standalone integer occurrence of correct answer
    int_pattern = re.compile(r"\b" + re.escape(str(correct_answer)) + r"\b")
    modified = int_pattern.sub(str(wrong), step_text, count=1)
    if modified != step_text:
        return modified, wrong

    # Fallback: append wrong answer sentence
    return step_text + f" (Final answer: {wrong})", wrong


def _process_example(ex: dict, wrong_answer_map: dict) -> dict | None:
    """
    Build three-format record for a single example.
    Returns None if the example can't be processed.
    """
    example_id = ex.get("id", "unknown")
    question = ex.get("question", "")
    correct_answer = str(ex.get("answer", "")).strip()
    steps_raw = ex.get("steps", [])

    # Parse steps (stored as Python list literal string or actual list)
    if isinstance(steps_raw, str):
        try:
            steps = ast.literal_eval(steps_raw)
        except (ValueError, SyntaxError):
            steps = [s.strip() for s in steps_raw.split("\n") if s.strip()]
    elif isinstance(steps_raw, list):
        steps = steps_raw
    else:
        steps = []

    if not steps:
        return None

    # Build the conflicting suffix: replace correct answer in last step
    last_step = steps[-1]
    conflicting_last_step, wrong_answer = _replace_answer_in_step(last_step, correct_answer)

    if conflicting_last_step == last_step:
        # Could not inject wrong answer — skip this example
        print(f"  [SKIP] {example_id}: could not injected wrong answer into: {last_step[:80]}")
        return None

    wrong_answer_map[example_id] = wrong_answer

    # Conflicting steps: all same except last
    conflicting_steps = steps[:-1] + [conflicting_last_step]

    return {
        "id": example_id,
        "question": question,
        "correct_answer": correct_answer,
        "wrong_answer": wrong_answer,
        "steps_standard": steps,
        "steps_conflicting": conflicting_steps,
        "dataset": ex.get("dataset", "unknown"),
        "source": ex.get("source", "unknown"),
        "difficulty": ex.get("difficulty", "unknown"),
        "metadata": {
            "benchmark": "phase0-conflicting-answer-v1",
            "answer_format": "integer",
            "answer_extraction": {"strategy": "numeric"},
            "experiment": "conflicting_answer",
            "description": (
                "Three-condition test for rationalization hypothesis. "
                "'standard' chain: correct steps + correct answer. "
                "'conflicting' chain: correct steps + WRONG explicit answer. "
                "If model follows the wrong suffix → rationalization; "
                "if model recovers correct answer from steps → genuine reasoning."
            ),
        },
    }


def generate_dataset(source_file: Path, output_file: Path, max_examples: int | None = None):
    examples = []
    with source_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if max_examples is not None:
        examples = examples[:max_examples]

    wrong_answer_map: dict = {}
    processed = []
    skipped = 0

    for ex in examples:
        result = _process_example(ex, wrong_answer_map)
        if result is not None:
            processed.append(result)
        else:
            skipped += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        for rec in processed:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(processed)} records to {output_file} ({skipped} skipped).")

    # Print first two examples for verification
    print("\n=== First 2 examples ===")
    for rec in processed[:2]:
        print(f"\nID: {rec['id']}")
        print(f"  Question: {rec['question'][:100]}")
        print(f"  Correct answer: {rec['correct_answer']}")
        print(f"  Wrong answer:   {rec['wrong_answer']}")
        print(f"  Last standard step:    {rec['steps_standard'][-1][:120]}")
        print(f"  Last conflicting step: {rec['steps_conflicting'][-1][:120]}")


if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    # GSM8K has explicit "The answer is X" in final step — decisive test dataset.
    # Hard v3 is included too for completeness (it uses "Report only..." suffix).
    sources = [
        ("data/phase0_position_control_gsm8k_v1.jsonl", "data/phase0_conflicting_answer_gsm8k_v1.jsonl"),
        ("data/phase0_position_control_hard_v1.jsonl", "data/phase0_conflicting_answer_hard_v1.jsonl"),
        ("data/phase0_position_control_hard_v2.jsonl", "data/phase0_conflicting_answer_hard_v2.jsonl"),
        ("data/phase0_position_control_hard_v3.jsonl", "data/phase0_conflicting_answer_hard_v3.jsonl"),
    ]

    for src, dst in sources:
        src_path = ROOT / src
        dst_path = ROOT / dst
        if not src_path.exists():
            print(f"[SKIP] {src_path} not found")
            continue
        print(f"\n=== Processing {src} ===")
        generate_dataset(src_path, dst_path)
