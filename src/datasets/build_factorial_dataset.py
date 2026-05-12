#!/usr/bin/env python3
"""
Build the 2x2 Factorial Dataset: Reasoning Correctness × Answer-Line Correctness.

This separates the causal contributions of:
  (1) Reasoning content (correct vs. wrong intermediate steps)
  (2) Answer text (correct vs. wrong explicit answer)

DESIGN
------
4 conditions:
  A: Correct reasoning + Correct answer  [standard chain]
  B: Correct reasoning + Wrong answer    [conflicting chain -- already have]
  C: Wrong reasoning + Correct answer    [NEW -- shows answer text can rescue bad reasoning]
  D: Wrong reasoning + Wrong answer      [NEW -- double-wrong control]

KEY PREDICTION (if answer-text dominates):
  - Model should follow answer text regardless of reasoning quality
  - Accuracy: A ≈ C > D ≈ B (answer text determines outcome, not reasoning)

ALTERNATIVE (if reasoning dominates):
  - Accuracy: A ≈ B > C ≈ D (reasoning content determines outcome)

DATA SOURCES
------------
  phase1_gsm8k_1000.jsonl    -- gold steps + correct answer + question
  phase0_conflicting_answer_gsm8k_v2.jsonl -- aligned wrong_answer for each question

CORRUPTION METHOD
-----------------
Subtle semantic corruption matching scripts/run_tpu_easydel.py:
  - Number perturbation: replace numeric values (±20%)
  - Operator swap: replace +/- in multi-step arithmetic
  - Wrong variable: substitute wrong intermediate value

Output: phase0_factorial_2x2_gsm8k.jsonl
  Each record: {id, question, correct_answer, wrong_answer,
                condition, chain, steps_corrrupted: bool, answer_line_wrong: bool}
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path


random.seed(42)

# ---------------------------------------------------------------------------
# Subtle semantic corruption helpers (matches run_tpu_easydel.py approach)
# ---------------------------------------------------------------------------

def _perturb_number(match: re.Match) -> str:
    """Perturb a numeric value by ±20-50%."""
    original = match.group(0)
    try:
        val = float(original.replace(",", ""))
        # Perturb by 25-50%
        factor = random.choice([0.5, 0.6, 0.75, 1.33, 1.5, 2.0])
        perturbed = val * factor
        if perturbed == int(perturbed):
            return str(int(perturbed))
        return f"{perturbed:.1f}"
    except ValueError:
        return original


def _perturb_operator(text: str) -> str:
    """Swap arithmetic operators in the text."""
    swaps = [("+", "-"), ("-", "+"), ("×", "÷"), ("*", "/")]
    for a, b in swaps:
        if a in text and b not in text:
            return text.replace(a, b, 1)
        if b in text and a not in text:
            return text.replace(b, a, 1)
    return text


def semantic_corrupt_step(step: str) -> str:
    """Apply subtle semantic corruption to a single reasoning step.

    Strategy:
    1. Try to perturb a number in the step.
    2. If no number, try to swap an operator.
    3. If neither, append a wrong logical qualifier.
    """
    # Try number perturbation
    number_pattern = re.compile(r"\b\d+(?:\.\d+)?\b")
    numbers = list(number_pattern.finditer(step))
    if numbers:
        # Choose a random number to perturb
        choice = random.choice(numbers)
        return step[:choice.start()] + _perturb_number(choice) + step[choice.end():]

    # Try operator swap
    op_perturbed = _perturb_operator(step)
    if op_perturbed != step:
        return op_perturbed

    # Fallback: negate the statement
    qualifiers = ["incorrectly", "approximately", "mistakenly", ""]
    q = random.choice([q for q in qualifiers if q])
    return f"However, {q} " + step.lower()


def corrupt_steps(steps: list[str]) -> list[str]:
    """Corrupt approximately 2/3 of steps with semantic errors."""
    corrupted = []
    for i, step in enumerate(steps):
        # Corrupt all but the first step (which sets up the problem correctly)
        # and add errors progressively to maintain some plausibility
        if i > 0 and i < len(steps) - 1:
            corrupted.append(semantic_corrupt_step(step))
        elif i == len(steps) - 1 and len(steps) > 1:
            corrupted.append(semantic_corrupt_step(step))
        else:
            corrupted.append(step)
    return corrupted


# ---------------------------------------------------------------------------
# Chain formatting
# ---------------------------------------------------------------------------
def format_chain(steps: list[str], answer: str, include_answer_line: bool = True,
                 answer_line_content: str | None = None) -> str:
    """Format steps + optional answer line into a chain string."""
    lines = []
    for i, step in enumerate(steps, 1):
        lines.append(f"Step {i}: {step}")
    if include_answer_line:
        ans = answer_line_content if answer_line_content is not None else answer
        lines.append(f"The answer is {ans}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_gold_data(path: str) -> dict[str, dict]:
    """Load phase1 GSM8K data keyed by question text."""
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            question = str(rec.get("question") or "")
            if question:
                # Clean GSM8K-style computation annotations
                steps = rec.get("steps") or []
                clean_steps = [re.sub(r"<<[^>]*>>", "", s).strip() for s in steps]
                answer = re.sub(r"####\s*", "", str(rec.get("answer") or "")).strip()
                answer = re.sub(r"<<[^>]*>>", "", answer).strip()
                data[question] = {
                    "id": rec.get("id", ""),
                    "question": question,
                    "steps": clean_steps,
                    "answer": answer,
                }
    return data


def load_conflicting_data(path: str) -> dict[str, dict]:
    """Load conflicting-answer dataset keyed by question text."""
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            question = str(rec.get("question") or rec.get("prompt") or "")
            if question:
                data[question] = rec
    return data


# ---------------------------------------------------------------------------
# Build the 4 conditions
# ---------------------------------------------------------------------------
def build_factorial_dataset(
    gold_path: str,
    conflicting_path: str,
    output_path: str,
    n_examples: int = 300,
):
    gold_data = load_gold_data(gold_path)
    conflicting_data = load_conflicting_data(conflicting_path)

    # Find examples present in BOTH datasets
    common_questions = set(gold_data.keys()) & set(conflicting_data.keys())
    print(f"Gold examples: {len(gold_data)}", flush=True)
    print(f"Conflicting examples: {len(conflicting_data)}", flush=True)
    print(f"Common examples (by question text): {len(common_questions)}", flush=True)

    # Sort for reproducibility, then sample
    sorted_questions = sorted(common_questions)
    sample = sorted_questions[:n_examples]
    print(f"Using {len(sample)} examples for factorial design", flush=True)

    records = []
    for question in sample:
        gold = gold_data[question]
        conflicting = conflicting_data[question]

        correct_answer = gold["answer"]
        wrong_answer = str(conflicting.get("wrong_answer") or "")
        original_steps = gold["steps"]

        if not correct_answer or not wrong_answer or not original_steps:
            continue

        # Corrupt the steps for conditions C and D
        corrupted_steps = corrupt_steps(list(original_steps))

        # Condition A: standard chain (correct reasoning + correct answer)
        chain_A = format_chain(original_steps, correct_answer)

        # Condition B: conflicting chain (correct reasoning + wrong answer)
        chain_B = format_chain(original_steps, correct_answer, answer_line_content=wrong_answer)

        # Condition C: corrupted reasoning + correct answer
        chain_C = format_chain(corrupted_steps, correct_answer)

        # Condition D: corrupted reasoning + wrong answer
        chain_D = format_chain(corrupted_steps, correct_answer, answer_line_content=wrong_answer)

        base = {
            "id": gold["id"],
            "question": question,
            "correct_answer": correct_answer,
            "wrong_answer": wrong_answer,
            "n_steps": len(original_steps),
        }

        for cond, chain, steps_corrupted, answer_wrong in [
            ("A_correct_reasoning_correct_answer", chain_A, False, False),
            ("B_correct_reasoning_wrong_answer", chain_B, False, True),
            ("C_wrong_reasoning_correct_answer", chain_C, True, False),
            ("D_wrong_reasoning_wrong_answer", chain_D, True, True),
        ]:
            records.append({
                **base,
                "condition": cond,
                "chain": chain,
                "steps": original_steps if not steps_corrupted else corrupted_steps,
                "steps_corrupted": steps_corrupted,
                "answer_line_wrong": answer_wrong,
            })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    n_unique = len(records) // 4
    print(f"\nDataset written to {output_path}", flush=True)
    print(f"Total records: {len(records)} ({n_unique} examples × 4 conditions)", flush=True)
    print(f"Conditions: A (SC), B (CC), C (WR+CA), D (WR+WA)", flush=True)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-data", default="data/phase1_gsm8k_1000.jsonl")
    parser.add_argument("--conflicting-data", default="data/phase0_conflicting_answer_gsm8k_v2.jsonl")
    parser.add_argument("--output", default="data/phase0_factorial_2x2_gsm8k.jsonl")
    parser.add_argument("--n-examples", type=int, default=300)
    args = parser.parse_args()

    build_factorial_dataset(
        args.gold_data,
        args.conflicting_data,
        args.output,
        n_examples=args.n_examples,
    )
