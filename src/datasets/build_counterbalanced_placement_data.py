#!/usr/bin/env python3
"""
Build counterbalanced answer-placement dataset from phase0_conflicting_answer_gsm8k_v2.jsonl.

For each source example, creates 3 variants:
  - prefix:  wrong/correct answer sentence placed FIRST, before computation steps
  - middle:  wrong/correct answer sentence placed in the MIDDLE of computation steps
  - suffix:  wrong/correct answer sentence placed LAST (standard condition)

Output: data/phase0_counterbalanced_placement_gsm8k_v1.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SOURCE = Path(__file__).parent.parent / "data" / "phase0_conflicting_answer_gsm8k_v2.jsonl"
OUTPUT = Path(__file__).parent.parent / "data" / "phase0_counterbalanced_placement_gsm8k_v1.jsonl"


def main() -> None:
    records: list[dict] = []
    skipped: list[tuple[str, str]] = []

    with SOURCE.open() as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    n_input = len(raw_lines)
    print(f"[info] Reading {n_input} examples from {SOURCE}")

    for raw in raw_lines:
        ex = json.loads(raw)
        eid = ex.get("id") or ex.get("example_id") or f"ex_{len(records)}"

        steps_std = ex.get("steps_standard", [])
        steps_conflict = ex.get("steps_conflicting", [])

        if not steps_std or not steps_conflict:
            skipped.append((eid, "missing steps_standard or steps_conflicting"))
            continue

        answer_step_correct = steps_std[-1]
        answer_step_wrong = steps_conflict[-1]

        # Verify the last step looks like a final answer line
        if "the answer is" not in answer_step_correct.lower():
            skipped.append((eid, f"last step does not contain 'the answer is': {answer_step_correct!r}"))
            continue

        computation_steps = steps_std[:-1]
        mid = max(1, len(computation_steps) // 2)

        base = {
            "original_id": eid,
            "question": ex.get("question", ""),
            "correct_answer": str(ex.get("correct_answer", "")),
            "wrong_answer": str(ex.get("wrong_answer", "")),
            "dataset": ex.get("dataset", "gsm8k"),
        }

        # --- PREFIX ---
        records.append({
            **base,
            "id": f"{eid}_prefix",
            "placement": "prefix",
            "steps_standard": [answer_step_correct] + computation_steps,
            "steps_conflicting": [answer_step_wrong] + computation_steps,
            "metadata": {"benchmark": "phase0-counterbalanced-placement-v1", "placement": "prefix"},
        })

        # --- MIDDLE ---
        records.append({
            **base,
            "id": f"{eid}_middle",
            "placement": "middle",
            "steps_standard": computation_steps[:mid] + [answer_step_correct] + computation_steps[mid:],
            "steps_conflicting": computation_steps[:mid] + [answer_step_wrong] + computation_steps[mid:],
            "metadata": {"benchmark": "phase0-counterbalanced-placement-v1", "placement": "middle"},
        })

        # --- SUFFIX ---
        records.append({
            **base,
            "id": f"{eid}_suffix",
            "placement": "suffix",
            "steps_standard": computation_steps + [answer_step_correct],
            "steps_conflicting": computation_steps + [answer_step_wrong],
            "metadata": {"benchmark": "phase0-counterbalanced-placement-v1", "placement": "suffix"},
        })

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Summary
    by_placement: dict[str, int] = {}
    for rec in records:
        p = rec["placement"]
        by_placement[p] = by_placement.get(p, 0) + 1

    print(f"\n=== DATA GENERATION SUMMARY ===")
    print(f"Total examples input:   {n_input}")
    print(f"Total records output:   {len(records)}")
    for placement, count in sorted(by_placement.items()):
        print(f"  {placement:8s}: {count}")
    if skipped:
        print(f"\nSkipped ({len(skipped)} examples):")
        for eid, reason in skipped:
            print(f"  {eid}: {reason}")
    else:
        print(f"\nNo examples skipped.")
    print(f"\nOutput written to: {OUTPUT}")


if __name__ == "__main__":
    main()
