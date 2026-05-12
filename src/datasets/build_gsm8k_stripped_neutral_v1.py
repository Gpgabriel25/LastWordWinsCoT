"""
Build a GSM8K benchmark study where the suffix step has its explicit answer removed,
using a NEUTRAL placeholder that does not induce re-verification or sign negation.

The original build_gsm8k_stripped_suffix_v1.py used "Let me verify this computation."
which caused sign-negation in Qwen 2.5-7B (base_acc=0.06) and Phi-3-mini.
This version uses a neutral completion marker that models treat as inert.

Key design goals:
1. Remove "the answer is X" from the suffix step (as before)
2. Replace with a phrase that doesn't instruct the model to re-verify
3. Allow the model to extract the answer from the preceding math steps
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "phase0_position_control_gsm8k_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "phase0_position_control_gsm8k_stripped_neutral_v1.jsonl"

ANSWER_AT_END = re.compile(r"(?:the answer is|= )[\s]*[-]?[0-9][0-9,]*\.?\s*$", re.IGNORECASE)


def strip_answer_from_step(step: str) -> str:
    """
    Replace the explicit answer in the final step with a neutral completion marker.
    E.g., "Therefore, the answer is 70000." -> "Therefore, the calculation above gives the result."
    Uses a neutral phrase that does NOT instruct the model to re-verify/re-compute.
    """
    cleaned = ANSWER_AT_END.sub("", step).strip().rstrip(".,;")
    if cleaned:
        return cleaned + ". The calculation above gives the result."
    return "The steps above give the result."


def process_record(record: dict) -> dict:
    steps = list(record["steps"])
    if steps:
        steps[-1] = strip_answer_from_step(steps[-1])
    new_record = dict(record)
    new_record["steps"] = steps
    new_record["id"] = record["id"].replace("gsm8k-", "gsm8k-neutral-stripped-")
    meta = dict(record.get("metadata", {}))
    meta["variant"] = "stripped_suffix_neutral"
    meta["original_id"] = record["id"]
    meta["format_note"] = (
        "suffix step has explicit answer removed and replaced with neutral completion marker. "
        "Replaces verify-inducing placeholder from v1 stripped format to avoid sign-negation artifact."
    )
    new_record["metadata"] = meta
    return new_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            raw = json.loads(line)
            records.append(process_record(raw))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(r, ensure_ascii=True) for r in records) + "\n",
        encoding="utf-8",
    )
    print({"output": str(args.output), "records": len(records)})


if __name__ == "__main__":
    main()
