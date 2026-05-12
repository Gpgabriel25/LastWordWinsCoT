from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "phase0_position_control_gsm8k_v1.jsonl"
ANSWER_PATTERN = re.compile(r"####\s*([-]?[0-9][0-9,]*)")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _require_datasets():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'datasets' package is required. Install it with `python -m pip install datasets`."
        ) from exc
    return load_dataset


def _normalize_answer(raw_answer: str) -> str | None:
    match = ANSWER_PATTERN.search(raw_answer)
    if not match:
        return None
    return match.group(1).replace(",", "")


def _split_steps(rationale: str, final_answer: str) -> list[str]:
    steps: list[str] = []
    for raw_line in rationale.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in SENTENCE_SPLIT.split(line) if part.strip()]
        steps.extend(parts or [line])
    if not steps:
        return []
    steps.append(f"Therefore, the answer is {final_answer}.")
    return steps


def build_records(split: str, limit: int, min_steps: int) -> list[dict[str, object]]:
    load_dataset = _require_datasets()
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    records: list[dict[str, object]] = []

    for row_index, row in enumerate(dataset):
        raw_answer = row["answer"]
        final_answer = _normalize_answer(raw_answer)
        if final_answer is None:
            continue
        rationale = raw_answer.split("####", 1)[0].strip()
        steps = _split_steps(rationale, final_answer)
        if len(steps) < min_steps:
            continue
        question = row["question"].strip()
        records.append(
            {
                "id": f"gsm8k-{split}-{row_index:05d}",
                "question": question,
                "steps": steps,
                "answer": final_answer,
                "dataset": "gsm8k",
                "split": split,
                "source": "openai/gsm8k",
                "metadata": {
                    "benchmark": "gsm8k",
                    "subset": "main",
                    "split": split,
                    "source_index": row_index,
                    "answer_format": "integer",
                    "answer_extraction": {
                        "strategy": "numeric",
                        "expected_normalized": final_answer,
                    },
                    "chain_length": len(steps),
                },
            }
        )
        if len(records) >= limit:
            break

    if len(records) < limit:
        raise SystemExit(
            f"Only found {len(records)} qualifying examples for split={split}, limit={limit}, min_steps={min_steps}."
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed GSM8K benchmark slice in the Phase 0 JSONL schema.")
    parser.add_argument("--split", default="test", help="Dataset split to use. Defaults to test.")
    parser.add_argument("--limit", type=int, default=100, help="Number of qualifying examples to write.")
    parser.add_argument("--min-steps", type=int, default=4, help="Minimum parsed chain length to keep an example.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    args = parser.parse_args()

    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.min_steps < 1:
        raise SystemExit("--min-steps must be at least 1")

    records = build_records(split=args.split, limit=args.limit, min_steps=args.min_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )
    print({"output": str(args.output), "records": len(records), "split": args.split, "min_steps": args.min_steps})


if __name__ == "__main__":
    main()
