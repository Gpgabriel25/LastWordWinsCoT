"""
Build a MATH benchmark position-control dataset in the Phase 0 JSONL schema.

Loads the Hendrycks MATH dataset, filters for problems with numeric answers,
parses solutions into reasoning steps, and appends a terminal "the answer is X"
statement (matching the GSM8K format that the position-control pipeline expects).

Also generates a neutral-stripped variant where the terminal answer statement is
replaced with a neutral filler.

Usage:
    python build_math_benchmark_v1.py [--limit 100] [--min-steps 4]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "phase0_position_control_math_v1.jsonl"
DEFAULT_OUTPUT_STRIPPED = ROOT / "data" / "phase0_position_control_math_stripped_neutral_v1.jsonl"

BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
NUMERIC_ONLY = re.compile(r"^-?\d+(?:[\.,]\d+)*$")


def _require_datasets():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'datasets' package is required. Install it with `python -m pip install datasets`."
        ) from exc
    return load_dataset


def _extract_boxed(solution: str) -> str | None:
    """Extract the content from the last \\boxed{} in the solution."""
    matches = list(BOXED_PATTERN.finditer(solution))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _is_numeric_answer(answer: str) -> bool:
    """Check if the answer is purely numeric (possibly with commas/decimals)."""
    cleaned = answer.replace(",", "").replace(" ", "")
    return bool(NUMERIC_ONLY.match(cleaned))


def _normalize_numeric(answer: str) -> str:
    """Normalize a numeric answer: remove commas, trailing zeros after decimal."""
    cleaned = answer.replace(",", "").strip()
    try:
        val = float(cleaned)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return cleaned


def _clean_latex(text: str) -> str:
    """Light LaTeX cleanup for step text (remove \\boxed{}, simplify common commands)."""
    # Remove \boxed{...} wrappers, keeping content
    text = BOXED_PATTERN.sub(r"\1", text)
    # Remove $ delimiters
    text = text.replace("$", "")
    # Simplify common LaTeX
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("\\frac", "frac").replace("\\left", "").replace("\\right", "")
    text = text.replace("\\text{", "").replace("\\mathrm{", "")
    # Clean stray braces
    text = re.sub(r"\{(\d+)\}", r"\1", text)
    return text.strip()


def _split_solution_steps(solution: str, final_answer: str) -> list[str]:
    """Split a MATH solution into reasoning steps and append answer statement."""
    # Remove the boxed answer from the solution text
    clean_sol = BOXED_PATTERN.sub("", solution).strip()
    
    steps: list[str] = []
    for raw_line in clean_sol.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Clean up LaTeX for readability
        line = _clean_latex(line)
        if not line or line in (".", ""):
            continue
        # Split on sentence boundaries
        parts = [p.strip() for p in SENTENCE_SPLIT.split(line) if p.strip()]
        steps.extend(parts or [line])
    
    if not steps:
        return []
    
    # Append terminal answer statement (matching GSM8K format)
    steps.append(f"Therefore, the answer is {final_answer}.")
    return steps


def _strip_answer_from_step(step: str) -> str:
    """Replace the explicit answer in the final step with a neutral completion marker."""
    answer_re = re.compile(r"(?:the answer is|= )[\s]*[-]?[0-9][0-9,]*\.?\s*$", re.IGNORECASE)
    cleaned = answer_re.sub("", step).strip().rstrip(".,;")
    if cleaned:
        return cleaned + ". The calculation above gives the result."
    return "The steps above give the result."


def build_records(split: str, limit: int, min_steps: int) -> tuple[list[dict], list[dict]]:
    """Build both original and neutral-stripped records."""
    load_dataset = _require_datasets()
    
    # Load MATH dataset — merge all subject configs from EleutherAI/hendrycks_math
    configs = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]
    all_rows = []
    for cfg in configs:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", cfg, split=split)
            for row in ds:
                row["subject"] = cfg
                all_rows.append(row)
        except Exception as e:
            print(f"Warning: Failed to load config '{cfg}': {e}")
    
    if not all_rows:
        raise SystemExit("Could not load any MATH dataset configs.")
    
    # Shuffle deterministically for diversity across subjects
    import hashlib
    all_rows.sort(key=lambda r: hashlib.md5(r.get("problem", "").encode()).hexdigest())
    dataset = all_rows
    
    original_records: list[dict] = []
    stripped_records: list[dict] = []
    
    for row_index, row in enumerate(dataset):
        solution = row.get("solution", "")
        
        # Extract the boxed answer
        boxed = _extract_boxed(solution)
        if boxed is None:
            continue
        
        # Only keep numeric answers for clean extraction
        if not _is_numeric_answer(boxed):
            continue
        
        final_answer = _normalize_numeric(boxed)
        question = row.get("problem", "").strip()
        level = row.get("level", "")
        subject = row.get("type", row.get("subject", ""))
        
        steps = _split_solution_steps(solution, final_answer)
        if len(steps) < min_steps:
            continue
        
        record_id = f"math-{split}-{row_index:05d}"
        
        # Original record (with "the answer is X")
        orig = {
            "id": record_id,
            "question": question,
            "steps": steps,
            "answer": final_answer,
            "dataset": "math",
            "split": split,
            "source": "hendrycks/competition_math",
            "metadata": {
                "benchmark": "math",
                "subset": subject,
                "split": split,
                "source_index": row_index,
                "level": level,
                "answer_format": "numeric",
                "answer_extraction": {
                    "strategy": "numeric",
                    "expected_normalized": final_answer,
                },
                "chain_length": len(steps),
            },
        }
        original_records.append(orig)
        
        # Neutral-stripped record (answer replaced with neutral filler)
        stripped_steps = list(steps)
        stripped_steps[-1] = _strip_answer_from_step(stripped_steps[-1])
        
        stripped = dict(orig)
        stripped["id"] = record_id.replace("math-", "math-neutral-stripped-")
        stripped["steps"] = stripped_steps
        stripped_meta = dict(orig["metadata"])
        stripped_meta["variant"] = "stripped_suffix_neutral"
        stripped_meta["original_id"] = record_id
        stripped_meta["format_note"] = (
            "suffix step has explicit answer removed and replaced with neutral completion marker."
        )
        stripped["metadata"] = stripped_meta
        stripped_records.append(stripped)
        
        if len(original_records) >= limit:
            break
    
    if len(original_records) < limit:
        print(f"Warning: Only found {len(original_records)} qualifying examples "
              f"(requested {limit}). Using all available.")
    
    return original_records, stripped_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MATH benchmark position-control datasets in Phase 0 JSONL schema."
    )
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--limit", type=int, default=100, help="Max examples to include")
    parser.add_argument("--min-steps", type=int, default=4, help="Minimum chain steps")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-stripped", type=Path, default=DEFAULT_OUTPUT_STRIPPED)
    args = parser.parse_args()
    
    original, stripped = build_records(
        split=args.split, limit=args.limit, min_steps=args.min_steps
    )
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(r, ensure_ascii=True) for r in original) + "\n",
        encoding="utf-8",
    )
    args.output_stripped.write_text(
        "\n".join(json.dumps(r, ensure_ascii=True) for r in stripped) + "\n",
        encoding="utf-8",
    )
    print(f"Original: {len(original)} records → {args.output}")
    print(f"Stripped: {len(stripped)} records → {args.output_stripped}")
    
    # Show a sample
    if original:
        print(f"\nSample original steps (first record):")
        for i, s in enumerate(original[0]["steps"]):
            print(f"  [{i}]: {s[:120]}")
        print(f"  Answer: {original[0]['answer']}")
        print(f"\nSample stripped last step:")
        print(f"  {stripped[0]['steps'][-1][:120]}")


if __name__ == "__main__":
    main()
