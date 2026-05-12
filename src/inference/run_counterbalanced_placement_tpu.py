#!/usr/bin/env python3
"""
Counterbalanced answer-placement experiment.

Tests whether answer-text prioritization depends on POSITION (suffix bias)
or is consistently triggered by the CONTENT of the answer sentence regardless
of where it appears in the chain.

Three placement conditions per example:
  - prefix:  answer sentence placed FIRST in the chain (before computation)
  - middle:  answer sentence placed in the MIDDLE of the chain
  - suffix:  answer sentence placed LAST (standard condition)

Each placement has 3 inference conditions:
  1. standard_chain   : computation steps + CORRECT answer at given position
  2. conflicting_chain: computation steps + WRONG answer at given position
  3. question_only    : question text only (no steps)

Key metric: followed_wrong_suffix_rate per placement condition.
  If prefix and middle follow-wrong rates are also high (>0.4):
    → answer-text tracking regardless of position (CONTENT-driven prioritization)
  If only suffix shows high follow-wrong rate:
    → recency/positional bias rather than content-driven prioritization

Dataset schema (phase0_counterbalanced_placement_gsm8k_v1.jsonl):
{
  "id": "gsm8k-test-XXXXX_prefix",
  "original_id": "gsm8k-test-XXXXX",
  "question": "...",
  "correct_answer": "70000",
  "wrong_answer": "70005",
  "placement": "prefix",
  "steps_standard": ["Therefore, the answer is 70000.", ...computation...],
  "steps_conflicting": ["Therefore, the answer is 70005.", ...computation...],
  ...
}

Usage (run on all workers of v5e-64 pod):
  python3 run_counterbalanced_placement_tpu.py \\
    --model-id Qwen/Qwen2.5-3B-Instruct \\
    --data-file ~/data/phase0_counterbalanced_placement_gsm8k_v1.jsonl \\
    --output ~/results_fixed/counterbalanced_placement_qwen3b.json \\
    --study-name counterbalanced_placement_qwen3b \\
    --progress-every 20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from ast import Add, BinOp, Constant, Div, Expression, Mult, Sub, UnaryOp, USub
from ast import parse as ast_parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Example:
    example_id: str
    original_id: str
    prompt: str
    correct_answer: str
    wrong_answer: str
    placement: str
    steps_standard: list[str]
    steps_conflicting: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Prediction:
    predicted_answer: str
    raw_output: str
    is_correct: bool      # matches correct_answer
    follows_wrong: bool   # matches wrong_answer (and not correct)


# ---------------------------------------------------------------------------
# Answer extraction helpers (copied from run_conflicting_answer_tpu.py)
# ---------------------------------------------------------------------------
ANSWER_PATTERNS = [
    re.compile(r"(?:answer|result|total)[\s\S]*?[:\s]+\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:days?|hours?|minutes?|dollars?|cents?|apples?|oranges?|people|items?|units?)?\s*\.?\s*$", re.IGNORECASE),
]


def _maybe_eval_expression(text: str) -> str | None:
    try:
        node = ast_parse(text, mode="eval")
    except SyntaxError:
        return None
    allowed = (Add, BinOp, Constant, Div, Expression, Mult, Sub, UnaryOp, USub)
    for n in __import__("ast").walk(node):
        if not isinstance(n, allowed):
            return None
    try:
        result = eval(compile(node, "<string>", "eval"))  # noqa: S307
        if isinstance(result, (int, float)):
            if isinstance(result, float) and result.is_integer():
                return str(int(result))
            return str(result)
    except Exception:
        return None
    return None


def _extract_leading_answer(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"-?\d+(?:\.\d+)?(?:\s+[A-Za-z][A-Za-z\-]*)*", stripped):
            return stripped
        if _maybe_eval_expression(stripped) is not None:
            return stripped
        return None
    return None


def extract_answer(text: str) -> str | None:
    leading = _extract_leading_answer(text)
    if leading is not None:
        return leading
    for pattern in ANSWER_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(1).strip()
    return None


def coerce_answer(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    evaluated = _maybe_eval_expression(stripped)
    if evaluated is not None:
        return evaluated
    number_matches = re.findall(r"-?\d+(?:\.\d+)?", stripped)
    return number_matches[-1] if number_matches else stripped


def normalize_answer(text: str) -> str:
    normalized = text.strip().lower().rstrip(".!,;:").replace(",", "")
    return " ".join(normalized.split())


def answers_match(pred: str, target: str) -> bool:
    """Check if two answer strings match (numeric-aware)."""
    pred_n = normalize_answer(pred)
    tgt_n = normalize_answer(target)
    if pred_n == tgt_n:
        return True
    try:
        return abs(float(pred_n.replace(",", "")) - float(tgt_n.replace(",", ""))) < 1e-6
    except (ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_examples(path: str) -> list[Example]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            eid = str(record.get("id") or record.get("example_id") or f"ex_{len(examples)}")
            original_id = str(record.get("original_id") or eid)
            question = record.get("question") or record.get("prompt") or ""
            correct_answer = str(record.get("correct_answer") or record.get("answer") or "")
            wrong_answer = str(record.get("wrong_answer") or "")
            placement = str(record.get("placement") or "suffix")
            steps_std = record.get("steps_standard", [])
            steps_conflict = record.get("steps_conflicting", [])
            examples.append(Example(
                example_id=eid,
                original_id=original_id,
                prompt=question,
                correct_answer=correct_answer,
                wrong_answer=wrong_answer,
                placement=placement,
                steps_standard=steps_std,
                steps_conflicting=steps_conflict,
                metadata=record.get("metadata", {}),
            ))
    return examples


# ---------------------------------------------------------------------------
# Prompt building (same format as existing experiments)
# ---------------------------------------------------------------------------
def build_chain_prompt(example: Example, steps: list[str]) -> str:
    if not steps:
        return (
            "Solve the problem. "
            "Return exactly one integer answer with no words, no explanation, "
            "and no punctuation other than a leading minus sign if needed.\n"
            f"Question: {example.prompt}\n"
            "Final answer:"
        )
    reasoning = "\n".join(f"Step {i + 1}: {step}" for i, step in enumerate(steps))
    return (
        "Solve the problem using the provided reasoning. "
        "Return exactly one integer answer with no words, no explanation, "
        "and no punctuation other than a leading minus sign if needed.\n"
        f"Question: {example.prompt}\n"
        f"Reasoning:\n{reasoning}\n"
        "Final answer:"
    )


# ---------------------------------------------------------------------------
# EasyDeL model wrapper — copied exactly from run_conflicting_answer_tpu.py
# ---------------------------------------------------------------------------
class EasyDeLReasoner:
    FIXED_BUF_LEN = 640
    MAX_NEW_TOKENS = 24

    def __init__(self, model, tokenizer):
        import jax

        self.model = model
        self.tokenizer = tokenizer
        self._mesh = model.config.mesh

        try:
            import easydel.infra.modeling_outputs as _mo

            original_is_array = _mo._is_array

            def _patched_is_array(array):
                if isinstance(array, jax.Array):
                    return True
                return original_is_array(array)

            _mo._is_array = _patched_is_array
            print("[init] Patched EasyDeL _is_array for multi-host compatibility", flush=True)
        except Exception as exc:
            print(f"[init] WARNING: could not patch _is_array: {exc}", flush=True)

    def predict(self, example: "Example", steps: list[str]) -> "Prediction":
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P

        prompt_text = build_chain_prompt(example, steps)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="np",
            truncation=True,
            max_length=512,
            padding=False,
        )

        raw_ids = jnp.array(encoded["input_ids"])    # [1, seq_len]
        orig_len = int(raw_ids.shape[1])
        pad_id = self.tokenizer.pad_token_id or 0
        eos_id = self.tokenizer.eos_token_id
        stop_ids = {eos_id}
        if pad_id and pad_id != 0:
            stop_ids.add(pad_id)

        buf_len = self.FIXED_BUF_LEN
        if orig_len > buf_len:
            raw_ids = raw_ids[:, :buf_len]
            orig_len = buf_len

        ids_buf = jnp.full((1, buf_len), pad_id, dtype=jnp.int32)
        mask_buf = jnp.zeros((1, buf_len), dtype=jnp.int32)
        ids_buf = ids_buf.at[0, :orig_len].set(raw_ids[0, :orig_len])
        mask_buf = mask_buf.at[0, :orig_len].set(1)

        replicated = NamedSharding(self._mesh, P())
        ids_buf = jax.device_put(ids_buf, replicated)
        mask_buf = jax.device_put(mask_buf, replicated)

        # Greedy decoding via Python loop — exactly as in run_conflicting_answer_tpu.py
        gen_tokens: list[int] = []
        cur_pos = orig_len
        for _ in range(self.MAX_NEW_TOKENS):
            if cur_pos >= buf_len:
                break
            with self._mesh:
                outputs = self.model(ids_buf, attention_mask=mask_buf)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = int(jnp.argmax(logits[0, cur_pos - 1, :]))
            if next_token_id in stop_ids:
                break
            gen_tokens.append(next_token_id)
            ids_buf = ids_buf.at[0, cur_pos].set(next_token_id)
            mask_buf = mask_buf.at[0, cur_pos].set(1)
            ids_buf = jax.device_put(ids_buf, replicated)
            mask_buf = jax.device_put(mask_buf, replicated)
            cur_pos += 1

        completion = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        extracted = extract_answer(completion)
        predicted_answer = coerce_answer(extracted) if extracted else completion.strip()
        is_correct = answers_match(predicted_answer, example.correct_answer)
        follows_wrong = (
            answers_match(predicted_answer, example.wrong_answer)
            and not answers_match(example.wrong_answer, example.correct_answer)
        )

        return Prediction(
            predicted_answer=predicted_answer,
            raw_output=completion,
            is_correct=is_correct,
            follows_wrong=follows_wrong,
        )


# ---------------------------------------------------------------------------
# EasyDeL model loading — copied exactly from run_conflicting_answer_tpu.py
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_id: str, sharding_axis_dims: tuple[int, ...]):
    import jax
    import jax.numpy as jnp
    from jax import lax
    import easydel as ed
    from transformers import AutoTokenizer

    process_index = jax.process_index()
    device_count = jax.device_count()
    print(
        f"[init] process_index={process_index} device_count={device_count} "
        f"model_id={model_id} sharding={sharding_axis_dims}",
        flush=True,
    )

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        model_id,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=lax.Precision.DEFAULT,
        auto_shard_model=True,
        sharding_axis_dims=sharding_axis_dims,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
            attn_dtype=jnp.bfloat16,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
        ),
        partition_axis=ed.PartitionAxis(),
    )
    if hasattr(model, "config"):
        model.config.attn_mechanism = ed.AttentionMechanisms.VANILLA

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[init] Model loaded. process_index={process_index}", flush=True)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------
CONDITION_NAMES = ["standard_chain", "conflicting_chain", "question_only"]
PLACEMENT_NAMES = ["prefix", "middle", "suffix"]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON so watchers never read a partially-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def run_placement_experiment(
    examples: list[Example],
    reasoner: EasyDeLReasoner,
    study_name: str,
    model_name: str,
    progress_every: int = 20,
    checkpoint_every: int = 0,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Run counterbalanced placement experiment: 3 placements × 3 conditions.

    Optimization: question_only inference is deduplicated per original_id.
    Since all 3 placements of an example share the same question, the question_only
    forward pass is run once and cached — reducing total passes from 4500 to 3500
    for a 1500-record dataset (22% speedup without any accuracy impact).
    """

    # Stats bucketed by (placement, condition)
    stats: dict[str, dict[str, dict[str, int]]] = {
        p: {c: {"total": 0, "correct": 0, "follows_wrong": 0} for c in CONDITION_NAMES}
        for p in PLACEMENT_NAMES
    }

    # Cache question_only predictions by original_id to avoid redundant inference
    question_only_cache: dict[str, Prediction] = {}

    per_example_results: list[dict[str, Any]] = []
    t0 = time.time()

    # Only coordinator writes checkpoints on multi-host TPU runs.
    coordinator_writes = False
    if checkpoint_every and checkpoint_path:
        import jax

        coordinator_writes = (jax.process_index() == 0)

    for ei, example in enumerate(examples, start=1):
        placement = example.placement
        if placement not in PLACEMENT_NAMES:
            placement = "suffix"  # fallback

        condition_steps = {
            "standard_chain": example.steps_standard,
            "conflicting_chain": example.steps_conflicting,
            "question_only": [],
        }

        condition_results: dict[str, dict] = {}
        for condition in CONDITION_NAMES:
            # Reuse cached question_only prediction for the same original example
            if condition == "question_only" and example.original_id in question_only_cache:
                pred = question_only_cache[example.original_id]
            else:
                pred = reasoner.predict(example, condition_steps[condition])
                if condition == "question_only":
                    question_only_cache[example.original_id] = pred

            stats[placement][condition]["total"] += 1
            if pred.is_correct:
                stats[placement][condition]["correct"] += 1
            if pred.follows_wrong:
                stats[placement][condition]["follows_wrong"] += 1

            condition_results[condition] = {
                "predicted": pred.predicted_answer,
                "raw_output": pred.raw_output[:200],
                "is_correct": pred.is_correct,
                "follows_wrong": pred.follows_wrong,
            }

        per_example_results.append({
            "example_id": example.example_id,
            "original_id": example.original_id,
            "placement": placement,
            "correct_answer": example.correct_answer,
            "wrong_answer": example.wrong_answer,
            "conditions": condition_results,
        })

        if coordinator_writes and checkpoint_every and (ei % checkpoint_every == 0):
            _write_json_atomic(
                Path(checkpoint_path),
                {
                    "study_name": study_name,
                    "model": model_name,
                    "n_examples": len(examples),
                    "completed_examples": ei,
                    "stats": stats,
                    "results": per_example_results,
                    "checkpoint_time": int(time.time()),
                },
            )
            print(
                f"[checkpoint] wrote {ei}/{len(examples)} to {checkpoint_path}",
                flush=True,
            )

        if progress_every and (ei % progress_every == 0 or ei == len(examples)):
            elapsed = time.time() - t0
            # Print per-placement conflicting rate
            parts = []
            for p in PLACEMENT_NAMES:
                s = stats[p]["conflicting_chain"]
                n = s["total"]
                if n > 0:
                    fw = s["follows_wrong"] / n
                    parts.append(f"{p}_fw={fw:.3f}")
            print(
                f"[progress] {ei}/{len(examples)} | " + " | ".join(parts) +
                f" | elapsed={elapsed:.0f}s",
                flush=True,
            )

    # Build per-placement summary
    placements_summary: dict[str, Any] = {}
    print("\n=== FINAL RESULTS BY PLACEMENT ===")
    for p in PLACEMENT_NAMES:
        placements_summary[p] = {}
        print(f"\n--- {p.upper()} ---")
        for c in CONDITION_NAMES:
            s = stats[p][c]
            n = s["total"]
            if n == 0:
                placements_summary[p][c] = {"n": 0}
                continue
            acc = s["correct"] / n
            fw = s["follows_wrong"] / n
            placements_summary[p][c] = {
                "n": n,
                "accuracy": round(acc, 4),
                "followed_wrong_rate": round(fw, 4),
            }
            print(f"  {c}: n={n}  accuracy={acc:.4f}  follow_wrong={fw:.4f}")

    # Interpretation
    print("\n=== INTERPRETATION ===")
    fw_by_placement = {}
    for p in PLACEMENT_NAMES:
        s = stats[p]["conflicting_chain"]
        n = s["total"]
        fw = s["follows_wrong"] / n if n > 0 else 0.0
        fw_by_placement[p] = fw

    high_placements = [p for p, fw in fw_by_placement.items() if fw > 0.4]
    if len(high_placements) >= 2:
        print(
            f"CONTENT-DRIVEN PRIORITIZATION: high follow-wrong in {high_placements} "
            f"— answer-text tracking is position-independent"
        )
    elif fw_by_placement.get("suffix", 0) > 0.4 and all(
        fw_by_placement.get(p, 0) <= 0.3 for p in ["prefix", "middle"]
    ):
        print(
            "RECENCY/SUFFIX BIAS: only suffix shows high follow-wrong "
            f"(prefix={fw_by_placement['prefix']:.3f}, middle={fw_by_placement['middle']:.3f}, "
            f"suffix={fw_by_placement['suffix']:.3f})"
        )
    else:
        print(
            f"MIXED OR AMBIGUOUS: "
            + ", ".join(f"{p}={fw:.3f}" for p, fw in fw_by_placement.items())
        )

    return {
        "study_name": study_name,
        "model": model_name,
        "n_examples": len(examples),
        "placements": placements_summary,
        "results": per_example_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Counterbalanced answer-placement rationalization experiment"
    )
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID")
    parser.add_argument("--data-file", required=True, help="Path to counterbalanced placement JSONL")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--study-name", default="counterbalanced_placement", help="Study name")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Write periodic checkpoint JSON every N examples (0 disables)")
    parser.add_argument("--checkpoint-path", default=None,
                        help="Checkpoint JSON path (default: <output>.checkpoint.json)")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--sharding", default="1,-1,1,1,1",
                        help="Sharding axis dims: dp,fsdp,ep,tp,sp (default: 1,-1,1,1,1)")
    args = parser.parse_args()
    sharding_axis_dims = tuple(int(x) for x in args.sharding.split(","))

    # Enable transparent hugepages
    try:
        with open("/sys/kernel/mm/transparent_hugepage/enabled", "w") as f:
            f.write("always")
    except Exception:
        pass

    print(f"[startup] Loading examples from {args.data_file}", flush=True)
    examples = load_examples(args.data_file)
    if args.max_examples:
        examples = examples[: args.max_examples]
    print(f"[startup] Loaded {len(examples)} examples", flush=True)

    # Brief integrity check
    placements_seen = set(ex.placement for ex in examples[:20])
    print(f"[startup] Placements seen in first 20 examples: {placements_seen}", flush=True)

    print(f"[startup] Loading model {args.model_id} with sharding={sharding_axis_dims}", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model_id, sharding_axis_dims)
    reasoner = EasyDeLReasoner(model, tokenizer)
    print("[startup] Ready.", flush=True)

    output = run_placement_experiment(
        examples=examples,
        reasoner=reasoner,
        study_name=args.study_name,
        model_name=args.model_id,
        progress_every=args.progress_every,
        checkpoint_every=max(0, args.checkpoint_every),
        checkpoint_path=args.checkpoint_path or (args.output + ".checkpoint.json"),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(out_path, output)
    print(f"\n[done] Results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
