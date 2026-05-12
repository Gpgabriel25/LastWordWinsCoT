#!/usr/bin/env python3
"""
2×2 Factorial inference: Reasoning quality × Answer-line content.

Conditions in the dataset (field: condition):
  A_correct_reasoning_correct_answer   : gold steps + correct answer line
  B_correct_reasoning_wrong_answer     : gold steps + wrong answer line
  C_corrupted_reasoning_correct_answer : corrupted steps + correct answer line
  D_corrupted_reasoning_wrong_answer   : corrupted steps + wrong answer line

Metrics per condition:
  - accuracy: fraction producing the correct final answer
  - followed_wrong: fraction producing the wrong answer (for B and D)

Decision predictions:
  Answer-text dominance : A ≈ C >> B ≈ D  (answer line controls output)
  Reasoning dominance   : A ≈ B >> C ≈ D  (reasoning quality controls output)
  Mixed                 : A > C >= D > B   (both factors matter, answer-line stronger)

Usage (run on all workers of v5e-64 pod):
  python3 run_factorial_inference_tpu.py \\
      --model-id Qwen/Qwen2.5-3B-Instruct \\
      --data-file ~/data/phase0_factorial_2x2_gsm8k.jsonl \\
      --output ~/results_fixed/factorial_gsm8k_qwen3b.json \\
      --n-examples 300 \\
      --sharding 1,-1,1,1,1 \\
      --progress-every 20
"""
from __future__ import annotations

# CRITICAL: Must init distributed BEFORE any JAX import that touches backend
import jax
jax.distributed.initialize()

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
# Atomic JSON write helper
# ---------------------------------------------------------------------------
def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write payload to a .tmp file then rename to avoid corrupt partial writes."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Example:
    example_id: str
    question: str
    correct_answer: str
    wrong_answer: str
    condition: str
    chain: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Prediction:
    condition: str
    predicted_answer: str
    raw_output: str
    is_correct: bool
    follows_wrong: bool


# ---------------------------------------------------------------------------
# Answer extraction (copied from run_conflicting_answer_tpu.py)
# ---------------------------------------------------------------------------
ANSWER_PATTERNS = [
    re.compile(r"(?:answer|result|total)[\s\S]*?[:\s]+\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:days?|hours?|minutes?|dollars?|cents?|apples?|oranges?"
               r"|people|items?|units?)?\s*\.?\s*$", re.IGNORECASE),
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
def load_examples(path: str, n_examples: int) -> list[Example]:
    """Load factorial dataset, up to n_examples PER CONDITION."""
    per_condition: dict[str, list[Example]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            condition = record.get("condition", "")
            if condition not in per_condition:
                per_condition[condition] = []
            if len(per_condition[condition]) >= n_examples:
                continue
            example_id = str(record.get("id") or f"ex_{sum(len(v) for v in per_condition.values())}")
            per_condition[condition].append(Example(
                example_id=example_id,
                question=record.get("question", ""),
                correct_answer=str(record.get("correct_answer", "")),
                wrong_answer=str(record.get("wrong_answer", "")),
                condition=condition,
                chain=record.get("chain", ""),
                metadata={k: record.get(k) for k in ("n_steps",)},
            ))
    # Flatten, interleaving conditions for balanced progress reporting
    all_conditions = sorted(per_condition.keys())
    examples: list[Example] = []
    max_count = max((len(v) for v in per_condition.values()), default=0)
    for i in range(max_count):
        for cond in all_conditions:
            if i < len(per_condition[cond]):
                examples.append(per_condition[cond][i])
    return examples


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(example: Example) -> str:
    return (
        "Solve the problem using the provided reasoning. "
        "Return exactly one integer answer with no words, no explanation, "
        "and no punctuation other than a leading minus sign if needed.\n"
        f"Question: {example.question}\n"
        f"Reasoning:\n{example.chain}\n"
        "Final answer:"
    )


# ---------------------------------------------------------------------------
# Model loading + inference (same EasyDeL pattern as run_conflicting_answer_tpu.py)
# ---------------------------------------------------------------------------
FIXED_BUF_LEN = 640
MAX_NEW_TOKENS = 24


def load_model(model_id: str, sharding: tuple[int, ...]):
    import os
    import jax
    import jax.numpy as jnp
    from jax import lax
    import easydel as ed
    from transformers import AutoTokenizer

    local_only = os.environ.get("HF_LOCAL_ONLY", "1") == "1"

    print(f"[init] process_index={jax.process_index()} device_count={jax.device_count()} "
          f"model_id={model_id} sharding={sharding} local_only={local_only}", flush=True)

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=local_only,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=lax.Precision.DEFAULT,
        auto_shard_model=True,
        sharding_axis_dims=sharding,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
            attn_dtype=jnp.bfloat16,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
        ),
        partition_axis=ed.PartitionAxis(),
    )

    # Monkey-patch _is_array for multi-host compatibility
    try:
        import easydel.infra.modeling_outputs as _mo
        _orig = _mo._is_array

        def _patched(arr):
            if isinstance(arr, jax.Array):
                return True
            return _orig(arr)
        _mo._is_array = _patched
        print("[init] Patched EasyDeL _is_array for multi-host compatibility", flush=True)
    except Exception as e:
        print(f"[init] WARNING: could not patch _is_array: {e}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[init] Model loaded. process_index={jax.process_index()}", flush=True)
    return model, tokenizer


def run_one(model, tokenizer, prompt: str) -> str:
    """Greedy decode using Python loop — proven pattern from run_conflicting_answer_tpu.py."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    encoded = tokenizer(
        prompt,
        return_tensors="np",
        truncation=True,
        max_length=512,
        padding=False,
    )

    raw_ids = jnp.array(encoded["input_ids"])
    orig_len = int(raw_ids.shape[1])
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    # Qwen2.5 uses 151643 (<|endoftext|>) as its primary stop marker
    stop_ids: set[int] = set()
    if eos_id is not None:
        stop_ids.add(int(eos_id))
    if pad_id is not None:
        stop_ids.add(int(pad_id))
    stop_ids.add(151643)

    buf_len = FIXED_BUF_LEN
    if orig_len > buf_len:
        raw_ids = raw_ids[:, :buf_len]
        orig_len = buf_len

    ids_buf = jnp.full((1, buf_len), pad_id, dtype=jnp.int32)
    mask_buf = jnp.zeros((1, buf_len), dtype=jnp.int32)
    ids_buf = ids_buf.at[0, :orig_len].set(raw_ids[0, :orig_len])
    mask_buf = mask_buf.at[0, :orig_len].set(1)

    mesh = model.config.mesh
    replicated = NamedSharding(mesh, P())
    ids_buf = jax.device_put(ids_buf, replicated)
    mask_buf = jax.device_put(mask_buf, replicated)

    gen_tokens: list[int] = []
    cur_pos = orig_len

    for _ in range(MAX_NEW_TOKENS):
        if cur_pos >= buf_len:
            break
        with mesh:
            outputs = model(ids_buf, attention_mask=mask_buf)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = int(jnp.argmax(logits[0, cur_pos - 1, :]))
        if next_token_id in stop_ids:
            break
        gen_tokens.append(next_token_id)
        ids_buf = ids_buf.at[0, cur_pos].set(next_token_id)
        mask_buf = mask_buf.at[0, cur_pos].set(1)
        cur_pos += 1

    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate(model, tokenizer, examples: list[Example],
             progress_every: int) -> dict[str, list[dict]]:
    """Returns per-condition list of result dicts."""
    results: dict[str, list[dict]] = {}
    total = len(examples)

    for i, ex in enumerate(examples):
        prompt = build_prompt(ex)
        raw = run_one(model, tokenizer, prompt)

        extracted = extract_answer(raw)
        if extracted is None:
            extracted = coerce_answer(raw)

        is_correct = answers_match(extracted, ex.correct_answer)
        follows_wrong = answers_match(extracted, ex.wrong_answer) and not is_correct

        result = {
            "example_id": ex.example_id,
            "condition": ex.condition,
            "correct_answer": ex.correct_answer,
            "wrong_answer": ex.wrong_answer,
            "predicted_answer": extracted,
            "raw_output": raw[:200],
            "is_correct": is_correct,
            "follows_wrong": follows_wrong,
        }
        results.setdefault(ex.condition, []).append(result)

        if (i + 1) % progress_every == 0:
            # Compute running stats per condition
            stats_parts = []
            for cond, cond_results in sorted(results.items()):
                n = len(cond_results)
                acc = sum(1 for r in cond_results if r["is_correct"]) / n
                fw = sum(1 for r in cond_results if r["follows_wrong"]) / n
                stats_parts.append(f"{cond[0]}:acc={acc:.3f},fw={fw:.3f}")
            print(f"[progress] {i+1}/{total} | {' | '.join(stats_parts)}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(results: dict[str, list[dict]]) -> dict:
    summary = {}
    for cond, cond_results in sorted(results.items()):
        n = len(cond_results)
        n_correct = sum(1 for r in cond_results if r["is_correct"])
        n_wrong = sum(1 for r in cond_results if r["follows_wrong"])
        short = cond.split("_")[0]  # "A", "B", "C", "D"
        summary[short] = {
            "condition_full": cond,
            "n": n,
            "accuracy": round(n_correct / n, 4) if n else 0.0,
            "followed_wrong_rate": round(n_wrong / n, 4) if n else 0.0,
        }

    # Key contrasts
    a_acc = summary.get("A", {}).get("accuracy", 0.0)
    b_fw = summary.get("B", {}).get("followed_wrong_rate", 0.0)
    c_acc = summary.get("C", {}).get("accuracy", 0.0)
    d_fw = summary.get("D", {}).get("followed_wrong_rate", 0.0)

    # Answer-line effect: does answer content dominate reasoning?
    # Under dominance: A_acc ≈ C_acc, B_fw ≈ D_fw
    # Reasoning effect: A_acc >> C_acc
    answer_line_dominance = (abs(a_acc - c_acc) < 0.10) and (abs(b_fw - d_fw) < 0.10)

    return {
        "conditions": summary,
        "contrasts": {
            "A_acc": a_acc,
            "B_followed_wrong": b_fw,
            "C_acc": c_acc,
            "D_followed_wrong": d_fw,
            "reasoning_effect_on_accuracy": round(a_acc - c_acc, 4),
            "reasoning_effect_on_fw": round(b_fw - d_fw, 4),
            "answer_line_effect_on_accuracy": round(a_acc - b_fw, 4),
            "answer_line_rescues_corrupted": round(c_acc, 4),
        },
        "interpretation": {
            "answer_text_dominates": answer_line_dominance,
            "reasoning_reduces_acc_by": round(a_acc - c_acc, 4),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2×2 factorial inference (TPU)")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--data-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-examples", type=int, default=300,
                   help="Max examples per condition (default 300)")
    p.add_argument("--sharding", default="1,-1,1,1,1",
                   help="EasyDeL sharding axes (comma-separated)")
    p.add_argument("--progress-every", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sharding = tuple(int(x) for x in args.sharding.split(","))

    model, tokenizer = load_model(args.model_id, sharding)

    # Only process on coordinator
    examples = load_examples(args.data_file, args.n_examples)
    print(f"[startup] Ready. {len(examples)} total examples across "
          f"{len(set(e.condition for e in examples))} conditions.", flush=True)

    t0 = time.time()
    results = evaluate(model, tokenizer, examples, args.progress_every)
    elapsed = time.time() - t0

    summary = aggregate(results)
    print(f"\n[done] Elapsed={elapsed:.0f}s", flush=True)
    print(f"[results] A(correct/correct) acc={summary['conditions'].get('A',{}).get('accuracy','?')}", flush=True)
    print(f"[results] B(correct/wrong) fw={summary['conditions'].get('B',{}).get('followed_wrong_rate','?')}", flush=True)
    print(f"[results] C(corrupted/correct) acc={summary['conditions'].get('C',{}).get('accuracy','?')}", flush=True)
    print(f"[results] D(corrupted/wrong) fw={summary['conditions'].get('D',{}).get('followed_wrong_rate','?')}", flush=True)
    print(f"[results] reasoning_effect={summary['contrasts']['reasoning_effect_on_accuracy']:+.4f} "
          f"(A-C; positive = gold reasoning helps)", flush=True)
    print(f"[results] c_acc={summary['contrasts']['C_acc']:.4f} "
          f"(answer-line rescues corrupted reasoning)", flush=True)
    print("[DONE] Factorial evaluation finished.", flush=True)

    if jax.process_index() == 0:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "model_id": args.model_id,
            "data_file": args.data_file,
            "n_per_condition": args.n_examples,
            "elapsed_s": round(elapsed, 1),
            "summary": summary,
            "per_example": {k: v for k, v in results.items()},
        }
        _write_json_atomic(output_path, out)
        print(f"[save] Results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
