#!/usr/bin/env python3
"""
Decisive rationalization test: conflicting-answer experiment.

Three conditions per example:
  1. standard_chain   : correct steps + correct explicit final answer
  2. conflicting_chain: correct steps + WRONG explicit final answer in suffix
  3. question_only    : question text only (no steps)

Key metric (condition 2): what fraction of outputs match the WRONG suffix answer
vs. the correct computed answer?

  followed_wrong_suffix_rate > 0.5  → RATIONALIZATION CONFIRMED
  followed_wrong_suffix_rate < 0.2  → GENUINE REASONING
  Otherwise                         → Mixed / inconclusive

Dataset schema (phase0_conflicting_answer_gsm8k_v1.jsonl):
{
  "id": "gsm8k-test-XXXXX",
  "question": "...",
  "correct_answer": "70000",
  "wrong_answer": "70005",
  "steps_standard": [..., "Therefore, the answer is 70000."],
  "steps_conflicting": [..., "Therefore, the answer is 70005."],
  ...
}

Usage (run on all workers of v5e-64 pod):
  python3 run_conflicting_answer_tpu.py \\
    --model-id Qwen/Qwen2.5-3B-Instruct \\
    --data-file ~/data/phase0_conflicting_answer_gsm8k_v1.jsonl \\
    --output ~/results_fixed/conflicting_gsm8k_qwen3b.json \\
    --study-name conflicting_gsm8k_qwen3b \\
    --progress-every 10
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
    prompt: str
    correct_answer: str
    wrong_answer: str
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
# Answer extraction helpers (copied from run_pillar2_truncation_tpu.py)
# ---------------------------------------------------------------------------
ANSWER_PATTERNS = [
    re.compile(r"\\boxed\{(-?\d[\d,]*(?:\.\d+)?)\}"),  # LaTeX \boxed{N} (DeepSeek-R1 family)
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
    # Strip <think>...</think> blocks (for models like DeepSeek-R1-Distill that use CoT tags)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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
    # Try numeric float comparison
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
            eid = str(record.get("example_id") or record.get("id") or f"ex_{len(examples)}")
            question = record.get("question") or record.get("prompt") or ""
            correct_answer = str(record.get("correct_answer") or record.get("answer") or "")
            wrong_answer = str(record.get("wrong_answer") or "")
            steps_std = record.get("steps_standard", [])
            steps_conflict = record.get("steps_conflicting", [])
            examples.append(Example(
                example_id=eid,
                prompt=question,
                correct_answer=correct_answer,
                wrong_answer=wrong_answer,
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
            "Return the answer as a single number only, nothing else.\n"
            f"Question: {example.prompt}\n"
            "Answer:"
        )
    reasoning = "\n".join(f"Step {i + 1}: {step}" for i, step in enumerate(steps))
    return (
        "Solve the problem using the provided reasoning. "
        "Return the answer as a single number only, nothing else.\n"
        f"Question: {example.prompt}\n"
        f"Reasoning:\n{reasoning}\n"
        "Answer:"
    )


# EasyDeL model wrapper — matches run_pillar2_truncation_tpu.py exactly.
# Key: use a Python for loop + with self._mesh: per step, NOT jax.lax.while_loop.
# EasyDeL's model() is internally JIT-compiled; mesh context activates sharding.
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
        stop_ids: set[int] = set()
        if eos_id is not None:
            stop_ids.add(int(eos_id))
        if self.tokenizer.pad_token_id is not None:
            stop_ids.add(int(self.tokenizer.pad_token_id))
        # Qwen commonly emits <|endoftext|> as the actual stop token.
        stop_ids.add(151643)

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

        # Greedy decoding via Python loop — exactly as in run_pillar2_truncation_tpu.py
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
# EasyDeL model loading
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

    # Cap max_position_embeddings to avoid OOM when EasyDeL creates the causal mask.
    # Models with 128K context (e.g., DeepSeek-R1-Distill, some Qwen-2.5 variants)
    # trigger a (max_pos × max_pos) bool allocation = 16 GB on a single device.
    # Since FIXED_BUF_LEN=640 we never process sequences longer than 640 tokens;
    # capping to 1024 is safe and reduces the mask to ~1 MB.
    _CAUSAL_MASK_CAP = 1024
    if hasattr(model, "config"):
        orig_max_pos = getattr(model.config, "max_position_embeddings", None)
        if orig_max_pos is not None and orig_max_pos > _CAUSAL_MASK_CAP:
            model.config.max_position_embeddings = _CAUSAL_MASK_CAP
            # Clear any cached causal_mask (cached_property stores in instance __dict__)
            for _attr in ("causal_mask", "_causal_mask"):
                if _attr in model.__dict__:
                    del model.__dict__[_attr]
            print(
                f"[init] Capped max_position_embeddings {orig_max_pos}→{_CAUSAL_MASK_CAP} "
                f"to avoid causal-mask OOM",
                flush=True,
            )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[init] Model loaded. process_index={process_index}", flush=True)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------
CONDITION_NAMES = ["standard_chain", "conflicting_chain", "question_only"]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON so readers never observe partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def run_conflicting_experiment(
    examples: list[Example],
    reasoner: EasyDeLReasoner,
    study_name: str,
    model_name: str,
    progress_every: int = 10,
    checkpoint_every: int = 0,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Run the decisive 3-condition conflicting-answer experiment."""
    stats: dict[str, dict[str, int]] = {
        c: {"total": 0, "correct": 0, "follows_wrong": 0}
        for c in CONDITION_NAMES
    }
    per_example_results: list[dict[str, Any]] = []
    t0 = time.time()

    coordinator_writes = False
    if checkpoint_every and checkpoint_path:
        import jax

        coordinator_writes = (jax.process_index() == 0)

    for ei, example in enumerate(examples, start=1):
        condition_steps = {
            "standard_chain": example.steps_standard,
            "conflicting_chain": example.steps_conflicting,
            "question_only": [],
        }

        condition_results: dict[str, dict] = {}
        for condition in CONDITION_NAMES:
            pred = reasoner.predict(example, condition_steps[condition])
            stats[condition]["total"] += 1
            if pred.is_correct:
                stats[condition]["correct"] += 1
            if pred.follows_wrong:
                stats[condition]["follows_wrong"] += 1
            condition_results[condition] = {
                "predicted": pred.predicted_answer,
                "raw_output": pred.raw_output[:200],
                "is_correct": pred.is_correct,
                "follows_wrong_suffix": pred.follows_wrong,
            }

        per_example_results.append({
            "example_id": example.example_id,
            "correct_answer": example.correct_answer,
            "wrong_answer": example.wrong_answer,
            "last_step_standard": example.steps_standard[-1][:80] if example.steps_standard else "",
            "last_step_conflicting": example.steps_conflicting[-1][:80] if example.steps_conflicting else "",
            "conditions": condition_results,
        })

        if coordinator_writes and checkpoint_every and (ei % checkpoint_every == 0):
            _write_json_atomic(
                Path(checkpoint_path),
                {
                    "summary": {
                        "study_name": study_name,
                        "model": model_name,
                        "n_examples": len(examples),
                        "completed_examples": ei,
                        "stats": stats,
                    },
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
            n_done = ei
            parts = []
            for c in CONDITION_NAMES:
                s = stats[c]
                n = s["total"]
                if n > 0:
                    acc = s["correct"] / n
                    fw = s["follows_wrong"] / n
                    parts.append(f"{c}: acc={acc:.3f} fw={fw:.3f}")
            print(
                f"[progress] {ei}/{len(examples)} | " + " | ".join(parts) +
                f" | elapsed={elapsed:.0f}s",
                flush=True,
            )

    # Build summary
    summary_conditions: dict[str, Any] = {}
    print("\n=== FINAL RESULTS ===")
    for c in CONDITION_NAMES:
        s = stats[c]
        n = s["total"]
        if n == 0:
            continue
        acc = s["correct"] / n
        fw = s["follows_wrong"] / n
        summary_conditions[c] = {"n": n, "accuracy": acc, "followed_wrong_suffix_rate": fw}
        print(f"{c} (n={n}): accuracy={acc:.4f}  followed_wrong_suffix={fw:.4f}")
        if c == "conflicting_chain":
            if fw > 0.5:
                print("  >> RATIONALIZATION CONFIRMED: majority follow wrong explicit suffix")
            elif fw < 0.2:
                print("  >> GENUINE REASONING: majority recover correct answer from steps")
            else:
                print("  >> MIXED: intermediate signal")

    return {
        "summary": {
            "study_name": study_name,
            "model": model_name,
            "n_examples": len(examples),
            "conditions": summary_conditions,
        },
        "results": per_example_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Decisive conflicting-answer rationalization test")
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID")
    parser.add_argument("--data-file", required=True, help="Path to conflicting-answer JSONL")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--study-name", default="conflicting_answer", help="Study name")
    parser.add_argument("--progress-every", type=int, default=10)
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

    # Verify dataset integrity
    ok = 0
    for ex in examples[:3]:
        assert ex.correct_answer and ex.wrong_answer, f"Missing answers in {ex.example_id}"
        assert ex.steps_standard and ex.steps_conflicting, f"Missing steps in {ex.example_id}"
        assert ex.steps_standard[-1] != ex.steps_conflicting[-1], (
            f"Last steps are identical in {ex.example_id} — conflicting dataset not set up correctly"
        )
        ok += 1
    print(f"[startup] Dataset integrity check passed ({ok}/3 examples verified)", flush=True)

    print(f"[startup] Loading model {args.model_id} with sharding={sharding_axis_dims}", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model_id, sharding_axis_dims)
    reasoner = EasyDeLReasoner(model, tokenizer)
    print("[startup] Ready.", flush=True)

    import jax

    output = run_conflicting_experiment(
        examples=examples,
        reasoner=reasoner,
        study_name=args.study_name,
        model_name=args.model_id,
        progress_every=args.progress_every,
        checkpoint_every=max(0, args.checkpoint_every),
        checkpoint_path=args.checkpoint_path or (args.output + ".checkpoint.json"),
    )

    if jax.process_index() == 0:
        out_path = Path(args.output)
        _write_json_atomic(out_path, output)
        print(f"\n[done] Results written to {args.output}", flush=True)
    else:
        print("\n[done] Worker complete; coordinator writes final results", flush=True)


if __name__ == "__main__":
    main()
