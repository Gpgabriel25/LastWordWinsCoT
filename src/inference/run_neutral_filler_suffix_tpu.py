#!/usr/bin/env python3
"""
Neutral-filler suffix control experiment (W5).

Addresses the recency-bias alternative explanation for suffix FW=0.39:
  Is the effect driven by ANSWER-TEXT CONTENT at the suffix,
  or by END-OF-CHAIN RECENCY/POSITION alone?

Design (2 conditions on same 100 suffix examples):
  A. conflicting_suffix    : standard chain + "Therefore, the answer is [WRONG]." at end
     (baseline; expected FW approx 0.39 — replicates original finding)
  B. neutral_filler_suffix : same computation steps + neutral filler at end (no answer text)
     (expected FW approx 0 if content-driven, elevated if recency-driven)

Key metric: FW ratio = conflicting_fw / neutral_fw
  - ratio >> 1 (>=5x) -> CONTENT-DRIVEN (answer text causes FW, not position)
  - ratio approx 1    -> RECENCY BIAS (position alone causes FW)

Model: Qwen/Qwen2.5-3B-Instruct (matches original counterbalanced placement experiment)
Dataset: phase0_counterbalanced_placement_gsm8k_v1.jsonl (suffix examples only, first 100)

Usage (single-worker, 4 chips):
  python3 scripts/run_neutral_filler_suffix_tpu.py \
    --model-id Qwen/Qwen2.5-3B-Instruct \
    --data-file ~/data/phase0_counterbalanced_placement_gsm8k_v1.jsonl \
    --out-file ~/results_fixed/neutral_filler_suffix_qwen3b.json \
    --n 100
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from ast import Add, BinOp, Constant, Div, Expression, Mult, Sub, UnaryOp, USub
from ast import parse as ast_parse


NEUTRAL_FILLER = "The computation above confirms the reasoning."

# ---------------------------------------------------------------------------
# Answer extraction — copied verbatim from run_counterbalanced_placement_tpu.py
# ---------------------------------------------------------------------------
ANSWER_PATTERNS = [
    re.compile(r"(?:answer|result|total)[\s\S]*?[:\s]+\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:days?|hours?|minutes?|dollars?|cents?|apples?|oranges?|people|items?|units?)?\s*\.?\s*$", re.IGNORECASE),
]


def _maybe_eval_expression(text: str):
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


def _extract_leading_answer(text: str):
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


def extract_answer(text: str):
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
    return " ".join(text.strip().lower().rstrip(".!,;:").replace(",", "").split())


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
# Prompt building — same format as run_counterbalanced_placement_tpu.py
# ---------------------------------------------------------------------------
def build_chain_prompt(question: str, steps: list) -> str:
    if not steps:
        return (
            "Solve the problem. "
            "Return exactly one integer answer with no words, no explanation, "
            "and no punctuation other than a leading minus sign if needed.\n"
            f"Question: {question}\n"
            "Final answer:"
        )
    reasoning = "\n".join(f"Step {i + 1}: {step}" for i, step in enumerate(steps))
    return (
        "Solve the problem using the provided reasoning. "
        "Return exactly one integer answer with no words, no explanation, "
        "and no punctuation other than a leading minus sign if needed.\n"
        f"Question: {question}\n"
        f"Reasoning:\n{reasoning}\n"
        "Final answer:"
    )


# ---------------------------------------------------------------------------
# EasyDeL model loading — copied from run_counterbalanced_placement_tpu.py
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_id: str, sharding_axis_dims: tuple):
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
# EasyDeL reasoner — copied from run_counterbalanced_placement_tpu.py
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
            print("[init] Patched EasyDeL _is_array", flush=True)
        except Exception as exc:
            print(f"[init] WARNING: could not patch _is_array: {exc}", flush=True)

    def predict(self, question: str, correct_answer: str, wrong_answer: str,
                steps: list) -> dict:
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P

        prompt_text = build_chain_prompt(question, steps)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="np",
            truncation=True,
            max_length=512,
            padding=False,
        )

        raw_ids = jnp.array(encoded["input_ids"])
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

        gen_tokens: list = []
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
        predicted = coerce_answer(extracted) if extracted else completion.strip()
        is_correct = answers_match(predicted, correct_answer)
        follows_wrong = (
            answers_match(predicted, wrong_answer)
            and not answers_match(wrong_answer, correct_answer)
        )
        return {
            "predicted": predicted,
            "raw_output": completion[:100],
            "is_correct": is_correct,
            "follows_wrong": follows_wrong,
        }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="W5 Neutral-filler suffix control")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--n", type=int, default=100, help="Number of suffix examples")
    parser.add_argument("--sharding", default="1,-1,1,1,1")
    args = parser.parse_args()

    # Enable transparent hugepages
    try:
        with open("/sys/kernel/mm/transparent_hugepage/enabled", "w") as f:
            f.write("always")
    except Exception:
        pass

    sharding = tuple(int(x) for x in args.sharding.split(","))
    print(f"[startup] Loading model {args.model_id} sharding={sharding}", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model_id, sharding)
    reasoner = EasyDeLReasoner(model, tokenizer)
    print("[startup] Model ready.", flush=True)

    with open(args.data_file) as f:
        all_data = [json.loads(l) for l in f if l.strip()]
    suffix_examples = [e for e in all_data if e["placement"] == "suffix"][:args.n]
    print(f"[startup] Using {len(suffix_examples)} suffix examples.", flush=True)

    conflicting_results = []
    neutral_results = []
    t0 = time.time()

    for i, ex in enumerate(suffix_examples):
        q = ex["question"]
        correct = str(ex["correct_answer"])
        wrong = str(ex["wrong_answer"])

        # Condition A: conflicting chain (wrong answer in last step)
        res_c = reasoner.predict(q, correct, wrong, list(ex["steps_conflicting"]))
        conflicting_results.append(res_c)

        # Condition B: neutral filler (same intermediate steps, no answer text)
        # steps_conflicting and steps_standard share identical intermediate steps here;
        # we use steps_conflicting[:-1] to be explicit about replacing only the last step
        neutral_steps = list(ex["steps_conflicting"][:-1]) + [NEUTRAL_FILLER]
        res_n = reasoner.predict(q, correct, wrong, neutral_steps)
        neutral_results.append(res_n)

        if (i + 1) % 10 == 0:
            fw_c = sum(r["follows_wrong"] for r in conflicting_results) / (i + 1)
            fw_n = sum(r["follows_wrong"] for r in neutral_results) / (i + 1)
            elapsed = time.time() - t0
            print(
                f"[progress] {i+1}/{len(suffix_examples)}"
                f"  conflicting_FW={fw_c:.3f}"
                f"  neutral_FW={fw_n:.3f}"
                f"  elapsed={elapsed:.0f}s",
                flush=True,
            )

    n = len(suffix_examples)
    conflict_fw = sum(r["follows_wrong"] for r in conflicting_results) / n
    neutral_fw = sum(r["follows_wrong"] for r in neutral_results) / n
    conflict_acc = sum(r["is_correct"] for r in conflicting_results) / n
    neutral_acc = sum(r["is_correct"] for r in neutral_results) / n
    fw_ratio = conflict_fw / max(neutral_fw, 1e-6)

    print(f"\n===== W5 NEUTRAL FILLER RESULTS =====")
    print(f"  Model:               {args.model_id}")
    print(f"  N:                   {n}")
    print(f"  Neutral filler:      '{NEUTRAL_FILLER}'")
    print(f"  Conflicting FW:      {conflict_fw:.3f}  (acc={conflict_acc:.3f})")
    print(f"  Neutral filler FW:   {neutral_fw:.3f}  (acc={neutral_acc:.3f})")
    print(f"  FW ratio:            {fw_ratio:.1f}x")
    if neutral_fw < 0.08:
        conclusion = "content-dependent"
        print("  CONCLUSION: FW is CONTENT-DEPENDENT (answer text, not position/recency)")
    elif neutral_fw > 0.20:
        conclusion = "recency-bias"
        print("  CONCLUSION: RECENCY BIAS -- elevated FW even without explicit answer text")
    else:
        conclusion = "ambiguous"
        print("  CONCLUSION: AMBIGUOUS")

    output = {
        "model_id": args.model_id,
        "n": n,
        "neutral_filler_text": NEUTRAL_FILLER,
        "conflicting_suffix": {
            "fw_rate": round(conflict_fw, 4),
            "accuracy": round(conflict_acc, 4),
        },
        "neutral_filler_suffix": {
            "fw_rate": round(neutral_fw, 4),
            "accuracy": round(neutral_acc, 4),
        },
        "fw_ratio": round(fw_ratio, 2),
        "conclusion": conclusion,
        "per_example": [
            {
                "id": ex["id"],
                "correct": ex["correct_answer"],
                "wrong": ex["wrong_answer"],
                "conflicting": conflicting_results[i],
                "neutral": neutral_results[i],
            }
            for i, ex in enumerate(suffix_examples)
        ],
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[done] Saved to {args.out_file}", flush=True)


if __name__ == "__main__":
    main()
