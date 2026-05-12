#!/usr/bin/env python3
"""
Early-Stop Generation Probe: measure answer availability vs. reasoning steps revealed.

This script supports reference-chain and post-hoc self-prefix probes.
For a live decode-time branch-from-prefix commitment test, use the dedicated
prefix-branch probe runner.

DESIGN
------
Two modes:

  --mode reference  (fast):
    Use pre-written GSM8K steps from phase1_gsm8k_1000.jsonl.
    For each example, build probe prompts at k=0,1,...,K steps revealed.
    Measure accuracy vs. step count.
    This answers: "how much of the reference chain does the model need?"

    --mode generation  (post-hoc self-prefix):
        Generate the model's own chain first, parse it into steps, then probe
        prefixes of that completed self-generated chain.
        This is useful supporting evidence, but it is still a consumption-style
        prefix probe rather than a live decode-time commitment test.

KEY PREDICTION
--------------
If rationalization holds, accuracy should plateau quickly (after 1-2 steps).
If genuine reasoning holds, accuracy should rise gradually through the chain.

RESEARCH CLAIM SUPPORTED
-------------------------
"The model produces the correct answer after seeing only a fraction of the
chain, suggesting the answer is committed to early during generation and
later steps serve as post-hoc elaboration."

Usage:
  python3 run_early_stop_probe_tpu.py \\
    --model-id Qwen/Qwen2.5-3B-Instruct \\
    --data-file ~/data/phase1_gsm8k_1000.jsonl \\
    --output ~/results_fixed/early_stop_probe_qwen3b.json \\
    --mode reference \\
    --n-examples 200 \\
    --progress-every 10

  python3 run_early_stop_probe_tpu.py \\
    --model-id Qwen/Qwen2.5-3B-Instruct \\
    --data-file ~/data/phase1_gsm8k_1000.jsonl \\
    --output ~/results_fixed/early_stop_probe_gen_qwen3b.json \\
    --mode generation \\
    --n-examples 100 \\
    --progress-every 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Example:
    example_id: str
    question: str
    steps: list[str]          # reference steps (from dataset)
    correct_answer: str


@dataclass
class StepProbeResult:
    """Accuracy at each step of chain revelation."""
    example_id: str
    question: str
    correct_answer: str
    n_total_steps: int
    step_predictions: list[str]   # predicted answer at each k (k=0 means no steps)
    step_correct: list[bool]      # correct at each k
    full_chain_prediction: str | None = None
    full_chain_correct: bool | None = None
    # In generation mode: the model's own generated chain
    generated_chain: str | None = None
    generated_steps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Answer normalization / matching (from run_conflicting_answer_tpu.py)
# ---------------------------------------------------------------------------
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


def extract_answer(text: str) -> str | None:
    """Extract numerical answer from model output."""
    # Pattern: "the answer is X"
    m = re.search(r"(?:the\s+)?(?:final\s+)?answer(?:\s+is)?\s*[:\s]?\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
                  text, re.IGNORECASE)
    if m:
        return m.group(1).strip().replace(",", "")
    # Pattern: "= X" at end of line
    m = re.search(r"=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).strip().replace(",", "")
    # Pattern: "#### X" (GSM8K style)
    m = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).strip().replace(",", "")
    # Last number in output
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None


# ---------------------------------------------------------------------------
# Step parsing
# ---------------------------------------------------------------------------
def parse_chain_into_steps(chain_text: str) -> list[str]:
    """Parse a free-form CoT into discrete steps.

    Splits on numbered patterns (Step 1:, 1., etc.) or double newlines.
    """
    # Try numbered step pattern first: "Step 1:", "1.", "Step 1 -"
    numbered = re.split(
        r"(?:^|\n)\s*(?:Step\s+)?\d+[.:)]\s*",
        chain_text,
        flags=re.IGNORECASE,
    )
    steps = [s.strip() for s in numbered if s.strip()]
    if len(steps) >= 2:
        return steps

    # Try newline-based splitting
    lines = [l.strip() for l in chain_text.split("\n") if l.strip()]
    if len(lines) >= 2:
        return lines

    # Single block: treat as one step
    return [chain_text.strip()] if chain_text.strip() else []


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_probe_prompt_reference(question: str, steps: list[str], k: int) -> str:
    """Build a probe prompt revealing exactly k reference steps.

    k=0: question only (no steps revealed).
    k>0: first k steps revealed.
    """
    if k == 0:
        return (
            "You are given a math problem. "
            "Provide the final numerical answer as a single number with no explanation.\n\n"
            f"Problem: {question}\n\n"
            "Final answer:"
        )
    step_text = "\n".join(
        f"Step {i+1}: {steps[i]}" for i in range(min(k, len(steps)))
    )
    return (
        "You are given a math problem and some reasoning steps. "
        "Based on this partial reasoning, provide the final numerical answer "
        "as a single number with no explanation.\n\n"
        f"Problem: {question}\n\n"
        f"Reasoning so far:\n{step_text}\n\n"
        "Final answer:"
    )


def build_probe_prompt_generation(question: str, generated_steps: list[str], k: int) -> str:
    """Same structure but using model's own generated steps."""
    return build_probe_prompt_reference(question, generated_steps, k)


def build_generation_prompt(question: str) -> str:
    """Prompt for generating a full CoT chain."""
    return (
        "Solve this math problem step by step. "
        "Number each step. State the final answer as a number after 'The answer is:'.\n\n"
        f"Problem: {question}\n\nSolution:\n"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_examples(path: str, n: int | None = None) -> list[Example]:
    examples = []
    data_path = Path(path).expanduser()
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            eid = str(rec.get("id") or rec.get("example_id") or f"ex_{len(examples)}")
            question = str(rec.get("question") or rec.get("prompt") or "")
            # The JSONL has 'steps' as a list and 'answer' as the final answer
            steps = rec.get("steps") or []
            if isinstance(steps, str):
                steps = parse_chain_into_steps(steps)
            correct_answer = str(rec.get("answer") or rec.get("correct_answer") or "")
            # Clean up GSM8K-style <<expression=value>> annotations from steps
            clean_steps = [re.sub(r"<<[^>]*>>", "", s).strip() for s in steps]
            correct_answer = re.sub(r"<<[^>]*>>", "", correct_answer).strip()
            # Remove trailing #### marker if present
            correct_answer = re.sub(r"####\s*", "", correct_answer).strip()
            if question and correct_answer:
                examples.append(Example(
                    example_id=eid,
                    question=question,
                    steps=clean_steps,
                    correct_answer=correct_answer,
                ))
            if n is not None and len(examples) >= n:
                break
    return examples


# ---------------------------------------------------------------------------
# Model loading + inference (same EasyDeL pattern as run_conflicting_answer_tpu.py)
# ---------------------------------------------------------------------------
FIXED_BUF_LEN = 640


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def load_model_and_tokenizer(model_id: str, sharding: str = "1,-1,1,1,1"):
    import jax
    import jax.numpy as jnp
    from jax import lax
    import easydel as ed
    from transformers import AutoTokenizer

    process_index = jax.process_index()
    device_count = jax.device_count()
    print(f"[init] process_index={process_index} device_count={device_count}", flush=True)

    axis_dims = tuple(int(x) for x in sharding.split(","))
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        model_id,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=lax.Precision.DEFAULT,
        auto_shard_model=True,
        sharding_axis_dims=axis_dims,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
            attn_dtype=jnp.bfloat16,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
        ),
        partition_axis=ed.PartitionAxis(),
    )

    # Patch _is_array in the module that actually uses it (modeling_outputs)
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

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[init] Model loaded. process_index={process_index}", flush=True)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference helpers (greedy decode via Python loop)
# ---------------------------------------------------------------------------
def _greedy_decode(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Greedy token-by-token generation — proven pattern from run_conflicting_answer_tpu.py."""
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
    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id
    stop_ids: set[int] = set()
    if eos_id is not None:
        stop_ids.add(int(eos_id))
    if tokenizer.pad_token_id is not None:
        stop_ids.add(int(tokenizer.pad_token_id))
    # Qwen2.5 commonly emits <|endoftext|> as the actual stop token.
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

    for _ in range(max_new_tokens):
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


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 24) -> str:
    """Short inference for probing (answer extraction, 24 tokens)."""
    return _greedy_decode(model, tokenizer, prompt, max_new_tokens)


def run_generation_inference(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Longer inference for full CoT chain generation."""
    return _greedy_decode(model, tokenizer, prompt, max_new_tokens)


# ---------------------------------------------------------------------------
# Core experiment: reference mode
# ---------------------------------------------------------------------------
def probe_reference_example(
    example: Example,
    model,
    tokenizer,
    max_steps: int | None = None,
) -> StepProbeResult:
    """Probe accuracy at each step k=0..K using reference chain steps."""
    total_steps = len(example.steps)
    K = total_steps
    if max_steps is not None:
        K = min(K, max_steps)

    step_predictions = []
    step_correct = []

    for k in range(K + 1):
        prompt = build_probe_prompt_reference(example.question, example.steps, k)
        raw = run_inference(model, tokenizer, prompt)
        pred = extract_answer(raw) or raw.strip()
        correct = answers_match(pred, example.correct_answer)
        step_predictions.append(pred)
        step_correct.append(correct)

    full_chain_prediction = step_predictions[-1] if step_predictions else None
    full_chain_correct = step_correct[-1] if step_correct else None
    if K < total_steps:
        prompt = build_probe_prompt_reference(example.question, example.steps, total_steps)
        raw = run_inference(model, tokenizer, prompt)
        full_chain_prediction = extract_answer(raw) or raw.strip()
        full_chain_correct = answers_match(full_chain_prediction, example.correct_answer)

    return StepProbeResult(
        example_id=example.example_id,
        question=example.question,
        correct_answer=example.correct_answer,
        n_total_steps=total_steps,
        step_predictions=step_predictions,
        step_correct=step_correct,
        full_chain_prediction=full_chain_prediction,
        full_chain_correct=full_chain_correct,
    )


# ---------------------------------------------------------------------------
# Core experiment: generation mode
# ---------------------------------------------------------------------------
def probe_generation_example(
    example: Example,
    model,
    tokenizer,
    max_steps: int | None = None,
) -> StepProbeResult:
    """Generate model's own chain, then probe accuracy at each step k=0..K."""
    # Phase 1: Generate full CoT
    gen_prompt = build_generation_prompt(example.question)
    generated_chain = run_generation_inference(model, tokenizer, gen_prompt)

    # Parse into steps
    generated_steps = parse_chain_into_steps(generated_chain)
    total_steps = len(generated_steps)
    K = total_steps
    if max_steps is not None:
        K = min(K, max_steps)

    step_predictions = []
    step_correct = []

    for k in range(K + 1):
        prompt = build_probe_prompt_generation(example.question, generated_steps, k)
        raw = run_inference(model, tokenizer, prompt)
        pred = extract_answer(raw) or raw.strip()
        correct = answers_match(pred, example.correct_answer)
        step_predictions.append(pred)
        step_correct.append(correct)

    full_chain_prediction = step_predictions[-1] if step_predictions else None
    full_chain_correct = step_correct[-1] if step_correct else None
    if K < total_steps:
        prompt = build_probe_prompt_generation(example.question, generated_steps, total_steps)
        raw = run_inference(model, tokenizer, prompt)
        full_chain_prediction = extract_answer(raw) or raw.strip()
        full_chain_correct = answers_match(full_chain_prediction, example.correct_answer)

    return StepProbeResult(
        example_id=example.example_id,
        question=example.question,
        correct_answer=example.correct_answer,
        n_total_steps=total_steps,
        step_predictions=step_predictions,
        step_correct=step_correct,
        full_chain_prediction=full_chain_prediction,
        full_chain_correct=full_chain_correct,
        generated_chain=generated_chain,
        generated_steps=generated_steps,
    )


# ---------------------------------------------------------------------------
# Aggregate step accuracies
# ---------------------------------------------------------------------------
def aggregate_step_accuracies(results: list[StepProbeResult]) -> dict[str, Any]:
    """Compute accuracy at each step position across all examples."""
    from collections import defaultdict

    # Bucket by step index (k=0 is no-step baseline, k=max is last probed prefix)
    step_correct_counts: dict[int, int] = defaultdict(int)
    step_totals: dict[int, int] = defaultdict(int)
    max_k = max(len(r.step_correct) for r in results) - 1

    for r in results:
        K_example = len(r.step_correct) - 1
        for k in range(K_example + 1):
            step_correct_counts[k] += int(r.step_correct[k])
            step_totals[k] += 1

    # Also compute "fractional position" normalized accuracies
    # Each example's k gets normalized to its fraction of the probed chain
    BINS = 5  # 0%, 25%, 50%, 75%, 100%
    bin_correct: dict[int, int] = defaultdict(int)
    bin_totals: dict[int, int] = defaultdict(int)

    for r in results:
        K = max(len(r.step_correct) - 1, 1)
        for k, correct in enumerate(r.step_correct):
            frac = k / K
            # Assign to bin 0-4
            b = min(int(frac * BINS), BINS - 1)
            bin_correct[b] += int(correct)
            bin_totals[b] += 1

    step_accuracies = {
        k: step_correct_counts[k] / step_totals[k]
        for k in sorted(step_totals.keys())
    }
    bin_accuracies = {
        b: bin_correct[b] / bin_totals[b]
        for b in sorted(bin_totals.keys())
    }
    last_probed_accuracy = (
        sum(int(r.step_correct[-1]) for r in results) / len(results)
        if results else 0.0
    )
    full_chain_values = [
        r.full_chain_correct if r.full_chain_correct is not None else r.step_correct[-1]
        for r in results
    ]
    full_chain_accuracy = (
        sum(int(value) for value in full_chain_values) / len(full_chain_values)
        if full_chain_values else 0.0
    )

    return {
        "n_examples": len(results),
        "max_steps": max_k,
        "step_accuracies": {str(k): v for k, v in step_accuracies.items()},
        "step_counts": {str(k): step_totals[k] for k in sorted(step_totals.keys())},
        "normalized_bin_accuracies": {
            f"bin_{b}_({b*100//BINS}pct-{(b+1)*100//BINS}pct)": bin_accuracies[b]
            for b in sorted(bin_accuracies.keys())
        },
        # Key summary stats
        "accuracy_at_step_0": step_accuracies.get(0, 0.0),
        "accuracy_at_step_1": step_accuracies.get(1, 0.0),
        "accuracy_at_step_2": step_accuracies.get(2, 0.0),
        "accuracy_at_last_probed_prefix": last_probed_accuracy,
        "accuracy_at_full_chain": full_chain_accuracy,
        "early_saturation_ratio": (
            step_accuracies.get(1, 0.0) / max(full_chain_accuracy, 1e-9)
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Early-stop generation probe on TPU")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["reference", "generation"], default="reference",
                        help="reference: use pre-written chains; generation: model generates own chain")
    parser.add_argument("--n-examples", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=8,
                        help="Max steps to probe per example (cap to reduce cost)")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Write coordinator-only checkpoint JSON every N examples (0 disables)")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Checkpoint JSON path (default: <output>.checkpoint.json)")
    parser.add_argument("--sharding", type=str, default="1,-1,1,1,1")
    args = parser.parse_args()

    import jax
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.sharding)
    process_index = jax.process_index()

    examples = load_examples(args.data_file, n=args.n_examples)
    if process_index == 0:
        print(f"[startup] Ready. Loaded {len(examples)} examples. Mode={args.mode}", flush=True)

    results: list[StepProbeResult] = []
    t0 = time.time()
    checkpoint_path = Path((args.checkpoint_path or (args.output + ".checkpoint.json"))).expanduser()

    for idx, example in enumerate(examples):
        if args.mode == "reference":
            result = probe_reference_example(
                example, model, tokenizer, max_steps=args.max_steps
            )
        else:
            result = probe_generation_example(
                example, model, tokenizer, max_steps=args.max_steps
            )
        results.append(result)

        if process_index == 0 and (idx + 1) % args.progress_every == 0:
            done = idx + 1
            elapsed = time.time() - t0
            agg = aggregate_step_accuracies(results)
            acc_0 = agg["accuracy_at_step_0"]
            acc_1 = agg["accuracy_at_step_1"]
            acc_full = agg["accuracy_at_full_chain"]
            ratio = agg["early_saturation_ratio"]
            print(
                f"[progress] {done}/{len(examples)} | "
                f"acc_k0={acc_0:.3f} acc_k1={acc_1:.3f} acc_full={acc_full:.3f} "
                f"early_ratio={ratio:.3f} | elapsed={elapsed:.0f}s",
                flush=True,
            )

        if process_index == 0 and args.checkpoint_every > 0 and (idx + 1) % args.checkpoint_every == 0:
            agg = aggregate_step_accuracies(results)
            _write_json_atomic(
                checkpoint_path,
                {
                    "study_name": f"early_stop_probe_{args.mode}",
                    "model_id": args.model_id,
                    "mode": args.mode,
                    "n_examples": len(results),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "aggregated": agg,
                    "per_example": [
                        {
                            "example_id": r.example_id,
                            "correct_answer": r.correct_answer,
                            "n_total_steps": r.n_total_steps,
                            "step_predictions": r.step_predictions,
                            "step_correct": r.step_correct,
                            "full_chain_prediction": r.full_chain_prediction,
                            "full_chain_correct": r.full_chain_correct,
                            "generated_chain": r.generated_chain,
                            "generated_steps": r.generated_steps,
                        }
                        for r in results
                    ],
                },
            )
            print(f"[checkpoint] wrote {len(results)}/{len(examples)} to {checkpoint_path}", flush=True)

    # Save results (coordinator only)
    if process_index == 0:
        agg = aggregate_step_accuracies(results)
        out = {
            "study_name": f"early_stop_probe_{args.mode}",
            "model_id": args.model_id,
            "mode": args.mode,
            "n_examples": len(results),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "aggregated": agg,
            "per_example": [
                {
                    "example_id": r.example_id,
                    "correct_answer": r.correct_answer,
                    "n_total_steps": r.n_total_steps,
                    "step_predictions": r.step_predictions,
                    "step_correct": r.step_correct,
                    "full_chain_prediction": r.full_chain_prediction,
                    "full_chain_correct": r.full_chain_correct,
                    "generated_chain": r.generated_chain,
                    "generated_steps": r.generated_steps,
                }
                for r in results
            ],
        }
        output_path = Path(args.output).expanduser()
        _write_json_atomic(output_path, out)
        print(f"[done] Results saved to {output_path}", flush=True)
        print(f"[summary] accuracy_at_k0={agg['accuracy_at_step_0']:.3f}", flush=True)
        print(f"[summary] accuracy_at_k1={agg['accuracy_at_step_1']:.3f}", flush=True)
        print(f"[summary] accuracy_at_full={agg['accuracy_at_full_chain']:.3f}", flush=True)
        print(f"[summary] early_saturation_ratio={agg['early_saturation_ratio']:.3f}", flush=True)
        print(f"[summary] step_accuracies={agg['step_accuracies']}", flush=True)


if __name__ == "__main__":
    main()
