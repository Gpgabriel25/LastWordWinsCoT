#!/usr/bin/env python3
"""
Live prefix-branch commitment probe on TPU.

This is the generation-time behavioral test that branches from the model's
exact self-generated prefix during decoding, rather than generating a full
chain first and then re-prompting on reconstructed prefixes.

Protocol
--------
1. Prompt the model to solve a problem step by step with numbered steps.
2. Generate its chain token by token.
3. Whenever a numbered step is completed, branch from that exact emitted
   prefix into an answer-only continuation by appending "The answer is:".
4. Record whether the branched answer is correct and whether it matches the
   model's own eventual free-run final answer.

Interpretation
--------------
- High early branch accuracy and high early match-to-final rates suggest early
  answer commitment.
- Low early branch accuracy with a late rise suggests gradual computation or at
  least no strong evidence for early commitment.

Usage:
  python3 run_prefix_branch_probe_tpu.py \
    --model-id Qwen/Qwen2.5-3B-Instruct \
    --data-file ~/data/phase0_conflicting_answer_gsm8k_v2.jsonl \
    --output ~/results_fixed/prefix_branch_probe_qwen3b.json \
    --study-name prefix_branch_probe_qwen3b \
    --n-examples 100 \
    --max-probe-steps 4 \
    --progress-every 5
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


STEP_HEADER_RE = re.compile(r"(?m)^\s*(?:Step\s+)?(\d+)[\.:)]\s*")
ANSWER_CUE_RE = re.compile(r"\b(?:the\s+)?answer\s+is\b", re.IGNORECASE)
FIXED_BUF_LEN = 640
MAX_BUF_LEN = 1024


@dataclass(slots=True)
class Example:
    example_id: str
    question: str
    correct_answer: str


@dataclass(slots=True)
class ProbeRecord:
    step_index: int
    prefix_preview: str
    predicted_answer: str
    raw_output: str
    is_correct: bool
    matches_final: bool | None = None


@dataclass(slots=True)
class PrefixBranchResult:
    example_id: str
    question: str
    correct_answer: str
    generated_chain: str
    generated_steps: list[str]
    final_generated_answer: str
    final_generated_parseable: bool
    final_generated_correct: bool
    probes: list[ProbeRecord] = field(default_factory=list)


def normalize_answer(text: str) -> str:
    normalized = text.strip().lower().rstrip(".!,:;").replace(",", "")
    return " ".join(normalized.split())


def answers_match(pred: str, target: str) -> bool:
    pred_n = normalize_answer(pred)
    tgt_n = normalize_answer(target)
    if pred_n == tgt_n:
        return True
    try:
        return abs(float(pred_n) - float(tgt_n)) < 1e-6
    except (TypeError, ValueError):
        return False


def extract_answer(text: str) -> str | None:
    patterns = [
        re.compile(
            r"(?:the\s+)?(?:final\s+)?answer(?:\s+is)?\s*[:\s]?\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)"),
        re.compile(r"=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*$", re.MULTILINE),
    ]
    for pattern in patterns:
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(1).strip().replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def load_examples(path: str, n: int | None = None) -> list[Example]:
    examples: list[Example] = []
    data_path = Path(path).expanduser()
    with data_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = str(record.get("question") or record.get("prompt") or "")
            correct_answer = str(record.get("correct_answer") or record.get("answer") or "")
            if not question or not correct_answer:
                continue
            correct_answer = re.sub(r"<<[^>]*>>", "", correct_answer).strip()
            correct_answer = re.sub(r"####\s*", "", correct_answer).strip()
            example_id = str(record.get("id") or record.get("example_id") or f"ex_{len(examples)}")
            examples.append(
                Example(
                    example_id=example_id,
                    question=question,
                    correct_answer=correct_answer,
                )
            )
            if n is not None and len(examples) >= n:
                break
    return examples


def build_generation_prompt(question: str) -> str:
    return (
        "Solve this math problem step by step. "
        "Number each step on its own line as 'Step N:'. "
        "After the reasoning, give the final answer after 'The answer is:'.\n\n"
        f"Problem: {question}\n\n"
        "Solution:\n"
    )


def build_branch_prompt(base_prompt: str, prefix_text: str) -> str:
    prefix = prefix_text.rstrip()
    if prefix:
        return f"{base_prompt}{prefix}\nThe answer is:"
    return f"{base_prompt}The answer is:"


def extract_completed_step_infos(
    prefix_text: str,
    finalize: bool = False,
) -> list[tuple[str, int]]:
    answer_match = ANSWER_CUE_RE.search(prefix_text)
    answer_pos = answer_match.start() if answer_match else None
    headers = [
        match
        for match in STEP_HEADER_RE.finditer(prefix_text)
        if answer_pos is None or match.start() < answer_pos
    ]

    step_infos: list[tuple[str, int]] = []
    for index, header in enumerate(headers):
        content_start = header.end()
        if index + 1 < len(headers):
            boundary = headers[index + 1].start()
        elif answer_pos is not None:
            boundary = answer_pos
        elif finalize:
            boundary = len(prefix_text)
        else:
            continue

        step_text = prefix_text[content_start:boundary].strip()
        if step_text:
            step_infos.append((step_text, boundary))
    return step_infos


def parse_completed_steps(prefix_text: str, finalize: bool = False) -> list[str]:
    return [step_text for step_text, _ in extract_completed_step_infos(prefix_text, finalize=finalize)]


def prefix_preview(prefix_text: str, limit: int = 240) -> str:
    compact = " ".join(prefix_text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


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
    axis_names = ("dp", "fsdp", "tp", "sp", "ep") if len(axis_dims) == 5 else ("dp", "fsdp", "tp", "sp")
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        model_id,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=lax.Precision.DEFAULT,
        auto_shard_model=True,
        sharding_axis_dims=axis_dims,
        sharding_axis_names=axis_names,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
            attn_dtype=jnp.bfloat16,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
        ),
        partition_axis=ed.PartitionAxis(),
    )

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

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[init] Model loaded. process_index={process_index}", flush=True)
    return model, tokenizer


def _greedy_decode(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    encoded = tokenizer(
        prompt,
        return_tensors="np",
        truncation=False,
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
    stop_ids.add(151643)

    required_len = orig_len + max_new_tokens
    buf_len = max(FIXED_BUF_LEN, required_len)
    if buf_len > MAX_BUF_LEN:
        raise ValueError(
            f"Prompt length {orig_len} + max_new_tokens {max_new_tokens} exceeds MAX_BUF_LEN={MAX_BUF_LEN}"
        )

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


def run_prefix_branch_example(
    example: Example,
    model,
    tokenizer,
    max_probe_steps: int,
    max_generation_tokens: int,
    max_answer_tokens: int,
) -> PrefixBranchResult:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    base_prompt = build_generation_prompt(example.question)
    probes: list[ProbeRecord] = []

    try:
        baseline_raw = _greedy_decode(
            model,
            tokenizer,
            build_branch_prompt(base_prompt, ""),
            max_answer_tokens,
        )
        baseline_pred = extract_answer(baseline_raw) or baseline_raw.strip()
        baseline_is_correct = answers_match(baseline_pred, example.correct_answer)
    except ValueError as exc:
        baseline_raw = f"[prompt_too_long] {exc}"
        baseline_pred = ""
        baseline_is_correct = False
    probes.append(
        ProbeRecord(
            step_index=0,
            prefix_preview="",
            predicted_answer=baseline_pred,
            raw_output=baseline_raw,
            is_correct=baseline_is_correct,
        )
    )

    encoded = tokenizer(
        base_prompt,
        return_tensors="np",
        truncation=False,
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
    stop_ids.add(151643)

    required_len = orig_len + max_generation_tokens
    buf_len = max(FIXED_BUF_LEN, required_len)
    if buf_len > MAX_BUF_LEN:
        raise ValueError(
            f"Base prompt length {orig_len} + max_generation_tokens {max_generation_tokens} exceeds MAX_BUF_LEN={MAX_BUF_LEN}"
        )

    ids_buf = jnp.full((1, buf_len), pad_id, dtype=jnp.int32)
    mask_buf = jnp.zeros((1, buf_len), dtype=jnp.int32)
    ids_buf = ids_buf.at[0, :orig_len].set(raw_ids[0, :orig_len])
    mask_buf = mask_buf.at[0, :orig_len].set(1)

    mesh = model.config.mesh
    replicated = NamedSharding(mesh, P())
    ids_buf = jax.device_put(ids_buf, replicated)
    mask_buf = jax.device_put(mask_buf, replicated)

    gen_tokens: list[int] = []
    decoded_prefix = ""
    cur_pos = orig_len
    last_probed_step = 0
    for _ in range(max_generation_tokens):
        if cur_pos >= buf_len:
            break
        with mesh:
            outputs = model(ids_buf, attention_mask=mask_buf)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = int(jnp.argmax(logits[0, cur_pos - 1, :]))
        if next_token_id in stop_ids:
            break
        gen_tokens.append(next_token_id)
        token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
        if token_text:
            decoded_prefix += token_text
        ids_buf = ids_buf.at[0, cur_pos].set(next_token_id)
        mask_buf = mask_buf.at[0, cur_pos].set(1)
        cur_pos += 1

        if (
            token_text
            and last_probed_step < max_probe_steps
            and "\n" in token_text
        ):
            completed_steps = extract_completed_step_infos(decoded_prefix)
            current_step = len(completed_steps)
            if current_step > last_probed_step:
                for step_index in range(last_probed_step + 1, min(current_step, max_probe_steps) + 1):
                    _, prefix_end = completed_steps[step_index - 1]
                    completed_prefix = decoded_prefix[:prefix_end].rstrip()
                    try:
                        branch_raw = _greedy_decode(
                            model,
                            tokenizer,
                            build_branch_prompt(base_prompt, completed_prefix),
                            max_answer_tokens,
                        )
                        branch_pred = extract_answer(branch_raw) or branch_raw.strip()
                        branch_is_correct = answers_match(branch_pred, example.correct_answer)
                    except ValueError as exc:
                        branch_raw = f"[prompt_too_long] {exc}"
                        branch_pred = ""
                        branch_is_correct = False
                    probes.append(
                        ProbeRecord(
                            step_index=step_index,
                            prefix_preview=prefix_preview(completed_prefix),
                            predicted_answer=branch_pred,
                            raw_output=branch_raw,
                            is_correct=branch_is_correct,
                        )
                    )
                last_probed_step = current_step

    generated_chain = decoded_prefix.strip()
    generated_steps = parse_completed_steps(generated_chain, finalize=True)
    final_answer_match = extract_answer(generated_chain)
    final_generated_answer = final_answer_match or ""
    final_generated_parseable = final_answer_match is not None
    final_generated_correct = final_generated_parseable and answers_match(
        final_generated_answer,
        example.correct_answer,
    )

    for probe in probes:
        if final_generated_parseable:
            probe.matches_final = answers_match(probe.predicted_answer, final_generated_answer)

    return PrefixBranchResult(
        example_id=example.example_id,
        question=example.question,
        correct_answer=example.correct_answer,
        generated_chain=generated_chain,
        generated_steps=generated_steps,
        final_generated_answer=final_generated_answer,
        final_generated_parseable=final_generated_parseable,
        final_generated_correct=final_generated_correct,
        probes=probes,
    )


def aggregate_prefix_branch(results: list[PrefixBranchResult]) -> dict[str, Any]:
    from collections import defaultdict

    probe_correct: dict[int, int] = defaultdict(int)
    probe_total: dict[int, int] = defaultdict(int)
    probe_match_final: dict[int, int] = defaultdict(int)
    probe_match_total: dict[int, int] = defaultdict(int)

    parseable_examples = 0
    final_correct = 0
    final_parseable = 0
    for result in results:
        if result.generated_steps:
            parseable_examples += 1
        if result.final_generated_parseable:
            final_parseable += 1
        if result.final_generated_correct:
            final_correct += 1
        for probe in result.probes:
            step = probe.step_index
            probe_total[step] += 1
            probe_correct[step] += int(probe.is_correct)
            if probe.matches_final is not None:
                probe_match_total[step] += 1
                probe_match_final[step] += int(probe.matches_final)

    probe_accuracies = {
        str(step): probe_correct[step] / probe_total[step]
        for step in sorted(probe_total)
    }
    match_final_rates = {
        str(step): probe_match_final[step] / probe_match_total[step]
        for step in sorted(probe_match_total)
    }
    full_generation_accuracy = final_correct / len(results) if results else 0.0
    step1_accuracy = probe_accuracies.get("1", 0.0)
    step1_match_final = match_final_rates.get("1", 0.0)

    return {
        "n_examples": len(results),
        "parseable_examples": parseable_examples,
        "parseable_fraction": parseable_examples / len(results) if results else 0.0,
        "final_answer_parseable_examples": final_parseable,
        "final_answer_parseable_fraction": final_parseable / len(results) if results else 0.0,
        "probe_accuracies": probe_accuracies,
        "probe_counts": {str(step): probe_total[step] for step in sorted(probe_total)},
        "match_final_counts": {str(step): probe_match_total[step] for step in sorted(probe_match_total)},
        "match_final_rates": match_final_rates,
        "full_generation_accuracy": full_generation_accuracy,
        "step1_accuracy": step1_accuracy,
        "step1_match_final_rate": step1_match_final,
        "early_commitment_ratio_step1": (
            step1_accuracy / full_generation_accuracy if full_generation_accuracy > 0.0 else None
        ),
    }


def init_running_progress_stats() -> dict[str, Any]:
    from collections import defaultdict

    return {
        "probe_correct": defaultdict(int),
        "probe_total": defaultdict(int),
        "probe_match_final": defaultdict(int),
        "probe_match_total": defaultdict(int),
        "parseable_examples": 0,
        "final_correct": 0,
    }


def update_running_progress_stats(stats: dict[str, Any], result: PrefixBranchResult) -> None:
    if result.generated_steps:
        stats["parseable_examples"] += 1
    if result.final_generated_correct:
        stats["final_correct"] += 1

    for probe in result.probes:
        step = probe.step_index
        stats["probe_total"][step] += 1
        stats["probe_correct"][step] += int(probe.is_correct)
        if probe.matches_final is not None:
            stats["probe_match_total"][step] += 1
            stats["probe_match_final"][step] += int(probe.matches_final)


def snapshot_running_progress(stats: dict[str, Any], n_seen: int) -> dict[str, float]:
    step1_total = stats["probe_total"].get(1, 0)
    step1_match_total = stats["probe_match_total"].get(1, 0)
    return {
        "step1_accuracy": (
            stats["probe_correct"].get(1, 0) / step1_total if step1_total else 0.0
        ),
        "step1_match_final_rate": (
            stats["probe_match_final"].get(1, 0) / step1_match_total if step1_match_total else 0.0
        ),
        "full_generation_accuracy": stats["final_correct"] / max(n_seen, 1),
        "parseable_fraction": stats["parseable_examples"] / max(n_seen, 1),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live prefix-branch commitment probe on TPU")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--study-name", default="prefix_branch_probe")
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--max-probe-steps", type=int, default=4)
    parser.add_argument("--max-generation-tokens", type=int, default=256)
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=10,
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
        print(
            f"[startup] Ready. Loaded {len(examples)} examples for live prefix-branch probing.",
            flush=True,
        )

    results: list[PrefixBranchResult] = []
    progress_stats = init_running_progress_stats()
    t0 = time.time()
    checkpoint_path = Path((args.checkpoint_path or (args.output + ".checkpoint.json"))).expanduser()
    for idx, example in enumerate(examples, start=1):
        result = run_prefix_branch_example(
            example=example,
            model=model,
            tokenizer=tokenizer,
            max_probe_steps=args.max_probe_steps,
            max_generation_tokens=args.max_generation_tokens,
            max_answer_tokens=args.max_answer_tokens,
        )
        results.append(result)
        update_running_progress_stats(progress_stats, result)

        if process_index == 0 and (idx % args.progress_every == 0 or idx == len(examples)):
            agg = snapshot_running_progress(progress_stats, idx)
            elapsed = time.time() - t0
            print(
                f"[progress] {idx}/{len(examples)} | "
                f"step1_acc={agg['step1_accuracy']:.3f} "
                f"step1_match_final={agg['step1_match_final_rate']:.3f} "
                f"full_gen_acc={agg['full_generation_accuracy']:.3f} "
                f"parseable={agg['parseable_fraction']:.3f} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

        if process_index == 0 and args.checkpoint_every > 0 and (idx % args.checkpoint_every == 0):
            aggregated = aggregate_prefix_branch(results)
            payload = {
                "study_name": args.study_name,
                "model_id": args.model_id,
                "mode": "prefix_branch_generation",
                "n_examples": len(results),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "aggregated": aggregated,
                "per_example": [
                    {
                        "example_id": result.example_id,
                        "correct_answer": result.correct_answer,
                        "generated_chain": result.generated_chain,
                        "generated_steps": result.generated_steps,
                        "n_generated_steps": len(result.generated_steps),
                        "final_generated_answer": result.final_generated_answer,
                        "final_generated_parseable": result.final_generated_parseable,
                        "final_generated_correct": result.final_generated_correct,
                        "probes": [
                            {
                                "step_index": probe.step_index,
                                "prefix_preview": probe.prefix_preview,
                                "predicted_answer": probe.predicted_answer,
                                "raw_output": probe.raw_output,
                                "is_correct": probe.is_correct,
                                "matches_final": probe.matches_final,
                            }
                            for probe in result.probes
                        ],
                    }
                    for result in results
                ],
            }
            _write_json_atomic(checkpoint_path, payload)
            print(f"[checkpoint] wrote {len(results)}/{len(examples)} to {checkpoint_path}", flush=True)

    if process_index == 0:
        aggregated = aggregate_prefix_branch(results)
        payload = {
            "study_name": args.study_name,
            "model_id": args.model_id,
            "mode": "prefix_branch_generation",
            "n_examples": len(results),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "aggregated": aggregated,
            "per_example": [
                {
                    "example_id": result.example_id,
                    "correct_answer": result.correct_answer,
                    "generated_chain": result.generated_chain,
                    "generated_steps": result.generated_steps,
                    "n_generated_steps": len(result.generated_steps),
                    "final_generated_answer": result.final_generated_answer,
                    "final_generated_parseable": result.final_generated_parseable,
                    "final_generated_correct": result.final_generated_correct,
                    "probes": [
                        {
                            "step_index": probe.step_index,
                            "prefix_preview": probe.prefix_preview,
                            "predicted_answer": probe.predicted_answer,
                            "raw_output": probe.raw_output,
                            "is_correct": probe.is_correct,
                            "matches_final": probe.matches_final,
                        }
                        for probe in result.probes
                    ],
                }
                for result in results
            ],
        }
        output_path = Path(args.output).expanduser()
        _write_json_atomic(output_path, payload)
        print(f"[done] Results saved to {output_path}", flush=True)
        print(
            f"[summary] step1_acc={aggregated['step1_accuracy']:.3f} "
            f"step1_match_final={aggregated['step1_match_final_rate']:.3f} "
            f"full_generation_accuracy={aggregated['full_generation_accuracy']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()