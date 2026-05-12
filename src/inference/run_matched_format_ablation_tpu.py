#!/usr/bin/env python3
"""Qwen-2.5-7B N=300 matched format ablation with self-matched stripped generation.

Loads standard GSM8K-v1 data, generates neutral-stripped versions by removing
the terminal answer statement from the last step, then runs position-corruption
sweeps on both formats. Every example is its own match — no ID alignment needed.
"""
from __future__ import annotations

import jax
jax.distributed.initialize()

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any


def strip_answer_from_last_step(steps: list[str]) -> list[str]:
    """Remove terminal answer statement from the last step of the chain."""
    if not steps:
        return steps
    result = list(steps)
    last = result[-1]
    # Patterns: "The answer is X", "Therefore, the answer is X.", "So the answer is X"
    last = re.sub(r',?\s*(the|so|therefore[,]?)\s+answer\s+is\s+\$?\s*-?\d[\d,]*\.?\s*\.?\s*$', '', last, flags=re.IGNORECASE)
    last = re.sub(r'[Tt]he\s+answer\s+is\s+\$?\s*-?\d[\d,]*\.?\s*\.?$', '', last)
    if not last.strip():
        # If stripping makes the step empty, remove it
        result = result[:-1]
    else:
        result[-1] = last.strip()
    # Add neutral marker
    result.append("Based on the calculation above:")
    return result


def load_model(model_id: str, sharding: tuple[int, ...]):
    import jax.numpy as jnp
    from jax import lax
    import easydel as ed
    from transformers import AutoTokenizer

    local_only = os.environ.get("HF_LOCAL_ONLY", "1") == "1"

    print(f"[init] process={jax.process_index()}/{jax.device_count()} "
          f"model={model_id} sharding={sharding} local_only={local_only}", flush=True)

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

    try:
        import easydel.infra.modeling_outputs as _mo
        _orig = _mo._is_array
        def _patched(arr):
            if isinstance(arr, jax.Array):
                return True
            return _orig(arr)
        _mo._is_array = _patched
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[init] Model loaded.", flush=True)
    return model, tokenizer


def extract_answer(text: str) -> str | None:
    """Extract numeric answer from model output."""
    patterns = [
        re.compile(r"(?:answer|result|total)[\s\S]*?[:\s]+\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)"),
        re.compile(r"\b(-?\d[\d,]*(?:\.\d+)?)\s*\.?\s*$"),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(1).replace(",", "")
    return None


def answers_match(a: str, b: str) -> bool:
    try:
        return abs(float(str(a).replace(",", "")) - float(str(b).replace(",", ""))) < 1e-6
    except (ValueError, TypeError):
        return str(a).strip() == str(b).strip()


def semantic_corrupt(step: str, rng: random.Random) -> str:
    """Subtle semantic corruption: swap operators, perturb numbers."""
    step = re.sub(r'\bplus\b', 'minus', step, flags=re.IGNORECASE)
    step = re.sub(r'\bminus\b', 'plus', step, flags=re.IGNORECASE)
    step = step.replace('+', '@PLUS@').replace('-', '@MINUS@')
    step = step.replace('@PLUS@', '-').replace('@MINUS@', '+')
    step = re.sub(r'\b(\d+)\b', lambda m: str(int(m.group(1)) + rng.choice([-1, 1])), step)
    return step


def corrupt_steps(steps: list[str], target: str, rng: random.Random) -> list[str]:
    """Corrupt steps in target region (prefix/middle/suffix)."""
    n = len(steps)
    if target == "prefix":
        indices = list(range(max(0, n - 4), n))
    elif target == "middle":
        mid = n // 2
        indices = list(range(max(1, mid - 1), min(n - 1, mid + 2)))
    else:  # suffix
        indices = list(range(max(0, n - 2), n))

    result = list(steps)
    for i in indices:
        if i < n:
            result[i] = semantic_corrupt(result[i], rng)
    return result


def run_one(model, tokenizer, prompt: str, max_new: int = 16) -> str:
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    encoded = tokenizer(prompt, return_tensors="np", truncation=True, max_length=1024)
    ids = jnp.array(encoded["input_ids"])
    seq_len = int(ids.shape[1])

    pad_id = tokenizer.pad_token_id or 0
    buf = jnp.full((1, seq_len + max_new), pad_id, dtype=jnp.int32)
    mask = jnp.zeros((1, seq_len + max_new), dtype=jnp.int32)
    buf = buf.at[0, :seq_len].set(ids[0, :seq_len])
    mask_in = mask.at[0, :seq_len].set(1)
    buf_len = seq_len + max_new

    mesh = model.config.mesh
    rep = NamedSharding(mesh, P())
    buf = jax.device_put(buf, rep)
    mask = jax.device_put(mask_in, rep)

    eos_id = tokenizer.eos_token_id
    stop_ids = {int(eos_id)} if eos_id else set()

    for pos in range(seq_len, buf_len):
        with mesh:
            outputs = model(buf, attention_mask=mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_id = int(jnp.argmax(logits[0, pos - 1, :]))
        if next_id in stop_ids:
            break
        buf = buf.at[0, pos].set(next_id)
        mask = mask.at[0, pos].set(1)

    return tokenizer.decode(buf[0, seq_len:].tolist(), skip_special_tokens=True).strip()


def build_prompt(question: str, steps: list[str]) -> str:
    chain = "\n".join(steps)
    return (
        "Solve the problem step by step.\n"
        f"Question: {question}\n"
        f"{chain}\n"
        "Answer:"
    )


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-examples", type=int, default=300)
    p.add_argument("--sharding", default="1,-1,1,1,1")
    p.add_argument("--progress-every", type=int, default=20)
    args = p.parse_args()

    sharding = tuple(int(x) for x in args.sharding.split(","))
    model, tokenizer = load_model(args.model_id, sharding)

    with open(args.data_file) as f:
        records = [json.loads(l) for l in f if l.strip()]
    records = records[:args.n_examples]
    print(f"[startup] {len(records)} examples", flush=True)

    rng = random.Random(42)

    targets = ["prefix", "middle", "suffix"]
    conditions = ["baseline"] + targets

    # Results: per-example condition -> dict
    all_results = []
    t0 = time.time()

    for i, rec in enumerate(records):
        q = rec["question"]
        answer = str(rec.get("answer", rec.get("correct_answer", "")))
        std_steps = [str(s) for s in rec["steps"]]
        str_steps = strip_answer_from_last_step(std_steps)

        row = {"id": rec.get("id", str(i)), "answer": answer}

        for fmt_name, steps in [("standard", std_steps), ("stripped", str_steps)]:
            # Baseline
            prompt_base = build_prompt(q, steps)
            raw = run_one(model, tokenizer, prompt_base)
            ext = extract_answer(raw) or ""
            row[f"{fmt_name}_baseline_raw"] = raw[:200]
            row[f"{fmt_name}_baseline"] = answers_match(ext, answer)

            # Corruption conditions
            for tgt in targets:
                corr_steps = corrupt_steps(steps, tgt, rng)
                prompt_corr = build_prompt(q, corr_steps)
                raw_c = run_one(model, tokenizer, prompt_corr)
                ext_c = extract_answer(raw_c) or ""
                row[f"{fmt_name}_{tgt}"] = answers_match(ext_c, answer)

        all_results.append(row)

        if (i + 1) % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(records) - i - 1) / rate if rate > 0 else 0
            # Quick summary
            n = len(all_results)
            std_corr = sum(1 for r in all_results[-20:] if r["standard_baseline"])
            m = min(20, n)
            print(f"[progress] {i+1}/{len(records)} rate={rate:.3f}/s eta={eta:.0f}s "
                  f"std_baseline_acc={std_corr/m:.2f}", flush=True)

    # Aggregate
    def acc(key):
        vals = [r[key] for r in all_results if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    def delta(fmt, tgt):
        return acc(f"{fmt}_{tgt}") - acc(f"{fmt}_baseline")

    summary = {
        "n": len(all_results),
        "standard": {
            "baseline": acc("standard_baseline"),
            **{t: acc(f"standard_{t}") for t in targets},
            **{f"delta_{t}": delta("standard", t) for t in targets},
        },
        "stripped": {
            "baseline": acc("stripped_baseline"),
            **{t: acc(f"stripped_{t}") for t in targets},
            **{f"delta_{t}": delta("stripped", t) for t in targets},
        },
    }

    if jax.process_index() == 0:
        elapsed = time.time() - t0
        out = {"model_id": args.model_id, "n": len(all_results),
               "elapsed_s": round(elapsed, 1), "summary": summary}
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "qwen7b_n300_matched_ablation.json"
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("w") as fh:
            json.dump(out, fh, indent=2)
        tmp.replace(out_path)
        print(f"\n[save] {out_path} elapsed={elapsed:.0f}s", flush=True)
        print(f"[results] std baseline={summary['standard']['baseline']:.3f} "
              f"suffix_delta={summary['standard']['delta_suffix']:+.3f}", flush=True)
        print(f"[results] str baseline={summary['stripped']['baseline']:.3f} "
              f"suffix_delta={summary['stripped']['delta_suffix']:+.3f}", flush=True)
        print(f"[results] attenuation={abs(summary['standard']['delta_suffix'])/(abs(summary['stripped']['delta_suffix'])+1e-9):.1f}x", flush=True)

    print("[DONE] Matched format ablation complete.", flush=True)


if __name__ == "__main__":
    main()
