#!/usr/bin/env python3
"""Integrate Qwen3 campaign results into paper/main.tex.

Reads the result JSONs from results_fixed/, computes significance tests,
generates new LaTeX macros, and produces sed commands to update the paper.

Usage:
    python3 scripts/integrate_qwen3_results.py --results-dir results_fixed
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def fisher_exact_p(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact test p-value (a*d < b*c direction)."""
    n = a + b + c + d
    def log_factorial(x):
        return sum(math.log(i) for i in range(1, x + 1))
    
    row1 = a + b; row2 = c + d; col1 = a + c; col2 = b + d
    p_cutoff = (log_factorial(row1) + log_factorial(row2) + 
                log_factorial(col1) + log_factorial(col2) - log_factorial(n))
    
    p_val = 0.0
    for x in range(min(row1, col1) + 1):
        y = row1 - x; z = col1 - x; w = row2 - z
        if y < 0 or z < 0 or w < 0:
            continue
        log_p = (log_factorial(row1) + log_factorial(row2) + 
                 log_factorial(col1) + log_factorial(col2) - 
                 log_factorial(n) - log_factorial(x) - 
                 log_factorial(y) - log_factorial(z) - log_factorial(w))
        if log_p <= p_cutoff + 1e-10:
            p_val += math.exp(log_p - p_cutoff) * math.exp(p_cutoff - p_cutoff)
    # Normalize
    total = 0.0
    for x in range(min(row1, col1) + 1):
        y = row1 - x; z = col1 - x; w = row2 - z
        if y < 0 or z < 0 or w < 0:
            continue
        log_p = (log_factorial(row1) + log_factorial(row2) + 
                 log_factorial(col1) + log_factorial(col2) - 
                 log_factorial(n) - log_factorial(x) - 
                 log_factorial(y) - log_factorial(z) - log_factorial(w))
        total += math.exp(log_p)
    
    p_sum = 0.0
    for x in range(min(row1, col1) + 1):
        y = row1 - x; z = col1 - x; w = row2 - z
        if y < 0 or z < 0 or w < 0:
            continue
        log_p = (log_factorial(row1) + log_factorial(row2) + 
                 log_factorial(col1) + log_factorial(col2) - 
                 log_factorial(n) - log_factorial(x) - 
                 log_factorial(y) - log_factorial(z) - log_factorial(w))
        this_p = math.exp(log_p)
        ref_p = (log_factorial(row1) + log_factorial(row2) + 
                 log_factorial(col1) + log_factorial(col2) - 
                 log_factorial(n) - log_factorial(a) - 
                 log_factorial(b) - log_factorial(c) - log_factorial(d))
        if log_p <= ref_p + 1e-10:
            p_sum += this_p
    return min(p_sum, 1.0)


def format_p(p: float) -> str:
    if p < 1e-10:
        exp = int(math.floor(math.log10(p)))
        return f"$p{{<}}10^{{{exp}}}$"
    elif p < 0.001:
        return f"$p{{<}}0.001$"
    else:
        return f"$p{{=}}{p:.3f}$"


def process_combined(data: dict, prefix: str) -> dict[str, str]:
    """Generate LaTeX macros from combined experiment results."""
    macros = {}
    model = data.get("model", "unknown")
    
    # Standard format ablation
    std = data.get("standard_format_ablation", {})
    macros[f"{prefix}STDN"] = str(std.get("n", "?"))
    macros[f"{prefix}STDBASE"] = f"{std.get('baseline_acc', 0):.3f}"
    macros[f"{prefix}STDQO"] = f"{std.get('qo_acc', 0):.3f}"
    macros[f"{prefix}STDMID"] = f"{std.get('middle_acc', 0):.3f}"
    macros[f"{prefix}STDPRE"] = f"{std.get('prefix_acc', 0):.3f}"
    macros[f"{prefix}STDSUF"] = f"{std.get('suffix_acc', 0):.3f}"
    
    # Neutral-stripped format ablation
    ns = data.get("neutral_stripped_format_ablation", {})
    macros[f"{prefix}NSN"] = str(ns.get("n", "?"))
    macros[f"{prefix}NSBASE"] = f"{ns.get('baseline_acc', 0):.3f}"
    macros[f"{prefix}NSQO"] = f"{ns.get('qo_acc', 0):.3f}"
    macros[f"{prefix}NSMID"] = f"{ns.get('middle_acc', 0):.3f}"
    macros[f"{prefix}NSPRE"] = f"{ns.get('prefix_acc', 0):.3f}"
    macros[f"{prefix}NSSUF"] = f"{ns.get('suffix_acc', 0):.3f}"
    
    # Conflicting answer
    cc = data.get("conflicting_answer", {})
    macros[f"{prefix}CCN"] = str(cc.get("n", "?"))
    macros[f"{prefix}SCACC"] = f"{cc.get('sc_acc', 0):.3f}"
    macros[f"{prefix}CCACC"] = f"{cc.get('cc_acc', 0):.3f}"
    macros[f"{prefix}FW"] = f"{cc.get('fw', 0):.3f}"
    macros[f"{prefix}QOACC"] = f"{cc.get('qo_acc', 0):.3f}"
    
    # Compute suffix delta for standard format
    if std.get("baseline_acc") is not None and std.get("suffix_acc") is not None:
        delta = std["suffix_acc"] - std["baseline_acc"]
        macros[f"{prefix}STDDELTA"] = f"{delta:+.3f}"
    
    return macros


def process_factorial(data: dict, prefix: str) -> dict[str, str]:
    """Generate LaTeX macros from factorial experiment results."""
    macros = {}
    summary = data.get("summary", data)
    conditions = summary.get("conditions", {})
    contrasts = summary.get("contrasts", {})
    
    for cond_key in ["A", "B", "C", "D"]:
        cond = conditions.get(cond_key, conditions.get(f"condition_{cond_key}", {}))
        acc = cond.get("accuracy", "?")
        if isinstance(acc, (int, float)):
            macros[f"{prefix}F{cond_key}"] = f"{acc:.2f}"
        fw = cond.get("followed_wrong_rate", None)
        if fw is not None and isinstance(fw, (int, float)):
            macros[f"{prefix}F{cond_key}FW"] = f"{fw:.2f}"
    
    if contrasts:
        re = contrasts.get("reasoning_effect_on_accuracy", None)
        if re is not None:
            macros[f"{prefix}FREASON"] = f"{re:+.4f}"
        ca = contrasts.get("C_acc", None)
        if ca is not None:
            macros[f"{prefix}FCCACC"] = f"{ca:.4f}"
    
    macros[f"{prefix}FN"] = str(data.get("n_per_condition", "?"))
    return macros


def parse_cc_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize conflicting-answer summary across flat/nested schemas."""
    summary = data.get("summary", data)
    conds = summary.get("conditions", {})
    sc_cond = conds.get("standard_chain", {})
    cc_cond = conds.get("conflicting_chain", {})
    qo_cond = conds.get("question_only", {})
    out = {
        "n": summary.get("n", summary.get("n_examples", data.get("n", data.get("n_examples", "?")))),
        "sc_acc": sc_cond.get("accuracy", summary.get("sc_acc", data.get("sc_acc"))),
        "cc_acc": cc_cond.get("accuracy", summary.get("cc_acc", data.get("cc_acc"))),
        "fw": cc_cond.get("followed_wrong_suffix_rate", summary.get("fw", data.get("fw", data.get("followed_wrong_rate")))),
        "qo_acc": qo_cond.get("accuracy", summary.get("qo_acc", data.get("qo_acc"))),
    }
    return out


def parse_ablation_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize format-ablation summary across schemas."""
    summary = data.get("summary", data)
    return {
        "n": summary.get("n", summary.get("n_examples", data.get("n", data.get("n_examples", "?")))),
        "baseline_acc": summary.get("baseline_acc", summary.get("baseline_accuracy", data.get("baseline_acc", data.get("baseline_accuracy")))),
        "qo_acc": summary.get("qo_acc", data.get("qo_acc")),
        "middle_acc": summary.get("middle_acc", summary.get("corrupted_accuracy", {}).get("middle", data.get("middle_acc"))),
        "prefix_acc": summary.get("prefix_acc", summary.get("corrupted_accuracy", {}).get("prefix", data.get("prefix_acc"))),
        "suffix_acc": summary.get("suffix_acc", summary.get("corrupted_accuracy", {}).get("suffix", data.get("suffix_acc"))),
    }


def generate_macro_block(macros: dict[str, str], label: str) -> str:
    """Generate a LaTeX macro definition block."""
    lines = [f"%% ========== {label} =========="]
    for name, value in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results_fixed")
    parser.add_argument("--output", default="reports/qwen3_integration.md")
    args = parser.parse_args()
    
    rdir = Path(args.results_dir)
    all_macros = {}
    report_lines = ["# Qwen3 Campaign Results Integration\n"]
    
    # Process Qwen3-8B combined (preferred) or Phase 1 per-file outputs (fallback)
    f8b = rdir / "qwen3_8b_combined_summary.json"
    if f8b.exists():
        data = json.loads(f8b.read_text())
        macros = process_combined(data, "QTEIGHT")
        all_macros.update(macros)
        report_lines.append("## Qwen3-8B Combined Results")
        report_lines.append(f"Model: {data.get('model', '?')}")
        
        std = data.get("standard_format_ablation", {})
        report_lines.append(f"\n### Standard Format Ablation (N={std.get('n', '?')})")
        report_lines.append(f"- Baseline: {std.get('baseline_acc', '?')}")
        report_lines.append(f"- QO: {std.get('qo_acc', '?')}")
        report_lines.append(f"- Mid: {std.get('middle_acc', '?')}")
        report_lines.append(f"- Pre: {std.get('prefix_acc', '?')}")
        report_lines.append(f"- Suf: {std.get('suffix_acc', '?')}")
        
        ns = data.get("neutral_stripped_format_ablation", {})
        report_lines.append(f"\n### Neutral-Stripped Format Ablation (N={ns.get('n', '?')})")
        report_lines.append(f"- Baseline: {ns.get('baseline_acc', '?')}")
        report_lines.append(f"- Suf: {ns.get('suffix_acc', '?')}")
        
        cc = data.get("conflicting_answer", {})
        report_lines.append(f"\n### Conflicting Answer (N={cc.get('n', '?')})")
        report_lines.append(f"- SC: {cc.get('sc_acc', '?')}")
        report_lines.append(f"- CC: {cc.get('cc_acc', '?')}")
        report_lines.append(f"- FW: {cc.get('fw', '?')}")
        report_lines.append(f"- QO: {cc.get('qo_acc', '?')}")
        report_lines.append("")
    else:
        report_lines.append(f"**MISSING**: {f8b}")
        fstd = rdir / "qwen3_8b_standard_format_ablation.json"
        fns = rdir / "qwen3_8b_neutral_stripped_format_ablation.json"
        fcc = rdir / "qwen3_8b_conflicting_answer.json"
        if fstd.exists() or fns.exists() or fcc.exists():
            report_lines.append("\n## Qwen3-8B Phase 1 Fallback Integration")
            if fstd.exists():
                std = parse_ablation_summary(json.loads(fstd.read_text()))
                all_macros["QTEIGHTSCACC"] = f"{(std.get('baseline_acc') or 0):.3f}"
                if std.get("qo_acc") is not None:
                    all_macros["QTEIGHTQOACC"] = f"{std['qo_acc']:.3f}"
                report_lines.append(f"- Standard file: {fstd.name} (N={std.get('n', '?')})")
                report_lines.append(f"  - baseline={std.get('baseline_acc', '?')}, qo={std.get('qo_acc', '?')}")
            else:
                report_lines.append(f"- **MISSING**: {fstd.name}")

            if fns.exists():
                ns = parse_ablation_summary(json.loads(fns.read_text()))
                for macro, key in [
                    ("QTEIGHTNSBASE", "baseline_acc"),
                    ("QTEIGHTNSMID", "middle_acc"),
                    ("QTEIGHTNSPRE", "prefix_acc"),
                    ("QTEIGHTNSSUF", "suffix_acc"),
                ]:
                    val = ns.get(key)
                    if val is not None:
                        all_macros[macro] = f"{val:.3f}"
                report_lines.append(f"- Neutral-stripped file: {fns.name} (N={ns.get('n', '?')})")
                report_lines.append(f"  - base={ns.get('baseline_acc', '?')}, mid={ns.get('middle_acc', '?')}, pre={ns.get('prefix_acc', '?')}, suf={ns.get('suffix_acc', '?')}")
            else:
                report_lines.append(f"- **MISSING**: {fns.name}")

            if fcc.exists():
                cc = parse_cc_summary(json.loads(fcc.read_text()))
                for macro, key in [
                    ("QTEIGHTCCACC", "cc_acc"),
                    ("QTEIGHTFW", "fw"),
                    ("QTEIGHTSCACC", "sc_acc"),
                    ("QTEIGHTQOACC", "qo_acc"),
                ]:
                    val = cc.get(key)
                    if isinstance(val, (int, float)):
                        all_macros[macro] = f"{val:.3f}"
                report_lines.append(f"- Conflicting file: {fcc.name} (N={cc.get('n', '?')})")
                report_lines.append(f"  - sc={cc.get('sc_acc', '?')}, cc={cc.get('cc_acc', '?')}, fw={cc.get('fw', '?')}, qo={cc.get('qo_acc', '?')}")
            else:
                report_lines.append(f"- **MISSING**: {fcc.name}")
            report_lines.append("")
        else:
            report_lines.append("")
    
    # Process Qwen3-8B factorial
    f8bfact = rdir / "factorial_gsm8k_qwen3_8b.json"
    if f8bfact.exists():
        data = json.loads(f8bfact.read_text())
        macros = process_factorial(data, "QTEIGHT")
        all_macros.update(macros)
        report_lines.append("## Qwen3-8B Factorial Results")
        summary = data.get("summary", data)
        conditions = summary.get("conditions", {})
        for k in ["A", "B", "C", "D"]:
            cond = conditions.get(k, conditions.get(f"condition_{k}", {}))
            report_lines.append(f"- Condition {k}: acc={cond.get('accuracy', '?')}, fw={cond.get('followed_wrong_rate', '?')}")
        report_lines.append("")
    else:
        report_lines.append(f"**MISSING**: {f8bfact}\n")
    
    # Process Qwen3-32B combined
    f32b = rdir / "qwen3_32b_combined_summary.json"
    if f32b.exists():
        data = json.loads(f32b.read_text())
        macros = process_combined(data, "QTTHIRTYTWO")
        all_macros.update(macros)
        report_lines.append("## Qwen3-32B Combined Results")
        
        cc = data.get("conflicting_answer", {})
        report_lines.append(f"\n### Conflicting Answer (N={cc.get('n', '?')})")
        report_lines.append(f"- SC: {cc.get('sc_acc', '?')}")
        report_lines.append(f"- CC: {cc.get('cc_acc', '?')}")
        report_lines.append(f"- FW: {cc.get('fw', '?')}")
        report_lines.append(f"- QO: {cc.get('qo_acc', '?')}")
        report_lines.append("")
    else:
        report_lines.append(f"**MISSING**: {f32b}\n")
    
    # Generate LaTeX macro block
    report_lines.append("## Generated LaTeX Macros\n")
    report_lines.append("```latex")
    if any(k.startswith("QTEIGHT") for k in all_macros):
        block = generate_macro_block(
            {k: v for k, v in all_macros.items() if k.startswith("QTEIGHT")},
            "QWEN3-8B EXPERIMENT MACROS"
        )
        report_lines.append(block)
    if any(k.startswith("QTTHIRTYTWO") for k in all_macros):
        block = generate_macro_block(
            {k: v for k, v in all_macros.items() if k.startswith("QTTHIRTYTWO")},
            "QWEN3-32B EXPERIMENT MACROS"
        )
        report_lines.append(block)
    report_lines.append("```\n")
    
    # Write report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_lines))
    print(f"Report written to {out_path}")
    print(f"Total macros generated: {len(all_macros)}")
    
    # Print summary
    if all_macros:
        print("\n=== Key Results ===")
        for key in sorted(all_macros):
            if any(x in key for x in ["SCACC", "CCACC", "FW", "QOACC", "STDSUF", "NSSU"]):
                print(f"  {key} = {all_macros[key]}")


if __name__ == "__main__":
    main()
