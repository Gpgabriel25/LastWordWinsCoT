#!/usr/bin/env python3
"""Generate publication-quality figures for the CoT rationalization paper.

Usage:
    python3 scripts/generate_figures.py --results-dir results_fixed/ --output-dir paper/figures/

Generates:
  - fig_reversal.pdf: Main reversal bar chart (Figure 1)
  - fig_cross_model.pdf: Cross-model comparison
  - fig_format_ablation.pdf: Format ablation (if gsm8k_stripped results present)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_result(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_accuracies(result: dict) -> dict[str, float]:
    """Return {baseline, prefix, middle, suffix} accuracies from a result file."""
    s = result["summary"]
    out = {"baseline": s["baseline_accuracy"]}
    if isinstance(s["corrupted_accuracy"], dict):
        out.update(s["corrupted_accuracy"])
    return out


def extract_qonly(result: dict) -> float:
    """Return question-only baseline accuracy."""
    return result["summary"]["baseline_accuracy"]


MODEL_LABELS = {
    "qwen3b": "Qwen 2.5-3B",
    "phi3mini": "Phi-3-mini",
    "qwen7b": "Qwen 2.5-7B",
}

SLICE_LABELS = {
    "hard_v3": "Hard-v3\n(no answer in suffix)",
    "gsm8k_v1": "GSM8K-v1\n(answer in suffix)",
    "gsm8k_stripped": "GSM8K-stripped\n(answer removed)",
    "gsm8k_stripped_v1": "GSM8K-stripped\n(answer removed)",
    "gsm8k_v2": "GSM8K-v2",
}


# ---------------------------------------------------------------------------
# Figure 1: The reversal
# ---------------------------------------------------------------------------

def fig_reversal(results_dir: Path, output_dir: Path):
    """Side-by-side bar chart showing the format-sensitivity reversal."""
    # We need hard_v3.qwen3b_chain and gsm8k_v1.qwen3b_chain
    hard_path = results_dir / "hard_v3.qwen3b_chain.json"
    gsm8k_path = results_dir / "gsm8k_v1.qwen3b_chain.json"

    if not hard_path.exists() or not gsm8k_path.exists():
        print(f"[SKIP] fig_reversal: need {hard_path.name} and {gsm8k_path.name}")
        return

    hard = extract_accuracies(load_result(hard_path))
    gsm8k = extract_accuracies(load_result(gsm8k_path))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)

    positions = ["baseline", "prefix", "middle", "suffix"]
    labels = ["Baseline", "Prefix", "Middle", "Suffix"]
    x = np.arange(len(positions))
    width = 0.6

    # Hard-v3 (prefix is the killer)
    hard_vals = [hard.get(p, 0) for p in positions]
    colors_hard = ["#4C72B0" if p != "prefix" else "#C44E52" for p in positions]
    bars1 = ax1.bar(x, hard_vals, width, color=colors_hard, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.set_title("Hard-v3: no answer in suffix", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=hard["baseline"], color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    # Add value labels
    for bar, val in zip(bars1, hard_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # GSM8K-v1 (suffix is the killer)
    gsm8k_vals = [gsm8k.get(p, 0) for p in positions]
    colors_gsm8k = ["#4C72B0" if p != "suffix" else "#C44E52" for p in positions]
    bars2 = ax2.bar(x, gsm8k_vals, width, color=colors_gsm8k, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("GSM8K-v1: answer in suffix", fontsize=10, fontweight="bold")
    ax2.axhline(y=gsm8k["baseline"], color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    for bar, val in zip(bars2, gsm8k_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#4C72B0", label="Intact / non-critical"),
        mpatches.Patch(facecolor="#C44E52", label="Catastrophic drop"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
              fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    # Caption box
    fig.text(0.5, -0.10,
             "Same model, same protocol, opposite conclusion.\n"
             "The load-bearing position tracks answer placement, not computation.",
             ha="center", va="top", fontsize=9, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.8))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = output_dir / "fig_reversal.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: Cross-model comparison
# ---------------------------------------------------------------------------

def fig_cross_model(results_dir: Path, output_dir: Path):
    """Grouped bar chart showing reversal holds across models."""
    models = ["qwen3b", "phi3mini"]
    slices = ["hard_v3", "gsm8k_v1"]

    data = {}
    for model in models:
        for sl in slices:
            chain_path = results_dir / f"{sl}.{model}_chain.json"
            qonly_path = results_dir / f"{sl}.{model}_qonly.json"
            if chain_path.exists():
                data[(sl, model)] = extract_accuracies(load_result(chain_path))
            if qonly_path.exists():
                data[(sl, model, "qonly")] = extract_qonly(load_result(qonly_path))

    if len(data) < 4:
        print(f"[SKIP] fig_cross_model: need all 4 chain results, found {len([k for k in data if len(k)==2])}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=True)

    targets = ["prefix", "middle", "suffix"]
    x = np.arange(len(targets))
    width = 0.35
    colors = ["#4C72B0", "#55A868"]

    for i, sl in enumerate(slices):
        ax = axes[i]
        for j, model in enumerate(models):
            key = (sl, model)
            if key not in data:
                continue
            accs = data[key]
            vals = [accs.get(t, 0) for t in targets]
            baseline = accs["baseline"]
            # Compute delta from baseline
            deltas = [v - baseline for v in vals]
            bars = ax.bar(x + j * width - width/2, deltas, width,
                         label=MODEL_LABELS.get(model, model),
                         color=colors[j], edgecolor="white", linewidth=0.5)
            for bar, delta in zip(bars, deltas):
                label_y = bar.get_height() if delta >= 0 else bar.get_height() - 0.03
                va = "bottom" if delta >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width()/2, label_y + 0.01 * (1 if delta >= 0 else -1),
                       f"{delta:+.3f}", ha="center", va=va, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in targets], fontsize=9)
        ax.set_title(SLICE_LABELS.get(sl, sl), fontsize=10, fontweight="bold")
        ax.axhline(y=0, color="black", linewidth=0.5)
        if i == 0:
            ax.set_ylabel("$\\Delta$ Accuracy from baseline", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="lower left" if sl == "hard_v3" else "upper right")

    fig.suptitle("Cross-Model Corruption Sensitivity", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = output_dir / "fig_cross_model.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: Format ablation (GSM8K-stripped)
# ---------------------------------------------------------------------------

def fig_format_ablation(results_dir: Path, output_dir: Path):
    """Bar chart comparing GSM8K-v1 vs GSM8K-stripped suffix sensitivity,
    now including Qwen 2.5-7B original and neutral-stripped arms."""

    # --- Load 3B and Phi-3-mini data (existing arms) ---
    models_3b = ["qwen3b", "phi3mini"]
    data_old = {}
    for model in models_3b:
        for variant in ["gsm8k_v1", "gsm8k_stripped_v1", "gsm8k_stripped"]:
            chain_path = results_dir / f"{variant}.{model}_chain.json"
            if chain_path.exists():
                accs = extract_accuracies(load_result(chain_path))
                key = "gsm8k_stripped" if "stripped" in variant else variant
                data_old[(key, model)] = accs

    # --- Load 7B arms ---
    path_7b_orig = results_dir.parent / "logs" / "pillar1_qwen7b_gsm8k_1000.json"
    # Prefer Phase C (matched instruction) over Phase B / old file
    path_7b_neutral_phaseC = results_dir / "phaseC_neutral_stripped_matched.json"
    path_7b_neutral_old = results_dir / "gsm8k_neutral_stripped.qwen7b_chain.json"
    if path_7b_neutral_phaseC.exists():
        path_7b_neutral = path_7b_neutral_phaseC
        print("[fig_format_ablation] Using Phase C neutral-stripped data (matched instruction)")
    else:
        path_7b_neutral = path_7b_neutral_old

    data_7b_orig = extract_accuracies(load_result(path_7b_orig)) if path_7b_orig.exists() else None
    data_7b_neutral = extract_accuracies(load_result(path_7b_neutral)) if path_7b_neutral.exists() else None

    if ("gsm8k_v1", "qwen3b") not in data_old and data_7b_orig is None:
        print("[SKIP] fig_format_ablation: no usable results found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    # ---- Left panel: 3B and Phi-3-mini ----
    ax = axes[0]
    groups, tick_labels, colors_list = [], [], []
    for model in models_3b:
        for variant in ["gsm8k_v1", "gsm8k_stripped"]:
            key = (variant, model)
            if key in data_old:
                accs = data_old[key]
                suffix_drop = accs["baseline"] - accs.get("suffix", accs["baseline"])
                groups.append(suffix_drop)
                mlabel = MODEL_LABELS.get(model, model)
                vlabel = "Original" if variant == "gsm8k_v1" else "Stripped\n(verify)"
                tick_labels.append(f"{mlabel}\n{vlabel}")
                colors_list.append("#C44E52" if variant == "gsm8k_v1" else "#55A868")

    x = np.arange(len(groups))
    bars = ax.bar(x, groups, 0.5, color=colors_list, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, groups):
        va = "bottom" if val >= 0 else "top"
        offset = 0.01 if val >= 0 else -0.01
        ax.text(bar.get_x() + bar.get_width()/2, val + offset,
                f"{val:+.3f}", ha="center", va=va, fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel(r"Suffix accuracy drop (baseline $-$ corrupted)", fontsize=9)
    ax.set_title("3B & Phi-3-mini: Format Ablation", fontsize=10, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.7)
    ax.set_ylim(-0.25, 0.90)

    # ---- Right panel: Qwen 2.5-7B original vs neutral-stripped ----
    ax2 = axes[1]
    groups2, tick_labels2, colors2 = [], [], []
    if data_7b_orig is not None:
        drop_orig = data_7b_orig["baseline"] - data_7b_orig.get("suffix", data_7b_orig["baseline"])
        groups2.append(drop_orig)
        tick_labels2.append("Qwen 2.5-7B\nOriginal\n($N{=}1000$)")
        colors2.append("#C44E52")
    if data_7b_neutral is not None:
        drop_neutral = data_7b_neutral["baseline"] - data_7b_neutral.get("suffix", data_7b_neutral["baseline"])
        groups2.append(drop_neutral)
        neutral_label = "Qwen 2.5-7B\nNeutral-stripped\n(Phase C, $N{=}100$)" if path_7b_neutral_phaseC.exists() else "Qwen 2.5-7B\nNeutral-stripped\n($N{=}100$)"
        tick_labels2.append(neutral_label)
        colors2.append("#4C72B0")

    if groups2:
        x2 = np.arange(len(groups2))
        bars2 = ax2.bar(x2, groups2, 0.45, color=colors2, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars2, groups2):
            va = "bottom" if val >= 0 else "top"
            offset = 0.01 if val >= 0 else -0.01
            ax2.text(bar.get_x() + bar.get_width()/2, val + offset,
                     f"{val:+.3f}", ha="center", va=va, fontsize=9)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(tick_labels2, fontsize=8.5)

        # Annotate sign inversion
        if len(groups2) == 2:
            ax2.annotate("sign\ninversion", xy=(0.5, 0), xytext=(0.5, 0.28),
                         ha="center", fontsize=8, color="#333333",
                         arrowprops=dict(arrowstyle="<->", color="#666666", lw=1.4))

        ax2.axhline(y=0, color="black", linewidth=0.7)
        ax2.set_ylabel(r"Suffix accuracy drop (baseline $-$ corrupted)", fontsize=9)
        ax2.set_title("Qwen 2.5-7B: Sign Inversion ($p{=}0.013$)", fontsize=10, fontweight="bold")
        ax2.set_ylim(-0.25, 0.70)

    legend_elements = [
        mpatches.Patch(facecolor="#C44E52", label='Original format ("the answer is X")'),
        mpatches.Patch(facecolor="#55A868", label="Stripped + verify placeholder"),
        mpatches.Patch(facecolor="#4C72B0", label="Stripped + neutral placeholder"),
    ]
    fig.legend(handles=legend_elements, fontsize=8, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = output_dir / "fig_format_ablation.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 4: Question-only control comparison
# ---------------------------------------------------------------------------

def fig_qonly_control(results_dir: Path, output_dir: Path):
    """Bar chart showing question-only vs chain baseline accuracy."""
    models = ["qwen3b", "phi3mini"]
    slices = ["hard_v3", "gsm8k_v1"]

    data = []
    for sl in slices:
        for model in models:
            chain_path = results_dir / f"{sl}.{model}_chain.json"
            qonly_path = results_dir / f"{sl}.{model}_qonly.json"
            if chain_path.exists() and qonly_path.exists():
                chain_acc = extract_accuracies(load_result(chain_path))["baseline"]
                qonly_acc = extract_qonly(load_result(qonly_path))
                data.append({
                    "slice": sl, "model": model,
                    "chain": chain_acc, "qonly": qonly_acc,
                })

    if len(data) < 4:
        print(f"[SKIP] fig_qonly_control: found {len(data)}/4 pairs")
        return

    fig, ax = plt.subplots(figsize=(6, 3.2))

    x = np.arange(len(data))
    width = 0.35

    chain_vals = [d["chain"] for d in data]
    qonly_vals = [d["qonly"] for d in data]

    bars1 = ax.bar(x - width/2, chain_vals, width, label="Chain baseline",
                   color="#4C72B0", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, qonly_vals, width, label="Question-only",
                   color="#DD8452", edgecolor="white", linewidth=0.5)

    for bar, val in zip(list(bars1) + list(bars2), chain_vals + qonly_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    labels = [f"{SLICE_LABELS.get(d['slice'], d['slice']).split(chr(10))[0]}\n{MODEL_LABELS.get(d['model'], d['model'])}"
              for d in data]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("Question-Only Controls", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out_path = output_dir / "fig_qonly_control.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", required=True, help="Directory with campaign result JSONs")
    parser.add_argument("--output-dir", default="paper/figures/", help="Output directory for figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results: {results_dir}")
    print(f"Output:  {output_dir}")
    print(f"Available results: {sorted(f.name for f in results_dir.glob('*.json'))}")
    print()

    generated = []
    for gen_fn in [fig_reversal, fig_cross_model, fig_format_ablation, fig_qonly_control]:
        result = gen_fn(results_dir, output_dir)
        if result:
            generated.append(result)

    print(f"\nGenerated {len(generated)} figures.")


if __name__ == "__main__":
    main()
