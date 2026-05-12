#!/usr/bin/env python3
"""Plot protocol-uniform FW/QO conditioning summary across scale points.

This script reads completed conflicting-answer artifacts and produces a compact
figure for the manuscript scale section.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class Point:
    label: str
    path: Path
    fw: float
    qo: float
    sc: float
    cc: float
    n: int


def wilson_interval(phat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (phat, phat)
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * ((phat * (1.0 - phat) / n + (z * z) / (4.0 * n * n)) ** 0.5)
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


def load_point(label: str, path: Path) -> Point:
    data = json.loads(path.read_text())
    cond = data["summary"]["conditions"]
    fw = float(cond["conflicting_chain"]["followed_wrong_suffix_rate"])
    qo = float(cond["question_only"]["accuracy"])
    sc = float(cond["standard_chain"]["accuracy"])
    cc = float(cond["conflicting_chain"]["accuracy"])
    n = int(cond["conflicting_chain"]["n"])
    return Point(label=label, path=path, fw=fw, qo=qo, sc=sc, cc=cc, n=n)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    results = repo / "results_fixed"
    out = repo / "paper" / "figures" / "fig_scale_conditioning_summary.pdf"

    spec = [
        ("Qwen 7B", results / "conflicting_gsm8k_qwen7b_v1_n100_fixed.json"),
        ("Qwen 14B", results / "conflicting_gsm8k_qwen14b_v1_n100.json"),
        ("Phi-4 14B", results / "conflicting_gsm8k_phi4_n200.json"),
        ("Qwen 32B", results / "conflicting_gsm8k_qwen32b_n100.json"),
        (
            "DeepSeek-R1-7B",
            results / "conflicting_gsm8k_deepseek_r1_distill_7b_n200.json",
        ),
    ]

    points = [load_point(label, path) for label, path in spec]

    labels = [p.label for p in points]
    fw = [p.fw for p in points]
    qo = [p.qo for p in points]
    gap = [p.fw - p.qo for p in points]

    fw_err = []
    qo_err = []
    for p in points:
        fw_lo, fw_hi = wilson_interval(p.fw, p.n)
        qo_lo, qo_hi = wilson_interval(p.qo, p.n)
        fw_err.append((p.fw - fw_lo, fw_hi - p.fw))
        qo_err.append((p.qo - qo_lo, qo_hi - p.qo))

    x = list(range(len(points)))

    plt.rcParams.update({"font.size": 10})
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.8, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.4]})

    width = 0.34
    ax0.bar([i - width / 2 for i in x], fw, width=width, label="FW (followed-wrong)", color="#c0392b")
    ax0.bar([i + width / 2 for i in x], qo, width=width, label="QO (question-only)", color="#2c7fb8")

    for i, p in enumerate(points):
        ax0.errorbar(
            i - width / 2,
            p.fw,
            yerr=[[fw_err[i][0]], [fw_err[i][1]]],
            fmt="none",
            ecolor="black",
            capsize=3,
            lw=0.8,
        )
        ax0.errorbar(
            i + width / 2,
            p.qo,
            yerr=[[qo_err[i][0]], [qo_err[i][1]]],
            fmt="none",
            ecolor="black",
            capsize=3,
            lw=0.8,
        )

    ax0.set_ylim(0.0, 1.05)
    ax0.set_ylabel("Rate")
    ax0.set_title("Protocol-uniform conditioning check across scale points (GSM8K-v1, n=100 each)")
    ax0.legend(loc="upper right", frameon=False)
    ax0.grid(axis="y", alpha=0.2)

    colors = ["#8e44ad" if g >= 0 else "#7f8c8d" for g in gap]
    ax1.bar(x, gap, color=colors)
    ax1.axhline(0.0, color="black", lw=0.9)
    ax1.set_ylabel("FW - QO")
    ax1.set_ylim(-0.4, 1.05)
    ax1.grid(axis="y", alpha=0.2)

    for i, g in enumerate(gap):
        ax1.text(i, g + (0.03 if g >= 0 else -0.05), f"{g:+.2f}", ha="center", va="bottom" if g >= 0 else "top", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=10)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
