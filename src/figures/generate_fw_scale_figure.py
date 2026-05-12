#!/usr/bin/env python3
"""Figure: followed-wrong rate across model families and scales.

Shows the answer-text override mechanism at the behavioral level:
at 3B-7B, FW ≈ 0.6-1.0 across all families; by 14B-32B, it attenuates.
This directly visualizes the scale-dependent dissociation between
override and format-determination.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data: (label, params_billions, fw, is_instruct, family)
data = [
    ("Qwen-2.5-3B",      3.0,  0.680,  True,  "Qwen-2.5"),
    ("Qwen-2.5-7B",      7.6,  0.590,  True,  "Qwen-2.5"),
    ("Qwen-2.5-14B",    14.7,  0.060,  True,  "Qwen-2.5"),
    ("Qwen-2.5-32B",    32.0,  0.010,  True,  "Qwen-2.5"),
    ("Qwen-7B-Base",     7.6,  0.900,  False, "Qwen-base"),
    ("Phi-3-mini",       3.8,  0.470,  True,  "Phi"),
    ("Phi-4",           14.0,  0.300,  True,  "Phi"),
    ("Mistral-7B",       7.2,  0.980,  True,  "Mistral"),
    ("DS-R1-Distill-7B", 7.0,  0.980,  True,  "Distill"),
    ("Qwen3-8B",         8.0,  0.730,  True,  "Qwen3"),
    ("Qwen3-14B",       14.0,  0.480,  True,  "Qwen3"),
    # Additional: 7B direct
    ("Qwen-2.5-7B-dir",  7.6,  1.000,  True,  "Qwen-2.5-dir"),
]

fig, ax = plt.subplots(figsize=(5.5, 3.0))

families = ["Qwen-2.5", "Qwen2.5-dir", "Qwen-base", "Phi", "Mistral", "Distill", "Qwen3"]
colors = {
    "Qwen-2.5": "#2166ac",
    "Qwen-2.5-dir": "#4393c3",
    "Qwen-base": "#92c5de",
    "Phi": "#4d9221",
    "Mistral": "#b2182b",
    "Distill": "#d6604d",
    "Qwen3": "#762a83",
}
markers = {
    "Qwen-2.5": "o",
    "Qwen-2.5-dir": "s",
    "Qwen-base": "D",
    "Phi": "^",
    "Mistral": "v",
    "Distill": "<",
    "Qwen3": "*",
}

for family in families:
    pts = [(p, fw) for _, p, fw, _, f in data if f == family and p > 0]
    if pts:
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=colors[family], marker=markers[family],
                markersize=8, linewidth=1.5, alpha=0.85, label=family)

# Qwen3 separate
pts3 = [(p, fw) for _, p, fw, _, f in data if f == "Qwen3"]
if pts3:
    pts3.sort()
    xs, ys = zip(*pts3)
    ax.plot(xs, ys, color=colors["Qwen3"], marker=markers["Qwen3"],
            markersize=10, linewidth=2.0, alpha=0.9, label="Qwen3")

ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
ax.set_xlabel("Model parameters (billions)", fontsize=9)
ax.set_ylabel("Followed-wrong rate (FW)", fontsize=9)
ax.set_title("Answer-text override attenuates with model scale", fontsize=10)
ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
ax.set_xscale("log")
ax.set_xticks([3, 7, 8, 14, 32])
ax.set_xticklabels(["3B", "7B", "8B", "14B", "32B"])
ax.set_ylim(-0.05, 1.1)
ax.grid(alpha=0.3)

# Annotation
ax.annotate("Override dominant\n(CC acc → 0)", xy=(5, 0.85), fontsize=7,
            ha="center", color="#666666",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.6))
ax.annotate("Override fades\n(CC acc → 0.4–0.9)", xy=(12, 0.45), fontsize=7,
            ha="center", color="#666666",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.6))
ax.annotate("Near-zero\n(CC acc → 0.94)", xy=(30, 0.15), fontsize=7,
            ha="center", color="#666666",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.6))

fig.tight_layout(pad=1.0)
fig.savefig("paper/figures/fig_fw_scale.pdf", dpi=150, bbox_inches="tight")
print("Saved paper/figures/fig_fw_scale.pdf")
