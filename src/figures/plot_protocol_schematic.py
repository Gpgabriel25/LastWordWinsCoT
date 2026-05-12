#!/usr/bin/env python3
"""
Generate a conceptual schematic figure for the CoT corruption protocol.

Shows the three-condition design:
  (A) Standard chain  → suffix corruption → answer follows answer text
  (B) Stripped chain  → suffix corruption → answer follows reasoning
  (C) Conflicting answer → answer follows conflicting text
Plus the Question-Only (QO) baseline.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "../paper/figures/fig_protocol_schematic.pdf")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 1, figsize=(8.5, 5.0))
ax = axes
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

# Color palette
C_QUESTION   = "#d0e8ff"   # light blue
C_REASON_OK  = "#d4f7d4"   # light green
C_SUFFIX_ANS = "#ffe4b5"   # moccasin (answer in suffix)
C_SUFFIX_CORRUPT = "#ffd0d0"  # light red (corrupted region)
C_SUFFIX_EMPTY = "#e8e8e8"  # gray (removed)
C_CONFLICTING = "#ffe0f0"  # light pink (conflicting answer)
C_QO         = "#f0f0ff"   # very light purple
C_ARROW      = "#444444"
C_ACCURACY_UP = "#2ca02c"
C_ACCURACY_DOWN = "#d62728"

EDGE_COLOR = "#888888"
FONTSIZE = 7.5
LABEL_FS  = 7.0
TITLE_FS  = 8.5

def box(ax, x, y, w, h, color, label, label_fs=FONTSIZE, ec=EDGE_COLOR, lw=0.8, alpha=1.0):
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.05",
                           facecolor=color, edgecolor=ec, linewidth=lw, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=label_fs, wrap=True,
            multialignment="center")

def hbar(ax, x, y, w, h, blocks):
    """Draw a horizontal chain of colored blocks.
    blocks: list of (frac, color, label)
    """
    cx = x
    for frac, color, label in blocks:
        bw = w * frac
        box(ax, cx, y, bw, h, color, label, label_fs=LABEL_FS)
        cx += bw

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.2, style="->"):
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(5, 6.75, "Corruption Protocol: Three-Condition Design",
        ha="center", va="center", fontsize=TITLE_FS, fontweight="bold")

ROW_H = 0.42
ROW_GAP = 0.18
LABEL_X = 0.05
CHAIN_X = 1.4
CHAIN_W = 6.0
ACC_X = 7.6

# ---------------------------------------------------------------------------
# Row labels header
# ---------------------------------------------------------------------------
ax.text(LABEL_X, 6.3, "Condition", ha="left", va="center", fontsize=LABEL_FS,
        fontweight="bold", color="#333333")
ax.text(ACC_X + 0.5, 6.3, "Suffix Δ", ha="center", va="center", fontsize=LABEL_FS,
        fontweight="bold", color="#333333")
ax.text(CHAIN_X + CHAIN_W / 2, 6.3, "CoT Chain structure (Q = question, R = reasoning steps)",
        ha="center", va="center", fontsize=LABEL_FS, color="#555555", style="italic")

ax.axhline(6.15, xmin=0.01, xmax=0.99, color="#aaaaaa", lw=0.7, linestyle="--")

# ---- Row 1: Standard chain, no corruption (baseline) ----------------------
Y1 = 5.55
ax.text(LABEL_X, Y1 + ROW_H / 2, "(1) Standard\nbaseline", ha="left", va="center",
        fontsize=LABEL_FS, color="#333333")
hbar(ax, CHAIN_X, Y1, CHAIN_W, ROW_H, [
    (0.20, C_QUESTION,   "Q"),
    (0.55, C_REASON_OK,  "R₁  R₂  R₃  …  Rₙ"),
    (0.25, C_SUFFIX_ANS, "answer = X"),
])
ax.text(ACC_X + 0.5, Y1 + ROW_H / 2, "—", ha="center", va="center",
        fontsize=FONTSIZE, color="#555555")

# ---- Row 2: Standard chain, suffix corrupted --------------------------------
Y2 = Y1 - ROW_H - ROW_GAP
ax.text(LABEL_X, Y2 + ROW_H / 2, "(2) Suffix\ncorrupt\n(standard)", ha="left", va="center",
        fontsize=LABEL_FS, color="#333333")
hbar(ax, CHAIN_X, Y2, CHAIN_W, ROW_H, [
    (0.20, C_QUESTION,      "Q"),
    (0.55, C_REASON_OK,     "R₁  R₂  R₃  …  Rₙ"),
    (0.25, C_SUFFIX_CORRUPT, "answer = X′ ✗"),
])
ax.text(ACC_X + 0.5, Y2 + ROW_H / 2, "−0.76", ha="center", va="center",
        fontsize=FONTSIZE, fontweight="bold", color=C_ACCURACY_DOWN)
arrow(ax, CHAIN_X + CHAIN_W * 0.875, Y1, CHAIN_X + CHAIN_W * 0.875, Y2 + ROW_H,
      color=C_ACCURACY_DOWN, lw=1.0)

# ---- Row 3: Stripped chain, suffix corrupted --------------------------------
Y3 = Y2 - ROW_H - ROW_GAP
ax.text(LABEL_X, Y3 + ROW_H / 2, "(3) Suffix\ncorrupt\n(stripped)", ha="left", va="center",
        fontsize=LABEL_FS, color="#333333")
hbar(ax, CHAIN_X, Y3, CHAIN_W, ROW_H, [
    (0.20, C_QUESTION,      "Q"),
    (0.55, C_REASON_OK,     "R₁  R₂  R₃  …  Rₙ"),
    (0.25, C_SUFFIX_CORRUPT, "corrupted ✗"),
])
ax.text(ACC_X + 0.5, Y3 + ROW_H / 2, "−0.04", ha="center", va="center",
        fontsize=FONTSIZE, fontweight="bold", color=C_ACCURACY_UP)
# bracket showing collapse
ax.annotate("", xy=(ACC_X + 0.3, Y3 + ROW_H * 0.5),
            xytext=(ACC_X + 0.3, Y2 + ROW_H * 0.5),
            arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.0))
ax.text(ACC_X + 0.0, (Y2 + Y3) / 2 + ROW_H / 2, "19×\ncollapse", ha="right", va="center",
        fontsize=6.5, color="#555555")

# ---- Row 4: Conflicting answer  --------------------------------
Y4 = Y3 - ROW_H - ROW_GAP
ax.text(LABEL_X, Y4 + ROW_H / 2, "(4) Conflicting\nanswer", ha="left", va="center",
        fontsize=LABEL_FS, color="#333333")
hbar(ax, CHAIN_X, Y4, CHAIN_W, ROW_H, [
    (0.20, C_QUESTION,    "Q"),
    (0.55, C_REASON_OK,   "R₁  R₂  R₃  …  Rₙ"),
    (0.25, C_CONFLICTING, "answer = X′ (wrong)"),
])
ax.text(ACC_X + 0.5, Y4 + ROW_H / 2, "FW\nratio", ha="center", va="center",
        fontsize=LABEL_FS, color="#555555")

# ---- Row 5: Question-only baseline  --------------------------------
Y5 = Y4 - ROW_H - ROW_GAP * 1.5
ax.text(LABEL_X, Y5 + ROW_H / 2, "(5) Question\nonly (QO)", ha="left", va="center",
        fontsize=LABEL_FS, color="#333333")
hbar(ax, CHAIN_X, Y5, CHAIN_W, ROW_H, [
    (0.20, C_QUESTION, "Q"),
    (0.80, C_QO,       "← no chain, answer requested directly →"),
])
ax.text(ACC_X + 0.5, Y5 + ROW_H / 2, "control", ha="center", va="center",
        fontsize=LABEL_FS, color="#555555")

# ---- Horizontal separator before QO ----------------------------------------
ax.axhline(Y5 + ROW_H + ROW_GAP * 0.5, xmin=0.01, xmax=0.99,
           color="#cccccc", lw=0.6, linestyle=":")

# ---- Legend box  ----------------------------------------
legend_elements = [
    mpatches.Patch(facecolor=C_QUESTION,      edgecolor=EDGE_COLOR, label="Question (Q)"),
    mpatches.Patch(facecolor=C_REASON_OK,     edgecolor=EDGE_COLOR, label="Reasoning steps (intact)"),
    mpatches.Patch(facecolor=C_SUFFIX_ANS,    edgecolor=EDGE_COLOR, label="Answer suffix (original)"),
    mpatches.Patch(facecolor=C_SUFFIX_CORRUPT, edgecolor=EDGE_COLOR, label="Corrupted region"),
    mpatches.Patch(facecolor=C_CONFLICTING,   edgecolor=EDGE_COLOR, label="Conflicting answer text"),
    mpatches.Patch(facecolor=C_QO,            edgecolor=EDGE_COLOR, label="No chain (QO control)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=6.2,
          ncol=3, framealpha=0.92, edgecolor="#aaaaaa",
          bbox_to_anchor=(0.99, 0.0))

# ---- Key insight annotation ---------------------------------------------------
ax.text(5.0, Y5 - 0.35,
        "Key: suffix sensitivity collapses 19× when the explicit answer statement is removed "
        "(rows 2→3),\n"
        "identifying answer placement—not reasoning content—as the mechanism.",
        ha="center", va="center", fontsize=6.8,
        style="italic", color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffbe6", edgecolor="#cccc88", lw=0.8))

# ---------------------------------------------------------------------------
fig.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
plt.close()
print(f"Saved: {OUT_PATH}")
