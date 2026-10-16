"""Schematic of the analysis windows (baseline | before | after) for a
hypothetical repository with the threshold in June 2025 and N = 14. Uses
the rules of analysis.py (BASELINE_MONTHS = 24, COLLECTION_START = 2022-01,
CUTOFF_MONTH = 2026-08). Reads no data.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs("figures", exist_ok=True)

START, CUTOFF, THR, BASELINE = (
    pd.Period("2022-01", "M"),
    pd.Period("2026-08", "M"),
    pd.Period("2025-06", "M"),
    24,
)
N = (CUTOFF - THR).n
start_before = THR - N
start_base = max(START, start_before - BASELINE)
x = lambda p: (p - START).n  # months since 2022-01

BLUE, ORANGE, SHADE, GRAY, LIGHT = "#5672b3", "#b35672", "#333333", "#888888", "#c2d3bb"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
# exact column width of the ACM sigconf template (241 pt)
fig, ax = plt.subplots(figsize=(241 / 72, 1.55))
WINDOWS = [
    (start_base, start_before, LIGHT, SHADE, "Baseline\n(up to 24 months)"),
    (start_before, THR, BLUE, "white", "Before\n($N$ months)"),
    (THR, CUTOFF, ORANGE, "white", "After\n($N$ months)"),
]
for a, b, color, txt, name in WINDOWS:
    ax.barh(0, x(b) - x(a), left=x(a), height=0.6, color=color, edgecolor="white", linewidth=1.5)
    ax.text(
        (x(a) + x(b)) / 2,
        0,
        name,
        ha="center",
        va="center",
        color=txt,
        fontsize=7.5,
        fontweight="bold",
        linespacing=1.1,
    )
ax.text(
    (x(start_base) + x(start_before)) / 2,
    -0.42,
    "identifies\nexisting members",
    ha="center",
    va="top",
    color=SHADE,
    fontsize=7.5,
    linespacing=1.1,
)
ax.annotate(
    "",
    xy=(x(start_before), -0.47),
    xytext=(x(CUTOFF), -0.47),
    arrowprops={"arrowstyle": "|-|", "color": GRAY, "lw": 0.6, "mutation_scale": 2},
)
ax.text(
    (x(start_before) + x(CUTOFF)) / 2,
    -0.52,
    "newcomer = first PR\nsince baseline start",
    ha="center",
    va="top",
    color=SHADE,
    fontsize=7.5,
    linespacing=1.1,
)
ax.annotate(
    "threshold",
    xy=(x(THR), 0.3),
    xytext=(x(THR), 0.78),
    ha="center",
    va="bottom",
    fontsize=8,
    color=SHADE,
    arrowprops={"arrowstyle": "-|>", "color": SHADE, "lw": 0.8},
)
years = [pd.Period(f"{a}-01", "M") for a in range(2022, 2027)]
ax.set_xticks([x(p) for p in years], [str(p.year) for p in years], fontsize=7.5)
ax.set_xlim(-3, x(CUTOFF) + 1)
ax.set_ylim(-1.15, 1.08)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRAY)
ax.tick_params(axis="x", colors=SHADE, length=2)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.14)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/windows_timeline.{ext}", dpi=300)
