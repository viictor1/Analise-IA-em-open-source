"""Newcomer retention before and after adoption (RQ1). Reads the outputs of
analysis.py.

Shows the aggregate retention of eligible newcomers at each horizon k (1, 3
and 6 months). The test (paired Wilcoxon per repository, with BH) is not
drawn; the subtitle only says whether any k was significant.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs("figures", exist_ok=True)

INPUT_DIR = "results"
FIGURE = "newcomer_retention"
ALPHA = 0.05

COLOR_BEFORE, COLOR_AFTER = "#86b6ef", "#1c5cab"
SHADE, SHADE_2 = "#0b0b0b", "#52514e"

glob = pd.read_csv(f"{INPUT_DIR}/rq1_retention_global.csv")
glob = glob[glob["comparison"] == "after_vs_before"].sort_values("k")
some_sig = (glob["mwu_p_fdr"] < ALPHA).any()

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#8a8a85",
        "axes.linewidth": 0.6,
        "xtick.color": SHADE,
        "text.color": SHADE,
    }
)
fig, ax = plt.subplots(figsize=(4.6, 2.9))

x = range(len(glob))
width = 0.38
for offset, col, color, label in (
    (-width / 2, "aggregated_rate_2", COLOR_BEFORE, "Before AI adoption"),
    (width / 2, "aggregated_rate_1", COLOR_AFTER, "After AI adoption"),
):
    bars = ax.bar(
        [i + offset for i in x],
        glob[col] * 100,
        width=width,
        color=color,
        edgecolor="white",
        linewidth=1.5,
        label=label,
    )
    ax.bar_label(
        bars, labels=[f"{v:.1%}" for v in glob[col]], padding=3, fontsize=8.5, color=SHADE_2
    )

ax.set_xticks(list(x))
ax.set_xticklabels([f"within {k} month{'s' if k > 1 else ''}" for k in glob["k"]])
ax.tick_params(axis="x", length=0, pad=6)
ax.set_yticks([])
ax.set_ylim(0, glob[["aggregated_rate_1", "aggregated_rate_2"]].max().max() * 100 * 1.35)
ax.set_title("Share of newcomers who came back", loc="left", pad=20, fontweight="bold")
ax.text(
    0,
    1.04,
    "no significant difference before vs. after"
    if not some_sig
    else "significant difference in at least one follow-up window",
    transform=ax.transAxes,
    fontsize=8.5,
    color=SHADE_2,
)
ax.legend(loc="upper left", frameon=False, handlelength=1.0, ncol=2, bbox_to_anchor=(0.0, 1.0))

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/{FIGURE}.{ext}", bbox_inches="tight", dpi=300)
print(f"Figure saved to {FIGURE}.pdf and {FIGURE}.png")
print(
    "  "
    + "; ".join(
        f"k={r.k} before {r.aggregated_rate_2:.1%} after {r.aggregated_rate_1:.1%} q={r.mwu_p_fdr:.2f}"
        for r in glob.itertuples()
    )
)
