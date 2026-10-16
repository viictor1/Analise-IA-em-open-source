"""Monthly newcomer flow in the repositories without any AI-evidence PR, over
the last 12 full months before the cutoff. Reads the outputs of analysis.py.

Plots the mean number of newcomers per repository in each month. The trend
(Mann-Kendall per repository and Wilcoxon of the taus against zero) comes
from rq1_no_ai_trend.csv.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs("figures", exist_ok=True)

INPUT_DIR = "results"
FIGURE = "newcomer_entry_no_ai"

COLOR = "#7a7a74"
SHADE, SHADE_2 = "#0b0b0b", "#52514e"

month = pd.read_csv(f"{INPUT_DIR}/rq1_no_ai_newcomers_month.csv")
trend = pd.read_csv(f"{INPUT_DIR}/rq1_no_ai_trend.csv").iloc[0]
n_repos = month["repositories"].max()

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#8a8a85",
        "axes.linewidth": 0.6,
        "xtick.color": SHADE_2,
        "text.color": SHADE,
    }
)
fig, ax = plt.subplots(figsize=(4.6, 2.9))

x = list(range(len(month)))
v = month["mean_per_repository"].tolist()
ax.bar(x, v, width=0.78, color=COLOR, edgecolor="white", linewidth=1.5)
top = max(v)
ax.text(
    sum(x) / len(x),
    top * 1.17,
    "No AI adoption",
    ha="center",
    fontsize=9,
    color=COLOR,
    fontweight="bold",
)
for i in (x[0], x[-1]):
    ax.text(
        i, v[i] + top * 0.025, f"{v[i]:.1f}", ha="center", va="bottom", fontsize=8.5, color=SHADE_2
    )

ax.set_xticks(x)
ax.set_xticklabels([pd.Period(m, freq="M").strftime("%b") for m in month["Month"]])
ax.tick_params(axis="x", length=0, pad=4)
ax.set_xlabel(
    f"Months ({pd.Period(month['Month'].iloc[0], freq='M').strftime('%b %Y')} to "
    f"{pd.Period(month['Month'].iloc[-1], freq='M').strftime('%b %Y')})",
    color=SHADE_2,
    fontsize=8.5,
)
ax.set_yticks([])
ax.set_ylim(0, top * 1.3)
ax.set_title("New contributors arriving per month", loc="left", pad=20, fontweight="bold")
ax.text(
    0,
    1.04,
    f"average per repository ({n_repos} repositories without AI)",
    transform=ax.transAxes,
    fontsize=8.5,
    color=SHADE_2,
)
print(f"  trend: median tau = {trend['tau_median']:.2f}, Wilcoxon p = {trend['wilcoxon_p']:.2f}")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/{FIGURE}.{ext}", bbox_inches="tight", dpi=300)
print(
    f"Figure saved to {FIGURE}.pdf and {FIGURE}.png  ({month['Month'].min()} to {month['Month'].max()}, "
    f"{n_repos} repos, {int(month['newcomers'].sum())} newcomers)"
)

# Monthly boxplot, with the same layout and y axis (symlog, 0 to 100) as the
# adopters' boxplot so that both figures can be compared.
FIGURE_BOX = "newcomer_entry_no_ai_box"
GRID = "#e1e0d9"

rep = pd.read_csv(f"{INPUT_DIR}/rq1_no_ai_newcomers_per_repository.csv")
months = list(month["Month"])

fig, ax = plt.subplots(figsize=(7.0, 3.2))
for i, m in enumerate(months):
    v = rep.loc[rep["Month"] == m, "newcomers"]
    ax.boxplot(
        v,
        positions=[i],
        widths=0.42,
        whis=(0, 100),
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=COLOR, edgecolor=COLOR, linewidth=0.8),
        medianprops=dict(color="white", linewidth=1.6),
        whiskerprops=dict(color=COLOR, linewidth=0.9),
        capprops=dict(color=COLOR, linewidth=0.9),
    )
    med = v.median()
    ax.text(i + 0.25, med, f"{med:g}", ha="left", va="center", fontsize=7.5, color=SHADE_2)
ax.set_yscale("symlog", linthresh=1, linscale=0.4)
ax.set_ylim(0, 400)
ax.set_yticks([0, 1, 2, 5, 10, 20, 50, 100])
ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "50", "100"])
ax.tick_params(axis="y", length=0, labelsize=8, colors=SHADE_2)
ax.grid(axis="y", color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
ax.set_xticks(range(len(months)))
ax.set_xticklabels([pd.Period(m, freq="M").strftime("%b") for m in months])
ax.tick_params(axis="x", length=0, pad=4)
ax.set_xlim(-0.6, len(months) - 1 + 0.8)
ax.set_xlabel(
    f"Months ({pd.Period(months[0], freq='M').strftime('%b %Y')} to "
    f"{pd.Period(months[-1], freq='M').strftime('%b %Y')})",
    color=SHADE_2,
    fontsize=8.5,
)
ax.set_title("New contributors arriving per month", loc="left", pad=20, fontweight="bold")
ax.text(
    0,
    1.04,
    f"newcomers per repository ({n_repos} repositories without AI)",
    transform=ax.transAxes,
    fontsize=8.5,
    color=SHADE_2,
)
ax.text(
    (len(months) - 1) / 2,
    260,
    "No AI adoption",
    ha="center",
    fontsize=9,
    color=COLOR,
    fontweight="bold",
)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/{FIGURE_BOX}.{ext}", bbox_inches="tight", dpi=300)
print(f"Figure saved to {FIGURE_BOX}.pdf and {FIGURE_BOX}.png")
print(
    "  medians: "
    + ", ".join(f"{m}={rep.loc[rep['Month'] == m, 'newcomers'].median():g}" for m in months)
)
