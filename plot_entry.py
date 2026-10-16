"""Newcomer entry figures for RQ1 (before vs. after). Reads the outputs of
analysis.py.

The main figure shows the mean number of newcomers per repository in each
month relative to the threshold, only in the months where every repository
is present (N ranges from 6 to 21 months, and the extreme months keep only
the early, larger adopters).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INPUT_DIR = "results"
FIGURE = "newcomer_entry"

COLOR_BEFORE, COLOR_AFTER = "#86b6ef", "#1c5cab"
SHADE, SHADE_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

rel = pd.read_csv(f"{INPUT_DIR}/rq1_newcomers_relative_month.csv")
n_repos = rel["repositories"].max()
rel = rel[rel["repositories"] == n_repos]

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

top = rel["mean_per_repository"].max()
for before, color, label in (
    (True, COLOR_BEFORE, "Before AI adoption"),
    (False, COLOR_AFTER, "After AI adoption"),
):
    g = rel[(rel["relative_month"] < 0) == before]
    ax.bar(
        g["relative_month"],
        g["mean_per_repository"],
        width=0.78,
        color=color,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.text(
        g["relative_month"].mean(),
        top * 1.17,
        label,
        ha="center",
        fontsize=9,
        color=color if not before else "#3b78c4",
        fontweight="bold",
    )
    for m in (g["relative_month"].min(), g["relative_month"].max()):
        v = g.loc[g["relative_month"] == m, "mean_per_repository"].iloc[0]
        ax.text(
            m, v + top * 0.025, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, color=SHADE_2
        )
ax.plot([-0.5, -0.5], [0, top * 1.12], color="#8a8a85", linewidth=0.8, linestyle="--")

ax.set_xticks(rel["relative_month"])
ax.set_xticklabels([f"{m:+d}" if m else "0" for m in rel["relative_month"]])
ax.tick_params(axis="x", length=0, pad=4)
ax.set_xlabel("Months before and after AI adoption", color=SHADE_2, fontsize=8.5)
ax.set_yticks([])
ax.set_ylim(0, top * 1.3)
ax.set_title("New contributors arriving per month", loc="left", pad=20, fontweight="bold")
ax.text(
    0,
    1.04,
    f"average per repository ({n_repos} repositories)",
    transform=ax.transAxes,
    fontsize=8.5,
    color=SHADE_2,
)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/{FIGURE}.{ext}", bbox_inches="tight", dpi=300)
print(
    f"Figure saved to {FIGURE}.pdf and {FIGURE}.png  (months {rel['relative_month'].min()} "
    f"a {rel['relative_month'].max()}, {n_repos} repos)"
)

# Full period: per-repository index (newcomers in the month / the repository's
# own mean in the before window, median across repositories). Months with fewer
# than MIN_REPOS_INDEX repositories are dropped, and months without every
# repository are drawn in a lighter shade.
MIN_REPOS_INDEX = 20
FIGURE_INDEX = "newcomer_entry_index"

ind = pd.read_csv(f"{INPUT_DIR}/rq1_newcomers_index_relative_month.csv")
ind = ind[ind["repositories"] >= MIN_REPOS_INDEX]
n_all = ind["repositories"].max()

fig, ax = plt.subplots(figsize=(7.0, 2.9))
for r in ind.itertuples():
    color = COLOR_BEFORE if r.relative_month < 0 else COLOR_AFTER
    ax.bar(
        r.relative_month,
        r.median_index,
        width=0.8,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        alpha=1.0 if r.repositories == n_all else 0.45,
    )
ax.axhline(1.0, color="#8a8a85", linewidth=0.8, linestyle=":")
top = ind["median_index"].max()
ax.plot([-0.5, -0.5], [0, top * 1.12], color="#8a8a85", linewidth=0.8, linestyle="--")
ax.text(-0.5, top * 1.14, "AI adoption threshold", fontsize=7, color=SHADE_2, ha="center")
ax.text(
    ind["relative_month"].min() - 0.4,
    top * 0.86,
    f"1.0 = the repository's own monthly mean before adoption\n"
    f"Lighter bars: fewer than {n_all} repositories in the month",
    fontsize=7,
    color=SHADE_2,
    va="top",
    linespacing=1.4,
)
ax.set_xticks(ind["relative_month"])
ax.set_xticklabels(
    [
        f"{m}\n{n}" if m % 2 == 0 else f"{m}\n"
        for m, n in zip(ind["relative_month"], ind["repositories"])
    ],
    fontsize=6.5,
)
ax.text(
    -0.012,
    -0.138,
    "repos",
    transform=ax.transAxes,
    fontsize=6.5,
    color=SHADE_2,
    ha="right",
    va="center",
)
ax.set_xlabel("Month relative to the repository's adoption threshold")
ax.set_ylabel("Newcomers / before-window mean\n(median across repositories)")
ax.set_ylim(0, top * 1.25)
ax.set_xlim(ind["relative_month"].min() - 0.7, ind["relative_month"].max() + 0.7)
ax.grid(axis="y", color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
ax.set_title(
    f"Monthly newcomer entry, each repository relative to itself ({n_all} repositories)",
    loc="left",
    pad=4,
)
from matplotlib.patches import Patch
import os

os.makedirs("figures", exist_ok=True)
ax.legend(
    handles=[
        Patch(color=COLOR_BEFORE, label="Before adoption"),
        Patch(color=COLOR_AFTER, label="After adoption"),
    ],
    loc="upper left",
    bbox_to_anchor=(0.0, 1.0),
    frameon=False,
    handlelength=1.0,
)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/{FIGURE_INDEX}.{ext}", bbox_inches="tight", dpi=300)
print(
    f"Figure saved to {FIGURE_INDEX}.pdf and {FIGURE_INDEX}.png  (months "
    f"{ind['relative_month'].min()} a {ind['relative_month'].max()}, >= {MIN_REPOS_INDEX} repos)"
)

# Boxplot per relative month: newcomers per repository, over the same months
# as the bar figure. Whiskers go from min to max; symlog axis because the tail
# reaches ~150 and some months are zero.
FIGURE_BOX = "newcomer_entry_box"

win = pd.read_csv(f"{INPUT_DIR}/rq1_newcomers_windows_per_repository.csv")
months = sorted(rel["relative_month"])
win = win[win["relative_month"].isin(months)]
assert (win.groupby("relative_month")["Repository"].nunique() == n_repos).all()

fig, ax = plt.subplots(figsize=(7.0, 3.2))
for m in months:
    v = win.loc[win["relative_month"] == m, "newcomers"]
    color = COLOR_BEFORE if m < 0 else COLOR_AFTER
    ax.boxplot(
        v,
        positions=[m],
        widths=0.42,
        whis=(0, 100),
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=color, edgecolor=color, linewidth=0.8),
        medianprops=dict(color="white", linewidth=1.6),
        whiskerprops=dict(color=color, linewidth=0.9),
        capprops=dict(color=color, linewidth=0.9),
    )
    med = v.median()
    ax.text(m + 0.25, med, f"{med:g}", ha="left", va="center", fontsize=7.5, color=SHADE_2)
ax.plot([-0.5, -0.5], [0, 200], color="#8a8a85", linewidth=0.8, linestyle="--")
ax.set_yscale("symlog", linthresh=1, linscale=0.4)
ax.set_ylim(0, 400)
ax.set_yticks([0, 1, 2, 5, 10, 20, 50, 100])
ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "50", "100"])
ax.tick_params(axis="y", length=0, labelsize=8, colors=SHADE_2)
ax.grid(axis="y", color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
ax.set_xticks(months)
ax.set_xticklabels([f"{m:+d}" if m else "0" for m in months])
ax.tick_params(axis="x", length=0, pad=4)
ax.set_xlim(months[0] - 0.6, months[-1] + 0.8)
ax.set_xlabel("Months before and after AI adoption", color=SHADE_2, fontsize=8.5)
ax.set_title("New contributors arriving per month", loc="left", pad=20, fontweight="bold")
ax.text(
    0,
    1.04,
    f"newcomers per repository ({n_repos} repositories)",
    transform=ax.transAxes,
    fontsize=8.5,
    color=SHADE_2,
)
for before, color, label in (
    (True, "#3b78c4", "Before AI adoption"),
    (False, COLOR_AFTER, "After AI adoption"),
):
    ms = [m for m in months if (m < 0) == before]
    ax.text(sum(ms) / len(ms), 260, label, ha="center", fontsize=9, color=color, fontweight="bold")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/{FIGURE_BOX}.{ext}", bbox_inches="tight", dpi=300)
print(f"Figure saved to {FIGURE_BOX}.pdf and {FIGURE_BOX}.png")
print(
    "  medians: "
    + ", ".join(
        f"{m:+d}={win.loc[win['relative_month'] == m, 'newcomers'].median():g}" for m in months
    )
)

# Same boxplot with each box colored by its Scott-Knott ESD rank
# (rq1_months_scott_knott.csv), then one panel per language.
COLORS_SK = ["#2a78d6", "#eb6834", "#1b9e77", "#7570b3", "#c9a227", "#d03b8a"]
sk = pd.read_csv(f"{INPUT_DIR}/rq1_months_scott_knott.csv")
assert (sk.loc[sk["language"] == "global", "repositories"] == n_repos).all()
assert sorted(sk["relative_month"].unique()) == months


def _boxes_sk(
    ax, data: pd.DataFrame, rank: dict, width: float, shows_median: bool, ms=None, cutoffs=(-0.5,)
):
    ms = months if ms is None else ms
    for m in ms:
        v = data.loc[data["relative_month"] == m, "newcomers"]
        color = COLORS_SK[(rank[m] - 1) % len(COLORS_SK)]
        ax.boxplot(
            v,
            positions=[m],
            widths=width,
            whis=(0, 100),
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color, edgecolor=color, linewidth=0.8),
            medianprops=dict(color="white", linewidth=1.4),
            whiskerprops=dict(color=color, linewidth=0.8),
            capprops=dict(color=color, linewidth=0.8),
        )
        if shows_median:
            ax.text(
                m + 0.25,
                v.median(),
                f"{v.median():g}",
                ha="left",
                va="center",
                fontsize=7.5,
                color=SHADE_2,
            )
    for c in cutoffs:
        ax.plot([c, c], [0, 200], color="#8a8a85", linewidth=0.8, linestyle="--")
    ax.set_yscale("symlog", linthresh=1, linscale=0.4)
    ax.set_ylim(0, 400)
    ax.set_yticks([0, 1, 2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "50", "100"])
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(ms[0] - 0.6, ms[-1] + 0.8)


def _rank(language: str, tab: pd.DataFrame = None) -> dict:
    tab = sk if tab is None else tab
    d = tab[tab["language"] == language]
    return dict(zip(d["relative_month"], d["scott_knott_esd"]))


def _legend_sk(ax, ranks, **kw):
    ax.legend(
        handles=[
            Patch(color=COLORS_SK[(r - 1) % len(COLORS_SK)], label=f"Rank {r}")
            for r in sorted(set(ranks))
        ],
        frameon=False,
        fontsize=7.5,
        **kw,
    )


FIGURE_BOX_SK = "newcomer_entry_box_sk"
rank_global = _rank("global")
fig, ax = plt.subplots(figsize=(7.0, 3.2))
_boxes_sk(ax, win, rank_global, 0.42, True)
ax.tick_params(axis="y", length=0, labelsize=8, colors=SHADE_2)
ax.set_xticks(months)
ax.set_xticklabels([f"{m:+d}" if m else "0" for m in months])
ax.tick_params(axis="x", length=0, pad=4)
ax.set_xlabel("Months before and after AI adoption", color=SHADE_2, fontsize=8.5)
ax.set_title("New contributors arriving per month", loc="left", pad=20, fontweight="bold")
ax.text(
    0,
    1.04,
    f"newcomers per repository ({n_repos} repositories) · color = Scott-Knott ESD rank "
    f"(rank 1 = highest medians)",
    transform=ax.transAxes,
    fontsize=7.5,
    color=SHADE_2,
)
for before, label in ((True, "Before AI adoption"), (False, "After AI adoption")):
    ms = [m for m in months if (m < 0) == before]
    ax.text(
        sum(ms) / len(ms), 260, label, ha="center", fontsize=9, color=SHADE_2, fontweight="bold"
    )
_legend_sk(ax, rank_global.values(), loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/{FIGURE_BOX_SK}.{ext}", bbox_inches="tight", dpi=300)
print(
    f"Figure saved to {FIGURE_BOX_SK}.pdf and {FIGURE_BOX_SK}.png  (ranks "
    + ", ".join(f"{m:+d}={r}" for m, r in rank_global.items())
    + ")"
)

FIGURE_BOX_SK_LANG = "newcomer_entry_box_sk_language"
langs = [l for l in sk["language"].unique() if l != "global"]
langs = sorted(langs)
fig, axs = plt.subplots(4, 4, figsize=(7.0, 7.4), sharex=True, sharey=True)
for ax, l in zip(axs.flat, langs):
    _boxes_sk(ax, win[win["Language"] == l], _rank(l), 0.55, False)
    n = int(sk.loc[sk["language"] == l, "repositories"].iloc[0])
    ax.set_title(f"{l} ({n})", loc="left", fontsize=8.5, fontweight="bold", pad=3)
    ax.tick_params(length=0, labelsize=7, colors=SHADE_2, labelbottom=True)
    ax.set_xticks([-6, -1, 0, 5])
    ax.set_xticklabels(["-6", "-1", "0", "+5"])
for ax in axs.flat[len(langs) :]:
    ax.axis("off")
_legend_sk(axs.flat[len(langs)], sk["scott_knott_esd"], loc="center left")
fig.suptitle(
    "New contributors per month by language",
    x=0.02,
    y=1.0,
    ha="left",
    va="top",
    fontweight="bold",
    fontsize=10,
)
fig.text(
    0.02,
    0.972,
    "color = Scott-Knott ESD rank within each language (rank 1 = highest medians; "
    "colors are not comparable across panels) · x axis = months from AI adoption",
    fontsize=7,
    color=SHADE_2,
    va="top",
)
fig.tight_layout(rect=(0, 0, 1, 0.955))
for ext in ("pdf", "png"):
    fig.savefig(f"figures/{FIGURE_BOX_SK_LANG}.{ext}", bbox_inches="tight", dpi=300)
print(
    f"Figure saved to {FIGURE_BOX_SK_LANG}.pdf and {FIGURE_BOX_SK_LANG}.png  ({len(langs)} languages)"
)

# Three windows: the six months before -6 (-12..-7) form the "earlier" window
# and the SK-ESD is run again on the 18 months (rq1_months3_*.csv).
FIGURE_BOX_SK3 = "newcomer_entry_box_sk3"
sk3 = pd.read_csv(f"{INPUT_DIR}/rq1_months3_scott_knott.csv")
win3 = pd.read_csv(f"{INPUT_DIR}/rq1_months3_newcomers.csv")
months3 = sorted(sk3["relative_month"].unique())
assert (sk3.loc[sk3["language"] == "global", "repositories"] == n_repos).all()
assert win3.groupby("relative_month")["Repository"].nunique().eq(n_repos).all()
CUTOFFS3 = (-6.5, -0.5)
rank3 = _rank("global", sk3)

fig, ax = plt.subplots(figsize=(7.0, 3.2))
_boxes_sk(ax, win3, rank3, 0.5, False, months3, CUTOFFS3)
for m in months3:
    v = win3.loc[win3["relative_month"] == m, "newcomers"]
    ax.text(
        m,
        0.05,
        f"{v.median():g}",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=SHADE_2,
        transform=ax.get_xaxis_transform(),
    )
ax.tick_params(axis="y", length=0, labelsize=8, colors=SHADE_2)
ax.set_xticks(months3)
ax.set_xticklabels([f"{m:+d}" if m else "0" for m in months3], fontsize=7.5)
ax.tick_params(axis="x", length=0, pad=4)
ax.set_xlabel(
    "Months before and after AI adoption (numbers at the bottom = medians)",
    color=SHADE_2,
    fontsize=8.5,
)
ax.set_title("New contributors arriving per month", loc="left", pad=20, fontweight="bold")
ax.text(
    0,
    1.04,
    f"newcomers per repository ({n_repos} repositories) · color = Scott-Knott ESD rank "
    f"(rank 1 = highest medians)",
    transform=ax.transAxes,
    fontsize=7.5,
    color=SHADE_2,
)
for (a, b), label in (
    ((-12, -7), "Earlier"),
    ((-6, -1), "Before AI adoption"),
    ((0, 5), "After AI adoption"),
):
    ax.text((a + b) / 2, 260, label, ha="center", fontsize=9, color=SHADE_2, fontweight="bold")
_legend_sk(ax, rank3.values(), loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/{FIGURE_BOX_SK3}.{ext}", bbox_inches="tight", dpi=300)
print(
    f"Figure saved to {FIGURE_BOX_SK3}.pdf and {FIGURE_BOX_SK3}.png  (ranks "
    + ", ".join(f"{m:+d}={r}" for m, r in rank3.items())
    + ")"
)

FIGURE_BOX_SK3_LANG = "newcomer_entry_box_sk3_language"
langs3 = sorted(l for l in sk3["language"].unique() if l != "global")
fig, axs = plt.subplots(4, 4, figsize=(7.0, 7.4), sharex=True, sharey=True)
for ax, l in zip(axs.flat, langs3):
    _boxes_sk(ax, win3[win3["Language"] == l], _rank(l, sk3), 0.6, False, months3, CUTOFFS3)
    n = int(sk3.loc[sk3["language"] == l, "repositories"].iloc[0])
    ax.set_title(f"{l} ({n})", loc="left", fontsize=8.5, fontweight="bold", pad=3)
    ax.tick_params(length=0, labelsize=7, colors=SHADE_2, labelbottom=True)
    ax.set_xticks([-12, -6, 0, 5])
    ax.set_xticklabels(["-12", "-6", "0", "+5"])
for ax in axs.flat[len(langs3) :]:
    ax.axis("off")
_legend_sk(axs.flat[len(langs3)], sk3["scott_knott_esd"], loc="center left")
fig.suptitle(
    "New contributors per month by language",
    x=0.02,
    y=1.0,
    ha="left",
    va="top",
    fontweight="bold",
    fontsize=10,
)
fig.text(
    0.02,
    0.972,
    "color = Scott-Knott ESD rank within each language (rank 1 = highest medians; "
    "colors are not comparable across panels) · dashed lines at -6 and 0",
    fontsize=7,
    color=SHADE_2,
    va="top",
)
fig.tight_layout(rect=(0, 0, 1, 0.955))
for ext in ("pdf", "png"):
    fig.savefig(f"figures/{FIGURE_BOX_SK3_LANG}.{ext}", bbox_inches="tight", dpi=300)
print(
    f"Figure saved to {FIGURE_BOX_SK3_LANG}.pdf and {FIGURE_BOX_SK3_LANG}.png  ({len(langs3)} languages)"
)
