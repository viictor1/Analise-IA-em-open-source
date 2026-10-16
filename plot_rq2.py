"""RQ2 figures. Reads the outputs of analysis.py.

rq2_acceptance: acceptance rate per repository for the 2 groups of window 1
    (before) and the 5 groups of window 2 (after), boxplot + points.
rq2_gfi: who closes the good first issues, % of each window's GFIs per group
    (stacked bars).
rq2_gfi_per_repo: % of merged PRs that close a GFI per repository and group.
rq2_resolution: median resolution time (merged only) per repository, log
    scale.
rq2_resolution_3windows, rq2_acceptance_3windows, rq2_gfi_3windows: the same
    with the earlier window [thr-2N, thr-N) (Newcomer and Prior Contributor,
    no AI only).
A repository enters a group only with at least MIN_N_PER_REPO PRs, the same
minimum as the analysis.
"""

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_DIR = "results"
MIN_N_PER_REPO = 5
# Okabe-Ito palette; blue/orange = newcomer/veteran as in the RQ1 figures
BLUE, ORANGE = "#0072b2", "#e69f00"
PURPLE, GREEN, RED = "#cc79a7", "#009e73", "#222222"
SHADE, SHADE_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

# (window, CSV group, paper label, color)
GROUPS = [
    ("window1", "Newcomer", "Newcomer", BLUE),
    ("window1", "Veteran", "Prior\nContributor", ORANGE),
    ("window2", "Newcomer", "Newcomer", BLUE),
    ("window2", "Veteran", "Prior\nContributor", ORANGE),
    ("window2", "Newcomer_AI", "Newcomer\nwith AI", PURPLE),
    ("window2", "Veteran_AI", "Prior\nContributor\nwith AI", GREEN),
    ("window2", "AI_Agent", "AI Agent", RED),
]
POS = [0, 1, 2.6, 3.6, 4.6, 5.6, 6.6]

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8a85",
        "axes.linewidth": 0.6,
        "xtick.color": SHADE_2,
        "ytick.color": SHADE_2,
        "text.color": SHADE,
    }
)


def _box(ax, v, pos, color, log=False, whis=1.5):
    """Boxplot without fliers. On a log axis the whiskers (1.5 IQR) are computed
    on the log scale; box and median do not change.
    """
    v = np.asarray(v, dtype=float)
    s = matplotlib.cbook.boxplot_stats(v, whis=whis)[0]
    if log:
        assert (v > 0).all()
        lq1, lq3 = np.log(s["q1"]), np.log(s["q3"])
        lo, hi = np.exp(lq1 - whis * (lq3 - lq1)), np.exp(lq3 + whis * (lq3 - lq1))
        s["whislo"], s["whishi"] = v[v >= lo].min(), v[v <= hi].max()
    ax.bxp(
        [s],
        positions=[pos],
        widths=0.6,
        showfliers=False,
        showcaps=False,
        patch_artist=True,
        boxprops=dict(facecolor="none", edgecolor=color, linewidth=1.1),
        medianprops=dict(color=color, linewidth=2.2),
        whiskerprops=dict(color=color, linewidth=0.8),
    )


def _figure(
    d,
    col_val,
    ylabel,
    title,
    fmt,
    name,
    log=False,
    points=True,
    groups=GROUPS,
    pos_x=POS,
    separators=(1.8,),
    labels_window=((0.5, "Before adoption"), (4.6, "After adoption")),
    figsize=(6.8, 3.8),
):
    fig, ax = plt.subplots(figsize=figsize)
    if log:
        ax.set_yscale("log")
    for k, ((j, g, lbl, color), pos) in enumerate(zip(groups, pos_x)):
        v = d.loc[(d["window"] == j) & (d["group"] == g), col_val].dropna()
        _box(ax, v, pos, color, log=log and points, whis=1.5 if points else (5, 95))
        if points:
            jit = np.random.default_rng(k).uniform(-0.2, 0.2, len(v))
            ax.scatter(pos + jit, v, s=7, color=color, alpha=0.4, linewidths=0)
        if not points:
            ax.annotate(
                f"{len(v):,} PRs",
                (pos, -0.02),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=7,
                color=SHADE_2,
            )
        ax.annotate(
            fmt(v.median()),
            (pos, 1.0),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=8,
            color=SHADE_2,
            fontweight="bold",
        )
    ax.set_xticks(pos_x)
    ax.set_xticklabels([lbl for _, _, lbl, _ in groups])
    ax.tick_params(axis="x", length=0, pad=4 if points else 14)
    for x in separators:
        ax.axvline(x, color="#8a8a85", linewidth=0.8, linestyle=":")
    for x, t in labels_window:
        ax.annotate(
            t,
            (x, 1.08),
            xycoords=("data", "axes fraction"),
            ha="center",
            fontsize=9,
            color=SHADE_2,
            fontweight="bold",
        )
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=SHADE_2)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10, pad=24)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  {name}.pdf/.png")


ac = pd.read_csv(f"{INPUT_DIR}/rq2_acceptance_rate_per_repository.csv")
# annotate the median of the repositories that pass the minimum
_figure(
    ac,
    "rate",
    "Acceptance rate (per repository)",
    "Acceptance rate by contribution group",
    lambda m: f"{m:.0%}",
    "rq2_acceptance",
)

rt = pd.read_csv(f"{INPUT_DIR}/rq2_resolution_time_per_repository_descriptive.csv")
rt = rt[rt["n"] >= MIN_N_PER_REPO]
_figure(
    rt,
    "median_days",
    "Median days to merge (per repository, log)",
    "Resolution time of merged PRs by contribution group",
    lambda m: f"{m:.2f}d",
    "rq2_resolution",
    log=True,
)

# 3 windows of N months: the earlier window [thr-2N, thr-N) before the two of
# rq2_resolution (230 repositories).
rt3 = pd.read_csv(f"{INPUT_DIR}/rq2_resolution_3windows_per_repository.csv")
rt3 = rt3[rt3["n"] >= MIN_N_PER_REPO]
_figure(
    rt3,
    "median_days",
    "Median days to merge (per repository, log)",
    "Resolution time of merged PRs by contribution group",
    lambda m: f"{m:.2f}d",
    "rq2_resolution_3windows",
    log=True,
    groups=[
        ("window0", "Newcomer", "Newcomer", BLUE),
        ("window0", "Veteran", "Prior\nContributor", ORANGE),
    ]
    + GROUPS,
    pos_x=[0, 1] + [p + 2.6 for p in POS],
    separators=(1.8, 4.4),
    labels_window=((0.5, "Earlier"), (3.1, "Before adoption"), (7.2, "After adoption")),
    figsize=(8.6, 3.8),
)

ac3 = pd.read_csv(f"{INPUT_DIR}/rq2_acceptance_3windows_per_repository.csv")
_figure(
    ac3,
    "rate",
    "Acceptance rate (per repository)",
    "Acceptance rate by contribution group",
    lambda m: f"{m:.0%}",
    "rq2_acceptance_3windows",
    groups=[
        ("window0", "Newcomer", "Newcomer", BLUE),
        ("window0", "Veteran", "Prior\nContributor", ORANGE),
    ]
    + GROUPS,
    pos_x=[0, 1] + [p + 2.6 for p in POS],
    separators=(1.8, 4.4),
    labels_window=((0.5, "Earlier"), (3.1, "Before adoption"), (7.2, "After adoption")),
    figsize=(8.6, 3.8),
)

# GFI per repository: % of the group's merged PRs that close a GFI, in the
# repositories that use GFIs in some window; annotation = median.
gfi = pd.read_csv(f"{INPUT_DIR}/rq2_gfi_per_repository_windows.csv").assign(
    pct=lambda x: x["rate"] * 100
)
fig, ax = plt.subplots(figsize=(6.8, 3.8))
for k, ((j, g, lbl, color), pos) in enumerate(zip(GROUPS, POS)):
    v = gfi.loc[(gfi["window"] == j) & (gfi["group"] == g), "pct"]
    _box(ax, v, pos, color)
    jit = np.random.default_rng(k).uniform(-0.2, 0.2, len(v))
    ax.scatter(pos + jit, v, s=7, color=color, alpha=0.4, linewidths=0)
    ax.annotate(
        f"{v.median():.2f}%",
        (pos, 1.0),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="bottom",
        fontsize=8,
        color=SHADE_2,
        fontweight="bold",
    )
    ax.annotate(
        f"{len(v)} repos",
        (pos, -0.02),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
        fontsize=7,
        color=SHADE_2,
    )
ax.set_xticks(POS)
ax.set_xticklabels([lbl for _, _, lbl, _ in GROUPS])
ax.tick_params(axis="x", length=0, pad=14)
ax.axvline(1.8, color="#8a8a85", linewidth=0.8, linestyle=":")
for x, t in ((0.5, "Before adoption"), (4.6, "After adoption")):
    ax.annotate(
        t,
        (x, 1.08),
        xycoords=("data", "axes fraction"),
        ha="center",
        fontsize=9,
        color=SHADE_2,
        fontweight="bold",
    )
# square-root axis: most repositories are near 0
ax.set_yscale("function", functions=(lambda x: np.sqrt(np.clip(x, 0, None)), np.square))
ax.set_ylim(0, 60)
ax.set_yticks([0, 1, 5, 10, 20, 40])
ax.grid(axis="y", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.set_ylabel("Merged PRs closing a GFI (%, per repository)", color=SHADE_2)
ax.set_title(
    "Good first issue resolution by contribution group",
    loc="left",
    fontweight="bold",
    fontsize=10,
    pad=24,
)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/rq2_gfi_per_repo.{ext}", dpi=200, bbox_inches="tight")
print("  rq2_gfi_per_repo.pdf/.png")

# Who closes the GFIs: one stacked bar per window, slice = % of the GFIs closed
# by a merged PR of the group (a GFI closed by 2 groups counts in both).
ORDER = [
    ("Newcomer", "Newcomer", BLUE),
    ("Newcomer_AI", "Newcomer with AI", PURPLE),
    ("Veteran", "Prior Contributor", ORANGE),
    ("Veteran_AI", "Prior Contributor with AI", GREEN),
    ("AI_Agent", "AI Agent", RED),
]


def _bars_gfi(q, windows, name):
    """Stacked bar per window (top to bottom, the reverse order of windows)."""
    fig, ax = plt.subplots(figsize=(6.8, 2.4 + 0.55 * (len(windows) - 2)))
    small = []
    for y, (j, lbl_j) in enumerate(windows):
        dj = q[q["window"] == j].set_index("group")
        x0 = 0.0
        for g, lbl, color in ORDER:
            if g not in dj.index:
                continue
            w = dj.loc[g, "pct_assignments"]
            ax.barh(
                y,
                w,
                left=x0,
                color=color,
                height=0.6,
                edgecolor="white",
                linewidth=0.8,
                label=lbl if y == 0 else None,
            )
            if w >= 5:
                ax.text(
                    x0 + w / 2,
                    y,
                    f"{w:.0f}%\n({dj.loc[g, 'gfis']})",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white",
                    fontweight="bold",
                )
            else:  # narrow slice: label below the bar, with a leader line
                small.append((x0 + w / 2, y, f"{w:.0f}% ({dj.loc[g, 'gfis']})", color))
            x0 += w
        ax.annotate(
            f"{dj['gfis_distinct_window'].iloc[0]:,} GFIs",
            (100.5, y),
            va="center",
            fontsize=8,
            color=SHADE_2,
            annotation_clip=False,
        )
    for k, (xc, y, txt, color) in enumerate(sorted(small, reverse=True)):
        ax.annotate(
            txt,
            (xc, y - 0.3),
            xytext=(xc - 3 - 14 * k, y - 0.62),
            ha="center",
            va="center",
            fontsize=7.5,
            color=color,
            fontweight="bold",
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=color, linewidth=0.8),
        )
    ax.set_ylim(-0.75, len(windows) - 0.6)
    ax.set_yticks(range(len(windows)))
    ax.set_yticklabels([r for _, r in windows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.set_xlabel("Share of good first issues closed by a merged PR", color=SHADE_2)
    ax.spines["left"].set_visible(False)
    ax.legend(ncol=3, fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    fig.text(
        0.99,
        0.0,
        "GFIs in all repositories; an issue closed by two groups counts in both",
        ha="right",
        va="top",
        fontsize=6,
        color=SHADE_2,
    )
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  {name}.pdf/.png")


_bars_gfi(
    pd.read_csv(f"{INPUT_DIR}/rq2_gfi_who_closes.csv"),
    (("window2", "After adoption"), ("window1", "Before adoption")),
    "rq2_gfi",
)
_bars_gfi(
    pd.read_csv(f"{INPUT_DIR}/rq2_gfi_who_closes_3windows.csv"),
    (("window2", "After adoption"), ("window1", "Before adoption"), ("window0", "Earlier")),
    "rq2_gfi_3windows",
)

# Paired per repository (what the paired Wilcoxon tests): each gray line is a
# repository linking group A to group B; the black line links the medians. The
# annotated count comes from repos_1_greater/repos_1_less of the analysis and
# is checked against the points drawn.
PAIRS_ROWS = [
    # (window A, group A, window B, group B, label A, label B, test file, g1, g2)
    (
        "window2",
        "Newcomer",
        "window2",
        "Veteran",
        "Newcomer",
        "Prior\nContributor",
        "window2",
        "Newcomer",
        "Veteran",
    ),
    (
        "window2",
        "Newcomer_AI",
        "window2",
        "Newcomer",
        "Newcomer\nwith AI",
        "Newcomer",
        "window2",
        "Newcomer_AI",
        "Newcomer",
    ),
    (
        "window2",
        "AI_Agent",
        "window2",
        "Newcomer",
        "AI Agent",
        "Newcomer",
        "window2",
        "AI_Agent",
        "Newcomer",
    ),
    (
        "window1",
        "Newcomer",
        "window2",
        "Newcomer",
        "Newcomer\nbefore",
        "Newcomer\nafter",
        "between_windows",
        "Newcomer[after]",
        "Newcomer[before]",
    ),
]
METRICS_ROWS = [
    (
        rt,
        "median_days",
        "rq2_resolution_time_median_per_repo",
        "Median days to merge (log)",
        True,
        "slower",
    ),
    (ac, "rate", "rq2_acceptance_rate", "Acceptance rate", False, "lower"),
]
COLOR_WORST, COLOR_BEST = "#d55e00", "#0072b2"

fig, axes = plt.subplots(2, len(PAIRS_ROWS), figsize=(7.2, 5.4), sharey="row")
for lin, (d, col, file, ylabel, log, worst) in enumerate(METRICS_ROWS):
    for c, (already, ga, jb, gb, ra, rb, jt, g1, g2) in enumerate(PAIRS_ROWS):
        ax = axes[lin, c]
        a = d[(d["window"] == already) & (d["group"] == ga)].set_index("Repository")[col]
        b = d[(d["window"] == jb) & (d["group"] == gb)].set_index("Repository")[col]
        pair = pd.concat([a, b], axis=1, join="inner", keys=["a", "b"]).dropna()
        # "worse" for B: slower time, lower acceptance
        b_worst = pair["b"] > pair["a"] if log else pair["b"] < pair["a"]
        a_worst = pair["a"] > pair["b"] if log else pair["a"] < pair["b"]
        for (_, r), bad in zip(pair.iterrows(), a_worst if c < 3 else b_worst):
            ax.plot(
                [0, 1],
                [r["a"], r["b"]],
                color=COLOR_WORST if bad else COLOR_BEST,
                alpha=0.25,
                linewidth=0.6,
            )
        ax.scatter(
            np.zeros(len(pair)), pair["a"], s=5, color=SHADE_2, alpha=0.5, linewidths=0, zorder=3
        )
        ax.scatter(
            np.ones(len(pair)), pair["b"], s=5, color=SHADE_2, alpha=0.5, linewidths=0, zorder=3
        )
        ax.plot(
            [0, 1],
            [pair["a"].median(), pair["b"].median()],
            color=SHADE,
            linewidth=2.2,
            marker="o",
            markersize=5,
            zorder=4,
        )
        # check against the test
        t = pd.read_csv(f"{INPUT_DIR}/{file}_{jt}.csv")
        t = t[(t["group_1"] == g1) & (t["group_2"] == g2)].iloc[0]
        n_a_worst = int(a_worst.sum()) if c < 3 else int(b_worst.sum())
        col_t = "repos_1_greater" if log else "repos_1_less"
        assert n_a_worst == int(t[col_t]) and len(pair) == int(t["n1"]), (file, g1, g2)
        sig = "*" if t["mwu_p_fdr"] < 0.05 else " (n.s.)"
        who = ra if c < 3 else rb
        ax.set_title(
            f"{who.replace(chr(10), ' ')} {worst}\nin {n_a_worst}/{len(pair)} repos{sig}",
            fontsize=7.5,
            color=SHADE_2,
        )
        if log:
            ax.set_yscale("log")
        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([ra, rb], fontsize=7.5)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[lin, 0].set_ylabel(ylabel, color=SHADE_2)
axes[1, 0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
fig.suptitle(
    "Each line is one repository; the thick line joins the medians",
    x=0.01,
    ha="left",
    fontweight="bold",
    fontsize=10,
)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/rq2_paired_repos.{ext}", dpi=200, bbox_inches="tight")
print("  rq2_paired_repos.pdf/.png")

# Matrix of all paired comparisons: cell (row, column) = % of shared
# repositories where the row group does better than the column group (faster,
# higher acceptance). * = p_fdr < 0.05 in the paired Wilcoxon. Ties excluded.
from matplotlib.colors import LinearSegmentedColormap
import os

os.makedirs("figures", exist_ok=True)

MAT_GROUPS = [  # (key, label)
    ("window1:Newcomer", "Newcomer (before)"),
    ("window1:Veteran", "Prior Contr. (before)"),
    ("window2:Newcomer", "Newcomer"),
    ("window2:Newcomer_AI", "Newcomer with AI"),
    ("window2:Veteran", "Prior Contr."),
    ("window2:Veteran_AI", "Prior Contr. with AI"),
    ("window2:AI_Agent", "AI Agent"),
]
MAT_IDX = {k: i for i, (k, _) in enumerate(MAT_GROUPS)}


def _key(block, g):
    if block != "between_windows":
        return f"{block}:{g}"
    g, when = g.rstrip("]").split("[")
    return f"{'window2' if when == 'after' else 'window1'}:{g}"


CMAP = LinearSegmentedColormap.from_list("div", [COLOR_WORST, "#f4f4f2", COLOR_BEST])
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), sharey=True)
for ax, (file, title, col_best, col_worst) in zip(
    axes,
    [
        (
            "rq2_resolution_time_median_per_repo",
            "Resolution time (row faster)",
            "repos_1_less",
            "repos_1_greater",
        ),
        ("rq2_acceptance_rate", "Acceptance rate (row higher)", "repos_1_greater", "repos_1_less"),
    ],
):
    n = len(MAT_GROUPS)
    frac = np.full((n, n), np.nan)
    txt = {}
    for block in ("window1", "window2", "between_windows"):
        for _, r in pd.read_csv(f"{INPUT_DIR}/{file}_{block}.csv").iterrows():
            i, j = MAT_IDX[_key(block, r["group_1"])], MAT_IDX[_key(block, r["group_2"])]
            m, p = int(r[col_best]), int(r[col_worst])
            sig = "*" if r["mwu_p_fdr"] < 0.05 else ""
            frac[i, j], frac[j, i] = m / (m + p), p / (m + p)
            txt[i, j], txt[j, i] = f"{m}/{m + p}{sig}", f"{p}/{m + p}{sig}"
    ax.imshow(np.ma.masked_invalid(frac), cmap=CMAP, vmin=0, vmax=1)
    for (i, j), t in txt.items():
        strong = abs(frac[i, j] - 0.5) > 0.3
        ax.text(j, i, t, ha="center", va="center", fontsize=5.6, color="white" if strong else SHADE)
    for i, j in [(i, j) for i in range(n) for j in range(n) if np.isnan(frac[i, j])]:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#fafafa", zorder=0))
    ax.set_xticks(range(n))
    ax.set_xticklabels([r for _, r in MAT_GROUPS], rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r for _, r in MAT_GROUPS], fontsize=7)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9, fontweight="bold")
fig.suptitle(
    "Repositories where the row group does better than the column group "
    "(blank = not tested; * = significant)",
    x=0.01,
    ha="left",
    fontweight="bold",
    fontsize=9,
)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/rq2_matrix_paired.{ext}", dpi=200, bbox_inches="tight")
print("  rq2_matrix_paired.pdf/.png")

# Time relative to the Prior Contributor of the same repository: each point is
# the ratio between the group's median and the Prior Contributor's median in the
# same repository and window (1 = equal), the same pairing as the Wilcoxon.
REL = [
    ("window1", "Newcomer", "Newcomer", BLUE),
    ("window2", "Newcomer", "Newcomer", BLUE),
    ("window2", "Newcomer_AI", "Newcomer\nwith AI", PURPLE),
    ("window2", "Veteran_AI", "Prior\nContributor\nwith AI", GREEN),
    ("window2", "AI_Agent", "AI Agent", RED),
]
POS_REL = [0, 1.6, 2.6, 3.6, 4.6]
# cropped axis: the few repositories outside it are drawn as triangles on the
# edge (the box uses all of them)
Y_REL = (1 / 64, 1024)
fig, ax = plt.subplots(figsize=(6.8, 3.4))
ax.set_yscale("log", base=2)
n_outside = 0
for k, ((j, g, lbl, color), pos) in enumerate(zip(REL, POS_REL)):
    w = rt[rt["window"] == j].pivot(index="Repository", columns="group", values="median_days")
    v = (w[g] / w["Veteran"]).dropna()
    t = pd.read_csv(f"{INPUT_DIR}/rq2_resolution_time_median_per_repo_{j}.csv")
    t = t[(t["group_1"] == g) & (t["group_2"] == "Veteran")].iloc[0]
    assert int((v > 1).sum()) == int(t["repos_1_greater"]) and len(v) == int(t["n1"]), g
    assert t["mwu_p_fdr"] < 0.05, g
    _box(ax, v, pos, color, log=True)
    jit = np.random.default_rng(k).uniform(-0.2, 0.2, len(v))
    within, bottom, top = (v >= Y_REL[0]) & (v <= Y_REL[1]), v < Y_REL[0], v > Y_REL[1]
    ax.scatter(pos + jit[within], v[within], s=5, color=color, alpha=0.3, linewidths=0)
    for m, lim, sel in (("v", Y_REL[0], bottom), ("^", Y_REL[1], top)):
        ax.scatter(
            pos + jit[sel],
            np.full(sel.sum(), lim),
            s=12,
            marker=m,
            color=color,
            alpha=0.6,
            linewidths=0,
            clip_on=False,
        )
    n_outside += int((bottom | top).sum())
    ax.annotate(
        f"{v.median():.1f}×",
        (pos, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, 11),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        color=color,
        fontweight="bold",
    )
    ax.annotate(
        f"{int((v > 1).sum())}/{len(v)}",
        (pos, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
        color=SHADE_2,
    )
ax.set_ylim(Y_REL)
ax.axhline(1, color=ORANGE, linewidth=1.2)
ax.annotate(
    "Prior Contributor\n= 1×",
    (POS_REL[-1] + 0.45, 1),
    ha="left",
    va="center",
    fontsize=7.5,
    color=ORANGE,
    fontweight="bold",
    annotation_clip=False,
)
ax.set_xticks(POS_REL)
ax.set_xticklabels([lbl for _, _, lbl, _ in REL])
ax.tick_params(axis="x", length=0, pad=4)
ax.axvline(0.8, color="#8a8a85", linewidth=0.8, linestyle=":")
for x, t in ((0, "Before adoption"), (3.1, "After adoption")):
    ax.annotate(
        t,
        (x, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, 26),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        color=SHADE_2,
        fontweight="bold",
    )
ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([8.0**e for e in range(-2, 4)]))
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}×" if y >= 1 else f"1/{1 / y:g}×")
)
ax.grid(axis="y", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.set_ylabel("Median time to merge relative to\nPrior Contributor (per repository)", color=SHADE_2)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/rq2_resolution_relative.{ext}", dpi=200, bbox_inches="tight")
print(f"  rq2_resolution_relative.pdf/.png ({n_outside} repos outside the axis, on the edge)")
