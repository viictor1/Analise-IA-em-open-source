"""RQ1 breakdown figures. Reads the outputs of analysis.py.

breakdowns_size: newcomer growth (log2 after/before) by community size,
    with moving medians.
breakdowns_threshold: the same, split by whether the threshold holds when
    only signals from prior members are used.
breakdowns_language: newcomer change per language (+-6 months).
breakdowns_{size,stars}_box: newcomer growth per community size and stars
    quartile.
breakdowns_threshold_box / breakdowns_stars_threshold_box: growth per
    quartile, split by threshold origin.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.makedirs("figures", exist_ok=True)

INPUT_DIR = "results"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a9a8a2"
SHADE, SHADE_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
MOVING_WINDOW = 41
Y_LIM = (-2.6, 3.6)

r = pd.read_csv(f"{INPUT_DIR}/rq1_before_after_per_repository.csv")
origin = pd.read_csv(f"{INPUT_DIR}/rq1_threshold_origin_per_repository.csv").set_index("Repository")
r["insiders"] = r["Repository"].map(origin["threshold_insiders"]).notna()

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


def _axis_y(ax, label):
    ticks = [-2, -1, 0, 1, 2, 3]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{(2.0**t - 1) * 100:+.0f}%" if t else "0%" for t in ticks])
    ax.set_ylim(*Y_LIM)
    ax.axhline(0, color="#8a8a85", linewidth=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(label, color=SHADE_2)


def _moving_median(x, y):
    d = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    m = d["y"].rolling(MOVING_WINDOW, center=True, min_periods=MOVING_WINDOW).median()
    return d["x"], m


def _axis_size(ax):
    ax.set_xscale("log")
    ax.set_xlabel(
        "Contributors per month before AI adoption (people opening PRs, log scale)", color=SHADE_2
    )


def _points(ax, x, y, color):
    ax.scatter(x, np.clip(y, *Y_LIM), s=14, color=color, alpha=0.35, linewidths=0)


def _save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  {name}.pdf/.png")


# Newcomers by community size
fig, ax = plt.subplots(figsize=(6.4, 3.8))
x = r["baseline_authors_per_month"]
for col, color, lbl in (("log2_newcomers", BLUE, "Newcomers"),):
    _points(ax, x, r[col], color)
    mx, my = _moving_median(x, r[col])
    ax.plot(mx, my, color=color, linewidth=2.2, label=f"{lbl} (moving median)")
_axis_y(ax, "Change in newcomers (per repository)")
_axis_size(ax)
ax.legend(frameon=False, loc="upper right")
ax.set_title("Newcomer entry by number of contributors", loc="left", fontweight="bold", fontsize=10)
_save(fig, "breakdowns_size")

# Newcomers by threshold origin
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for ins, color, lbl in (
    (False, ORANGE, "Threshold set by newcomers' own AI PRs"),
    (True, BLUE, "Threshold holds with prior members' AI PRs"),
):
    g = r[r["insiders"] == ins]
    _points(ax, g["baseline_authors_per_month"], g["log2_newcomers"], color)
    mx, my = _moving_median(g["baseline_authors_per_month"], g["log2_newcomers"])
    ax.plot(mx, my, color=color, linewidth=2.2, label=f"{lbl} ({len(g)})")
_axis_y(ax, "Change in newcomers (per repository)")
_axis_size(ax)
ax.legend(frameon=False, loc="upper right")
ax.set_title(
    "Newcomer entry by number of contributors and adoption origin",
    loc="left",
    fontweight="bold",
    fontsize=10,
)
_save(fig, "breakdowns_threshold")

# By language
MIN_REPOS = 10
X_MAX = 300  # %; the few repositories above it are drawn as triangles on the edge
# newcomer change (%) per repository, 6 months after / 6 months before - 1
per_lang = {
    l: (2 ** g["log2_newcomers_fixed"].dropna() - 1) * 100 for l, g in r.groupby("Language")
}
lang = sorted(per_lang, key=lambda l: per_lang[l].median())
RED = "#d03b2a"
fig, ax = plt.subplots(figsize=(6.8, 5.2))
for i, l in enumerate(lang):
    v = per_lang[l]
    color = BLUE if len(v) >= MIN_REPOS else GRAY
    jitter = np.random.default_rng(i).uniform(-0.22, 0.22, len(v))
    within, fell = (v <= X_MAX) & (v >= 0), v < 0
    ax.scatter(v[within], i + jitter[within], s=10, color=color, alpha=0.3, linewidths=0, zorder=1)
    ax.scatter(v[fell], i + jitter[fell], s=16, color=RED, alpha=0.85, linewidths=0, zorder=3)
    above = v > X_MAX
    ax.scatter(
        np.full(above.sum(), X_MAX),
        i + jitter[above],
        s=16,
        marker=">",
        color=color,
        alpha=0.6,
        linewidths=0,
        zorder=1,
    )
    ax.boxplot(
        v,
        positions=[i],
        orientation="horizontal",
        widths=0.5,
        showfliers=False,
        showcaps=False,
        patch_artist=True,
        zorder=2,
        boxprops=dict(facecolor=color, alpha=0.18, edgecolor="none"),
        medianprops=dict(color=color, linewidth=2.6, solid_capstyle="butt"),
        whiskerprops=dict(linewidth=0),
    )
    ax.text(
        X_MAX + 25,
        i,
        f"{v.median():+.0f}%",
        va="center",
        ha="left",
        fontsize=8.5,
        color=color,
        fontweight="bold",
    )
    ax.text(
        X_MAX + 105,
        i,
        f"{fell.sum()}/{len(v)}",
        va="center",
        ha="left",
        fontsize=8.5,
        color=RED if fell.any() else SHADE_2,
    )
ax.text(X_MAX + 25, len(lang) - 0.35, "median", va="bottom", ha="left", fontsize=7.5, color=SHADE_2)
ax.text(X_MAX + 105, len(lang) - 0.35, "fell", va="bottom", ha="left", fontsize=7.5, color=RED)
ax.set_yticks(range(len(lang)))
ax.set_yticklabels([f"{l} ({len(per_lang[l])})" for l in lang])
ax.set_ylim(-0.6, len(lang) - 0.4)
ticks = [-100, 0, 100, 200, 300]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:+d}%" if t else "0%" for t in ticks])
ax.set_xlim(-110, X_MAX + 15)
ax.axvline(0, color="#8a8a85", linewidth=0.9, zorder=1)
med = (2 ** r["log2_newcomers_fixed"].median() - 1) * 100
ax.axvline(med, color=SHADE_2, linewidth=0.9, linestyle="--", zorder=1)
ax.text(
    med,
    len(lang) - 0.45,
    f" all repos ({med:+.0f}%)",
    color=SHADE_2,
    fontsize=7.5,
    va="bottom",
    ha="left",
)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlabel(
    "Change in newcomers, 6 months after vs. 6 months before AI adoption (per repository)",
    color=SHADE_2,
)
ax.set_title("Newcomer growth by language", loc="left", fontweight="bold", fontsize=10, pad=28)
_v = r["log2_newcomers_fixed"].dropna()
ax.text(
    0,
    1.045,
    f"red: repos that lost newcomers ({(_v < 0).sum()} of {len(_v)}) · gray: < {MIN_REPOS} repos",
    transform=ax.transAxes,
    fontsize=8,
    color=SHADE_2,
)
_save(fig, "breakdowns_language")

# By language: tests and Scott-Knott ESD, only languages with MIN_REPOS
# repositories. Color = SK-ESD rank; * = growth != 0 (Wilcoxon, BH).
test_lang = pd.read_csv(f"{INPUT_DIR}/rq1_fixed_language.csv").set_index("language")
pairs_lang = pd.read_csv(f"{INPUT_DIR}/rq1_fixed_language_pairs.csv")
assert set(test_lang.index) == {l for l, v in per_lang.items() if len(v) >= MIN_REPOS}
COLORS_SK = ["#2a78d6", "#eb6834", "#1b9e77", "#7570b3", "#c9a227", "#d03b8a"]
lang_sk = sorted(test_lang.index, key=lambda l: per_lang[l].median())
fig, ax = plt.subplots(figsize=(6.8, 4.8))
for i, l in enumerate(lang_sk):
    v, t = per_lang[l], test_lang.loc[l]
    color = COLORS_SK[(int(t["scott_knott_esd"]) - 1) % len(COLORS_SK)]
    jitter = np.random.default_rng(i).uniform(-0.22, 0.22, len(v))
    within = v <= X_MAX
    ax.scatter(v[within], i + jitter[within], s=10, color=color, alpha=0.35, linewidths=0, zorder=1)
    ax.scatter(
        np.full((~within).sum(), X_MAX),
        i + jitter[~within],
        s=16,
        marker=">",
        color=color,
        alpha=0.6,
        linewidths=0,
        zorder=1,
    )
    ax.boxplot(
        v,
        positions=[i],
        orientation="horizontal",
        widths=0.5,
        showfliers=False,
        showcaps=False,
        patch_artist=True,
        zorder=2,
        boxprops=dict(facecolor=color, alpha=0.2, edgecolor="none"),
        medianprops=dict(color=color, linewidth=2.6, solid_capstyle="butt"),
        whiskerprops=dict(linewidth=0),
    )
    sig = "*" if t["wilcoxon_p_fdr"] < 0.05 else ""
    ax.text(
        X_MAX + 25,
        i,
        f"{v.median():+.0f}%{sig}",
        va="center",
        ha="left",
        fontsize=8.5,
        color=color,
        fontweight="bold",
    )
    ax.text(
        X_MAX + 135,
        i,
        f"{int(t['scott_knott_esd'])}",
        va="center",
        ha="center",
        fontsize=8.5,
        color=color,
        fontweight="bold",
    )
ax.text(
    X_MAX + 25, len(lang_sk) - 0.35, "median", va="bottom", ha="left", fontsize=7.5, color=SHADE_2
)
ax.text(
    X_MAX + 135, len(lang_sk) - 0.35, "rank", va="bottom", ha="center", fontsize=7.5, color=SHADE_2
)
ax.set_yticks(range(len(lang_sk)))
ax.set_yticklabels([f"{l} ({len(per_lang[l])})" for l in lang_sk])
ax.set_ylim(-0.6, len(lang_sk) - 0.4)
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:+d}%" if t else "0%" for t in ticks])
ax.set_xlim(-110, X_MAX + 15)
ax.axvline(0, color="#8a8a85", linewidth=0.9, zorder=1)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlabel(
    "Change in newcomers, 6 months after vs. 6 months before AI adoption (per repository)",
    color=SHADE_2,
)
ax.set_title(
    "Newcomer growth by language: Scott-Knott ESD",
    loc="left",
    fontweight="bold",
    fontsize=10,
    pad=38,
)
_kw = test_lang.iloc[0]
ax.text(
    0,
    1.085,
    "color = SK-ESD rank · * growth ≠ 0 (Wilcoxon, BH-adjusted p < 0.05)",
    transform=ax.transAxes,
    fontsize=7.5,
    color=SHADE_2,
)
ax.text(
    0,
    1.045,
    f"between languages: Kruskal-Wallis p = {_kw['kruskal_p']:.2f} · pairwise Mann-Whitney "
    f"significant after BH: {int(pairs_lang['significant'].sum())} of {len(pairs_lang)}",
    transform=ax.transAxes,
    fontsize=7.5,
    color=SHADE_2,
)
_save(fig, "breakdowns_language_sk")


# Boxplots per community size and stars quartile
QUARTILES = ["Q1\n(smallest)", "Q2", "Q3", "Q4\n(largest)"]
kw = pd.read_csv(f"{INPUT_DIR}/rq1_before_after_kruskal.csv").set_index(["breakdown", "variable"])


def _boxes(ax, d, groups, colors, labels, col_cat, col_val, labels_x=QUARTILES):
    """Boxplot + points per quartile, one box per group. Points outside Y_LIM are
    not drawn (the box uses all of them).
    """
    width = 0.36 if len(groups) > 1 else 0.5
    for k, (selection, color, lbl) in enumerate(zip(groups, colors, labels)):
        for i, q in enumerate(["Q1", "Q2", "Q3", "Q4"]):
            v = (
                d.loc[selection & (d[col_cat] == q), col_val]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            pos = i + (k - (len(groups) - 1) / 2) * width
            ax.boxplot(
                v,
                positions=[pos],
                widths=width * 0.8,
                showfliers=False,
                showcaps=False,
                patch_artist=True,
                boxprops=dict(facecolor="none", edgecolor=color, linewidth=1.1),
                medianprops=dict(color=color, linewidth=2.2),
                whiskerprops=dict(color=color, linewidth=0.8),
            )
            jit = np.random.default_rng(10 * i + k).uniform(-width * 0.3, width * 0.3, len(v))
            ax.scatter(pos + jit, v, s=7, color=color, alpha=0.25, linewidths=0)
            ax.text(
                pos,
                Y_LIM[1] - 0.15,
                f"{2 ** v.median() - 1:+.0%}",
                ha="center",
                va="top",
                fontsize=8,
                color=color,
                fontweight="bold",
            )
        ax.plot([], [], color=color, linewidth=2.2, label=lbl)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels_x)
    ax.tick_params(axis="x", length=0)


def _xlabel(ax, label, breakdown, variable):
    p = kw.loc[(breakdown, variable), "p"]
    p = f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
    ax.set_xlabel(f"{label}   ·   Kruskal-Wallis on newcomer growth, p = {p}", color=SHADE_2)


def _header(ax, title):
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10, pad=22)
    if len(ax.get_legend_handles_labels()[0]) > 1:
        ax.legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(-0.01, 1.0),
            ncols=2,
            fontsize=8,
            handlelength=1.4,
            borderaxespad=0.2,
        )


r_long = pd.concat(
    [
        r.assign(group="newc", value=r["log2_newcomers"]),
        r.assign(group="vet", value=r["log2_veterans"]),
    ],
    ignore_index=True,
)
NEWC_VET = dict(
    groups=[r_long["group"] == "newc", r_long["group"] == "vet"],
    colors=[BLUE, ORANGE],
    labels=["Newcomers", "Veterans"],
)
PURPLE, GREEN = "#8a4fc9", "#2a9d6f"
_did = r["log2_newcomers"].notna()
ORIGIN = dict(
    groups=[~r["insiders"], r["insiders"]],
    colors=[PURPLE, GREEN],
    labels=[
        f"Adoption dated by newcomers' AI PRs ({(~r['insiders'] & _did).sum()})",
        f"Confirmed by prior members ({(r['insiders'] & _did).sum()})",
    ],
)
# star ranges from the cut points between quartiles
c1, c2, c3 = (
    f"{r.loc[r['tier_stars'] == q, 'stars'].min() / 1000:.1f}" for q in ("Q2", "Q3", "Q4")
)
t1, t2, t3 = (
    f"{r.loc[r['tier_size'] == q, 'baseline_authors_per_month'].min():.0f}"
    for q in ("Q2", "Q3", "Q4")
)
LBL_SIZE = [f"Q1\n<{t1}", f"Q2\n{t1}–{t2}", f"Q3\n{t2}–{t3}", f"Q4\n≥{t3}"]
LBL_STARS = [f"Q1\n<{c1}k", f"Q2\n{c1}–{c2}k", f"Q3\n{c2}–{c3}k", f"Q4\n≥{c3}k"]

for tier, label_x, labels_x, suffix in (
    ("size", "Contributors per month (quartile)", LBL_SIZE, "size"),
    ("stars", "GitHub stars (quartile)", LBL_STARS, "stars"),
):
    col_cat = f"tier_{tier}"
    name_t = "contributor" if tier == "size" else "star"

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    _boxes(
        ax,
        r,
        groups=[r["log2_newcomers"].notna()],
        colors=[BLUE],
        labels=["Newcomers"],
        col_cat=col_cat,
        col_val="log2_newcomers",
        labels_x=labels_x,
    )
    _axis_y(ax, "Change in newcomers (per repository)")
    _xlabel(ax, label_x, tier, "level_change_newcomers")
    _header(
        ax,
        f"Newcomer entry by {name_t} quartile"
        if tier == "stars"
        else "Newcomer entry by number of contributors",
    )
    _save(fig, f"breakdowns_{suffix}_box")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    _boxes(ax, r, col_cat=col_cat, col_val="log2_newcomers", labels_x=labels_x, **ORIGIN)
    _axis_y(ax, "Change in newcomers (per repository)")
    ax.set_xlabel(label_x, color=SHADE_2)
    _header(ax, "Newcomer entry by adoption origin")
    _save(fig, "breakdowns_threshold_box" if tier == "size" else "breakdowns_stars_threshold_box")
