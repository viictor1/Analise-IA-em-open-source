"""RQ2 breakdown figures. Reads the outputs of analysis.py.

rq2_breakdowns_{stars,size}_acceptance: acceptance rate per repository in
    each quartile, for the 2 groups of window 1 and the 4 human groups of
    window 2 (AI Agent is left out, as in the breakdown tests).
rq2_breakdowns_{stars,size}_resolution: the same with the median days to
    merge per repository (log axis).
rq2_breakdowns_language: change in acceptance per repository (after - before)
    of newcomers and veterans without AI, per language.
Quartiles are the RQ1 ones and the Kruskal-Wallis p across categories comes
from rq2_breakdowns_kruskal_between_categories.csv; the script checks that
the number of repositories drawn matches the test.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.makedirs("figures", exist_ok=True)

INPUT_DIR = "results"
MIN_N_PER_REPO = 5
MIN_REPOS_BREAKDOWN = 10  # same minimum as the Kruskal-Wallis between categories
BLUE, ORANGE, PURPLE, GREEN = "#0072b2", "#e69f00", "#cc79a7", "#009e73"
GRAY = "#a9a8a2"
SHADE, SHADE_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

# (window, CSV group, label, color, offset within the quartile)
GROUPS = [
    ("window1", "Newcomer", "Newcomer", BLUE, -0.66),
    ("window1", "Veteran", "Prior Contributor", ORANGE, -0.46),
    ("window2", "Newcomer", "Newcomer", BLUE, -0.14),
    ("window2", "Veteran", "Prior Contributor", ORANGE, 0.06),
    ("window2", "Newcomer_AI", "Newcomer with AI", PURPLE, 0.26),
    ("window2", "Veteran_AI", "Prior Contributor with AI", GREEN, 0.46),
]
WIDTH = 0.16
STEP = 1.6
QS = ["Q1", "Q2", "Q3", "Q4"]

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
    """Boxplot without fliers, as in plot_rq2.py (whiskers computed on the log scale)."""
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
        widths=WIDTH * 0.85,
        showfliers=False,
        showcaps=False,
        patch_artist=True,
        boxprops=dict(facecolor="none", edgecolor=color, linewidth=1.0),
        medianprops=dict(color=color, linewidth=2.0),
        whiskerprops=dict(color=color, linewidth=0.7),
    )


def _save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  {name}.pdf/.png")


def _fmt_p(p):
    if pd.isna(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else (f"= {p:.3f}" if p < 0.01 else f"= {p:.2f}")


r = pd.read_csv(f"{INPUT_DIR}/rq1_before_after_per_repository.csv")
tiers = r.set_index("Repository")[["tier_stars", "tier_size", "Language"]]
kw = pd.read_csv(f"{INPUT_DIR}/rq2_breakdowns_kruskal_between_categories.csv")

ac = pd.read_csv(f"{INPUT_DIR}/rq2_acceptance_rate_per_repository.csv")
rt = pd.read_csv(f"{INPUT_DIR}/rq2_resolution_time_per_repository_descriptive.csv")
rt = rt[rt["n"] >= MIN_N_PER_REPO]
assert ac["Repository"].isin(tiers.index).all() and rt["Repository"].isin(tiers.index).all()
ac = ac.drop(columns="Language").join(tiers, on="Repository")
rt = rt.join(tiers, on="Repository")

c1, c2, c3 = (
    f"{r.loc[r['tier_stars'] == q, 'stars'].min() / 1000:.1f}" for q in ("Q2", "Q3", "Q4")
)
t1, t2, t3 = (
    f"{r.loc[r['tier_size'] == q, 'baseline_authors_per_month'].min():.0f}"
    for q in ("Q2", "Q3", "Q4")
)
LBL = {
    "stars": [f"Q1\n<{c1}k", f"Q2\n{c1}–{c2}k", f"Q3\n{c2}–{c3}k", f"Q4\n≥{c3}k"],
    "size": [f"Q1\n<{t1}", f"Q2\n{t1}–{t2}", f"Q3\n{t2}–{t3}", f"Q4\n≥{t3}"],
}
AXIS_X = {
    "stars": "GitHub stars (quartile)",
    "size": "Contributors per month before AI adoption (quartile)",
}
METRICS = [
    (
        ac,
        "rate",
        "acceptance_rate_per_repo",
        "Acceptance rate (per repository)",
        False,
        lambda m: f"{m:.0%}",
        "acceptance",
        "Acceptance rate by contribution group",
    ),
    (
        rt,
        "median_days",
        "resolution_time_median_per_repo",
        "Median days to merge (per repository, log)",
        True,
        lambda m: f"{m:.1f}d",
        "resolution",
        "Resolution time of merged PRs by contribution group",
    ),
]


for breakdown in ("stars", "size"):
    col_cat = f"tier_{breakdown}"
    for d, col, met_kw, ylabel, log, fmt, suffix, title in METRICS:
        fig, ax = plt.subplots(figsize=(9.2, 4.2))
        if log:
            ax.set_yscale("log")
        footer = []
        for k, (j, g, lbl, color, offset) in enumerate(GROUPS):
            n_kw = 0
            for i, q in enumerate(QS):
                v = d.loc[(d["window"] == j) & (d["group"] == g) & (d[col_cat] == q), col].dropna()
                pos = i * STEP + offset
                _box(ax, v, pos, color, log=log)
                jit = np.random.default_rng(10 * i + k).uniform(-WIDTH * 0.3, WIDTH * 0.3, len(v))
                ax.scatter(pos + jit, v, s=5, color=color, alpha=0.3, linewidths=0)
                ax.annotate(
                    fmt(v.median()),
                    (pos, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 2),
                    textcoords="offset points",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=SHADE_2,
                )
                n_kw += len(v) if len(v) >= MIN_REPOS_BREAKDOWN else 0
            # check against the test between categories
            t = kw[
                (kw["breakdown"] == breakdown)
                & (kw["metric"] == met_kw)
                & (kw["window"] == j)
                & (kw["group"] == g)
            ].iloc[0]
            assert n_kw == int(t["repositories"]), (
                breakdown,
                met_kw,
                j,
                g,
                n_kw,
                t["repositories"],
            )
            when = "before" if j == "window1" else "after"
            footer.append(f"{lbl} ({when}) p {_fmt_p(t['p_fdr'])}")
        # gray band behind the 2 groups of window 1 in each quartile
        for i in range(4):
            ax.axvspan(i * STEP - 0.78, i * STEP - 0.34, color="#f1f0eb", zorder=0, linewidth=0)
        ax.annotate(
            "before",
            (-0.56, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 26),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=SHADE_2,
        )
        ax.annotate(
            "after",
            (0.16, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 26),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=SHADE_2,
        )
        ax.set_xticks([i * STEP for i in range(4)])
        ax.set_xticklabels(LBL[breakdown])
        ax.set_xlim(-0.95, 3 * STEP + 0.65)
        ax.tick_params(axis="x", length=0)
        if log:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        else:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylabel(ylabel, color=SHADE_2)
        ax.set_xlabel(AXIS_X[breakdown], color=SHADE_2)
        for _, _, lbl, color, _ in GROUPS[2:]:
            ax.plot([], [], color=color, linewidth=2.2, label=lbl)
        ax.legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.32, 1.09),
            ncols=4,
            fontsize=7.5,
            handlelength=1.4,
        )
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10, pad=44)
        fig.tight_layout()
        fig.text(
            0.01,
            -0.01,
            "Kruskal-Wallis across quartiles (FDR): "
            + "; ".join(footer[:3])
            + ";\n"
            + "; ".join(footer[3:]),
            ha="left",
            va="top",
            fontsize=6.5,
            color=SHADE_2,
        )
        _save(fig, f"rq2_breakdowns_{breakdown}_{suffix}")


# By language: change in acceptance (after - before)
w = ac.pivot_table(index=["Repository", "Language"], columns=["group", "window"], values="rate")
var = {
    g: ((w[(g, "window2")] - w[(g, "window1")]) * 100).dropna().reset_index(level=1)
    for g in ("Newcomer", "Veteran")
}
SERIES = [("Newcomer", "Newcomer", BLUE, 0.17), ("Veteran", "Prior Contributor", ORANGE, -0.17)]
for g, *_ in SERIES:
    n = var[g].groupby("Language").size()
    t = kw[
        (kw["breakdown"] == "language")
        & (kw["metric"] == "var_acceptance_rate_per_repo")
        & (kw["group"] == g)
    ].iloc[0]
    assert n[n >= MIN_REPOS_BREAKDOWN].sum() == int(t["repositories"]), (g, n.to_dict())

lang = sorted(
    var["Newcomer"]["Language"].unique(),
    key=lambda l: var["Newcomer"].loc[var["Newcomer"]["Language"] == l, 0].median(),
)
X_LIM = (-60, 60)
fig, ax = plt.subplots(figsize=(6.8, 6.0))
for i, l in enumerate(lang):
    for k, (g, lbl, color, offset) in enumerate(SERIES):
        v = var[g].loc[var[g]["Language"] == l, 0]
        if v.empty:
            continue
        c = color if len(v) >= MIN_REPOS_BREAKDOWN else GRAY
        y = i + offset
        jit = np.random.default_rng(10 * i + k).uniform(-0.1, 0.1, len(v))
        ax.scatter(np.clip(v, *X_LIM), y + jit, s=8, color=c, alpha=0.35, linewidths=0, zorder=1)
        ax.boxplot(
            v,
            positions=[y],
            orientation="horizontal",
            widths=0.26,
            showfliers=False,
            showcaps=False,
            patch_artist=True,
            zorder=2,
            boxprops=dict(facecolor=c, alpha=0.18, edgecolor="none"),
            medianprops=dict(color=c, linewidth=2.4, solid_capstyle="butt"),
            whiskerprops=dict(linewidth=0),
        )
        ax.text(
            X_LIM[1] + 8 + 32 * k,
            i,
            (f"{v.median():+.0f}" if round(v.median()) else "0"),
            va="center",
            ha="left",
            fontsize=8,
            color=SHADE_2 if len(v) >= MIN_REPOS_BREAKDOWN else GRAY,
            fontweight="bold",
        )
n_newc = var["Newcomer"].groupby("Language").size()
n_vet = var["Veteran"].groupby("Language").size()
ax.set_yticks(range(len(lang)))
ax.set_yticklabels([f"{l} ({n_newc.get(l, 0)}/{n_vet.get(l, 0)})" for l in lang])
ax.set_ylim(-0.6, len(lang) - 0.3)
for k, (_, lbl, color, _) in enumerate(SERIES):
    ax.text(
        X_LIM[1] + 8 + 32 * k,
        len(lang) - 0.45,
        lbl.replace(" ", "\n"),
        va="bottom",
        ha="left",
        fontsize=7,
        color=color,
        fontweight="bold",
    )
ax.set_xlim(X_LIM[0] - 5, X_LIM[1] + 5)
ax.set_xticks([-60, -30, 0, 30, 60])
ax.set_xticklabels(["≤−60", "−30", "0", "+30", "≥+60"])
ax.axvline(0, color="#8a8a85", linewidth=0.9, zorder=1)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlabel(
    "Change in acceptance rate, after − before AI adoption (p.p., per repository)", color=SHADE_2
)
ax.set_title(
    "Acceptance change by language (PRs without AI)",
    loc="left",
    fontweight="bold",
    fontsize=10,
    pad=30,
)
_p = {
    g: kw[
        (kw["breakdown"] == "language")
        & (kw["metric"] == "var_acceptance_rate_per_repo")
        & (kw["group"] == g)
    ].iloc[0]["p_fdr"]
    for g, *_ in SERIES
}
ax.text(
    0,
    1.03,
    f"median in p.p. on the right · (n) = repos newcomer/prior contributor · "
    f"gray: < {MIN_REPOS_BREAKDOWN} repos\nKruskal-Wallis across languages (FDR): "
    f"newcomer p {_fmt_p(_p['Newcomer'])}, prior contributor p {_fmt_p(_p['Veteran'])}",
    transform=ax.transAxes,
    fontsize=7,
    color=SHADE_2,
)
_save(fig, "rq2_breakdowns_language")


# Breakdown summary: for each key RQ2 comparison and breakdown, how many
# categories go in the direction of the global result (median paired
# difference < 0) and how many are significant in that direction (BH < 0.05).
_brk = pd.read_csv("results/rq2_breakdowns.csv")
_KEY = [
    (
        "acceptance_rate_per_repo",
        "between_windows",
        "Newcomer[after]",
        "Newcomer[before]",
        "acceptance_newcomer_after_vs_before",
    ),
    (
        "acceptance_rate_per_repo",
        "between_windows",
        "Veteran[after]",
        "Veteran[before]",
        "acceptance_prior_after_vs_before",
    ),
    (
        "acceptance_rate_per_repo",
        "window2",
        "Newcomer_AI",
        "Newcomer",
        "acceptance_newcomer_ai_vs_no_ai",
    ),
    (
        "resolution_time_median_per_repo",
        "window2",
        "Newcomer_AI",
        "Newcomer",
        "time_newcomer_ai_vs_no_ai",
    ),
]
_rows = []
for base, block, g1, g2, name in _KEY:
    for brk, d in _brk[
        (_brk.metric_base == base)
        & (_brk.block == block)
        & (_brk.group_1 == g1)
        & (_brk.group_2 == g2)
    ].groupby("breakdown"):
        _neg = d.paired_median_diff < 0
        _rows.append(
            {
                "comparison": name,
                "breakdown": brk,
                "categories": len(d),
                "difference_negative": int(_neg.sum()),
                "significant_negative": int((_neg & (d.mwu_p_fdr < 0.05)).sum()),
                "difference_positive": int((d.paired_median_diff > 0).sum()),
                "significant_positive": int(
                    ((d.paired_median_diff > 0) & (d.mwu_p_fdr < 0.05)).sum()
                ),
            }
        )
pd.DataFrame(_rows).to_csv("results/rq2_breakdowns_summary.csv", index=False)
