"""Staggered difference-in-differences for RQ1 and RQ2 with the imputation
estimator of Borusyak, Jaravel and Spiess (BJS). Runs after analysis.py,
from which it reads the thresholds, windows and breakdowns.

Model. Repository and calendar-month fixed effects are fitted on
pre-adoption observations only (repositories that have not adopted yet act
as controls). In each repository-month, y0 = repo FE + month FE is the
counterfactual without adoption. The effect is the mean of y - y0 over
months 0..+5 (and month 0 alone), up to 2026-01, after which no repository
is left without adoption to act as a control. 95% CIs come from a bootstrap
over repositories, refitting the fixed effects in each replicate.

Placebo (pre-trend). The same model with a fake adoption at month -3: the
fixed effects use only the months before -3 and the "effect" is the mean of
y - y0 over months -3..-1, all pre-adoption. Values near zero support
parallel trends.

Panels:
- RQ1: log1p of monthly newcomers (rq1_newcomers_windows_per_repository.csv),
  in the 236 repositories with a threshold.
- RQ2: the Newcomer and Veteran (Prior Contributor in the paper) groups,
  PRs without AI only, for acceptance (merged / closed) and log1p of the
  median resolution time of merged PRs, per repository and creation month,
  without the 17 repositories that merge outside GitHub. A repository-month
  needs at least MIN_PRS PRs of the group.

Outputs in results/:
- did_bjs.csv: effect and placebo of each panel.
- did_bjs_breakdowns*.csv: the RQ1 effect per language, stars quartile and
  community size quartile (categories with at least MIN_REPOS
  repositories), with Kruskal-Wallis on the per-repository mean effect.
- did_sawtooth.csv: share of debuts among Newcomer PRs per relative month.
Figures: did_bjs_rq1, did_bjs_rq2 and did_bjs_breakdowns (PDF and PNG).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal

os.makedirs("figures", exist_ok=True)

CUTOFF_MONTH, LAST_CAL, START = "2026-08", "2026-01", "2022-01"
MIN_PRS = {"Newcomer": 3, "Veteran": 5}
H_BEFORE, H_AFTER = 12, 6  # figure months (the effect uses only 0..H_AFTER-1)
B, B_BREAKDOWNS, MIN_REPOS = 200, 300, 10
PLACEBO = -3  # fake adoption: the placebo estimates the "effect" over PLACEBO..-1
BREAKDOWNS = {"language": "Language", "stars": "tier_stars", "size": "tier_size"}
OUTPUT_DIR = "results"

# data
COLS = ["Repository", "PR_Number", "Created_At", "Closed_At", "State", "PR_Author"]
hum = pd.read_csv("data/human_prs.csv", usecols=COLS, low_memory=False)
hum = hum[
    hum.PR_Author.fillna("").str.strip() != ""
]  # deleted accounts are dropped from the control
ai = pd.read_csv("data/ai_prs.csv", usecols=COLS + ["PR_Author_Category"], low_memory=False)

thr = pd.read_csv(f"{OUTPUT_DIR}/rq1_threshold_origin_per_repository.csv").set_index("Repository")[
    "threshold"
]
j1 = pd.read_csv(f"{OUTPUT_DIR}/rq1_newcomers_windows_per_repository.csv")
n_months = j1[j1.window == "after"].groupby("Repository").relative_month.max() + 1
outside = set(pd.read_csv(f"{OUTPUT_DIR}/rq2_repos_external_merge.csv").repo)
repos = [r for r in thr.index if r not in outside]
brk = pd.read_csv(f"{OUTPUT_DIR}/rq1_before_after_per_repository.csv").set_index("Repository")

d = pd.concat([hum.assign(PR_Author_Category="Human"), ai]).rename(
    columns={
        "Repository": "Repository",
        "PR_Number": "PR",
        "Created_At": "Created_At",
        "Closed_At": "Closed",
        "State": "State",
        "PR_Author": "Author",
        "PR_Author_Category": "Cat",
    }
)
d["Created_At"] = pd.to_datetime(d.Created_At, utc=True, errors="coerce")
d["Closed"] = pd.to_datetime(d.Closed, utc=True, errors="coerce")
d["Month"] = d.Created_At.dt.strftime("%Y-%m")
d["State"] = d.State.str.lower().str.strip()
d = d[(d.Month >= START) & (d.Month < CUTOFF_MONTH) & d.Repository.isin(repos)]
p_thr = d.Repository.map(thr).map(lambda t: pd.Period(t, "M"))
d["relative_month"] = [(pd.Period(m, "M") - t).n for m, t in zip(d.Month, p_thr)]
d = d[d.relative_month >= -d.Repository.map(n_months)]
d["days"] = np.where(
    d.State == "merged", (d.Closed - d.Created_At).dt.total_seconds() / 86400, np.nan
)
_ai = set(zip(ai.Repository, ai.PR_Number))
d_no_ai = d[[(r, p) not in _ai for r, p in zip(d.Repository, d.PR)]]

# RQ2 groups (PRs without AI only), with the same rule as analysis.py: the
# newcomer of a window is a human author whose first PR (with or without AI)
# since the start of the baseline falls in it. Newcomer = that author's PRs
# without AI in their window; Veteran = everyone else.
a = pd.concat([hum, ai[ai.PR_Author_Category == "Human"]])[
    ["Repository", "PR_Author", "Created_At"]
]
a = a[a.PR_Author.fillna("").str.strip() != ""]
a["Created_At"] = pd.to_datetime(a.Created_At, utc=True, errors="coerce")
a["Month"] = a.Created_At.dt.strftime("%Y-%m")
a = a[a.Repository.isin(repos) & (a.Month < CUTOFF_MONTH)]
start = {
    r: str(max(pd.Period(START, "M"), pd.Period(thr[r], "M") - n_months[r] - 24)) for r in repos
}
a = a[a.Month >= a.Repository.map(start)]
first = a.groupby(["Repository", "PR_Author"]).Month.min().reset_index()
first["rel"] = [
    (pd.Period(m, "M") - pd.Period(thr[r], "M")).n for r, m in zip(first.Repository, first.Month)
]
first = first[first.rel >= -first.Repository.map(n_months)]
win_newc = {(r, u): rel >= 0 for r, u, rel in zip(first.Repository, first.PR_Author, first.rel)}
is_newc = [
    win_newc.get((r, u)) == (t >= 0)
    for r, u, t in zip(d_no_ai.Repository, d_no_ai.Author, d_no_ai.relative_month)
]
d_newc, d_vet = d_no_ai[is_newc], d_no_ai[[not e for e in is_newc]]


def panel(x, metric, min_n):
    """One value per repository-month: log1p of the median time of merged PRs, or
    the acceptance (merged / closed), with at least min_n PRs.
    """
    if metric == "resolution_time":
        m = (
            x[x.days >= 0]
            .groupby(["Repository", "Month", "relative_month"])
            .days.agg(["median", "size"])
        )
        m = m[m["size"] >= min_n]
        y = np.log1p(m["median"])
    else:
        f = (
            x[x.State.isin(["merged", "closed"])]
            .groupby(["Repository", "Month", "relative_month"])
            .State
        )
        m = f.agg(n="size", merged=lambda e: (e == "merged").sum())
        m = m[m["n"] >= min_n]
        y = m["merged"] / m["n"]
    return y.rename("y").reset_index()


# (rq, group, metric) -> panel; log scale except for acceptance
PANELS = {
    ("RQ1", "newcomers", "newcomers_per_month"): j1.assign(y=np.log1p(j1.newcomers))[
        ["Repository", "Month", "relative_month", "y"]
    ],
    ("RQ2", "Newcomer", "acceptance"): panel(d_newc, "acceptance", MIN_PRS["Newcomer"]),
    ("RQ2", "Newcomer", "resolution_time"): panel(d_newc, "resolution_time", MIN_PRS["Newcomer"]),
    ("RQ2", "Veteran", "acceptance"): panel(d_vet, "acceptance", MIN_PRS["Veteran"]),
    ("RQ2", "Veteran", "resolution_time"): panel(d_vet, "resolution_time", MIN_PRS["Veteran"]),
}


# model
def y0(p, cutoff=0):
    """Counterfactual y0 (repo FE + month FE, fitted on the months before cutoff)
    in each repository-month of the figure window. cutoff = 0 is the model;
    cutoff = PLACEBO fakes the adoption a few months earlier.
    """
    pre = p[p.relative_month < cutoff]
    rs, ms = sorted(pre.Repository.unique()), sorted(pre.Month.unique())
    ri, mi = {r: i for i, r in enumerate(rs)}, {m: i for i, m in enumerate(ms)}
    X = np.zeros((len(pre), len(rs) + len(ms) - 1))
    X[np.arange(len(pre)), pre.Repository.map(ri).values] = 1
    c = pre.Month.map(mi).values
    X[np.arange(len(pre))[c > 0], len(rs) + c[c > 0] - 1] = 1
    b, *_ = np.linalg.lstsq(X, pre.y.values, rcond=None)
    fr, fc = dict(zip(rs, b[: len(rs)])), dict(zip(ms, np.r_[0, b[len(rs) :]]))
    w = p[
        p.relative_month.between(-H_BEFORE, H_AFTER - 1)
        & (p.Month <= LAST_CAL)
        & p.Month.isin(ms)
        & p.Repository.isin(rs)
    ].copy()
    w["y0"] = w.Repository.map(fr) + w.Month.map(fc)
    return w


def effect(w):
    """Mean of y - y0 over months 0..+5 and in month 0 alone."""
    pos = w[w.relative_month >= 0]
    eff = pos.y - pos.y0
    return eff.mean(), eff[pos.relative_month == 0].mean()


def placebo(w):
    """Mean of y - y0 over months PLACEBO..-1. Uses only pre-adoption months, so a
    value near zero means no pre-trend.
    """
    pla = w[w.relative_month.between(PLACEBO, -1)]
    return (pla.y - pla.y0).mean()


def resamples(p, n, rng):
    """Repository bootstrap: each sampled repository enters as a new one."""
    groups = {r: g for r, g in p.groupby("Repository")}
    names = np.array(list(groups))
    for _ in range(n):
        s = rng.choice(names, len(names), replace=True)
        yield pd.concat([groups[r].assign(Repository=f"{r}#{k}") for k, r in enumerate(s)])


def fmt(v, scale):
    return f"{np.expm1(v) * 100:+.1f}%" if scale == "log" else f"{v * 100:+.1f} pp"


rng = np.random.default_rng(42)
rows, WINDOWS = [], {}
for (rq, group, metric), p in PANELS.items():
    scale = "pp" if metric == "acceptance" else "log"
    w, wp = y0(p), y0(p, PLACEBO)
    eff, eff0 = effect(w)
    pl = placebo(wp)
    boot = np.array([(*effect(y0(q)), placebo(y0(q, PLACEBO))) for q in resamples(p, B, rng)])
    (lo, lo0, lop), (hi, hi0, hip) = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    rows.append(
        {
            "rq": rq,
            "group": group,
            "metric": metric,
            "scale": scale,
            "repos": w.Repository.nunique(),
            "obs": len(w),
            "effect_0a5": eff,
            "ci_inf": lo,
            "ci_sup": hi,
            "effect_month0": eff0,
            "ci_month0_inf": lo0,
            "ci_month0_sup": hi0,
            "text_0a5": f"{fmt(eff, scale)} [{fmt(lo, scale)}, {fmt(hi, scale)}]",
            "text_month0": f"{fmt(eff0, scale)} [{fmt(lo0, scale)}, {fmt(hi0, scale)}]",
            "placebo": pl,
            "ci_placebo_inf": lop,
            "ci_placebo_sup": hip,
            "obs_placebo": int(wp.relative_month.between(PLACEBO, -1).sum()),
            "text_placebo": f"{fmt(pl, scale)} [{fmt(lop, scale)}, {fmt(hip, scale)}]",
        }
    )
    WINDOWS[(rq, group, metric)] = w
    print(
        f"{rq} {group} {metric}: 0..+5 {rows[-1]['text_0a5']}  month 0 {rows[-1]['text_month0']}  "
        f"placebo {rows[-1]['text_placebo']}  {w.Repository.nunique()} repos"
    )
did = pd.DataFrame(rows)
did.to_csv(f"{OUTPUT_DIR}/did_bjs.csv", index=False)

# RQ1 breakdowns
p1 = PANELS[("RQ1", "newcomers", "newcomers_per_month")]
est = WINDOWS[("RQ1", "newcomers", "newcomers_per_month")]
est = est[est.relative_month >= 0].assign(eff=lambda e: e.y - e.y0)
per_repo = (
    est.groupby("Repository")
    .eff.mean()
    .rename("effect_0a5")
    .to_frame()
    .join(est[est.relative_month == 0].groupby("Repository").eff.mean().rename("effect_month0"))
)
per_repo = per_repo.join(brk[list(BREAKDOWNS.values())])
CATS = {
    name: [c for c, n in per_repo[col].value_counts().sort_index().items() if n >= MIN_REPOS]
    for name, col in BREAKDOWNS.items()
}


def summary(e, col, c):
    """Aggregate effect of the category (mean over repository-months), 0..+5 and month 0."""
    e = e[e.orig.map(brk[col]) == c]
    return e.eff.mean(), e.loc[e.relative_month == 0, "eff"].mean()


est = est.assign(orig=est.Repository)
rng_brk = np.random.default_rng(42)
boot = {(name, c): [] for name, cats in CATS.items() for c in cats}
for q in resamples(p1, B_BREAKDOWNS, rng_brk):
    e = y0(q)
    e = e[e.relative_month >= 0].assign(
        eff=lambda x: x.y - x.y0, orig=lambda x: x.Repository.str.split("#").str[0]
    )
    for name, cats in CATS.items():
        for c in cats:
            boot[(name, c)].append(summary(e, BREAKDOWNS[name], c))

pct = lambda v: np.expm1(v) * 100
rows, kw = [], []
for name, col in BREAKDOWNS.items():
    for c in CATS[name]:
        m, m0 = summary(est, col, c)
        (lo, lo0), (hi, hi0) = np.nanpercentile(np.array(boot[(name, c)]), [2.5, 97.5], axis=0)
        rows.append(
            {
                "breakdown": name,
                "category": c,
                "repositories": int((per_repo[col] == c).sum()),
                "effect_0a5_pct": pct(m),
                "ci_0a5_inf": pct(lo),
                "ci_0a5_sup": pct(hi),
                "effect_month0_pct": pct(m0),
                "ci_month0_inf": pct(lo0),
                "ci_month0_sup": pct(hi0),
            }
        )
    for measure, column in (("months_0a5", "effect_0a5"), ("month_0", "effect_month0")):
        samples = [per_repo.loc[per_repo[col] == c, column].dropna() for c in CATS[name]]
        h, p = kruskal(*samples)
        kw.append(
            {
                "breakdown": name,
                "measure": measure,
                "categories": len(CATS[name]),
                "repositories": sum(len(s) for s in samples),
                "H": h,
                "p": p,
            }
        )
brk_out, kwd = pd.DataFrame(rows), pd.DataFrame(kw)
# BH over the 6 Kruskal-Wallis tests (3 breakdowns x months 0..+5 and month 0)
_o = kwd["p"].rank(method="first").astype(int)
_adj = (kwd["p"] * len(kwd) / _o).to_numpy()
_ord = np.argsort(kwd["p"].to_numpy())
_adj[_ord] = np.minimum.accumulate(_adj[_ord][::-1])[::-1]
kwd["p_fdr"] = np.minimum(_adj, 1)
brk_out.to_csv(f"{OUTPUT_DIR}/did_bjs_breakdowns.csv", index=False)
kwd.to_csv(f"{OUTPUT_DIR}/did_bjs_breakdowns_kruskal.csv", index=False)
per_repo.to_csv(f"{OUTPUT_DIR}/did_bjs_breakdowns_per_repository.csv")
print(brk_out.round(1).to_string(index=False))
print(kwd.round(4).to_string(index=False))

# sawtooth (Threats to Validity)
pri = a.groupby(["Repository", "PR_Author"]).Created_At.min()
x = d_newc.assign(
    debut=[
        pri.get((r, u)) == t for r, u, t in zip(d_newc.Repository, d_newc.Author, d_newc.Created_At)
    ]
)
s = (
    x[x.relative_month.between(-12, 11)]
    .groupby("relative_month")
    .agg(prs=("debut", "size"), debuts=("debut", "sum"), repos=("Repository", "nunique"))
)
s["fraction_debut"] = s.debuts / s.prs
s.to_csv(f"{OUTPUT_DIR}/did_sawtooth.csv")
print(s.loc[-3:2].round(3).to_string())

# figures
BLUE, ORANGE, SHADE, GRAY = "#2a78d6", "#eb6834", "#333333", "#888888"
LABEL = {
    ("RQ1", "newcomers", "newcomers_per_month"): ("Newcomers per month", "Newcomers per month"),
    ("RQ2", "Newcomer", "acceptance"): ("Newcomer · acceptance", "Acceptance (%)"),
    ("RQ2", "Newcomer", "resolution_time"): (
        "Newcomer · resolution time",
        "Days to merge (median)",
    ),
    ("RQ2", "Veteran", "acceptance"): ("Prior Contributor · acceptance", "Acceptance (%)"),
    ("RQ2", "Veteran", "resolution_time"): (
        "Prior Contributor · resolution time",
        "Days to merge (median)",
    ),
}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)


def draw(ax, key):
    w = WINDOWS[key]
    c = did[(did.rq == key[0]) & (did.group == key[1]) & (did.metric == key[2])].iloc[0]
    tr = np.expm1 if c.scale == "log" else (lambda v: v * 100)
    g = w.groupby("relative_month")[["y", "y0"]].mean()
    n = w.groupby("relative_month").Repository.nunique()
    before, after = g[g.index < 0], g[g.index >= 0]
    ax.axvline(-0.5, color=GRAY, lw=0.7, ls="--")
    ax.plot(g.index, tr(g.y), "o", color=BLUE, ms=2.8, label="Observed (repo mean)")
    ax.plot(before.index, tr(before.y0), color=ORANGE, lw=1.4, label="Model (repo and month FE)")
    ax.plot(after.index, tr(after.y0), color=ORANGE, lw=1.2, ls=":", label="Counterfactual")
    title, ylabel = LABEL[key]
    ax.set_title(
        f"{title}\nmonth 0 {c.text_month0}\nmonths 0..+5 {c.text_0a5}",
        fontsize=FS,
        color=SHADE,
        linespacing=1.15,
    )
    ticks = list(range(-H_BEFORE, H_AFTER, 3))
    ax.set_xticks(ticks, [f"{m}\n{n.get(m, 0)}" for m in ticks], fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylabel(ylabel, fontsize=FS, color=SHADE)
    style(ax)


FS, FS_TICK, FS_LEG = 8, 7, 7
XLABEL = "Month relative to AI adoption\n(below, no. of repos)"
KEYS = list(PANELS)
fig, ax = plt.subplots(figsize=(241 / 72, 2.6))
draw(ax, KEYS[0])
ax.legend(frameon=False, fontsize=FS_LEG, loc="upper left", handlelength=1.6, borderaxespad=0.2)
ax.set_xlabel(XLABEL, fontsize=FS, color=SHADE)
fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/did_bjs_rq1.{ext}", dpi=300)

fig, axs = plt.subplots(2, 2, figsize=(0.9 * 506 / 72, 5.0))
for ax, key in zip(axs.flat, KEYS[1:]):  # rows: Newcomer, Veteran; columns: acceptance, time
    draw(ax, key)
axs[0][0].legend(
    frameon=False, fontsize=FS_LEG, loc="lower left", handlelength=1.6, borderaxespad=0.2
)
for ax in axs[1]:
    ax.set_xlabel(XLABEL, fontsize=FS, color=SHADE)
fig.tight_layout(pad=0.2, h_pad=0.8, w_pad=1.5)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/did_bjs_rq2.{ext}", dpi=300)

# Breakdowns: boxplot + points of each repository's mean effect over 0..+5. The
# diamond is the aggregate effect of the category, with its bootstrap CI.
jitter = np.random.default_rng(42)
fig, axs = plt.subplots(
    1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [11, 4, 4]}, sharey=True
)
for ax, (name, col) in zip(axs, BREAKDOWNS.items()):
    o = brk_out[brk_out.breakdown == name].reset_index(drop=True)
    vals = [pct(per_repo.loc[per_repo[col] == c, "effect_0a5"].values) for c in o.category]
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.boxplot(
        vals,
        positions=range(len(o)),
        widths=0.55,
        showfliers=False,
        medianprops={"color": BLUE, "lw": 2},
        boxprops={"color": GRAY},
        whiskerprops={"color": GRAY},
        capprops={"color": GRAY},
    )
    for i, v in enumerate(vals):
        ax.scatter(i + jitter.uniform(-0.18, 0.18, len(v)), v, s=9, color=BLUE, alpha=0.35, lw=0)
    ax.errorbar(
        range(len(o)),
        o.effect_0a5_pct,
        yerr=[o.effect_0a5_pct - o.ci_0a5_inf, o.ci_0a5_sup - o.effect_0a5_pct],
        fmt="D",
        color=ORANGE,
        ms=6,
        capsize=3,
        lw=1.5,
        label="Category effect (95% CI)",
    )
    ax.set_xticks(
        range(len(o)), [f"{c}\n{n}" for c, n in zip(o.category, o.repositories)], fontsize=12
    )
    k = kwd[(kwd.breakdown == name) & (kwd.measure == "months_0a5")].iloc[0]
    title = {"language": "Language", "stars": "Stars", "size": "Community size"}[name]
    ax.set_title(f"{title}\n(Kruskal-Wallis p = {k.p:.2f})", fontsize=14, color=SHADE)
    ax.tick_params(axis="y", labelsize=12)
    style(ax)
axs[0].set_ylabel("Effect on newcomers per month, months 0..+5 (%)", color=SHADE, fontsize=12)
axs[0].set_ylim(-75, 150)
axs[0].legend(frameon=False, fontsize=12, loc="upper left")
fig.suptitle(
    "Staggered DiD (BJS) by breakdown: effect of adoption on newcomers per repository "
    "(below, no. of repos; axis clipped at −75% and +150%)",
    fontsize=13,
    color=SHADE,
)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/did_bjs_breakdowns.{ext}", dpi=200, bbox_inches="tight")
