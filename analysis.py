"""Quantitative analysis of RQ1 and RQ2.

RQ1: monthly newcomer flow, comparing the "before" and "after" windows of
each repository, and newcomer retention.
RQ2: resolution time and acceptance rate per repository, testing the groups
within each window (2 before, 5 after) and then comparing the windows, plus
good first issues.
Both RQs are reported globally and per language, stars quartile and
community size quartile. RQ3 (survey) is in survey_analysis.py.
"""

import csv
import os
import re
import sys
import unicodedata
from itertools import combinations

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

INPUT_AI = "data/ai_prs.csv"
INPUT_HUMAN = "data/human_prs.csv"

OUTPUT_DIR = "results"

# Data were collected in August 2026, so that month and later ones are left
# out of every time series (partially collected months).
CUTOFF_MONTH = "2026-08"

# The RQ1 windows are computed per repository below, from the first month of
# a pair of consecutive months with an AI signal. Both windows have N months;
# MIN_WINDOW_MONTHS is the minimum N. Before them comes a baseline of up to
# BASELINE_MONTHS months, bounded by the start of the collection, that
# identifies the existing members.
MIN_WINDOW_MONTHS = 6
BASELINE_MONTHS = 24
COLLECTION_START = "2022-01"

ALPHA = 0.05

CLIFF_THRESHOLDS = [
    (0.15, "negligible"),
    (0.33, "small"),
    (0.47, "medium"),
    (float("inf"), "large"),
]

MIN_N_PER_REPO = 5
MIN_N_CELL = 30  # minimum merged PRs per cell (language x group, month x group)


os.makedirs(OUTPUT_DIR, exist_ok=True)


_RE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RE_TIME = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")


def read_csv_robust(path: str) -> pd.DataFrame:
    corrected, discarded, rows = 0, 0, []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        n = len(header)
        idx_data = next(
            (i for i, c in enumerate(header) if c.strip().lower() in ("data", "created_at")),
            8,
        )
        for row in reader:
            if len(row) == n + 1:
                d = row[idx_data].strip()
                t = row[idx_data + 1].strip() if idx_data + 1 < len(row) else ""
                if _RE_DATE.match(d) and _RE_TIME.match(t):
                    row = row[:idx_data] + [d + " " + t] + row[idx_data + 2 :]
                    corrected += 1
            if len(row) == n:
                rows.append(row)
            else:
                discarded += 1
    if corrected:
        print(f"  fixed {corrected} rows (date and time joined)")
    if discarded:
        print(f"  dropped {discarded} rows (unexpected number of columns)")
    return pd.DataFrame(rows, columns=header)


def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().strip().lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "ai_tool": "AI_Tool",
        "detection_source": "Detection_Source",
        "language": "Language",
        "stars": "Stars",
        "repository": "Repository",
        "pr_number": "PR_Number",
        "title": "Title",
        "link": "Link",
        "created_at": "Created_At",
        "updated_at": "Updated_At",
        "closed_at": "Closed_At",
        "state": "State",
        "pr_author": "Author",
        "pr_author_category": "Author_Category",
        "signal_date": "Signal_Date",
        "closes_gfi": "Closes_GFI",
        "gfi_issue_numbers": "GFI_Issue_Numbers",
    }
    df = df.rename(columns={c: c.strip() for c in df.columns})
    rename = {c: mapping[_strip_accents(c)] for c in df.columns if _strip_accents(c) in mapping}
    return df.rename(columns=rename)


def cliffs_delta_from_u(u_stat: float, n1: int, n2: int) -> float:
    return (2.0 * u_stat) / (n1 * n2) - 1.0


def paired_rank_biserial(v1: pd.Series, v2: pd.Series) -> float:
    """Paired rank-biserial correlation (Kerby 2014), the effect size of the
    Wilcoxon signed-rank test: (sum of ranks of positive differences - sum of
    ranks of negative ones) / sum of ranks. Zero differences are dropped.
    """
    d = (v1 - v2).to_numpy()
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    signed_ranks = stats.rankdata(np.abs(d))
    return (signed_ranks[d > 0].sum() - signed_ranks[d < 0].sum()) / signed_ranks.sum()


def interpret_cliffs_delta(d: float) -> str:
    ad = abs(d)
    for threshold, label in CLIFF_THRESHOLDS:
        if ad < threshold:
            return label
    return "large"


def scott_knott_esd(groups: dict[str, pd.Series]) -> dict[str, int]:
    """Non-parametric Scott-Knott ESD (Tantithamthavorn et al., TSE 2018),
    ported from PartitionNonParametric of the R package ScottKnottESD 3.0.0
    (version="np"). Sorts the groups by median (largest first) and splits the
    list where the Kruskal-Wallis H between the two halves is largest. A split
    is kept only if Cliff's delta between the first and last group of the
    segment is not negligible, and the procedure repeats on each half.
    Returns the rank of each group (1 = largest medians). Two differences
    from the package: the negligible threshold is 0.15 (the package uses
    0.147), and groups are named in median order.
    """
    names = sorted(groups, key=lambda n: groups[n].median(), reverse=True)
    ranks: dict[str, int] = {}

    def _negligible(a: pd.Series, b: pd.Series) -> bool:
        u = stats.mannwhitneyu(a, b).statistic
        # round so that a delta exactly at the threshold (e.g. 15/100) is not lost to
        # floating-point error
        return (
            interpret_cliffs_delta(round(cliffs_delta_from_u(u, len(a), len(b)), 10))
            == "negligible"
        )

    def _split(segment: list[str]):
        if len(segment) == 1 or _negligible(groups[segment[0]], groups[segment[-1]]):
            rank = len(set(ranks.values())) + 1
            ranks.update({n: rank for n in segment})
            return
        h = [
            stats.kruskal(
                pd.concat([groups[n] for n in segment[:i]]),
                pd.concat([groups[n] for n in segment[i:]]),
            ).statistic
            for i in range(1, len(segment))
        ]
        cutoff = int(np.argmax(h)) + 1
        _split(segment[:cutoff])
        _split(segment[cutoff:])

    _split(names)
    return ranks


def apply_fdr(df: pd.DataFrame, p_col: str = "mwu_p", alpha: float = ALPHA) -> pd.DataFrame:
    """Benjamini-Hochberg (FDR) correction over the rows of a DataFrame of
    pairwise comparisons, treated as one family of tests. BH never makes a
    non-significant pair significant, so `significant`/`effect`/`magnitude`
    can be recomputed on top of run_group_comparison's output.
    """
    df = df.copy()
    n = len(df)
    if n == 0:
        df["mwu_p_fdr"] = pd.Series(dtype=float)
        return df

    pvals = df[p_col].to_numpy(dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    ranks = np.arange(1, n + 1)
    q_raw = ranked * n / ranks
    # enforce monotonicity
    q_monotone = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_monotone = np.clip(q_monotone, 0, 1)
    q_values = np.empty(n)
    q_values[order] = q_monotone

    df["mwu_p_fdr"] = q_values
    df["significant"] = df["mwu_p_fdr"] < alpha
    # clear effect/magnitude of pairs that were significant only before the
    # correction
    became_ns = (~df["significant"]) & df["effect"].notna()
    df.loc[became_ns, "effect"] = None
    df.loc[became_ns, "magnitude"] = None
    df["effect"] = df["effect"].astype("float64")
    df["magnitude"] = df["magnitude"].astype("object")
    return df


# Shapiro-Wilk result of each run_group_comparison call (rq_shapiro.csv).
_shapiro_rows = []


def run_group_comparison(
    vectors: dict[str, pd.Series], metric_name: str, paired: bool = False, shapiro: bool = True
) -> pd.DataFrame:
    """paired=False: Mann-Whitney U between the vectors (independent
    observations, e.g. resolution time per PR).
    paired=True: per-repository metrics (index = Repository). Each pair is
    compared on the repositories they share, with the Wilcoxon signed-rank
    test. The p-value goes to the mwu_p column and `test` says which test
    was used. repos_1_greater/repos_1_less/repos_tied count the
    repositories where group 1 is above, below or equal to group 2.
    The effect size is Cliff's delta for Mann-Whitney and the paired
    rank-biserial for Wilcoxon (`effect_measure`), with the same magnitude
    thresholds (CLIFF_THRESHOLDS).
    shapiro=False skips the Shapiro-Wilk test.
    """
    shapiro_results = {}
    for name, v in vectors.items() if shapiro else ():
        v_clean = v.dropna()
        if len(v_clean) < 3:
            shapiro_results[name] = (None, None)
            continue
        # sort before sampling: DuckDB returns rows in varying order
        sample = v_clean.sort_values(ignore_index=True).sample(
            min(len(v_clean), 5000), random_state=42
        )
        stat, p = stats.shapiro(sample)
        shapiro_results[name] = (stat, p)
        _shapiro_rows.append(
            {"metric": metric_name, "group": name, "n": len(v_clean), "W": stat, "p": p}
        )

    if shapiro:
        print(f"\n  [{metric_name}] Shapiro-Wilk per group:")
    for name, (stat, p) in shapiro_results.items():
        n = len(vectors[name].dropna())
        if p is None:
            print(f"    {name:<24} n={n:<8} (insufficient N)")
        else:
            normal = "normal" if p >= ALPHA else "non-normal"
            print(f"    {name:<24} n={n:<8} p={p:.4g}  ({normal})")

    rows = []
    for g1, g2 in combinations(vectors.keys(), 2):
        v1, v2 = vectors[g1].dropna(), vectors[g2].dropna()
        if paired:
            pair = pd.concat([v1, v2], axis=1, join="inner")
            v1, v2 = pair.iloc[:, 0], pair.iloc[:, 1]
        if len(v1) < 3 or len(v2) < 3:
            continue
        u_stat, p_mwu = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        if paired:
            # all differences are zero: Wilcoxon undefined, no difference
            p_mwu = stats.wilcoxon(v1, v2).pvalue if (v1.to_numpy() != v2.to_numpy()).any() else 1.0

        row = {
            "metric": metric_name,
            "group_1": g1,
            "group_2": g2,
            "test": "paired_wilcoxon" if paired else "mann_whitney",
            "n1": len(v1),
            "n2": len(v2),
            "median_1": v1.median(),
            "median_2": v2.median(),
            "paired_median_diff": (v1 - v2).median() if paired else np.nan,
            "repos_1_greater": int((v1 > v2).sum()) if paired else np.nan,
            "repos_1_less": int((v1 < v2).sum()) if paired else np.nan,
            "repos_tied": int((v1 == v2).sum()) if paired else np.nan,
            "mwu_p": p_mwu,
            "significant": p_mwu < ALPHA,
            "effect_measure": "paired_rank_biserial" if paired else "cliffs_delta",
        }
        if p_mwu < ALPHA:
            delta = (
                paired_rank_biserial(v1, v2)
                if paired
                else cliffs_delta_from_u(u_stat, len(v1), len(v2))
            )
            row["effect"] = delta
            row["magnitude"] = interpret_cliffs_delta(delta)
        else:
            row["effect"] = None
            row["magnitude"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # fixed dtypes even when effect/magnitude are all None
    df["effect"] = df["effect"].astype("float64")
    df["magnitude"] = df["magnitude"].astype("object")
    return df


def overall_kruskal(vectors: dict[str, pd.Series], metric_name: str) -> dict:
    """Kruskal-Wallis across all groups of a window, before the pairwise tests.
    Groups with fewer than 3 observations are left out.
    """
    samples = {n: v.dropna() for n, v in vectors.items() if len(v.dropna()) >= 3}
    h = stats.kruskal(*samples.values())
    return {
        "metric": metric_name,
        "groups": ", ".join(samples),
        "n_total": sum(map(len, samples.values())),
        "H": h.statistic,
        "p": h.pvalue,
    }


if not os.path.isfile(INPUT_AI) or not os.path.isfile(INPUT_HUMAN):
    print(f"Error: '{INPUT_AI}' or '{INPUT_HUMAN}' not found. Run classify_prs.py first.")
    sys.exit(1)

print(f"Reading '{INPUT_AI}'...")
df_ai = normalize_columns(read_csv_robust(INPUT_AI))
print(f"Reading '{INPUT_HUMAN}'...")
df_h = normalize_columns(read_csv_robust(INPUT_HUMAN))

df_ai["Closes_GFI"] = df_ai["Closes_GFI"].astype(str).str.lower() == "true"
df_h["Closes_GFI"] = df_h["Closes_GFI"].astype(str).str.lower() == "true"

con = duckdb.connect()
con.register("ai_raw", df_ai)
con.register("h_raw", df_h)

# Human PRs without an author login (deleted accounts) are left out of the
# control and of the newcomer base.
CUTOFF = f"AND strftime('%Y-%m', TRY_CAST(COALESCE(NULLIF(Signal_Date, ''), Created_At) AS TIMESTAMP)) < '{CUTOFF_MONTH}'"

con.execute(f"""
CREATE OR REPLACE VIEW ai AS
SELECT
    AI_Tool, Detection_Source, Language, TRY_CAST(Stars AS INTEGER) AS Stars,
    Repository, PR_Number, Title,
    TRY_CAST(Created_At AS TIMESTAMP) AS Created_At,
    TRY_CAST(Closed_At AS TIMESTAMP) AS Closed_At,
    LOWER(TRIM(State)) AS State,
    Author_Category, Author,
    strftime('%Y-%m', TRY_CAST(Created_At AS TIMESTAMP)) AS Month,
    TRY_CAST(Signal_Date AS TIMESTAMP) AS Signal_Date,
    strftime('%Y-%m', TRY_CAST(Signal_Date AS TIMESTAMP)) AS Signal_Month,
    Closes_GFI, GFI_Issue_Numbers
FROM ai_raw
WHERE 1=1 {CUTOFF}
""")

con.execute(f"""
CREATE OR REPLACE VIEW humans AS
SELECT
    Language, TRY_CAST(Stars AS INTEGER) AS Stars,
    Repository, PR_Number, Title, Author,
    TRY_CAST(Created_At AS TIMESTAMP) AS Created_At,
    TRY_CAST(Closed_At AS TIMESTAMP) AS Closed_At,
    LOWER(TRIM(State)) AS State,
    strftime('%Y-%m', TRY_CAST(Created_At AS TIMESTAMP)) AS Month,
    Closes_GFI, GFI_Issue_Numbers
FROM h_raw
WHERE NULLIF(TRIM(Author), '') IS NOT NULL {CUTOFF}
""")

# All PRs opened by human accounts (control + Human AI), the base of the
# newcomer definition: someone whose first PR is AI-assisted is also a
# newcomer. The ai view cuts by Signal_Date when present, and a PR created in
# August may have a July signal, so this side also cuts by creation date.
con.execute(f"""
CREATE OR REPLACE VIEW humans_all AS
SELECT Repository, Language, Author, PR_Number, Created_At, Month FROM humans
UNION ALL
SELECT Repository, Language, Author, PR_Number, Created_At, Month FROM ai
WHERE Author_Category = 'Human' AND Month < '{CUTOFF_MONTH}'
  AND NULLIF(TRIM(Author), '') IS NOT NULL
""")

print(f"\nAI PRs (after cutoff): {con.execute('SELECT COUNT(*) FROM ai').fetchone()[0]:,}")
print(f"Human PRs (after cutoff): {con.execute('SELECT COUNT(*) FROM humans').fetchone()[0]:,}\n")

# RQ1 thresholds, per repository: the first month of a pair of consecutive
# months with an AI signal. Repositories without such a pair, or whose window
# (N months, from the threshold to the cutoff) is shorter than
# MIN_WINDOW_MONTHS, are left out of the whole RQ1 analysis.
_signals_per_repo = con.execute(f"""
    SELECT DISTINCT Repository, Signal_Month FROM ai
    WHERE Signal_Month IS NOT NULL AND Signal_Month < '{CUTOFF_MONTH}'
""").df()

_cutoff_period = pd.Period(CUTOFF_MONTH, freq="M")


def _threshold_of_repo(months_str: list[str]) -> str | None:
    months = sorted(pd.Period(m, freq="M") for m in months_str)
    return next(
        (str(m) for m, nxt in zip(months, months[1:]) if nxt == m + 1),
        None,
    )


def _row_threshold(repo: str, thr: str) -> dict | None:
    """Intervals of the repository for a threshold (None if N is below the minimum)."""
    thr_period = pd.Period(thr, freq="M")
    n_months = int((_cutoff_period - thr_period).n)
    if n_months < MIN_WINDOW_MONTHS:
        return None
    return {
        "Repository": repo,
        "threshold": thr,
        "baseline_start": str(
            max(pd.Period(COLLECTION_START, freq="M"), thr_period - n_months - BASELINE_MONTHS)
        ),
        "baseline_end": str(thr_period - n_months),
        "n_months": n_months,
    }


def _thresholds(signals: pd.DataFrame) -> pd.DataFrame:
    """Threshold of each repository from its months with an AI signal (columns
    Repository, Signal_Month). Also used with prior-member signals only.
    """
    rows = []
    for repo, group in signals.groupby("Repository"):
        thr = _threshold_of_repo(group["Signal_Month"].tolist())
        if thr is not None and (row := _row_threshold(repo, thr)):
            rows.append(row)
    return pd.DataFrame(rows)


df_repo_threshold = _thresholds(_signals_per_repo)
_n_repos_with_signal = _signals_per_repo["Repository"].nunique()
_n_discarded = _n_repos_with_signal - len(df_repo_threshold)
print(
    f"Repositories with a valid RQ1 threshold: {len(df_repo_threshold)} of "
    f"{_n_repos_with_signal} with some AI signal "
    f"({_n_discarded} dropped: no consecutive pair or window < "
    f"{MIN_WINDOW_MONTHS} months)"
)
print(
    f"  N months per repo: min={df_repo_threshold['n_months'].min()}, "
    f"median={df_repo_threshold['n_months'].median():.0f}, "
    f"max={df_repo_threshold['n_months'].max()}"
)
_months_baseline = df_repo_threshold.apply(
    lambda r: (pd.Period(r["baseline_end"], freq="M") - pd.Period(r["baseline_start"], freq="M")).n,
    axis=1,
)
print(
    f"  Baseline < {BASELINE_MONTHS} months (collection starts in {COLLECTION_START}): "
    f"{(_months_baseline < BASELINE_MONTHS).sum()} repos "
    f"(min={_months_baseline.min()} months)"
)

con.register("repo_threshold", df_repo_threshold)

# Three contiguous intervals per repository (Steinmacher et al. 2013):
# baseline = [max(COLLECTION_START, threshold-N-BASELINE_MONTHS), threshold-N),
# which identifies the existing members; before = [threshold-N, threshold);
# after = [threshold, CUTOFF_MONTH). The newcomer of a window is an author whose
# first PR in the repository, counted from the start of the baseline, falls in
# it; the entry month is the month of that PR. RQ1 follows the monthly flow in
# each window, tests its trend (Mann-Kendall) and compares before and after per
# repository (paired Wilcoxon on level and trend). The sensitivity analysis
# drops the first SENSITIVITY_MONTHS months of each window.
SENSITIVITY_MONTHS = 3

print("\n" + "=" * 70)
print("RQ1 - NEWCOMER ENTRY FLOW ANALYSIS (per-repository threshold)")
print("=" * 70)

# Newcomers per Author+Repository pair in both windows. The newcomers view
# (after only) defines the window-2 newcomers of RQ2, and newcomers_before
# those of window 1.
con.execute("""
    CREATE OR REPLACE VIEW newcomers_windows AS
    WITH first AS (
        SELECT h.Author, h.Repository, rt.threshold, rt.baseline_end,
               MIN(h.Month) AS first_month
        FROM humans_all h
        JOIN repo_threshold rt ON h.Repository = rt.Repository
        WHERE h.Month >= rt.baseline_start
        GROUP BY h.Author, h.Repository, rt.threshold, rt.baseline_end
    )
    SELECT Author, Repository, threshold,
           CASE WHEN first_month >= threshold THEN 'after' ELSE 'before' END AS "window",
           first_month
    FROM first
    WHERE first_month >= baseline_end
""")
con.execute("""
    CREATE OR REPLACE VIEW newcomers AS
    SELECT Author, Repository, threshold, first_month
    FROM newcomers_windows WHERE "window" = 'after'
""")
con.execute("""
    CREATE OR REPLACE VIEW newcomers_before AS
    SELECT Author, Repository, threshold, first_month
    FROM newcomers_windows WHERE "window" = 'before'
""")


def _month_idx(month: str, threshold: str) -> int:
    return (pd.Period(month, freq="M") - pd.Period(threshold, freq="M")).n


# Full monthly series per repository (months without newcomers = 0), indexed by
# the month relative to the threshold (0 = threshold month; negative before).
_counts = con.execute("""
    SELECT Repository, first_month AS Month, COUNT(*) AS newcomers
    FROM newcomers_windows GROUP BY Repository, first_month
""").df()
_cnt_idx = {(r, m): n for r, m, n in _counts.itertuples(index=False)}
# Distinct human authors with a PR in the month (newcomers + veterans).
_active = con.execute("""
    SELECT h.Repository, h.Month, COUNT(DISTINCT h.Author) AS active
    FROM humans_all h JOIN repo_threshold rt ON h.Repository = rt.Repository
    WHERE h.Month >= rt.baseline_end GROUP BY h.Repository, h.Month
""").df()
_active_idx = {(r, m): n for r, m, n in _active.itertuples(index=False)}
_lang_repo = dict(
    con.execute(
        "SELECT Repository, ANY_VALUE(Language) FROM humans_all GROUP BY Repository"
    ).fetchall()
)

_rows_series = []
for rt in df_repo_threshold.itertuples(index=False):
    thr = pd.Period(rt.threshold, freq="M")
    for k in range(-rt.n_months, rt.n_months):
        month = str(thr + k)
        _rows_series.append(
            {
                "Repository": rt.Repository,
                "Language": _lang_repo.get(rt.Repository),
                "window": "after" if k >= 0 else "before",
                "Month": month,
                "relative_month": k,
                "newcomers": _cnt_idx.get((rt.Repository, month), 0),
                "active": _active_idx.get((rt.Repository, month), 0),
            }
        )
df_newcomers_windows = pd.DataFrame(_rows_series)
# Share of newcomers among the active human authors of the month, which
# separates "more newcomers" from "busier project".
df_newcomers_windows["newcomer_fraction"] = df_newcomers_windows[
    "newcomers"
] / df_newcomers_windows["active"].where(df_newcomers_windows["active"] > 0)
df_newcomers_windows.to_csv(f"{OUTPUT_DIR}/rq1_newcomers_windows_per_repository.csv", index=False)
_per_window = df_newcomers_windows.groupby(["Repository", "window"])["newcomers"].sum().unstack()
print(
    f"  rq1_newcomers_windows_per_repository.csv  newcomers 'before'="
    f"{int(_per_window['before'].sum()):,}, 'after'={int(_per_window['after'].sum()):,}; "
    f"median per repo {_per_window['before'].median():g} x {_per_window['after'].median():g}; "
    f"'after' > 'before' in {(_per_window['after'] > _per_window['before']).sum()} "
    f"of {len(_per_window)} repos"
)

# Series of each window (the before window goes from -N to -1).
df_newcomers_repo = (
    df_newcomers_windows[df_newcomers_windows["relative_month"] >= 0]
    .drop(columns="window")
    .reset_index(drop=True)
)
df_newcomers_repo.to_csv(f"{OUTPUT_DIR}/rq1_newcomers_per_repository.csv", index=False)
print(
    f"  rq1_newcomers_per_repository.csv  ({df_newcomers_repo['Repository'].nunique()} "
    f"repositories, {len(df_newcomers_repo)} repo-months in 'after')"
)
df_newcomers_before = (
    df_newcomers_windows[df_newcomers_windows["relative_month"] < 0]
    .drop(columns="window")
    .reset_index(drop=True)
)

# Aggregated views: per calendar month (sum) and per month relative to the
# threshold (mean per repository, since fewer repositories reach month k as k
# grows).
df_newcomers_global = (
    df_newcomers_repo.groupby("Month")["newcomers"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "newcomers", "count": "repositories"})
    .reset_index()
)
df_newcomers_global.to_csv(f"{OUTPUT_DIR}/rq1_newcomers_global.csv", index=False)

df_newcomers_relative = (
    df_newcomers_windows.groupby("relative_month")["newcomers"]
    .agg(["sum", "count", "mean"])
    .rename(columns={"sum": "newcomers", "count": "repositories", "mean": "mean_per_repository"})
    .reset_index()
)
df_newcomers_relative.to_csv(f"{OUTPUT_DIR}/rq1_newcomers_relative_month.csv", index=False)

# Per-repository index: newcomers in the month / the repository's monthly mean
# in the before window (1 = same as before). Each repository is compared with
# itself, so the extreme months (only early, larger adopters) do not rise just
# by composition. Repositories without newcomers before are left out.
_base_before = (
    df_newcomers_windows[df_newcomers_windows["window"] == "before"]
    .groupby("Repository")["newcomers"]
    .mean()
)
_index = df_newcomers_windows["newcomers"] / df_newcomers_windows["Repository"].map(
    _base_before.where(_base_before > 0)
)
df_newcomers_index = (
    _index.groupby(df_newcomers_windows["relative_month"])
    .agg(
        repositories="count",
        median_index="median",
        index_q1=lambda s: s.quantile(0.25),
        index_q3=lambda s: s.quantile(0.75),
    )
    .reset_index()
)
df_newcomers_index.to_csv(f"{OUTPUT_DIR}/rq1_newcomers_index_relative_month.csv", index=False)
print(
    f"  rq1_newcomers_index_relative_month.csv  ({(_base_before > 0).sum()} of "
    f"{len(_base_before)} repos with newcomers 'before')"
)

df_newcomers_language = (
    df_newcomers_repo.groupby(["Language", "Month"])["newcomers"].sum().reset_index()
)
df_newcomers_language.to_csv(f"{OUTPUT_DIR}/rq1_newcomers_per_language.csv", index=False)
print(
    "  rq1_newcomers_global.csv, rq1_newcomers_relative_month.csv, rq1_newcomers_per_language.csv"
)


# Trend per repository: Mann-Kendall (Kendall's tau between relative month and
# count) + Sen's slope (newcomers/month)
def _mann_kendall_p(y: np.ndarray) -> float:
    """Two-sided p of the classic Mann-Kendall test (Hamed and Rao 1998, as in
    pymannkendall): statistic S, variance corrected for ties in y and
    continuity correction. scipy's kendalltau lacks the correction.
    """
    n = len(y)
    s = sum(np.sign(y[j + 1 :] - y[j]).sum() for j in range(n - 1))
    _, ties = np.unique(y, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5) - (ties * (ties - 1) * (2 * ties + 5)).sum()) / 18
    z = 0.0 if s == 0 else (s - np.sign(s)) / np.sqrt(var_s)
    return 2 * stats.norm.sf(abs(z))


def _trend_repos(series: pd.DataFrame, skip: int = 0, col: str = "newcomers") -> pd.DataFrame:
    rows = []
    for (repo, lang), g in series.groupby(["Repository", "Language"]):
        # `skip` counts from the first month of the window (0 after, -N before)
        g = g[
            (g["relative_month"] >= g["relative_month"].min() + skip) & g[col].notna()
        ].sort_values("relative_month")
        y = g[col].to_numpy()
        x = g["relative_month"].to_numpy()
        if len(y) < MIN_WINDOW_MONTHS - skip or np.all(y == y[0]):
            tau, p, sen = np.nan, np.nan, 0.0
        else:
            tau = stats.kendalltau(x, y).statistic
            p = _mann_kendall_p(y)
            sen = stats.theilslopes(y, x)[0]
        rows.append(
            {
                "Repository": repo,
                "Language": lang,
                "n_months": len(y),
                "total_newcomers": int(y.sum()),
                "mean": y.mean() if len(y) else np.nan,
                "kendall_tau": tau,
                "mk_p": p,
                "sen_slope": sen,
            }
        )
    df = pd.DataFrame(rows)
    testable = df["mk_p"].notna()
    df["mk_p_fdr"] = np.nan
    if testable.any():
        df.loc[testable, "mk_p_fdr"] = stats.false_discovery_control(
            df.loc[testable, "mk_p"], method="bh"
        )
    df["trend"] = np.select(
        [df["mk_p_fdr"] < ALPHA, df["mk_p_fdr"].notna()],
        [np.where(df["kendall_tau"] > 0, "increasing", "decreasing"), "no trend"],
        default="constant",
    )
    return df


def _summary_trend(df: pd.DataFrame, label: str) -> dict:
    taus = df["kendall_tau"].dropna()
    w = stats.wilcoxon(taus) if len(taus) >= 10 else None
    cnt = df["trend"].value_counts()
    return {
        "breakdown": label,
        "repositories": len(df),
        "testable": len(taus),
        "increasing": int(cnt.get("increasing", 0)),
        "decreasing": int(cnt.get("decreasing", 0)),
        "no_trend": int(cnt.get("no trend", 0)),
        "constant": int(cnt.get("constant", 0)),
        "tau_positive": int((taus > 0).sum()),
        "tau_negative": int((taus < 0).sum()),
        "tau_median": taus.median(),
        "wilcoxon_W": w.statistic if w else np.nan,
        "wilcoxon_p": w.pvalue if w else np.nan,
    }


# Newcomers who already had a PR in the repository before the start of the
# baseline ("returning"), per month relative to the threshold, in both windows.
con.execute("""
    CREATE OR REPLACE VIEW newcomers_returning AS
    SELECT DISTINCT n.Author, n.Repository
    FROM newcomers_windows n
    JOIN repo_threshold rt ON n.Repository = rt.Repository
    JOIN humans_all h ON h.Author = n.Author AND h.Repository = n.Repository
    WHERE h.Month < rt.baseline_start
""")
_ret = con.execute("""
    SELECT n.Repository, n."window", n.first_month AS Month, COUNT(*) AS newcomers,
           COUNT(r.Author) AS "returning"
    FROM newcomers_windows n
    LEFT JOIN newcomers_returning r ON n.Author = r.Author AND n.Repository = r.Repository
    GROUP BY 1, 2, 3
""").df()
_ret["relative_month"] = [
    _month_idx(m, t)
    for m, t in zip(
        _ret["Month"],
        _ret["Repository"].map(df_repo_threshold.set_index("Repository")["threshold"]),
    )
]
df_ret = _ret.groupby(["window", "relative_month"])[["newcomers", "returning"]].sum().reset_index()
df_ret["returning_fraction"] = df_ret["returning"] / df_ret["newcomers"]
df_ret.to_csv(f"{OUTPUT_DIR}/rq1_returning_relative_month.csv", index=False)
_ret_j = _ret.groupby("window")[["newcomers", "returning"]].sum()
print(
    "  rq1_returning_relative_month.csv  returning: "
    + "; ".join(
        f"{j} {r.returning:,} of {r.newcomers:,} ({r.returning / r.newcomers:.1%})"
        for j, r in _ret_j.iterrows()
    )
)

# Trend in each window: the newcomer series and two controls for overall
# growth (active human authors and the share of newcomers among them).
SERIES_RQ1 = {
    "newcomers": "newcomers",
    "active": "control:active_authors",
    "newcomer_fraction": "control:newcomer_fraction",
}
WINDOWS_RQ1 = {"before": df_newcomers_before, "after": df_newcomers_repo}
trend = {
    (j, col): _trend_repos(series, col=col)
    for j, series in WINDOWS_RQ1.items()
    for col in SERIES_RQ1
}
trend_sens = {
    (j, "newcomers"): _trend_repos(series, skip=SENSITIVITY_MONTHS)
    for j, series in WINDOWS_RQ1.items()
}
df_trend = trend[("after", "newcomers")]
df_trend.to_csv(f"{OUTPUT_DIR}/rq1_trend_per_repository.csv", index=False)
trend[("before", "newcomers")].to_csv(
    f"{OUTPUT_DIR}/rq1_trend_per_repository_before.csv", index=False
)
trend_sens[("after", "newcomers")].to_csv(
    f"{OUTPUT_DIR}/rq1_trend_per_repository_no_{SENSITIVITY_MONTHS}_months.csv", index=False
)

_summaries = []
for j in WINDOWS_RQ1:
    _summaries.append(_summary_trend(trend[(j, "newcomers")], j))
    _summaries.append(
        _summary_trend(trend_sens[(j, "newcomers")], f"{j}:no_{SENSITIVITY_MONTHS}_first_months")
    )
    for col in ("active", "newcomer_fraction"):
        _summaries.append(_summary_trend(trend[(j, col)], f"{j}:{SERIES_RQ1[col]}"))
    for lang, g in trend[(j, "newcomers")].groupby("Language"):
        _summaries.append(_summary_trend(g, f"{j}:language:{lang}"))
df_trend_summary = pd.DataFrame(_summaries)
# BH across the testable languages, one family per window.
df_trend_summary["wilcoxon_p_fdr"] = np.nan
for j in WINDOWS_RQ1:
    _lang = (
        df_trend_summary["breakdown"].str.startswith(f"{j}:language:")
        & df_trend_summary["wilcoxon_p"].notna()
    )
    df_trend_summary.loc[_lang, "wilcoxon_p_fdr"] = stats.false_discovery_control(
        df_trend_summary.loc[_lang, "wilcoxon_p"], method="bh"
    )
df_trend_summary.to_csv(f"{OUTPUT_DIR}/rq1_trend_summary.csv", index=False)
print("  rq1_trend_per_repository*.csv, rq1_trend_summary.csv")
print(df_trend_summary.round(4).to_string(index=False))


# Before x after per repository. Level = monthly mean of the series in the
# window; trend = Kendall's tau. Both windows have the same N.
def _table_windows(tends: dict) -> pd.DataFrame:
    df = df_repo_threshold[["Repository", "threshold", "n_months"]].copy()
    for (j, col), t in tends.items():
        t = t.set_index("Repository")
        df[f"mean_{col}_{j}"] = df["Repository"].map(t["mean"])
        df[f"tau_{col}_{j}"] = df["Repository"].map(t["kendall_tau"])
    return df


def _compare_windows(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Paired Wilcoxon after x before per repository, on the level and the trend
    of each series. delta > 0 = larger after.
    """
    d = df.set_index("Repository")
    parts = [
        run_group_comparison(
            {"after": d[f"{measure}_{col}_after"], "before": d[f"{measure}_{col}_before"]},
            f"{measure}_{col}",
            paired=True,
            shapiro=False,
        ).assign(series=col, measure=measure)
        for col in cols
        for measure in ("mean", "tau")
    ]
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)


df_rq1_repo = _table_windows(trend)
df_rq1_repo.insert(1, "Language", df_rq1_repo["Repository"].map(_lang_repo))
for j in WINDOWS_RQ1:
    t = trend[(j, "newcomers")].set_index("Repository")
    df_rq1_repo[f"total_newcomers_{j}"] = df_rq1_repo["Repository"].map(t["total_newcomers"])
    df_rq1_repo[f"trend_{j}"] = df_rq1_repo["Repository"].map(t["trend"])
# Relative change of the level (NaN if no newcomers before) and tau
# difference, used to compare breakdowns.
df_rq1_repo["level_change_newcomers"] = (
    df_rq1_repo["mean_newcomers_after"]
    / df_rq1_repo["mean_newcomers_before"].where(df_rq1_repo["mean_newcomers_before"] > 0)
    - 1
)
df_rq1_repo["tau_diff_newcomers"] = (
    df_rq1_repo["tau_newcomers_after"] - df_rq1_repo["tau_newcomers_before"]
)

# Community size = monthly mean of distinct human authors in the baseline,
# measured before both windows so that it is not inflated by the growth RQ1
# measures. It is a monthly mean because the baseline spans 13 to 24 months.
_author_months = (
    con.execute("""
    SELECT h.Repository, COUNT(DISTINCT h.Author || '|' || h.Month) AS author_months
    FROM humans_all h JOIN repo_threshold rt ON h.Repository = rt.Repository
    WHERE h.Month >= rt.baseline_start AND h.Month < rt.baseline_end
    GROUP BY h.Repository
""")
    .df()
    .set_index("Repository")["author_months"]
)
df_rq1_repo["baseline_authors_per_month"] = (
    df_rq1_repo["Repository"].map(_author_months).fillna(0)
    / _months_baseline.set_axis(df_repo_threshold["Repository"])
    .reindex(df_rq1_repo["Repository"])
    .to_numpy()
)
# Stars from results/repo_metrics.csv (repo_stats.py).
df_rq1_repo["stars"] = df_rq1_repo["Repository"].map(
    pd.read_csv("results/repo_metrics.csv").set_index("repo")["stars"]
)
N_TIERS = 4


def _quartiles(v: pd.Series) -> pd.Series:
    return pd.qcut(v.rank(method="first"), N_TIERS, labels=[f"Q{i + 1}" for i in range(N_TIERS)])


df_rq1_repo["tier_stars"] = _quartiles(df_rq1_repo["stars"])
df_rq1_repo["tier_size"] = _quartiles(df_rq1_repo["baseline_authors_per_month"])

# AI intensity = share of after-window PRs with AI evidence. The no_newcomers
# version drops all PRs of after-window newcomers: about a quarter of the AI
# PRs after adoption are theirs, which would make the association circular.
_intensity = con.execute(f"""
    WITH prs AS (
        SELECT Repository, Author, Month, 1 AS is_ai FROM ai WHERE Month < '{CUTOFF_MONTH}'
        UNION ALL
        SELECT Repository, Author, Month, 0 FROM humans
    ),
    prs_after AS (
        SELECT p.*, n.Author IS NOT NULL AS by_newcomer
        FROM prs p JOIN repo_threshold rt ON p.Repository = rt.Repository
        LEFT JOIN newcomers n ON n.Repository = p.Repository AND n.Author = p.Author
        WHERE p.Month >= rt.threshold
    )
    SELECT Repository, SUM(is_ai) AS prs_ai, COUNT(*) AS prs_total,
           SUM(is_ai) / COUNT(*) AS fraction_ai,
           SUM(is_ai) FILTER (WHERE NOT by_newcomer) AS prs_ai_no_newcomers,
           SUM(is_ai) FILTER (WHERE NOT by_newcomer)
               / COUNT(*) FILTER (WHERE NOT by_newcomer) AS fraction_ai_no_newcomers,
           SUM(is_ai) FILTER (WHERE by_newcomer) AS prs_ai_by_newcomers
    FROM prs_after GROUP BY Repository
""").df()
df_rq1_repo = df_rq1_repo.merge(_intensity, on="Repository", how="left")
print(
    f"\n  AI PRs 'after' opened by 'after' newcomers: "
    f"{int(_intensity['prs_ai_by_newcomers'].sum()):,} of {int(_intensity['prs_ai'].sum()):,} "
    f"({_intensity['prs_ai_by_newcomers'].sum() / _intensity['prs_ai'].sum():.1%})"
)

# Newcomers and veterans per window, for any set of thresholds (the main one,
# the prior-member one and the placebo pseudo-thresholds). The veteran of a
# window is an author with a PR in it who is not a newcomer of that window,
# counted in author-months. did = log2(newcomers A/B) - log2(veterans A/B),
# undefined when a count is zero.
_author_month = con.execute("SELECT DISTINCT Repository, Author, Month FROM humans_all").df()
# months as integers (year*12 + month): the placebo calls this function
# hundreds of times
_author_month["m"] = pd.to_numeric(_author_month["Month"].str[:4]) * 12 + pd.to_numeric(
    _author_month["Month"].str[5:7]
)
_author_month = _author_month.drop(columns="Month")


def _windows_repo(thr: pd.DataFrame, with_tau: bool = True) -> pd.DataFrame:
    t = thr[["Repository"]].copy()
    for c in ("threshold", "baseline_start", "baseline_end"):
        t[c] = pd.to_numeric(thr[c].str[:4]) * 12 + pd.to_numeric(thr[c].str[5:7])
    d = _author_month.merge(t, on="Repository")
    d = d[d["m"] >= d["baseline_start"]].copy()
    d["first"] = d.groupby(["Repository", "Author"])["m"].transform("min")
    in_baseline = d["m"] < d["baseline_end"]
    baseline_authors_per_month = d[in_baseline].groupby("Repository").size()
    d = d[~in_baseline]
    after = d["m"] >= d["threshold"]
    d["veteran"] = d["first"] < np.where(after, d["threshold"], d["baseline_end"])
    d["debut"] = d["m"] == d["first"]
    d["window"] = np.where(after, "after", "before")
    cnt = d.groupby(["Repository", "window"])[["debut", "veteran"]].sum().unstack("window")
    r = thr[["Repository", "threshold", "n_months"]].set_index("Repository")
    for j in ("before", "after"):
        r[f"total_newcomers_{j}"] = (
            cnt.get(("debut", j), pd.Series(dtype=float)).reindex(r.index).fillna(0)
        )
        r[f"veterans_author_month_{j}"] = (
            cnt.get(("veteran", j), pd.Series(dtype=float)).reindex(r.index).fillna(0)
        )
        r[f"mean_newcomers_{j}"] = r[f"total_newcomers_{j}"] / r["n_months"]
    # Same period for every repository (+-MIN_WINDOW_MONTHS months around the
    # threshold, as in the newcomer_entry figure).
    fixed = d[
        d["debut"]
        & (d["m"] >= d["threshold"] - MIN_WINDOW_MONTHS)
        & (d["m"] < d["threshold"] + MIN_WINDOW_MONTHS)
    ]
    fixed = fixed.groupby(["Repository", "window"]).size().unstack("window")
    for j in ("before", "after"):
        r[f"newcomers_fixed_{j}"] = fixed.get(j, pd.Series(dtype=float)).reindex(r.index).fillna(0)
    m01 = d[d["debut"] & after & (d["m"] <= d["threshold"] + 1)]
    r["newcomers_months_0_1"] = m01.groupby("Repository").size().reindex(r.index).fillna(0)
    months_base = (t["baseline_end"] - t["baseline_start"]).set_axis(t["Repository"])
    r["baseline_authors_per_month"] = (
        baseline_authors_per_month.reindex(r.index).fillna(0) / months_base
    )
    # tau of each window (skipped in the placebo, where it is the slow part)
    if with_tau:
        est = d[d["debut"]].groupby(["Repository", "m"]).size()
        series = []
        for repo, t0, n in zip(t["Repository"], t["threshold"], thr["n_months"]):
            series += [
                {
                    "Repository": repo,
                    "Language": "",
                    "relative_month": k,
                    "newcomers": est.get((repo, t0 + k), 0),
                }
                for k in range(-n, n)
            ]
        series = pd.DataFrame(series)
        for j, s in (
            ("before", series[series["relative_month"] < 0]),
            ("after", series[series["relative_month"] >= 0]),
        ):
            r[f"tau_newcomers_{j}"] = _trend_repos(s).set_index("Repository")["kendall_tau"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r["level_change_newcomers"] = (
            r["mean_newcomers_after"]
            / r["mean_newcomers_before"].where(r["mean_newcomers_before"] > 0)
            - 1
        )
        r["log2_newcomers"] = np.log2(r["total_newcomers_after"] / r["total_newcomers_before"])
        r["log2_veterans"] = np.log2(
            r["veterans_author_month_after"] / r["veterans_author_month_before"]
        )
        r["log2_newcomers_fixed"] = np.log2(
            r["newcomers_fixed_after"] / r["newcomers_fixed_before"]
        )
    r = r.replace([np.inf, -np.inf], np.nan)
    r["did_newcomers_veterans"] = r["log2_newcomers"] - r["log2_veterans"]
    # Newcomers of months 0 and 1 over the before-window monthly mean: whether the
    # threshold falls on a peak of newcomer arrivals.
    r["peak_months_0_1"] = (r["newcomers_months_0_1"] / 2) / r["mean_newcomers_before"].where(
        r["mean_newcomers_before"] > 0
    )
    return r.reset_index()


_win_main = _windows_repo(df_repo_threshold).set_index("Repository")
_conf = df_rq1_repo.set_index("Repository")
for _c in (
    "total_newcomers_before",
    "total_newcomers_after",
    "mean_newcomers_before",
    "mean_newcomers_after",
    "tau_newcomers_before",
    "tau_newcomers_after",
    "baseline_authors_per_month",
):
    assert np.allclose(_win_main[_c], _conf.loc[_win_main.index, _c], equal_nan=True), _c
_COLS_VET = [
    "veterans_author_month_before",
    "veterans_author_month_after",
    "log2_newcomers",
    "log2_veterans",
    "did_newcomers_veterans",
    "newcomers_months_0_1",
    "peak_months_0_1",
    "newcomers_fixed_before",
    "newcomers_fixed_after",
    "log2_newcomers_fixed",
]
df_rq1_repo = df_rq1_repo.merge(
    _win_main[_COLS_VET], left_on="Repository", right_index=True, how="left"
)
df_rq1_repo.to_csv(f"{OUTPUT_DIR}/rq1_before_after_per_repository.csv", index=False)
# Breakdown categories of each repository, reused in RQ2.
BREAKDOWNS = {"language": "Language", "stars": "tier_stars", "size": "tier_size"}
breakdown_repo = df_rq1_repo.set_index("Repository")[list(BREAKDOWNS.values())]
print("\n  rq1_before_after_per_repository.csv  ranges:")
for name in ("stars", "baseline_authors_per_month"):
    tier = "tier_stars" if name == "stars" else "tier_size"
    print(
        f"    {name}: "
        + "; ".join(
            f"{q} {g[name].min():,.1f}–{g[name].max():,.1f}"
            for q, g in df_rq1_repo.groupby(tier, observed=True)
        )
    )

# Global, with the sensitivity that drops the first months of each window.
# Each breakdown is one BH family.
df_ad_global = pd.concat(
    [
        apply_fdr(_compare_windows(df_rq1_repo, list(SERIES_RQ1))).assign(breakdown="global"),
        apply_fdr(_compare_windows(_table_windows(trend_sens), ["newcomers"])).assign(
            breakdown=f"no_{SENSITIVITY_MONTHS}_first_months"
        ),
    ],
    ignore_index=True,
)
df_ad_global.to_csv(f"{OUTPUT_DIR}/rq1_before_after_global.csv", index=False)
print("\n  rq1_before_after_global.csv (paired Wilcoxon after x before per repository)")
print(
    df_ad_global[
        [
            "breakdown",
            "series",
            "measure",
            "n1",
            "median_1",
            "median_2",
            "paired_median_diff",
            "repos_1_greater",
            "repos_1_less",
            "mwu_p_fdr",
            "effect",
            "magnitude",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# Breakdowns: language, stars quartile and community size quartile. Within
# each category, the same paired Wilcoxon (BH across the categories of the
# breakdown, one family per measure). Across categories, Kruskal-Wallis on the
# relative change of the level and on newcomer growth net of veteran growth
# (did). Tau differences are not compared across categories, because tau
# measures consistency and grows with repository size. Categories with fewer
# than MIN_REPOS_BREAKDOWN repositories are descriptive only.
MIN_REPOS_BREAKDOWN = 10
_brk_desc, _brk_tests, _brk_kw = [], [], []
for name, col_brk in BREAKDOWNS.items():
    testable = []
    for cat, g in df_rq1_repo.groupby(col_brk, observed=True):
        _brk_desc.append(
            {
                "breakdown": name,
                "category": cat,
                "repositories": len(g),
                "newcomers_before": int(g["total_newcomers_before"].sum()),
                "newcomers_after": int(g["total_newcomers_after"].sum()),
                "repos_after_greater": int(
                    (g["total_newcomers_after"] > g["total_newcomers_before"]).sum()
                ),
                "level_change_median": g["level_change_newcomers"].median(),
                "tau_diff_median": g["tau_diff_newcomers"].median(),
                "repos_with_did": int(g["did_newcomers_veterans"].notna().sum()),
                "log2_veterans_median": g["log2_veterans"].median(),
                "did_median": g["did_newcomers_veterans"].median(),
            }
        )
        if len(g) >= MIN_REPOS_BREAKDOWN:
            testable.append(g)
            _brk_tests.append(
                _compare_windows(g, ["newcomers"]).assign(breakdown=name, category=cat)
            )
    for var in ("level_change_newcomers", "did_newcomers_veterans"):
        kw = stats.kruskal(*[g[var].dropna() for g in testable])
        _brk_kw.append(
            {
                "breakdown": name,
                "variable": var,
                "categories": len(testable),
                "repositories": sum(g[var].notna().sum() for g in testable),
                "H": kw.statistic,
                "p": kw.pvalue,
            }
        )
_tests = pd.concat(_brk_tests, ignore_index=True)
_tests = pd.concat(
    [apply_fdr(g) for _, g in _tests.groupby(["breakdown", "measure"])], ignore_index=True
)
# separate pivot so that numeric columns do not become object
_wide = pd.concat(
    [
        _tests.pivot(
            index=["breakdown", "category"],
            columns="measure",
            values=["paired_median_diff", "mwu_p_fdr", "effect"],
        ),
        _tests.pivot(index=["breakdown", "category"], columns="measure", values=["magnitude"]),
    ],
    axis=1,
)
_wide.columns = [f"{m}_{v}" for v, m in _wide.columns]
df_brk = pd.DataFrame(_brk_desc).merge(
    _wide.reset_index(), on=["breakdown", "category"], how="left"
)
df_brk.to_csv(f"{OUTPUT_DIR}/rq1_before_after_breakdowns.csv", index=False)
df_brk_kw = pd.DataFrame(_brk_kw)
df_brk_kw["p_fdr"] = stats.false_discovery_control(df_brk_kw["p"], method="bh")
df_brk_kw.to_csv(f"{OUTPUT_DIR}/rq1_before_after_kruskal.csv", index=False)
print("\n  rq1_before_after_breakdowns.csv, rq1_before_after_kruskal.csv")
print(df_brk.round(3).to_string(index=False))
print(df_brk_kw.round(4).to_string(index=False))

# Per language on log2_newcomers_fixed (+-6 months around the threshold): in
# each language, one-sample Wilcoxon against 0 with BH across languages;
# between languages, Kruskal-Wallis, pairwise Mann-Whitney (BH) and the
# Scott-Knott ESD grouping. Only languages with MIN_REPOS_BREAKDOWN repositories.
_lang = {l: g["log2_newcomers_fixed"].dropna() for l, g in df_rq1_repo.groupby("Language")}
_lang = {l: v for l, v in _lang.items() if len(v) >= MIN_REPOS_BREAKDOWN}
_sk_lang = scott_knott_esd(_lang)
_kw_lang = stats.kruskal(*_lang.values())
df_lang_fixed = pd.DataFrame(
    [
        {
            "language": l,
            "repositories": len(v),
            "median_log2": v.median(),
            "median_change_pct": (2 ** v.median() - 1) * 100,
            "repos_grew": int((v > 0).sum()),
            "repos_fell": int((v < 0).sum()),
            "wilcoxon_p": stats.wilcoxon(v).pvalue,
            "effect": paired_rank_biserial(v, pd.Series(0.0, index=v.index)),
            "effect_measure": "paired_rank_biserial",
            "scott_knott_esd": _sk_lang[l],
            "kruskal_H": _kw_lang.statistic,
            "kruskal_p": _kw_lang.pvalue,
        }
        for l, v in _lang.items()
    ]
).sort_values("median_log2", ascending=False)
df_lang_fixed["wilcoxon_p_fdr"] = stats.false_discovery_control(
    df_lang_fixed["wilcoxon_p"], method="bh"
)
df_lang_fixed["magnitude"] = df_lang_fixed["effect"].map(interpret_cliffs_delta)
df_lang_fixed.to_csv(f"{OUTPUT_DIR}/rq1_fixed_language.csv", index=False)
df_lang_pairs = run_group_comparison(_lang, "log2_newcomers_fixed", shapiro=False)
df_lang_pairs = apply_fdr(df_lang_pairs)
df_lang_pairs.to_csv(f"{OUTPUT_DIR}/rq1_fixed_language_pairs.csv", index=False)
print("\n  rq1_fixed_language.csv, rq1_fixed_language_pairs.csv")
print(df_lang_fixed.round(4).to_string(index=False))
print(
    f"  significant pairs (p_fdr < {ALPHA}): {int(df_lang_pairs['significant'].sum())} of {len(df_lang_pairs)}"
)

# Scott-Knott ESD over the months of the newcomer_entry_box figure: the groups
# are the 12 months from -6 to +5, each with the newcomers of that month in
# each repository. Since the same repositories appear in every month, the
# paired Wilcoxon between the 66 pairs of months (BH) is reported too. First
# globally, then per language with MIN_REPOS_BREAKDOWN repositories (one BH
# family per language).
MONTHS_SK = range(-6, 6)
_months = df_newcomers_windows[df_newcomers_windows["relative_month"].isin(MONTHS_SK)].pivot(
    index="Repository", columns="relative_month", values="newcomers"
)
assert _months.notna().all().all() and len(_months) == df_repo_threshold["Repository"].nunique()
_lang_months = df_newcomers_windows.drop_duplicates("Repository").set_index("Repository")[
    "Language"
]
_n_lang = _lang_months.value_counts()
_n_lang = _n_lang[_n_lang >= MIN_REPOS_BREAKDOWN]
assert _n_lang.to_dict() == {l: len(v) for l, v in _lang.items()}


def _window_sk(m: int) -> str:
    return "earlier" if m < -6 else ("before" if m < 0 else "after")


def _sk_months(tab: pd.DataFrame, months, prefix: str):
    """SK-ESD and paired Wilcoxon (BH per language) between the months of `tab`
    (repos x relative month), globally and per language in _n_lang. Writes
    {prefix}_scott_knott.csv and {prefix}_pairs.csv.
    """
    sk, pairs = [], []
    for label, t in [("global", tab)] + [
        (l, tab[_lang_months.reindex(tab.index) == l]) for l in _n_lang.index
    ]:
        groups = {m: t[m] for m in months}
        rank = scott_knott_esd(groups)
        sk += [
            {
                "language": label,
                "relative_month": m,
                "window": _window_sk(m),
                "repositories": len(v),
                "median": v.median(),
                "mean": v.mean(),
                # same scale as the DiD figures: expm1(mean of log1p)
                "geometric_mean": np.expm1(np.log1p(v).mean()),
                "scott_knott_esd": rank[m],
            }
            for m, v in groups.items()
        ]
        p = apply_fdr(run_group_comparison(groups, "newcomers_month", paired=True, shapiro=False))
        p.insert(0, "language", label)
        p["window_1"] = p["group_1"].map(_window_sk)
        p["window_2"] = p["group_2"].map(_window_sk)
        pairs.append(p)
    df_sk = pd.DataFrame(sk)
    df_sk.to_csv(f"{OUTPUT_DIR}/{prefix}_scott_knott.csv", index=False)
    df_p = pd.concat(pairs, ignore_index=True)
    df_p.to_csv(f"{OUTPUT_DIR}/{prefix}_pairs.csv", index=False)
    print(f"\n  {prefix}_scott_knott.csv, {prefix}_pairs.csv")
    print(
        df_sk.pivot(
            index="language", columns="relative_month", values="scott_knott_esd"
        ).to_string()
    )
    return df_sk, df_p


df_sk_months, df_pairs_months = _sk_months(_months, MONTHS_SK, "rq1_months")

# Third window: the 6 months before -6 (-12..-7). The newcomer of a month is
# still the author of the first PR since the start of the baseline, so months
# -6..+5 are the same as above. In repositories with N < 12, part of -12..-7
# falls in the baseline, where the history is up to 6 months shorter.
MONTHS_SK3 = range(-12, 6)
_first_all = con.execute("""
    WITH first AS (
        SELECT h.Author, h.Repository, MIN(h.Month) AS first_month
        FROM humans_all h JOIN repo_threshold rt ON h.Repository = rt.Repository
        WHERE h.Month >= rt.baseline_start
        GROUP BY h.Author, h.Repository
    )
    SELECT Repository, first_month AS Month, COUNT(*) AS newcomers
    FROM first GROUP BY Repository, first_month
""").df()
_cnt_all = {(r, m): n for r, m, n in _first_all.itertuples(index=False)}
assert (
    df_repo_threshold["baseline_start"]
    <= df_repo_threshold["threshold"].map(lambda t: str(pd.Period(t, freq="M") - 12))
).all()
_months3 = pd.DataFrame(
    {
        m: {
            rt.Repository: _cnt_all.get(
                (rt.Repository, str(pd.Period(rt.threshold, freq="M") + m)), 0
            )
            for rt in df_repo_threshold.itertuples(index=False)
        }
        for m in MONTHS_SK3
    }
)
_months3.index.name = "Repository"
# months -6..+5 match the RQ1 count
assert (
    _months3[list(MONTHS_SK)]
    .sort_index()
    .equals(_months.sort_index().astype(_months3.dtypes.iloc[0]))
)
_months3.reset_index().melt(
    id_vars="Repository", var_name="relative_month", value_name="newcomers"
).assign(
    Language=lambda d: d["Repository"].map(_lang_months),
    window=lambda d: d["relative_month"].map(_window_sk),
).to_csv(f"{OUTPUT_DIR}/rq1_months3_newcomers.csv", index=False)
df_sk_months3, df_pairs_months3 = _sk_months(_months3, MONTHS_SK3, "rq1_months3")
_ad = df_pairs_months3[df_pairs_months3["window_1"] != df_pairs_months3["window_2"]].groupby(
    ["language", "window_1", "window_2"]
)
print("  significant pairs between windows (p_fdr < 0.05):")
print(
    pd.DataFrame({"significant": _ad["significant"].sum(), "pairs": _ad.size()})
    .loc["global"]
    .to_string()
)


def _spearman_partial(df: pd.DataFrame, x: str, y: str, controls: list[str]):
    d = df.dropna(subset=[x, y, *controls])
    Z = np.column_stack([np.ones(len(d))] + [stats.rankdata(d[c]) for c in controls])
    res = lambda v: (
        stats.rankdata(d[v]) - Z @ np.linalg.lstsq(Z, stats.rankdata(d[v]), rcond=None)[0]
    )
    r = stats.pearsonr(res(x), res(y))
    return r.statistic, r.pvalue, len(d)


# Association of AI intensity with the after-window trend and with the
# before x after change, controlling for size and N. The main family uses the
# intensity without after-window newcomer PRs; BH per family.
df_intensity = pd.DataFrame(
    [
        {
            "family_bh": fam,
            "variable": y,
            "ai": ai_col,
            **dict(
                zip(
                    ["spearman_partial", "p", "repositories"],
                    _spearman_partial(
                        df_rq1_repo, ai_col, y, ["baseline_authors_per_month", "n_months"]
                    ),
                )
            ),
        }
        for fam, cols in (
            ("no_newcomers", ("fraction_ai_no_newcomers", "prs_ai_no_newcomers")),
            ("all_circular", ("fraction_ai", "prs_ai")),
        )
        for y in (
            "tau_newcomers_after",
            "tau_newcomer_fraction_after",
            "level_change_newcomers",
            "tau_diff_newcomers",
        )
        for ai_col in cols
    ]
)
df_intensity["p_fdr"] = df_intensity.groupby("family_bh")["p"].transform(
    lambda p: stats.false_discovery_control(p, method="bh")
)
df_intensity.to_csv(f"{OUTPUT_DIR}/rq1_intensity_ai.csv", index=False)
print("\n  rq1_intensity_ai.csv")
print(df_intensity.round(4).to_string(index=False))

# Repositories without AI: newcomer flow over the last 12 months. Repositories
# without any AI-evidence PR, followed over the 12 full months before the
# cutoff, with the same newcomer rule (threshold at the start of the year,
# N = 12). Descriptive reference for the adopters' series (plot_no_ai.py).
NO_AI_MONTHS = 12
_no_ai_start = str(pd.Period(CUTOFF_MONTH, freq="M") - NO_AI_MONTHS)
_no_ai_base = str(pd.Period(CUTOFF_MONTH, freq="M") - NO_AI_MONTHS - BASELINE_MONTHS)
_no_ai = con.execute(f"""
    WITH repos AS (
        SELECT DISTINCT Repository FROM humans
        WHERE Repository NOT IN (SELECT DISTINCT Repository FROM ai_raw)
    ),
    first AS (
        SELECT h.Repository, h.Author, MIN(h.Month) AS first_month
        FROM humans_all h JOIN repos r USING (Repository)
        WHERE h.Month >= '{_no_ai_base}'
        GROUP BY 1, 2
    )
    SELECT Repository, first_month AS Month, COUNT(*) AS newcomers
    FROM first WHERE first_month >= '{_no_ai_start}'
    GROUP BY 1, 2
""").df()
_repos_no_ai = con.execute(
    "SELECT DISTINCT Repository FROM humans "
    "WHERE Repository NOT IN (SELECT DISTINCT Repository FROM ai_raw)"
).df()["Repository"]
_cnt_no_ai = {(r, m): n for r, m, n in _no_ai.itertuples(index=False)}
df_no_ai = pd.DataFrame(
    [
        {
            "Repository": r,
            "Language": _lang_repo.get(r),
            "Month": str(pd.Period(_no_ai_start, freq="M") + k),
            "relative_month": k,
            "newcomers": _cnt_no_ai.get((r, str(pd.Period(_no_ai_start, freq="M") + k)), 0),
        }
        for r in sorted(_repos_no_ai)
        for k in range(NO_AI_MONTHS)
    ]
)
df_no_ai.to_csv(f"{OUTPUT_DIR}/rq1_no_ai_newcomers_per_repository.csv", index=False)
df_no_ai_month = (
    df_no_ai.groupby("Month")["newcomers"]
    .agg(["sum", "count", "mean"])
    .rename(columns={"sum": "newcomers", "count": "repositories", "mean": "mean_per_repository"})
    .reset_index()
)
df_no_ai_month.to_csv(f"{OUTPUT_DIR}/rq1_no_ai_newcomers_month.csv", index=False)
df_no_ai_trend = _trend_repos(df_no_ai)
pd.DataFrame(
    [_summary_trend(df_no_ai_trend, f"no_ai:{_no_ai_start}..{df_no_ai['Month'].max()}")]
).to_csv(f"{OUTPUT_DIR}/rq1_no_ai_trend.csv", index=False)
print(
    f"\n  rq1_no_ai_newcomers_*.csv, rq1_no_ai_trend.csv  ({len(_repos_no_ai)} repos without AI, "
    f"{_no_ai_start} to {df_no_ai['Month'].max()}, {int(df_no_ai['newcomers'].sum()):,} newcomers)"
)
print(pd.DataFrame([_summary_trend(df_no_ai_trend, "no_ai")]).round(4).to_string(index=False))

# Newcomer retention. A newcomer is retained at k if they open any PR (with or
# without AI) in the k months after the entry month; a second PR in the entry
# month does not count. Follow-up stays inside the window, so only newcomers
# who entered at least k months before its end are eligible. In the after
# window, newcomers are split by whether their first PR from the entry month
# on has AI evidence. The per-repository rate needs MIN_N_PER_REPO eligible
# newcomers. BH: one family per k.
RETENTION_K = 3
RETENTION_K_SENS = (1, 6)
_KS_RET = (RETENTION_K, *RETENTION_K_SENS)

df_ret_newc = con.execute(f"""
    WITH nxt AS (
        SELECT n.Repository, n.Author, MIN(h.Month) AS nxt_month
        FROM newcomers_windows n
        JOIN humans_all h ON h.Author = n.Author AND h.Repository = n.Repository
        WHERE h.Month > n.first_month
        GROUP BY 1, 2
    ),
    -- from the entry month on (a returning author has PRs before the baseline)
    debut AS (
        SELECT h.Repository, h.Author, h.PR_Number,
               ROW_NUMBER() OVER (
                   PARTITION BY h.Repository, h.Author
                   ORDER BY h.Created_At, TRY_CAST(h.PR_Number AS BIGINT)
               ) AS rn
        FROM humans_all h
        JOIN newcomers_windows n ON h.Author = n.Author AND h.Repository = n.Repository
        WHERE h.Month >= n.first_month
    ),
    ai_human AS (
        SELECT DISTINCT Repository, PR_Number FROM ai
        WHERE Author_Category = 'Human' AND Month < '{CUTOFF_MONTH}'
    )
    SELECT n.Repository, n.Author, n."window", n.first_month, n.threshold,
           p.nxt_month, i.PR_Number IS NOT NULL AS debut_with_ai
    FROM newcomers_windows n
    JOIN debut e ON e.Repository = n.Repository AND e.Author = n.Author AND e.rn = 1
    LEFT JOIN nxt p ON p.Repository = n.Repository AND p.Author = n.Author
    LEFT JOIN ai_human i ON i.Repository = e.Repository AND i.PR_Number = e.PR_Number
""").df()
assert len(df_ret_newc) == con.execute("SELECT COUNT(*) FROM newcomers_windows").fetchone()[0]


def _month_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str[:4]) * 12 + pd.to_numeric(s.str[5:7])


_m0 = _month_int(df_ret_newc["first_month"])
_nxt = _month_int(df_ret_newc["nxt_month"])
_end = np.where(
    df_ret_newc["window"] == "before",
    _month_int(df_ret_newc["threshold"]) - 1,
    _month_int(pd.Series([CUTOFF_MONTH])).iloc[0] - 1,
)
_after = df_ret_newc["window"] == "after"
_cells_ret = {
    "before": ~_after,
    "after": _after,
    "after_debut_with_ai": _after & df_ret_newc["debut_with_ai"],
    "after_debut_no_ai": _after & ~df_ret_newc["debut_with_ai"],
}
df_ret_repo = df_rq1_repo[["Repository", "Language", "tier_stars", "tier_size"]].copy()
for k in _KS_RET:
    _elig = _end - _m0 >= k
    _retained = _elig & (_nxt - _m0 <= k)
    for cell, mask in _cells_ret.items():
        g = (
            pd.DataFrame({"e": _elig & mask, "r": _retained & mask})
            .groupby(df_ret_newc["Repository"])
            .sum()
        )
        e = df_ret_repo["Repository"].map(g["e"]).fillna(0).astype(int)
        r = df_ret_repo["Repository"].map(g["r"]).fillna(0).astype(int)
        df_ret_repo[f"eligible_k{k}_{cell}"] = e
        df_ret_repo[f"retained_k{k}_{cell}"] = r
        df_ret_repo[f"rate_k{k}_{cell}"] = (r / e).where(e >= MIN_N_PER_REPO)
df_ret_repo[f"diff_rate_k{RETENTION_K}"] = (
    df_ret_repo[f"rate_k{RETENTION_K}_after"] - df_ret_repo[f"rate_k{RETENTION_K}_before"]
)
df_ret_repo.to_csv(f"{OUTPUT_DIR}/rq1_retention_per_repository.csv", index=False)
print(
    f"\n  rq1_retention_per_repository.csv  ({len(df_ret_newc):,} newcomers; "
    f"debut with AI 'after': {int(df_ret_newc.loc[_after, 'debut_with_ai'].sum()):,} "
    f"of {int(_after.sum()):,})"
)

# comparison -> ((label, cell), (label, cell)); delta > 0 = larger in the first
COMPARISONS_RET = {
    "after_vs_before": (("after", "after"), ("before", "before")),
    "debut_with_ai_vs_no_ai": (
        ("debut_with_ai", "after_debut_with_ai"),
        ("debut_no_ai", "after_debut_no_ai"),
    ),
}


def _compare_retention(d: pd.DataFrame, k: int, comparisons: dict) -> pd.DataFrame:
    """Paired Wilcoxon on the per-repository rates (index = Repository), with
    the aggregate (summed eligible and retained) on each side.
    """
    parts = []
    for name, ((l1, c1), (l2, c2)) in comparisons.items():
        desc = {}
        for i, c in ((1, c1), (2, c2)):
            e, r = int(d[f"eligible_k{k}_{c}"].sum()), int(d[f"retained_k{k}_{c}"].sum())
            desc |= {f"eligible_{i}": e, f"retained_{i}": r, f"aggregated_rate_{i}": r / e}
        parts.append(
            run_group_comparison(
                {l1: d[f"rate_k{k}_{c1}"], l2: d[f"rate_k{k}_{c2}"]},
                f"retention_k{k}",
                paired=True,
                shapiro=False,
            ).assign(k=k, comparison=name, **desc)
        )
    return pd.concat(parts, ignore_index=True)


_d_ret = df_ret_repo.set_index("Repository")
df_ret_global = pd.concat(
    [
        apply_fdr(_compare_retention(_d_ret, k, COMPARISONS_RET)).assign(family_bh=f"k{k}")
        for k in _KS_RET
    ],
    ignore_index=True,
)
_first = [
    "family_bh",
    "k",
    "comparison",
    "group_1",
    "group_2",
    "eligible_1",
    "retained_1",
    "aggregated_rate_1",
    "eligible_2",
    "retained_2",
    "aggregated_rate_2",
]
df_ret_global = df_ret_global[_first + [c for c in df_ret_global.columns if c not in _first]]
df_ret_global.to_csv(f"{OUTPUT_DIR}/rq1_retention_global.csv", index=False)
print("\n  rq1_retention_global.csv (paired Wilcoxon on per-repository rates)")
print(
    df_ret_global[
        [
            "k",
            "group_1",
            "group_2",
            "eligible_1",
            "aggregated_rate_1",
            "eligible_2",
            "aggregated_rate_2",
            "n1",
            "median_1",
            "median_2",
            "paired_median_diff",
            "mwu_p",
            "mwu_p_fdr",
            "effect",
            "magnitude",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# Breakdowns (main k): after x before within each category (BH across the
# categories) and Kruskal-Wallis on the rate difference between them (BH
# across breakdowns). A category is testable with at least
# MIN_REPOS_BREAKDOWN repositories with a rate in both windows.
_k = RETENTION_K
_ret_desc, _ret_tests, _ret_kw = [], [], []
for name, col_brk in BREAKDOWNS.items():
    testable = []
    for cat, g in df_ret_repo.groupby(col_brk, observed=True):
        with_rate = g[f"diff_rate_k{_k}"].notna()
        _ret_desc.append(
            {
                "breakdown": name,
                "category": cat,
                "repositories": len(g),
                "repositories_with_rate": int(with_rate.sum()),
                **{
                    f"{v}_{j}": int(g[f"{v}_k{_k}_{j}"].sum())
                    for j in ("before", "after")
                    for v in ("eligible", "retained")
                },
                "median_rate_before": g.loc[with_rate, f"rate_k{_k}_before"].median(),
                "median_rate_after": g.loc[with_rate, f"rate_k{_k}_after"].median(),
                "diff_rate_median": g[f"diff_rate_k{_k}"].median(),
            }
        )
        if with_rate.sum() >= MIN_REPOS_BREAKDOWN:
            testable.append(g)
            _ret_tests.append(
                _compare_retention(
                    g.set_index("Repository"),
                    _k,
                    {"after_vs_before": COMPARISONS_RET["after_vs_before"]},
                ).assign(breakdown=name, category=cat)
            )
    kw = stats.kruskal(*[g[f"diff_rate_k{_k}"].dropna() for g in testable])
    _ret_kw.append(
        {
            "breakdown": name,
            "variable": f"diff_rate_k{_k}",
            "categories": len(testable),
            "repositories": sum(g[f"diff_rate_k{_k}"].notna().sum() for g in testable),
            "H": kw.statistic,
            "p": kw.pvalue,
        }
    )
_tests_ret = pd.concat(
    [apply_fdr(g) for _, g in pd.concat(_ret_tests, ignore_index=True).groupby("breakdown")],
    ignore_index=True,
)
df_ret_brk = pd.DataFrame(_ret_desc).merge(
    _tests_ret[
        [
            "breakdown",
            "category",
            "n1",
            "paired_median_diff",
            "mwu_p",
            "mwu_p_fdr",
            "effect",
            "magnitude",
        ]
    ].rename(columns={"n1": "repositories_paired"}),
    on=["breakdown", "category"],
    how="left",
)
df_ret_brk.to_csv(f"{OUTPUT_DIR}/rq1_retention_breakdowns.csv", index=False)
df_ret_kw = pd.DataFrame(_ret_kw)
df_ret_kw["p_fdr"] = stats.false_discovery_control(df_ret_kw["p"], method="bh")
df_ret_kw.to_csv(f"{OUTPUT_DIR}/rq1_retention_kruskal.csv", index=False)
print(f"\n  rq1_retention_breakdowns.csv, rq1_retention_kruskal.csv (k={_k})")
print(df_ret_brk.round(3).to_string(index=False))
print(df_ret_kw.round(4).to_string(index=False))

# RQ1 robustness
# 1) Threshold origin: in how many repositories the months of the pair have AI
#    signals only from PRs of after-window newcomers.
# 2) Prior-member threshold ("insiders"): the same rule, counting only AI
#    signals from authors whose first PR in the repository is from an earlier
#    month. Sensitivity only; the main definition does not change.
# 3) Placebo: pseudo-thresholds drawn from the adopters' distribution for
#    repositories without AI and with sporadic AI (no valid threshold).
# 4) AI debut: how much of the newcomer increase comes from newcomers whose
#    first PR has AI evidence, and the level without them.
print("\n" + "=" * 70)
print("RQ1 - ROBUSTNESS")
print("=" * 70)

_signals_author = con.execute(f"""
    WITH first_pr AS (
        SELECT Repository, Author, MIN(Month) AS m0 FROM humans_all GROUP BY 1, 2
    )
    SELECT i.Repository, i.Signal_Month,
           i.Author_Category = 'Human' AND n.Author IS NOT NULL AS by_newcomer_after,
           i.Author_Category <> 'Human' OR COALESCE(p.m0 < i.Signal_Month, FALSE) AS by_prior_member
    FROM ai i
    LEFT JOIN newcomers n ON n.Repository = i.Repository AND n.Author = i.Author
    LEFT JOIN first_pr p ON p.Repository = i.Repository AND p.Author = i.Author
    WHERE i.Signal_Month IS NOT NULL AND i.Signal_Month < '{CUTOFF_MONTH}'
""").df()

# 2) Prior-member threshold.
df_thr_insiders = _thresholds(
    _signals_author.loc[
        _signals_author["by_prior_member"], ["Repository", "Signal_Month"]
    ].drop_duplicates()
)
_thr_ins = df_thr_insiders.set_index("Repository")["threshold"]

# 1) Origin of the main threshold, per repository.
_only_newc = _signals_author.groupby(["Repository", "Signal_Month"])["by_newcomer_after"].all()
_origin = df_rq1_repo[["Repository", "threshold", "tier_size"]].copy()
_origin["month0_only_newcomers"] = [
    _only_newc.get((r, t), False) for r, t in zip(_origin["Repository"], _origin["threshold"])
]
_origin["month1_only_newcomers"] = [
    _only_newc.get((r, str(pd.Period(t, freq="M") + 1)), False)
    for r, t in zip(_origin["Repository"], _origin["threshold"])
]
_origin["threshold_insiders"] = _origin["Repository"].map(_thr_ins)
_origin.to_csv(f"{OUTPUT_DIR}/rq1_threshold_origin_per_repository.csv", index=False)

# PRs with a commit signal dropped because a Bot/Organization-typed account
# outside bots.csv opened them (classify_prs.py), per repository, by creation
# month relative to the threshold (empty threshold: repository without one).
_auto = pd.read_csv(f"{OUTPUT_DIR}/automation_with_ai_evidence.csv")
_auto = _auto[
    (_auto["Drop_Reason"] == "account_type")
    & _auto["Detection_Source"].isin(["commit-author", "commit-coauthor"])
].copy()
_auto["threshold"] = _auto["Repository"].map(_origin.set_index("Repository")["threshold"])
_auto["before_threshold"] = _auto["threshold"].notna() & (
    _auto["Created_At"].str[:7] < _auto["threshold"]
)
_auto_thr = (
    _auto.groupby("Repository", sort=False)
    .agg(
        threshold=("threshold", "first"),
        prs=("PR_Number", "size"),
        prs_before_threshold=("before_threshold", "sum"),
    )
    .reset_index()
    .sort_values("prs", ascending=False)
)
_auto_thr.to_csv(f"{OUTPUT_DIR}/automation_account_type_threshold.csv", index=False)
print(
    f"  automation_account_type_threshold.csv: {_auto_thr['prs'].sum()} PRs, "
    f"{_auto_thr.loc[_auto_thr['threshold'].notna(), 'prs'].sum()} in repositories with a "
    f"threshold, {_auto_thr['prs_before_threshold'].sum()} before it"
)


def _summary_origin(g: pd.DataFrame, label: str) -> dict:
    return {
        "breakdown": label,
        "repositories": len(g),
        "some_month_only_newcomers": int(
            (g["month0_only_newcomers"] | g["month1_only_newcomers"]).sum()
        ),
        "two_months_only_newcomers": int(
            (g["month0_only_newcomers"] & g["month1_only_newcomers"]).sum()
        ),
        "with_threshold_insiders": int(g["threshold_insiders"].notna().sum()),
        "same_threshold": int((g["threshold_insiders"] == g["threshold"]).sum()),
    }


df_origin = pd.DataFrame(
    [_summary_origin(_origin, "global")]
    + [_summary_origin(g, f"size:{q}") for q, g in _origin.groupby("tier_size", observed=True)]
)
df_origin.to_csv(f"{OUTPUT_DIR}/rq1_threshold_origin.csv", index=False)
print(
    f"\n  Prior-member threshold: {len(df_thr_insiders)} repos "
    f"({len(set(df_thr_insiders['Repository']) - set(df_rq1_repo['Repository']))} without the main threshold)"
)
print("  rq1_threshold_origin.csv")
print(df_origin.to_string(index=False))


def _with_breakdowns(r: pd.DataFrame) -> pd.DataFrame:
    """Language, stars and quartiles recomputed within the sample."""
    r = r.copy()
    r["Language"] = r["Repository"].map(_lang_repo)
    r["stars"] = (
        r["Repository"]
        .map(df_rq1_repo.set_index("Repository")["stars"])
        .fillna(
            r["Repository"].map(pd.read_csv("results/repo_metrics.csv").set_index("repo")["stars"])
        )
    )
    r["tier_stars"] = _quartiles(r["stars"])
    r["tier_size"] = _quartiles(r["baseline_authors_per_month"])
    return r


_rob_repo = {"main": df_rq1_repo, "insiders": _with_breakdowns(_windows_repo(df_thr_insiders))}
_rob_repo["insiders"].to_csv(f"{OUTPUT_DIR}/rq1_insiders_per_repository.csv", index=False)

# Global: level, tau and newcomers x veterans (paired Wilcoxon; BH per threshold).
_rob_global = []
for name, r in _rob_repo.items():
    d = r.set_index("Repository")
    parts = [
        run_group_comparison(
            {"after": d["mean_newcomers_after"], "before": d["mean_newcomers_before"]},
            "mean_newcomers",
            paired=True,
            shapiro=False,
        ),
        run_group_comparison(
            {"after": d["tau_newcomers_after"], "before": d["tau_newcomers_before"]},
            "tau_newcomers",
            paired=True,
            shapiro=False,
        ),
        run_group_comparison(
            {"newcomers": d["log2_newcomers"], "veterans": d["log2_veterans"]},
            "log2_growth_newcomers_vs_veterans",
            paired=True,
            shapiro=False,
        ),
    ]
    _rob_global.append(
        apply_fdr(pd.concat(parts, ignore_index=True)).assign(threshold=name, repositories=len(r))
    )
df_rob_global = pd.concat(_rob_global, ignore_index=True)
df_rob_global.to_csv(f"{OUTPUT_DIR}/rq1_robustness_global.csv", index=False)
print("\n  rq1_robustness_global.csv")
print(
    df_rob_global[
        [
            "threshold",
            "metric",
            "n1",
            "median_1",
            "median_2",
            "paired_median_diff",
            "mwu_p_fdr",
            "effect",
            "magnitude",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# Breakdowns: descriptive per category and Kruskal-Wallis across them (BH per
# threshold).
_VARS_ROB = {
    "main": ("peak_months_0_1",),
    "insiders": ("level_change_newcomers", "did_newcomers_veterans", "peak_months_0_1"),
}
_rob_desc, _rob_kw = [], []
for name, r in _rob_repo.items():
    for brk, col_brk in BREAKDOWNS.items():
        testable = []
        for cat, g in r.groupby(col_brk, observed=True):
            _rob_desc.append(
                {
                    "threshold": name,
                    "breakdown": brk,
                    "category": cat,
                    "repositories": len(g),
                    **{
                        f"{v}_median": g[v].median()
                        for v in (
                            "level_change_newcomers",
                            "did_newcomers_veterans",
                            "peak_months_0_1",
                        )
                    },
                }
            )
            if len(g) >= MIN_REPOS_BREAKDOWN:
                testable.append(g)
        for var in _VARS_ROB[name]:
            kw = stats.kruskal(*[g[var].dropna() for g in testable])
            _rob_kw.append(
                {
                    "threshold": name,
                    "breakdown": brk,
                    "variable": var,
                    "categories": len(testable),
                    "repositories": sum(g[var].notna().sum() for g in testable),
                    "H": kw.statistic,
                    "p": kw.pvalue,
                }
            )
df_rob_desc = pd.DataFrame(_rob_desc)
df_rob_desc.to_csv(f"{OUTPUT_DIR}/rq1_robustness_breakdowns.csv", index=False)
df_rob_kw = pd.DataFrame(_rob_kw)
df_rob_kw["p_fdr"] = df_rob_kw.groupby("threshold")["p"].transform(
    lambda p: stats.false_discovery_control(p, method="bh")
)
df_rob_kw.to_csv(f"{OUTPUT_DIR}/rq1_robustness_kruskal.csv", index=False)
print("\n  rq1_robustness_breakdowns.csv, rq1_robustness_kruskal.csv")
print(df_rob_desc[df_rob_desc["breakdown"] == "size"].round(3).to_string(index=False))
print(df_rob_kw.round(4).to_string(index=False))

# 3) Placebo. Groups: repositories without any AI PR in the dataset and
# repositories with an AI signal before the cutoff but no valid threshold. Each
# draw gives every placebo repository a threshold from the adopters'
# distribution; adopters x placebo by Mann-Whitney on newcomer growth and did.
PLACEBO_DRAWS = 100
_rng = np.random.default_rng(42)
_repos_with_signal = set(_signals_per_repo["Repository"])
_placebo = []
for name, r in _rob_repo.items():
    groups = {
        "no_ai": sorted(_repos_no_ai),
        "ai_sporadic": sorted(_repos_with_signal - set(r["Repository"])),
    }
    for group, repos in groups.items():
        res = {v: [] for v in ("log2_newcomers", "log2_newcomers_fixed", "did_newcomers_veterans")}
        for _ in range(PLACEBO_DRAWS):
            thr = pd.DataFrame(
                [
                    _row_threshold(rp, t)
                    for rp, t in zip(repos, _rng.choice(r["threshold"].to_numpy(), len(repos)))
                ]
            )
            pl = _windows_repo(thr, with_tau=False)
            for v in res:
                a, b = r[v].dropna(), pl[v].dropna()
                u = stats.mannwhitneyu(a, b, alternative="two-sided")
                res[v].append(
                    (len(b), b.median(), u.pvalue, cliffs_delta_from_u(u.statistic, len(a), len(b)))
                )
        for v, lst in res.items():
            arr = np.array(lst)
            _placebo.append(
                {
                    "threshold": name,
                    "group_placebo": group,
                    "variable": v,
                    "repos_adopters": int(r[v].notna().sum()),
                    "median_adopters": r[v].median(),
                    "repos_placebo": len(repos),
                    "repos_placebo_with_value": np.median(arr[:, 0]),
                    "median_placebo": np.median(arr[:, 1]),
                    "p_median": np.median(arr[:, 2]),
                    "fraction_draws_p_less_alpha": (arr[:, 2] < ALPHA).mean(),
                    "cliffs_delta_median": np.median(arr[:, 3]),
                    "draws": PLACEBO_DRAWS,
                }
            )
df_placebo = pd.DataFrame(_placebo)
df_placebo.to_csv(f"{OUTPUT_DIR}/rq1_placebo.csv", index=False)
print("\n  rq1_placebo.csv (median across draws)")
print(df_placebo.round(4).to_string(index=False))

# 4) AI debut (main threshold).
_ai_debut = (
    df_ret_newc[_after & df_ret_newc["debut_with_ai"]]
    .groupby("Repository")
    .size()
    .reindex(df_rq1_repo["Repository"])
    .fillna(0)
)
_d = df_rq1_repo.set_index("Repository")
_no_ai_debut = (_d["total_newcomers_after"] - _ai_debut) / _d["n_months"]
_tot_a, _tot_d, _tot_ai = (
    _d["total_newcomers_before"].sum(),
    _d["total_newcomers_after"].sum(),
    _ai_debut.sum(),
)
df_ai_debut = run_group_comparison(
    {"after_no_ai_debut": _no_ai_debut, "before": _d["mean_newcomers_before"]},
    "mean_newcomers",
    paired=True,
    shapiro=False,
).assign(
    newcomers_before=int(_tot_a),
    newcomers_after=int(_tot_d),
    newcomers_after_ai_debut=int(_tot_ai),
    fraction_ai_debut=_tot_ai / _tot_d,
    share_of_increase=_tot_ai / (_tot_d - _tot_a),
)
df_ai_debut.to_csv(f"{OUTPUT_DIR}/rq1_ai_debut.csv", index=False)
print("\n  rq1_ai_debut.csv")
print(df_ai_debut.round(4).T.to_string())

print("\n" + "=" * 70)
print("RQ2 — COMPARING METRICS ACROSS CONTRIBUTION GROUPS")
print("=" * 70)

# Aligned subset for RQ2 window 2: only repositories with a valid threshold and
# only PRs created from that repository's threshold on, for every group. The
# window-2 newcomers are the RQ1 `newcomers` view.
con.execute("""
    CREATE OR REPLACE VIEW humans_al AS
    SELECT h.* FROM humans h
    JOIN repo_threshold rt ON h.Repository = rt.Repository
    WHERE h.Month >= rt.threshold
""")
con.execute(f"""
    CREATE OR REPLACE VIEW ai_al AS
    SELECT i.* FROM ai i
    JOIN repo_threshold rt ON i.Repository = rt.Repository
    WHERE strftime('%Y-%m', i.Created_At) >= rt.threshold
      AND strftime('%Y-%m', i.Created_At) < '{CUTOFF_MONTH}'
""")
for _all, _al in (("ai", "ai_al"), ("humans", "humans_al")):
    _n0 = con.execute(f"SELECT COUNT(*) FROM {_all}").fetchone()[0]
    _n1 = con.execute(f"SELECT COUNT(*) FROM {_al}").fetchone()[0]
    print(f"  aligned subset ({_al}): {_n1:,} of {_n0:,} PRs")

print(
    f"  newcomers identified (RQ1 definition, Author+Repository pairs): "
    f"{con.execute('SELECT COUNT(*) FROM newcomers').fetchone()[0]:,}"
)

# Debut PR of each newcomer (used to build the survey sample).
df_newcomers_prs = con.execute(f"""
    WITH first_pr AS (
        SELECT h.Repository, h.PR_Number, h.Author, h.Created_At, h.Language,
               ROW_NUMBER() OVER (
                   PARTITION BY h.Author, h.Repository ORDER BY h.Created_At
               ) AS rn
        FROM humans_all h
        JOIN newcomers n ON h.Author = n.Author AND h.Repository = n.Repository
        WHERE h.Month >= n.threshold
    )
    , ai_human AS (
        SELECT Repository, PR_Number, Author, Created_At FROM ai
        WHERE Author_Category = 'Human' AND Month < '{CUTOFF_MONTH}'
    )
    -- AI_Debut: the debut PR itself is Human AI. N_AI_PRs: Human AI PRs of
    -- the newcomer in the repository, from the debut to the cutoff.
    SELECT p.Repository, p.PR_Number, p.Author, p.Created_At, p.Language,
           EXISTS (SELECT 1 FROM ai_human i
                   WHERE i.Repository = p.Repository
                     AND i.PR_Number = p.PR_Number) AS AI_Debut,
           (SELECT COUNT(*) FROM ai_human i
            WHERE i.Repository = p.Repository AND i.Author = p.Author
              AND i.Created_At >= p.Created_At) AS N_AI_PRs
    FROM first_pr p
    WHERE rn = 1
    ORDER BY p.Repository, p.Created_At, p.PR_Number
""").df()
df_newcomers_prs.to_csv(f"{OUTPUT_DIR}/newcomers_prs.csv", index=False)
print(f"  newcomers_prs.csv  ({len(df_newcomers_prs)} debut PRs)")

# RQ2 groups. Window 1 (before, [baseline_end, threshold)): PRs without AI
# only, from that window's newcomers (Newcomer) and from everyone else
# (Veteran). Window 2 (after, aligned subset): AI_Agent, plus human PRs split
# into newcomer/veteran and with/without AI. Groups are disjoint within each
# window. Human AI PRs without a login cannot be assigned and are dropped. A
# window-1 newcomer is a veteran in window 2.
con.execute("""
    CREATE OR REPLACE VIEW humans_before AS
    SELECT h.* FROM humans h
    JOIN repo_threshold rt ON h.Repository = rt.Repository
    WHERE h.Month >= rt.baseline_end AND h.Month < rt.threshold
""")
_NEWC = "n.Author = x.Author AND n.Repository = x.Repository"
GROUPS_RQ2 = {
    "window1": {
        "Newcomer": f"SELECT x.* FROM humans_before x JOIN newcomers_before n ON {_NEWC}",
        "Veteran": f"SELECT x.* FROM humans_before x ANTI JOIN newcomers_before n ON {_NEWC}",
    },
    "window2": {
        "AI_Agent": "SELECT x.* FROM ai_al x WHERE x.Author_Category = 'AI_Agent'",
        "Newcomer_AI": (
            f"SELECT x.* FROM ai_al x JOIN newcomers n ON {_NEWC} WHERE x.Author_Category = 'Human'"
        ),
        "Newcomer": f"SELECT x.* FROM humans_al x JOIN newcomers n ON {_NEWC}",
        "Veteran_AI": (
            f"SELECT x.* FROM ai_al x ANTI JOIN newcomers n ON {_NEWC} "
            "WHERE x.Author_Category = 'Human' AND NULLIF(TRIM(x.Author), '') IS NOT NULL"
        ),
        "Veteran": f"SELECT x.* FROM humans_al x ANTI JOIN newcomers n ON {_NEWC}",
    },
}
# Long table (one PR per row) with everything RQ2 uses. days = resolution
# time, merged PRs only.
df_rq2 = pd.concat(
    [
        con.execute(f"""
        SELECT '{j}' AS "window", '{g}' AS "group", Repository, Language, PR_Number,
               Author, State, Month, Closes_GFI, CAST(GFI_Issue_Numbers AS VARCHAR) AS GFI_Issue_Numbers,
               CASE WHEN State = 'merged' AND Closed_At IS NOT NULL
                    THEN (epoch(Closed_At) - epoch(Created_At)) / 86400.0 END AS days
        FROM ({sql}) z
    """).df()
        for j, groups in GROUPS_RQ2.items()
        for g, sql in groups.items()
    ],
    ignore_index=True,
)
assert not df_rq2.duplicated(["window", "Repository", "PR_Number"]).any(), (
    "a PR fell into two groups of the same window"
)
_n_ai_before = con.execute("""
    SELECT COUNT(*) FROM ai i JOIN repo_threshold rt ON i.Repository = rt.Repository
    WHERE i.Month >= rt.baseline_end AND i.Month < rt.threshold
""").fetchone()[0]
print(f"  PRs per window and group (AI PRs in window 1, outside RQ2: {_n_ai_before:,}):")
print(df_rq2.groupby(["window", "group"], sort=False).size().to_string())
# Repositories where GitHub's merge does not reflect acceptance (Gerrit,
# mailing lists, bots or maintainers that land the commit and close the PR,
# projects that do not accept external PRs) record accepted PRs as closed.
# They are left out of the main acceptance and resolution time analyses; the
# version with them is a sensitivity analysis (_with_external_merge, window 2
# and between windows only). Merge rate from results/repo_metrics.csv.
MIN_MERGE_RATE = 0.10
_rm = pd.read_csv("results/repo_metrics.csv")
_rm["merge_rate"] = _rm["merged_prs"] / _rm["prs"]
# Cases the 10% rule misses, added by hand after checking samples of PRs
# closed without merge on the GitHub timeline. "error" = (1 - merge rate) x
# share of closed-unmerged PRs that were integrated outside GitHub.
# - lwouis/alt-tab-macos: never uses GitHub's merge (0 merged of 206 PRs in
#   the period); acceptance is constantly 0
# - grpc/grpc: the copybara-service bot lands the commit and closes the PR;
#   error 0.63
# - angular/angular: until 2025-10 the project's merge tool closed the PR;
#   error 0.36
# - gnachman/iTerm2: the maintainer merges by hand ("Merged as <sha>"); error 0.34
# - php/php-src: pushing the contributor's commit closes the PR; error 0.33
# - NationalSecurityAgency/ghidra: integration through internal branches;
#   error 0.31
# - Genymobile/scrcpy: external PRs are merged by hand ("Merged into dev");
#   error 0.25
# Kept (error < 0.12): iSoron/uhabits 0.11, nodejs/node 0.08, junegunn/fzf
# 0.07, p0deje/Maccy 0.02, YunaiV/ruoyi-vue-pro 0.00; no other RQ2 repository
# goes above 0.07.
EXTERNAL_MERGE_MANUAL = [
    "lwouis/alt-tab-macos",
    "angular/angular",
    "grpc/grpc",
    "NationalSecurityAgency/ghidra",
    "Genymobile/scrcpy",
    "php/php-src",
    "gnachman/iTerm2",
]
_rm_outside = _rm[
    ((_rm["merge_rate"] < MIN_MERGE_RATE) | _rm["repo"].isin(EXTERNAL_MERGE_MANUAL))
    & _rm["repo"].isin(df_repo_threshold["Repository"])
]
_rm_outside[["repo", "prs", "merged_prs", "merge_rate"]].sort_values("merge_rate").to_csv(
    f"{OUTPUT_DIR}/rq2_repos_external_merge.csv", index=False
)
WITH_EXTERNAL_MERGE = False  # True only in the sensitivity analysis


# resolution time per PR (merged), the Mann-Whitney sample
df_rq2.loc[
    (df_rq2["days"] >= 0) & ~df_rq2["Repository"].isin(_rm_outside["repo"]),
    ["window", "group", "Repository", "PR_Number", "days"],
].to_csv(f"{OUTPUT_DIR}/rq2_resolution_time_per_pr.csv", index=False)


def _no_outside(d: pd.DataFrame) -> pd.DataFrame:
    """Drops the repositories that merge outside GitHub (except in the
    sensitivity analysis): there a merged PR is the exception, so neither
    acceptance nor resolution time measure the project's process.
    """
    return d if WITH_EXTERNAL_MERGE else d[~d["Repository"].isin(_rm_outside["repo"])]


def _resolution(d: pd.DataFrame) -> pd.Series:
    d = _no_outside(d)
    return d["days"][d["days"] >= 0].reset_index(drop=True)


def _acceptance(d: pd.DataFrame) -> pd.Series:
    """Merged / closed (merged + closed) per repository, with minimum N, without
    the repositories that merge outside GitHub (except in the sensitivity).
    """
    f = d[d["State"].isin(["merged", "closed"])]
    f = _no_outside(f)
    g = f.groupby("Repository")["State"].agg(closed="size", merged=lambda e: (e == "merged").sum())
    g = g[g["closed"] >= MIN_N_PER_REPO]
    return (g["merged"] / g["closed"]).rename("rate")


def _median_repo(d: pd.DataFrame) -> pd.Series:
    """Median resolution time per repository, with minimum N. This is the main
    time test (paired by repository): PRs of one repository are not
    independent and large repositories dominate the per-PR Mann-Whitney,
    which is kept as a sensitivity analysis.
    """
    d = _no_outside(d)
    m = d[d["days"] >= 0].groupby("Repository")["days"].agg(["median", "size"])
    return m.loc[m["size"] >= MIN_N_PER_REPO, "median"]


# metric -> (function, paired by repository)
METRICS_RQ2 = {
    "resolution_time_days": (_resolution, False),
    "acceptance_rate_per_repo": (_acceptance, True),
    "resolution_time_median_per_repo": (_median_repo, True),
}
# Pairs between windows: (window-1 group, window-2 group).
PAIRS_BETWEEN_WINDOWS = [
    ("Newcomer", "Newcomer"),
    ("Newcomer", "Newcomer_AI"),
    ("Veteran", "Veteran"),
    ("Veteran", "Veteran_AI"),
]


def _vectors(d: pd.DataFrame, window: str, metric: str, exclude=()) -> dict[str, pd.Series]:
    f = METRICS_RQ2[metric][0]
    dj = d[d["window"] == window]
    return {g: f(dj[dj["group"] == g]) for g in GROUPS_RQ2[window] if g not in exclude}


def _kruskal_safe(vectors: dict[str, pd.Series], metric_name: str) -> dict:
    try:
        return overall_kruskal(vectors, metric_name)
    except (TypeError, ValueError, IndexError):  # fewer than 2 groups with data
        return {"metric": metric_name, "groups": ", ".join(vectors), "H": np.nan, "p": np.nan}


def _rq2(
    d: pd.DataFrame,
    label: str = "",
    exclude_paired=(),
    fdr: bool = True,
    shapiro: bool = True,
    min_repos: int = 0,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Runs RQ2 on the long table `d`: groups within each window (Kruskal-Wallis
    first in window 2) and pairs between windows. Each (block, metric) is one
    BH family. Between windows, group 1 is the after window (delta > 0 =
    larger after). exclude_paired drops groups from the per-repository
    metrics (AI_Agent in the breakdowns). With min_repos, a group enters the
    tests only with at least min_repos repositories, and a paired comparison
    only with min_repos shared repositories.
    """
    out, kw = {}, []
    if min_repos:
        _m = _no_outside(d)
        _repos_pr = _m[_m["days"] >= 0].groupby(["window", "group"])["Repository"].nunique()
    for metric, (_, paired) in METRICS_RQ2.items():
        excl = exclude_paired if paired else ()
        v1 = _vectors(d, "window1", metric, excl)
        v2 = _vectors(d, "window2", metric, excl)
        if min_repos:
            v1, v2 = (
                {
                    g: v
                    for g, v in vs.items()
                    if (len(v) if paired else _repos_pr.get((j, g), 0)) >= min_repos
                }
                for j, vs in (("window1", v1), ("window2", v2))
            )
        kw.append(_kruskal_safe(v2, f"{metric}[window2{label}]"))
        blocks = {
            "window1": run_group_comparison(
                v1, f"{metric}[window1{label}]", paired=paired, shapiro=shapiro
            ),
            "window2": run_group_comparison(
                v2, f"{metric}[window2{label}]", paired=paired, shapiro=shapiro
            ),
            "between_windows": pd.concat(
                [
                    run_group_comparison(
                        {f"{g2}[after]": v2[g2], f"{g1}[before]": v1[g1]},
                        f"{metric}[between_windows{label}]",
                        paired=paired,
                        shapiro=False,
                    )
                    for g1, g2 in PAIRS_BETWEEN_WINDOWS
                    if g1 in v1 and g2 in v2
                ]
                or [pd.DataFrame()],
                ignore_index=True,
            ),
        }
        for block, df in blocks.items():
            if min_repos and paired and not df.empty:
                df = df[df["n1"] >= min_repos].reset_index(drop=True)
            out[f"{metric}:{block}"] = apply_fdr(df) if fdr else df
    return out, kw


def _gap(d: pd.DataFrame) -> pd.DataFrame:
    """Newcomer - veteran gap (PRs without AI only) per repository in each
    window, compared between windows with the paired Wilcoxon. For
    acceptance, gap < 0 = newcomers accepted less; for median time,
    gap > 0 = newcomers slower.
    """
    parts = []
    for metric in ("acceptance_rate_per_repo", "resolution_time_median_per_repo"):
        gaps = {}
        for j in ("window1", "window2"):
            v = _vectors(d, j, metric)
            gaps[j] = (v["Newcomer"] - v["Veteran"]).dropna()
        parts.append(
            run_group_comparison(
                {"gap[after]": gaps["window2"], "gap[before]": gaps["window1"]},
                f"gap_{metric}",
                paired=True,
                shapiro=False,
            )
        )
    return apply_fdr(pd.concat(parts, ignore_index=True))


def _save(out: dict[str, pd.DataFrame], suffix: str = "", blocks=None):
    for key, df in out.items():
        metric, block = key.split(":")
        if blocks is not None and key not in blocks:
            continue
        name = {
            "resolution_time_days": "resolution_time",
            "acceptance_rate_per_repo": "acceptance_rate",
            "resolution_time_median_per_repo": "resolution_time_median_per_repo",
        }[metric]
        df.to_csv(f"{OUTPUT_DIR}/rq2_{name}_{block}{suffix}.csv", index=False)


_rq2_out, _rq2_kw = _rq2(df_rq2)
_rq2_kw = [{**k, "breakdown": "global"} for k in _rq2_kw]
_save(_rq2_out)
df_gap = _gap(df_rq2)
df_gap.to_csv(f"{OUTPUT_DIR}/rq2_gap_between_windows.csv", index=False)
print(
    "\n  rq2_{resolution_time,acceptance_rate,resolution_time_median_per_repo}_"
    "{window1,window2,between_windows}.csv, rq2_gap_between_windows.csv (FDR per block)"
)
for key, df in _rq2_out.items():
    print(f"\n  [{key}]")
    print(
        df[
            [
                "group_1",
                "group_2",
                "n1",
                "n2",
                "median_1",
                "median_2",
                "paired_median_diff",
                "repos_1_greater",
                "repos_1_less",
                "mwu_p_fdr",
                "effect",
                "magnitude",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
print("\n  [newcomer - veteran gap, after x before]")
print(
    df_gap[
        [
            "metric",
            "n1",
            "median_1",
            "median_2",
            "paired_median_diff",
            "mwu_p_fdr",
            "effect",
            "magnitude",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# Rate per repository, window and group (base of the paired tests), and the
# number of repositories per group and language.
df_acc_repo = pd.concat(
    {
        (j, g): _acceptance(df_rq2[(df_rq2["window"] == j) & (df_rq2["group"] == g)])
        for j, groups in GROUPS_RQ2.items()
        for g in groups
    },
    names=["window", "group", "Repository"],
).reset_index()
df_acc_repo["Language"] = df_acc_repo["Repository"].map(_lang_repo)
df_acc_repo.to_csv(f"{OUTPUT_DIR}/rq2_acceptance_rate_per_repository.csv", index=False)
df_acc_repo.groupby(["window", "group", "Language"]).size().rename(
    "repositories"
).reset_index().to_csv(f"{OUTPUT_DIR}/rq2_acceptance_repos_per_language.csv", index=False)

# Descriptive resolution time per repository, window and group.
_desc = (
    _no_outside(df_rq2)[df_rq2["days"] >= 0]
    .groupby(["Repository", "window", "group"])["days"]
    .agg(
        n="size",
        median_days="median",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
    )
    .reset_index()
)
_desc[_desc["n"] >= MIN_N_PER_REPO].to_csv(
    f"{OUTPUT_DIR}/rq2_resolution_time_per_repository_descriptive.csv", index=False
)

# Median across repositories of each group and window, unpaired (each group
# uses all its repositories with minimum N, so it differs slightly from the
# medians of the paired tests).
_med_acc = df_acc_repo.groupby(["window", "group"])["rate"].agg(
    repos_acceptance="size", median_acceptance="median"
)
_med_res = (
    _desc[_desc["n"] >= MIN_N_PER_REPO]
    .groupby(["window", "group"])["median_days"]
    .agg(repos_resolution="size", median_resolution_days="median")
)
_med_acc.join(_med_res, how="outer").reset_index().to_csv(
    f"{OUTPUT_DIR}/rq2_medians_groups.csv", index=False
)

# Resolution time with 3 windows of N months: window 0 ("earlier") =
# [thr-2N, thr-N), right before the before window, to see whether time was
# already changing before adoption. PRs without AI only, with Newcomer and
# Veteran as in window 1. The window-0 newcomer is the author of the first PR
# since the start of their own baseline. Repositories where window 0 would
# start before COLLECTION_START are left out of it. Descriptive reference.
_collection_start = pd.Period(COLLECTION_START, freq="M")
df_repo_window0 = pd.DataFrame(
    [
        {
            "Repository": r["Repository"],
            "j0_start": str(pd.Period(r["threshold"], freq="M") - 2 * r["n_months"]),
            "j0_end": r["baseline_end"],
            "lb_start": str(
                max(
                    _collection_start,
                    pd.Period(r["threshold"], freq="M") - 2 * r["n_months"] - BASELINE_MONTHS,
                )
            ),
        }
        for _, r in df_repo_threshold.iterrows()
        if pd.Period(r["threshold"], freq="M") - 2 * r["n_months"] >= _collection_start
    ]
)
_months_lb0 = df_repo_window0.apply(
    lambda r: (pd.Period(r["j0_start"], freq="M") - pd.Period(r["lb_start"], freq="M")).n, axis=1
)
con.register("repo_window0", df_repo_window0)
print(
    f"\n  RQ2 with 3 windows: window 0 in {len(df_repo_window0)} of {len(df_repo_threshold)} repos "
    f"(newcomer baseline: {(_months_lb0 >= BASELINE_MONTHS).sum()} with {BASELINE_MONTHS} months, "
    f"{(_months_lb0 >= 12).sum()} with >= 12, {(_months_lb0 < 12).sum()} with < 12)"
)
con.execute("""
    CREATE OR REPLACE VIEW newcomers_window0 AS
    WITH first AS (
        SELECT h.Author, h.Repository, r.j0_start, r.j0_end, MIN(h.Month) AS first_month
        FROM humans_all h
        JOIN repo_window0 r ON h.Repository = r.Repository
        WHERE h.Month >= r.lb_start
        GROUP BY h.Author, h.Repository, r.j0_start, r.j0_end
    )
    SELECT Author, Repository, first_month FROM first
    WHERE first_month >= j0_start AND first_month < j0_end
""")
con.execute("""
    CREATE OR REPLACE VIEW humans_window0 AS
    SELECT h.* FROM humans h
    JOIN repo_window0 r ON h.Repository = r.Repository
    WHERE h.Month >= r.j0_start AND h.Month < r.j0_end
""")
_GROUPS_J0 = {
    "Newcomer": f"SELECT x.* FROM humans_window0 x JOIN newcomers_window0 n ON {_NEWC}",
    "Veteran": f"SELECT x.* FROM humans_window0 x ANTI JOIN newcomers_window0 n ON {_NEWC}",
}
df_rq2_j0 = pd.concat(
    [
        con.execute(f"""
        SELECT 'window0' AS "window", '{g}' AS "group", Repository, PR_Number, State,
               Closes_GFI, CAST(GFI_Issue_Numbers AS VARCHAR) AS GFI_Issue_Numbers,
               CASE WHEN State = 'merged' AND Closed_At IS NOT NULL
                    THEN (epoch(Closed_At) - epoch(Created_At)) / 86400.0 END AS days
        FROM ({sql}) z
    """).df()
        for g, sql in _GROUPS_J0.items()
    ],
    ignore_index=True,
)
assert not df_rq2_j0.duplicated(["Repository", "PR_Number"]).any(), (
    "a PR fell into both groups of window 0"
)
_rq2_3j = pd.concat(
    [
        df_rq2[
            [
                "window",
                "group",
                "Repository",
                "PR_Number",
                "State",
                "Closes_GFI",
                "GFI_Issue_Numbers",
                "days",
            ]
        ],
        df_rq2_j0,
    ],
    ignore_index=True,
)

_desc3 = (
    _no_outside(_rq2_3j)[_rq2_3j["days"] >= 0]
    .groupby(["Repository", "window", "group"])["days"]
    .agg(
        n="size",
        median_days="median",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
    )
    .reset_index()
)
_desc3 = _desc3[_desc3["n"] >= MIN_N_PER_REPO]
_desc3.to_csv(f"{OUTPUT_DIR}/rq2_resolution_3windows_per_repository.csv", index=False)

# Paired Wilcoxon on per-repository medians, one BH family per block. Group 1
# is always the most recent window (effect > 0 = slower later).
_med3 = {
    (j, g): _median_repo(_rq2_3j[(_rq2_3j["window"] == j) & (_rq2_3j["group"] == g)])
    for j, g in _rq2_3j[["window", "group"]].drop_duplicates().itertuples(index=False)
}
_LBL_J = {"window0": "earlier", "window1": "before", "window2": "after"}
_BLOCKS_3J = {
    "window0": [(("window0", "Newcomer"), ("window0", "Veteran"))],
    "earlier_before": [(("window1", g), ("window0", g)) for g in ("Newcomer", "Veteran")],
    "earlier_after": [(("window2", g), ("window0", g)) for g in ("Newcomer", "Veteran")],
}
_pairs3 = []
for block, pairs in _BLOCKS_3J.items():
    _df = pd.concat(
        [
            run_group_comparison(
                {f"{a[1]}[{_LBL_J[a[0]]}]": _med3[a], f"{b[1]}[{_LBL_J[b[0]]}]": _med3[b]},
                f"resolution_time_median_per_repo[{block}]",
                paired=True,
                shapiro=False,
            )
            for a, b in pairs
        ],
        ignore_index=True,
    )
    _pairs3.append(apply_fdr(_df).assign(block=block))
df_rq2_3j_pairs = pd.concat(_pairs3, ignore_index=True)
df_rq2_3j_pairs.to_csv(f"{OUTPUT_DIR}/rq2_resolution_3windows_pairs.csv", index=False)
print(
    "  rq2_resolution_3windows_{per_repository,pairs}.csv. Repos per window and group "
    f"(n >= {MIN_N_PER_REPO} merged):"
)
print(_desc3.groupby(["window", "group"]).size().to_string())
print(
    df_rq2_3j_pairs[
        [
            "block",
            "group_1",
            "group_2",
            "n1",
            "median_1",
            "median_2",
            "paired_median_diff",
            "repos_1_greater",
            "repos_1_less",
            "mwu_p_fdr",
            "effect",
            "magnitude",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# Acceptance rate with the same 3 windows (same repositories, groups and
# blocks). Effect > 0 = higher acceptance in the most recent window.
df_acc3_repo = pd.concat(
    {
        (j, g): _acceptance(_rq2_3j[(_rq2_3j["window"] == j) & (_rq2_3j["group"] == g)])
        for j, g in _rq2_3j[["window", "group"]].drop_duplicates().itertuples(index=False)
    },
    names=["window", "group", "Repository"],
).reset_index()
df_acc3_repo.to_csv(f"{OUTPUT_DIR}/rq2_acceptance_3windows_per_repository.csv", index=False)
_acc3 = {
    (j, g): d.set_index("Repository")["rate"]
    for (j, g), d in df_acc3_repo.groupby(["window", "group"])
}
_pairs_acc3 = []
for block, pairs in _BLOCKS_3J.items():
    _df = pd.concat(
        [
            run_group_comparison(
                {f"{a[1]}[{_LBL_J[a[0]]}]": _acc3[a], f"{b[1]}[{_LBL_J[b[0]]}]": _acc3[b]},
                f"acceptance_rate_per_repo[{block}]",
                paired=True,
                shapiro=False,
            )
            for a, b in pairs
        ],
        ignore_index=True,
    )
    _pairs_acc3.append(apply_fdr(_df).assign(block=block))
df_rq2_acc3_pairs = pd.concat(_pairs_acc3, ignore_index=True)
df_rq2_acc3_pairs.to_csv(f"{OUTPUT_DIR}/rq2_acceptance_3windows_pairs.csv", index=False)
print(
    "  rq2_acceptance_3windows_{per_repository,pairs}.csv. Repos per window and group "
    f"(n >= {MIN_N_PER_REPO} closed):"
)
print(df_acc3_repo.groupby(["window", "group"]).size().to_string())
print(
    df_rq2_acc3_pairs[
        [
            "block",
            "group_1",
            "group_2",
            "n1",
            "median_1",
            "median_2",
            "paired_median_diff",
            "repos_1_greater",
            "repos_1_less",
            "mwu_p_fdr",
            "effect",
            "magnitude",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# Sensitivity analyses (window 2, and between windows when affected).
# robobun (oven-sh/bun) alone is about half of AI_Agent: window 2 without it.
SENSITIVITY_AUTHOR = "robobun"
_no_robobun = df_rq2[~((df_rq2["group"] == "AI_Agent") & (df_rq2["Author"] == SENSITIVITY_AUTHOR))]
_out, _kw = _rq2(_no_robobun, f"|no {SENSITIVITY_AUTHOR}", shapiro=False)
_save(
    _out,
    f"_no_{SENSITIVITY_AUTHOR}",
    blocks={
        "resolution_time_days:window2",
        "acceptance_rate_per_repo:window2",
        "resolution_time_median_per_repo:window2",
    },
)
_rq2_kw += [{**k, "breakdown": f"no_{SENSITIVITY_AUTHOR}"} for k in _kw]

# Censoring: AI PRs concentrate in the last months, which had less time to be
# closed before the collection. RQ2 is repeated with PRs created up to
# CENSORING_LIMIT_MONTH, and the share of open PRs per group is written.
CENSORING_LIMIT_MONTH = "2026-05"
df_rq2.assign(
    is_open=df_rq2["State"] == "open", after_limit=df_rq2["Month"] > CENSORING_LIMIT_MONTH
).groupby(["window", "group"], sort=False).agg(
    prs=("State", "size"),
    fraction_open=("is_open", "mean"),
    fraction_after_limit=("after_limit", "mean"),
).reset_index().to_csv(f"{OUTPUT_DIR}/rq2_censoring_per_group.csv", index=False)
_out, _kw = _rq2(
    df_rq2[df_rq2["Month"] <= CENSORING_LIMIT_MONTH],
    f"|until {CENSORING_LIMIT_MONTH}",
    shapiro=False,
)
_save(
    _out,
    f"_until_{CENSORING_LIMIT_MONTH}",
    blocks={
        "resolution_time_days:window2",
        "acceptance_rate_per_repo:window2",
        "resolution_time_days:between_windows",
        "acceptance_rate_per_repo:between_windows",
        "resolution_time_median_per_repo:window2",
        "resolution_time_median_per_repo:between_windows",
    },
)
_rq2_kw += [{**k, "breakdown": f"until_{CENSORING_LIMIT_MONTH}"} for k in _kw]

WITH_EXTERNAL_MERGE = True
_out, _ = _rq2(df_rq2, "|with external merge", shapiro=False)
WITH_EXTERNAL_MERGE = False
_save(
    _out,
    "_with_external_merge",
    blocks={
        "acceptance_rate_per_repo:window2",
        "acceptance_rate_per_repo:between_windows",
        "resolution_time_median_per_repo:window2",
        "resolution_time_median_per_repo:between_windows",
        "resolution_time_days:window2",
        "resolution_time_days:between_windows",
    },
)
print(
    f"\n  sensitivity: rq2_*_no_{SENSITIVITY_AUTHOR}.csv, rq2_*_until_{CENSORING_LIMIT_MONTH}.csv, "
    f"rq2_censoring_per_group.csv, rq2_*_with_external_merge.csv ({len(_rm_outside)} repos with "
    f"threshold, merge < {MIN_MERGE_RATE:.0%} or {len(EXTERNAL_MERGE_MANUAL)} by hand, left out of "
    f"the main acceptance and resolution time)"
)

# Breakdowns: language, stars quartile and community size quartile. Within each
# category, the same RQ2 (AI_Agent left out of the per-repository metrics). BH
# across the categories of the breakdown, one family per (breakdown, metric,
# block). Groups and pairs need MIN_REPOS_BREAKDOWN repositories.
_brk = []
for name, col in BREAKDOWNS.items():
    for cat, repos in breakdown_repo.groupby(col, observed=True).groups.items():
        _out, _kw = _rq2(
            df_rq2[df_rq2["Repository"].isin(repos)],
            f"|{name}={cat}",
            exclude_paired=("AI_Agent",),
            fdr=False,
            min_repos=MIN_REPOS_BREAKDOWN,
        )
        _rq2_kw += [{**k, "breakdown": name, "category": cat} for k in _kw]
        _brk += [
            df.assign(
                breakdown=name, category=cat, metric_base=key.split(":")[0], block=key.split(":")[1]
            )
            for key, df in _out.items()
            if not df.empty
        ]
df_rq2_brk = pd.concat(
    [
        apply_fdr(g)
        for _, g in pd.concat(_brk, ignore_index=True).groupby(
            ["breakdown", "metric_base", "block"]
        )
    ],
    ignore_index=True,
)
df_rq2_brk.to_csv(f"{OUTPUT_DIR}/rq2_breakdowns.csv", index=False)

# Overall Kruskal-Wallis of window 2 (global, sensitivity and breakdowns), BH
# within each breakdown and base metric.
df_rq2_kw = pd.DataFrame(_rq2_kw)
df_rq2_kw["metric_base"] = df_rq2_kw["metric"].str.split("[", regex=False).str[0]
df_rq2_kw["p_fdr"] = np.nan
for _, idx in (
    df_rq2_kw[df_rq2_kw["p"].notna()].groupby(["breakdown", "metric_base"]).groups.items()
):
    df_rq2_kw.loc[idx, "p_fdr"] = stats.false_discovery_control(
        df_rq2_kw.loc[idx, "p"], method="bh"
    )
df_rq2_kw.to_csv(f"{OUTPUT_DIR}/rq2_kruskal.csv", index=False)

# Across categories: Kruskal-Wallis on the per-repository metric of each
# window and group, and on the after - before change in acceptance and in
# median resolution time (days) of newcomers and veterans without AI. Only
# categories with MIN_REPOS_BREAKDOWN repositories.
_vet_repo = {
    (metric, j, g): v
    for metric in ("acceptance_rate_per_repo", "resolution_time_median_per_repo")
    for j in GROUPS_RQ2
    for g, v in _vectors(df_rq2, j, metric).items()
}
for g in ("Newcomer", "Veteran"):
    for metric in ("acceptance_rate_per_repo", "resolution_time_median_per_repo"):
        _vet_repo[(f"var_{metric}", "after-before", g)] = (
            _vet_repo[(metric, "window2", g)] - _vet_repo[(metric, "window1", g)]
        ).dropna()
_kw_cat = []
for name, col in BREAKDOWNS.items():
    cats = breakdown_repo[col]
    for (metric, j, g), v in _vet_repo.items():
        samples = [
            s
            for _, s in v.groupby(cats.reindex(v.index), observed=True)
            if len(s) >= MIN_REPOS_BREAKDOWN
        ]
        h, p = stats.kruskal(*samples) if len(samples) >= 2 else (np.nan, np.nan)
        _kw_cat.append(
            {
                "breakdown": name,
                "metric": metric,
                "window": j,
                "group": g,
                "categories": len(samples),
                "repositories": sum(map(len, samples)),
                "H": h,
                "p": p,
            }
        )
df_rq2_kw_cat = pd.DataFrame(_kw_cat)
_t = df_rq2_kw_cat["p"].notna()
df_rq2_kw_cat["p_fdr"] = np.nan
df_rq2_kw_cat.loc[_t, "p_fdr"] = stats.false_discovery_control(
    df_rq2_kw_cat.loc[_t, "p"], method="bh"
)
df_rq2_kw_cat.to_csv(f"{OUTPUT_DIR}/rq2_breakdowns_kruskal_between_categories.csv", index=False)
print(
    f"\n  rq2_breakdowns.csv ({len(df_rq2_brk)} comparisons), rq2_kruskal.csv, "
    f"rq2_breakdowns_kruskal_between_categories.csv"
)
print(
    df_rq2_kw[df_rq2_kw["breakdown"] == "global"][["metric", "groups", "n_total", "H", "p_fdr"]]
    .round(4)
    .to_string(index=False)
)
print(df_rq2_kw_cat.round(4).to_string(index=False))

# Group composition (Threats to Validity)
df_newcomers_unique_after = con.execute("SELECT COUNT(DISTINCT Author) FROM newcomers").fetchone()[
    0
]
print(f"\n  Unique newcomers (after adoption, by author): {df_newcomers_unique_after:,}")
_n = df_rq2.groupby(["window", "group"]).size()
_comp = {
    # human control PRs without an author login (excluded)
    "control_prs_no_author": con.execute(
        "SELECT COUNT(*) FROM h_raw WHERE NULLIF(TRIM(Author), '') IS NULL"
    ).fetchone()[0],
    **{f"prs_{j}_{g}": int(n) for (j, g), n in _n.items()},
    "prs_ai_window1_excluded": _n_ai_before,
    # window-2 Human AI PRs without a login (neither newcomer nor veteran)
    "human_ai_window2_no_login": con.execute(
        "SELECT COUNT(*) FROM ai_al WHERE Author_Category = 'Human' "
        "AND NULLIF(TRIM(Author), '') IS NULL"
    ).fetchone()[0],
    "ai_agent_prs_robobun": int(
        ((df_rq2["group"] == "AI_Agent") & (df_rq2["Author"] == SENSITIVITY_AUTHOR)).sum()
    ),
    # newcomers: Author+Repository pairs and distinct people
    "newcomers_pairs_author_repo_window1": con.execute(
        "SELECT COUNT(*) FROM newcomers_before"
    ).fetchone()[0],
    "newcomers_pairs_author_repo": con.execute("SELECT COUNT(*) FROM newcomers").fetchone()[0],
    "newcomers_authors_distinct": df_newcomers_unique_after,
}
# Newcomer PRs in window 2 from newcomers with 10 or more PRs in the group
_per_newcomer = (
    df_rq2[(df_rq2["window"] == "window2") & (df_rq2["group"] == "Newcomer")]
    .groupby(["Author", "Repository"])
    .size()
)
_comp["newcomer_prs_by_newcomers_10plus"] = int(_per_newcomer[_per_newcomer >= 10].sum())
# activity of repositories with and without a valid threshold
_with_signal = set(con.execute("SELECT DISTINCT Repository FROM ai").df()["Repository"])
_thr = set(df_repo_threshold["Repository"])
_comp["median_prs_repos_threshold"] = _rm.loc[_rm["repo"].isin(_thr), "prs"].median()
_comp["median_prs_repos_signal_no_threshold"] = _rm.loc[
    _rm["repo"].isin(_with_signal - _thr), "prs"
].median()
pd.Series(_comp).to_csv(f"{OUTPUT_DIR}/composition_groups.csv", header=["value"])
print(f"  composition_groups.csv: {_comp}")


print("\n" + "=" * 70)
print("RQ2 - GOOD FIRST ISSUES IN WINDOW 2")
print("=" * 70)

# Window 2 with its 5 groups. Repositories that "use GFIs": at least one merged
# human PR without AI (Newcomer or Veteran) that closes a GFI, in either
# window. Defined by the groups without AI only, so that repositories are not
# selected by the outcome of the AI groups. Merged PRs only.
_j2 = df_rq2[df_rq2["window"] == "window2"]
_repos_gfi = set(
    df_rq2.loc[
        df_rq2["group"].isin(["Newcomer", "Veteran"])
        & df_rq2["Closes_GFI"]
        & (df_rq2["State"] == "merged"),
        "Repository",
    ]
)
gfi_base = _j2[(_j2["State"] == "merged") & _j2["Repository"].isin(_repos_gfi)]
GROUPS_GFI = list(GROUPS_RQ2["window2"])
print(f"  repos with threshold: {len(_thr)} | repos using GFIs: {len(_repos_gfi)}")

# activity of the repositories that use GFIs vs. the others with a threshold
_m = _rm.set_index("repo")
_prs = {
    name: _m.loc[sorted(rs & set(_m.index)), "prs"]
    for name, rs in (("uses_gfi", _repos_gfi), ("no_gfi", _thr - _repos_gfi))
}
_mw = stats.mannwhitneyu(_prs["uses_gfi"], _prs["no_gfi"], alternative="two-sided")
pd.DataFrame(
    [
        {
            "group": name,
            "n_repos": len(v),
            "median_prs": v.median(),
            "median_merged_prs": _m.loc[v.index, "merged_prs"].median(),
            "mwu_p_prs": _mw.pvalue,
        }
        for name, v in _prs.items()
    ]
).to_csv(f"{OUTPUT_DIR}/rq2_gfi_repos_activity.csv", index=False)

# diagnostics: PRs that close a GFI but are not merged, per group
print("\n  Closes_GFI per State (before the merged filter):")
print(_j2[_j2["Closes_GFI"]].groupby(["group", "State"]).size().to_string())


def _gfi_agg(d: pd.DataFrame, per: list[str]) -> pd.DataFrame:
    a = d.groupby(per)["Closes_GFI"].agg(prs_close_gfi="sum", total_merged="size").reset_index()
    a["pct_gfi_in_group"] = (100.0 * a["prs_close_gfi"] / a["total_merged"]).round(2)
    return a


# Global: % of the group's merged PRs that close a GFI.
df_gfi_global = _gfi_agg(gfi_base, ["group"]).sort_values("pct_gfi_in_group", ascending=False)
df_gfi_global.to_csv(f"{OUTPUT_DIR}/rq2_gfi_global.csv", index=False)
print("\n  rq2_gfi_global.csv")
print(df_gfi_global.to_string(index=False))


# Per repository (minimum N) + Kruskal-Wallis + pairs.
def _gfi_repo(d: pd.DataFrame) -> pd.DataFrame:
    r = _gfi_agg(d, ["Repository", "group"])
    r = r[r["total_merged"] >= MIN_N_PER_REPO].copy()
    r["rate"] = r["prs_close_gfi"] / r["total_merged"]
    return r


def _gfi_vectors(r: pd.DataFrame) -> dict[str, pd.Series]:
    return {g: r.loc[r["group"] == g].set_index("Repository")["rate"] for g in GROUPS_GFI}


df_gfi_repo = _gfi_repo(gfi_base)
df_gfi_repo.drop(columns="pct_gfi_in_group").to_csv(
    f"{OUTPUT_DIR}/rq2_gfi_per_repository.csv", index=False
)
_gfi_kw = [overall_kruskal(_gfi_vectors(df_gfi_repo), "gfi_rate_per_repo")]
df_gfi_test = apply_fdr(
    run_group_comparison(_gfi_vectors(df_gfi_repo), "gfi_rate_per_repo", paired=True)
)
df_gfi_test.to_csv(f"{OUTPUT_DIR}/rq2_gfi_test.csv", index=False)
print(
    f"  rq2_gfi_per_repository.csv ({df_gfi_repo['Repository'].nunique()} repos), "
    f"rq2_gfi_test.csv (FDR applied, {len(df_gfi_test)} pairs)"
)
print(
    df_gfi_test[["group_1", "group_2", "n1", "mwu_p_fdr", "effect", "magnitude"]]
    .round(4)
    .to_string(index=False)
)

# Censoring, same limit as RQ2. Repositories that use GFIs are still defined
# over the whole period.
_gfi_cens = _gfi_repo(gfi_base[gfi_base["Month"] <= CENSORING_LIMIT_MONTH])
_gfi_kw.append(
    overall_kruskal(_gfi_vectors(_gfi_cens), f"gfi_rate_per_repo[until {CENSORING_LIMIT_MONTH}]")
)
apply_fdr(
    run_group_comparison(
        _gfi_vectors(_gfi_cens),
        f"gfi_rate_per_repo[until {CENSORING_LIMIT_MONTH}]",
        paired=True,
    )
).to_csv(f"{OUTPUT_DIR}/rq2_gfi_test_until_{CENSORING_LIMIT_MONTH}.csv", index=False)
pd.DataFrame(_gfi_kw).to_csv(f"{OUTPUT_DIR}/rq2_gfi_kruskal.csv", index=False)

# Share still open and median time to merge of newcomer PRs without AI, with
# and without GFI, per creation period.
_newc = _j2[(_j2["group"] == "Newcomer") & _j2["Repository"].isin(_repos_gfi)]
_newc.assign(
    after_limit=_newc["Month"] > CENSORING_LIMIT_MONTH, is_open=_newc["State"] == "open"
).groupby(["Closes_GFI", "after_limit"]).agg(
    prs=("State", "size"),
    fraction_open=("is_open", "mean"),
    median_days_merge=("days", lambda x: x[x >= 0].median()),
).reset_index().rename(columns={"Closes_GFI": "closes_gfi"}).to_csv(
    f"{OUTPUT_DIR}/rq2_gfi_censoring_newcomers.csv", index=False
)

# Per language (minimum N per cell).
df_gfi_lang = _gfi_agg(gfi_base, ["Language", "group"])
df_gfi_lang = df_gfi_lang[df_gfi_lang["total_merged"] >= MIN_N_CELL]
df_gfi_lang.to_csv(f"{OUTPUT_DIR}/rq2_gfi_per_language.csv", index=False)


# Timeline: monthly share, not raw counts. Cells with small N are NaN.
def gfi_timeline(per_language: bool = False) -> pd.DataFrame:
    cols = ["Language", "Month", "group"] if per_language else ["Month", "group"]
    df = _gfi_agg(gfi_base, cols).drop(columns="pct_gfi_in_group")
    df["pct_gfi"] = np.where(
        df["total_merged"] >= MIN_N_CELL, 100.0 * df["prs_close_gfi"] / df["total_merged"], np.nan
    )
    return df


df_gfi_timeline = gfi_timeline()
df_gfi_timeline.to_csv(f"{OUTPUT_DIR}/rq2_gfi_timeline.csv", index=False)
gfi_timeline(per_language=True).to_csv(
    f"{OUTPUT_DIR}/rq2_gfi_timeline_per_language.csv", index=False
)
print(
    f"  rq2_gfi_test_until_{CENSORING_LIMIT_MONTH}.csv, rq2_gfi_kruskal.csv, "
    f"rq2_gfi_censoring_newcomers.csv, rq2_gfi_per_language.csv, rq2_gfi_timeline*.csv"
)

print(f"\n  % of merged PRs that close a GFI, per month (cells with N < {MIN_N_CELL} = NaN)")
_wide = df_gfi_timeline.pivot(index="Month", columns="group", values="pct_gfi").sort_index()
_cols_ai = [c for c in ("AI_Agent", "Newcomer_AI", "Veteran_AI") if c in _wide.columns]
if _cols_ai and _wide[_cols_ai].notna().any().any():
    print(_wide.loc[_wide[_cols_ai].dropna(how="all").index.min() :].round(1).to_string())
else:
    print("  (no month with enough N in the AI groups)")


# GFI before x after: the same per-repository rate in both windows. The
# repositories that use GFIs are defined over both windows together.
# Pairs between windows as in RQ2 (PAIRS_BETWEEN_WINDOWS), paired Wilcoxon on
# shared repositories; group 1 = after.
_jj = df_rq2[df_rq2["State"] == "merged"]
_repos_gfi_jj = set(
    _jj.loc[_jj["group"].isin(["Newcomer", "Veteran"]) & _jj["Closes_GFI"], "Repository"]
)
_gfi_jj = pd.concat(
    [
        _gfi_repo(_jj[(_jj["window"] == j) & _jj["Repository"].isin(_repos_gfi_jj)]).assign(
            window=j
        )
        for j in ("window1", "window2")
    ],
    ignore_index=True,
)
_gfi_jj.drop(columns="pct_gfi_in_group").to_csv(
    f"{OUTPUT_DIR}/rq2_gfi_per_repository_windows.csv", index=False
)
_vj = {
    j: {
        g: _gfi_jj.loc[(_gfi_jj["window"] == j) & (_gfi_jj["group"] == g)].set_index("Repository")[
            "rate"
        ]
        for g in GROUPS_RQ2[j]
    }
    for j in ("window1", "window2")
}
apply_fdr(run_group_comparison(_vj["window1"], "gfi_rate_per_repo[window1]", paired=True)).to_csv(
    f"{OUTPUT_DIR}/rq2_gfi_window1.csv", index=False
)
df_gfi_between = apply_fdr(
    pd.concat(
        [
            run_group_comparison(
                {f"{g2}[after]": _vj["window2"][g2], f"{g1}[before]": _vj["window1"][g1]},
                "gfi_rate_per_repo[between_windows]",
                paired=True,
                shapiro=False,
            )
            for g1, g2 in PAIRS_BETWEEN_WINDOWS
        ],
        ignore_index=True,
    )
)
df_gfi_between.to_csv(f"{OUTPUT_DIR}/rq2_gfi_between_windows.csv", index=False)
print(
    f"\n  rq2_gfi_per_repository_windows.csv ({len(_repos_gfi_jj)} repos use GFIs in the 2 windows, "
    f"the same set as window 2: {len(_repos_gfi)}), rq2_gfi_window1.csv, rq2_gfi_between_windows.csv"
)
print(
    df_gfi_between[
        ["group_1", "group_2", "n1", "median_1", "median_2", "mwu_p_fdr", "effect", "magnitude"]
    ]
    .round(4)
    .to_string(index=False)
)


# Who closes the GFIs: each GFI (repository + issue number) closed by a merged
# PR, counted in the PR's group in each window. pct = % of the window's
# distinct GFIs. A GFI closed by PRs of 2 groups counts in both
# (gfis_shared), so pct can add up to slightly more than 100.
def _who_closes(d: pd.DataFrame) -> pd.DataFrame:
    gi = d[(d["State"] == "merged") & d["Closes_GFI"]].assign(
        issue=lambda x: (
            x["GFI_Issue_Numbers"].astype(str).str.replace(r"\.0$", "", regex=True).str.split(",")
        )
    )
    gi = gi.explode("issue").assign(issue=lambda x: x["issue"].str.strip())
    gi = gi[gi["issue"].str.fullmatch(r"\d+")].drop_duplicates(
        ["window", "group", "Repository", "issue"]
    )
    n_groups = gi.groupby(["window", "Repository", "issue"])["group"].transform("size")
    tot = gi.drop_duplicates(["window", "Repository", "issue"]).groupby("window").size()
    q = (
        gi.assign(shared=n_groups > 1)
        .groupby(["window", "group"])
        .agg(gfis=("issue", "size"), gfis_shared=("shared", "sum"), repos=("Repository", "nunique"))
        .reset_index()
    )
    q["gfis_distinct_window"] = q["window"].map(tot)
    q["pct"] = 100.0 * q["gfis"] / q["gfis_distinct_window"]
    # slice of the stacked bar (plot_rq2.py): % of assignments, adds up to 100
    q["pct_assignments"] = 100.0 * q["gfis"] / q.groupby("window")["gfis"].transform("sum")
    return q


df_gfi_who = _who_closes(df_rq2)
df_gfi_who.to_csv(f"{OUTPUT_DIR}/rq2_gfi_who_closes.csv", index=False)
print("\n  rq2_gfi_who_closes.csv")
print(df_gfi_who.round(1).to_string(index=False))


# GFI with 3 windows of N months: the same earlier window as time and
# acceptance. Repositories that use GFIs are defined over the 3 windows
# together; BH blocks as for time.
_jj3 = _rq2_3j[_rq2_3j["State"] == "merged"]
_repos_gfi_3j = set(
    _jj3.loc[_jj3["group"].isin(["Newcomer", "Veteran"]) & _jj3["Closes_GFI"], "Repository"]
)
_gfi_3j = pd.concat(
    [
        _gfi_repo(_jj3[(_jj3["window"] == j) & _jj3["Repository"].isin(_repos_gfi_3j)]).assign(
            window=j
        )
        for j in ("window0", "window1", "window2")
    ],
    ignore_index=True,
)
_gfi_3j.drop(columns="pct_gfi_in_group").to_csv(
    f"{OUTPUT_DIR}/rq2_gfi_3windows_per_repository.csv", index=False
)
_v3 = {
    (j, g): d.set_index("Repository")["rate"] for (j, g), d in _gfi_3j.groupby(["window", "group"])
}
df_gfi_3j_pairs = pd.concat(
    [
        apply_fdr(
            pd.concat(
                [
                    run_group_comparison(
                        {f"{a[1]}[{_LBL_J[a[0]]}]": _v3[a], f"{b[1]}[{_LBL_J[b[0]]}]": _v3[b]},
                        f"gfi_rate_per_repo[{block}]",
                        paired=True,
                        shapiro=False,
                    )
                    for a, b in pairs
                ],
                ignore_index=True,
            )
        ).assign(block=block)
        for block, pairs in _BLOCKS_3J.items()
    ],
    ignore_index=True,
)
df_gfi_3j_pairs.to_csv(f"{OUTPUT_DIR}/rq2_gfi_3windows_pairs.csv", index=False)
df_gfi_who3 = _who_closes(_rq2_3j)
df_gfi_who3.to_csv(f"{OUTPUT_DIR}/rq2_gfi_who_closes_3windows.csv", index=False)
print(
    f"\n  rq2_gfi_3windows_{{per_repository,pairs}}.csv ({len(_repos_gfi_3j)} repos use GFIs in the 3 windows), "
    "rq2_gfi_who_closes_3windows.csv. Repos per window and group:"
)
print(_gfi_3j.groupby(["window", "group"]).size().to_string())
print(
    df_gfi_3j_pairs[
        [
            "block",
            "group_1",
            "group_2",
            "n1",
            "median_1",
            "median_2",
            "repos_1_greater",
            "repos_1_less",
            "mwu_p_fdr",
            "effect",
            "magnitude",
        ]
    ]
    .round(4)
    .to_string(index=False)
)
print(df_gfi_who3.round(1).to_string(index=False))


pd.DataFrame(_shapiro_rows).assign(normal=lambda d: d["p"] >= ALPHA).to_csv(
    f"{OUTPUT_DIR}/rq_shapiro.csv", index=False
)

print("\n" + "=" * 70)
print(f"Done. Files in ./{OUTPUT_DIR}/")
print("=" * 70)
con.close()
