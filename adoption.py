"""Descriptive characterization of AI adoption.

From the classified AI-evidence PRs, writes results/adoption_*.csv (per tool,
per language, repository ranking, timeline and the summary numbers quoted in
the paper) and figures/adoption_timeline.pdf.

Same cutoff as analysis.py: only PRs whose Signal_Date month (or Created_At,
if empty) is before CUTOFF_MONTH. The month of each PR in the timeline is also
the Signal_Date month.
"""

import csv
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import classify_prs

os.makedirs("figures", exist_ok=True)

INPUT_AI = "data/ai_prs.csv"
OUTPUT_DIR = "results"
FIGURE = "figures/adoption_timeline.pdf"

CUTOFF_MONTH = "2026-08"
# Before 2025 there are only 12 PRs in total; they stay in the timeline CSV
# but are not drawn.
FIGURE_START_MONTH = "2025-01"

TOOL_NAMES = {"Copilot": "GitHub Copilot"}
TOP_TOOLS = 4
TOP_LANGUAGES = 5
TOP_REPOSITORIES = 10

GROUPS = [("Human", "Human AI"), ("AI_Agent", "AI Agent")]
STATES = [
    ("merged", "Merged", "#2a78d6"),
    ("closed", "Closed", "#eb6834"),
    ("open", "Open", "#1baf7a"),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

con = duckdb.connect()
con.execute(f"""
CREATE OR REPLACE VIEW ai AS
SELECT
    AI_Tool, Language, Repository, PR_Author, PR_Author_Category,
    LOWER(TRIM(State)) AS State,
    strftime('%Y-%m', TRY_CAST(COALESCE(NULLIF(Signal_Date, ''), Created_At)
                               AS TIMESTAMP)) AS Month
FROM read_csv('{INPUT_AI}', all_varchar = true)
WHERE Month < '{CUTOFF_MONTH}'
""")

total = con.execute("SELECT COUNT(*) FROM ai").fetchone()[0]
# AI PRs collected before the month cutoff, to report how many the cutoff removes.
total_no_cutoff = con.execute(
    f"SELECT COUNT(*) FROM read_csv('{INPUT_AI}', all_varchar = true)"
).fetchone()[0]
n_repos = con.execute("SELECT COUNT(DISTINCT Repository) FROM ai").fetchone()[0]


def pct(n: int) -> str:
    return f"{100 * n / total:.1f}%"


# Per tool
df_tool = con.execute("""
    SELECT AI_Tool AS tool, COUNT(*) AS prs,
           COUNT(DISTINCT Repository) AS repositories
    FROM ai GROUP BY 1 ORDER BY prs DESC, tool
""").df()
df_tool["tool"] = df_tool["tool"].replace(TOOL_NAMES)
df_tool.to_csv(f"{OUTPUT_DIR}/adoption_per_tool.csv", index=False)

# Per language
df_lang = con.execute("""
    SELECT Language AS language, COUNT(*) AS prs,
           COUNT(DISTINCT Repository) AS repositories
    FROM ai GROUP BY 1 ORDER BY prs DESC, language
""").df()
df_lang.to_csv(f"{OUTPUT_DIR}/adoption_per_language.csv", index=False)

# Repository ranking
df_rank = con.execute("""
    SELECT Repository AS repository, Language AS language, COUNT(*) AS prs
    FROM ai GROUP BY 1, 2 ORDER BY prs DESC, repository
""").df()
df_rank.to_csv(f"{OUTPUT_DIR}/adoption_ranking_repositories.csv", index=False)

# Timeline: month x group x state
df_time = con.execute("""
    SELECT Month, PR_Author_Category AS "group",
           COUNT(*) AS prs,
           COUNT(*) FILTER (WHERE State = 'merged') AS merged,
           COUNT(*) FILTER (WHERE State = 'closed') AS closed,
           COUNT(*) FILTER (WHERE State = 'open')   AS open
    FROM ai GROUP BY 1, 2 ORDER BY 1, 2
""").df()
df_time.to_csv(f"{OUTPUT_DIR}/adoption_timeline.csv", index=False)

# Summary numbers quoted in the paper: repositories with AI, share of the top 2
# and top 4 tools, share of the top 1 and top 10 repositories, share in 2026 and
# share opened by humans (Human AI).
_n_sample = len(list(Path("data/raw_prs").glob("*.json")))
_top = df_tool["prs"].to_numpy()
pd.DataFrame(
    [
        {
            "repos_sample": _n_sample,
            "repos_with_ai": n_repos,
            "pct_repos_with_ai": 100 * n_repos / _n_sample,
            "prs_ai_collected": total_no_cutoff,
            "prs_ai_cut": total_no_cutoff - total,
            "prs_ai": total,
            "tools": len(df_tool),
            "pct_top2_tools": 100 * _top[:2].sum() / total,
            "pct_top4_tools": 100 * _top[:4].sum() / total,
            "pct_top1_repo": 100 * df_rank["prs"].iloc[0] / total,
            "pct_top10_repos": 100 * df_rank["prs"].iloc[:10].sum() / total,
            "pct_prs_2026": 100 * df_time.loc[df_time["Month"] >= "2026", "prs"].sum() / total,
            "pct_human_ai": 100 * df_time.loc[df_time["group"] == "Human", "prs"].sum() / total,
        }
    ]
).to_csv(f"{OUTPUT_DIR}/adoption_summary.csv", index=False)

# Timeline per repository: PRs per repository and month, with the share opened
# by AI_Agent accounts and by robobun alone.
HIGHLIGHTED_AUTHOR = "robobun"
df_repo_month = con.execute(f"""
    SELECT Repository AS repository, Month, COUNT(*) AS prs,
           COUNT(*) FILTER (WHERE PR_Author_Category = 'AI_Agent') AS prs_ai_agent,
           COUNT(*) FILTER (WHERE PR_Author = '{HIGHLIGHTED_AUTHOR}') AS prs_{HIGHLIGHTED_AUTHOR}
    FROM ai GROUP BY 1, 2 ORDER BY 1, 2
""").df()
df_repo_month.to_csv(f"{OUTPUT_DIR}/adoption_repository_month.csv", index=False)

# Growth of the TOP_GROWTH repositories with most PRs: monthly mean in
# Sep-Dec 2025 vs. Apr-Jul 2026, peak month and AI_Agent share.
TOP_GROWTH = 20
INITIAL_WINDOW = ("2025-09", "2025-12")
FINAL_WINDOW = ("2026-04", "2026-07")
LATE_PEAK_MONTH = "2026-03"
_top = df_rank.head(TOP_GROWTH)["repository"]
_rm = df_repo_month[df_repo_month["repository"].isin(_top)]


def _mean_window(g: pd.DataFrame, window: tuple[str, str]) -> float:
    months_window = pd.period_range(*window, freq="M").strftime("%Y-%m")
    return g.set_index("Month")["prs"].reindex(months_window, fill_value=0).mean()


df_growth = pd.DataFrame(
    [
        {
            "repository": repo,
            "prs": int(g["prs"].sum()),
            "pct_ai_agent": 100 * g["prs_ai_agent"].sum() / g["prs"].sum(),
            "initial_mean": _mean_window(g, INITIAL_WINDOW),
            "final_mean": _mean_window(g, FINAL_WINDOW),
            "peak_month": g.loc[g["prs"].idxmax(), "Month"],
        }
        for repo, g in _rm.groupby("repository")
    ]
).sort_values("prs", ascending=False)
df_growth["tripled"] = df_growth["final_mean"] >= 3 * df_growth["initial_mean"]
df_growth["late_peak"] = df_growth["peak_month"] >= LATE_PEAK_MONTH
df_growth.to_csv(f"{OUTPUT_DIR}/adoption_top{TOP_GROWTH}_growth.csv", index=False)

# Numbers quoted in the text
print(
    f"AI-evidence PRs: {total:,} in {n_repos} repositories "
    f"({100 * n_repos / 450:.1f}% of 450), {len(df_tool)} tools"
)

top_f = df_tool.head(TOP_TOOLS)
print(f"\nTop {TOP_TOOLS} tools:")
print(top_f.to_string(index=False))
others_f = total - top_f["prs"].sum()
print(f"Other {len(df_tool) - TOP_TOOLS} tools: {others_f:,}")
print(f"Top 2: {pct(top_f['prs'].head(2).sum())} | Top 4: {pct(top_f['prs'].sum())}")

top_l = df_lang.head(TOP_LANGUAGES)
rest_l = df_lang.iloc[TOP_LANGUAGES:]
print(f"\nTop {TOP_LANGUAGES} languages:")
print(top_l.to_string(index=False))
print(
    f"Other {len(rest_l)} languages: {rest_l['prs'].sum():,} PRs, "
    f"{rest_l['repositories'].sum()} repositories"
)
print(f"Top {TOP_LANGUAGES}: {pct(top_l['prs'].sum())}")
less = df_lang.sort_values("repositories").iloc[0]
print(f"Lowest coverage: {less['language']} ({less['repositories']} of 30)")

top_r = df_rank.head(TOP_REPOSITORIES)
print(f"\nTop {TOP_REPOSITORIES} repositories:")
print(top_r.to_string(index=False))
print(f"Top {TOP_REPOSITORIES}: {pct(top_r['prs'].sum())}")

_ok = df_growth["tripled"] & df_growth["late_peak"]
print(
    f"\nTripled the monthly mean ({INITIAL_WINDOW[0]}..{INITIAL_WINDOW[1]} -> "
    f"{FINAL_WINDOW[0]}..{FINAL_WINDOW[1]}) and peak from {LATE_PEAK_MONTH} on: "
    f"top {TOP_REPOSITORIES}: {_ok.head(TOP_REPOSITORIES).sum()} | "
    f"top {TOP_GROWTH}: {_ok.sum()}"
)
print(df_growth.round(1).to_string(index=False))

# Explicit evidence of robobun: robobun is in authors.csv, so all its PRs are
# pr-author matches. To show which tool the evidence outside the account
# points to, the heuristics are re-applied to its raw PRs with an authors.csv
# without the robobun pattern.
_rows = [
    r
    for r in csv.DictReader(open(classify_prs.PATTERNS_DIR / "authors.csv", encoding="utf-8"))
    if r["pattern"].strip().lower() != HIGHLIGHTED_AUTHOR
]
with tempfile.NamedTemporaryFile(
    "w", suffix=".csv", delete=False, newline="", encoding="utf-8"
) as f:
    _w = csv.DictWriter(f, fieldnames=_rows[0].keys())
    _w.writeheader()
    _w.writerows(_rows)
classify_prs.AUTHORS, classify_prs.AUTHORS_NAMES = classify_prs.load_patterns(
    Path(f.name), word_boundary=True
)
os.unlink(f.name)

_bun = json.loads((classify_prs.RAW_DIR / "oven-sh__bun.json").read_text(encoding="utf-8"))
_evid = Counter()
for _pr in _bun["prs"]:
    if ((_pr.get("author") or {}).get("login") or "") != HIGHLIGHTED_AUTHOR:
        continue
    _r = classify_prs.classify_pr(_pr, "Human", None)
    _evid[(_r["ai"], _r["source"]) if _r else ("", "no_evidence")] += 1
df_evid = pd.DataFrame([{"tool": t, "source": s, "prs": n} for (t, s), n in _evid.items()])
df_evid.to_csv(f"{OUTPUT_DIR}/{HIGHLIGHTED_AUTHOR}_evidence.csv", index=False)
_with = df_evid[df_evid["source"] != "no_evidence"]
print(
    f"\n{HIGHLIGHTED_AUTHOR}: {df_evid['prs'].sum():,} PRs, {_with['prs'].sum():,} with explicit "
    f"evidence, tools: {sorted(_with['tool'].unique())}"
)

# PRs whose branch starts with serena/ (pattern removed from branches.csv) and
# their authors, to show that none comes from the agent. Files are filtered by
# their bytes before parsing the JSON.
REMOVED_PREFIX = "serena/"
_re_prefix = re.compile(rb'"headRefName":\s*"' + re.escape(REMOVED_PREFIX.encode()), re.I)
_rows_prefix = []
for fpath in sorted(classify_prs.RAW_DIR.glob("*.json")):
    raw = fpath.read_bytes()
    if not _re_prefix.search(raw):
        continue
    data = json.loads(raw)
    for _pr in data["prs"]:
        if not (_pr.get("headRefName") or "").lower().startswith(REMOVED_PREFIX):
            continue
        _names = {
            ((c["commit"].get("author") or {}).get("name") or "")
            for c in (_pr.get("commits") or {}).get("nodes", [])
        }
        _rows_prefix.append(
            {
                "repository": f"{data['owner']}/{data['repo']}",
                "pr": _pr.get("number"),
                "author": (_pr.get("author") or {}).get("login"),
                "commit_author_names": "; ".join(sorted(_names)),
                "created_at": _pr.get("createdAt"),
                "branch": _pr.get("headRefName"),
            }
        )
df_prefix = pd.DataFrame(_rows_prefix)
df_prefix.to_csv(f"{OUTPUT_DIR}/branches_serena.csv", index=False)
print(
    f"\nPRs with branch {REMOVED_PREFIX}: {len(df_prefix)} from "
    f"{df_prefix['author'].nunique()} authors"
)
print(df_prefix.groupby(["author", "commit_author_names"]).size().to_string())

# Figure: one acmart column (~3.33 in), TrueType fonts embedded.
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8a85",
        "axes.linewidth": 0.6,
        "xtick.color": "#4a4a46",
        "ytick.color": "#4a4a46",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

months = pd.period_range(FIGURE_START_MONTH, pd.Period(CUTOFF_MONTH, freq="M") - 1, freq="M")
labels = [m.strftime("%Y-%m") for m in months]
x = range(len(months))

fig, axes = plt.subplots(len(GROUPS), 1, figsize=(3.33, 2.9), sharex=True)
for ax, (category, name) in zip(axes, GROUPS):
    g = df_time[df_time["group"] == category].set_index("Month").reindex(labels, fill_value=0)
    base = pd.Series(0, index=labels)
    for col, label, color in STATES:
        ax.bar(
            x,
            g[col],
            bottom=base,
            width=0.8,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
        base = base + g[col]
    ax.set_title(f"{name} ({int(g['prs'].sum()):,} PRs)", loc="left", pad=3)
    ax.set_ylabel("Pull requests")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", color="#e4e4df", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.margins(x=0.01)

axes[0].legend(loc="upper left", frameon=False, ncol=3, handlelength=1.0, columnspacing=1.0)
ticks = [i for i, m in enumerate(months) if m.month in (1, 4, 7, 10)]
axes[-1].set_xticks(ticks)
axes[-1].set_xticklabels([months[i].strftime("%b\n%Y") for i in ticks])

fig.tight_layout(h_pad=0.8)
fig.savefig(FIGURE, bbox_inches="tight")
print(f"\nFigure saved to {FIGURE}")
