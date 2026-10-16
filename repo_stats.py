#!/usr/bin/env python3
"""Descriptive metrics of the sampled repositories and the LaTeX tables.

Usage:
    python repo_stats.py data/selected_repos.csv

Writes results/repo_metrics.csv (one row per repository; already collected
repositories are skipped) and results/tables.tex.
"""

import csv
import os
import statistics as st
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()
raw_tokens = os.getenv("GITHUB_TOKENS", "")
GITHUB_TOKENS = [t.strip() for t in raw_tokens.split(",") if t.strip()]

if not GITHUB_TOKENS and os.getenv("GITHUB_TOKEN"):
    GITHUB_TOKENS = [os.getenv("GITHUB_TOKEN").strip()]

if not GITHUB_TOKENS:
    sys.exit("No token found: set GITHUB_TOKENS (comma-separated) or GITHUB_TOKEN in .env")

_idx = 0  # index of the token currently in use
_blocked_until = {}  # token index -> epoch time when it can be used again
OUT_CSV = "results/repo_metrics.csv"
OUT_TEX = "results/tables.tex"
# "contributors" is a legacy column (no longer collected), kept so that an
# existing repo_metrics.csv stays compatible.
FIELDS = [
    "repo",
    "language",
    "stars",
    "forks",
    "contributors",
    "prs",
    "merged_prs",
    "issues",
    "commits",
    "age_years",
    "days_since_push",
    "archived",
]

GQL = """
query($o:String!,$n:String!){
  repository(owner:$o,name:$n){
    stargazerCount forkCount createdAt pushedAt isArchived
    pullRequests{totalCount}
    merged:pullRequests(states:MERGED){totalCount}
    issues{totalCount}
    defaultBranchRef{target{... on Commit{history{totalCount}}}}
  }
}"""


def retry(fn, *args, tries=4):
    for i in range(tries):
        try:
            return fn(*args)
        except Exception as exc:  # network errors, 5xx, secondary rate limits
            if i == tries - 1:
                raise
            print(f"  retry after error: {exc}", file=sys.stderr)
            time.sleep(2 ** (i + 1))


def headers():
    return {"Authorization": f"Bearer {GITHUB_TOKENS[_idx]}"}


def mark_limited(reset_ts=None):
    """Blocks the current token and switches to a usable one (or sleeps until the
    earliest reset).
    """
    global _idx
    _blocked_until[_idx] = reset_ts or time.time() + 60
    while True:
        now = time.time()
        for k in range(1, len(GITHUB_TOKENS) + 1):
            j = (_idx + k) % len(GITHUB_TOKENS)
            if _blocked_until.get(j, 0) <= now:
                _idx = j
                return
        wait = min(_blocked_until.values()) - now + 5
        print(f"  all tokens rate-limited; sleeping {wait:.0f}s", file=sys.stderr)
        time.sleep(max(wait, 1))


def reset_from(resp):
    """Epoch time at which the limit lifts, from Retry-After or X-RateLimit-Reset."""
    if resp.headers.get("Retry-After"):
        return time.time() + int(resp.headers["Retry-After"]) + 5
    if resp.headers.get("X-RateLimit-Reset"):
        return int(resp.headers["X-RateLimit-Reset"]) + 5
    return None


def api(method, url, **kw):
    """requests wrapper that rotates tokens when the current one is rate-limited."""
    while True:
        r = requests.request(method, url, headers=headers(), timeout=60, **kw)
        limited = r.status_code in (403, 429) and (
            r.headers.get("X-RateLimit-Remaining") == "0"
            or r.headers.get("Retry-After")
            or "rate limit" in r.text.lower()
        )
        if limited:
            mark_limited(reset_from(r))
            continue
        return r


def graphql(owner, name):
    while True:
        r = api(
            "POST",
            "https://api.github.com/graphql",
            json={"query": GQL, "variables": {"o": owner, "n": name}},
        )
        r.raise_for_status()
        data = r.json()
        errors = data.get("errors")
        if errors:
            if any(e.get("type") == "RATE_LIMITED" for e in errors):
                mark_limited(reset_from(r))
                continue
            raise RuntimeError(errors)
        return data["data"]["repository"]


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def collect(rows):
    done = {}
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="") as f:
            done = {r["repo"]: r for r in csv.DictReader(f)}
    new_file = not os.path.exists(OUT_CSV)
    now = datetime.now(timezone.utc)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, row in enumerate(rows, 1):
            repo = f"{row['owner']}/{row['project']}"
            if repo in done:
                continue
            print(f"[{i}/{len(rows)}] {repo}")
            g = retry(graphql, row["owner"], row["project"])
            hist = ((g.get("defaultBranchRef") or {}).get("target") or {}).get("history") or {}
            out = {
                "repo": repo,
                "language": row["main language"],
                "stars": g["stargazerCount"],
                "forks": g["forkCount"],
                "contributors": "",
                "prs": g["pullRequests"]["totalCount"],
                "merged_prs": g["merged"]["totalCount"],
                "issues": g["issues"]["totalCount"],
                "commits": hist.get("totalCount", 0),
                "age_years": round((now - parse_ts(g["createdAt"])).days / 365.25, 2),
                "days_since_push": max(0, (now - parse_ts(g["pushedAt"])).days),
                "archived": int(g["isArchived"]),
            }
            w.writerow(out)
            f.flush()
            done[repo] = out
            time.sleep(0.3)
    return done


def esc(s):
    return s.replace("_", r"\_").replace("#", r"\#").replace("&", r"\&")


def num(x, dec=0):
    x = float(x)
    if dec == 0:
        return f"{round(x):,}".replace(",", r"\,")
    return f"{x:.{dec}f}"


def col(records, key):
    """Numeric values of a column, ignoring blanks (e.g. missing contributors)."""
    return [float(r[key]) for r in records if r[key] not in ("", None)]


def agg(fn, values, dec=0):
    """Apply min/median/max; return '--' when there is no value."""
    return num(fn(values), dec) if values else "--"


def make_tables(records, lang_order, top5):
    for r in records:  # PR merge rate (%), only defined for repos with at least one PR
        prs, merged = float(r["prs"]), float(r["merged_prs"])
        r["merge_rate"] = round(100 * merged / prs, 2) if prs > 0 else ""

    def mmm(rs, key, dec=0):
        v = col(rs, key)
        return f"{agg(min, v, dec)} & {agg(st.median, v, dec)} & {agg(max, v, dec)}"

    def med(rs, key, dec=0):
        return agg(st.median, col(rs, key), dec)

    # ---- Table 1: overall min / median / max ----
    metrics = [
        ("Stars", "stars", 0),
        ("Forks", "forks", 0),
        ("Pull requests (all)", "prs", 0),
        ("Merged pull requests", "merged_prs", 0),
        (r"PR merge rate (\%)", "merge_rate", 2),
        ("Issues (all)", "issues", 0),
        ("Commits (default branch)", "commits", 0),
        ("Repository age (years)", "age_years", 1),
        ("Days since last push", "days_since_push", 0),
    ]
    t1 = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Descriptive statistics of the 450 sampled repositories. "
        r"Counts cover each repository's whole history, and days since last push were measured when the metrics were collected. The PR merge rate is the share of merged pull requests among all pull requests.}",
        r"\label{tab:sample-overall}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Min} & \textbf{Median} & \textbf{Max} \\",
        r"\midrule",
    ]
    for label, key, dec in metrics:
        t1.append(f"{label} & {mmm(records, key, dec)} \\\\")
    t1 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    # ---- Table 2: per language, min / median / max of stars and PRs ----
    t2 = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Sample composition by main programming language: number of repositories (N) "
        r"and minimum, median and maximum of stars and pull requests.}",
        r"\label{tab:sample-by-language}",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{\textbf{Stars}} & \multicolumn{3}{c}{\textbf{Pull requests}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"\textbf{Language} & \textbf{N} & Min & Median & Max & Min & Median & Max \\",
        r"\midrule",
    ]
    for lang in lang_order:
        rs = [r for r in records if r["language"] == lang]
        t2.append(f"{esc(lang)} & {len(rs)} & {mmm(rs, 'stars')} & {mmm(rs, 'prs')} \\\\")
    t2 += [
        r"\midrule",
        f"All & {len(records)} & {mmm(records, 'stars')} & {mmm(records, 'prs')} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    # ---- Table 3: per language, other medians + top-5 repositories ----
    t3 = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Median activity profile per language and the five most-starred repositories of each.}",
        r"\label{tab:sample-profile}",
        r"\small",
        r"\begin{tabular}{lrrrrp{6.5cm}}",
        r"\toprule",
        r"\textbf{Language} & \textbf{Commits} & \textbf{Forks} & \textbf{Age (yr)} & "
        r"\textbf{Merge rate (\%)} & \textbf{Top-5 repositories} \\",
        r"\midrule",
    ]
    for lang in lang_order:
        rs = [r for r in records if r["language"] == lang]
        names = ", ".join(r"\texttt{" + esc(n) + "}" for n in top5[lang])
        t3.append(
            f"{esc(lang)} & {med(rs, 'commits')} & {med(rs, 'forks')} & "
            f"{med(rs, 'age_years', 1)} & {med(rs, 'merge_rate', 1)} & {names} \\\\"
        )
    t3 += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(t1) + "\n\n" + "\n".join(t2) + "\n\n" + "\n".join(t3) + "\n"


def main():
    with open(sys.argv[1], newline="") as f:
        rows = list(csv.DictReader(f))
    lang_order = list(OrderedDict.fromkeys(r["main language"] for r in rows))
    top5 = {
        l: [
            r["project"]
            for r in sorted(
                (x for x in rows if x["main language"] == l), key=lambda x: -int(x["stars"])
            )[:5]
        ]
        for l in lang_order
    }
    done = collect(rows)
    records = [done[f"{r['owner']}/{r['project']}"] for r in rows]
    for r in records:
        r["days_since_push"] = max(0, int(float(r["days_since_push"])))
    with open(OUT_TEX, "w") as f:
        f.write(make_tables(records, lang_order, top5))
    archived = sum(int(r["archived"]) for r in records)
    print(f"\nWrote {OUT_TEX}. Archived repos in sample: {archived}")


if __name__ == "__main__":
    main()
