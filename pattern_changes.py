"""Effect of each change to the pattern lists of Robbes et al.

For each change in CHANGES, the PRs that the change can affect are classified
twice with the classify_prs.py rules: with the current lists in patterns/ and
with the lists without that change. The PRs whose outcome differs are counted
by transition. The outcome of a PR is its group (ai, control,
automation_with_ai, automation) and, with AI evidence, its Detection_Source;
the false positives of data/false_positives.csv count as control, as after
post_classification.py.

Writes:
  results/pattern_changes.csv           change, group and source without the
                                        change, group and source with it,
                                        PRs and distinct PR authors
  results/pattern_changes_accounts.csv  the 13 User-typed automation accounts
                                        added to bots.csv, with their PRs and
                                        rank among the control accounts by
                                        number of PRs without that change

Usage:
    python pattern_changes.py
"""

import csv
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import classify_prs as cp

OUTPUT_FILE = "results/pattern_changes.csv"
OUTPUT_ACCOUNTS = "results/pattern_changes_accounts.csv"
FP_FILE = "data/false_positives.csv"
HUMAN_FILE = "data/human_prs.csv"

COPILOT_NOREPLY = r"\d+\+Copilot@users\.noreply\.github\.com"
# specific identifier -> the generic token of the same tool it used to follow
REORDERED = {
    "aider@aider.chat": "aider",
    "claude@anthropic.com": "Claude",
    "copilot-swe-agent": "Copilot",
    "cline@example.com": "cline",
    "chatgpt@chatgpt.com": "ChatGPT",
    "chatgpt@openai.com": "ChatGPT",
    "chatGPT@no-reply.com": "ChatGPT",
    "qwen-coder@alibabacloud.com": "Qwen-Coder",
}
ROBOBUN = "robobun"
ACCOUNTS_13 = [
    "BrewTestBot",
    "fw-bot",
    "engine-flutter-autoroll",
    "miss-islington",
    "robot-clickhouse",
    "robot-clickhouse-ci-1",
    "robot-clickhouse-ci-2",
    "robot-ch-test-poll",
    "robot-ch-test-poll2",
    "robot-ch-test-poll3",
    "robot-ch-test-poll4",
    "grafanabot",
    "angular-robot",
]
DEEPSOURCE = "deepsource-autofix"
SERENA = "serena/"


def _read(name: str) -> list[dict]:
    with open(cp.PATTERNS_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _compile(rows: list[dict], **kwargs):
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=["pattern", "type", "subtype"])
        w.writeheader()
        w.writerows({k: r[k] for k in ("pattern", "type", "subtype")} for r in rows)
    try:
        return cp.load_patterns(Path(f.name), **kwargs)
    finally:
        os.unlink(f.name)


def _without(rows: list[dict], patterns) -> list[dict]:
    patterns = {p.lower() for p in patterns}
    return [r for r in rows if r["pattern"].strip().lower() not in patterns]


def _old_order(rows: list[dict]) -> list[dict]:
    """The specific identifiers of REORDERED placed after their generic token."""
    specific = {r["pattern"].strip(): r for r in rows if r["pattern"].strip() in REORDERED}
    out = []
    for r in rows:
        if r["pattern"].strip() in specific:
            continue
        out.append(r)
        out.extend(s for p, s in specific.items() if REORDERED[p] == r["pattern"].strip())
    assert len(out) == len(rows), "a generic token of REORDERED is missing from authors.csv"
    return out


AUTHORS, BOTS, BRANCHES = _read("authors.csv"), _read("bots.csv"), _read("branches.csv")
_serena_row = {"pattern": SERENA, "type": "AI-PR", "subtype": "Serena"}
_deepsource_row = {"pattern": DEEPSOURCE, "type": "AI", "subtype": "DeepSource"}

# change -> (authors, bots, branches without the change, bytes regex of the PRs it can affect)
CHANGES = {
    "add_copilot_noreply": (
        _without(AUTHORS, [COPILOT_NOREPLY]),
        BOTS,
        BRANCHES,
        rb"\d\+copilot@users\.noreply\.github\.com",
    ),
    "reorder_specific_first": (
        _old_order(AUTHORS),
        BOTS,
        BRANCHES,
        b"|".join(re.escape(p.encode()) for p in REORDERED),
    ),
    "add_robobun": (_without(AUTHORS, [ROBOBUN]), BOTS, BRANCHES, re.escape(ROBOBUN.encode())),
    "add_13_automation_accounts": (
        AUTHORS,
        _without(BOTS, ACCOUNTS_13),
        BRANCHES,
        b"|".join(re.escape(a.encode()) for a in ACCOUNTS_13),
    ),
    "move_deepsource_to_bots": (
        AUTHORS + [_deepsource_row],
        _without(BOTS, [DEEPSOURCE]),
        BRANCHES,
        re.escape(DEEPSOURCE.encode()),
    ),
    "remove_serena_branch": (
        AUTHORS,
        BOTS,
        BRANCHES + [_serena_row],
        rb'"headRefName":\s*"' + re.escape(SERENA.encode()),
    ),
}


def _lists(authors, bots, branches):
    return (
        _compile(authors, word_boundary=True),
        _compile(bots, word_boundary=True),
        _compile(branches, anchor_prefix=True),
    )


CURRENT = (
    (cp.AUTHORS, cp.AUTHORS_NAMES),
    (cp.BOTS, cp.BOTS_NAMES),
    (cp.BRANCHES, cp.BRANCHES_NAMES),
)
VARIANTS = {name: _lists(a, b, br) for name, (a, b, br, _) in CHANGES.items()}
FILTERS = {name: re.compile(rx, re.I) for name, (_, _, _, rx) in CHANGES.items()}


def _use(lists):
    (cp.AUTHORS, cp.AUTHORS_NAMES), (cp.BOTS, cp.BOTS_NAMES), (cp.BRANCHES, cp.BRANCHES_NAMES) = (
        lists
    )


FP_LINKS = set(pd.read_csv(FP_FILE)["Link"])


def outcome(pr: dict, link: str) -> tuple[str, str]:
    """(group, Detection_Source) with the lists in use, as in classify_prs.main()."""
    category, tool = cp.resolve_pr_author_category(pr)
    result = cp.classify_pr(pr, category, tool)
    if result is not None and not cp._pr_author_is_automation(pr, category, result["source"]):
        if link in FP_LINKS:
            return "control", ""
        return "ai", result["source"]
    if category == "Human" and not cp._pr_author_is_automation(pr, category, None):
        return "control", ""
    if result is not None:
        return "automation_with_ai", result["source"]
    return "automation", ""


def main():
    transitions = Counter()
    authors = defaultdict(set)
    accounts_control = Counter()
    accounts_13 = {a.lower() for a in ACCOUNTS_13}

    for fpath in sorted(cp.RAW_DIR.glob("*.json")):
        raw = fpath.read_bytes()
        hit = [name for name, rx in FILTERS.items() if rx.search(raw)]
        if not hit:
            continue
        data = json.loads(raw)
        repo = f"{data['owner']}/{data['repo']}"
        for pr in data["prs"]:
            text = json.dumps(pr).encode()
            changes = [name for name in hit if FILTERS[name].search(text)]
            if not changes:
                continue
            link = f"https://github.com/{repo}/pull/{pr['number']}"
            login = (pr.get("author") or {}).get("login") or ""
            _use(CURRENT)
            now = outcome(pr, link)
            for name in changes:
                _use(VARIANTS[name])
                before = outcome(pr, link)
                if name == "add_13_automation_accounts" and before[0] == "control":
                    if login.lower().removesuffix("[bot]") in accounts_13:
                        accounts_control[login] += 1
                if before != now:
                    transitions[(name, *before, *now)] += 1
                    authors[(name, *before, *now)].add(login)
        _use(CURRENT)

    rows = [
        {
            "change": k[0],
            "group_without": k[1],
            "source_without": k[2],
            "group_with": k[3],
            "source_with": k[4],
            "prs": n,
            "pr_authors": len(authors[k]),
        }
        for k, n in transitions.items()
    ]
    df = pd.DataFrame(rows).sort_values(["change", "prs"], ascending=[True, False])
    df.to_csv(OUTPUT_FILE, index=False)
    print(df.to_string(index=False))

    # Rank of the 13 accounts among the control accounts by PRs, had they
    # stayed out of bots.csv.
    control = pd.read_csv(HUMAN_FILE, usecols=["PR_Author"])["PR_Author"].dropna().value_counts()
    control = pd.concat([control, pd.Series(accounts_control)])
    rank = control.rank(method="min", ascending=False).astype(int)
    acc = pd.DataFrame(
        [
            {"account": a, "prs_control": int(control.get(a, 0)), "rank_control": rank.get(a)}
            for a in sorted(accounts_control, key=lambda a: -accounts_control[a])
        ]
    )
    acc.to_csv(OUTPUT_ACCOUNTS, index=False)
    print()
    print(acc.to_string(index=False))


if __name__ == "__main__":
    main()
