"""Collected PRs per group, before and after the month cutoff.

The cutoff uses the PR creation month (Created_At < CUTOFF_MONTH), as in the
RQs. The adoption characterization dates AI PRs by their first signal, so the
column with_ai_by_signal reports that count too.

Groups (disjoint, they add up to the collected total):
  ai_agent, human_ai          data/ai_prs.csv (PR_Author_Category)
  control                     data/human_prs.csv, with login
  control_deleted_account     data/human_prs.csv, without login
  automation                  the remaining raw PRs (bot/organization
                              accounts and the known-bots list), with the 13
                              User-typed automation accounts counted
                              separately in automation_13_accounts; the
                              automation PRs with AI evidence
                              (results/automation_with_ai_evidence.csv, from
                              classify_prs.py) are also split out in
                              automation_with_ai and automation_without_ai
"""

import ast
import glob
import json

import pandas as pd

CUTOFF_MONTH = "2026-08"
OUTPUT_FILE = "results/cutoff_counts.csv"
AUTOMATION_AI_FILE = "results/automation_with_ai_evidence.csv"
# the 13 User-typed automation accounts added to bots.csv
ACCOUNTS_13 = {
    "angular-robot",
    "brewtestbot",
    "engine-flutter-autoroll",
    "fw-bot",
    "grafanabot",
    "miss-islington",
    "robot-ch-test-poll",
    "robot-ch-test-poll2",
    "robot-ch-test-poll3",
    "robot-ch-test-poll4",
    "robot-clickhouse",
    "robot-clickhouse-ci-1",
    "robot-clickhouse-ci-2",
}


def _login(author):
    if isinstance(author, str):
        try:
            author = ast.literal_eval(author)
        except (ValueError, SyntaxError):
            return None
    return (author or {}).get("login") if isinstance(author, dict) else None


raw = {}
for f in sorted(glob.glob("data/raw_prs/*.json")):
    d = json.load(open(f, encoding="utf-8"))
    repo = f"{d['owner']}/{d['repo']}"
    for pr in d["prs"]:
        raw[(repo.lower(), str(pr["number"]))] = (
            str(pr["createdAt"])[:7],
            _login(pr.get("author")),
        )


def _key(link: str):
    p = link.rstrip("/").split("/")
    return (f"{p[-4]}/{p[-3]}".lower(), p[-1])


ai = pd.read_csv(
    "data/ai_prs.csv",
    usecols=["Link", "Created_At", "Signal_Date", "PR_Author_Category"],
    dtype=str,
)
hum = pd.read_csv("data/human_prs.csv", usecols=["Link", "Created_At", "PR_Author"], dtype=str)
ai["month"] = ai["Created_At"].str[:7]
ai["signal_month"] = pd.to_datetime(
    ai["Signal_Date"].fillna(ai["Created_At"]), utc=True, errors="coerce"
).dt.strftime("%Y-%m")
hum["month"] = hum["Created_At"].str[:7]

keys_class = set(map(_key, ai["Link"])) | set(map(_key, hum["Link"]))
rest = [
    (m, (l or "").lower().removesuffix("[bot]"), k)
    for k, (m, l) in raw.items()
    if k not in keys_class
]
auto_ai = pd.read_csv(AUTOMATION_AI_FILE, usecols=["Repository", "PR_Number"], dtype=str)
keys_auto_ai = set(zip(auto_ai["Repository"].str.lower(), auto_ai["PR_Number"]))


def _row(group, months, signal=None):
    months = pd.Series(months)
    r = {
        "group": group,
        "collected": len(months),
        "until_2026_07": int((months < CUTOFF_MONTH).sum()),
    }
    if signal is not None:
        r["with_ai_by_signal"] = int((pd.Series(signal) < CUTOFF_MONTH).sum())
    return r


agent = ai["PR_Author_Category"] == "AI_Agent"
no_login = hum["PR_Author"].isna() | (hum["PR_Author"].str.strip() == "")
rows = [
    _row("total", [m for m, _ in raw.values()]),
    _row("with_ai", ai["month"], ai["signal_month"]),
    _row("ai_agent", ai.loc[agent, "month"], ai.loc[agent, "signal_month"]),
    _row("human_ai", ai.loc[~agent, "month"], ai.loc[~agent, "signal_month"]),
    _row("control", hum.loc[~no_login, "month"]),
    _row("control_deleted_account", hum.loc[no_login, "month"]),
    _row("automation", [m for m, _, _ in rest]),
    _row("automation_with_ai", [m for m, _, k in rest if k in keys_auto_ai]),
    _row("automation_without_ai", [m for m, _, k in rest if k not in keys_auto_ai]),
    _row("automation_13_accounts", [m for m, l, _ in rest if l in ACCOUNTS_13]),
    _row(
        "automation_13_accounts_with_ai",
        [m for m, l, k in rest if l in ACCOUNTS_13 and k in keys_auto_ai],
    ),
]
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)
print(df.to_string(index=False))
assert len(keys_auto_ai) == len(auto_ai) and keys_auto_ai <= {k for _, _, k in rest}, (
    f"{AUTOMATION_AI_FILE} does not match the automation PRs"
)
summed = (
    df.set_index("group").loc[["with_ai", "control", "control_deleted_account", "automation"]].sum()
)
print(
    f"\nsum of groups: {summed['collected']:,} collected, {summed['until_2026_07']:,} until 2026-07 "
    f"(total {df.loc[0, 'collected']:,} / {df.loc[0, 'until_2026_07']:,})"
)
