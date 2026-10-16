"""Collection gaps in the 13 repositories collected through the search API.

In these repositories a removed pull request broke the GraphQL listing, so
fetch_prs.py fell back to the search API and checked completeness by
enumerating PR numbers (fields unloadable_numbers and recovered_by_number of
the raw JSON). Writes one row per affected repository plus a TOTAL row with
the share of removed PRs over the whole collection.
"""

import glob
import json

import pandas as pd

rows, total_450 = [], 0
for f in sorted(glob.glob("data/raw_prs/*.json")):
    d = json.load(open(f, encoding="utf-8"))
    total_450 += len(d["prs"])
    if "unloadable_numbers" not in d:
        continue
    n, u = len(d["prs"]), len(d["unloadable_numbers"])
    rows.append(
        {
            "repo": f"{d['owner']}/{d['repo']}",
            "collected": n,
            "removed": u,
            "pct_removed": 100 * u / (n + u),
            "recovered_by_number": len(d["recovered_by_number"]),
            "search_since": d.get("search_fallback_from"),
        }
    )

t = pd.DataFrame(rows)
tot = {
    "repo": "TOTAL",
    "collected": t.collected.sum(),
    "removed": t.removed.sum(),
    "pct_removed": 100 * t.removed.sum() / (t.collected.sum() + t.removed.sum()),
    "recovered_by_number": t.recovered_by_number.sum(),
    "search_since": None,
    "repos_with_gap": len(t),
    "collected_450": total_450,
    "pct_removed_450": 100 * t.removed.sum() / (total_450 + t.removed.sum()),
}
out = pd.concat([t, pd.DataFrame([tot])], ignore_index=True)
out.to_csv("results/collection_gaps.csv", index=False)
print(out.round(3).to_string(index=False))
