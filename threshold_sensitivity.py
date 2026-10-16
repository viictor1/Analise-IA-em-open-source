"""Sensitivity of the 180-day Signal_Date_Suspect threshold
(SIGNAL_DATE_GAP_THRESHOLD_DAYS in classify_prs.py).

Reads the final classified dataset instead of reclassifying the raw JSON, so
it measures exactly the PRs used in the analyses. For commit signals the CSV
stores the chosen Signal_Date and the committed - authored gap
(Signal_Date_Gap_Days), which is enough to rebuild both dates:
  suspect (gap > 180): Signal_Date = committed, authored = committed - gap
  normal  (gap <= 180): Signal_Date = authored,  committed = authored + gap
The gap is stored with one decimal place, so the rebuilt date is off by at
most about one hour, which does not affect monthly counts.

Writes results/gap_distribution.csv and results/threshold_sensitivity.csv.
"""

import os

import pandas as pd

INPUT_AI = "data/ai_prs.csv"
OUTPUT_DIR = "results"

CUTOFF_MONTH = "2026-08"
ADOPTED_THRESHOLD = 180
THRESHOLDS = [90, 180, 365]

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_AI, dtype=str, keep_default_na=False)
signal = pd.to_datetime(df["Signal_Date"], utc=True)
gap = pd.to_numeric(df["Signal_Date_Gap_Days"], errors="coerce")
suspect = df["Signal_Date_Suspect"] == "True"

authored = signal.where(~suspect, signal - pd.to_timedelta(gap, unit="D"))
committed = signal.where(suspect, signal + pd.to_timedelta(gap, unit="D"))

# Gap distribution (only commit signals have a gap)
with_gap = gap.notna()
df_gap = df.loc[with_gap, ["Repository", "PR_Number", "AI_Tool", "Detection_Source"]].assign(
    gap_days=gap[with_gap]
)
df_gap.sort_values("gap_days").to_csv(f"{OUTPUT_DIR}/gap_distribution.csv", index=False)

g = gap[with_gap]
print(f"PRs with a commit signal: {len(g):,}")
print(f"  gap < 7 days:        {100 * (g < 7).mean():.1f}%")
print(f"  7 <= gap <= 180 days: {100 * ((g >= 7) & (g <= 180)).mean():.1f}%")
print(f"  gap > 180 days:      {100 * (g > 180).mean():.2f}% ({(g > 180).sum()} PRs)")

# Monthly counts under each threshold
counts = {}
for threshold in THRESHOLDS:
    data = signal.where(gap.isna(), committed.where(gap > threshold, authored))
    month = data.dt.strftime("%Y-%m")
    counts[f"threshold_{threshold}d"] = month[month < CUTOFF_MONTH].value_counts()

df_sens = pd.DataFrame(counts).fillna(0).astype(int).sort_index().rename_axis("month")
df_sens.to_csv(f"{OUTPUT_DIR}/threshold_sensitivity.csv")

base = df_sens[f"threshold_{ADOPTED_THRESHOLD}d"]
print(f"\nPRs with AI evidence before {CUTOFF_MONTH}, per threshold:")
print(df_sens.sum().to_string())
for threshold in THRESHOLDS:
    if threshold == ADOPTED_THRESHOLD:
        continue
    diff = (df_sens[f"threshold_{threshold}d"] - base).abs()
    month = diff.idxmax()
    print(
        f"\n{threshold} vs {ADOPTED_THRESHOLD} days: largest monthly difference of "
        f"{diff.max()} PRs in {month} ({100 * diff.max() / base[month]:.1f}% "
        f"of the {base[month]:,} PRs of that month)"
    )

# PR author, branch and label signals (including every AI_Agent PR) use
# Created_At and have no gap, so they do not depend on the threshold.
agent = df["PR_Author_Category"] == "AI_Agent"
print(f"\nAI_Agent PRs with a gap (affected by the threshold): {int((agent & with_gap).sum())}")
