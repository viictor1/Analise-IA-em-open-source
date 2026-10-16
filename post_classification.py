"""Post-processing of the classify_prs.py outputs.

  (i)  Confirmed false positives (data/false_positives.csv) are moved from
       the AI file to the human file, with the AI columns cleared.
  (ii) Implausible Signal_Date values (year < 2000, e.g. a commit dated
       1970-01-01) are replaced by the PR creation date.

Both files are rewritten with LF line endings. The script is idempotent.

Usage:
  python post_classification.py                     # default paths
  python post_classification.py <ai_csv> <human_csv>
"""

import csv
import os
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)

FP_FILE = "data/false_positives.csv"
OUTPUT_FILE = "data/ai_prs.csv"
OUTPUT_FILE_HUMAN = "data/human_prs.csv"

MIN_PLAUSIBLE_YEAR = 2000
SOURCE_SUMMARY = "results/detection_per_source.csv"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        return header, list(reader)


def write_csv(path, header, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, path)


def to_human_format(row, idx):
    row = list(row)
    row[idx["AI_Tool"]] = ""
    row[idx["Detection_Source"]] = ""
    row[idx["Signal_Date"]] = ""
    row[idx["Signal_Date_Gap_Days"]] = ""
    row[idx["Signal_Date_Suspect"]] = "False"
    return row


def main():
    csv_ai = sys.argv[1] if len(sys.argv) > 1 else OUTPUT_FILE
    human_csv = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE_HUMAN

    with open(FP_FILE, newline="", encoding="utf-8") as f:
        fp_rows = list(csv.DictReader(f))
    fp_links = {r["Link"] for r in fp_rows}
    print(f"{len(fp_links)} false positives in {FP_FILE}")

    header, ai_rows = read_csv(csv_ai)
    header_h, h_rows = read_csv(human_csv)
    if header != header_h:
        sys.exit("Error: the two CSV headers differ.")
    idx = {col: i for i, col in enumerate(header)}
    i_link = idx["Link"]

    human_links = {row[i_link] for row in h_rows}
    ai_kept, moved, already_in_human = [], 0, 0
    for row in ai_rows:
        if row[i_link] in fp_links:
            if row[i_link] in human_links:
                already_in_human += 1
            else:
                h_rows.append(to_human_format(row, idx))
                human_links.add(row[i_link])
                moved += 1
            print(
                f"  FP -> human: {row[i_link]} ({row[idx['AI_Tool']]}, {row[idx['Detection_Source']]})"
            )
            continue
        ai_kept.append(row)
    ai_rows = ai_kept

    missing = fp_links - human_links
    if missing:
        print(f"  WARNING: FPs not found in either CSV: {sorted(missing)}")

    corrected_dates = 0
    for row in ai_rows:
        sd = row[idx["Signal_Date"]]
        if sd and sd[:4].isdigit() and int(sd[:4]) < MIN_PLAUSIBLE_YEAR:
            print(f"  Signal_Date {sd} -> {row[idx['Created_At']]}: {row[i_link]}")
            row[idx["Signal_Date"]] = row[idx["Created_At"]]
            corrected_dates += 1

    dup_ai = len(ai_rows) - len({r[i_link] for r in ai_rows})
    dup_h = len(h_rows) - len(human_links)
    if dup_ai or dup_h:
        print(f"  WARNING: duplicated links: AI={dup_ai}, human={dup_h}")

    write_csv(csv_ai, header, ai_rows)
    write_csv(human_csv, header, h_rows)

    # AI PRs per detection source, before and after removing the false positives.
    fp_source = Counter(r["Detection_Source"] for r in fp_rows)
    ai_source = Counter(row[idx["Detection_Source"]] for row in ai_rows)
    os.makedirs(os.path.dirname(SOURCE_SUMMARY), exist_ok=True)
    with open(SOURCE_SUMMARY, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["detection_source", "prs_before_fp", "false_positives", "prs_ai"])
        for source in sorted(set(ai_source) | set(fp_source), key=lambda k: -ai_source[k]):
            w.writerow(
                [
                    source,
                    ai_source[source] + fp_source[source],
                    fp_source[source],
                    ai_source[source],
                ]
            )
    print(
        f"{SOURCE_SUMMARY}: "
        + ", ".join(
            f"{k}={ai_source[k] + fp_source[k]:,} (-{fp_source[k]} FP)"
            for k in sorted(ai_source, key=lambda k: -ai_source[k])
        )
    )

    print(f"\nFPs moved to the human file: {moved} (already there: {already_in_human})")
    print(f"Signal_Date values corrected: {corrected_dates}")
    print(f"AI: {len(ai_rows):,} PRs | Human: {len(h_rows):,} PRs (written with LF)")


if __name__ == "__main__":
    main()
