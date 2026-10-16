#!/usr/bin/env python3
"""Repository selection.

Fetches candidate repositories per language from the GitHub Search API
(sorted by stars), applies the automatic filters and shows the survivors one
at a time for manual confirmation: 1 = real software project, 0 = rejected
(curated lists, roadmaps, tutorials and similar).

Automatic filters:
  - not archived
  - created before START, so every repository has a pre-AI baseline
  - last push on or after LAST_ACTIVE
  - at least MIN_PRS pull requests created between START and END
Repositories that fail are logged with label "x" and never shown. Filter
results are cached in data/activity_cache.csv.

Every decision is appended to --log right away, so the run can be stopped and
resumed. The confirmed sample is written to --output.

Usage:
    python fetch_most_starred.py
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
raw_tokens = os.getenv("GITHUB_TOKENS", "")
GITHUB_TOKENS = [t.strip() for t in raw_tokens.split(",") if t.strip()]

if not GITHUB_TOKENS and os.getenv("GITHUB_TOKEN"):
    GITHUB_TOKENS = [os.getenv("GITHUB_TOKEN").strip()]

LANGUAGES = [
    "C",
    "C#",
    "C++",
    "Dart",
    "Go",
    "Java",
    "JavaScript",
    "Kotlin",
    "Objective-C",
    "PHP",
    "Python",
    "Ruby",
    "Rust",
    "Swift",
    "TypeScript",
]

# Collection window and automatic filters
START = "2022-01-01"  # start of the window; only repos created before this date are kept
END = "2026-08-31"  # end of the window
LAST_ACTIVE = "2025-09-01"  # last push must be >= this date (the 12 months before END)
MIN_PRS = 50  # minimum PRs created inside the window

API = "https://api.github.com"
SEARCH_URL = f"{API}/search/repositories"
CACHE = Path("data/activity_cache.csv")
CACHE_FIELDS = ["full_name", "archived", "created", "pushed", "prs", "passed", "reason"]
FIELDS = ["url", "main language", "stars", "owner", "project", "branch"]


def load_labeled(log_path):
    """owner/project -> row, from a previous run's log."""
    labeled = {}
    if log_path.exists():
        with log_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labeled[f"{row['owner']}/{row['project']}"] = row
    return labeled


def load_cache():
    if not CACHE.exists():
        return {}
    with CACHE.open(newline="", encoding="utf-8") as f:
        return {r["full_name"]: r for r in csv.DictReader(f)}


_token_idx = 0
_blocked_until = {}


def next_token():
    """Round-robin over GITHUB_TOKENS, skipping tokens still rate limited.
    Sleeps until the earliest reset when all of them are blocked."""
    global _token_idx
    n = len(GITHUB_TOKENS)
    while True:
        now = time.time()
        for _ in range(n):
            i = _token_idx % n
            _token_idx += 1
            if _blocked_until.get(i, 0) <= now:
                return i
        wait = max(min(_blocked_until.values()) - now, 1)
        print(f"  all tokens rate limited, sleeping {wait:.0f}s", file=sys.stderr)
        time.sleep(wait)


def search_get(session, url, params):
    """GET on a search endpoint, rotating tokens when one hits the rate limit."""
    while True:
        i = next_token()
        resp = session.get(
            url, params=params, timeout=30, headers={"Authorization": f"Bearer {GITHUB_TOKENS[i]}"}
        )
        if resp.status_code in (403, 429) and "rate limit" in resp.text.lower():
            retry_after = resp.headers.get("Retry-After")
            if retry_after:  # secondary rate limit
                until = time.time() + float(retry_after) + 1
            else:  # primary limit; margin avoids hitting the old window right at reset
                until = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60)) + 2
            _blocked_until[i] = until
            continue
        resp.raise_for_status()
        return resp


def pr_count(session, full_name, period):
    params = {"q": f"repo:{full_name} is:pr created:{period}", "per_page": 1}
    resp = search_get(session, f"{API}/search/issues", params)
    time.sleep(2.1 / len(GITHUB_TOKENS))
    return resp.json()["total_count"]


def check(session, full_name, cache, info=None):
    """Returns (passed, reason) for the automatic filters. Results are cached."""
    if full_name in cache:
        c = cache[full_name]
        return c["passed"] == "1", c["reason"]

    if info is None:
        resp = session.get(f"{API}/repos/{full_name}", timeout=30)
        info = resp.json() if resp.status_code == 200 else {}
    created = (info.get("created_at") or "")[:10]
    pushed = (info.get("pushed_at") or "")[:10]
    archived = bool(info.get("archived", True))

    prs = 0
    if archived:
        reason = "archived or unavailable"
    elif not created or created >= START:
        reason = f"created {created or '?'}"
    elif not pushed or pushed < LAST_ACTIVE:
        reason = f"last push {pushed or '?'}"
    else:
        prs = pr_count(session, full_name, f"{START}..{END}")
        reason = "ok" if prs >= MIN_PRS else f"only {prs} PRs in window"
    passed = reason == "ok"

    row = {
        "full_name": full_name,
        "archived": int(archived),
        "created": created,
        "pushed": pushed,
        "prs": prs,
        "passed": int(passed),
        "reason": reason,
    }
    is_new = not CACHE.exists()
    with CACHE.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)
    cache[full_name] = {k: str(v) for k, v in row.items()}
    return passed, reason


def fetch_page(language, session, page, include_forks):
    query = f"language:{language} created:<{START} archived:false pushed:>={LAST_ACTIVE}"
    if not include_forks:
        query += " fork:false"
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": 100, "page": page}
    resp = search_get(session, SEARCH_URL, params)
    return resp.json().get("items", [])


def prompt_label(item, language, confirmed, target):
    print(f"\n[{language}] {confirmed}/{target} confirmed so far")
    print(f"  {item['full_name']}  ({item['stargazers_count']} stars)")
    print(f"  {item['html_url']}")
    if item.get("description"):
        print(f"  {item['description']}")
    if item.get("topics"):
        print(f"  topics: {', '.join(item['topics'])}")
    while True:
        ans = input("  valid project? [1=yes / 0=no / q=quit]: ").strip().lower()
        if ans in ("1", "0"):
            return ans
        if ans == "q":
            return None
        print("  please enter 1, 0, or q")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/selected_repos.csv"),
        help="final dataset of confirmed (label=1) projects",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("data/repo_selection_log.csv"),
        help="full audit trail of every repo shown, with your 1/0 (or x for automatic rejections) -- also used to resume",
    )
    parser.add_argument("--per-language", type=int, default=30)
    parser.add_argument("--include-forks", action="store_true")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="pages of 100 to pull per language before giving up (10 = GitHub search's 1000-result cap)",
    )
    args = parser.parse_args()

    if not GITHUB_TOKENS:
        sys.exit("Set GITHUB_TOKENS (comma separated) or GITHUB_TOKEN in your environment or .env.")

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {GITHUB_TOKENS[0]}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )

    already = load_labeled(args.log)
    cache = load_cache()

    to_check = [k for k, r in already.items() if r["label"] == "1"]
    demoted = False
    for i, key in enumerate(to_check, 1):
        ok, reason = check(session, key, cache)
        if ok:
            print(f"  [{i}/{len(to_check)}] {key}: ok")
        else:
            print(f"  [{i}/{len(to_check)}] {key}: removed ({reason})")
            already[key]["label"] = "x"
            demoted = True
    if demoted:
        with args.log.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS + ["label"])
            w.writeheader()
            w.writerows(already.values())

    is_new_log = not args.log.exists()
    log_f = args.log.open("a", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_f, fieldnames=FIELDS + ["label"])
    if is_new_log:
        log_writer.writeheader()

    quit_requested = False
    for lang in LANGUAGES:
        if quit_requested:
            break

        confirmed = sum(
            1 for r in already.values() if r["main language"] == lang and r["label"] == "1"
        )
        if confirmed >= args.per_language:
            print(f"{lang}: already has {confirmed}/{args.per_language} confirmed, skipping")
            continue

        print(f"\n=== {lang} ({confirmed}/{args.per_language} confirmed so far) ===")
        for page in range(1, args.max_pages + 1):
            items = fetch_page(lang, session, page, args.include_forks)
            if not items:
                break

            for item in items:
                key = item["full_name"]
                if key in already:
                    continue

                row = {
                    "url": item["html_url"],
                    "main language": lang,
                    "stars": item["stargazers_count"],
                    "owner": item["owner"]["login"],
                    "project": item["name"],
                    "branch": item["default_branch"],
                }

                ok, reason = check(session, key, cache, info=item)
                if not ok:
                    print(f"  skipped {key}: {reason}")
                    row["label"] = "x"
                    already[key] = row
                    log_writer.writerow(row)
                    log_f.flush()
                    continue

                ans = prompt_label(item, lang, confirmed, args.per_language)
                if ans is None:
                    quit_requested = True
                    break

                row["label"] = ans
                already[key] = row
                log_writer.writerow(row)
                log_f.flush()

                if ans == "1":
                    confirmed += 1
                    if confirmed >= args.per_language:
                        break

            if quit_requested or confirmed >= args.per_language:
                break
            time.sleep(2)

        if confirmed < args.per_language and not quit_requested:
            print(
                f"  warning: only {confirmed}/{args.per_language} confirmed for {lang} "
                f"after {args.max_pages} pages -- raise --max-pages",
                file=sys.stderr,
            )

    log_f.close()

    by_lang = defaultdict(list)
    for r in already.values():
        if r["label"] == "1":
            by_lang[r["main language"]].append(r)

    short = {
        lang: len(by_lang.get(lang, []))
        for lang in LANGUAGES
        if len(by_lang.get(lang, [])) < args.per_language
    }
    if short:
        print(
            f"\nNot writing {args.output} -- these languages don't have {args.per_language} confirmed yet:",
            file=sys.stderr,
        )
        for lang, count in short.items():
            print(f"  {lang}: {count}/{args.per_language}", file=sys.stderr)
        print(
            "Rerun (add --max-pages if you've exhausted 10 pages) to keep labeling and reach exactly "
            f"{args.per_language} for each language.",
            file=sys.stderr,
        )
        sys.exit(1)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for lang in LANGUAGES:
            rows = sorted(by_lang[lang], key=lambda r: int(r["stars"]), reverse=True)
            for r in rows[: args.per_language]:
                writer.writerow({k: r[k] for k in FIELDS})

    print(
        f"\nWrote exactly {args.per_language} confirmed repos per language "
        f"({args.per_language * len(LANGUAGES)} total) to {args.output}"
    )
    print(f"Full label log at {args.log}")


if __name__ == "__main__":
    main()
