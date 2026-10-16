"""Offline classification of the collected PRs.

Reads the raw JSON written by fetch_prs.py and matches each PR against the
pattern lists of Robbes et al. (patterns/authors.csv, branches.csv,
labels.csv and bots.csv), writing two CSVs: PRs with AI evidence
(data/ai_prs.csv) and the human control group (data/human_prs.csv).

Detection sources (Detection_Source), checked in this order, first match
wins: pr-author > branch > label > commit-author > commit-coauthor. Commit
signals are evaluated commit by commit, author before co-author.

Author patterns are matched per field (login, email, name). A match that
comes only from the free-text name field, or a partial match inside a login
or email, needs corroboration: a full login/email match for the same tool and
the same identity among the PR's commits. Branch patterns are anchored to the
start of the branch name; author, bot and label patterns use word boundaries,
and email patterns must match the whole address.

Signal_Date is the PR creation time for pr-author, branch and label signals,
and the authored date of the matching commit for commit signals. When the
authored and committed dates of that commit are more than
SIGNAL_DATE_GAP_THRESHOLD_DAYS apart (a rebased long-lived branch), the
signal is flagged (Signal_Date_Suspect) and the committed date is used.

Closes_GFI tells whether the PR closes an issue with a label from GFI_LABELS
(adapted from Tan, Zhou and Sun, 2020).

PRs opened by automation accounts (bots.csv, or accounts GitHub types as Bot
or Organization) are dropped, unless the account resolves to AI_Agent or the
AI evidence is a branch match: several coding agents run as GitHub Apps and
are typed as Bot too.

Output columns (both CSVs):
  AI_Tool, Detection_Source, Language, Stars, Repository, PR_Number,
  Title, Link, Created_At, Updated_At, Closed_At, State, PR_Author,
  PR_Author_Category, Signal_Date, Signal_Date_Gap_Days,
  Signal_Date_Suspect, Closes_GFI, GFI_Issue_Numbers

Usage:
    python classify_prs.py
"""

import re
import csv
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass

RAW_DIR = Path("data/raw_prs")
PATTERNS_DIR = Path("patterns")
OUTPUT_FILE = "data/ai_prs.csv"
OUTPUT_FILE_HUMAN = "data/human_prs.csv"

DEBUG = True

SIGNAL_DATE_GAP_THRESHOLD_DAYS = 180


def _sanitize(name: str) -> str:
    return re.sub(r"\W+", "_", name)


def _make_bot_suffix_optional(pattern: str) -> str:
    """Makes a trailing "[bot]" optional: the GraphQL API returns bot logins
    without the suffix that the web UI shows.
    """
    m = re.search(r"\\?\[bot\\?\]$", pattern)
    if not m:
        return pattern
    return pattern[: m.start()] + r"(?:\[bot\])?"


def load_patterns(
    csv_path: Path, word_boundary: bool = False, anchor_prefix: bool = False
) -> tuple[re.Pattern, dict[str, str]]:
    """Returns (compiled_regex, sanitized_name -> original_name map).

    word_boundary: the match cannot be part of a larger identifier.
    anchor_prefix: the match must start at the beginning of the string.
    """
    patterns: dict[str, list[str]] = defaultdict(list)
    name_map: dict[str, str] = {}

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_name = row["subtype"].strip()
            tool_name = _sanitize(original_name)
            pattern = row["pattern"].strip()
            if not pattern:
                continue
            if anchor_prefix:
                pattern = f"^(?:{pattern})"
            if word_boundary:
                if "@" in pattern:
                    # full email addresses must match exactly, never as a substring
                    pattern = f"^(?:{pattern})$"
                else:
                    pattern = _make_bot_suffix_optional(pattern)
                    pattern = f"(?<!\\w)(?:{pattern})(?!\\w)"
            patterns[tool_name].append(pattern)
            name_map[tool_name] = original_name

    parts = []
    for name, pats in patterns.items():
        combined = "|".join(pats)
        parts.append(f"(?P<{name}>{combined})")

    if not parts:
        raise ValueError(f"No patterns loaded from {csv_path}")

    # case-insensitive: logins are usually lowercase while some subtypes use
    # display casing (e.g. "Claude")
    compiled = re.compile("|".join(parts), re.IGNORECASE)
    return compiled, name_map


AUTHORS, AUTHORS_NAMES = load_patterns(PATTERNS_DIR / "authors.csv", word_boundary=True)
BRANCHES, BRANCHES_NAMES = load_patterns(PATTERNS_DIR / "branches.csv", anchor_prefix=True)
LABELS, LABELS_NAMES = load_patterns(PATTERNS_DIR / "labels.csv", word_boundary=True)
BOTS, BOTS_NAMES = load_patterns(PATTERNS_DIR / "bots.csv", word_boundary=True)

CO_AUTH = re.compile(r"[Cc]o-[Aa]uthored-[Bb]y:\s*([^<]+)<([^>]+)>")


# Adapted from Tan, Zhou and Sun (2020): the most frequent labels coded as
# newcomer-suitable. "help wanted" alone is too ambiguous and is left out.
GFI_LABELS = {
    "good first issue",
    "easy",
    "low hanging fruit",
    "minor bug",
    "easy pick",
    "easy to fix",
    "good first bug",
    "beginner",
    "good first contribution",
    "good first task",
    "newbie",
    "starter bug",
    "beginner-task",
    "easy-pick",
    "minor feature",
    "help wanted (easy)",
    "up-for-grabs",
    "good-first-bug",
}


@dataclass
class Author:
    login: str
    name: str
    email: str

    def __str__(self):
        return f"{self.login or ''} | {self.name or ''} <{self.email or ''}>"


def _pr_author(pr: dict) -> Author:
    author_info = pr.get("author") or {}
    return Author(
        login=author_info.get("login", "") or "",
        name=author_info.get("name", "") or "",
        email="",
    )


def extract_co_authors(commit_message: str) -> list[Author]:
    co_authors = []
    for name, email in re.findall(CO_AUTH, commit_message or ""):
        co_authors.append(Author(login="", name=name.strip(), email=email.strip()))
    return co_authors


def author_matches_with_source(author: Author) -> tuple[str, str, dict, str | None, str | None]:
    """Checks login, email and name separately against AUTHORS, then BOTS.
    Returns (type, sanitized_tool, name_map, matched_field, matched_text),
    with type in {"AI", "Bot", "Human"}. Whether the match needs
    corroboration is decided by the callers.
    """
    fields = (("login", author.login), ("email", author.email), ("name", author.name))

    for field_name, value in fields:
        if not value:
            continue
        m = AUTHORS.search(value)
        if m:
            return "AI", m.lastgroup or "", AUTHORS_NAMES, field_name, m.group()

    for field_name, value in fields:
        if not value:
            continue
        m = BOTS.search(value)
        if m:
            return "Bot", m.lastgroup or "", BOTS_NAMES, field_name, m.group()

    return "Human", "", {}, None, None


def author_matches(author: Author) -> tuple[str, str, dict]:
    """Backward-compatible wrapper (drops matched_field/matched_text)."""
    t, tool, names, _field, _matched_text = author_matches_with_source(author)
    return t, tool, names


def _needs_corroboration(
    field: str | None, value: str | None = None, matched_text: str | None = None
) -> bool:
    """True if a match via `field` needs corroboration: always for `name`, and
    for `login`/`email` whenever `matched_text` is only part of the value
    (ignoring a trailing "[bot]" on logins).
    """
    if field == "name":
        return True
    if field not in ("login", "email") or value is None or matched_text is None:
        return False
    core = value.strip().lower()
    mt = matched_text.strip().lower()
    if field == "login" and core.endswith("[bot]"):
        core = core[: -len("[bot]")]
    return mt != core


def _pr_has_corroborating_signal(pr: dict, tool_key: str, candidate: Author) -> bool:
    """True if another commit author/co-author of the PR matches tool_key through
    a login or email of the same identity as `candidate`. The identity check
    avoids corroboration by unrelated commits in backport PRs.
    """
    commits = (pr.get("commits") or {}).get("nodes", [])
    for commit_raw in commits:
        c = commit_raw["commit"]
        author_info = c.get("author") or {}
        user_info = author_info.get("user") or {}
        main_author = Author(
            login=user_info.get("login", ""),
            name=author_info.get("name", ""),
            email=author_info.get("email", ""),
        )
        candidates = [main_author] + extract_co_authors(c.get("message", ""))
        for cand in candidates:
            if candidate.login and cand.login and cand.login == candidate.login:
                pass
            elif candidate.email and cand.email and cand.email == candidate.email:
                pass
            else:
                continue
            t, tool, _names, field, matched_text = author_matches_with_source(cand)
            value = getattr(cand, field) if field else None
            if (
                t == "AI"
                and tool == tool_key
                and field in ("login", "email")
                and not _needs_corroboration(field, value, matched_text)
            ):
                return True
    return False


def is_ai_pr_branch(branch_name: str) -> tuple[bool, str]:
    m = BRANCHES.search(branch_name or "")
    if m:
        return True, BRANCHES_NAMES.get(m.lastgroup, m.lastgroup or "")
    return False, ""


def is_ai_pr_label(labels: list[str]) -> tuple[bool, str]:
    joined = " ".join(labels)
    m = LABELS.search(joined)
    if m:
        return True, LABELS_NAMES.get(m.lastgroup, m.lastgroup or "")
    return False, ""


def pr_closes_gfi(pr: dict) -> tuple[bool, str]:
    """Returns (closes_gfi, comma-joined issue numbers): whether an issue closed
    by the PR has a label in GFI_LABELS (exact, case-folded match).
    """
    closing_issues = (pr.get("closingIssuesReferences") or {}).get("nodes", [])
    matched_numbers = []
    for issue in closing_issues:
        labels = [l["name"].strip().lower() for l in (issue.get("labels") or {}).get("nodes", [])]
        if any(name in GFI_LABELS for name in labels):
            matched_numbers.append(str(issue.get("number", "")))
    return bool(matched_numbers), ",".join(matched_numbers)


def _parse_iso(ts: str):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _signal_date_gap_days(authored: str, committed: str) -> float | None:
    """Days between a commit's authoredDate and committedDate."""
    a, c = _parse_iso(authored), _parse_iso(committed)
    if a is None or c is None:
        return None
    return (c - a).total_seconds() / 86400.0


def resolve_pr_author_category(pr: dict) -> tuple[str, str | None]:
    """Category of the account that opened the PR (AI_Agent or Human) and the
    tool name for AI_Agent. A Bot/Organization-typed account that is not in
    bots.csv and pushes a branch matching branches.csv is an AI_Agent.
    """
    author = _pr_author(pr)
    t, tool, names, field, matched_text = author_matches_with_source(author)

    if t == "AI":
        tool_display = names.get(tool, tool)
        value = getattr(author, field) if field else None
        if not _needs_corroboration(field, value, matched_text) or _pr_has_corroborating_signal(
            pr, tool, author
        ):
            return "AI_Agent", tool_display

    if t != "Bot":
        author_info = pr.get("author") or {}
        typename = author_info.get("__typename") or ""
        if typename in ("Bot", "Organization"):
            branch_hit, branch_tool = is_ai_pr_branch(pr.get("headRefName", ""))
            if branch_hit:
                return "AI_Agent", branch_tool

    return "Human", None


def _pr_author_is_automation(pr: dict, author_category: str, detection_source: str | None) -> bool:
    """True if the PR was opened by out-of-scope automation: a bots.csv match
    (always), or a Bot/Organization-typed account, unless the account resolves
    to AI_Agent or the AI evidence is a branch match. Label and commit signals
    do not get this exemption, since an automation account can carry someone
    else's AI-assisted work.
    """
    author = _pr_author(pr)
    t, _tool, _names, _field, _matched_text = author_matches_with_source(author)
    if t == "Bot":
        return True
    if author_category == "AI_Agent":
        return False
    if detection_source == "branch":
        return False
    author_info = pr.get("author") or {}
    typename = author_info.get("__typename") or ""
    return typename in ("Bot", "Organization")


def classify_pr(pr: dict, author_category: str, author_tool: str | None) -> dict | None:
    """Returns {"ai", "source", "signal_date", "gap_days", "suspect"} for the
    first AI signal in precedence order, or None. gap_days and suspect only
    apply to commit signals.
    """
    if author_category == "AI_Agent":
        author = _pr_author(pr)
        t, tool, _names, field, matched_text = author_matches_with_source(author)
        direct = t == "AI" and (
            not _needs_corroboration(field, getattr(author, field) if field else None, matched_text)
            or _pr_has_corroborating_signal(pr, tool, author)
        )
        source = "pr-author" if direct else "branch"
        return {
            "ai": author_tool,
            "source": source,
            "signal_date": pr.get("createdAt", ""),
            "gap_days": None,
            "suspect": False,
        }

    branch_hit, branch_tool = is_ai_pr_branch(pr.get("headRefName", ""))
    if branch_hit:
        return {
            "ai": branch_tool,
            "source": "branch",
            "signal_date": pr.get("createdAt", ""),
            "gap_days": None,
            "suspect": False,
        }

    labels = [l["name"] for l in (pr.get("labels") or {}).get("nodes", [])]
    label_hit, label_tool = is_ai_pr_label(labels)
    if label_hit:
        return {
            "ai": label_tool,
            "source": "label",
            "signal_date": pr.get("createdAt", ""),
            "gap_days": None,
            "suspect": False,
        }

    commits = (pr.get("commits") or {}).get("nodes", [])
    for commit_raw in commits:
        c = commit_raw["commit"]
        author_info = c.get("author") or {}
        user_info = author_info.get("user") or {}
        main_author = Author(
            login=user_info.get("login", ""),
            name=author_info.get("name", ""),
            email=author_info.get("email", ""),
        )
        t, tool, names, field, matched_text = author_matches_with_source(main_author)
        if t == "AI":
            tool_display = names.get(tool, tool)
            value = getattr(main_author, field) if field else None
            if not _needs_corroboration(field, value, matched_text) or _pr_has_corroborating_signal(
                pr, tool, main_author
            ):
                authored = c.get("authoredDate", "")
                committed = c.get("committedDate", "")
                gap = _signal_date_gap_days(authored, committed)
                suspect = gap is not None and gap > SIGNAL_DATE_GAP_THRESHOLD_DAYS
                return {
                    "ai": tool_display,
                    "source": "commit-author",
                    "signal_date": (committed or authored) if suspect else (authored or committed),
                    "gap_days": gap,
                    "suspect": suspect,
                }

        for co_author in extract_co_authors(c.get("message", "")):
            t, tool, names, field, matched_text = author_matches_with_source(co_author)
            if t == "AI":
                tool_display = names.get(tool, tool)
                value = getattr(co_author, field) if field else None
                if not _needs_corroboration(
                    field, value, matched_text
                ) or _pr_has_corroborating_signal(pr, tool, co_author):
                    authored = c.get("authoredDate", "")
                    committed = c.get("committedDate", "")
                    gap = _signal_date_gap_days(authored, committed)
                    suspect = gap is not None and gap > SIGNAL_DATE_GAP_THRESHOLD_DAYS
                    return {
                        "ai": tool_display,
                        "source": "commit-coauthor",
                        "signal_date": (committed or authored)
                        if suspect
                        else (authored or committed),
                        "gap_days": gap,
                        "suspect": suspect,
                    }

    return None


def main():
    if not RAW_DIR.exists():
        print(f"Error: {RAW_DIR}/ does not exist. Run fetch_prs.py first.")
        sys.exit(1)

    files = sorted(RAW_DIR.glob("*.json"))
    if not files:
        print(f"Error: no JSON files in {RAW_DIR}/. Run fetch_prs.py first.")
        sys.exit(1)

    print(f"\nClassifying {len(files)} repositories offline\n")

    HEADER = [
        "AI_Tool",
        "Detection_Source",
        "Language",
        "Stars",
        "Repository",
        "PR_Number",
        "Title",
        "Link",
        "Created_At",
        "Updated_At",
        "Closed_At",
        "State",
        "PR_Author",
        "PR_Author_Category",
        "Signal_Date",
        "Signal_Date_Gap_Days",
        "Signal_Date_Suspect",
        "Closes_GFI",
        "GFI_Issue_Numbers",
    ]

    stats = defaultdict(int)
    stats_source = defaultdict(int)
    total_prs_processed = 0
    total_prs_ai = 0
    total_prs_human = 0
    total_prs_gfi = 0
    total_signal_date_suspect = 0
    total_prs_automation_dropped = 0
    total_prs_no_signal_automation = 0
    no_signal_automation_by_login = Counter()

    # PRs dropped only because the account is Bot/Organization-typed and the AI
    # signal is a commit signal, per login (diagnostics only).
    automation_drop_typename_only = 0
    automation_drop_typename_only_by_login = Counter()

    with (
        open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile,
        open(OUTPUT_FILE_HUMAN, "w", newline="", encoding="utf-8") as csvfile_human,
    ):
        writer = csv.writer(csvfile)
        writer.writerow(HEADER)

        writer_human = csv.writer(csvfile_human)
        writer_human.writerow(HEADER)

        for fpath in files:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            owner, repo = data["owner"], data["repo"]
            repo_full = f"{owner}/{repo}"
            prs = data["prs"]
            total_prs_processed += len(prs)

            found_in_repo = 0
            human_in_repo = 0
            for pr in prs:
                author_category, author_tool = resolve_pr_author_category(pr)
                result = classify_pr(pr, author_category, author_tool)
                closes_gfi, gfi_issue_numbers = pr_closes_gfi(pr)
                if closes_gfi:
                    total_prs_gfi += 1

                state = "merged" if pr.get("mergedAt") else (pr.get("state", "") or "").lower()
                closed_at = pr.get("mergedAt") or pr.get("closedAt") or ""

                row_base = [
                    data["language"],
                    data["stars"],
                    repo_full,
                    pr["number"],
                    pr.get("title", ""),
                    f"https://github.com/{repo_full}/pull/{pr['number']}",
                    pr.get("createdAt", ""),
                    pr.get("updatedAt", ""),
                    closed_at,
                    state,
                    (pr.get("author") or {}).get("login", ""),
                    author_category,
                ]

                # PRs with AI evidence opened by automation accounts are dropped.
                if result is not None and not _pr_author_is_automation(
                    pr, author_category, result["source"]
                ):
                    found_in_repo += 1
                    total_prs_ai += 1
                    stats[result["ai"]] += 1
                    stats_source[(result["ai"], result["source"])] += 1
                    if result["suspect"]:
                        total_signal_date_suspect += 1
                    gap_str = f"{result['gap_days']:.1f}" if result["gap_days"] is not None else ""
                    writer.writerow(
                        [
                            result["ai"],
                            result["source"],
                            *row_base,
                            result["signal_date"],
                            gap_str,
                            result["suspect"],
                            closes_gfi,
                            gfi_issue_numbers,
                        ]
                    )
                elif author_category == "Human" and not _pr_author_is_automation(
                    pr, author_category, None
                ):
                    human_in_repo += 1
                    total_prs_human += 1
                    writer_human.writerow(
                        [
                            "",
                            "",
                            *row_base,
                            "",
                            "",
                            False,
                            closes_gfi,
                            gfi_issue_numbers,
                        ]
                    )
                elif result is not None:
                    total_prs_automation_dropped += 1

                    pr_author_info = pr.get("author") or {}
                    pr_author_login = pr_author_info.get("login", "")
                    t_check, _tool_check, _names_check, _field_check, _matched_check = (
                        author_matches_with_source(_pr_author(pr))
                    )
                    if t_check != "Bot" and result["source"] in (
                        "commit-author",
                        "commit-coauthor",
                    ):
                        automation_drop_typename_only += 1
                        automation_drop_typename_only_by_login[pr_author_login] += 1

                    if (
                        total_prs_automation_dropped <= 50
                        or total_prs_automation_dropped % 150 == 1
                    ):
                        author = pr.get("author") or {}
                        print(
                            f"  [DROP-AUTOMATION] {repo_full}#{pr['number']} | "
                            f"author={author.get('login', '')!r} "
                            f"typename={author.get('__typename', '')!r} | "
                            f"tool={result['ai']!r} source={result['source']!r} | "
                            f"{pr.get('createdAt', '')} | "
                            f"https://github.com/{repo_full}/pull/{pr['number']}",
                            flush=True,
                        )
                else:
                    # No AI signal and the account is automation (e.g. plain Dependabot
                    # PRs).
                    total_prs_no_signal_automation += 1
                    author = pr.get("author") or {}
                    no_signal_automation_by_login[author.get("login", "")] += 1
                    if (
                        total_prs_no_signal_automation <= 50
                        or total_prs_no_signal_automation % 5000 == 1
                    ):
                        print(
                            f"  [NO-SIGNAL-AUTOMATION] {repo_full}#{pr['number']} | "
                            f"author={author.get('login', '')!r} "
                            f"typename={author.get('__typename', '')!r} | "
                            f"{pr.get('createdAt', '')}",
                            flush=True,
                        )

            if DEBUG:
                print(
                    f"  [{repo_full}] {len(prs)} PRs processed, {found_in_repo} with AI and {human_in_repo} human control",
                    flush=True,
                )

    print(f"\n{'═' * 62}")
    print("  SUMMARY PER TOOL")
    print(f"{'═' * 62}")
    for ai, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {ai:<24} {count:>6,}")

    print(f"\n{'─' * 62}")
    print("  SUMMARY PER DETECTION SOURCE")
    print(f"{'─' * 62}")
    for (ai, source), count in sorted(stats_source.items(), key=lambda x: -x[1]):
        print(f"    {ai:<20} [{source:<20}] {count:>6,}")

    print(f"\n{'═' * 62}")
    print(
        f"  Total processado: {total_prs_processed:,} | IA: {total_prs_ai:,} | "
        f"Human: {total_prs_human:,} | Closes GFI: {total_prs_gfi:,}"
    )
    print(
        f"  Suspect Signal_Date (gap > {SIGNAL_DATE_GAP_THRESHOLD_DAYS}d): {total_signal_date_suspect:,}"
    )
    print(f"  Automation with AI signal dropped: {total_prs_automation_dropped:,}")
    print(
        f"  Automation (typename only, commit-author/coauthor): {automation_drop_typename_only:,}"
    )
    print(f"\n{'─' * 62}")
    print("  TOP ACCOUNTS: typename-only drop")
    print(f"{'─' * 62}")
    for login, count in automation_drop_typename_only_by_login.most_common(20):
        print(f"    {login:<30} {count:>6,}")

    print(f"  Automation without AI signal: {total_prs_no_signal_automation:,}")
    assert total_prs_processed == (
        total_prs_ai
        + total_prs_human
        + total_prs_automation_dropped
        + total_prs_no_signal_automation
    ), (
        f"Counts do not add up: processed={total_prs_processed:,} vs sum={total_prs_ai + total_prs_human + total_prs_automation_dropped + total_prs_no_signal_automation:,}"
    )

    print(f"\n{'─' * 62}")
    print("  TOP ACCOUNTS: automation without AI signal")
    print(f"{'─' * 62}")
    for login, count in no_signal_automation_by_login.most_common(20):
        print(f"    {login:<30} {count:>6,}")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()
