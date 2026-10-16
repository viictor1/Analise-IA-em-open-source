"""Raw PR collection through the GitHub GraphQL API.

One paginated query per repository pulls every PR (any state) with its
commits, author, labels and closing issues (with their labels). Any
sub-connection truncated on the first page is completed with follow-up
queries scoped to that PR or issue.

Each completed page is checkpointed to <owner>__<repo>.partial.jsonl, so a
repository that fails midway resumes from the last good cursor. The
checkpoint is removed once <owner>__<repo>.json is written.

Some repositories list PRs that GitHub can no longer load (removed PRs), and
any page of the pullRequests connection that contains one fails. When the
connection cannot be paged even at PAGE_SIZE_FLOOR, the repository switches
to the search API in created-date windows ("search_fallback_from" in the
JSON). If a search page also fails at the floor, the search goes item by item
and skips the failing item ("search_skipped" in the JSON).

Usage:
    python fetch_prs.py
"""

import os
import sys
import json
import time
import base64
import random
import csv
import threading
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

# Configuration
load_dotenv()
raw_tokens = os.getenv("GITHUB_TOKENS", "")
GITHUB_TOKENS = [t.strip() for t in raw_tokens.split(",") if t.strip()]

if not GITHUB_TOKENS and os.getenv("GITHUB_TOKEN"):
    GITHUB_TOKENS = [os.getenv("GITHUB_TOKEN").strip()]

PROJECTS_FILE = "data/selected_repos.csv"
RAW_DIR = Path("data/raw_prs")  # 1 JSON file per repo
STATE_FILE = Path("data/fetch_state.json")  # repos already collected

MAX_REPOS = None  # e.g. 10 to test on a small slice; None = all
START_DATE = "2022-01-01"  # PRs created before this are out of scope
MAX_PAGES_PER_REPO = 20000  # emergency valve, should not fire on a healthy repo

# One worker per token (each token has its own rate-limit budget), capped low
# because concurrent queries on very large repositories trigger GitHub's
# execution timeout.
MAX_WORKERS = 6

PAGE_SIZE = 100  # PRs per page; shrinks per-repo via dynamic backoff
PAGE_SIZE_FLOOR = 10
# Embedded connection sizes of the main query. Lower values reduce the query
# cost; anything truncated is completed by the follow-up queries.
COMMITS_PER_PR = 15
LABELS_PER_PR = 10
CLOSING_ISSUES_PER_PR = 3
LABELS_PER_ISSUE = 10
MAX_RETRIES = 10
# Query-cost failures (502/504 or GitHub's execution-timeout error) shrink the
# page size after a few attempts instead of using all MAX_RETRIES.
GATEWAY_RETRY_LIMIT = 3
RATE_LIMIT_FLOOR = 80  # below this, wait for reset instead of risking a 403

# Cap on the exponential backoff, so a failing page shrinks quickly.
MAX_BACKOFF_SECS = 30

# If the main query keeps failing at PAGE_SIZE_FLOOR, retry this many times and
# then switch the repository to the search fallback.
FLOOR_FAILURE_LIMIT = 2
FLOOR_FAILURE_WAIT_SECS = 20

# Search fallback. Search returns at most 1000 results per query, so the
# created-date window slides down to the oldest PR seen every 1000 results.
SEARCH_PAGE_SIZE = 50
SEARCH_MAX_RESULTS = 1000
SEARCH_FLOOR_FAILURE_LIMIT = 5
SEARCH_FLOOR_FAILURE_WAIT_SECS = 60
# Item-by-item crawl (pageSize=1) used when a search page fails at the floor:
# failures tolerated on one item before skipping it, and how many good items
# before going back to PAGE_SIZE_FLOOR.
SEARCH_CRAWL_FAILURE_LIMIT = 2
SEARCH_CRAWL_RESTORE_AFTER = 25

PAGE_SIZE_SUBCONN = 100  # page size for completing a truncated sub-connection
MAX_SUBPAGES_PER_PR = 50  # safety valve for that sub-pagination

DEBUG = True

GRAPHQL_URL = "https://api.github.com/graphql"

# Queries
PR_FIELDS = """
        number
        title
        bodyText
        headRefName
        state
        createdAt
        updatedAt
        closedAt
        mergedAt
        author { __typename login ... on User { name } }
        labels(first: %(labels_per_pr)d) {
          pageInfo { endCursor hasNextPage }
          nodes { name }
        }
        commits(first: %(commits_per_pr)d) {
          pageInfo { endCursor hasNextPage }
          nodes {
            commit {
              oid
              message
              authoredDate
              committedDate
              author { name email user { login } }
            }
          }
        }
        closingIssuesReferences(first: %(closing_issues_per_pr)d) {
          pageInfo { endCursor hasNextPage }
          nodes {
            number
            title
            createdAt
            labels(first: %(labels_per_issue)d) {
              pageInfo { endCursor hasNextPage }
              nodes { name }
            }
          }
        }
""" % {
    "labels_per_pr": LABELS_PER_PR,
    "commits_per_pr": COMMITS_PER_PR,
    "closing_issues_per_pr": CLOSING_ISSUES_PER_PR,
    "labels_per_issue": LABELS_PER_ISSUE,
}

QUERY = """
query($owner: String!, $name: String!, $pageSize: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(
      first: $pageSize
      after: $cursor
      orderBy: { field: CREATED_AT, direction: DESC }
    ) {
      totalCount
      pageInfo { endCursor hasNextPage }
      nodes {%(pr_fields)s}
    }
  }
  rateLimit { cost remaining resetAt }
}
""" % {"pr_fields": PR_FIELDS}

# Fallback: same PR fields through the search API (see _fetch_via_search).
SEARCH_QUERY = """
query($q: String!, $pageSize: Int!, $cursor: String) {
  search(query: $q, type: ISSUE, first: $pageSize, after: $cursor) {
    issueCount
    pageInfo { endCursor hasNextPage }
    nodes {
      ... on PullRequest {%(pr_fields)s}
    }
  }
  rateLimit { cost remaining resetAt }
}
""" % {"pr_fields": PR_FIELDS}

# Probe of a single search item, used to record the number of an item that the
# full query cannot load.
LIGHT_SEARCH = """
query($q: String!, $cursor: String) {
  search(query: $q, type: ISSUE, first: 1, after: $cursor) {
    nodes { ... on PullRequest { number createdAt } }
  }
}
"""

# Follow-up queries, scoped to one PR or issue by number.
PR_COMMITS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $pageSize: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      commits(first: $pageSize, after: $cursor) {
        pageInfo { endCursor hasNextPage }
        nodes {
          commit {
            oid
            message
            authoredDate
            committedDate
            author { name email user { login } }
          }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
"""

PR_LABELS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $pageSize: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      labels(first: $pageSize, after: $cursor) {
        pageInfo { endCursor hasNextPage }
        nodes { name }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
"""

PR_CLOSING_ISSUES_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $pageSize: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      closingIssuesReferences(first: $pageSize, after: $cursor) {
        pageInfo { endCursor hasNextPage }
        nodes {
          number
          title
          createdAt
          labels(first: %(labels_per_issue)d) {
            pageInfo { endCursor hasNextPage }
            nodes { name }
          }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
""" % {"labels_per_issue": LABELS_PER_ISSUE}

ISSUE_LABELS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $pageSize: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      labels(first: $pageSize, after: $cursor) {
        pageInfo { endCursor hasNextPage }
        nodes { name }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
"""


# Helpers
def _backoff(attempt: int) -> float:
    return min(2**attempt, MAX_BACKOFF_SECS) + random.uniform(0, 1)


def _load_projects(filename: str) -> list[dict]:
    projects = []
    with open(filename, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row["stars"]
            try:
                stars = int(float(raw.replace("k", "")) * (1000 if "k" in raw else 1))
            except Exception:
                stars = 0
            projects.append(
                {
                    "owner": row["owner"],
                    "repo": row["project"],
                    "language": row["main language"],
                    "stars": stars,
                }
            )
    return projects


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"done": []}


_state_lock = threading.Lock()


def _save_state(state: dict) -> None:
    with _state_lock:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _repo_out_path(owner: str, repo: str) -> Path:
    safe = f"{owner}__{repo}".replace("/", "_")
    return RAW_DIR / f"{safe}.json"


def _partial_path(owner: str, repo: str) -> Path:
    return _repo_out_path(owner, repo).with_suffix(".partial.jsonl")


def _load_checkpoint(path: Path) -> dict:
    """Reads the checkpoint JSONL into a state dict. A truncated last line is
    discarded and that page is fetched again.
    """
    st = {
        "prs": [],
        "cursor": None,
        "page": 0,
        "page_size": PAGE_SIZE,
        "finished": False,
        "mode": "graphql",
        "hi": None,
        "fetched": 0,
        "window_new": 0,
        "last_created": None,
        "fallback_from": None,
        "skipped": [],
    }
    if not path.exists():
        return st

    good_bytes = 0
    with path.open("rb") as f:
        for raw in f:
            if not raw.endswith(b"\n"):
                break
            try:
                rec = json.loads(raw)
            except ValueError:
                break
            good_bytes += len(raw)
            st["prs"].extend(rec["prs"])
            st["cursor"], st["page"] = rec["next_cursor"], rec["page"]
            st["page_size"], st["finished"] = rec["page_size"], rec["finished"]
            st["mode"] = rec.get("mode", "graphql")
            for k in ("hi", "fetched", "window_new", "last_created", "fallback_from", "skipped"):
                if k in rec:
                    st[k] = rec[k]

    if good_bytes < path.stat().st_size:
        with path.open("r+b") as f:
            f.truncate(good_bytes)
    return st


def _append_checkpoint(
    path: Path, page: int, prs: list, next_cursor, page_size: int, finished: bool, **extra
) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "page": page,
                    "prs": prs,
                    "next_cursor": next_cursor,
                    "page_size": page_size,
                    "finished": finished,
                    **extra,
                }
            )
            + "\n"
        )


def _cursor_offset(cursor) -> int:
    """GitHub's search cursor is base64('cursor:<offset>'). None = offset 0."""
    if not cursor:
        return 0
    return int(base64.b64decode(cursor).decode().split(":")[1])


def _make_cursor(offset: int) -> str:
    return base64.b64encode(f"cursor:{offset}".encode()).decode()


# Each worker thread is pinned to one token, so the budgets of several tokens
# are spent in parallel.
_thread_local = threading.local()
_token_assign_counter = itertools.count()


class RateBudget:
    """Per-token rate-limit budget (remaining/reset_at). A thread switches token
    only when its own token runs low.
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.state = {t: {"remaining": None, "reset_at": None} for t in tokens}
        self._lock = threading.Lock()

    def _my_idx(self) -> int:
        if not hasattr(_thread_local, "idx"):
            _thread_local.idx = next(_token_assign_counter) % len(self.tokens)
        return _thread_local.idx

    def get_token(self) -> str:
        return self.tokens[self._my_idx()]

    def update(self, token: str, rate_limit: dict) -> None:
        with self._lock:
            entry = self.state.get(token)
            if entry is None:
                return
            entry["remaining"] = rate_limit.get("remaining")
            reset_str = rate_limit.get("resetAt")
            if reset_str:
                entry["reset_at"] = datetime.fromisoformat(reset_str.replace("Z", "+00:00"))

    def mark_exhausted(self, token: str) -> None:
        """Marks the token as limited when the response has no rateLimit block."""
        with self._lock:
            entry = self.state.get(token)
            if entry is not None:
                entry["remaining"] = 0

    def _find_healthy_locked(self, exclude_idx: int):
        """Index of another token that looks healthy, or None. Call with self._lock held."""
        for offset in range(1, len(self.tokens)):
            cand_idx = (exclude_idx + offset) % len(self.tokens)
            entry = self.state[self.tokens[cand_idx]]
            if entry["remaining"] is None or entry["remaining"] >= RATE_LIMIT_FLOOR:
                return cand_idx
        return None

    def switch_to_healthy_or_wait(self) -> None:
        """Moves the calling thread to a healthy token, or sleeps until the earliest
        reset when every token is low.
        """
        idx = self._my_idx()
        with self._lock:
            healthy_idx = self._find_healthy_locked(idx) if len(self.tokens) > 1 else None
            if healthy_idx is not None:
                _thread_local.idx = healthy_idx
                if DEBUG:
                    print(
                        f"  thread switching to token {healthy_idx + 1}/{len(self.tokens)}",
                        flush=True,
                    )
                return
            dated = [e["reset_at"] for e in self.state.values() if e["reset_at"]]
        wait_secs = 60.0
        if dated:
            wait_secs = max((min(dated) - datetime.now(timezone.utc)).total_seconds(), 0) + 5
        if DEBUG:
            print(f"  all tokens low, waiting {wait_secs:.0f}s for nearest reset", flush=True)
        time.sleep(wait_secs)

    def wait_or_rotate(self) -> None:
        """Called between requests: leaves the pinned token if it is low."""
        token = self.tokens[self._my_idx()]
        with self._lock:
            entry = self.state[token]
            remaining, reset_at = entry["remaining"], entry["reset_at"]
        if remaining is None or reset_at is None or remaining >= RATE_LIMIT_FLOOR:
            return
        self.switch_to_healthy_or_wait()


_budget = RateBudget(GITHUB_TOKENS)

# Bounds the number of concurrent HTTP calls across all threads, since
# GitHub's secondary rate limit reacts to overall request volume.
MAX_CONCURRENT_REQUESTS = max(10, 4 * len(GITHUB_TOKENS)) if GITHUB_TOKENS else 10
_inflight = threading.Semaphore(MAX_CONCURRENT_REQUESTS)


class RepoNotFound(Exception):
    pass


class GatewayOverloaded(RuntimeError):
    """Raised when a query keeps failing with 502/504 or the execution-timeout
    error, i.e. it is too expensive. Callers shrink pageSize.
    """

    pass


def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


def _capture_header_rate_limit(token: str, resp: requests.Response) -> None:
    """Reads the X-RateLimit-* headers, which are present even when the query fails."""
    remaining_h = resp.headers.get("X-RateLimit-Remaining")
    reset_h = resp.headers.get("X-RateLimit-Reset")
    if remaining_h is None or reset_h is None:
        return
    try:
        _budget.update(
            token,
            {
                "remaining": int(remaining_h),
                "resetAt": datetime.fromtimestamp(int(reset_h), tz=timezone.utc).isoformat(),
            },
        )
    except (TypeError, ValueError):
        pass


def _run_query(session: requests.Session, query: str, variables: dict) -> dict:
    last_err = None
    gateway_timeout = False
    gateway_failures = 0  # consecutive 502/504 or exec-timeout failures

    # Describes the failing query in the final error message.
    if "q" in variables:  # search fallback query
        tag = f"search[{variables['q']}] cursor={variables.get('cursor')}"
    else:
        tag = f"{variables.get('owner')}/{variables.get('name')}" + (
            f"#{variables['number']}"
            if "number" in variables
            else f" cursor={variables.get('cursor')}"
        )

    for attempt in range(1, MAX_RETRIES + 1):
        current_token = _budget.get_token()
        headers = {"Authorization": f"bearer {current_token}"}

        try:
            with _inflight:
                resp = session.post(
                    GRAPHQL_URL,
                    json={"query": query, "variables": variables},
                    headers=headers,
                    timeout=30,
                )
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            last_err = e
            gateway_timeout = False
            backoff = _backoff(attempt)
            if DEBUG:
                print(
                    f"    network error (attempt {attempt}): {e}, retrying in {backoff:.1f}s",
                    flush=True,
                )
            time.sleep(backoff)
            continue

        _capture_header_rate_limit(current_token, resp)

        if resp.status_code in (502, 503, 504):
            last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            gateway_timeout = True
            gateway_failures += 1
            if gateway_failures >= GATEWAY_RETRY_LIMIT:
                if DEBUG:
                    print(
                        f"    HTTP {resp.status_code}: {gateway_failures} query-cost failures in a row, "
                        f"giving up early (attempt {attempt}/{MAX_RETRIES}) so page_size can shrink",
                        flush=True,
                    )
                break
            backoff = _backoff(attempt)
            if DEBUG:
                print(
                    f"    HTTP {resp.status_code} (attempt {attempt}), retrying in {backoff:.1f}s",
                    flush=True,
                )
            time.sleep(backoff)
            continue

        if resp.status_code == 403:
            # HTTP 403 is the secondary rate limit; switching tokens does not help, so
            # back off and respect Retry-After.
            last_err = RuntimeError(f"HTTP 403 (secondary rate limit): {resp.text[:300]}")
            gateway_timeout = False
            retry_after = int(resp.headers.get("Retry-After", 30))
            if DEBUG:
                print(
                    f"    403 secondary rate limit, backing off {retry_after}s (not switching token)",
                    flush=True,
                )
            time.sleep(retry_after)
            continue

        if resp.status_code != 200:
            raise RuntimeError(f"[{tag}] GraphQL HTTP {resp.status_code}: {resp.text[:300]}")

        try:
            payload = resp.json()
        except ValueError as e:
            last_err = RuntimeError(f"invalid JSON body on HTTP 200: {e}")
            gateway_timeout = False
            backoff = _backoff(attempt)
            if DEBUG:
                print(
                    f"    invalid JSON response (attempt {attempt}): {e}, retrying in {backoff:.1f}s",
                    flush=True,
                )
            time.sleep(backoff)
            continue

        if "errors" in payload:
            errs = payload["errors"]
            msgs = [e.get("message", "") for e in errs]

            if any("rate limit" in m.lower() and "exceeded" in m.lower() for m in msgs):
                is_secondary = any("secondary" in m.lower() for m in msgs)
                if is_secondary:
                    # Secondary rate limit: back off instead of switching tokens.
                    last_err = RuntimeError(f"secondary rate limit: {msgs}")
                    gateway_timeout = False
                    backoff = 30 + random.uniform(0, 15)
                    if DEBUG:
                        print(
                            f"    secondary rate limit hit, backing off {backoff:.0f}s (not switching token)",
                            flush=True,
                        )
                    time.sleep(backoff)
                    continue

                # Primary per-token budget exhausted: move to a healthier token.
                _budget.mark_exhausted(current_token)
                last_err = RuntimeError(f"rate limit exceeded: {msgs}")
                gateway_timeout = False
                if DEBUG:
                    print("    rate limit exceeded on this token, switching/waiting", flush=True)
                _budget.switch_to_healthy_or_wait()
                continue

            if any("Could not resolve to a Repository" in m for m in msgs):
                raise RepoNotFound(msgs)

            # "Not Found" / "Could not resolve to an Issue": the PR or issue of this
            # follow-up query no longer exists. The field comes back null, which the
            # extractors treat as empty, so accept it instead of retrying.
            if any("not found" in m.lower() or "could not resolve to" in m.lower() for m in msgs):
                if DEBUG:
                    print(
                        f"    field not found (deleted/private/cross-repo PR or issue?), accepting null: {msgs[:2]}",
                        flush=True,
                    )
                data = payload.get("data") or {}
                rl = data.get("rateLimit")
                if rl:
                    _budget.update(current_token, rl)
                return data

            # Any other GraphQL error may come with partially null data, so retry. The
            # execution-timeout error is treated like a 502/504.
            is_transient_exec_error = any(
                "something went wrong while executing your query" in m.lower() for m in msgs
            )
            last_err = RuntimeError(f"GraphQL errors: {msgs}")
            gateway_timeout = is_transient_exec_error
            if is_transient_exec_error:
                gateway_failures += 1
                if gateway_failures >= GATEWAY_RETRY_LIMIT:
                    if DEBUG:
                        print(
                            f"    {gateway_failures} query-cost failures in a row, giving up early "
                            f"(attempt {attempt}/{MAX_RETRIES}) so page_size can shrink",
                            flush=True,
                        )
                    break
            backoff = _backoff(attempt)
            if DEBUG:
                kind = (
                    "transient query execution error"
                    if is_transient_exec_error
                    else "GraphQL error"
                )
                print(
                    f"    {kind} (attempt {attempt}): {msgs[:2]}, retrying in {backoff:.1f}s",
                    flush=True,
                )
            time.sleep(backoff)
            continue

        data = payload["data"]
        # Main query only: pullRequests came back null without errors; retry.
        if (
            query is QUERY
            and data.get("repository") is not None
            and data["repository"].get("pullRequests") is None
        ):
            last_err = RuntimeError("repository.pullRequests came back null with no GraphQL errors")
            gateway_timeout = False
            backoff = _backoff(attempt)
            if DEBUG:
                print(
                    f"    null pullRequests with no errors (attempt {attempt}), retrying in {backoff:.1f}s",
                    flush=True,
                )
            time.sleep(backoff)
            continue
        rl = data.get("rateLimit")
        if rl:
            _budget.update(current_token, rl)
        return data

    if gateway_timeout:
        raise GatewayOverloaded(f"[{tag}] Failed after {attempt} attempts: {last_err}")
    raise RuntimeError(f"[{tag}] Failed after {attempt} attempts: {last_err}")


def _fetch_remaining(
    session: requests.Session, query: str, variables_base: dict, cursor: str, extractor, label: str
) -> tuple[list[dict], bool]:
    """Pages a sub-connection (PR commits/labels/closing issues, or issue labels)
    from `cursor`. extractor(data) returns the {nodes, pageInfo} connection.
    Returns (extra_nodes, truncated). On repeated query-cost failures the page
    size is halved; if it still fails at the floor, returns what it has.
    """
    extra_nodes = []
    pages = 0
    page_size = variables_base["pageSize"]
    while True:
        pages += 1
        variables = {**variables_base, "pageSize": page_size, "cursor": cursor}
        try:
            data = _run_query(session, query, variables)
        except GatewayOverloaded:
            if page_size <= PAGE_SIZE_FLOOR:
                print(
                    f"    {label}: gave up at pageSize={page_size}, "
                    f"{len(extra_nodes)} extra items kept, flagged truncated",
                    flush=True,
                )
                return extra_nodes, True
            page_size = max(PAGE_SIZE_FLOOR, page_size // 2)
            pages -= 1
            if DEBUG:
                print(
                    f"    {label}: query too expensive, shrinking sub-pageSize to {page_size}",
                    flush=True,
                )
            continue

        conn = extractor(data)
        if conn is None:
            break
        extra_nodes.extend(conn["nodes"])

        if not conn["pageInfo"]["hasNextPage"]:
            return extra_nodes, False

        cursor = conn["pageInfo"]["endCursor"]
        if pages >= MAX_SUBPAGES_PER_PR:
            print(
                f"    {label}: hit sub-pagination safety valve "
                f"({pages} extra pages, {len(extra_nodes)} extra items), flagged truncated",
                flush=True,
            )
            return extra_nodes, True

        _budget.wait_or_rotate()

    return extra_nodes, False


def _complete_connection(
    session: requests.Session,
    node: dict,
    field: str,
    query: str,
    variables_base: dict,
    extractor,
    label: str,
) -> None:
    """If node[field] came back truncated, pages through the rest and merges
    it in-place. Sets node[f'{field}_truncated'] either way."""
    conn = node.get(field) or {}
    page_info = conn.get("pageInfo") or {}
    truncated_key = f"{field}_truncated"

    if not page_info.get("hasNextPage"):
        node[truncated_key] = False
        return

    extra, truncated = _fetch_remaining(
        session,
        query,
        variables_base,
        page_info.get("endCursor"),
        extractor,
        label,
    )
    conn.setdefault("nodes", []).extend(extra)
    node[truncated_key] = truncated

    if DEBUG:
        print(
            f"    {label}: +{len(extra)} extra via pagination "
            f"(total {len(conn['nodes'])})" + (" — TRUNCATED" if truncated else ""),
            flush=True,
        )


def _complete_pr_subconnections(
    session: requests.Session, owner: str, repo: str, node: dict
) -> None:
    """Completes the truncated sub-connections of one PR (and the labels of its
    closing issues), fetching them concurrently.
    """
    base_pr = {
        "owner": owner,
        "name": repo,
        "number": node["number"],
        "pageSize": PAGE_SIZE_SUBCONN,
    }

    pr_jobs = [
        (
            "commits",
            PR_COMMITS_QUERY,
            (lambda data: ((data.get("repository") or {}).get("pullRequest") or {}).get("commits")),
            f"PR #{node['number']} commits",
        ),
        (
            "labels",
            PR_LABELS_QUERY,
            (lambda data: ((data.get("repository") or {}).get("pullRequest") or {}).get("labels")),
            f"PR #{node['number']} labels",
        ),
        (
            "closingIssuesReferences",
            PR_CLOSING_ISSUES_QUERY,
            (
                lambda data: ((data.get("repository") or {}).get("pullRequest") or {}).get(
                    "closingIssuesReferences"
                )
            ),
            f"PR #{node['number']} closing issues",
        ),
    ]

    with ThreadPoolExecutor(max_workers=len(pr_jobs)) as pool:
        futures = [
            pool.submit(
                _complete_connection, session, node, field, query, base_pr, extractor, label
            )
            for field, query, extractor, label in pr_jobs
        ]
        for f in as_completed(futures):
            f.result()  # re-raise any exception from the three fetches

    issues = (node.get("closingIssuesReferences") or {}).get("nodes", [])
    if not issues:
        return

    def _complete_issue_labels(issue: dict) -> None:
        base_issue = {
            "owner": owner,
            "name": repo,
            "number": issue["number"],
            "pageSize": PAGE_SIZE_SUBCONN,
        }
        _complete_connection(
            session,
            issue,
            "labels",
            ISSUE_LABELS_QUERY,
            base_issue,
            extractor=lambda data: ((data.get("repository") or {}).get("issue") or {}).get(
                "labels"
            ),
            label=f"issue #{issue['number']} labels (closed by PR #{node['number']})",
        )

    with ThreadPoolExecutor(max_workers=len(issues)) as pool:
        futures = [pool.submit(_complete_issue_labels, issue) for issue in issues]
        for f in as_completed(futures):
            f.result()


def _search_q(owner: str, repo: str, hi) -> str:
    lo = f"{START_DATE}T00:00:00Z"
    span = f"{lo}..{hi}" if hi else f">={lo}"
    return f"repo:{owner}/{repo} is:pr created:{span} sort:created-desc"


def _fetch_via_search(
    session: requests.Session,
    owner: str,
    repo: str,
    ckpt: Path,
    all_prs: list,
    page: int,
    page_size: int,
    hi,
    cursor,
    fetched: int,
    window_new: int,
    last_created,
    fallback_from,
    skipped: list,
):
    """Fallback for repositories whose pullRequests connection cannot be paged.
    Walks the PRs newest to oldest through the search API in created-date
    windows; when a window is full it restarts just below the oldest
    createdAt seen. Removed PRs are usually not indexed, so they are absent.
    If a search page fails at PAGE_SIZE_FLOOR, crawls item by item and skips
    the failing item. Returns (prs, truncated, info).
    """
    seen = {p["number"] for p in all_prs}
    floor_failures = 0
    crawl_ok = 0
    truncated = False

    while True:
        _budget.wait_or_rotate()

        if fetched + page_size > SEARCH_MAX_RESULTS:
            if not last_created:
                raise RuntimeError(
                    f"[{owner}/{repo}] search window full but no PR seen to slide from"
                )
            new_hi = last_created
            if window_new == 0:  # whole window already collected: force progress
                dt = datetime.fromisoformat(last_created.replace("Z", "+00:00")) - timedelta(
                    seconds=1
                )
                new_hi = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            hi, cursor, fetched, window_new = new_hi, None, 0, 0
            if DEBUG:
                print(
                    f"    {owner}/{repo}: search window full, restarting at created<={hi}",
                    flush=True,
                )

        variables = {"q": _search_q(owner, repo, hi), "pageSize": page_size, "cursor": cursor}
        try:
            data = _run_query(session, SEARCH_QUERY, variables)
        except GatewayOverloaded:
            if page_size <= PAGE_SIZE_FLOOR:
                floor_failures += 1
                limit = SEARCH_CRAWL_FAILURE_LIMIT if page_size == 1 else SEARCH_FLOOR_FAILURE_LIMIT
                if floor_failures > limit:
                    if page_size > 1:
                        # The page is stuck: crawl item by item to find the one that fails.
                        page_size, floor_failures, crawl_ok = 1, 0, 0
                        print(
                            f"    {owner}/{repo}: search stuck at the floor, crawling with pageSize=1",
                            flush=True,
                        )
                        continue
                    off = _cursor_offset(cursor)
                    num = None
                    try:
                        lite = _run_query(
                            session, LIGHT_SEARCH, {"q": variables["q"], "cursor": cursor}
                        )
                        first = ((lite.get("search") or {}).get("nodes") or [None])[0]
                        num = first.get("number") if first else None
                    except Exception:
                        pass
                    skipped.append({"offset": off, "number": num, "window_hi": hi})
                    print(
                        f"    {owner}/{repo}: window item {off + 1} skipped (PR #{num})", flush=True
                    )
                    cursor = _make_cursor(off + 1)
                    fetched += 1
                    floor_failures, crawl_ok = 0, 0
                    continue
                print(
                    f"    {owner}/{repo}: search failed at the floor, waiting "
                    f"{1 if page_size == 1 else SEARCH_FLOOR_FAILURE_WAIT_SECS}s "
                    f"({floor_failures}/{limit})",
                    flush=True,
                )
                time.sleep(1 if page_size == 1 else SEARCH_FLOOR_FAILURE_WAIT_SECS)
                continue
            page_size = max(PAGE_SIZE_FLOOR, page_size // 2)
            if DEBUG:
                print(
                    f"    {owner}/{repo}: search too expensive, pageSize down to {page_size}",
                    flush=True,
                )
            continue
        floor_failures = 0

        if page_size == 1:
            crawl_ok += 1
            if crawl_ok >= SEARCH_CRAWL_RESTORE_AFTER:
                page_size = PAGE_SIZE_FLOOR  # long clean stretch: leave the crawl
                if DEBUG:
                    print(
                        f"    {owner}/{repo}: {crawl_ok} items ok in a row, back to pageSize={page_size}",
                        flush=True,
                    )

        result = data.get("search")
        if result is None:
            raise RuntimeError(f"[{owner}/{repo}] search came back null")

        raw_nodes = result["nodes"]
        prs_here = [n for n in raw_nodes if n and "number" in n]
        fetched += len(raw_nodes)
        if prs_here:
            last_created = prs_here[-1]["createdAt"]  # DESC: last node is the oldest

        kept = []
        for node in prs_here:
            if node["number"] in seen:
                continue
            seen.add(node["number"])
            _complete_pr_subconnections(session, owner, repo, node)
            kept.append(node)
        window_new += len(kept)
        all_prs.extend(kept)
        page += 1

        end_cursor = result["pageInfo"]["endCursor"]
        has_next = result["pageInfo"]["hasNextPage"]
        # Saturated window: search stops at 1000 results with hasNextPage=False, but
        # issueCount shows older PRs in the same window.
        saturated = (not has_next) and result["issueCount"] > fetched
        if saturated:
            fetched = SEARCH_MAX_RESULTS  # forces the window-slide block on the next loop
        finished = (not has_next) and not saturated
        _append_checkpoint(
            ckpt,
            page,
            kept,
            end_cursor,
            page_size,
            finished,
            mode="search",
            hi=hi,
            fetched=fetched,
            window_new=window_new,
            last_created=last_created,
            fallback_from=fallback_from,
            skipped=skipped,
        )

        if DEBUG:
            rl = data.get("rateLimit")
            print(
                f"    search page {page}: +{len(kept)} PRs (of {len(raw_nodes)} on page, "
                f"window issueCount={result['issueCount']}"
                + (f", cost={rl['cost']}, remaining={rl['remaining']})" if rl else ")"),
                flush=True,
            )

        if finished:
            break

        if page >= MAX_PAGES_PER_REPO:
            truncated = True
            print(
                f"    {owner}/{repo}: hit emergency valve of {MAX_PAGES_PER_REPO} pages "
                f"({len(all_prs)} PRs collected) — DATA IS INCOMPLETE, investigate this repo manually",
                flush=True,
            )
            break

        cursor = end_cursor

    return all_prs, truncated, {"search_fallback_from": fallback_from, "search_skipped": skipped}


def fetch_repo_prs(
    session: requests.Session, owner: str, repo: str
) -> tuple[list[dict], bool, dict]:
    """Pages a repository's PRs newest to oldest until START_DATE, with a
    checkpoint per page. pageSize is halved on repeated query-cost failures;
    if the query still fails FLOOR_FAILURE_LIMIT times at the floor, the
    repository continues through _fetch_via_search.
    Returns (prs, truncated, info).
    """
    ckpt = _partial_path(owner, repo)
    st = _load_checkpoint(ckpt)
    all_prs, cursor, page, page_size = st["prs"], st["cursor"], st["page"], st["page_size"]
    if st["finished"]:
        return (
            all_prs,
            False,
            {"search_fallback_from": st["fallback_from"], "search_skipped": st["skipped"]},
        )
    if page and DEBUG:
        print(
            f"    {owner}/{repo}: resuming from checkpoint "
            f"({len(all_prs)} PRs, {page} pages, pageSize={page_size}, mode={st['mode']})",
            flush=True,
        )
    if st["mode"] == "search":
        return _fetch_via_search(
            session,
            owner,
            repo,
            ckpt,
            all_prs,
            page,
            page_size,
            st["hi"],
            cursor,
            st["fetched"],
            st["window_new"],
            st["last_created"],
            st["fallback_from"],
            st["skipped"],
        )

    truncated = False
    oldest_seen = None
    floor_failures = 0
    cutoff = datetime.fromisoformat(START_DATE).replace(tzinfo=timezone.utc)

    while True:
        _budget.wait_or_rotate()
        page += 1
        variables = {"owner": owner, "name": repo, "pageSize": page_size, "cursor": cursor}
        try:
            data = _run_query(session, QUERY, variables)
        except GatewayOverloaded:
            page -= 1
            if page_size <= PAGE_SIZE_FLOOR:
                floor_failures += 1
                if floor_failures > FLOOR_FAILURE_LIMIT:
                    fallback_from = all_prs[-1]["createdAt"] if all_prs else None
                    print(
                        f"    {owner}/{repo}: PR connection cannot be paged even at the floor (probably a removed PR "
                        f"still listed), switching to search by date windows from "
                        f"{fallback_from or 'the start'}",
                        flush=True,
                    )
                    return _fetch_via_search(
                        session,
                        owner,
                        repo,
                        ckpt,
                        all_prs,
                        page,
                        SEARCH_PAGE_SIZE,
                        fallback_from,
                        None,
                        0,
                        0,
                        fallback_from,
                        fallback_from,
                        [],
                    )
                print(
                    f"    {owner}/{repo}: failed at the floor, waiting {FLOOR_FAILURE_WAIT_SECS}s "
                    f"({floor_failures}/{FLOOR_FAILURE_LIMIT})",
                    flush=True,
                )
                time.sleep(FLOOR_FAILURE_WAIT_SECS)
                continue
            page_size = max(PAGE_SIZE_FLOOR, page_size // 2)
            if DEBUG:
                print(
                    f"    {owner}/{repo}: 502/504 exhausted retries, shrinking pageSize to {page_size}",
                    flush=True,
                )
            continue
        floor_failures = 0

        repo_data = data.get("repository")
        if repo_data is None:
            if DEBUG:
                print(
                    f"    {owner}/{repo}: repository returned null (private/deleted?)", flush=True
                )
            return all_prs, truncated, {}

        rl = data.get("rateLimit")

        pr_connection = repo_data.get("pullRequests")
        if pr_connection is None:
            raise RuntimeError(f"[{owner}/{repo}] pullRequests null (cursor={cursor})")
        page_nodes = [n for n in pr_connection["nodes"] if n]

        # DESC order: the first PR older than the cutoff ends the collection.
        hit_cutoff = False
        kept_this_page = []
        for node in page_nodes:
            created = datetime.fromisoformat(node["createdAt"].replace("Z", "+00:00"))
            if oldest_seen is None or created < oldest_seen:
                oldest_seen = created
            if created < cutoff:
                hit_cutoff = True
                break
            _complete_pr_subconnections(session, owner, repo, node)
            kept_this_page.append(node)

        all_prs.extend(kept_this_page)

        end_cursor = pr_connection["pageInfo"]["endCursor"]
        finished = hit_cutoff or not pr_connection["pageInfo"]["hasNextPage"]
        _append_checkpoint(ckpt, page, kept_this_page, end_cursor, page_size, finished)

        if DEBUG:
            print(
                f"    page {page}: +{len(kept_this_page)} PRs "
                f"(of {len(page_nodes)} on page"
                + (", cutoff reached" if hit_cutoff else "")
                + (f", cost={rl['cost']}, remaining={rl['remaining']})" if rl else ")"),
                flush=True,
            )

        if finished:
            break

        if page >= MAX_PAGES_PER_REPO:
            truncated = True
            oldest_str = oldest_seen.isoformat() if oldest_seen else "unknown"
            print(
                f"    {owner}/{repo}: hit emergency valve of {MAX_PAGES_PER_REPO} pages "
                f"({len(all_prs)} PRs collected), oldest PR seen: {oldest_str}, "
                f"cutoff: {START_DATE} — DATA IS INCOMPLETE, investigate this repo manually",
                flush=True,
            )
            break

        cursor = end_cursor

    return all_prs, truncated, {}


# Main
def _process_repo(proj: dict, done: set) -> tuple[str, int, str]:
    """Runs in a thread. Returns (key, n_prs, status) for logging in main."""
    key = f"{proj['owner']}/{proj['repo']}"
    session = _new_session()
    out_path = _repo_out_path(proj["owner"], proj["repo"])

    try:
        prs, truncated, info = fetch_repo_prs(session, proj["owner"], proj["repo"])
    except RepoNotFound:
        return key, 0, "not_found"
    except Exception as exc:
        return key, 0, f"error: {exc}"

    out_path.write_text(
        json.dumps(
            {
                "owner": proj["owner"],
                "repo": proj["repo"],
                "language": proj["language"],
                "stars": proj["stars"],
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "truncated": truncated,
                "search_fallback_from": info.get("search_fallback_from"),
                "search_skipped": info.get("search_skipped"),
                "prs": prs,
            },
            indent=None,
        ),
        encoding="utf-8",
    )
    _partial_path(proj["owner"], proj["repo"]).unlink(missing_ok=True)
    return key, len(prs), ("truncated" if truncated else "ok")


def main():
    if not GITHUB_TOKENS:
        print("GITHUB_TOKENS or GITHUB_TOKEN is empty in .env")
        sys.exit(1)

    RAW_DIR.mkdir(exist_ok=True)
    projects = _load_projects(PROJECTS_FILE)
    if MAX_REPOS:
        projects = projects[:MAX_REPOS]
    state = _load_state()
    done = set(state["done"])

    pending = [p for p in projects if f"{p['owner']}/{p['repo']}" not in done]

    print(
        f"\nGraphQL collection — {len(projects)} repos, {len(done)} already collected, "
        f"{len(pending)} pending"
    )
    print(f"   Tokens: {len(GITHUB_TOKENS)}   Workers: {MAX_WORKERS}")
    print(f"   Output: {RAW_DIR}/<owner>__<repo>.json")
    print(f"   State: {STATE_FILE}\n")

    start_ts = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process_repo, proj, done): proj for proj in pending}

        for future in as_completed(futures):
            proj = futures[future]
            key = f"{proj['owner']}/{proj['repo']}"
            completed += 1

            try:
                _key, n_prs, status = future.result()
            except Exception as exc:
                print(
                    f"  [{completed}/{len(pending)}] {key}: unhandled exception: {exc}", flush=True
                )
                continue

            if status == "not_found":
                print(f"  [{completed}/{len(pending)}] {key} not found, skipping", flush=True)
            elif status.startswith("error"):
                print(
                    f"  [{completed}/{len(pending)}] {key}: {status}, will retry next run",
                    flush=True,
                )
                continue  # not marked done, retried later
            elif status == "truncated":
                elapsed = time.time() - start_ts
                print(
                    f"  [{completed}/{len(pending)}] {key}: {n_prs} PRs, TRUNCATED "
                    f"(hit emergency valve), flagged in the JSON with truncated=true",
                    flush=True,
                )
            else:
                elapsed = time.time() - start_ts
                print(
                    f"  [{completed}/{len(pending)}] {key}: {n_prs} PRs "
                    f"({elapsed / completed:.1f}s/repo on average)",
                    flush=True,
                )

            done.add(key)
            state["done"] = sorted(done)
            _save_state(state)

    elapsed = time.time() - start_ts
    print(f"\n{'=' * 62}")
    print(f"  Collection complete: {len(done)}/{len(projects)} repos in {elapsed / 60:.1f} min")
    print("  Next step: python classify_prs.py")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
