"""Completes the repositories that fetch_prs.py collected through the search API.

In these repositories (search_fallback_from set in the raw JSON) a removed PR
broke the GraphQL listing, and the search index can also miss PRs. Every PR
number between the oldest and the newest collected PR is probed by number:

- a number that loads as a PR but is not in the JSON is fetched with all the
  fields of fetch_prs.py, appended to "prs" with recovered_via = "number" and
  listed in "recovered_by_number" ("recover_failed" if the fetch fails);
- a number that fails with an INTERNAL error is a removed PR that cannot be
  loaded, and is listed in "unloadable_numbers";
- a number that is not a PR (an issue) is ignored.

Numbers above the newest collected PR are not probed, so PRs opened after the
collection are not added.

Usage:
    python recover_by_number.py            # updates data/raw_prs/*.json
    python recover_by_number.py --check    # writes nothing, compares with the JSON
"""

import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor

import fetch_prs as fp

PROBE_BATCH = 100  # PR numbers per probe query (one alias each)
PROBE_RETRIES = 5
INTERNAL_CONFIRM = 3  # single-number probes before a number counts as removed

PR_BY_NUMBER_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {%(pr_fields)s}
  }
  rateLimit { cost remaining resetAt }
}
""" % {"pr_fields": fp.PR_FIELDS}


def _probe(session, owner: str, repo: str, nums: list[int]) -> dict[int, str]:
    """Returns, for each number, "pr:<createdAt>", "not_pr", "internal" or "failed".
    Uses its own request loop, since fp._run_query hides per-alias errors."""
    aliases = " ".join(f"p{n}: pullRequest(number: {n}) {{ number createdAt }}" for n in nums)
    query = (
        f'query {{ repository(owner: "{owner}", name: "{repo}") {{ {aliases} }} '
        f"rateLimit {{ remaining resetAt }} }}"
    )
    for attempt in range(1, PROBE_RETRIES + 1):
        token = fp._budget.get_token()
        resp = None
        try:
            with fp._inflight:
                resp = session.post(
                    fp.GRAPHQL_URL,
                    json={"query": query},
                    headers={"Authorization": f"bearer {token}"},
                    timeout=90,
                )
            payload = resp.json() if resp.status_code == 200 else {}
        except Exception:
            payload = {}
        if resp is not None and resp.status_code in (401, 404):
            raise RuntimeError(f"{owner}/{repo}: HTTP {resp.status_code}: {resp.text[:200]}")
        if resp is not None:
            fp._capture_header_rate_limit(token, resp)

        errors = payload.get("errors") or []
        if any("rate limit" in (e.get("message") or "").lower() for e in errors):
            fp._budget.mark_exhausted(token)
            fp._budget.switch_to_healthy_or_wait()
            continue
        repository = (payload.get("data") or {}).get("repository")
        if repository is None:
            time.sleep(fp._backoff(attempt))
            continue

        out = {}
        for n in nums:
            node = repository.get(f"p{n}")
            out[n] = f"pr:{node['createdAt']}" if node else "not_pr"
        for e in errors:
            path = e.get("path") or []
            if e.get("type") == "INTERNAL" and len(path) >= 2:
                out[int(path[1][1:])] = "internal"
        return out

    # The whole batch keeps failing: split it until the failing number is alone.
    if len(nums) == 1:
        return {nums[0]: "failed"}
    mid = len(nums) // 2
    return {**_probe(session, owner, repo, nums[:mid]), **_probe(session, owner, repo, nums[mid:])}


def _fetch_pr(session, owner: str, repo: str, number: int) -> dict | None:
    data = fp._run_query(
        session, PR_BY_NUMBER_QUERY, {"owner": owner, "name": repo, "number": number}
    )
    node = (data.get("repository") or {}).get("pullRequest")
    if not node:
        return None
    fp._complete_pr_subconnections(session, owner, repo, node)
    node["recovered_via"] = "number"
    return node


def process(path, check: bool) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    owner, repo = d["owner"], d["repo"]
    session = fp._new_session()

    # In --check mode the earlier recoveries count as missing, so the probe
    # repeats the original recovery and can be compared with the JSON.
    have = {p["number"] for p in d["prs"] if not (check and "recovered_via" in p)}
    lo, hi = min(have), max(have)

    status = {}
    for start in range(lo, hi + 1, PROBE_BATCH):
        status.update(_probe(session, owner, repo, list(range(start, min(start + PROBE_BATCH, hi + 1)))))

    # INTERNAL can be transient: a number counts as removed only if it keeps failing alone.
    internal = []
    for n in sorted(k for k, v in status.items() if v == "internal"):
        if all(_probe(session, owner, repo, [n])[n] == "internal" for _ in range(INTERNAL_CONFIRM)):
            internal.append(n)
        else:
            status[n] = _probe(session, owner, repo, [n])[n]

    missing = sorted(
        n
        for n, v in status.items()
        if v.startswith("pr:") and v[3:] >= fp.START_DATE and n not in have
    )
    failed_probe = sorted(n for n, v in status.items() if v == "failed")

    summary = {
        "repo": f"{owner}/{repo}",
        "probed": hi - lo + 1,
        "missing": len(missing),
        "unloadable": len(internal),
        "probe_failed": failed_probe,
    }
    if check:
        old_rec, old_unl = set(d.get("recovered_by_number", [])), set(d.get("unloadable_numbers", []))
        summary.update(
            {
                "json_recovered": len(old_rec),
                "json_unloadable": len(old_unl),
                "recovered_only_now": sorted(set(missing) - old_rec),
                "recovered_only_json": sorted(old_rec - set(missing)),
                "unloadable_only_now": sorted(set(internal) - old_unl),
                "unloadable_only_json": sorted(old_unl - set(internal)),
            }
        )
        return summary

    recovered, failed = [], []
    for n in missing:
        node = _fetch_pr(session, owner, repo, n)
        if node:
            d["prs"].append(node)
            recovered.append(n)
        else:
            failed.append(n)
    d["recovered_by_number"] = sorted(set(d.get("recovered_by_number", [])) | set(recovered))
    d["recover_failed"] = failed
    d["unloadable_numbers"] = internal
    path.write_text(json.dumps(d, indent=None), encoding="utf-8")
    return summary


def main():
    if not fp.GITHUB_TOKENS:
        sys.exit("GITHUB_TOKENS or GITHUB_TOKEN is empty in .env")
    check = "--check" in sys.argv[1:]

    paths = []
    for path in sorted(fp.RAW_DIR.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("search_fallback_from"):
            paths.append(path)
    print(f"{len(paths)} repositories collected through the search API", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, min(len(paths), len(fp.GITHUB_TOKENS)))) as pool:
        for s in pool.map(lambda p: process(p, check), paths):
            print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
