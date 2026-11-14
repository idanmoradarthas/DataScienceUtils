"""Publish the Tier 2 dedup result on the pull request: a sticky comment and a separate check row.

Input is the JSON written by ``pr_dedup_report.py report --result-file``. Two things are published:

* One sticky comment, found by a hidden marker and authored by ``github-actions[bot]``, updated in place on every
  push. It is created only when there is something to say (findings or an incomplete scan); once a later push is clean
  an existing comment is updated to say so instead of being deleted, so the thread stays readable.
* A check run named ``Tier 2 dedup findings (advisory)``: ``neutral`` when there are findings, ``success`` when clean,
  ``failure`` when the scan could not complete. GitHub Actions jobs cannot finish neutral, so the row is the neutral
  signal; the dedup job itself stays green on findings and goes red only when the scan could not run.

Needs ``pull-requests: write`` and ``checks: write``. A read-only token (fork or Dependabot PRs) is not an error: the
script prints a notice and the result stays in the job summary.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from pr_dedup_report import COMMENT_MARKER

CHECK_NAME = "Tier 2 dedup findings (advisory)"
BOT_LOGIN = "github-actions[bot]"
_MAX_PAGES = 10
_RETRIES = 3


class ApiError(Exception):
    """GitHub API call failed."""

    def __init__(self, status: int, message: str) -> None:
        """Store the HTTP status (0 for network errors) and message."""
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def _error_body(exc: urllib.error.HTTPError) -> str:
    """Return the first 300 characters of an HTTP error body, or its reason if the body cannot be read."""
    try:
        return exc.read().decode("utf-8", errors="replace")[:300]
    except Exception:
        return str(exc.reason)


class GitHubApi:
    """Minimal GitHub REST client with retries for transient failures."""

    def __init__(self, token: str, api_url: str = "https://api.github.com", sleep: Any = time.sleep) -> None:
        """Create a client authenticated with ``token``."""
        self._token = token
        self._api_url = api_url.rstrip("/")
        self._sleep = sleep

    def request(self, method: str, path: str, payload: dict | None = None) -> Any:
        """Call the API and return the decoded JSON body; raise ApiError on failure."""
        url = path if path.startswith("http") else f"{self._api_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "tier2-dedup-publisher",
        }
        last: ApiError | None = None
        for attempt in range(_RETRIES):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = resp.read().decode("utf-8")
                    return json.loads(body) if body else None
            except urllib.error.HTTPError as exc:
                last = ApiError(exc.code, _error_body(exc))
                if exc.code < 500 and exc.code != 429:
                    raise last from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last = ApiError(0, str(exc))
            if attempt < _RETRIES - 1:
                self._sleep(2**attempt)
        assert last is not None
        raise last


def find_sticky_comment(api: Any, repo: str, pr_number: int) -> dict | None:
    """Return the bot's sticky dedup comment on the PR, or None."""
    for page in range(1, _MAX_PAGES + 1):
        comments = api.request("GET", f"/repos/{repo}/issues/{pr_number}/comments?per_page=100&page={page}")
        for comment in comments or []:
            if COMMENT_MARKER in (comment.get("body") or "") and (comment.get("user") or {}).get("login") == BOT_LOGIN:
                return comment
        if not comments or len(comments) < 100:
            break
    return None


def publish_comment(api: Any, repo: str, pr_number: int, result: dict) -> str:
    """Create or update the sticky comment as required by the result; return what happened."""
    existing = find_sticky_comment(api, repo, pr_number)
    body = result["comment_markdown"]
    if existing is not None:
        if (existing.get("body") or "") == body:
            return "comment unchanged"
        api.request("PATCH", f"/repos/{repo}/issues/comments/{existing['id']}", {"body": body})
        return "comment updated"
    if result.get("comment_needed"):
        api.request("POST", f"/repos/{repo}/issues/{pr_number}/comments", {"body": body})
        return "comment created"
    return "no comment needed"


def publish_check_run(api: Any, repo: str, sha: str, result: dict, run_url: str = "") -> str:
    """Create or update the neutral/success/failure check row on the PR head commit; return what happened."""
    if result["status"] == "skipped":
        return "no check row needed"
    summary = result["comment_markdown"].replace(COMMENT_MARKER + "\n", "", 1)
    payload: dict[str, Any] = {
        "name": CHECK_NAME,
        "head_sha": sha,
        "status": "completed",
        "conclusion": result["check_conclusion"],
        "output": {"title": result["check_title"], "summary": summary[:60000]},
    }
    if run_url:
        payload["details_url"] = run_url
    query = urllib.parse.urlencode({"check_name": CHECK_NAME})
    existing = api.request("GET", f"/repos/{repo}/commits/{sha}/check-runs?{query}")
    runs = [r for r in (existing or {}).get("check_runs", []) if r.get("name") == CHECK_NAME]
    if runs:
        api.request("PATCH", f"/repos/{repo}/check-runs/{runs[0]['id']}", payload)
        return "check row updated"
    api.request("POST", f"/repos/{repo}/check-runs", payload)
    return "check row created"


def publish(api: Any, repo: str, pr_number: int, sha: str, result: dict, run_url: str = "") -> list[str]:
    """Publish comment and check row. A read-only token (HTTP 403) downgrades to a notice."""
    outcomes: list[str] = []
    for label, action in (
        ("comment", lambda: publish_comment(api, repo, pr_number, result)),
        ("check row", lambda: publish_check_run(api, repo, sha, result, run_url)),
    ):
        try:
            outcomes.append(action())
        except ApiError as exc:
            if exc.status == 403:
                outcomes.append(f"{label} skipped: token is read-only ({exc})")
                print(
                    f"::notice title=Tier 2 dedup (advisory)::Could not write the {label} (read-only token, "
                    "for example a fork or Dependabot PR). The result is in the job summary."
                )
            else:
                raise
    return outcomes


def main() -> int:
    """CLI entry point; return the process exit code."""
    parser = argparse.ArgumentParser(description="Publish the Tier 2 dedup result on the pull request.")
    parser.add_argument("--result", required=True, help="Path to the result JSON from pr_dedup_report.py")
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    parser.add_argument("--sha", required=True, help="PR head commit SHA the check row is attached to")
    parser.add_argument("--run-url", default="", help="URL of the workflow run")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("::error title=Tier 2 dedup (advisory)::GITHUB_TOKEN is not set; cannot publish the dedup result.")
        return 1
    result_path = Path(args.result)
    if not result_path.is_file():
        print(f"::error title=Tier 2 dedup (advisory)::Result file '{result_path}' not found; nothing to publish.")
        return 1

    result = json.loads(result_path.read_text(encoding="utf-8"))
    api = GitHubApi(token, os.environ.get("GITHUB_API_URL", "https://api.github.com"))
    try:
        outcomes = publish(api, args.repo, args.pr, args.sha, result, args.run_url)
    except ApiError as exc:
        print(f"::error title=Tier 2 dedup (advisory)::Could not publish the dedup result: {exc}")
        return 1
    for outcome in outcomes:
        print(outcome)
    return 0


if __name__ == "__main__":
    sys.exit(main())
