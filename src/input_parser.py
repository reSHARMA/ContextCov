"""
Input module: parse GitHub file paths and fetch raw file content.

Accepts paths like: raw/owner/repo/CLAUDE.md
and resolves: owner=owner, repo=repo, file_path=CLAUDE.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass(frozen=True)
class GitHubFileInput:
    """Parsed GitHub file reference: owner, repo, path within repo, optional branch."""

    owner: str
    repo: str
    file_path: str
    branch: Optional[str] = None

    @property
    def repo_slug(self) -> str:
        """Return owner/repo."""
        return f"{self.owner}/{self.repo}"

    def raw_url(self, branch: Optional[str] = None) -> str:
        """URL for raw file content on GitHub. Uses self.branch if set, else argument, else 'main'."""
        ref = branch or self.branch or "main"
        return (
            f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/{ref}/{self.file_path}"
        )


def _strip_github_url_prefix(path: str) -> str:
    """
    Normalize a pasted GitHub URL to the internal `raw/owner/repo/...` form.

    Accepts the URLs users actually copy out of a browser:
      https://github.com/owner/repo/blob/main/AGENTS.md
      https://raw.githubusercontent.com/owner/repo/main/AGENTS.md
    Anything else is returned unchanged.
    """
    s = path.strip()
    # github.com/owner/repo/blob/<ref>/<path> already carries its own "blob" marker.
    for prefix in ("https://github.com/", "http://github.com/", "github.com/"):
        if s.lower().startswith(prefix):
            return "raw/" + s[len(prefix):].lstrip("/")
    # raw.githubusercontent.com/owner/repo/<ref>/<path> has no "blob" marker, so
    # insert one; otherwise the ref would be parsed as part of the file path.
    for prefix in (
        "https://raw.githubusercontent.com/",
        "http://raw.githubusercontent.com/",
        "raw.githubusercontent.com/",
    ):
        if s.lower().startswith(prefix):
            rest = s[len(prefix):].lstrip("/").split("/")
            if len(rest) >= 4:
                return "raw/" + "/".join(rest[:2] + ["blob"] + rest[2:])
            return "raw/" + "/".join(rest)
    return s


def parse_github_file_path(path: str) -> Optional[GitHubFileInput]:
    """
    Parse a path in one of these forms:

    - raw/owner/repo/path/to/file.md
    - raw/owner/repo/blob/branch/path/to/file.md   (branch taken from path)
    - https://github.com/owner/repo/blob/branch/path/to/file.md
    - https://raw.githubusercontent.com/owner/repo/branch/path/to/file.md

    Returns a GitHubFileInput with owner, repo, file_path, and optional branch,
    or None if the format is invalid.
    """
    if not isinstance(path, str):
        return None
    path = _strip_github_url_prefix(path)
    if not path.strip().startswith("raw/"):
        return None
    rest = path.strip()[4:].lstrip("/")
    parts = rest.split("/")
    if len(parts) < 3:
        return None
    owner, repo = parts[0], parts[1]
    if not owner or not repo:
        return None

    # GitHub web-URL style: owner/repo/blob/branch/rest
    if len(parts) >= 4 and parts[2].lower() == "blob":
        branch = parts[3]
        file_path_in_repo = "/".join(parts[4:]) if len(parts) > 4 else ""
        if not branch or not file_path_in_repo:
            return None
        return GitHubFileInput(
            owner=owner,
            repo=repo,
            file_path=file_path_in_repo,
            branch=branch,
        )

    # Simple form: owner/repo/path/to/file
    file_path_in_repo = "/".join(parts[2:])
    if not file_path_in_repo:
        return None
    return GitHubFileInput(owner=owner, repo=repo, file_path=file_path_in_repo)


def fetch_raw_content(
    inp: GitHubFileInput,
    branch: str = "main",
    session: Optional[requests.Session] = None,
    timeout: float = 30.0,
    try_fallback_branch_on_404: bool = True,
) -> str:
    """
    Fetch raw file content from GitHub for the given GitHubFileInput.

    Uses GITHUB_TOKEN from environment if set for higher rate limits.
    If try_fallback_branch_on_404 is True and the request returns 404, retries
    with the other common default branch (main <-> master) for repos that use
    a different default.
    Raises requests.HTTPError on 4xx/5xx after any fallback attempt.
    """
    url = inp.raw_url(branch=branch)
    headers: dict[str, str] = {"Accept": "application/vnd.github.raw"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    sess = session or requests.Session()
    resp = sess.get(url, headers=headers, timeout=timeout)
    if resp.status_code == 404 and try_fallback_branch_on_404:
        other = "master" if branch == "main" else "main"
        url_fallback = inp.raw_url(branch=other)
        resp2 = sess.get(url_fallback, headers=headers, timeout=timeout)
        if resp2.status_code == 200:
            return resp2.text
        resp2.raise_for_status()
    resp.raise_for_status()
    return resp.text
