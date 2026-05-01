"""
Clone a GitHub repo into .data for running checks.

Used by scan when --repo owner/repo is given: ensures the repo exists under .data/<owner>_<repo>
and returns that path as the root to scan.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


DEFAULT_DATA_DIR = ".data"


def clone_repo(
    owner: str,
    repo: str,
    parent_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    branch: str = "main",
) -> Path:
    """
    Clone https://github.com/<owner>/<repo> into <parent_dir>/<owner>_<repo>.
    If the directory already exists, do nothing and return it.
    Returns the path to the cloned repo root.
    """
    parent = Path(parent_dir).resolve()
    dir_name = f"{owner}_{repo}".replace("/", "_")
    dest = parent / dir_name

    if dest.is_dir() and (dest / ".git").is_dir():
        return dest

    parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{owner}/{repo}.git"
    # Try requested branch first, then clone default branch
    for try_branch in (branch, None):
        cmd = ["git", "clone", "--depth", "1", url, str(dest)]
        if try_branch:
            cmd[2:2] = ["--branch", try_branch]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            print("Error: git not found. Install git to clone repos.", file=sys.stderr)
            raise
        if r.returncode == 0:
            return dest
        if try_branch and "Remote branch" in (r.stderr or ""):
            continue
        print(f"Error: git clone failed: {r.stderr or r.stdout or 'unknown'}", file=sys.stderr)
        raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
