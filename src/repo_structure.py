"""
Minified repository file structure for check generators.

Used by SOURCE_CHECK, ARCH_DETERMINISTIC, ARCH_SEMANTIC, and PROCESS_CHECK
generators so that generated checks are correct and precise for the repo.
Process checks use it for artifact-based scope (e.g. pnpm-lock.yaml,
pyproject.toml paths).

Ignore list: when the repo has a .gitignore, we derive directory names to skip
from it (so we don't hide paths the repo cares about, e.g. .github/). Otherwise
we use a minimal fallback so only .git is skipped.
"""

from __future__ import annotations

from pathlib import Path

# Minimal fallback when no .gitignore: only skip .git to avoid walking the object db
_FALLBACK_IGNORE_DIRS = frozenset({".git"})


def _ignore_dirs_from_gitignore(repo_root: Path) -> frozenset[str] | None:
    """
    Read .gitignore under repo_root and return a set of directory names to skip.
    We only add a dir when the pattern clearly refers to a whole directory:
    - Single segment (e.g. "node_modules", ".venv") with no globs, or
    - Single segment with trailing slash (e.g. "dist/").
    We do NOT add the first segment of paths like "packages/backend/settings.py",
    since that would skip the whole tree instead of one file. Returns None if no .gitignore.
    """
    gitignore = repo_root / ".gitignore"
    if not gitignore.is_file():
        return None
    try:
        raw = gitignore.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    out: set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        # Only consider whole-directory patterns: one segment, no path sep (or only trailing /)
        if line.endswith("/"):
            line = line[:-1]
        if "/" in line:
            # Multi-segment path (e.g. packages/backend/settings.py) -> skip; don't add first segment
            continue
        if "*" in line or "?" in line:
            continue
        out.add(line)
    out.add(".git")
    return frozenset(out)


def get_ignore_dirs(repo_root: str | Path) -> frozenset[str]:
    """
    Return directory names to skip when walking the repo: from .gitignore if
    present, else minimal fallback (.git only). Used by minified_file_structure
    and by graph_builder when building the dependency graph.
    """
    root = Path(repo_root).resolve()
    from_gitignore = _ignore_dirs_from_gitignore(root)
    return from_gitignore if from_gitignore is not None else _FALLBACK_IGNORE_DIRS

# Max paths to include (keep prompt small)
_DEFAULT_MAX_PATHS = 300

# Max depth below repo root (e.g. 5 => root/a/b/c/d/e)
_DEFAULT_MAX_DEPTH = 6


def minified_file_structure(
    repo_root: str | Path,
    *,
    ignore_dirs: frozenset[str] | None = None,
    max_paths: int = _DEFAULT_MAX_PATHS,
    max_depth: int = _DEFAULT_MAX_DEPTH,
) -> str:
    """
    Produce a minified listing of file paths under repo_root for use in prompts.

    Skips directories using ignore_dirs if provided, else .gitignore-derived
    list when present, else a minimal fallback (.git only). Does not skip
    dot-directories by default so paths like .github/ are included.
    Returns a single string with one path per line (relative to repo_root), sorted.
    Truncates if over max_paths.
    """
    root = Path(repo_root).resolve()
    if not root.is_dir():
        return ""

    if ignore_dirs is not None:
        ignore = ignore_dirs
    else:
        from_gitignore = _ignore_dirs_from_gitignore(root)
        ignore = from_gitignore if from_gitignore is not None else _FALLBACK_IGNORE_DIRS
    paths: list[str] = []

    def _walk(dir_path: Path, depth: int) -> None:
        if depth > max_depth or len(paths) >= max_paths:
            return
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for p in entries:
            if len(paths) >= max_paths:
                return
            rel = p.relative_to(root)
            rel_str = str(rel).replace("\\", "/")
            if p.is_dir():
                if p.name in ignore:
                    continue
                _walk(p, depth + 1)
            else:
                paths.append(rel_str)

    _walk(root, 0)
    paths.sort()
    if len(paths) >= max_paths:
        return "\n".join(paths) + f"\n... (truncated at {max_paths} paths)"
    return "\n".join(paths)
