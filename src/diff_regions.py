"""
Parse unified git diffs to locate **changed line numbers in the new (post-image) file**.

Used to run static checks only on the code that was added or modified, not on unchanged
lines in touched files or on untouched files elsewhere in the repo.
"""

from __future__ import annotations

import re
from collections import defaultdict
# @@ -old_start,old_count +new_start,new_count @@
_HUNK_RE = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@"
)


def _path_from_diff_plus_line(line: str) -> str | None:
    """Return repo-relative path from a ``+++ b/path`` line, or None for ``/dev/null``."""
    line = line.rstrip("\n")
    if not line.startswith("+++ "):
        return None
    rest = line[4:].strip()
    if rest == "/dev/null":
        return None
    if rest.startswith('"') and rest.endswith('"'):
        rest = rest[1:-1]
    if rest.startswith("b/"):
        rest = rest[2:]
    elif rest.startswith("a/"):
        rest = rest[2:]
    out = rest.strip().replace("\\", "/")
    return out or None


def _path_from_diff_minus_line(line: str) -> str | None:
    """Return repo-relative path from a ``--- a/path`` line, or None for ``/dev/null``."""
    line = line.rstrip("\n")
    if not line.startswith("--- "):
        return None
    rest = line[4:].strip()
    if rest == "/dev/null":
        return None
    if rest.startswith('"') and rest.endswith('"'):
        rest = rest[1:-1]
    if rest.startswith("a/"):
        rest = rest[2:]
    elif rest.startswith("b/"):
        rest = rest[2:]
    out = rest.strip().replace("\\", "/")
    return out or None


def new_file_paths_from_unified_diff(diff: str) -> set[str]:
    """
    Paths (POSIX, repo-relative) that this unified diff **adds** as new files
    (``new file mode`` and/or ``--- /dev/null`` before ``+++ b/...``).
    """
    if not (diff or "").strip():
        return set()
    new_paths: set[str] = set()
    block: list[str] = []
    for raw in diff.splitlines():
        if raw.startswith("diff --git "):
            _new_paths_from_git_block(block, new_paths)
            block = [raw]
        else:
            block.append(raw)
    _new_paths_from_git_block(block, new_paths)
    return new_paths


def _new_paths_from_git_block(lines: list[str], new_paths: set[str]) -> None:
    if not lines:
        return
    body = "\n".join(lines)
    is_new = "new file mode" in body or "\n--- /dev/null\n" in body or body.startswith("--- /dev/null\n")
    if not is_new:
        return
    for ln in lines:
        if ln.startswith("+++ "):
            p = _path_from_diff_plus_line(ln)
            if p:
                new_paths.add(p)
            break


def touched_paths_from_unified_diff(diff: str) -> set[str]:
    """All repo-relative paths referenced on ``---`` / ``+++`` lines (old and new sides)."""
    if not (diff or "").strip():
        return set()
    out: set[str] = set()
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            p = _path_from_diff_plus_line(raw)
            if p:
                out.add(p)
        elif raw.startswith("--- "):
            p = _path_from_diff_minus_line(raw)
            if p:
                out.add(p)
    return out


def build_neighbor_directory_listing_hint(
    root: str | Path,
    diff_text: str,
    *,
    max_dirs: int = 14,
    max_names_per_dir: int = 36,
    max_top_level: int = 24,
) -> str:
    """
    Short textual hint for LLM review: top-level dirs plus file names in directories
    that contain paths touched by ``diff_text`` (post-image paths only where possible).
    """
    from pathlib import Path

    r = Path(root).resolve()
    touched = touched_paths_from_unified_diff(diff_text)
    dirs: set[str] = set()
    for p in touched:
        parts = Path(p.replace("\\", "/")).parts
        if len(parts) > 1:
            dirs.add(str(Path(*parts[:-1]).as_posix()))
        dirs.add(".")
    lines_out: list[str] = []
    try:
        top = sorted(
            x.name
            for x in r.iterdir()
            if x.is_dir()
            and not x.name.startswith(".")
            and x.name not in ("__pycache__", "node_modules", ".venv", "venv")
        )[:max_top_level]
        if top:
            lines_out.append(f"Repository top-level dirs: {', '.join(top)}")
    except OSError:
        pass
    for d in sorted(dirs)[:max_dirs]:
        dp = r / d
        if not dp.is_dir():
            continue
        try:
            names = sorted(
                x.name
                for x in dp.iterdir()
                if x.is_file()
                and not x.name.startswith(".")
                and "__pycache__" not in x.name
            )[:max_names_per_dir]
            label = d if d != "." else "(repo root)"
            lines_out.append(f"Files under {label!r}: {', '.join(names)}")
        except OSError:
            continue
    return "\n".join(lines_out) if lines_out else ""


def arch_det_violation_in_diff_scope(message: str, diff_paths: set[str]) -> bool:
    """
    Heuristic: a deterministic arch violation is **attributed to the patch** if the message
    mentions a path (or module form) that appears in the unified diff (new, old, or renamed side).
    """
    from pathlib import Path

    msg = message or ""
    if not msg.strip() or not diff_paths:
        return False
    msg_l = msg.lower()
    for p in diff_paths:
        p = (p or "").replace("\\", "/").strip()
        if not p:
            continue
        if p.lower() in msg_l:
            return True
        base = Path(p).name
        if len(base) >= 3 and base.lower() in msg_l:
            return True
        mod = p.replace("/", ".")
        if mod.endswith(".py"):
            mod = mod[:-3]
        if len(mod) > 3 and mod.lower() in msg_l:
            return True
    return False


def filter_arch_det_per_check_results_by_diff(
    per_check_results: list[tuple[dict, list[str]]],
    diff_text: str,
) -> tuple[list[tuple[dict, list[str]]], int, int]:
    """
    Drop arch_det violations whose messages do not reference any path touched by ``diff_text``.

    Returns ``(filtered_per_check_results, kept_count, dropped_count)``.
    If the diff yields no paths, returns the input unchanged (kept = original total).
    """
    paths = touched_paths_from_unified_diff(diff_text)
    if not paths:
        total = sum(len(v) for _, v in per_check_results)
        return per_check_results, total, 0
    out: list[tuple[dict, list[str]]] = []
    kept_total = 0
    dropped_total = 0
    for check_info, violations_list in per_check_results:
        kept = [m for m in violations_list if arch_det_violation_in_diff_scope(m, paths)]
        dropped_total += len(violations_list) - len(kept)
        kept_total += len(kept)
        out.append((check_info, kept))
    return out, kept_total, dropped_total


def parse_unified_diff_new_line_numbers(diff: str) -> dict[str, set[int]]:
    """
    Map **repo-relative POSIX path** -> set of **1-based** line numbers in the **new** file
    that correspond to ``+`` lines in the diff (added or replacement text in the new version).

    Context lines (`` ``) and pure deletions (``-``) do not add to the set for the new file,
    except that after a ``-`` block, ``+`` lines are recorded at their new-file positions.

    Empty or non-unified input returns {}.
    """
    if not (diff or "").strip():
        return {}

    result: dict[str, set[int]] = defaultdict(set)
    current: str | None = None
    new_line = 1

    for raw in diff.splitlines():
        if raw.startswith("diff --git "):
            current = None
            continue
        if raw.startswith("Binary files ") and " differ" in raw:
            current = None
            continue
        if raw.startswith("+++ "):
            current = _path_from_diff_plus_line(raw)
            continue
        if current is None:
            continue
        if raw.startswith("@@"):
            m = _HUNK_RE.match(raw)
            if m:
                new_line = int(m.group(3))
            continue
        if raw.startswith("+++") or raw.startswith("---"):
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            result[current].add(new_line)
            new_line += 1
        elif raw.startswith("-"):
            pass
        elif raw.startswith(" "):
            new_line += 1
        elif raw.startswith("\\"):
            # "\ No newline at end of file"
            continue

    return dict(result)


def normalize_changed_paths_for_root(
    root: str,
    changed: dict[str, set[int]],
) -> dict[str, set[int]]:
    """
    Normalize keys from :func:`parse_unified_diff_new_line_numbers` so they match
    ``Path.relative_to(root).as_posix()`` strings.
    Drops paths that do not exist under ``root``.
    """
    from pathlib import Path

    r = Path(root).resolve()
    out: dict[str, set[int]] = {}
    for rel, lines in changed.items():
        rel = rel.replace("\\", "/").lstrip("./")
        p = r / rel
        try:
            p.resolve().relative_to(r)
        except ValueError:
            continue
        if not p.is_file():
            continue
        key = p.resolve().relative_to(r).as_posix()
        out[key] = set(lines)
    return out
