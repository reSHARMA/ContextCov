"""
Static Check Runner: executes generated Python checks (Tree-Sitter or regex) on source files.

Parses each file at most once (using tree-sitter when available), then runs all applicable
checks in a restricted scope with `tree`, `source_bytes`, and `language`. Checks set
`result = "FAIL"` on violation. Supports regex-only checks when tree-sitter is not used.
"""

from __future__ import annotations

import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional SOURCE_CHECK violation metadata: checks may set these in exec scope before result="FAIL".
_STATIC_VIOLATION_OPT_KEYS = ("line", "line_end", "column", "snippet", "detail")
_TRACEBACK_MAX_LEN = 8000

# Optional tree-sitter: if missing, only "text" (regex) checks run
_PARSER_CACHE: Dict[str, Any] = {}
_TS_AVAILABLE = False
_TS_LANGUAGES_AVAILABLE = False

try:
    import tree_sitter
    _TS_AVAILABLE = True
except ImportError:
    tree_sitter = None  # type: ignore

try:
    from tree_sitter_languages import get_parser, get_language
    _TS_LANGUAGES_AVAILABLE = True
except ImportError:
    get_parser = None  # type: ignore
    get_language = None  # type: ignore


def detect_language(file_path: str | Path) -> str:
    """
    Infer primary language from file extension for parser selection.
    Returns one of: python, typescript, javascript, text.
    """
    path = Path(file_path)
    suffix = (path.suffix or "").lower()
    if suffix in (".py", ".pyi"):
        return "python"
    if suffix in (".ts", ".tsx"):
        return "typescript"
    if suffix in (".js", ".jsx", ".mjs", ".cjs"):
        return "javascript"
    if suffix in (".md", ".txt", ".json", ".yml", ".yaml", ".html", ".css", ".scss"):
        return "text"
    return "text"


# Set on the first failed parser construction so the backend can be reported as
# broken rather than silently degrading every AST check to a vacuous pass.
_TS_BACKEND_ERROR: str | None = None

# Directives of AST checks skipped because the parser backend is unusable.
# Drained by run_source_checks_for_repo into its stats.
_SKIPPED_NO_PARSER: List[str] = []


def _get_parser(lang: str):
    """Return a tree-sitter Parser for the given language, or None if not available."""
    global _TS_BACKEND_ERROR
    if not _TS_AVAILABLE or not _TS_LANGUAGES_AVAILABLE or get_parser is None:
        return None
    if lang == "typescript":
        # Many bundles expose TypeScript as "typescript" or use "javascript" for both
        for name in ("typescript", "javascript"):
            try:
                return get_parser(name)
            except Exception as e:
                _TS_BACKEND_ERROR = f"{type(e).__name__}: {e}"
                continue
        return None
    try:
        return get_parser(lang)
    except Exception as e:
        _TS_BACKEND_ERROR = f"{type(e).__name__}: {e}"
        return None


def tree_sitter_backend_ok() -> bool:
    """
    True when tree-sitter is installed AND can actually build a parser.

    An incompatible tree-sitter / tree-sitter-languages pair imports cleanly but
    raises when constructing a Language, which would otherwise leave every
    AST-based check running with tree=None and reporting a pass it never made.
    """
    if not _TS_AVAILABLE or not _TS_LANGUAGES_AVAILABLE or get_parser is None:
        return False
    return _get_parser("python") is not None


def tree_sitter_backend_error() -> str | None:
    """The error from the last failed parser construction, if any."""
    return _TS_BACKEND_ERROR


def _static_check_violation_dict(
    *,
    message: str,
    check_index: int,
    target_lang: str,
    file_lang: str,
    exec_globals: Optional[Dict[str, Any]] = None,
    exception: Optional[BaseException] = None,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "message": message,
        "check_index": check_index,
        "target_lang": target_lang,
        "file_lang": file_lang,
    }
    if exec_globals is not None:
        for k in _STATIC_VIOLATION_OPT_KEYS:
            if k not in exec_globals:
                continue
            v = exec_globals.get(k)
            if v is not None:
                d[k] = v
    if exception is not None:
        d["exception_type"] = type(exception).__name__
        tb = traceback.format_exc()
        if len(tb) > _TRACEBACK_MAX_LEN:
            tb = tb[:_TRACEBACK_MAX_LEN] + "\n... (truncated)"
        d["traceback"] = tb
    return d


def run_static_checks(
    file_path: str | Path,
    checks: List[Dict[str, Any]],
    *,
    encoding: str = "utf-8",
    line_subset: Optional[set[int]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Run a list of static checks on a file. Each check must have "code" and optionally "target_lang".

    line_subset
        If set, only **lines with these 1-based line numbers** from the file are passed as
        ``source_text`` / ``source_bytes`` (and the tree is re-parsed from that slice). This scopes
        checks to **changed** code from a unified diff. Tree-sitter / AST checks may be less
        reliable on fragments than on whole files.

    Returns a list of (check directive or fallback id, violation_dict) for each violation.
    violation_dict always includes ``message``, ``check_index``, ``target_lang``, ``file_lang``.
    On ``result = "FAIL"``, optional keys from the check's global scope are copied if set:
    ``line``, ``line_end``, ``column``, ``snippet``, ``detail``.
    On check runtime errors, ``exception_type`` and ``traceback`` are added.
    """
    path = Path(file_path)
    if not path.is_file():
        return [(str(path), {"message": "File not found"})]
    try:
        raw_bytes = path.read_bytes()
    except Exception as e:
        return [(str(path), {"message": f"Read error: {e}"})]

    full_text = raw_bytes.decode(encoding, errors="replace")
    line_list = full_text.splitlines()

    if line_subset is not None:
        if not line_subset:
            return []
        selected = [line_list[i - 1] for i in sorted(line_subset) if 1 <= i <= len(line_list)]
        source_text = "\n".join(selected)
        source_bytes = source_text.encode(encoding)
    else:
        source_text = full_text
        source_bytes = raw_bytes

    file_lang = detect_language(path)
    violations: List[Tuple[str, str]] = []

    # Build tree once per language if we have any tree-sitter check for this file
    tree = None
    language = None
    parser = None
    if _TS_AVAILABLE and _TS_LANGUAGES_AVAILABLE:
        parser = _get_parser(file_lang)
        if parser is not None:
            tree = parser.parse(source_bytes)
            try:
                if get_language is not None:
                    for lang_name in (file_lang, "javascript" if file_lang == "typescript" else None):
                        if lang_name is None:
                            continue
                        try:
                            language = get_language(lang_name)
                            break
                        except Exception:
                            continue
                    else:
                        language = None
                else:
                    language = getattr(parser, "language", None)
            except Exception:
                language = None

    for i, check in enumerate(checks):
        code = check.get("code") or ""
        target_lang = (check.get("target_lang") or "text").strip().lower()
        if target_lang in ("ts",):
            target_lang = "typescript"
        if target_lang in ("js",):
            target_lang = "javascript"

        # If check expects a specific language and we don't have a tree for it, skip or run with None
        run_tree = tree
        run_language = language
        if target_lang not in ("text",) and file_lang != target_lang:
            # File is not in the target language; optionally skip or run with tree=None
            run_tree = None
            run_language = None
        elif target_lang not in ("text",) and run_tree is None and not tree_sitter_backend_ok():
            # The check targets this file's language but the tree-sitter backend is
            # broken. Running it with tree=None makes it report a pass it never
            # evaluated, so skip it and record that it was skipped instead.
            _SKIPPED_NO_PARSER.append(str(check.get("directive") or f"check_{i}"))
            continue

        exec_globals = {
            "tree": run_tree,
            "source_bytes": source_bytes,
            "source_text": source_text,
            "language": run_language,
            "re": re,
            "result": None,
            "diff_changed_lines": line_subset,
        }
        directive = check.get("directive", f"check_{i}")
        try:
            exec(code, exec_globals)
        except Exception as e:
            violations.append(
                (
                    directive,
                    _static_check_violation_dict(
                        message=f"Check runtime error: {e}",
                        check_index=i,
                        target_lang=target_lang,
                        file_lang=file_lang,
                        exception=e,
                    ),
                )
            )
            continue
        if exec_globals.get("result") == "FAIL":
            # Generated checks often set no `message`. Falling back to a bare
            # "Constraint violated" tells the user nothing about which rule fired,
            # so use the directive (the rule text from the instruction file).
            msg_raw = exec_globals.get("message") or directive or "Constraint violated"
            msg = msg_raw if isinstance(msg_raw, str) else str(msg_raw)
            violations.append(
                (
                    directive,
                    _static_check_violation_dict(
                        message=msg,
                        check_index=i,
                        target_lang=target_lang,
                        file_lang=file_lang,
                        exec_globals=exec_globals,
                    ),
                )
            )
    return violations


def collect_checks_from_mapping(
    mapping: Dict[str, Dict[str, Any]],
    *,
    trigger_glob: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Flatten all SOURCE_CHECK strategies that have static_check from the mapping into a single
    list of check dicts (each with code, target_lang, and optionally directive for reporting).
    If trigger_glob is set, only include checks whose trigger matches (simple substring match).
    """
    checks = []
    for entry in mapping.values() if isinstance(mapping, dict) else []:
        if not isinstance(entry, dict):
            continue
        for s in entry.get("strategies") or []:
            if not isinstance(s, dict):
                continue
            if (s.get("type") or "").strip().upper() != "SOURCE_CHECK":
                continue
            sc = s.get("static_check")
            if not sc or not sc.get("code"):
                continue
            if trigger_glob is not None and trigger_glob not in str(s.get("trigger") or ""):
                continue
            checks.append({
                "code": sc.get("code", ""),
                "target_lang": sc.get("target_lang", "text"),
                "directive": s.get("directive", ""),
            })
    return checks


def run_static_checks_on_file(
    file_path: str | Path,
    mapping: Dict[str, Dict[str, Any]],
    *,
    trigger_glob: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Load checks from mapping and run them on the given file. Returns list of (directive, message).
    """
    checks = collect_checks_from_mapping(mapping, trigger_glob=trigger_glob)
    if not checks:
        return []
    return run_static_checks(file_path, checks)


def collect_checks_with_triggers_from_mapping(
    mapping: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Collect all SOURCE_CHECK strategies that have static_check, with their trigger pattern.
    Returns list of (trigger_str, check_dict) where check_dict has code, target_lang, directive.
    """
    out: List[Tuple[str, Dict[str, Any]]] = []
    for entry in mapping.values() if isinstance(mapping, dict) else []:
        if not isinstance(entry, dict):
            continue
        for s in entry.get("strategies") or []:
            if not isinstance(s, dict):
                continue
            if (s.get("type") or "").strip().upper() != "SOURCE_CHECK":
                continue
            sc = s.get("static_check")
            if not sc or not (sc.get("code") or "").strip():
                continue
            trigger = (s.get("trigger") or "*").strip()
            if not trigger:
                trigger = "*"
            check_dict = {
                "code": sc.get("code", ""),
                "target_lang": sc.get("target_lang", "text"),
                "directive": s.get("directive", ""),
            }
            out.append((trigger, check_dict))
    return out


# Top-level dirs to skip when expanding triggers (same as repo_structure)
_IGNORE_TOPLEVEL = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".tox", "dist", "build", ".next", ".contextcov"})


def _is_ignored(path: Path, root: Path) -> bool:
    """
    True if path lies inside a directory we never check (a virtualenv, vendored
    dependencies, build output, or ContextCov's own installed runtime).

    Trigger globs are resolved with Path.glob, which does not honour any skip
    list, so without this a trigger like "**/*.py" sweeps the user's venv and
    node_modules along with their source.
    """
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part in _IGNORE_TOPLEVEL for part in parts[:-1])


def _top_level_dirs(root: Path) -> List[str]:
    """Return top-level directory names under root (excluding ignored)."""
    root = root.resolve()
    if not root.is_dir():
        return []
    out = []
    try:
        for p in root.iterdir():
            if p.is_dir() and p.name not in _IGNORE_TOPLEVEL and not p.name.startswith("."):
                out.append(p.name)
    except OSError:
        pass
    return sorted(out)


def _expand_brace_pattern(pattern: str) -> List[str]:
    """
    Expand a single glob pattern that may contain {a,b,c} into a list of patterns
    (pathlib glob on some systems does not support braces). Only expands one brace set.
    """
    import re
    m = re.search(r"\{([^{}]+)\}", pattern)
    if not m:
        return [pattern]
    options = [x.strip() for x in m.group(1).split(",") if x.strip()]
    if not options:
        return [pattern]
    out = []
    for opt in options:
        out.append(pattern[: m.start()] + opt + pattern[m.end() :])
    return out


def _expand_extended_glob_optional(pattern: str) -> List[str]:
    """
    Expand ?(x) extended-glob style (0 or 1 occurrence of x) into pathlib-compatible
    patterns. E.g. "*.ts?(x)" -> ["*.ts", "*.tsx"]. Only expands one ?(...) per call.
    """
    import re
    # Match ?( ... ) - content inside is the optional part (no nested parens)
    m = re.search(r"\?\(([^()]*)\)", pattern)
    if not m:
        return [pattern]
    optional = m.group(1)
    before = pattern[: m.start()]
    after = pattern[m.end() :]
    return [before + after, before + optional + after]


def _normalize_trigger_pattern(pattern: str) -> List[str]:
    """
    Normalize a single pattern for pathlib: expand ?(...) then braces.
    Returns a list of pathlib-safe patterns.
    """
    out = [pattern]
    # Expand extended ?(...) first (may produce 2 patterns per ?(...))
    while True:
        expanded = []
        for p in out:
            expanded.extend(_expand_extended_glob_optional(p))
        if expanded == out:
            break
        out = expanded
    # Then expand braces
    result = []
    for p in out:
        result.extend(_expand_brace_pattern(p))
    return result


def _split_comma_outside_braces(s: str) -> List[str]:
    """Split by comma only when comma is outside {...}; preserves brace expressions like *.{a,b}."""
    out: List[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(s):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(s[start:i].strip())
            start = i + 1
    out.append(s[start:].strip())
    return [x for x in out if x]


def _trigger_to_pattern_list(trigger: str) -> List[str]:
    """Split trigger into list of patterns; | and comma (outside braces) as alternation. E.g. *.tsx,*.jsx or *.{ts,tsx}."""
    parts: List[str] = []
    for p in (x.strip() for x in trigger.split("|") if x.strip()):
        parts.extend(_split_comma_outside_braces(p))
    return parts


def _trigger_patterns_to_files(root: Path, trigger: str) -> List[Path]:
    """
    Resolve a trigger string (e.g. "src/**/*.py" or "README.md|docs/**/*.md") to a list of
    file paths under root. Trigger can contain | or comma for alternation (e.g. "*.tsx,*.jsx").
    Normalizes extended glob ?(...) to pathlib patterns (e.g. *.ts?(x) -> *.ts and *.tsx) and
    expands {a,b,c} braces. Uses pathlib glob.
    """
    seen: set = set()
    files: List[Path] = []
    root = root.resolve()
    for part in _trigger_to_pattern_list(trigger):
        for sub in _normalize_trigger_pattern(part):
            try:
                for path in root.glob(sub):
                    if path.is_file() and path not in seen and not _is_ignored(path, root):
                        seen.add(path)
                        files.append(path)
            except Exception:
                continue
    return files


def _expand_trigger_under_prefixes(root: Path, trigger: str) -> List[Path]:
    """
    When a trigger matches 0 files, try prepending each top-level directory so
    e.g. "frontend/**/*.tsx" also matches "packages/frontend/**/*.tsx".
    Returns deduplicated list of files from any expanded pattern that matched.
    """
    seen: set = set()
    files: List[Path] = []
    top_dirs = _top_level_dirs(root)
    for d in top_dirs:
        # Prepend prefix to each part of the trigger (support | and comma alternation)
        parts = _trigger_to_pattern_list(trigger)
        expanded_parts = [f"{d}/{p}" for p in parts]
        expanded_trigger = "|".join(expanded_parts)
        for path in _trigger_patterns_to_files(root, expanded_trigger):
            if path not in seen:
                seen.add(path)
                files.append(path)
    return files


def _expand_trigger_anywhere(root: Path, trigger: str) -> List[Path]:
    """
    When a trigger still matches 0 after prefix expansion, try matching it anywhere
    in the tree so e.g. "src/**/*.tsx" matches "packages/frontend/src/**/*.tsx".
    Uses pattern **/<trigger_part> for each part (after ?(...) and brace normalization).
    """
    seen: set = set()
    files: List[Path] = []
    root = root.resolve()
    for part in _trigger_to_pattern_list(trigger):
        for sub in _normalize_trigger_pattern(part):
            try:
                anywhere = "**/" + sub
                for path in root.glob(anywhere):
                    if path.is_file() and path not in seen and not _is_ignored(path, root):
                        seen.add(path)
                        files.append(path)
            except Exception:
                continue
    return files


def run_source_checks_for_repo(
    root: Path,
    mapping: Dict[str, Dict[str, Any]],
    *,
    encoding: str = "utf-8",
    return_stats: bool = False,
    expand_triggers: bool = True,
    source_scope: str = "full",
    diff_text: str | None = None,
) -> List[Tuple[Path, str, Dict[str, Any]]] | Tuple[
    List[Tuple[Path, str, Dict[str, Any]]], Dict[str, Any]
]:
    """
    Run all SOURCE_CHECK strategies from the mapping on the repo. Resolves each strategy's
    trigger to files under root, runs the check on each file, and returns violations.
    If expand_triggers is True (default), triggers that match 0 files are retried with each
    top-level directory prepended (e.g. frontend/** -> packages/frontend/**) for precision.
    If return_stats=True, returns (violations, stats) where stats has checks_loaded,
    trigger_file_counts, trigger_expanded (set of triggers that used expansion), files_checked.

    source_scope
        ``"full"`` (default): unchanged — all matching files, full file content per check.
        ``"diff"``: only files that appear in ``diff_text`` (unified git diff), and each check
        receives only the **lines added/changed in the new version** (see ``line_subset`` in
        :func:`run_static_checks`). Requires ``diff_text``.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        out: List[Tuple[Path, str, str]] = [(root, "", "Not a directory")]
        return (out, {"checks_loaded": 0, "trigger_file_counts": {}, "trigger_expanded": set(), "trigger_expanded_anywhere": set(), "files_checked": 0}) if return_stats else out

    trigger_check_pairs = collect_checks_with_triggers_from_mapping(mapping)
    if not trigger_check_pairs:
        return ([], {"checks_loaded": 0, "trigger_file_counts": {}, "trigger_expanded": set(), "trigger_expanded_anywhere": set(), "files_checked": 0}) if return_stats else []

    changed_map: dict[str, set[int]] = {}
    if source_scope == "diff":
        if not (diff_text or "").strip():
            empty_stats = {
                "checks_loaded": len(trigger_check_pairs),
                "trigger_file_counts": {},
                "trigger_expanded": set(),
                "trigger_expanded_anywhere": set(),
                "files_checked": 0,
                "per_check_file_counts": {},
            }
            return ([], empty_stats) if return_stats else []
        from src.diff_regions import (
            normalize_changed_paths_for_root,
            parse_unified_diff_new_line_numbers,
        )

        changed_map = normalize_changed_paths_for_root(
            root, parse_unified_diff_new_line_numbers(diff_text)
        )

    # Map each file to the list of checks that apply (by trigger)
    file_to_checks: Dict[Path, List[Dict[str, Any]]] = defaultdict(list)
    seen_checks_per_file: Dict[Path, set] = defaultdict(set)
    trigger_file_counts: Dict[str, int] = defaultdict(int)
    trigger_expanded: set = set()

    trigger_expanded_anywhere: set = set()
    # Per-check "sites": number of unique files each directive ran on.
    # (We use directive as the stable key for reporting.)
    per_check_file_sets: Dict[str, set] = defaultdict(set)
    for trigger, check_dict in trigger_check_pairs:
        files_for_trigger = _trigger_patterns_to_files(root, trigger)
        if not files_for_trigger and expand_triggers:
            files_for_trigger = _expand_trigger_under_prefixes(root, trigger)
            if files_for_trigger:
                trigger_expanded.add(trigger)
        if not files_for_trigger and expand_triggers:
            files_for_trigger = _expand_trigger_anywhere(root, trigger)
            if files_for_trigger:
                trigger_expanded_anywhere.add(trigger)
        trigger_file_counts[trigger] = len(files_for_trigger)
        directive = (check_dict.get("directive") or "").strip()
        if directive:
            # Track file sites per directive (union over triggers).
            per_check_file_sets[directive].update(files_for_trigger)
        for path in files_for_trigger:
            key = (check_dict.get("code"), check_dict.get("directive"))
            if key in seen_checks_per_file[path]:
                continue
            seen_checks_per_file[path].add(key)
            file_to_checks[path].append(check_dict)

    _SKIPPED_NO_PARSER.clear()
    violations: List[Tuple[Path, str, Dict[str, Any]]] = []
    for path in sorted(file_to_checks.keys()):
        checks = file_to_checks[path]
        if not checks:
            continue
        line_subset: Optional[set[int]] = None
        if source_scope == "diff":
            try:
                rel = path.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            line_subset = changed_map.get(rel)
            if not line_subset:
                continue
        for directive, vdict in run_static_checks(
            path, checks, encoding=encoding, line_subset=line_subset
        ):
            violations.append((path, directive, vdict))

    ast_backend_ok = tree_sitter_backend_ok()
    stats = {
        "checks_loaded": len(trigger_check_pairs),
        "trigger_file_counts": dict(trigger_file_counts),
        "trigger_expanded": trigger_expanded,
        "trigger_expanded_anywhere": trigger_expanded_anywhere,
        "files_checked": len(file_to_checks),
        "per_check_file_counts": {k: len(v) for k, v in per_check_file_sets.items()},
        "ast_backend_ok": ast_backend_ok,
        "ast_backend_error": tree_sitter_backend_error(),
        "checks_skipped_no_parser": len(_SKIPPED_NO_PARSER),
    }
    if not ast_backend_ok and _SKIPPED_NO_PARSER:
        print(
            f"[contextcov] WARNING: tree-sitter is installed but unusable "
            f"({tree_sitter_backend_error()}). Skipped {len(_SKIPPED_NO_PARSER)} "
            f"AST-based check run(s) instead of passing them silently. "
            f"Fix with: pip install 'tree-sitter>=0.20,<0.22' 'tree-sitter-languages==1.10.2'",
            file=sys.stderr,
            flush=True,
        )
    if return_stats:
        return (violations, stats)
    return violations
