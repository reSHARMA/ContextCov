"""
Arch Check Runner: runs deterministic (graph) and semantic (LLM-on-diff) checks.

Deterministic: load graph, exec generated Python with graph/nx, collect violations.
Semantic: build one prompt with all semantic rubrics + diff, call LLM, parse violations.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore


def run_deterministic_checks(
    graph: Any,
    checks: List[Dict[str, Any]],
) -> List[str]:
    """
    Run ARCH_DETERMINISTIC checks on the dependency graph.
    Each check must have "code". Returns list of violation messages.
    """
    if nx is None:
        return ["networkx not installed; cannot run deterministic arch checks"]
    violations = []
    for i, check in enumerate(checks):
        code = (check.get("code") or "").strip()
        if not code:
            continue
        context = {"graph": graph, "nx": nx, "result": (True, "")}
        try:
            exec(code, context)
            passed, msg = context.get("result", (True, ""))
            if not passed:
                violations.append(f"[Arch Violation] {msg}")
        except SyntaxError as e:
            violations.append(
                f"[Arch Check Error] check_{i}: generated code invalid (syntax). "
                f"Use break instead of return in loops. Detail: {e}"
            )
        except Exception as e:
            violations.append(f"[Arch Check Error] check_{i}: {e}")
    return violations


def run_deterministic_checks_per_check(
    graph: Any,
    checks: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], List[str]]]:
    """
    Run ARCH_DETERMINISTIC checks on the dependency graph, one at a time.
    Returns list of (check_dict, violation_messages) for each check.
    check_dict includes "code" and "directive".
    """
    if nx is None:
        return [({"directive": ""}, ["networkx not installed; cannot run deterministic arch checks"])]
    results: List[Tuple[Dict[str, Any], List[str]]] = []
    for check in checks:
        code = (check.get("code") or "").strip()
        directive = (check.get("directive") or "").strip()
        check_info = {"code": code, "directive": directive}
        if not code:
            results.append((check_info, []))
            continue
        context = {"graph": graph, "nx": nx, "result": (True, "")}
        violations: List[str] = []
        try:
            exec(code, context)
            passed, msg = context.get("result", (True, ""))
            if not passed:
                violations.append(f"[Arch Violation] {msg}")
        except SyntaxError as e:
            violations.append(
                f"[Arch Check Error] generated code invalid (syntax). "
                f"Use break instead of return in loops. Detail: {e}"
            )
        except Exception as e:
            violations.append(f"[Arch Check Error] {e}")
        results.append((check_info, violations))
    return results


def collect_deterministic_checks_from_mapping(mapping: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten ARCH_DETERMINISTIC strategies that have arch_deterministic_check.code."""
    checks = []
    for entry in mapping.values() if isinstance(mapping, dict) else []:
        if not isinstance(entry, dict):
            continue
        for s in entry.get("strategies") or []:
            if not isinstance(s, dict):
                continue
            if (s.get("type") or "").strip().upper() != "ARCH_DETERMINISTIC":
                continue
            ac = s.get("arch_deterministic_check")
            if not ac or not (ac.get("code") or "").strip():
                continue
            checks.append({"code": ac.get("code", ""), "directive": s.get("directive", "")})
    return checks


def get_git_diff(root: str | Path, staged_only: bool = True) -> str:
    """Return git diff (staged by default) as string. Empty if not a git repo or no diff."""
    root = Path(root)
    try:
        cmd = ["git", "diff", "--cached"] if staged_only else ["git", "diff"]
        out = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode != 0:
            return ""
        return out.stdout or ""
    except Exception:
        return ""


# Extensions to include in full-repo source snapshot (code only)
_SOURCE_SNAPSHOT_EXTENSIONS = {".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}

# Cap size and file count so we stay within LLM context
_DEFAULT_SNAPSHOT_MAX_CHARS = 200_000
_DEFAULT_SNAPSHOT_MAX_FILES = 400


def get_full_source_snapshot(
    root: str | Path,
    *,
    max_chars: int = _DEFAULT_SNAPSHOT_MAX_CHARS,
    max_files: int = _DEFAULT_SNAPSHOT_MAX_FILES,
) -> str:
    """
    Build a concatenated snapshot of source files under root for LLM review.
    Respects .gitignore-derived ignore dirs. Includes code files only; truncates
    when total length or file count exceeds limits. Returns a string with
    "--- path/to/file ---" headers and file contents.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        return ""

    try:
        from src.repo_structure import get_ignore_dirs
    except ImportError:
        return ""

    ignore = get_ignore_dirs(root)
    parts: list[str] = []
    total_chars = 0
    nfiles = 0

    for dir_path, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore]
        rel_dir = os.path.relpath(dir_path, root)
        if rel_dir == ".":
            rel_dir = ""
        for f in filenames:
            if nfiles >= max_files or total_chars >= max_chars:
                break
            ext = Path(f).suffix.lower()
            if ext not in _SOURCE_SNAPSHOT_EXTENSIONS:
                continue
            file_rel = (os.path.join(rel_dir, f) if rel_dir else f).replace("\\", "/")
            full = os.path.join(dir_path, f)
            try:
                raw = Path(full).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(raw) > 50_000:
                raw = raw[:50_000] + "\n... (truncated)\n"
            block = f"--- {file_rel} ---\n{raw}\n"
            if total_chars + len(block) > max_chars:
                block = block[: max_chars - total_chars] + "\n... (snapshot truncated)\n"
            parts.append(block)
            total_chars += len(block)
            nfiles += 1
        if nfiles >= max_files or total_chars >= max_chars:
            break

    if not parts:
        return ""
    out = "\n".join(parts)
    if total_chars >= max_chars or nfiles >= max_files:
        out += f"\n... (snapshot limited to {nfiles} files, ~{total_chars} chars)"
    return out


def run_semantic_checks(
    content: str,
    checks: List[Dict[str, Any]],
    *,
    client: Any,
    model: str,
    provider: str = "openai",
    temperature: float = 0.0,
    content_label: str = "Git Diff",
) -> List[str]:
    """
    Run ARCH_SEMANTIC checks by sending one LLM prompt: rules + content (diff or full source).
    Each check must have "description". content_label is used in the prompt (e.g. "Git Diff"
    or "Source code snapshot"). Returns list of violation strings.
    """
    if not checks:
        return []
    rules_text = "\n".join(f"- {c.get('description', '').strip() or 'Review for general quality'}" for c in checks)
    section_title = "DIFF" if content_label == "Git Diff" else "SOURCE SNAPSHOT"
    content_placeholder = "(no diff - empty or not a git repo)" if content_label == "Git Diff" else "(empty)"
    context_note = ""
    if content_label == "Source code snapshot":
        context_note = "\nThis is a complete snapshot of the repository source code (not a git diff or a pull request). Do not apply rules that require a PR description, PR title, PR checklist, or other PR-only context; only flag violations that apply to the code and structure shown.\n\n"
    prompt = f"""You are a Code Reviewer. Analyze the following {content_label} against these rules.{context_note}
RULES:
{rules_text}

{section_title}:
{content.strip() or content_placeholder}

If any rule is violated, list each violation. If all rules are satisfied, output no violations.
OUTPUT JSON only, no markdown:
{{ "violations": ["Rule X violated: explanation..."] }}
If no violations: {{ "violations": [] }}"""

    from llm import chat_completion_create

    try:
        resp = chat_completion_create(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = ""
        if getattr(resp, "choices", None) and len(resp.choices) > 0:
            msg = getattr(resp.choices[0], "message", None)
            if msg is not None:
                content = (getattr(msg, "content", None) or "").strip()
        if not content:
            return []
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        data = json.loads(text)
        violations = data.get("violations") or []
        return [v if isinstance(v, str) else str(v) for v in violations]
    except Exception as e:
        # Do not fail-open as a violation: an LLM/transport error is not a code
        # finding, and feeding it back to the agent as if it were a constraint
        # poisons the followup loop. Surface on stderr for visibility instead.
        import sys
        print(f"[arch_sem] LLM call failed (returning no violations): {e}", file=sys.stderr, flush=True)
        return []


def collect_semantic_checks_from_mapping(mapping: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten ARCH_SEMANTIC strategies that have arch_semantic_check.description."""
    checks = []
    for entry in mapping.values() if isinstance(mapping, dict) else []:
        if not isinstance(entry, dict):
            continue
        for s in entry.get("strategies") or []:
            if not isinstance(s, dict):
                continue
            if (s.get("type") or "").strip().upper() != "ARCH_SEMANTIC":
                continue
            ac = s.get("arch_semantic_check")
            if not ac or not (ac.get("description") or "").strip():
                continue
            checks.append({"description": ac.get("description", ""), "directive": s.get("directive", "")})
    return checks
