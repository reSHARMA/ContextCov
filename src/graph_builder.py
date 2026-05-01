"""
Dependency Graph Builder: scans a directory tree and builds a directed graph of file imports.

Nodes = file paths (relative to root_dir). Edges = "file A imports file B".
Used by ARCH_DETERMINISTIC checks to validate layering, cycles, and forbidden edges.

Python: ast module. TypeScript/JavaScript: regex for import/require.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore

# Default dirs to skip when walking
DEFAULT_IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "env", ".contextcov"}

# File extensions we parse for imports
PY_EXTENSIONS = {".py", ".pyi"}
JS_EXTENSIONS = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}

# Config/doc extensions added as nodes only (no edges) so "file must exist" arch checks can see them
CONFIG_EXTENSIONS = {".yml", ".yaml", ".md"}

# Regex for JS/TS: import x from 'path', import 'path', require('path')
JS_IMPORT_PATTERN = re.compile(
    r'''(?:import\s+.*\s+from\s+|import\s+)\s*['"]([^'"]+)['"]'''
)
JS_REQUIRE_PATTERN = re.compile(
    r'''require\s*\(\s*['"]([^'"]+)['"]\s*\)'''
)


def _source_roots(root_dir: str) -> List[str]:
    """
    Directories an absolute first-party import may be rooted at: the repo itself
    plus the conventional source dirs. Used to resolve `from core.models import X`,
    which is how most projects import their own code.
    """
    roots = [root_dir]
    for name in ("src", "lib"):
        candidate = os.path.join(root_dir, name)
        if os.path.isdir(candidate):
            roots.append(candidate)
    return roots


def _resolve_dotted_path(dotted: str, root_dir: str) -> Optional[str]:
    """
    Resolve a dotted module path (e.g. "core.models") against the repo's source
    roots, longest prefix first, to a file relative to root_dir. Returns None if
    it does not correspond to a file inside the repo (stdlib, third-party).
    """
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        return None
    for base in _source_roots(root_dir):
        for end in range(len(parts), 0, -1):
            target = os.path.normpath(os.path.join(base, *parts[:end]))
            # Never let a crafted module name escape the repo.
            if os.path.relpath(target, root_dir).startswith(".."):
                continue
            for candidate in (target + ".py", os.path.join(target, "__init__.py")):
                if os.path.isfile(candidate):
                    return os.path.relpath(candidate, str(root_dir)).replace("\\", "/")
    return None


def _resolve_python_import(
    node: ast.AST,
    from_file: str,
    root_dir: str,
) -> Optional[str]:
    """
    Resolve an ast Import or ImportFrom to a target file path (relative to root_dir).
    Returns None if we cannot resolve (e.g. standard library, third-party).
    """
    from_dir = os.path.dirname(from_file)
    if from_dir:
        from_abs = os.path.normpath(os.path.join(root_dir, from_dir))
    else:
        from_abs = root_dir

    if isinstance(node, ast.Import):
        # import foo / import foo.bar -> resolve the full dotted path, not just
        # its first segment, so `import core.models` binds to core/models.py.
        for alias in node.names:
            name = alias.name
            if name.startswith("_"):
                continue
            # Resolve the full dotted path first so `import core.models` binds to
            # core/models.py rather than to core/__init__.py.
            resolved = _resolve_dotted_path(name, root_dir)
            if resolved:
                return resolved
            # Fall back to a sibling of the importing file (implicit-relative style).
            head = name.split(".")[0]
            for ext in (".py", ""):
                candidate = os.path.join(from_abs, head + ext)
                if os.path.isfile(candidate):
                    return os.path.relpath(candidate, str(root_dir)).replace("\\", "/")
                init = os.path.join(from_abs, head, "__init__.py")
                if os.path.isfile(init):
                    return os.path.relpath(init, str(root_dir)).replace("\\", "/")
        return None

    if isinstance(node, ast.ImportFrom):
        level = getattr(node, "level", 0) or 0
        module = getattr(node, "module", None) or ""
        if level == 0 and not module:
            return None
        if level > 0:
            # Relative: .utils or ..utils
            parts = from_dir.replace("\\", "/").split("/") if from_dir else []
            up = level - 1
            if module:
                parts = parts[: max(0, len(parts) - up)] + module.split(".")
            else:
                parts = parts[: max(0, len(parts) - up)]
            if not parts:
                return None
            target_abs = os.path.normpath(os.path.join(root_dir, *parts))
        else:
            # Absolute import (e.g. from core.models import x). Resolve against the
            # repo's source roots; returns None for stdlib/third-party modules.
            return _resolve_dotted_path(module, root_dir)
        # Resolve to file
        if os.path.isfile(target_abs + ".py"):
            return os.path.relpath(target_abs + ".py", str(root_dir)).replace("\\", "/")
        if os.path.isfile(os.path.join(target_abs, "__init__.py")):
            return os.path.relpath(os.path.join(target_abs, "__init__.py"), str(root_dir)).replace("\\", "/")
        return None
    return None


def _collect_python_imports(file_path: str, root_dir: str) -> List[str]:
    """Return list of resolved file paths (relative) that this Python file imports."""
    targets = []
    full_path = os.path.join(root_dir, file_path)
    if not os.path.isfile(full_path):
        return targets
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, UnicodeDecodeError):
        return targets
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            t = _resolve_python_import(node, file_path, root_dir)
            if t and t != file_path:
                targets.append(t)
    return targets


def _resolve_js_import(spec: str, from_file: str, root_dir: str) -> Optional[str]:
    """Resolve JS/TS import spec to a file path relative to root_dir. Returns None if external."""
    if spec.startswith(".") or spec.startswith("/"):
        pass
    else:
        # Node module or alias - skip
        return None
    from_dir = os.path.dirname(from_file)
    from_abs = os.path.normpath(os.path.join(root_dir, from_dir))
    target_abs = os.path.normpath(os.path.join(from_abs, spec))
    # Try with extensions
    for ext in ("", ".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"):
        candidate = target_abs + ext if ext else target_abs
        if os.path.isfile(candidate):
            return os.path.relpath(candidate, root_dir).replace("\\", "/")
    if os.path.isfile(target_abs):
        return os.path.relpath(target_abs, root_dir).replace("\\", "/")
    return None


def _collect_js_imports(file_path: str, root_dir: str) -> List[str]:
    """Return list of resolved file paths that this JS/TS file imports."""
    targets = []
    full_path = os.path.join(root_dir, file_path)
    if not os.path.isfile(full_path):
        return targets
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return targets
    for pattern in (JS_IMPORT_PATTERN, JS_REQUIRE_PATTERN):
        for m in pattern.finditer(content):
            spec = m.group(1).strip()
            t = _resolve_js_import(spec, file_path, root_dir)
            if t and t != file_path:
                targets.append(t)
    return targets


def build_graph(
    root_dir: str | Path,
    *,
    ignore_dirs: Optional[Set[str]] = None,
    extensions: Optional[Set[str]] = None,
) -> Any:
    """
    Build a directed graph of file imports under root_dir.

    Returns a networkx DiGraph with nodes = file paths (relative), edges = imports.
    If networkx is not installed, returns None.
    """
    if nx is None:
        return None
    root_dir = Path(root_dir).resolve()
    ignore = ignore_dirs or DEFAULT_IGNORE_DIRS
    exts = extensions or (PY_EXTENSIONS | JS_EXTENSIONS)

    G = nx.DiGraph()
    code_exts = PY_EXTENSIONS | JS_EXTENSIONS
    for root, dirs, files in os.walk(root_dir):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore]
        rel_root = os.path.relpath(root, root_dir)
        if rel_root == ".":
            rel_root = ""
        for f in files:
            path = Path(f)
            suffix = path.suffix.lower()
            file_rel = (os.path.join(rel_root, f) if rel_root else f).replace("\\", "/")
            # Add ALL files as nodes so arch checks can verify file existence
            G.add_node(file_rel)
            # Only parse imports for code files
            if suffix not in code_exts:
                continue
            if suffix in PY_EXTENSIONS:
                targets = _collect_python_imports(file_rel, str(root_dir))
            elif suffix in JS_EXTENSIONS:
                targets = _collect_js_imports(file_rel, str(root_dir))
            else:
                continue
            for t in targets:
                if t and t != file_rel:
                    G.add_edge(file_rel, t)
    return G
