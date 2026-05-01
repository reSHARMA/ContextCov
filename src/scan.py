"""
ContextCov Scan: architectural gatekeeper.

Builds dependency graph, runs ARCH_DETERMINISTIC checks (graph), then ARCH_SEMANTIC checks
(LLM on git diff). Use on commit (hook) or explicitly: python -m src.scan
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root on path when run as python -m src.scan
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.graph_builder import build_graph
from src.arch_runner import (
    run_deterministic_checks,
    run_semantic_checks,
    get_git_diff,
    get_full_source_snapshot,
    collect_deterministic_checks_from_mapping,
    collect_semantic_checks_from_mapping,
)
from src.static_runner import run_source_checks_for_repo
from src.repo_clone import clone_repo, DEFAULT_DATA_DIR
from src.repo_structure import get_ignore_dirs, minified_file_structure


def _get_llm_client(prefix: str):
    from llm import get_model_and_client, get_provider_and_model_from_env
    client, model, temp = get_model_and_client(prefix)
    prov_result = get_provider_and_model_from_env(prefix)
    provider = prov_result[0] if not isinstance(prov_result[0], list) else prov_result[0][0]
    return client, model, temp, provider


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ContextCov architectural checks (SOURCE_CHECK, ARCH_DETERMINISTIC, ARCH_SEMANTIC).",
        epilog="Example: python -m src.scan --mapping owner_repo_AGENTS.mapping.json --root /path/to/repo",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=os.environ.get("CONTEXTCOV_MAPPING", ""),
        help="Path to mapping JSON (default: CONTEXTCOV_MAPPING or first *.mapping.json in --root)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Repository root to scan (default: current directory, or .data/<owner>_<repo> when --repo is set)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        metavar="OWNER/REPO",
        default="",
        help="Clone OWNER/REPO into .data and run checks there (e.g. Significant-Gravitas/AutoGPT)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Parent directory for cloned repos (default: .data)",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Branch to clone when using --repo (default: main)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip semantic (LLM) checks; only run deterministic graph checks",
    )
    parser.add_argument(
        "--diff-unstaged",
        action="store_true",
        help="Use full diff (unstaged) for semantic checks instead of staged only",
    )
    parser.add_argument(
        "--only-source",
        action="store_true",
        help="Run only SOURCE_CHECK (static) checks; skip ARCH_DETERMINISTIC and ARCH_SEMANTIC",
    )
    parser.add_argument(
        "--skip-source",
        action="store_true",
        help="Skip SOURCE_CHECK; run only ARCH_DETERMINISTIC and ARCH_SEMANTIC (arch checks)",
    )
    parser.add_argument(
        "--print-file-structure",
        action="store_true",
        help="Print minified repo file structure (paths relative to root) and exit. Use to verify what would be sent to SOURCE_CHECK generator.",
    )
    args = parser.parse_args()

    if args.repo.strip():
        owner_repo = args.repo.strip()
        if "/" not in owner_repo or owner_repo.count("/") != 1:
            print("Error: --repo must be OWNER/REPO (e.g. Significant-Gravitas/AutoGPT)", file=sys.stderr)
            return 1
        owner, repo = owner_repo.split("/", 1)
        cwd = Path.cwd()
        data_dir = cwd / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
        print(f"Cloning {owner}/{repo} into {data_dir}...", file=sys.stderr)
        root = clone_repo(owner, repo, data_dir, branch=args.branch)
        print(f"Using repo root: {root}", file=sys.stderr)
    else:
        root = Path(args.root or ".").resolve()
        if not root.is_dir():
            print(f"Error: root is not a directory: {root}", file=sys.stderr)
            return 1

    mapping_path = args.mapping.strip()
    if not mapping_path:
        # Look in cwd for mapping when repo was cloned (root is .data/...)
        search_root = root if (args.repo.strip() and root.exists()) else Path.cwd()
        for f in sorted(search_root.glob("*.mapping.json")):
            mapping_path = str(f)
            break
    if not mapping_path:
        print("Error: no --mapping and no *.mapping.json in root. Set CONTEXTCOV_MAPPING or pass --mapping.", file=sys.stderr)
        return 1
    mapping_path = Path(mapping_path)
    if not mapping_path.is_absolute():
        # Mapping file is relative to cwd (where user runs the command)
        mapping_path = (Path.cwd() / mapping_path).resolve()
    if not mapping_path.is_file():
        print(f"Error: mapping file not found: {mapping_path}", file=sys.stderr)
        return 1

    if args.print_file_structure:
        structure = minified_file_structure(root)
        print(f"# File structure (relative to root: {root}), {len(structure.splitlines())} paths:", file=sys.stderr)
        print(structure)
        return 0

    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)

    det_checks = collect_deterministic_checks_from_mapping(mapping)
    sem_checks = collect_semantic_checks_from_mapping(mapping)
    has_source = any(
        (s.get("type") or "").strip().upper() == "SOURCE_CHECK" and (s.get("static_check") or {}).get("code")
        for e in (mapping.values() if isinstance(mapping, dict) else [])
        if isinstance(e, dict)
        for s in (e.get("strategies") or [])
        if isinstance(s, dict)
    )

    if not det_checks and not sem_checks and not has_source:
        print("No checks in mapping (SOURCE_CHECK, ARCH_DETERMINISTIC, or ARCH_SEMANTIC). All good.")
        return 0

    any_failed = False

    # 1. Run SOURCE_CHECK (static checks on files)
    if has_source and not args.skip_source:
        print("Running source checks...", file=sys.stderr)
        source_violations, source_stats = run_source_checks_for_repo(root, mapping, return_stats=True)
        print(f"  Loaded {source_stats['checks_loaded']} SOURCE_CHECK(s), scanned {source_stats['files_checked']} file(s).", file=sys.stderr)
        expanded = source_stats.get("trigger_expanded") or set()
        expanded_anywhere = source_stats.get("trigger_expanded_anywhere") or set()
        for trig, count in sorted(source_stats["trigger_file_counts"].items(), key=lambda x: (-x[1], x[0])):
            zero = " (no files matched)" if count == 0 else ""
            exp = " (expanded under top-level dirs)" if trig in expanded else ""
            if trig in expanded_anywhere:
                exp = " (expanded **/trigger anywhere)"
            print(f"    trigger {trig!r}: {count} file(s){zero}{exp}", file=sys.stderr)
        if source_violations:
            any_failed = True
            print("Build Failed (Source):", file=sys.stderr)
            for path, directive, vdict in source_violations:
                try:
                    rel = path.relative_to(root)
                except ValueError:
                    rel = path
                msg = vdict.get("message", "") if isinstance(vdict, dict) else str(vdict)
                # message falls back to the directive when a check sets none, so
                # print the rule once rather than twice.
                if msg and msg.strip() != directive.strip():
                    print(f"  {rel}: {directive}", file=sys.stderr)
                    print(f"      {msg}", file=sys.stderr)
                else:
                    print(f"  {rel}: {directive}", file=sys.stderr)
        else:
            print("  Source checks passed.", file=sys.stderr)

    if args.only_source:
        print("Source checks passed (--only-source)." if not any_failed else "Source checks had failures.")
        return 1 if any_failed else 0

    # 2. Build graph (use .gitignore-derived ignore dirs when available)
    print("Building dependency graph...", file=sys.stderr)
    ignore_dirs = get_ignore_dirs(root)
    G = build_graph(root, ignore_dirs=set(ignore_dirs))
    if G is None:
        print("Warning: networkx or graph build failed; skipping deterministic checks.", file=sys.stderr)
    else:
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.", file=sys.stderr)

    # 3. Run deterministic checks
    if G is not None and det_checks:
        violations = run_deterministic_checks(G, det_checks)
        if violations:
            any_failed = True
            print("Build Failed (Deterministic):", file=sys.stderr)
            for v in violations:
                print(v, file=sys.stderr)
        else:
            print("  Deterministic arch checks passed.", file=sys.stderr)

    # 4. Run semantic checks (on diff if present, else on full repo source)
    if sem_checks and not args.no_semantic:
        diff = get_git_diff(root, staged_only=not args.diff_unstaged)
        if diff and diff.strip():
            content = diff
            content_label = "Git Diff"
            print("Running semantic review (on diff)...", file=sys.stderr)
        else:
            content = get_full_source_snapshot(root)
            content_label = "Source code snapshot"
            print("Running semantic review (on full repo source, no diff)...", file=sys.stderr)
        try:
            client, model, temp, provider = _get_llm_client("ARCH_SEMANTIC")
        except Exception:
            client, model, temp, provider = _get_llm_client("DEFAULT")
        sem_violations = run_semantic_checks(
            content,
            sem_checks,
            client=client,
            model=model,
            provider=provider,
            temperature=temp,
            content_label=content_label,
        )
        if sem_violations:
            any_failed = True
            print("Build Failed (Semantic):", file=sys.stderr)
            for v in sem_violations:
                print(v, file=sys.stderr)
        else:
            print("  Semantic arch checks passed.", file=sys.stderr)

    if any_failed:
        print("One or more check stages failed (see above).", file=sys.stderr)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
