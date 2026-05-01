"""
CLI for context-aware markdown analysis.

Usage:
  python -m src.cli raw/owner/repo/CLAUDE.md
  python -m src.cli raw/owner/repo/path/to/file.md [--branch main] [--no-compress]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

# Ensure project root is on path when run as python -m src.cli
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.input_parser import (
    GitHubFileInput,
    fetch_raw_content,
    parse_github_file_path,
)
from src.markdown_chunks import parse_markdown_to_scoped_segments
from src.contextual_compression import compress_segments
from src.stable_id import assign_stable_ids, build_compression_mapping
from src.router import route_constraints_parallel, strategies_to_dict_list
from src.static_check_generator import generate_static_checks_for_mapping
from src.process_check_generator import generate_process_checks_for_mapping
from src.arch_deterministic_generator import generate_arch_deterministic_checks_for_mapping
from src.arch_semantic_generator import generate_arch_semantic_checks_for_mapping


def _spinner(message: str, done_message: str | None = None):
    """
    Context manager: show an animated spinner on stderr while the block runs.
    If done_message is set, print it on exit (replacing the spinner line).
    """
    _stop = threading.Event()
    _done_msg = done_message

    def _run():
        chars = "|/-\\"
        i = 0
        while not _stop.wait(0.1):
            c = chars[i % len(chars)]
            print(f"\r{message} {c}  ", end="", file=sys.stderr, flush=True)
            i += 1
        if _done_msg:
            print(f"\r{_done_msg}    ", file=sys.stderr, flush=True)
        else:
            print("\r", end="", file=sys.stderr, flush=True)

    class _Ctx:
        def __enter__(self):
            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()
            return self

        def __exit__(self, *args):
            _stop.set()
            self._thread.join(timeout=1.0)

    return _Ctx()


def _source_digest(content: str) -> str:
    """SHA-256 of the instruction file the mapping was generated from."""
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _source_digest_path(mapping_path: str | Path) -> Path:
    """Sidecar recording the instruction-file digest (kept out of the mapping schema)."""
    return Path(str(mapping_path) + ".source.sha256")


def _read_source_digest(mapping_path: str | Path) -> str | None:
    path = _source_digest_path(mapping_path)
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _print_mapping_stats(mapping_path: str | Path) -> int:
    """Print strategy counts for an existing mapping. Returns a process exit code."""
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)
    n_compressed = sum(1 for e in mapping.values() if isinstance(e, dict) and e.get("compressed"))
    counts = {"SOURCE_CHECK": 0, "PROCESS_CHECK": 0, "ARCH_DETERMINISTIC": 0, "ARCH_SEMANTIC": 0}
    for e in mapping.values():
        if not isinstance(e, dict):
            continue
        for s in e.get("strategies") or []:
            if not isinstance(s, dict):
                continue
            t = (s.get("type") or "").strip().upper()
            if t in counts:
                counts[t] += 1
    print(
        f"Stats: {n_compressed} compressed segment(s) | SOURCE_CHECK: {counts['SOURCE_CHECK']} "
        f"| PROCESS_CHECK: {counts['PROCESS_CHECK']} | ARCH_DETERMINISTIC: {counts['ARCH_DETERMINISTIC']} "
        f"| ARCH_SEMANTIC: {counts['ARCH_SEMANTIC']}",
        file=sys.stderr,
    )
    return 0


def _pending_strategy_count(mapping: dict, strategy_type: str, check_key: str) -> int:
    """How many strategies of this type still need their check generated."""
    return len(
        [
            s
            for e in mapping.values()
            if isinstance(e, dict)
            for s in (e.get("strategies") or [])
            if isinstance(s, dict)
            and (s.get("type") or "").strip().upper() == strategy_type
            and not s.get(check_key)
        ]
    )


def _default_mapping_path(github_path: str) -> str:
    """Default mapping filename from input path: {owner}_{repo}_{file_stem}.mapping.json."""
    inp = parse_github_file_path(github_path)
    if inp is None:
        return "mapping.json"
    base = os.path.basename(inp.file_path)
    stem, _ = os.path.splitext(base)
    name = stem or base
    return f"{inp.owner}_{inp.repo}_{name}.mapping.json"


def _get_llm_client(prefix: str):
    """Resolve client/model/temp/provider from llm.py ({prefix}_LLM or DEFAULT_LLM)."""
    from llm import get_model_and_client, get_provider_and_model_from_env

    client, model, temp = get_model_and_client(prefix)
    prov_result = get_provider_and_model_from_env(prefix)
    provider = prov_result[0] if not isinstance(prov_result[0], list) else prov_result[0][0]
    return client, model, temp, provider


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch a markdown file from GitHub and analyze it with context-aware chunking and optional LLM compression."
    )
    parser.add_argument(
        "path",
        type=str,
        help="GitHub file path or URL, e.g. raw/owner/repo/AGENTS.md or https://github.com/owner/repo/blob/main/AGENTS.md",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Git branch (default: main)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Only output scoped segments (breadcrumb + content), skip LLM compression",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=" > ",
        help="Breadcrumb separator for display (default: ' > ')",
    )
    parser.add_argument(
        "--input-mapping",
        type=str,
        metavar="FILE",
        default=None,
        help="Use this mapping file if present (skip fetch/parse/compress). Default: {stem}.mapping.json from path",
    )
    parser.add_argument(
        "--output-mapping",
        type=str,
        metavar="FILE",
        default=None,
        help="Write mapping JSON here. Default: {stem}.mapping.json from path",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Run compression sequentially (no parallel LLM calls)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        metavar="N",
        default=None,
        help="Max parallel LLM requests (default: 5 or CONTEXTCOV_MAX_CONCURRENCY)",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        metavar="DIR",
        default=None,
        help="Path to repo root for check generation. If omitted, the repo is cloned from the GitHub path (owner/repo). Generators receive this path in the user prompt; claudecode also uses it as the subprocess working directory.",
    )
    parser.add_argument(
        "--local-readme",
        type=str,
        metavar="FILE",
        default=None,
        help="Read agent readme markdown from this file instead of fetching from GitHub (positional path still sets owner/repo for mapping name and clone).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the pipeline even if the mapping is already up to date with the instruction file. Unchanged sections are still reused; combine with --no-reuse for a full rebuild.",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Disable incremental reuse: recompress, reroute and regenerate every section instead of only the ones whose text changed.",
    )
    args = parser.parse_args()

    # Default mapping paths from input file name
    default_path = _default_mapping_path(args.path)
    input_mapping = args.input_mapping if args.input_mapping is not None else default_path
    output_mapping = args.output_mapping if args.output_mapping is not None else default_path

    inp = parse_github_file_path(args.path)
    if inp is None:
        print("Error: path must be in the form raw/owner/repo/path/to/file.md", file=sys.stderr)
        return 1

    branch = inp.branch or args.branch
    if args.local_readme:
        local_path = Path(args.local_readme).expanduser().resolve()
        if not local_path.is_file():
            print(f"Error: --local-readme not a file: {local_path}", file=sys.stderr)
            return 1
        content = local_path.read_text(encoding="utf-8", errors="replace")
        print(f"Loaded readme from {local_path} ({len(content)} chars).", file=sys.stderr)
    else:
        url = inp.raw_url(branch=branch)
        try:
            with _spinner(f"Fetching {url} ...", done_message=f"Fetched {url}"):
                content = fetch_raw_content(inp, branch=branch)
        except Exception as e:
            print(f"Error fetching file: {e}", file=sys.stderr)
            return 1

    # Reuse an existing mapping only if it was generated from this exact instruction
    # file. Reusing it after the file changed would silently enforce stale rules.
    if Path(input_mapping).is_file() and not args.force:
        recorded = _read_source_digest(input_mapping)
        if recorded is None:
            print(
                f"Mapping already exists: {input_mapping}. It predates change tracking, so it "
                f"cannot be verified against the current instruction file.",
                file=sys.stderr,
            )
            print("Re-run with --force to regenerate it.", file=sys.stderr)
            return _print_mapping_stats(input_mapping)
        if recorded == _source_digest(content):
            print(f"Mapping already exists and matches the instruction file: {input_mapping}.", file=sys.stderr)
            print("Run checks via: python -m src.scan | python -m src.setup_shims | static_runner.", file=sys.stderr)
            return _print_mapping_stats(input_mapping)
        print(
            f"Instruction file changed since {input_mapping} was generated; regenerating.",
            file=sys.stderr,
        )

    segments = parse_markdown_to_scoped_segments(content)
    print(f"Parsed {len(segments)} scoped segment(s).", file=sys.stderr)

    if args.no_compress:
        for seg in segments:
            bc = seg.breadcrumb_display(separator=args.separator)
            print(f"\n--- [{bc}] ---\n{seg.content}")
        return 0

    # Incremental update: a segment's StableID is hash(header_path || normalized
    # content), so it survives line shifts and changes only when that section's
    # text changes. Any segment whose StableID is already in the previous mapping
    # keeps its compression, routing and generated checks - we only spend LLM
    # calls on sections the user actually edited.
    segment_sid_pairs = assign_stable_ids(segments)
    prior_mapping: dict[str, Any] = {}
    if not args.no_reuse and Path(input_mapping).is_file():
        try:
            with open(input_mapping, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                prior_mapping = loaded
        except Exception as e:
            print(f"Warning: could not read {input_mapping} for reuse: {e}", file=sys.stderr)

    def _reusable(sid: str) -> dict[str, Any] | None:
        entry = prior_mapping.get(sid)
        if not isinstance(entry, dict):
            return None
        if not (entry.get("compressed") or "").strip():
            return None
        if not isinstance(entry.get("strategies"), list):
            return None
        return entry

    reused_entries: dict[int, dict[str, Any]] = {}
    todo_indices: list[int] = []
    for i, (_seg, sid) in enumerate(segment_sid_pairs):
        entry = _reusable(sid)
        if entry is not None:
            reused_entries[i] = entry
        else:
            todo_indices.append(i)

    total = len(segments)
    if prior_mapping:
        print(
            f"Incremental: reusing {len(reused_entries)} unchanged segment(s), "
            f"processing {len(todo_indices)} new or changed.",
            file=sys.stderr,
        )

    compressed_texts: list[str] = [""] * total
    for i, entry in reused_entries.items():
        compressed_texts[i] = entry.get("compressed") or ""

    if todo_indices:
        try:
            client, model, temp, provider = _get_llm_client("COMPRESSION")
        except Exception as e:
            print(f"Error initializing LLM (set COMPRESSION_LLM or DEFAULT_LLM): {e}", file=sys.stderr)
            return 1

        def _compression_progress(current: int, total_n: int) -> None:
            print(f"\rCompressing {current}/{total_n} ...  ", end="", file=sys.stderr, flush=True)

        fresh = compress_segments(
            [segments[i] for i in todo_indices],
            client=client,
            model=model,
            temperature=temp,
            provider=provider,
            progress_callback=_compression_progress,
            parallel=not args.no_parallel,
            max_concurrency=args.max_concurrency,
        )
        for pos, i in enumerate(todo_indices):
            compressed_texts[i] = fresh[pos].compressed
        print(f"\rCompressed {len(todo_indices)} segment(s).   ", file=sys.stderr, flush=True)

    # Router: only for new or changed segments; reused ones keep their strategies
    # (including any already-generated check code).
    strategies_per_item: list[list[dict[str, Any]]] = [[] for _ in range(total)]
    for i, entry in reused_entries.items():
        strategies_per_item[i] = entry.get("strategies") or []

    if todo_indices:
        try:
            route_client, route_model, route_temp, route_provider = _get_llm_client("ROUTER")
        except Exception as e:
            print(f"Error initializing router LLM (set ROUTER_LLM or DEFAULT_LLM): {e}", file=sys.stderr)
            return 1
        concurrency = args.max_concurrency
        if concurrency is None:
            try:
                concurrency = int(os.environ.get("CONTEXTCOV_MAX_CONCURRENCY", "5"))
            except ValueError:
                concurrency = 5
        concurrency = max(1, min(concurrency, len(todo_indices)))

        def _routing_progress(done: int, total_n: int) -> None:
            print(f"\rRouting {done}/{total_n} ...  ", end="", file=sys.stderr, flush=True)

        router_results = route_constraints_parallel(
            [compressed_texts[i] for i in todo_indices],
            client=route_client,
            model=route_model,
            provider=route_provider,
            temperature=route_temp,
            max_concurrency=concurrency,
            progress_callback=_routing_progress,
        )
        for pos, i in enumerate(todo_indices):
            strategies_per_item[i] = strategies_to_dict_list(router_results[pos].strategies)
        print(f"\rRouted {len(todo_indices)} segment(s).   ", file=sys.stderr, flush=True)

    # Build mapping: StableID -> { header_path, original_content, compressed [, strategies] }
    mapping = build_compression_mapping(
        segment_sid_pairs,
        compressed_texts,
        strategies_per_item=strategies_per_item,
    )
    # Static check generator: all SOURCE_CHECK strategies get generated Python code (Tree-Sitter/regex)
    try:
        gen_client, gen_model, gen_temp, gen_provider = _get_llm_client("STATIC_CHECK")
    except Exception:
        gen_client, gen_model, gen_temp, gen_provider = _get_llm_client("DEFAULT")
    gen_concurrency = args.max_concurrency
    if gen_concurrency is None:
        try:
            gen_concurrency = int(os.environ.get("CONTEXTCOV_MAX_CONCURRENCY", "5"))
        except ValueError:
            gen_concurrency = 5
    gen_concurrency = max(1, min(gen_concurrency, total))

    repo_root_path: Path | None = None
    if args.repo_root:
        repo_root_path = Path(args.repo_root).resolve()
    else:
        from src.repo_clone import clone_repo, DEFAULT_DATA_DIR

        try:
            data_dir = Path(DEFAULT_DATA_DIR).resolve()
            dir_name = f"{inp.owner}_{inp.repo}".replace("/", "_")
            existing = data_dir / dir_name
            if existing.is_dir() and (existing / ".git").is_dir():
                repo_root_path = existing.resolve()
            else:
                clone_branch = inp.branch or args.branch
                with _spinner(
                    f"Cloning {inp.owner}/{inp.repo} for check generation ...",
                    done_message=f"Cloned {inp.owner}/{inp.repo}",
                ):
                    cloned_root = clone_repo(inp.owner, inp.repo, DEFAULT_DATA_DIR, branch=clone_branch)
                repo_root_path = Path(cloned_root).resolve()
        except Exception as e:
            print(f"Could not use or clone repo for check generation: {e}. Proceeding without repo root.", file=sys.stderr)

    repo_root_str: str | None = (
        str(repo_root_path) if repo_root_path and repo_root_path.is_dir() else None
    )
    if repo_root_str:
        print(f"Check generation repo root: {repo_root_str}", file=sys.stderr)

    n_tasks = _pending_strategy_count(mapping, "SOURCE_CHECK", "static_check")
    if n_tasks > 0:
        def _gen_progress(done: int, total_n: int) -> None:
            print(f"\rGenerating static checks {done}/{total_n} ...  ", end="", file=sys.stderr, flush=True)
        generate_static_checks_for_mapping(
            mapping,
            client=gen_client,
            model=gen_model,
            provider=gen_provider,
            temperature=gen_temp,
            max_concurrency=gen_concurrency,
            progress_callback=_gen_progress,
            repo_root=repo_root_str,
        )
        print(f"\rGenerated static checks for {n_tasks} SOURCE_CHECK strategy(ies).   ", file=sys.stderr)
    proc_tasks = _pending_strategy_count(mapping, "PROCESS_CHECK", "process_check")
    if proc_tasks > 0:
        try:
            proc_client, proc_model, proc_temp, proc_provider = _get_llm_client("PROCESS_CHECK")
        except Exception:
            proc_client, proc_model, proc_temp, proc_provider = _get_llm_client("DEFAULT")
        def _proc_progress(done: int, total_n: int) -> None:
            print(f"\rGenerating process checks {done}/{total_n} ...  ", end="", file=sys.stderr, flush=True)
        generate_process_checks_for_mapping(
            mapping,
            client=proc_client,
            model=proc_model,
            provider=proc_provider,
            temperature=proc_temp,
            max_concurrency=gen_concurrency,
            progress_callback=_proc_progress,
            repo_root=repo_root_str,
        )
        print(f"\rGenerated process checks for {proc_tasks} PROCESS_CHECK strategy(ies).   ", file=sys.stderr)
    arch_det_tasks = _pending_strategy_count(mapping, "ARCH_DETERMINISTIC", "arch_deterministic_check")
    if arch_det_tasks > 0:
        try:
            ad_client, ad_model, ad_temp, ad_provider = _get_llm_client("ARCH_DETERMINISTIC")
        except Exception:
            ad_client, ad_model, ad_temp, ad_provider = _get_llm_client("DEFAULT")
        def _ad_progress(done: int, total_n: int) -> None:
            print(f"\rGenerating arch deterministic checks {done}/{total_n} ...  ", end="", file=sys.stderr, flush=True)
        generate_arch_deterministic_checks_for_mapping(
            mapping,
            client=ad_client,
            model=ad_model,
            provider=ad_provider,
            temperature=ad_temp,
            max_concurrency=gen_concurrency,
            progress_callback=_ad_progress,
            repo_root=repo_root_str,
        )
        print(f"\rGenerated arch deterministic checks for {arch_det_tasks} strategy(ies).   ", file=sys.stderr)
    arch_sem_tasks = _pending_strategy_count(mapping, "ARCH_SEMANTIC", "arch_semantic_check")
    if arch_sem_tasks > 0:
        try:
            as_client, as_model, as_temp, as_provider = _get_llm_client("ARCH_SEMANTIC")
        except Exception:
            as_client, as_model, as_temp, as_provider = _get_llm_client("DEFAULT")
        def _as_progress(done: int, total_n: int) -> None:
            print(f"\rGenerating arch semantic checks {done}/{total_n} ...  ", end="", file=sys.stderr, flush=True)
        generate_arch_semantic_checks_for_mapping(
            mapping,
            client=as_client,
            model=as_model,
            provider=as_provider,
            repo_root=repo_root_str,
            temperature=as_temp,
            max_concurrency=gen_concurrency,
            progress_callback=_as_progress,
        )
        print(f"\rGenerated arch semantic checks for {arch_sem_tasks} strategy(ies).   ", file=sys.stderr)
    with open(output_mapping, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    # Record which instruction file this was generated from, so a later run can
    # tell a stale mapping from a current one.
    _source_digest_path(output_mapping).write_text(_source_digest(content) + "\n", encoding="utf-8")
    print(f"Wrote mapping ({len(mapping)} entries) to {output_mapping}", file=sys.stderr)

    # Surface rules we could not compile into a check. These fail open, so
    # without this they would look enforced while doing nothing.
    ungenerated = [
        s
        for e in mapping.values()
        if isinstance(e, dict)
        for s in (e.get("strategies") or [])
        if isinstance(s, dict) and (s.get("static_check") or {}).get("generation_failed")
    ]
    total_static = len(
        [
            s
            for e in mapping.values()
            if isinstance(e, dict)
            for s in (e.get("strategies") or [])
            if isinstance(s, dict) and (s.get("type") or "").strip().upper() == "SOURCE_CHECK"
        ]
    )
    if ungenerated:
        print(
            f"Warning: {len(ungenerated)} of {total_static} source check(s) could not be generated "
            f"and will not be enforced:",
            file=sys.stderr,
        )
        for s in ungenerated[:10]:
            print(f"  - {str(s.get('directive') or '(no directive)')[:100]}", file=sys.stderr)
        if len(ungenerated) > 10:
            print(f"  ... and {len(ungenerated) - 10} more", file=sys.stderr)

    for (seg, _sid), text in zip(segment_sid_pairs, compressed_texts):
        bc = seg.breadcrumb_display(separator=args.separator)
        print(f"\n--- [{bc}] ---\n{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
