"""
Setup the ContextCov shim system: export process checks to .contextcov/compliance_db.json,
install runtime scripts, and create symlinks so PATH can prefer .contextcov/bin.

Usage:
  python -m src.setup_shims [--mapping FILE] [--contextcov-dir DIR]
  Default mapping: from cwd or CONTEXTCOV_MAPPING; default .contextcov dir: .contextcov in cwd.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Commands the dispatcher itself must run. Shimming any of these would make
# the shim re-enter the dispatcher, so they are refused at install time.
DISPATCHER_DEPENDENCIES = {"bash", "sh", "env", "python3", "basename", "dirname", "pwd", "cd"}

# Project root for finding contextcov_runtime templates
_SCRIPT_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _SCRIPT_DIR.parent


def build_compliance_db_from_mapping(mapping: dict) -> dict:
    """
    Build compliance_db.json content from a mapping (stable_id -> entry).
    Includes only PROCESS_CHECK strategies that have process_check.code.
    Returns a dict suitable for JSON: { "id": { "type": "PROCESS", "trigger": "...", "code": "..." }, ... }.
    """
    db = {}
    for sid, entry in mapping.items():
        if not isinstance(entry, dict):
            continue
        strategies = entry.get("strategies") or []
        for idx, s in enumerate(strategies):
            if not isinstance(s, dict):
                continue
            if (s.get("type") or "").strip().upper() != "PROCESS_CHECK":
                continue
            pc = s.get("process_check")
            if not pc or not (pc.get("code") or "").strip():
                continue
            raw_trigger = (s.get("trigger") or "").strip()
            if not raw_trigger:
                continue
            # Canonical trigger: only the binary (first word). Multi-word triggers like "npm test"
            # are normalized to "npm" so the shim for "npm" runs the check; the check code uses args.
            trigger = raw_trigger.split()[0]
            entry_id = f"{sid}_{idx}"
            db[entry_id] = {
                "type": "PROCESS",
                "trigger": trigger,
                "code": (pc.get("code") or "").strip(),
            }
            if isinstance(pc.get("priority"), (int, float)):
                db[entry_id]["priority"] = int(pc["priority"])
            if pc.get("enforcement_level") in ("block", "warning"):
                db[entry_id]["enforcement_level"] = pc["enforcement_level"]
    return db


def get_runtime_files_dir() -> Path:
    """Path to src/contextcov_runtime (templates for .contextcov/runtime)."""
    return _SCRIPT_DIR / "contextcov_runtime"


def setup(
    mapping_path: str | Path,
    contextcov_dir: str | Path = ".contextcov",
    *,
    dry_run: bool = False,
) -> None:
    """
    Create .contextcov with compliance_db.json, runtime scripts, and bin symlinks.

    mapping_path: path to mapping JSON (from CLI output).
    contextcov_dir: directory to create (e.g. .contextcov in repo root).
    """
    mapping_path = Path(mapping_path)
    contextcov_dir = Path(contextcov_dir).resolve()

    if not mapping_path.is_file():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)

    db = build_compliance_db_from_mapping(mapping)
    triggers = set(e.get("trigger", "").strip() for e in db.values() if (e.get("trigger") or "").strip())

    if dry_run:
        print(f"Would create {contextcov_dir}/ with compliance_db ({len(db)} process checks)")
        print(f"Triggers: {', '.join(sorted(triggers)) or '(none)'}")
        return

    runtime_src = get_runtime_files_dir()
    runtime_dst = contextcov_dir / "runtime"
    bin_dir = contextcov_dir / "bin"
    runtime_dst.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    # Write compliance_db.json
    db_path = contextcov_dir / "compliance_db.json"
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    print(f"Wrote {db_path} ({len(db)} process check(s))")

    # Copy runtime scripts (dispatcher, process runner, login-style shell wrapper)
    for name in ("check_process.py", "dispatcher.sh", "agent_shell.sh"):
        src = runtime_src / name
        if not src.is_file():
            print(f"Warning: template {src} not found", file=sys.stderr)
            continue
        dst = runtime_dst / name
        shutil.copy2(src, dst)
        if name.endswith(".sh"):
            # Pin the interpreter to an absolute path. A "#!/usr/bin/env bash"
            # line resolves through PATH, which begins with our own shim dir, so
            # a shimmed `env`/`bash` would re-enter the dispatcher forever.
            real_bash = shutil.which("bash", path=os.defpath) or shutil.which("bash")
            if real_bash:
                text = dst.read_text(encoding="utf-8")
                if text.startswith("#!"):
                    _, _, rest = text.partition("\n")
                    dst.write_text(f"#!{real_bash}\n{rest}", encoding="utf-8")
            os.chmod(dst, 0o755)
        print(f"Installed {dst}")

    dispatcher_path = runtime_dst / "dispatcher.sh"
    if not dispatcher_path.exists():
        print("Error: dispatcher.sh not installed.", file=sys.stderr)
        sys.exit(1)

    # Create symlinks in bin/ for each trigger
    dispatcher_abs = dispatcher_path.resolve()
    for cmd in sorted(triggers):
        # The trigger comes from an LLM-generated mapping; treat it as untrusted.
        # It must be a bare command name, never a path that could escape bin/.
        if not cmd or cmd in (".", "..") or os.path.basename(cmd) != cmd or cmd.startswith("."):
            print(f"Warning: skipping unsafe shim trigger {cmd!r} (not a bare command name)", file=sys.stderr)
            continue
        if cmd in DISPATCHER_DEPENDENCIES:
            print(
                f"Warning: skipping trigger {cmd!r}: the dispatcher itself needs it, "
                f"so shimming it would re-enter the dispatcher.",
                file=sys.stderr,
            )
            continue
        shim = bin_dir / cmd
        if shim.exists() or shim.is_symlink():
            shim.unlink()
        try:
            shim.symlink_to(dispatcher_abs)
        except OSError as e:
            print(f"Warning: could not create symlink {shim}: {e}", file=sys.stderr)
        else:
            print(f"Shim: {shim} -> dispatcher.sh")

    bin_abs = bin_dir.resolve()
    print(f"\nActivate with: export PATH={bin_abs}:$PATH")
    if not triggers:
        print("(No process checks in mapping; add PATH when you add PROCESS_CHECK strategies.)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Setup ContextCov shim directory and compliance DB from a mapping file.",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=os.environ.get("CONTEXTCOV_MAPPING", ""),
        help="Path to mapping JSON (default: CONTEXTCOV_MAPPING or first .mapping.json in cwd)",
    )
    parser.add_argument(
        "--contextcov-dir",
        type=str,
        default=".contextcov",
        help="Target directory for .contextcov (default: .contextcov)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done",
    )
    args = parser.parse_args()

    mapping = args.mapping.strip()
    if not mapping:
        # Default: first *.mapping.json in cwd
        cwd = Path.cwd()
        for f in sorted(cwd.glob("*.mapping.json")):
            mapping = str(f)
            break
        if not mapping:
            print("Error: no --mapping and no *.mapping.json in cwd. Set CONTEXTCOV_MAPPING or pass --mapping.", file=sys.stderr)
            return 1
    try:
        setup(mapping, args.contextcov_dir, dry_run=args.dry_run)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
