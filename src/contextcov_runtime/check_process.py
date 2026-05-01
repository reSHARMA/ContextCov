#!/usr/bin/env python3
"""
ContextCov Process Check Runner: runs generated Python checks for intercepted commands.

Loaded by the shim dispatcher. Reads .contextcov/compliance_db.json (relative to this
script's parent directory), finds all checks where type=PROCESS and trigger=command_name,
executes each check's code with args and env in scope. Blocks (exit 1) if any check
sets result = (False, message); otherwise exits 0 and the dispatcher runs the real binary.

Environment variables:
  CONTEXTCOV_DEBUG=1  - Enable debug logging for troubleshooting fail-open cases
"""

import json
import os
import sys

# Debug mode: set CONTEXTCOV_DEBUG=1 to log internal errors to a file
_DEBUG = os.environ.get("CONTEXTCOV_DEBUG", "").strip() in ("1", "true", "yes")


def _debug_log(message: str) -> None:
    """Log debug message to .contextcov/debug.log if CONTEXTCOV_DEBUG is enabled."""
    if not _DEBUG:
        return
    try:
        root = _contextcov_root()
        log_path = os.path.join(root, "debug.log")
        with open(log_path, "a", encoding="utf-8") as f:
            import datetime
            ts = datetime.datetime.now().isoformat()
            f.write(f"[{ts}] {message}\n")
    except Exception:
        pass  # Best effort logging


def _contextcov_root() -> str:
    """Directory containing .contextcov (parent of runtime/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_db() -> dict:
    """Load compliance_db.json from .contextcov/."""
    root = _contextcov_root()
    path = os.path.join(root, "compliance_db.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    if len(sys.argv) < 2:
        print("[ContextCov Error] check_process.py requires command name as first argument", file=sys.stderr)
        sys.exit(2)
    command_name = sys.argv[1]
    args = sys.argv[2:]

    db = load_db()
    if not db:
        sys.exit(0)  # No DB: allow command

    # Support both dict-of-entries { "id": { "type", "trigger", "code" } } and list format
    if isinstance(db, list):
        entries = [{"_id": i, **e} for i, e in enumerate(db) if isinstance(e, dict)]
    else:
        entries = [{"_id": kid, **v} for kid, v in db.items() if isinstance(v, dict)]

    checks = [
        e for e in entries
        if (e.get("type") or "").strip().upper() == "PROCESS"
        and (e.get("trigger") or "").strip() == command_name
        and (e.get("code") or "").strip()
    ]
    # Specificity: run higher-priority (scoped) checks first. Default priority 0.
    checks.sort(key=lambda e: -(e.get("priority") if isinstance(e.get("priority"), (int, float)) else 0))

    env_copy = dict(os.environ)
    cwd = os.getcwd()

    for check in checks:
        code = check.get("code", "").strip()
        # Contract: trigger, args, env, cwd. Generated checks may allow override via env CONTEXTCOV_ALLOW_UNSAFE=true.
        code_context = {
            "trigger": command_name,
            "args": args,
            "env": env_copy,
            "cwd": cwd,
            "os": os,
            "sys": sys,
            "result": (True, ""),
        }
        try:
            exec(code, code_context)
            passed, message = code_context.get("result", (True, ""))
            if not passed:
                msg = message or "Command blocked by ContextCov."
                print(f"[ContextCov Violation] {msg}", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"[ContextCov Internal Error] Could not run check: {e}", file=sys.stderr)
            _debug_log(f"Internal error in check {check.get('_id', 'unknown')}: {e}")
            _debug_log(f"  Command: {command_name} {' '.join(args)}")
            _debug_log(f"  Check code:\n{code[:500]}")
            # Fail open: allow command on internal error so workflow is not broken
            continue

    sys.exit(0)


if __name__ == "__main__":
    main()
