"""
Process Check Generator: produces Python check code for PROCESS_CHECK strategies.

Generated code runs with exec() at module scope (see docs/lessons_check_generation.md).
Do not use 'return'; use result = (True, "") or result = (False, "msg").

**Philosophy: Fail Closed (Aggressive Compliance).** The Agent README is treated as a
strict spec. We optimize for zero missed violations; false positives are acceptable.
If the README is vague, the tool enforces globally. Developers can make the README
more precise or use CONTEXTCOV_ALLOW_UNSAFE=true to override. No artifact/path checks
unless the rule explicitly restricts scope (e.g. "In the frontend/ directory only").
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# Domain knowledge: mandated tool → competitors to block (for "only use X" rules).
DOMAIN_COMPETITORS = """
**Domain — competitors to block when rule mandates one tool:**
- **Python:** pip ↔ uv, poetry, conda, pipenv, pdm | black, ruff, pytest ↔ poetry run format / poetry run test
- **Node:** npm ↔ yarn, pnpm, bun
- **E2E:** cypress ↔ Playwright (pnpm test / pnpm test-ui)
"""

PROCESS_CHECK_SYSTEM_PROMPT = """You are a DevOps Compliance Engineer. You generate Python code for a shim that intercepts shell commands. Use **denylist (negative) logic** where possible; avoid strict allowlists that block the whole system.

**Runtime contract:** Each invocation sees only **`trigger`**, **`args`**, **`env`**, **`cwd`** for that simulated shell command. You do **not** see the full codebase, open files, or a git diff.

**Repository awareness:** If you can access and inspect the codebase (layout, scripts, config, and how commands are usually run), use that to calibrate when to block or allow and to reduce false positives, especially for path-scoped rules.

**In-prompt self-validation loop (required):** Before finalizing your output, test your draft logic against representative command variants that are likely in this repo (valid and invalid forms). If your draft would over-block legitimate workflows, refine by tightening match conditions or adding explicit scope guards from the directive. There may be many potential command forms; sample a few representative allowed/blocked cases to estimate false-positive behavior, then refine once before final output.

**Precision within fail-closed:** When the README spells out **exact** allowed commands, path prefixes, or workspaces, encode that scope (e.g. gate on **`cwd`**, path under repo) so you block **only** what the text forbids. When the README is **vague**, the policy below favors recall — but **do not invent** extra forbidden binaries, scopes, or patterns that the README does not support (including extrapolating beyond a single example).

**Rule intent — Fail Closed (optimize for recall, accept false positives):**

1. **Exclusive Mandate ("only use X")**
   - This check runs when the user invoked a **competitor** (trigger = that competitor). **Block immediately** with a message like "README requires '<mandated tool>'. Blocked. Update the README to specify exceptions or set CONTEXTCOV_ALLOW_UNSAFE=true."
   - Do **not** require artifact checks (pnpm-lock.yaml, pyproject.toml) to block. Block the competitor command globally unless the rule explicitly says "only in path P".
   - Domain reference: Python → pip/uv/poetry/conda; Node → npm/yarn/pnpm/bun; E2E → cypress vs Playwright. Do not block unrelated commands (e.g. ls, git) unless the rule clearly includes them.

2. **Explicit Ban ("never use X")**
   - Block the forbidden command globally. Use broad keyword matching so variations (e.g. npm run format:fix, yarn prettier) are caught.

**Trigger rule:** Trigger is ALWAYS the binary name (single word). For subcommands (e.g. "npm test"), inspect `args` with **keyword matching** (e.g. "test", "t", "format", "fmt").

**Broad argument matching (required):** Do **not** use exact args equality. Use substring/keyword matching: e.g. args_str = " ".join(args).lower(); forbidden = ["format", "fmt", "prettier", "lint", "test", "generate", "t"]; if any(k in args_str or k in args for k in forbidden): block. This catches npm run format:fix, npm run prettier, yarn run test, npm t, etc.

**Variables in scope (use these; do not assume args[0] is the command name):**
- `trigger`: str — the command name that was invoked (e.g. "npm", "pip", "uv").
- `args`: List[str] — arguments after the command (e.g. for `npm run format` → args = ["run", "format"]).
- `env`: Dict[str, str] — environment variables (e.g. CI, GITHUB_ACTIONS, PWD).
- `cwd`: str — current working directory (e.g. for path-based scope).

**Execution:** Code runs with `exec()` at top level. Do **not** use `return`. Use `result = (True, "")` or `result = (False, "msg")` and if/else.

**Output contract:**
- **Allow:** `result = (True, "")`
- **Block:** `result = (False, "Clear message; suggest updating the README or CONTEXTCOV_ALLOW_UNSAFE=true.")`
**Escape hatch (required):** Generated code must start with: if env.get("CONTEXTCOV_ALLOW_UNSAFE") == "true": result = (True, ""). Then the rest of the logic. Block messages should suggest "update the README or set CONTEXTCOV_ALLOW_UNSAFE=true".

**Do not fixate on examples:** When the rule gives an example (e.g. a branch name, a command, or a file path), use it only to infer the **required structure or format**. Do not hardcode the example's literal tokens or substrings as the only condition for when the rule applies unless the rule explicitly states that. Prefer the rule's general intent: e.g. "when a ticket is referenced" means any common ticket-like pattern in the branch or context, and enforce the required format for all such cases; or validate all relevant inputs against the required pattern. If in doubt, apply the rule broadly rather than only when the example string appears verbatim.

**Response format:** JSON only, no markdown:
{
  "code": "result = (True, \\\"\\\")\\nif env.get(\\\"CONTEXTCOV_ALLOW_UNSAFE\\\") == \\\"true\\\": result = (True, \\\"\\\")\\nelse:\\n  # your logic ...",
  "priority": 0,
  "enforcement_level": "block"
}
- `priority`: 0 = global rule, 10 = scoped (path/env). Optional; default 0.
- `enforcement_level`: "block". Optional.
- `code`: valid Python, newlines as \\n, escape quotes. Must start with the escape hatch check.
""" + DOMAIN_COMPETITORS


def _build_user_prompt(
    compressed_segment: str,
    trigger: str,
    directive: str,
    repo_root: str | None = None,
) -> str:
    """Build user message including original compressed constraint and router output."""
    # Normalize trigger to single word for prompt (generator must only emit code for this binary)
    trigger_word = (trigger or "").strip().split()[0] if (trigger or "").strip() else (trigger or "")
    repo_block = ""
    if repo_root and repo_root.strip():
        repo_block = f"""
**Repository root:** `{repo_root.strip()}`
If you can access this directory, inspect it when path-scoped rules or project conventions matter; otherwise follow the global fail-closed rules below.
"""
    return f"""**Original constraint (compressed):**
{compressed_segment}

**Router directive:** {directive}
**Trigger (binary name — single word only):** {trigger_word}

**Task:**
1. Generate a **global** check: block the forbidden command everywhere. Do not gate on os.path.exists(...) or cwd unless the rule explicitly restricts to a folder.
2. Use **broad keyword matching** for args (e.g. format, fmt, prettier, test, t, lint, generate). Do not use exact argument equality — except when the directive says "only allow [exact command]": then allow only that exact args (e.g. poetry run test → args == ["run", "test"] only; block pytest, poetry run pytest, poetry run test -k unit, poetry run python -m pytest).
3. For "block direct X" (e.g. block pytest): block every invocation when trigger == X regardless of args (no keyword gating).
4. For branch-naming rules: validate branch on both git checkout -b and git push; extract branch from args (e.g. after "-b", or push ref) when env vars are empty. Use the example only to define the required format (e.g. namespace/id-slug); treat any branch that looks like it references a ticket or issue (e.g. ticket IDs in common forms) as in-scope for that format — do not require the branch to literally contain a substring from the example to apply the rule.
5. Start the code with the **escape hatch**: if env.get("CONTEXTCOV_ALLOW_UNSAFE") == "true": result = (True, ""). Then your blocking logic.
6. When blocking, suggest "update the README to specify exceptions or set CONTEXTCOV_ALLOW_UNSAFE=true".
Output JSON with "code" (must include escape hatch), and optionally "priority", "enforcement_level": "block".""" + repo_block


def _parse_generator_response(content: str) -> Dict[str, Any]:
    """Parse LLM JSON response into { code, priority?, enforcement_level? }. Returns empty dict on failure."""
    if not content or not content.strip():
        return {}
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
    except Exception:
        return {}
    code = data.get("code") or ""
    if not isinstance(code, str):
        code = str(code)
    out = {"code": code}
    if "priority" in data and isinstance(data["priority"], (int, float)):
        out["priority"] = int(data["priority"])
    if data.get("enforcement_level") in ("block", "warning"):
        out["enforcement_level"] = data["enforcement_level"]
    return out


def validate_process_check_code(code: str) -> Tuple[bool, str]:
    """
    Validate process check code: must compile (syntax valid). Code is exec'd at
    module scope so 'return' is invalid; the prompt forbids it.
    Returns (True, "") if valid, (False, "error message") otherwise.
    """
    if not code or not code.strip():
        return False, "empty code"
    try:
        compile(code, "<process_check>", "exec")
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    return True, ""


def generate_process_check(
    compressed_segment: str,
    strategy: Dict[str, Any],
    *,
    client: Any,
    model: str,
    provider: str = "openai",
    temperature: float = 0.2,
    repo_root: str | None = None,
) -> Dict[str, Any]:
    """
    Generate Python check code for one PROCESS_CHECK strategy.

    strategy must have "trigger" and "directive". Returns {"code": str, ...}.
    """
    from llm import chat_completion_create, is_claude_code_provider

    trigger = str(strategy.get("trigger") or "").strip()
    directive = str(strategy.get("directive") or "")
    user_content = _build_user_prompt(
        compressed_segment,
        trigger,
        directive,
        repo_root=repo_root,
    )
    cc_cwd = repo_root if repo_root and is_claude_code_provider(provider) else None
    messages = [
        {"role": "system", "content": PROCESS_CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    resp = chat_completion_create(
        client=client,
        model=model,
        messages=messages,
        provider=provider,
        temperature=temperature,
        claude_code_cwd=cc_cwd,
        response_format={"type": "json_object"},
    )
    content = ""
    if getattr(resp, "choices", None) and len(resp.choices) > 0:
        msg = getattr(resp.choices[0], "message", None)
        if msg is not None:
            content = (getattr(msg, "content", None) or "").strip()
    out = _parse_generator_response(content)
    if not out.get("code"):
        out["code"] = "result = (True, \"\")  # generator produced no code; allow by default"
    valid, err = validate_process_check_code(out.get("code", ""))
    if not valid and err:
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": f"The generated check failed validation: {err}. Fix the code (do not use 'return'; set result = (True, \"\") or result = (False, \"msg\")). Reply with the same JSON format.",
        })
        resp2 = chat_completion_create(
            client=client,
            model=model,
            messages=messages,
            provider=provider,
            temperature=temperature,
            claude_code_cwd=cc_cwd,
            response_format={"type": "json_object"},
        )
        content2 = ""
        if getattr(resp2, "choices", None) and len(resp2.choices) > 0:
            msg2 = getattr(resp2.choices[0], "message", None)
            if msg2 is not None:
                content2 = (getattr(msg2, "content", None) or "").strip()
        out2 = _parse_generator_response(content2)
        if out2.get("code"):
            valid2, _ = validate_process_check_code(out2["code"])
            if valid2:
                out = out2
    # Normalize optional fields for storage in strategy["process_check"]
    if "priority" not in out:
        out["priority"] = 0
    if "enforcement_level" not in out:
        out["enforcement_level"] = "block"
    return out


def _content_from_resp(resp: Any) -> str:
    if getattr(resp, "choices", None) and len(resp.choices) > 0:
        msg = getattr(resp.choices[0], "message", None)
        if msg is not None:
            return (getattr(msg, "content", None) or "").strip()
    return ""


def collect_process_check_tasks(mapping: Dict[str, Dict[str, Any]]) -> List[tuple]:
    """
    From a mapping, collect (stable_id, entry, strategy_index, compressed, strategy)
    for every strategy that has type PROCESS_CHECK and does not yet have process_check.
    """
    tasks = []
    for sid, entry in mapping.items():
        if not isinstance(entry, dict):
            continue
        strategies = entry.get("strategies") or []
        compressed = entry.get("compressed") or ""
        for idx, s in enumerate(strategies):
            if not isinstance(s, dict):
                continue
            if (s.get("type") or "").strip().upper() != "PROCESS_CHECK":
                continue
            if s.get("process_check"):
                continue
            tasks.append((sid, entry, idx, compressed, s))
    return tasks


def generate_process_checks_for_mapping(
    mapping: Dict[str, Dict[str, Any]],
    *,
    client: Any,
    model: str,
    provider: str = "openai",
    temperature: float = 0.2,
    max_concurrency: int = 5,
    progress_callback: Optional[Any] = None,
    repo_root: str | None = None,
) -> None:
    """
    For every PROCESS_CHECK strategy in mapping that lacks process_check, call the LLM
    generator and store the result in strategy["process_check"] = { "code": "..." }.
    Mutates mapping in place.
    """
    import llm
    from llm import is_claude_code_provider

    tasks = collect_process_check_tasks(mapping)
    if not tasks:
        return

    cc_cwd = repo_root if repo_root and is_claude_code_provider(provider) else None
    requests = []
    for sid, entry, idx, compressed, strategy in tasks:
        trigger = str(strategy.get("trigger") or "").strip()
        directive = str(strategy.get("directive") or "")
        user_content = _build_user_prompt(
            compressed,
            trigger,
            directive,
            repo_root=repo_root,
        )
        messages = [
            {"role": "system", "content": PROCESS_CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        requests.append({
            "client": client,
            "model": model,
            "messages": messages,
            "provider": provider,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "claude_code_cwd": cc_cwd,
            "_entry": entry,
            "_idx": idx,
        })
    responses = llm.run_parallel_chat_completions(
        requests,
        max_concurrency=max_concurrency,
        progress_callback=progress_callback,
    )
    for req, resp in zip(requests, responses):
        content = _content_from_resp(resp)
        out = _parse_generator_response(content)
        code = out.get("code") or ""
        if not code:
            code = "result = (True, \"\")  # generator produced no code; allow by default"
        valid, err = validate_process_check_code(code)
        if not valid and err:
            code = f"result = (True, \"\")  # validation failed: {err}"
        if "priority" not in out:
            out["priority"] = 0
        if "enforcement_level" not in out:
            out["enforcement_level"] = "block"
        entry = req["_entry"]
        idx = req["_idx"]
        strategies = entry.get("strategies") or []
        if idx < len(strategies):
            pc = {"code": code}
            if out.get("priority") is not None:
                pc["priority"] = out["priority"]
            if out.get("enforcement_level"):
                pc["enforcement_level"] = out["enforcement_level"]
            strategies[idx]["process_check"] = pc
