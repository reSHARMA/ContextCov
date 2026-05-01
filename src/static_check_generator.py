"""
Static Check Generator: produces Python check code (Tree-Sitter or regex) for SOURCE_CHECK strategies.

All compressed segments that are routed to SOURCE_CHECK go through this generator. The generator
receives both the original compressed segment (constraint) and the router output (directive, trigger)
and produces Python code that runs in the static_runner with `tree` and `source_bytes` in scope;
the code must set `result = "FAIL"` when a violation is found.

Uses STATIC_CHECK_LLM or DEFAULT_LLM. Supports Tree-Sitter (AST) and regex fallback.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

STATIC_CHECK_SYSTEM_PROMPT = """You are a Static Analysis Engineer. You generate Python code that checks source files for a specific constraint.

**Important:** These checks will run over the complete repository (every file matching the trigger). Generate a check for every provided constraint; do not skip constraints. To keep quality high, avoid brittle logic that flags legitimate code. Prefer precise, narrow conditions that only fail on clear violations.

**Coverage without brittleness:** Do not drop a constraint because it is hard or specific. Instead, choose one of:
- make the trigger more specific so the check runs only where the rule truly applies, and keep it strict there; or
- keep broader scope but make the check softer/conservative (high precision, lower false positives).
If forced to choose, keep the constraint covered while reducing brittleness via trigger refinement or conservative fail conditions.

**Repository awareness:** If you can access and inspect this codebase (files, directories, packages, naming, and tooling), do that before you finalize the check. Use what you learn to choose triggers and logic that match the real project and to avoid unnecessary false positives.

**In-prompt self-validation loop (required):** Before finalizing your output, run your draft check against the real repository using the provided script/venv command (when available in this environment). If your draft flags many legitimate files, revise it by (a) narrowing trigger scope and/or (b) tightening fail conditions, then run it again. There may be many violations; you do not need to inspect all of them. Sample a few representative flagged files/violations to estimate false-positive behavior, then refine to improve false-positive rate while still covering the constraint. Do at least one revise-and-rerun cycle so the final check is less brittle while still covering the constraint.

**Scoping to reduce false positives:** For rules that sound global (e.g. "all files must have copyright", "use tabs not spaces", "no double quotes"), narrow the trigger so the check does not run on files where the rule typically does not apply: e.g. exclude test/, scripts/, .devcontainer/, .github/, build/config files (e.g. *.json, *.yml at repo root), or third-party/tooling paths. When you know which paths exist, restrict the trigger (e.g. src/**/*.ts instead of **/*.ts) unless the rule explicitly requires every file in the repo.

- Your output must be correct and precise:
- **Trigger (file pattern):** The trigger determines which files this check runs on. It must be precise and match only the intended paths in the repo. Refine the trigger using whatever you can verify about the real layout.
- **Trigger pattern syntax (pathlib glob):** The runner uses pathlib glob. Supported:
  - `*` (any characters in a segment), `**` (any number of path segments)
  - `|` to separate alternatives (e.g. `README.md|docs/**/*.md`)
  - `{a,b,c}` for brace expansion (e.g. `*.{ts,tsx}` or `*.{css,scss,tsx,jsx}`). Commas inside braces are part of the pattern, not alternation.
  - Comma outside braces is also alternation (e.g. `*.tsx,*.jsx` matches .tsx and .jsx files).
  Do NOT use extended glob syntax such as `?(x)`, `*(x)`, `+(x)` — they will not match. Use `{ts,tsx}` or `*.tsx,*.jsx` instead.
- **Code:** Use paths and patterns that match the actual repo layout. Do not assume paths or conventions you have not verified.

**Runtime contract:** The runner calls your check **once per file** matching the trigger. In scope: **`tree`**, **`source_bytes`**, **`language`** for that file only. Not available: whole-repo file listing, other files’ contents, git diff, PR metadata. Do not assume multi-file or change-set context unless it is inferable from this file alone.

**Change-based / PR-only rules:** If the constraint is really about *what changed in a PR* (e.g. only new lines, changelog required per change, comparison to a base branch) and cannot be decided from **this file’s** content alone, still generate coverage by using **narrow triggers** onto file kinds where the rule is checkable and/or conservative file-level heuristics. Do not approximate “PR awareness” with blunt heuristics that flag normal, unchanged code.

**High-precision fail policy:** Set `result = "FAIL"` only when there is strong file-local evidence that the rule is violated. If uncertain, prefer refining trigger/logic over broad failure conditions.

**Context:** The check will run inside a runner that provides:
- `tree`: Tree-sitter Tree object (root_node is the AST root), or None if the file is not parsed (e.g. text/regex-only).
- `source_bytes`: Raw file content as bytes (use .decode("utf-8", errors="replace") for text).
- `language`: Tree-sitter Language object for queries, or None.

**Execution environment:** The runner executes your code with `exec(code, scope)` in a module-like scope (no enclosing function). Therefore:
- Do NOT use `nonlocal` in nested functions (e.g. inside a visitor or walk). It will raise "no binding for nonlocal" at runtime.
- If you need to mutate a variable from inside a nested function, use a mutable container: e.g. `found = [False]` and set `found[0] = True` inside the nested function; then check `found[0]` in the outer scope.
- When parsing strings character-by-line, do not clear a delimiter then call `len()` on the same name in one step (wrong: `in_string = None` then `i += len(in_string)`). Save the length first: `step = len(in_string); in_string = None; i += step`.

**Opt-out for "unless" rules:** When the constraint has an exception (e.g. "don't use X unless Y", "don't export Z unless used outside"), add a single allowlist comment so legitimate exceptions can opt out. Example: if the rule forbids `export interface Props` unless used elsewhere, allow a comment like `@external-props` or `external Props` in the file to skip the check. Document the comment in the code (e.g. "# Allow @external-props to mark intentional external use"). This reduces false positives where the exception applies but cannot be detected statically.

**Output contract:** Your code must set `result = "FAIL"` when the constraint is violated, and leave `result` unset (or set to "PASS") otherwise. The runner executes your code in a restricted scope with only these variables.

**Two modes:**
1. **Tree-Sitter (preferred for structure):** When the rule involves syntax (e.g. "no console.log", "async functions must return Promise"), use the `tree` and `language` to run a tree-sitter query. Example pattern for console.log:
   `query = language.query("(call_expression function: (member_expression object: (identifier) @obj property: (property_identifier) @prop) (#eq? @obj \\"console\\") (#eq? @prop \\"log\\"))")`
   Then `captures = query.captures(tree.root_node)` and if captures is non-empty, set `result = "FAIL"`.
2. **Regex (fallback):** When the rule is simple (e.g. "no TODO comments", "must contain X") or when no AST is needed, use the `source_bytes` as text and Python `re` module. Set `result = "FAIL"` when the forbidden pattern is found (or required pattern is missing). When matching line-by-line, skip lines that are clearly comments (e.g. after strip, start with `//`, `#`, or `*`) to avoid false positives from comment text.

**Response format (strict):** Reply with **one** JSON object and nothing else.

**Required shape (RFC 8259 JSON):**
- Top-level keys: **`target_lang`** (string, required), **`code`** (required), **`trigger`** (string, optional).
- **`target_lang`:** one of `"python"`, `"typescript"`, `"javascript"`, `"text"`.
- **`code`:** the full Python check as a **single JSON string**. Inside that string, newlines must be the two-character escape `\\n`, not real line breaks. Do not put raw newlines inside the JSON string value for `code` (invalid JSON). Do not wrap `code` in markdown or triple quotes.
- **`trigger`:** optional pathlib glob; if present it overrides the router trigger.

**Valid minimal example (note escaped newlines in `code`):**
{"target_lang":"text","code":"result = None\\n"}

**Forbidden:** prose before/after the object, markdown code fences (```), `json` labels, or a `code` field that is a multi-line bare string breaking JSON.

If you output "trigger", it overrides the router trigger so the check runs only on files matching your precise pattern.

- Use `target_lang: "text"` for regex-only checks that apply to any file.
- Use `target_lang: "python"` or `"typescript"` etc. when you use tree-sitter (so the runner picks the right parser).
- The `code` string must be valid Python. Do not use triple-quoted strings inside `code` in a way that breaks the outer JSON.
"""


def _build_user_prompt(
    compressed_segment: str,
    trigger: str,
    directive: str,
    repo_root: str | None = None,
) -> str:
    """Build user message including original compressed constraint and router output."""
    repo_block = ""
    if repo_root and repo_root.strip():
        helper_script = (_PROJECT_ROOT / "scripts" / "run_generated_static_check.py").resolve()
        venv_python = (_PROJECT_ROOT / "venv" / "bin" / "python").resolve()
        run_block = ""
        if helper_script.is_file():
            run_block = f"""
**Self-test command (recommended):**
1) Save your generated Python code into a temp file, e.g. `/tmp/generated_static_check.py`
2) Run:
`{venv_python} {helper_script} --repo-root "{repo_root.strip()}" --trigger "<trigger>" --target-lang "<target_lang>" --directive "<directive>" --code-file /tmp/generated_static_check.py --max-violations 50`
3) If results look noisy, refine trigger/code and rerun before final answer.
"""
        repo_block = f"""
**Repository root:** `{repo_root.strip()}`
If you can access this directory, inspect the codebase and use it to refine triggers and check logic so they fit the project and limit false positives. Use only pathlib-compatible glob: *, **, |, and {{a,b,c}}. Do not use extended globs like ?(x).
{run_block}
"""
    return f"""**Original constraint (compressed):**
{compressed_segment}

**Router directive:** {directive}
**Trigger (file pattern):** {trigger}
{repo_block}
Generate the static check JSON (target_lang + code, and optionally trigger for precision) as specified.

**Final line:** Output exactly one JSON object starting with `{{` and ending with `}}` — no markdown fences, no commentary. In the JSON, the `code` value must be one string with newlines only as the escape `\\n` (backslash then n)."""


def _strip_markdown_json_fences(text: str) -> str:
    """Remove optional ```json ... ``` or ``` ... ``` wrappers from model output."""
    s = text.strip()
    if not s.startswith("`"):
        return s
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            after_open = s[first_nl + 1 :]
            if after_open.rstrip().endswith("```"):
                inner = after_open.rstrip()[:-3].rstrip()
                return inner.strip()
    return s


def _parse_generator_response(content: str) -> Dict[str, Any]:
    """Parse LLM JSON response into { target_lang, code, trigger? }. Returns empty dict on failure."""
    if not content or not content.strip():
        return {}
    text = _strip_markdown_json_fences(content.strip())
    # Try direct parse.
    try:
        data = json.loads(text)
    except Exception:
        data = None
    # If mixed prose/fences, extract the first JSON object region.
    if not isinstance(data, dict):
        start = text.find("{")
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                data = parsed
                                break
                        except Exception:
                            break
            if isinstance(data, dict):
                break
            start = text.find("{", start + 1)
    if not isinstance(data, dict):
        return {}
    target_lang = (data.get("target_lang") or "text").strip().lower()
    if target_lang not in ("python", "typescript", "javascript", "text", "ts", "js"):
        if target_lang in ("ts",):
            target_lang = "typescript"
        elif target_lang in ("js",):
            target_lang = "javascript"
        else:
            target_lang = "text"
    code = data.get("code") or ""
    if isinstance(code, list):
        code = "\n".join(str(line) for line in code)
    elif not isinstance(code, str):
        code = str(code)
    out = {"target_lang": target_lang, "code": code}
    trigger = data.get("trigger")
    if isinstance(trigger, str) and trigger.strip():
        out["trigger"] = trigger.strip()
    return out


def validate_generated_check(code: str) -> Tuple[bool, str]:
    """
    Validate generated check code: syntax (ast.parse) and dry-run exec in runner-like scope.
    Returns (True, "") if valid, (False, "error message") if syntax or runtime error.
    """
    if not code or not code.strip():
        return False, "empty code"
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    exec_globals = {
        "tree": None,
        "source_bytes": b"",
        "source_text": "",
        "language": None,
        "re": re,
        "result": None,
    }
    try:
        exec(code, exec_globals)
    except Exception as e:
        return False, f"runtime error: {e}"
    return True, ""


def generate_static_check(
    compressed_segment: str,
    strategy: Dict[str, Any],
    *,
    client: Any,
    model: str,
    provider: str = "openai",
    temperature: float = 0.2,
    repo_root: str | None = None,
    max_validation_retries: int = 1,
) -> Dict[str, Any]:
    """
    Generate Python check code for one SOURCE_CHECK strategy.

    strategy must have "trigger" and "directive". Returns {"target_lang": str, "code": str}.
    After generation, validates code (syntax + dry-run exec); on failure retries up to
    max_validation_retries times with the validation error sent to the LLM.
    """
    from llm import chat_completion_create, is_claude_code_provider

    trigger = str(strategy.get("trigger") or "*")
    directive = str(strategy.get("directive") or "")
    user_content = _build_user_prompt(
        compressed_segment,
        trigger,
        directive,
        repo_root=repo_root,
    )
    cc_cwd = repo_root if repo_root and is_claude_code_provider(provider) else None
    messages = [
        {"role": "system", "content": STATIC_CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    out = None
    last_validation_error = ""
    for attempt in range(max_validation_retries + 1):
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
        code = (out.get("code") or "").strip()
        if not code or code == "result = None  # generator produced no code":
            ok, last_validation_error = False, "missing usable code (empty/unparseable/no-op)"
        else:
            ok, last_validation_error = validate_generated_check(code)
        if ok:
            break
        if attempt < max_validation_retries:
            messages.append(
                {
                    "role": "user",
                    "content": f"The generated check failed validation: {last_validation_error}. Fix the code and reply with the same JSON format (target_lang, code, optional trigger). Ensure valid Python syntax and that the code runs without error when executed with tree=None, source_bytes=b'', language=None, re=re, result=None (no nonlocal, escape regex parens like \\( for literal '(').",
                }
            )
    # out may contain "trigger" to override the strategy trigger
    return out


def _content_from_resp(resp: Any) -> str:
    if getattr(resp, "choices", None) and len(resp.choices) > 0:
        msg = getattr(resp.choices[0], "message", None)
        if msg is not None:
            return (getattr(msg, "content", None) or "").strip()
    return ""


def collect_source_check_tasks(mapping: Dict[str, Dict[str, Any]]) -> List[tuple]:
    """
    From a mapping (stable_id -> entry), collect (stable_id, entry, strategy_index) for every
    strategy that has type SOURCE_CHECK and does not yet have static_check.
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
            if (s.get("type") or "").strip().upper() != "SOURCE_CHECK":
                continue
            if s.get("static_check"):
                continue
            tasks.append((sid, entry, idx, compressed, s))
    return tasks


def generate_static_checks_for_mapping(
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
    For every SOURCE_CHECK strategy in mapping that lacks static_check, call the LLM generator
    and store the result in strategy["static_check"] = { "target_lang", "code" }.
    Mutates mapping in place.
    """
    import llm
    from llm import is_claude_code_provider

    tasks = collect_source_check_tasks(mapping)
    if not tasks:
        return

    cc_cwd = repo_root if repo_root and is_claude_code_provider(provider) else None
    requests = []
    for sid, entry, idx, compressed, strategy in tasks:
        trigger = str(strategy.get("trigger") or "*")
        directive = str(strategy.get("directive") or "")
        user_content = _build_user_prompt(
            compressed,
            trigger,
            directive,
            repo_root=repo_root,
        )
        messages = [
            {"role": "system", "content": STATIC_CHECK_SYSTEM_PROMPT},
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
            "_sid": sid,
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
        code = (out.get("code") or "").strip()
        if not code or code == "result = None  # generator produced no code":
            ok, validation_err = False, "missing usable code (empty/unparseable/no-op)"
        else:
            ok, validation_err = validate_generated_check(code)
        if not ok and validation_err:
            # One retry with validation error feedback
            retry_messages = req["messages"] + [
                {
                    "role": "user",
                    "content": f"The generated check failed validation: {validation_err}. Fix the code and reply with the same JSON format (target_lang, code, optional trigger). Ensure valid Python syntax and that the code runs without error when executed with tree=None, source_bytes=b'', language=None, re=re, result=None (no nonlocal, escape regex parens like \\( for literal '(').",
                }
            ]
            retry_resp = llm.chat_completion_create(
                client=req["client"],
                model=req["model"],
                messages=retry_messages,
                provider=req["provider"],
                temperature=req["temperature"],
                claude_code_cwd=req.get("claude_code_cwd"),
                response_format={"type": "json_object"},
            )
            retry_content = _content_from_resp(retry_resp)
            out = _parse_generator_response(retry_content)
            code2 = (out.get("code") or "").strip()
            # A check we could not generate must not become a check that fails
            # everything: result = 'FAIL' here would report a violation on every
            # file matching the trigger, manufacturing findings out of a
            # generation failure. Fail open, as the process and arch_det
            # generators already do, and record it so it can be reported.
            if not code2 or code2 == "result = None  # generator produced no code":
                out["code"] = f"result = None  # generation failed: {validation_err}"
                out["generation_failed"] = str(validation_err)
            else:
                ok2, err2 = validate_generated_check(code2)
                if not ok2:
                    out["code"] = f"result = None  # validation failed: {err2}"
                    out["generation_failed"] = str(err2)
        entry = req["_entry"]
        idx = req["_idx"]
        strategies = entry.get("strategies") or []
        if idx < len(strategies):
            strategies[idx]["static_check"] = {
                "target_lang": out.get("target_lang", "text"),
                "code": out.get("code", ""),
            }
            if out.get("generation_failed"):
                # Recorded so `generate` can tell the user this rule is not
                # actually being enforced, rather than leaving a silent no-op.
                strategies[idx]["static_check"]["generation_failed"] = out["generation_failed"]
            if out.get("trigger"):
                strategies[idx]["trigger"] = out["trigger"]
