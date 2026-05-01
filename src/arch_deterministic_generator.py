"""
Arch Deterministic Check Generator: produces Python code that queries a NetworkX graph.

All compressed segments routed to ARCH_DETERMINISTIC get generated code that runs in the
arch runner with `graph` (DiGraph) and `nx` in scope; code must set result = (False, "reason")
on violation. Uses ARCH_DETERMINISTIC_LLM or DEFAULT_LLM.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional, Tuple

ARCH_DET_SYSTEM_PROMPT = """You are a Software Architect. You generate Python code that checks a dependency graph for structural violations.

**Important:** These checks will run over the complete repository (the full dependency graph). Keep this in mind so that rules are not too strict: avoid conditions that would flag legitimate structure and cause false positives. Prefer precise, narrow checks that only fail on clear violations. Node paths in the graph match the actual repo file paths.

**Repository awareness:** If you can access and inspect the codebase, use it so your graph checks reference real paths, layers, and import patterns—avoid assuming directories or edges you have not verified.

**In-prompt self-validation loop (required):** Before finalizing your output, sanity-check your draft rule against the repository graph assumptions and likely path patterns. If your draft would flag many legitimate nodes/edges, refine by narrowing path scope, clarifying allowed boundaries, or requiring stronger evidence before failing. There may be many flagged edges; you do not need to inspect all of them—sample a few representative cases and refine once before final output.

**Context:** The check runs in a runner that provides:
- `graph`: A networkx DiGraph. Nodes are file paths (str, relative to repo root). Edges are imports (A -> B means A imports B).
- `nx`: The networkx module (e.g. nx.simple_cycles, nx.descendants).

**Runtime contract:** You only have **`graph`** and **`nx`** (full-repo import structure). You do **not** get “changed files since base”, PR metadata, or add/modify/delete sets unless **your code** obtains them (e.g. `subprocess` + `git` against the repo root given in the user message when needed).

**Directive triage — change-based vs structural:**
- If the **router directive** speaks of **added, modified, deleted, diff, PR, merge base, or comparison to another branch**, do **not** fail just because nodes exist under a path (vendored trees, fixtures, etc. always appear in `graph`). Either run **git** from the repo root (`git diff`, `git diff --name-only`, `git status`, etc.) to detect real changes in those paths, or **`result = (True, "")`** when you cannot implement that faithfully from the graph alone. **Never** treat “files exist under directory D” as “D was illegally modified.”
- If the directive is **structural** (forbidden imports between layers, cycles, package boundaries, “must not depend on”), graph-only checks are appropriate.

**Output contract:** Your code must set `result = (passed, message)` where:
- To **report a violation:** `result = (False, "Clear explanation including file names")`
- To **pass:** `result = (True, "")` or leave result unset (default is pass).

**Execution environment:** The code is run with exec() at module scope (no enclosing function). Therefore:
- Do NOT use `return` — it is invalid outside a function and will raise a SyntaxError. To exit a loop early, set `result = (False, "msg")` and use `break`; then check `result[0]` in an outer loop and break again if needed (see example below).

**Directive interpretation — avoid false positives:**
- "X and Y must exist" or "the only [category] extensions are X and Y" often means X and Y are **required** (must be present), not that **no others** are allowed. Only treat the set as exclusive (fail on any other item) when the rule explicitly says "no other", "only these and no others", or "fail if any other ... exist". When in doubt, enforce only that the required items exist.
- Set `result = (False, "...")` only when the graph evidence clearly violates the stated boundary/structure. If uncertain, refine scope/logic instead of broad failure conditions.

**Common patterns:**
- Layering: "folder A must not import folder B" -> get nodes in A, get nodes in B, for each node in A check graph.successors(node); if any successor is in B, violation.
- Circular dependencies: "No cycles in src/utils" -> subgraph = graph.subgraph(nodes in src/utils), if list(nx.simple_cycles(subgraph)): violation.
- File location: "React components must live under src/components" -> identify component files (e.g. by extension .tsx), check all have "src/components" in path.

**Example (services must not import controllers):**
result = (True, "")
services = [n for n in graph.nodes() if "/services/" in n or n.startswith("services/")]
controllers = [n for n in graph.nodes() if "/controllers/" in n or n.startswith("controllers/")]
for s in services:
    for neighbor in graph.successors(s):
        if neighbor in controllers:
            result = (False, f"Service {s} must not import controller {neighbor}")
            break
    if result[0] is False:
        break
(Use break and result; never use return.)

**Response format:** Reply with a JSON object only, no markdown:
{
  "code": "result = (True, \\\"\\\")\\n# ... your Python code ..."
}

The code string must be valid Python. Use newlines as \\n. Escape quotes. Do not use triple-quoted strings that would break JSON."""


def _build_user_prompt(
    compressed_segment: str,
    trigger: str,
    directive: str,
    repo_root: str | None = None,
) -> str:
    repo_block = ""
    if repo_root and repo_root.strip():
        repo_block = f"""
**Repository root:** `{repo_root.strip()}`
If you can access this directory, inspect layout and imports so the graph check uses real paths and layers. Graph nodes are file paths relative to this root.
"""
    return f"""**Original constraint (compressed):**
{compressed_segment}

**Router directive:** {directive}
**Trigger (file pattern / scope):** {trigger}
{repo_block}
Generate the arch deterministic check JSON (code only) as specified."""


def _parse_response(content: str) -> Dict[str, Any]:
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
    return {"code": code}


def validate_arch_deterministic_code(code: str) -> Tuple[bool, str]:
    """
    Validate generated arch check code: must be valid Python and must not use
    'return' at module scope (code is exec'd without an enclosing function).
    Returns (True, "") if valid, (False, "error message") otherwise.
    """
    if not code or not code.strip():
        return False, "empty code"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    def _has_return_outside_function(node: ast.AST, inside_function: bool) -> bool:
        if isinstance(node, ast.Return):
            return not inside_function
        now_inside = inside_function or isinstance(node, ast.FunctionDef)
        for child in ast.iter_child_nodes(node):
            if _has_return_outside_function(child, now_inside):
                return True
        return False

    if _has_return_outside_function(tree, False):
        return False, "generated code uses 'return' outside a function; use 'break' to exit loops and set result = (False, msg) before breaking"
    return True, ""


def generate_arch_deterministic_check(
    compressed_segment: str,
    strategy: Dict[str, Any],
    *,
    client: Any,
    model: str,
    provider: str = "openai",
    temperature: float = 0.2,
    repo_root: str | None = None,
) -> Dict[str, Any]:
    """Generate Python check code for one ARCH_DETERMINISTIC strategy. Returns { code }."""
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
        {"role": "system", "content": ARCH_DET_SYSTEM_PROMPT},
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
    out = _parse_response(content)
    if not out.get("code"):
        out["code"] = "result = (True, \"\")  # generator produced no code"
    valid, err = validate_arch_deterministic_code(out.get("code", ""))
    if not valid and err:
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": f"The generated check failed validation: {err}. Fix the code (use break instead of return to exit loops) and reply with the same JSON format (code only).",
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
        out2 = _parse_response(content2)
        if out2.get("code"):
            valid2, _ = validate_arch_deterministic_code(out2["code"])
            if valid2:
                out = out2
            else:
                out["code"] = out["code"].rstrip() + f"\n# validation failed: {err}"
        else:
            out["code"] = out["code"].rstrip() + f"\n# validation failed: {err}"
    return out


def _content_from_resp(resp: Any) -> str:
    if getattr(resp, "choices", None) and len(resp.choices) > 0:
        msg = getattr(resp.choices[0], "message", None)
        if msg is not None:
            return (getattr(msg, "content", None) or "").strip()
    return ""


def collect_arch_det_tasks(mapping: Dict[str, Dict[str, Any]]) -> List[tuple]:
    """Collect (entry, idx, compressed, strategy) for every ARCH_DETERMINISTIC without arch_deterministic_check."""
    tasks = []
    for entry in mapping.values():
        if not isinstance(entry, dict):
            continue
        strategies = entry.get("strategies") or []
        compressed = entry.get("compressed") or ""
        for idx, s in enumerate(strategies):
            if not isinstance(s, dict):
                continue
            if (s.get("type") or "").strip().upper() != "ARCH_DETERMINISTIC":
                continue
            if s.get("arch_deterministic_check"):
                continue
            tasks.append((entry, idx, compressed, s))
    return tasks


def generate_arch_deterministic_checks_for_mapping(
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
    """For every ARCH_DETERMINISTIC strategy missing arch_deterministic_check, generate and store code. Mutates mapping."""
    import llm
    from llm import is_claude_code_provider

    tasks = collect_arch_det_tasks(mapping)
    if not tasks:
        return

    cc_cwd = repo_root if repo_root and is_claude_code_provider(provider) else None
    requests = []
    for entry, idx, compressed, strategy in tasks:
        trigger = str(strategy.get("trigger") or "").strip()
        directive = str(strategy.get("directive") or "")
        user_content = _build_user_prompt(
            compressed,
            trigger,
            directive,
            repo_root=repo_root,
        )
        requests.append({
            "client": client,
            "model": model,
            "messages": [
                {"role": "system", "content": ARCH_DET_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
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
        out = _parse_response(content)
        code = out.get("code") or ""
        if not code:
            code = "result = (True, \"\")  # generator produced no code"
        valid, err = validate_arch_deterministic_code(code)
        if not valid and err:
            code = f"result = (True, \"\")  # validation failed: {err}"
        entry = req["_entry"]
        idx = req["_idx"]
        strategies = entry.get("strategies") or []
        if idx < len(strategies):
            strategies[idx]["arch_deterministic_check"] = {"code": code}
