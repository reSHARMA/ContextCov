"""
Arch Semantic Check Generator: produces a review rubric (description) for ARCH_SEMANTIC strategies.

The generator does not produce code; it produces a precise instruction for an LLM Code Reviewer
that will later judge a diff. Uses ARCH_SEMANTIC_LLM or DEFAULT_LLM.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

ARCH_SEM_SYSTEM_PROMPT = """You are a Technical Lead. The user gave a fuzzy or subjective rule. Translate it into a precise, actionable instruction for an LLM Code Reviewer.

**Runtime contract:** The reviewer may receive **either** (1) a **Git diff** or (2) a **bounded concatenated source snapshot** labeled as such — not necessarily a PR title/body/checklist. The rubric must be **usable in both cases**.

**Diff vs snapshot:** When the input is **snapshot-only** (no PR-only context), instruct the reviewer **not** to require PR titles, PR descriptions, review checklist items, or “this change must …” wording that only makes sense with a full PR. Judge only what appears in the supplied diff or snapshot.

**Important:** Keep criteria **narrow** so legitimate code is not flagged. Prefer pass/fail tests the reviewer can apply consistently to the **actual** paths and excerpts shown. When in doubt, err toward fewer false positives.

**Repository awareness:** If you can access and inspect the codebase, use typical paths, modules, and conventions you find there when wording the rubric so it matches how this project is organized.

**In-prompt self-validation loop (required):** Before finalizing your rubric, test it mentally against a few representative code snippets/diff hunks from this repo style (both compliant and non-compliant). If the rubric would likely flag too many legitimate changes, tighten criteria, scope, and examples. You do not need to enumerate every edge case; sample a few representative cases to reduce false positives, then refine once before output.

**Output:** A single "description" string. This will be shown to the reviewer along with the diff or snapshot. The description should:
- Be specific enough that the reviewer can decide pass/fail (e.g. "Check all string literals inside catch blocks; flag technical jargon or unhelpful codes; suggest plain-language alternatives").
- Focus on what to look for in the code/diff, not on tooling.

**Examples:**
- Input: "Make sure error messages are user-friendly."
  Output: "In the supplied diff or excerpts, check string literals inside error handling (try/catch, .catch()). Flag technical jargon (e.g. NullPointerException) or unhelpful codes (e.g. EXAMPLE_ERR). Suggest plain-language alternatives."
- Input: "Functions should be small and do one thing."
  Output: "For each function shown in the supplied diff or excerpts, check if it has a single clear responsibility. Flag functions that mix multiple concerns (e.g. I/O + parsing + formatting). Suggest splitting if > 30 lines or > 3 logical sections."
- Input: "Use descriptive variable names."
  Output: "In the supplied diff or code excerpts, flag single-letter names (except i,j in loops), vague names (data, thing, temp), or abbreviations that obscure meaning. Suggest more descriptive names."

**Response format:** Reply with a JSON object only, no markdown:
{
  "description": "Your precise review instruction in one or two sentences."
}"""


def _build_user_prompt(
    compressed_segment: str,
    directive: str,
    repo_root: str | None = None,
) -> str:
    repo_block = ""
    if repo_root and repo_root.strip():
        repo_block = f"""
**Repository root:** `{repo_root.strip()}`
If you can access this directory, review layout and conventions so the rubric fits this project and stays specific enough to limit false positives.
"""
    return f"""**Original constraint (compressed):**
{compressed_segment}

**Router directive:** {directive}
{repo_block}
Generate the review rubric (description only) as specified."""


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
    desc = data.get("description") or ""
    if not isinstance(desc, str):
        desc = str(desc)
    return {"description": desc.strip()}


def generate_arch_semantic_check(
    compressed_segment: str,
    strategy: Dict[str, Any],
    *,
    client: Any,
    model: str,
    provider: str = "openai",
    temperature: float = 0.2,
    repo_root: str | None = None,
) -> Dict[str, Any]:
    """Generate review rubric for one ARCH_SEMANTIC strategy. Returns { description }."""
    from llm import chat_completion_create, is_claude_code_provider

    directive = str(strategy.get("directive") or "")
    user_content = _build_user_prompt(
        compressed_segment,
        directive,
        repo_root=repo_root,
    )
    cc_cwd = repo_root if repo_root and is_claude_code_provider(provider) else None
    messages = [
        {"role": "system", "content": ARCH_SEM_SYSTEM_PROMPT},
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
    if not out.get("description"):
        out["description"] = "Review the diff for adherence to the stated directive."
    return out


def _content_from_resp(resp: Any) -> str:
    if getattr(resp, "choices", None) and len(resp.choices) > 0:
        msg = getattr(resp.choices[0], "message", None)
        if msg is not None:
            return (getattr(msg, "content", None) or "").strip()
    return ""


def collect_arch_sem_tasks(mapping: Dict[str, Dict[str, Any]]) -> List[tuple]:
    """Collect (entry, idx, compressed, strategy) for every ARCH_SEMANTIC without arch_semantic_check."""
    tasks = []
    for entry in mapping.values():
        if not isinstance(entry, dict):
            continue
        strategies = entry.get("strategies") or []
        compressed = entry.get("compressed") or ""
        for idx, s in enumerate(strategies):
            if not isinstance(s, dict):
                continue
            if (s.get("type") or "").strip().upper() != "ARCH_SEMANTIC":
                continue
            if s.get("arch_semantic_check"):
                continue
            tasks.append((entry, idx, compressed, s))
    return tasks


def generate_arch_semantic_checks_for_mapping(
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
    """For every ARCH_SEMANTIC strategy missing arch_semantic_check, generate and store description. Mutates mapping."""
    import llm
    from llm import is_claude_code_provider

    tasks = collect_arch_sem_tasks(mapping)
    if not tasks:
        return

    cc_cwd = repo_root if repo_root and is_claude_code_provider(provider) else None
    requests = []
    for entry, idx, compressed, strategy in tasks:
        directive = str(strategy.get("directive") or "")
        user_content = _build_user_prompt(
            compressed,
            directive,
            repo_root=repo_root,
        )
        requests.append({
            "client": client,
            "model": model,
            "messages": [
                {"role": "system", "content": ARCH_SEM_SYSTEM_PROMPT},
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
        if not out.get("description"):
            out["description"] = "Review the diff for adherence to the stated directive."
        entry = req["_entry"]
        idx = req["_idx"]
        strategies = entry.get("strategies") or []
        if idx < len(strategies):
            strategies[idx]["arch_semantic_check"] = {"description": out.get("description", "")}
