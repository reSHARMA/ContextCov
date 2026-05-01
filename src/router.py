"""
Intent Router: maps a compressed constraint to enforcement strategies.

Takes a natural-language constraint (e.g. from compressed segments) and returns
a list of strategies: SOURCE_CHECK, PROCESS_CHECK, ARCH_DETERMINISTIC (hard
structural gates), ARCH_SEMANTIC (soft LLM-as-judge). Uses ROUTER_LLM or
DEFAULT_LLM. Runs in parallel over many constraints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

ROUTER_SYSTEM_PROMPT = """You are a Senior DevOps Architect. Your goal is to analyze a natural language constraint from an 'Agent README' and determine the best technical strategy to enforce it programmatically.

You have four specialized enforcement engines:

1. SOURCE_CHECK: Analysis of file content (AST, Regex, Text). Best for coding style, forbidden patterns, library usage, imports within file content.
2. PROCESS_CHECK: Analysis of shell commands. Best for forbidden CLI tools, arguments, or environment variables.
3. ARCH_DETERMINISTIC: Structural rules that can be codified (graph/glob). Use when the rule describes a specific file path, import pattern, directory structure, layering, or dependency constraint. These are "hard" gates: fast, reproducible, no LLM at runtime. Examples: "core must not depend on ui", "all .tsx files must be in src/components", "no circular dependencies in src/utils".
4. ARCH_SEMANTIC: Subjective or qualitative rules that require "understanding". Use when the rule describes design patterns, "clean code" principles, or quality that needs an LLM-as-judge. These are "soft" gates: run e.g. on PRs. Examples: "business logic must not leak into controllers", "use descriptive variable names", "error messages must be user-friendly".

**Classification rule for architecture:** If the rule describes a specific file path, import pattern, or directory structure, classify as ARCH_DETERMINISTIC. If the rule describes a subjective quality, design pattern, or clean-code principle that requires reasoning, classify as ARCH_SEMANTIC.

**Input:** A specific constraint extracted from documentation.

**Output:** A JSON object only. No other text.
{
  "strategies": [
    {
      "type": "SOURCE_CHECK" | "PROCESS_CHECK" | "ARCH_DETERMINISTIC" | "ARCH_SEMANTIC",
      "confidence": 0.95,
      "trigger": "string",
      "directive": "string"
    }
  ]
}

**Rules:**
- A single constraint may require multiple strategies.
- If the constraint is vague or unenforceable (e.g., "Write clean code" with no specifics), return {"strategies": []}.
- For 'trigger', be specific: glob patterns for files ("*.ts", "src/core/*") or binary names for processes ("npm").
- **PROCESS_CHECK constraint:** The 'trigger' must be a SINGLE word — the executable (binary) name only (e.g. "npm", "yarn", "pytest"). If the rule targets a subcommand (e.g. "npm test", "yarn format"), set trigger to the binary only ("npm", "yarn") and include the subcommand in the directive (e.g. "Block 'npm test'; require 'pnpm test'"). The process check generator will then produce code that checks args (e.g. args[0] == 'test'). Do NOT use multi-word triggers like "npm test" or "git commit".
- Output only valid JSON. No markdown, no explanation."""

ROUTER_FEW_SHOT_USER = """Constraint: Use `pnpm` for package management."""

ROUTER_FEW_SHOT_ASSISTANT = """{"strategies": [{"type": "PROCESS_CHECK", "confidence": 0.95, "trigger": "npm", "directive": "Block execution of 'npm install' and suggest 'pnpm' instead."}, {"type": "PROCESS_CHECK", "confidence": 0.95, "trigger": "yarn", "directive": "Block execution of 'yarn' and suggest 'pnpm' instead."}, {"type": "SOURCE_CHECK", "confidence": 0.9, "trigger": "package-lock.json", "directive": "Fail if 'package-lock.json' exists; require 'pnpm-lock.yaml'."}]}"""

ROUTER_FEW_SHOT_ARCH_DET_USER = """Constraint: The `core` module must not depend on `ui`."""

ROUTER_FEW_SHOT_ARCH_DET_ASSISTANT = """{"strategies": [{"type": "ARCH_DETERMINISTIC", "confidence": 0.95, "trigger": "src/core/*", "directive": "Build dependency graph; fail if any node under src/core has an edge to any node under src/ui (layering violation)."}]}"""

ROUTER_FEW_SHOT_ARCH_SEM_USER = """Constraint: Functions should be small and do one thing."""

ROUTER_FEW_SHOT_ARCH_SEM_ASSISTANT = """{"strategies": [{"type": "ARCH_SEMANTIC", "confidence": 0.85, "trigger": "*.py", "directive": "LLM-as-judge: review changed functions for single-responsibility; flag functions that do multiple distinct tasks."}]}"""


@dataclass
class Strategy:
    """One enforcement strategy: type, confidence, trigger, directive."""

    type: str  # SOURCE_CHECK | PROCESS_CHECK | ARCH_DETERMINISTIC | ARCH_SEMANTIC
    confidence: float
    trigger: str
    directive: str


@dataclass
class RouterResult:
    """Result of routing one constraint: list of strategies (may be empty)."""

    strategies: List[Strategy] = field(default_factory=list)


def _parse_router_response(content: str) -> RouterResult:
    """Parse LLM JSON response into RouterResult. Returns empty strategies on parse error."""
    if not content or not content.strip():
        return RouterResult()
    text = content.strip()
    # Strip markdown code fence if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return RouterResult()
    strategies_data = data.get("strategies") or []
    strategies: List[Strategy] = []
    for s in strategies_data:
        if not isinstance(s, dict):
            continue
        stype = s.get("type") or "SOURCE_CHECK"
        conf = s.get("confidence")
        if conf is not None:
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.0
        else:
            conf = 0.0
        trigger = str(s.get("trigger") or "")
        directive = str(s.get("directive") or "")
        strategies.append(
            Strategy(type=stype, confidence=conf, trigger=trigger, directive=directive)
        )
    return RouterResult(strategies=strategies)


def _router_messages(constraint_text: str) -> List[dict]:
    """Build messages for router (system + few-shot + user constraint)."""
    return [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": ROUTER_FEW_SHOT_USER},
        {"role": "assistant", "content": ROUTER_FEW_SHOT_ASSISTANT},
        {"role": "user", "content": ROUTER_FEW_SHOT_ARCH_DET_USER},
        {"role": "assistant", "content": ROUTER_FEW_SHOT_ARCH_DET_ASSISTANT},
        {"role": "user", "content": ROUTER_FEW_SHOT_ARCH_SEM_USER},
        {"role": "assistant", "content": ROUTER_FEW_SHOT_ARCH_SEM_ASSISTANT},
        {"role": "user", "content": f"Constraint: {constraint_text}"},
    ]


def route_constraint(
    constraint_text: str,
    *,
    client: Any,
    model: str,
    provider: str = "openai",
    temperature: float = 0.2,
) -> RouterResult:
    """
    Route a single constraint to enforcement strategies via the LLM.

    Returns RouterResult with strategies (or empty list if vague/unenforceable).
    """
    from llm import chat_completion_create

    messages = _router_messages(constraint_text)
    resp = chat_completion_create(
        client=client,
        model=model,
        messages=messages,
        provider=provider,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    content = ""
    if getattr(resp, "choices", None) and len(resp.choices) > 0:
        msg = getattr(resp.choices[0], "message", None)
        if msg is not None:
            content = (getattr(msg, "content", None) or "").strip()
    return _parse_router_response(content)


def route_constraints_parallel(
    constraint_texts: List[str],
    *,
    client: Any,
    model: str,
    provider: str = "openai",
    temperature: float = 0.2,
    max_concurrency: int = 5,
    progress_callback: Optional[Any] = None,
) -> List[RouterResult]:
    """
    Route many constraints in parallel. Returns list of RouterResult in same order.
    """
    if not constraint_texts:
        return []
    from llm import run_parallel_chat_completions

    requests = []
    for text in constraint_texts:
        messages = _router_messages(text)
        requests.append({
            "client": client,
            "model": model,
            "messages": messages,
            "provider": provider,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        })
    responses = run_parallel_chat_completions(
        requests,
        max_concurrency=max_concurrency,
        progress_callback=progress_callback,
    )
    return [_parse_router_response(_content_from_resp(r)) for r in responses]


def _content_from_resp(resp: Any) -> str:
    if getattr(resp, "choices", None) and len(resp.choices) > 0:
        msg = getattr(resp.choices[0], "message", None)
        if msg is not None:
            return (getattr(msg, "content", None) or "").strip()
    return ""


def strategies_to_dict_list(strategies: List[Strategy]) -> List[dict]:
    """Serialize strategies for JSON (e.g. in mapping output)."""
    return [
        {
            "type": s.type,
            "confidence": s.confidence,
            "trigger": s.trigger,
            "directive": s.directive,
        }
        for s in strategies
    ]
