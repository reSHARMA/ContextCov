"""
LLM-based contextual compression (backward slicing) for scoped segments.

Takes a segment with header path + content and compresses it into a single
semantic statement: "In the context of Header1 and Header2, do Item."

Uses the project's llm module (llm.py) for client config and chat_completion_create
unless a custom chat_create callable is passed. COMPRESSION_LLM or DEFAULT_LLM
determines provider/model/temperature for segment compression. When parallel=True (default), requests
run in parallel with rate-limit retry (see llm.run_parallel_chat_completions).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

# Progress callback: (current_1based: int, total: int) -> None
ProgressCallback = Callable[[int, int], None]

from src.markdown_chunks import ScopedSegment


@dataclass
class CompressedSegment:
    """Result of compressing a scoped segment into one semantic statement."""

    original: ScopedSegment
    compressed: str


# Default system prompt for contextual compression (query rewriting / backward slicing).
DEFAULT_COMPRESSION_SYSTEM = """You are a technical editor. Given a markdown segment that includes a "breadcrumb" (section path) and content, produce a single, clear semantic statement that captures the rule or instruction.

- The last part of the content (the leaf) is the main instruction; the headers are context.
- Remove anything not relevant to that main instruction (backward slicing).
- Output one concise sentence or short paragraph, without markdown formatting or bullet points.
- Preserve technical terms and specifics. Do not add new information."""


def compress_segment(
    segment: ScopedSegment,
    *,
    client: Any,
    model: str,
    temperature: float = 0.2,
    system_prompt: str = DEFAULT_COMPRESSION_SYSTEM,
    chat_create: Optional[Callable[..., Any]] = None,
    provider: str = "openai",
) -> CompressedSegment:
    """
    Compress a single scoped segment into one semantic statement using the LLM.

    client, model, temperature: LLM configuration (e.g. from get_model_and_client).
    chat_create: optional function(client, model, messages, ...) for completions; defaults to standard chat.
    """
    full = segment.full_text(separator=" > ")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full},
    ]
    if chat_create is not None:
        resp = chat_create(
            client=client,
            model=model,
            messages=messages,
            provider=provider,
            temperature=temperature,
        )
    else:
        # Use project llm.py for OpenAI/Azure/etc. chat completions
        from llm import chat_completion_create

        resp = chat_completion_create(
            client=client,
            model=model,
            messages=messages,
            provider=provider,
            temperature=temperature,
        )
    content: str = ""
    if getattr(resp, "choices", None) and len(resp.choices) > 0:
        msg = getattr(resp.choices[0], "message", None)
        if msg is not None:
            content = (getattr(msg, "content", None) or "").strip()
    if not content:
        content = segment.content
    return CompressedSegment(original=segment, compressed=content)


def _content_from_response(resp: Any, fallback: str = "") -> str:
    """Extract message content from a chat completion response; use fallback if missing."""
    if getattr(resp, "choices", None) and len(resp.choices) > 0:
        msg = getattr(resp.choices[0], "message", None)
        if msg is not None:
            content = (getattr(msg, "content", None) or "").strip()
            if content:
                return content
    return fallback


def compress_segments(
    segments: List[ScopedSegment],
    *,
    client: Any,
    model: str,
    temperature: float = 0.2,
    system_prompt: str = DEFAULT_COMPRESSION_SYSTEM,
    chat_create: Optional[Callable[..., Any]] = None,
    provider: str = "openai",
    progress_callback: Optional[ProgressCallback] = None,
    parallel: bool = True,
    max_concurrency: Optional[int] = None,
) -> List[CompressedSegment]:
    """
    Compress each segment; returns list of CompressedSegment in same order.

    When parallel=True (default) and chat_create is not overridden, runs requests
    in parallel via llm.run_parallel_chat_completions with rate-limit retry.
    max_concurrency defaults to CONTEXTCOV_MAX_CONCURRENCY env or 5.
    """
    total = len(segments)
    if total == 0:
        return []

    use_parallel = parallel and chat_create is None
    concurrency = max_concurrency
    if concurrency is None:
        try:
            concurrency = int(os.environ.get("CONTEXTCOV_MAX_CONCURRENCY", "5"))
        except ValueError:
            concurrency = 5
    concurrency = max(1, min(concurrency, total))

    if use_parallel:
        from llm import run_parallel_chat_completions

        requests = []
        for seg in segments:
            full = seg.full_text(separator=" > ")
            requests.append({
                "client": client,
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full},
                ],
                "provider": provider,
                "temperature": temperature,
            })
        responses = run_parallel_chat_completions(
            requests,
            max_concurrency=concurrency,
            progress_callback=progress_callback,
        )
        return [
            CompressedSegment(
                original=seg,
                compressed=_content_from_response(resp, fallback=seg.content),
            )
            for seg, resp in zip(segments, responses)
        ]

    result: List[CompressedSegment] = []
    for i, seg in enumerate(segments):
        if progress_callback is not None:
            progress_callback(i + 1, total)
        result.append(
            compress_segment(
                seg,
                client=client,
                model=model,
                temperature=temperature,
                system_prompt=system_prompt,
                chat_create=chat_create,
                provider=provider,
            )
        )
    return result
