"""
Soft-wrapped source lines must not have their words glued together.

Instruction files are commonly wrapped at 80 columns. The span-token walker
joins a node's children with "", and a soft line break is a leaf with empty
content, so without explicit handling "tab characters\n  for indentation"
became "tab charactersfor indentation" - corrupting the text sent to the LLM
and changing the segment's StableID on a pure reflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.markdown_chunks import parse_markdown_to_scoped_segments  # noqa: E402


def _contents(markdown: str) -> list[str]:
    return [seg.content for seg in parse_markdown_to_scoped_segments(markdown)]


def test_soft_wrapped_bullet_keeps_word_separation() -> None:
    markdown = "# A\n\n## Style\n- Never use tab characters\n  for indentation.\n"

    assert "Never use tab characters for indentation." in _contents(markdown)


def test_wrapped_and_unwrapped_produce_the_same_text() -> None:
    wrapped = "# A\n\n## Style\n- Prefer four spaces\n  over tabs\n  everywhere.\n"
    flat = "# A\n\n## Style\n- Prefer four spaces over tabs everywhere.\n"

    assert _contents(wrapped) == _contents(flat)


def test_wrapped_paragraph_keeps_word_separation() -> None:
    markdown = "# A\n\n## Notes\nThis rule applies to every\nmodule in the repository.\n"

    assert "This rule applies to every module in the repository." in _contents(markdown)
