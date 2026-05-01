"""
Tests for the substrate of incremental check updates.

A segment's StableID is hash(header_path || normalized_content || dedup_index).
`python -m src.cli` reuses a prior mapping entry whenever its StableID reappears,
so unchanged sections keep their compression, routing and generated checks and
cost no LLM calls. These tests pin the property that makes that safe: editing one
section must not perturb the IDs of the others.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.markdown_chunks import parse_markdown_to_scoped_segments  # noqa: E402
from src.stable_id import assign_stable_ids, compute_stable_id, normalize_content  # noqa: E402

BASE = """# Agent Instructions

## Code style
- Never use tab characters for indentation.

## Architecture
- web/ must not import core/ directly.

## Workflow
- Never run `git push --force`.
"""


def _ids(markdown: str) -> dict[str, str]:
    """header_path -> stable_id for each segment."""
    segments = parse_markdown_to_scoped_segments(markdown)
    return {seg.header_path: sid for seg, sid in assign_stable_ids(segments)}


def test_editing_one_section_leaves_other_ids_untouched() -> None:
    edited = BASE.replace(
        "- Never run `git push --force`.",
        "- Never run `git push --force` or `git rebase`.",
    )
    before, after = _ids(BASE), _ids(edited)

    changed = {path for path in before if path in after and before[path] != after[path]}
    # Only the Workflow section may change; everything else must be reusable.
    assert all("Workflow" in path for path in changed), changed
    assert changed, "the edited section's id should have changed"


def test_inserting_a_section_does_not_shift_existing_ids() -> None:
    inserted = BASE.replace(
        "## Architecture",
        "## Testing\n- Run the suite before committing.\n\n## Architecture",
    )
    before, after = _ids(BASE), _ids(inserted)

    for path, sid in before.items():
        assert after.get(path) == sid, f"{path} shifted when an unrelated section was added"


def test_reformatting_whitespace_does_not_change_the_id() -> None:
    reflowed = BASE.replace(
        "- Never use tab characters for indentation.",
        "- Never use tab characters\n  for indentation.",
    )
    before, after = _ids(BASE), _ids(reflowed)

    style = [p for p in before if "Code style" in p]
    assert style, "expected a Code style segment"
    for path in style:
        assert after.get(path) == before[path], "whitespace-only reflow must not invalidate the entry"


def test_same_text_under_a_different_heading_gets_a_different_id() -> None:
    text = normalize_content("- Never use tabs.")
    assert compute_stable_id("Agent Instructions > Code style", text) != compute_stable_id(
        "Agent Instructions > Workflow", text
    )
