"""
Hierarchical markdown chunking (context-aware chunking) via AST.

We parse markdown to an AST (mistletoe), then walk the tree and emit one
ScopedSegment per content node (paragraph, list item, heading, code block).
Each segment carries the path of ancestor headings (parents) so the rest of
the pipeline (e.g. LLM compression) can use that context unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from mistletoe import Document
from mistletoe.block_token import (
    BlockCode,
    CodeFence,
    Heading,
    ListItem,
    Paragraph,
)


@dataclass(frozen=True)
class ScopedSegment:
    """
    A single segment with hierarchical context (breadcrumb).

    header_path: parent headers leading to this content (e.g. "Guidelines / Python / Naming").
    content: the markdown content of this segment (e.g. "- Use snake_case").
    """

    header_path: str
    content: str

    def breadcrumb_display(self, separator: str = " > ") -> str:
        """Return header_path with a readable separator."""
        if not self.header_path or self.header_path == "/":
            return ""
        return self.header_path.strip("/").replace("/", separator)

    def full_text(self, separator: str = " > ") -> str:
        """Content prefixed by breadcrumb for LLM context."""
        bc = self.breadcrumb_display(separator=separator)
        if not bc:
            return self.content
        return f"[{bc}]\n{self.content}"


def _node_to_text(node: Any) -> str:
    """Recursively get plain text from an AST node (block or span)."""
    if node is None:
        return ""
    # A line break inside a paragraph or list item is a leaf with empty content.
    # Without this, joining children with "" glues the surrounding words together
    # ("tab charactersfor indentation") for any soft-wrapped source line.
    if type(node).__name__ == "LineBreak":
        return " " if getattr(node, "soft", True) else "\n"
    # Prefer .content when present and node is effectively a leaf (no/empty children)
    if hasattr(node, "content"):
        children = getattr(node, "children", None)
        if not children:
            return getattr(node, "content", "") or ""
    children = getattr(node, "children", None)
    if children:
        return "".join(_node_to_text(c) for c in children)
    return ""


def _walk_ast(
    node: Any,
    header_stack: List[tuple[int, str]],
    header_path_separator: str,
    segments: List[ScopedSegment],
) -> None:
    """
    Walk the markdown AST; accumulate ancestor headings and emit one
    ScopedSegment per content node (Heading, Paragraph, ListItem, CodeFence, BlockCode).
    """
    node_type = type(node).__name__

    if node_type == "Heading":
        title = _node_to_text(node).strip()
        level = getattr(node, "level", 6)
        # Pop stack until we're at a lower level, then push this heading
        while header_stack and header_stack[-1][0] >= level:
            header_stack.pop()
        header_stack.append((level, title))
        # Optionally emit the heading itself as a segment (section title in context)
        path = header_path_separator.join(t for _, t in header_stack)
        if title:
            segments.append(ScopedSegment(header_path=path, content=title))
        return

    if node_type == "Paragraph":
        path = header_path_separator.join(t for _, t in header_stack)
        text = _node_to_text(node).strip()
        if text:
            segments.append(ScopedSegment(header_path=path, content=text))
        return

    if node_type == "ListItem":
        path = header_path_separator.join(t for _, t in header_stack)
        text = _node_to_text(node).strip()
        if text:
            segments.append(ScopedSegment(header_path=path, content=text))
        return

    if node_type in ("CodeFence", "BlockCode"):
        path = header_path_separator.join(t for _, t in header_stack)
        content = getattr(node, "content", None) or _node_to_text(node)
        if content and content.strip():
            segments.append(ScopedSegment(header_path=path, content=content.strip()))
        return

    # Container nodes: recurse into children (Document, List, Quote, etc.)
    children = getattr(node, "children", None)
    if children:
        for child in children:
            _walk_ast(child, header_stack, header_path_separator, segments)


_CONTENT_NODE_TYPES = ("Heading", "Paragraph", "ListItem", "CodeFence", "BlockCode")


def _count_ast_leaves(node: Any, count: List[int]) -> None:
    """Walk AST and count block-level content nodes (leaf nodes in the block tree)."""
    node_type = type(node).__name__
    if node_type in _CONTENT_NODE_TYPES:
        count[0] += 1
        return
    children = getattr(node, "children", None)
    if children:
        for child in children:
            _count_ast_leaves(child, count)


def count_markdown_ast_leaf_nodes(markdown_text: str) -> int:
    """
    Return the number of leaf (content) nodes in the markdown AST.
    These are block-level nodes: Heading, Paragraph, ListItem, CodeFence, BlockCode.
    """
    doc = Document(markdown_text)
    count: List[int] = [0]
    _count_ast_leaves(doc, count)
    return count[0]


def parse_markdown_to_scoped_segments(
    markdown_text: str,
    header_path_separator: str = "/",
) -> List[ScopedSegment]:
    """
    Parse markdown into scoped segments (context-aware chunks), as granular as possible.

    Uses an AST (mistletoe): we walk the tree and emit one segment per content node
    (heading, paragraph, list item, code block). Each segment carries the path of
    all ancestor headings (parents). The rest of the pipeline (e.g. LLM compression)
    uses these segments and their header_path unchanged.
    """
    doc = Document(markdown_text)
    header_stack: List[tuple[int, str]] = []
    segments: List[ScopedSegment] = []
    _walk_ast(doc, header_stack, header_path_separator, segments)
    return segments
