"""
Context-Aware Content Hash (CACH) — StableID for mapping original ↔ compressed segments.

A node is identified by "what it is and where it lives logically", not by position:
  StableID = hash(header_path || normalized_content || dedup_index)

This stays stable when lines shift (e.g. new intro added); only changes when the
context path or the content itself changes.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.markdown_chunks import ScopedSegment


def normalize_content(content: str) -> str:
    """
    Normalize segment content for stable hashing.

    - Strip leading/trailing whitespace.
    - Collapse internal runs of whitespace (including newlines) to a single space.
    """
    if not content:
        return ""
    s = content.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def compute_stable_id(
    header_path: str,
    normalized_content: str,
    dedup_index: int = 0,
) -> str:
    """
    Compute a stable ID for a segment: hash(context_path || content || dedup).

    Same path + same normalized content (+ same dedup_index) → same ID.
    """
    key = f"{header_path}||{normalized_content}||{dedup_index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def assign_stable_ids(segments: List[ScopedSegment]) -> List[Tuple[ScopedSegment, str]]:
    """
    Assign a unique StableID to each segment.

    Segments that share the same (header_path, normalized_content) in one run
    get a deduplication index (0, 1, 2, ...) so their IDs differ. Returns
    (segment, stable_id) for each segment in order.
    """
    # Group by (header_path, normalized_content) to assign dedup indices
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, seg in enumerate(segments):
        norm = normalize_content(seg.content)
        groups[(seg.header_path, norm)].append(i)

    # For each segment index, compute dedup_index within its group
    dedup: List[int] = [0] * len(segments)
    for indices in groups.values():
        for rank, idx in enumerate(indices):
            dedup[idx] = rank

    result: List[Tuple[ScopedSegment, str]] = []
    for i, seg in enumerate(segments):
        norm = normalize_content(seg.content)
        sid = compute_stable_id(seg.header_path, norm, dedup[i])
        result.append((seg, sid))
    return result


def build_compression_mapping(
    segment_stable_id_pairs: List[Tuple[ScopedSegment, str]],
    compressed_texts: List[str],
    strategies_per_item: Optional[List[List[Dict[str, Any]]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build the mapping DB: stable_id -> { header_path, original_content, compressed [, strategies] }.

    segment_stable_id_pairs and compressed_texts must be in the same order (one-to-one).
    If strategies_per_item is provided (same length), each entry gets "strategies": list of strategy dicts.
    """
    if len(segment_stable_id_pairs) != len(compressed_texts):
        raise ValueError(
            "segment_stable_id_pairs and compressed_texts must have the same length"
        )
    if strategies_per_item is not None and len(strategies_per_item) != len(compressed_texts):
        raise ValueError(
            "strategies_per_item must have the same length as compressed_texts"
        )
    mapping: Dict[str, Dict[str, Any]] = {}
    for i, ((seg, stable_id), compressed) in enumerate(zip(segment_stable_id_pairs, compressed_texts)):
        entry: Dict[str, Any] = {
            "stable_id": stable_id,
            "header_path": seg.header_path,
            "original_content": seg.content,
            "compressed": compressed,
        }
        if strategies_per_item is not None:
            entry["strategies"] = strategies_per_item[i]
        mapping[stable_id] = entry
    return mapping
