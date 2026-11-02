"""Shared interpretation of retrieval metadata signals."""

from __future__ import annotations

from typing import Any, Mapping


_BOOLEAN_ANCHOR_KEYS = (
    "answer_rescue",
    "anchor_match",
    "date_match",
    "id_match",
    "pinpoint_window",
)
_COUNT_ANCHOR_KEYS = (
    "object_term_hits",
    "entity_match_count",
    "structured_entity_match_count",
)


def is_anchor_chunk(meta: Mapping[str, Any] | None) -> bool:
    """Return whether retrieval metadata contains any strong anchor signal."""
    data = meta or {}
    if any(bool(data.get(key)) for key in _BOOLEAN_ANCHOR_KEYS):
        return True
    if bool(data.get("matched_ids")):
        return True
    for key in _COUNT_ANCHOR_KEYS:
        try:
            if int(data.get(key, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False
