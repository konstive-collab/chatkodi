"""Shared parsing helpers for structured local-LLM output."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object without greedy brace matching."""
    text = (raw or "").strip()
    if not text:
        return None
    text = re.sub(r"^\s*\`\`\`(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\`\`\`\s*$", "", text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start():])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


__all__ = ["extract_json_object"]
