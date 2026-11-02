"""Retrieval profiles derived from the shared deterministic query analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from core.query_understanding import QueryUnderstandingResult, analyze_question


@dataclass
class IntentProfile:
    name: str
    description: str = ""
    retrieval_k_base: int = 80
    retrieval_k_max: int = 480
    context_max_docs: int = 6
    context_max_chars: int = 32000
    date_strict: bool = False


def profile_name_for_intent(intent: QueryUnderstandingResult) -> str:
    """Map the shared answer intent to a retrieval tuning profile."""
    if intent.question_type == "comparison":
        return "compare"
    if intent.has_date_constraint:
        return "date_fact"
    if intent.identifiers:
        return "id_fact"
    if intent.question_type in {
        "summary",
        "range_summary",
        "timeline",
        "activity_report",
        "frequency",
        "count",
        "target",
    }:
        return "summary"
    return "fact_lookup"


class IntentRouter:
    """Select retrieval limits without reclassifying the user's question."""

    def __init__(self, intents_cfg: Dict[str, Any]):
        self.enabled = bool((intents_cfg or {}).get("enabled", True))
        self.profiles_cfg = (intents_cfg or {}).get("profiles", {}) or {}

    def _load_profile(self, name: str) -> IntentProfile:
        payload = self.profiles_cfg.get(name, {}) or {}
        return IntentProfile(
            name=name,
            description=str(payload.get("description", "")),
            retrieval_k_base=int(payload.get("retrieval_k_base", 80)),
            retrieval_k_max=int(payload.get("retrieval_k_max", 480)),
            context_max_docs=int(payload.get("context_max_docs", 6)),
            context_max_chars=int(payload.get("context_max_chars", 32000)),
            date_strict=bool(payload.get("date_strict", False)),
        )

    def route(self, question: str) -> IntentProfile:
        if not self.enabled:
            return IntentProfile(name="default")
        return self._load_profile(profile_name_for_intent(analyze_question(question)))


__all__ = ["IntentProfile", "IntentRouter", "profile_name_for_intent"]
