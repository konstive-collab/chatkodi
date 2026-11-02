"""Adaptive evidence budget and diversity selection for ChatKodi 1.7.5.

The goal is simple: do not let RAG/statistics collapse to one strong chunk,
but also never send an unbounded corpus to the LLM.  This module is purely
algorithmic and safe to use without LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple
import re


@dataclass(frozen=True)
class EvidenceBudget:
    max_docs: int
    chunks_per_doc: int
    max_records: int
    max_context_chunks: int
    max_context_chars: int
    require_diversity: bool = True
    reason: str = "balanced"
    notes: List[str] = field(default_factory=list)


def make_evidence_budget(intent: Any, *, corpus_docs: int = 0, total_chunks: int = 0, cfg: Dict[str, Any] | None = None) -> EvidenceBudget:
    """Return a conservative but query-aware budget.

    Range/statistical questions need breadth; exact facts need precision.
    The values are intentionally bounded to keep memory stable on 100+ docs.
    """
    cfg = cfg or {}
    qtype = getattr(intent, "question_type", "fact")
    is_range = bool(getattr(intent, "wants_range_aggregation", False) or getattr(intent, "date_mode", "none") in {"range", "discrete_dates"})
    wants_stats = bool(getattr(intent, "wants_statistics", False))
    wants_summary = bool(getattr(intent, "wants_summary", False))
    notes: List[str] = []

    if is_range and qtype in {"location", "method"}:
        # Вопросы вида "на каких направлениях работал X с ... по ..." требуют периода,
        # но не должны превращаться в широкую сводку по всему корпусу.
        max_docs = int(cfg.get("range_exact_max_docs", 8))
        chunks_per_doc = int(cfg.get("range_exact_chunks_per_doc", 4))
        max_records = 90
        max_context_chunks = 12
        max_context_chars = 10000
        reason = "range_scoped_exact"
        notes.append("range location/method query: scope by period, answer compactly")
    elif is_range and qtype == "target":
        max_docs = int(cfg.get("target_range_max_docs", 20))
        chunks_per_doc = int(cfg.get("target_range_chunks_per_doc", 6))
        max_records = 260
        max_context_chunks = 24
        max_context_chars = 16000
        reason = "target_range_listing"
        notes.append("target listing over range: collect all matching target evidence")
    elif is_range and qtype == "activity_report":
        max_docs = int(cfg.get("activity_range_max_docs", 18))
        chunks_per_doc = int(cfg.get("activity_range_chunks_per_doc", 7))
        max_records = 260
        max_context_chunks = 28
        max_context_chars = 17000
        reason = "activity_report_breadth"
        notes.append("activity report: collect actions/states for one entity across the whole period")
    elif is_range and qtype in {"fact", "binary_fact"}:
        max_docs = int(cfg.get("range_fact_max_docs", 10))
        chunks_per_doc = int(cfg.get("range_fact_chunks_per_doc", 4))
        max_records = 120
        max_context_chunks = 12
        max_context_chars = 10000
        reason = "range_scoped_fact"
        notes.append("date range scopes an exact fact query without enabling aggregation")
    elif is_range:
        max_docs = int(cfg.get("range_max_docs", 14))
        chunks_per_doc = int(cfg.get("range_chunks_per_doc", 7))
        max_records = 220
        max_context_chunks = 28
        max_context_chars = 18000
        reason = "range_breadth"
        notes.append("range query: keep multi-day/multi-document evidence")
    elif wants_stats or qtype in {"frequency", "count", "timeline", "comparison", "target", "activity_report"}:
        max_docs = int(cfg.get("stats_max_docs", 10))
        chunks_per_doc = int(cfg.get("stats_chunks_per_doc", 6))
        max_records = 180
        max_context_chunks = 22
        max_context_chars = 15000
        reason = "statistics_breadth"
        notes.append("statistics query: aggregate before presenting")
    elif qtype in {"binary_fact", "location", "method", "fact"}:
        max_docs = int(cfg.get("exact_max_docs", 6))
        chunks_per_doc = int(cfg.get("exact_chunks_per_doc", 3))
        max_records = 90
        max_context_chunks = 9
        max_context_chars = 9000
        reason = "exact_precision"
        notes.append("exact query: prefer best evidence")
    elif wants_summary or qtype in {"summary", "range_summary"}:
        max_docs = 10
        chunks_per_doc = 5
        max_records = 180
        max_context_chunks = 20
        max_context_chars = 16000
        reason = "summary_balanced"
    else:
        max_docs = 7
        chunks_per_doc = 4
        max_records = 120
        max_context_chunks = 12
        max_context_chars = 11000
        reason = "default_balanced"

    if corpus_docs >= 100 or total_chunks >= 900:
        # Big corpora need deterministic aggregation and smaller prompt contexts.
        max_context_chars = min(max_context_chars, 14000)
        max_context_chunks = min(max_context_chunks, 20)
        notes.append("large corpus: context capped")

    return EvidenceBudget(
        max_docs=max(1, max_docs),
        chunks_per_doc=max(1, chunks_per_doc),
        max_records=max(20, max_records),
        max_context_chunks=max(3, max_context_chunks),
        max_context_chars=max(3000, max_context_chars),
        require_diversity=bool(is_range or wants_stats or wants_summary),
        reason=reason,
        notes=notes,
    )


def _text_key(text: str, size: int = 220) -> str:
    t = re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()
    return t[:size]


def diversify_chunks(items: Iterable[Tuple[float, Dict[str, Any]]], *, max_items: int, per_doc_cap: int = 4) -> List[Tuple[float, Dict[str, Any]]]:
    """Small MMR-like selector: keep strong chunks, but prevent one doc/chunk from dominating."""
    ranked = sorted(list(items or []), key=lambda x: float(x[0] or 0.0), reverse=True)
    selected: List[Tuple[float, Dict[str, Any]]] = []
    per_doc: Dict[str, int] = {}
    seen_text = set()

    for score, meta in ranked:
        if len(selected) >= max_items:
            break
        meta = dict(meta or {})
        doc_id = str(meta.get("doc_id") or meta.get("source") or meta.get("doc_name") or "")
        if doc_id and per_doc.get(doc_id, 0) >= per_doc_cap:
            continue
        text = str(meta.get("text") or "")
        key = _text_key(text)
        if key and key in seen_text:
            continue
        if key:
            seen_text.add(key)
        selected.append((float(score), meta))
        if doc_id:
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1

    if len(selected) < max_items:
        # Backfill if diversity filters were too strict.
        existing = {(str(m.get("doc_id")), m.get("chunk_id"), m.get("start_char"), m.get("end_char")) for _, m in selected}
        for score, meta in ranked:
            key = (str((meta or {}).get("doc_id")), (meta or {}).get("chunk_id"), (meta or {}).get("start_char"), (meta or {}).get("end_char"))
            if key in existing:
                continue
            selected.append((float(score), dict(meta or {})))
            existing.add(key)
            if len(selected) >= max_items:
                break
    return selected
