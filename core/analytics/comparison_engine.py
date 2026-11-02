"""Простейшее сравнение периодов для статистических ответов."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable

from core.fact_extraction import EvidenceRecord


def compare_first_last(records: Iterable[EvidenceRecord]) -> Dict[str, object]:
    recs = [rec for rec in records if rec.event_dates]
    if len(recs) < 2:
        return {}
    recs.sort(key=lambda rec: min(rec.event_dates))
    first_date = min(recs[0].event_dates)
    last_date = max(recs[-1].event_dates)
    first = [rec for rec in recs if first_date in rec.event_dates]
    last = [rec for rec in recs if last_date in rec.event_dates]

    def _pack(items):
        methods = Counter(m for rec in items for m in rec.methods)
        targets = Counter(rec.target for rec in items if rec.target)
        return {
            'count': len(items),
            'top_methods': methods.most_common(3),
            'top_targets': targets.most_common(3),
        }

    return {
        'first_date': first_date,
        'last_date': last_date,
        'first': _pack(first),
        'last': _pack(last),
    }
