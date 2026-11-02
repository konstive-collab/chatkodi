"""Простые инструменты хронологии для ChatKodi 1.7."""

from __future__ import annotations

from typing import Dict, Iterable, List

from core.fact_extraction import EvidenceRecord


def build_timeline_summary(records: Iterable[EvidenceRecord], limit: int = 8) -> List[Dict[str, object]]:
    per_day: Dict[str, Dict[str, object]] = {}
    for rec in records:
        for date in rec.event_dates or ['без даты']:
            bucket = per_day.setdefault(date, {'date': date, 'count': 0, 'methods': set(), 'targets': set(), 'sources': set()})
            bucket['count'] += 1
            for method in rec.methods:
                bucket['methods'].add(method)
            if rec.target:
                bucket['targets'].add(rec.target)
            if rec.source:
                bucket['sources'].add(rec.source)
    items = []
    for date in sorted(per_day.keys()):
        item = per_day[date]
        items.append({
            'date': date,
            'count': int(item['count']),
            'methods': sorted(item['methods'])[:3],
            'targets': sorted(item['targets'])[:3],
            'sources': sorted(item['sources'])[:3],
        })
    return items[:limit]
