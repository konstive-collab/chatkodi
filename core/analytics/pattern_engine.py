"""Выделение повторяющихся паттернов поверх evidence records."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List

from core.fact_extraction import EvidenceRecord


def build_pattern_summary(records: Iterable[EvidenceRecord], limit: int = 6) -> List[Dict[str, object]]:
    counter = Counter()
    for rec in records:
        methods = rec.methods or ['без метода']
        directions = rec.directions or ['без направления']
        targets = [rec.target] if rec.target else ['без цели']
        for method in methods[:2]:
            for direction in directions[:1]:
                for target in targets[:1]:
                    counter[f'{direction} | {method} | {target}'] += 1
    return [{'pattern': key, 'count': int(value)} for key, value in counter.most_common(limit)]
