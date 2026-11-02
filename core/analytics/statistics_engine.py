"""Агрегирующая статистика по извлечённым evidence records.

Важно: статистика не должна автоматически считать каждую строку как отдельный
факт.  Для activity-report единица счёта — «эпизод активности» (обычно
документ/день/раздел объекта), а не все дочерние предложения под заголовком.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Dict, Iterable, List, Any
import re

from core.fact_extraction import EvidenceRecord, dedupe_evidence_records
from core.domain_lexicon import get_domain_lexicon
from core.knowledge_units import get_count_unit
from core.query_understanding import normalize_question


def _top(counter: Counter, limit: int = 5) -> List[tuple[str, int]]:
    return [(str(k), int(v)) for k, v in counter.most_common(limit) if k]


def _natural_source_key(value: str) -> tuple:
    text = str(value or '').replace('\\', '/').split('/')[-1].lower()
    parts = re.split(r'(\d+)', text)
    return tuple(int(x) if x.isdigit() else x for x in parts)


def _ordered_sources(recs: List[EvidenceRecord]) -> List[str]:
    seen = set()
    out: List[str] = []
    for rec in recs:
        source = str(getattr(rec, 'source', '') or '').strip()
        if not source or source in seen:
            continue
        seen.add(source)
        out.append(source)
    return sorted(out, key=_natural_source_key)


def _coverage_by_source(recs: List[EvidenceRecord]) -> List[dict]:
    by_source: Dict[str, dict] = {}
    for rec in recs:
        source = str(getattr(rec, 'source', '') or '').strip()
        if not source:
            continue
        item = by_source.setdefault(source, {'source': source, 'count': 0, 'dates': set(), 'targets': set(), 'methods': set(), 'directions': set()})
        item['count'] += 1
        for d in getattr(rec, 'event_dates', []) or []:
            if d:
                item['dates'].add(str(d))
        if getattr(rec, 'target', ''):
            item['targets'].add(str(rec.target))
        for m in getattr(rec, 'methods', []) or []:
            if m:
                item['methods'].add(str(m))
        for direction in getattr(rec, 'directions', []) or []:
            if direction:
                item['directions'].add(str(direction))
    rows = []
    for item in by_source.values():
        rows.append({
            'source': item['source'],
            'count': int(item['count']),
            'dates': sorted(item['dates']),
            'targets': sorted(item['targets']),
            'methods': sorted(item['methods']),
            'directions': sorted(item['directions']),
        })
    return sorted(rows, key=lambda x: _natural_source_key(x['source']))


def _build_periodicity(top_dates: List[tuple[str, int]]) -> Dict[str, object] | None:
    if not top_dates:
        return None
    counts = [count for _date, count in top_dates]
    active_days = len(counts)
    total = sum(counts)
    peak = max(counts)
    avg = round(total / max(1, active_days), 2)
    if active_days <= 1:
        label = 'единичный эпизод'
    elif peak >= max(3, avg * 1.8):
        label = 'всплеск активности'
    elif active_days >= 4 and peak <= avg * 1.35:
        label = 'регулярная активность'
    else:
        label = 'эпизодическая активность'
    return {
        'label': label,
        'active_days': active_days,
        'avg_per_active_day': avg,
        'peak_per_day': peak,
    }


_DOMAIN = get_domain_lexicon()

_ACTOR_METHOD_RE = re.compile(
    rf"^\s*{_DOMAIN.actor_pattern}\s*(?:№|#)?\s*\d{{1,8}}\b",
    re.IGNORECASE,
)


def _focus_terms(intent: Any | None) -> List[str]:
    if intent is None:
        return []
    out = []
    for item in list(getattr(intent, 'object_terms', []) or []):
        norm = normalize_question(str(item))
        if norm:
            out.append(norm)
    # Для рБпЛА №11 добавляем варианты без точного форматирования.
    q = normalize_question(str(getattr(intent, 'raw_question', '') or ''))
    for ident in (getattr(intent, 'identifiers', []) or []):
        ident = str(ident).lstrip('0') or '0'
        if _DOMAIN.mentions_family(q, 'рбпла'):
            out.extend(_DOMAIN.reference_terms('рбпла', ident))
    return list(dict.fromkeys(out))


def _sanitize_methods(methods: Iterable[str], *, intent: Any | None = None) -> List[str]:
    """Убирает из поля «способ» акторов/разделы, которые не являются методом.

    Пример старой ошибки: для запроса по рБпЛА №11 статистика показывала
    «Способы: рБпЛА №11 — 6», хотя рБпЛА №11 — главный объект отчёта, а не
    способ.  Аналогично не считаем ШГР/ДШБ/группы способами применения.
    """
    focus = _focus_terms(intent)
    out: List[str] = []
    seen = set()
    for method in methods or []:
        value = re.sub(r"\s+", " ", str(method or '').strip(" ,.;:-—–"))
        if not value:
            continue
        norm = normalize_question(value)
        if not norm:
            continue
        if any(term and (norm == term or norm in term or term in norm) for term in focus):
            continue
        if _ACTOR_METHOD_RE.match(value):
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(value)
    return out


def _episode_key(rec: EvidenceRecord, intent: Any | None) -> tuple:
    meta = getattr(rec, 'metadata', {}) or {}
    dates = tuple(sorted(str(x) for x in (getattr(rec, 'event_dates', []) or []) if x))
    if not dates:
        dates = ('undated',)
    directions = tuple(sorted(normalize_question(x) for x in (getattr(rec, 'directions', []) or []) if x))
    locations = tuple(sorted(normalize_question(x) for x in (getattr(rec, 'locations', []) or []) if x))
    place = directions or locations
    section_family = normalize_question(str(meta.get('section_family') or ''))
    section_number = str(meta.get('section_number') or '').lstrip('0')
    focus = tuple(_focus_terms(intent))
    # Для подробной работы одного объекта ключ не должен включать action/target:
    # иначе каждая строка под одним заголовком станет отдельным «эпизодом».
    return (
        str(getattr(rec, 'source', '') or ''),
        dates,
        section_family,
        section_number,
        place,
        focus,
    )


def _merge_episode(a: EvidenceRecord, b: EvidenceRecord, intent: Any | None) -> EvidenceRecord:
    # Оставляем более информативную строку как evidence, но объединяем поля для статистики.
    keep = a
    other = b
    if (len(str(getattr(b, 'sentence', '') or '')), float(getattr(b, 'score', 0.0) or 0.0)) > (len(str(getattr(a, 'sentence', '') or '')), float(getattr(a, 'score', 0.0) or 0.0)):
        keep, other = b, a

    def union(attr: str, limit: int = 12) -> List[str]:
        values = list(getattr(keep, attr, []) or []) + list(getattr(other, attr, []) or [])
        out, seen = [], set()
        for value in values:
            text = re.sub(r"\s+", " ", str(value or '').strip())
            norm = normalize_question(text)
            if not text or norm in seen:
                continue
            seen.add(norm)
            out.append(text)
            if len(out) >= limit:
                break
        return out

    merged = replace(
        keep,
        methods=_sanitize_methods(union('methods'), intent=intent),
        locations=union('locations'),
        directions=union('directions'),
        event_dates=union('event_dates'),
        quantities=list(dict.fromkeys(list(getattr(keep, 'quantities', []) or []) + list(getattr(other, 'quantities', []) or [])))[:12],
        score=max(float(getattr(keep, 'score', 0.0) or 0.0), float(getattr(other, 'score', 0.0) or 0.0)),
        confidence=max(float(getattr(keep, 'confidence', 0.0) or 0.0), float(getattr(other, 'confidence', 0.0) or 0.0)),
    )
    # Если в выбранной evidence-строке нет цели/действия, но в другой есть — подтянем.
    if not getattr(merged, 'target', '') and getattr(other, 'target', ''):
        merged.target = other.target
    if not getattr(merged, 'action', '') and getattr(other, 'action', ''):
        merged.action = other.action
    return merged


def _aggregate_activity_episodes(records: List[EvidenceRecord], intent: Any | None = None) -> List[EvidenceRecord]:
    """Сворачивает дочерние строки одного раздела/дня в один эпизод активности."""
    by_key: Dict[tuple, EvidenceRecord] = {}
    for rec in records or []:
        rec = replace(rec, methods=_sanitize_methods(getattr(rec, 'methods', []) or [], intent=intent))
        key = _episode_key(rec, intent)
        if key in by_key:
            by_key[key] = _merge_episode(by_key[key], rec, intent)
        else:
            by_key[key] = rec
    out = list(by_key.values())
    out.sort(key=lambda x: (_natural_source_key(getattr(x, 'source', '') or ''), tuple(getattr(x, 'event_dates', []) or [])))
    return out


def build_statistics(records: Iterable[EvidenceRecord], *, count_unit: str = "claim", intent: Any | None = None) -> Dict[str, object]:
    unit = get_count_unit(count_unit)
    raw_recs = list(records or [])
    recs = dedupe_evidence_records(raw_recs)
    raw_count_after_dedupe = len(recs)
    if unit.key == 'activity_episode':
        recs = _aggregate_activity_episodes(recs, intent=intent)

    methods = Counter()
    directions = Counter()
    locations = Counter()
    actions = Counter()
    targets = Counter()
    by_source = Counter()
    by_date = Counter()
    quantities = []
    method_target_pairs = Counter()
    direction_method_pairs = Counter()
    polarity = Counter()
    date_polarities: Dict[str, set[str]] = {}

    for rec in recs:
        by_source[str(rec.source)] += 1
        polarity[str(rec.polarity or 'neutral')] += 1
        if rec.action:
            actions[rec.action] += 1
        if rec.target:
            targets[rec.target] += 1
        for item in _sanitize_methods(getattr(rec, 'methods', []) or [], intent=intent):
            methods[item] += 1
            if rec.target:
                method_target_pairs[f"{item} → {rec.target}"] += 1
            for direction in rec.directions:
                direction_method_pairs[f"{direction} → {item}"] += 1
        for item in rec.directions:
            directions[item] += 1
        for item in rec.locations:
            locations[item] += 1
        for item in rec.event_dates:
            date_polarities.setdefault(str(item), set()).add(str(rec.polarity or "neutral"))
            if str(rec.polarity or "neutral") != "negative":
                by_date[item] += 1
        quantities.extend(rec.quantities)

    active_dates = sorted(by_date.items(), key=lambda x: x[0])
    timeline = [{"date": d, "count": int(c)} for d, c in active_dates[:24]]
    date_span = None
    if active_dates:
        date_span = {"start": active_dates[0][0], "end": active_dates[-1][0], "active_days": len(active_dates)}

    avg_quantity = round(sum(quantities) / len(quantities), 2) if quantities else None
    max_quantity = max(quantities) if quantities else None
    top_dates = _top(by_date, limit=32)
    periodicity = _build_periodicity(top_dates)
    if periodicity is None and len(recs) >= 2:
        periodicity = {
            'label': 'повторяемая активность',
            'active_days': len(top_dates) or 0,
            'avg_per_active_day': round(len(recs), 2),
            'peak_per_day': len(recs),
        }

    dominant = []
    for label, values in [
        ('способ', _top(methods, 1)),
        ('направление', _top(directions, 1)),
        ('цель', _top(targets, 1)),
        ('действие', _top(actions, 1)),
    ]:
        if values:
            dominant.append({'kind': label, 'value': values[0][0], 'count': values[0][1]})

    total_records = len(recs)
    if unit.key == "date":
        total_records = len(by_date)
    elif unit.key == "source_document":
        total_records = len(by_source)
    elif unit.key == "target":
        total_records = len(targets)
    elif unit.key == "method":
        total_records = len(methods)
    elif unit.key == "location":
        total_records = len(set(directions) | set(locations))
    conflicting_dates = sorted(
        date for date, values in date_polarities.items()
        if "positive" in values and "negative" in values
    )

    return {
        'total_records': total_records,
        'raw_fact_records': len(raw_recs),
        'raw_deduped_fact_records': raw_count_after_dedupe,
        'count_unit': unit.key,
        'count_unit_label': unit.label,
        'count_unit_short': unit.short,
        'count_unit_description': unit.description,
        'top_methods': _top(methods, limit=10),
        'top_directions': _top(directions, limit=10),
        'top_locations': _top(locations, limit=10),
        'top_actions': _top(actions, limit=10),
        'top_targets': _top(targets, limit=12),
        'top_sources': _top(by_source, limit=32),
        'all_sources': _ordered_sources(recs),
        'source_coverage': _coverage_by_source(recs),
        'timeline': timeline,
        'top_dates': top_dates,
        'date_span': date_span,
        'avg_quantity': avg_quantity,
        'max_quantity': max_quantity,
        'top_method_target_pairs': _top(method_target_pairs),
        'top_direction_method_pairs': _top(direction_method_pairs),
        'polarity': dict(polarity),
        'conflicting_dates': conflicting_dates,
        'dominant_patterns': dominant,
        'periodicity': periodicity,
    }
