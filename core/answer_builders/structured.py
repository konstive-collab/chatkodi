"""Сборка детерминированных ответов поверх evidence records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re

from core.analytics import build_timeline_summary, build_pattern_summary, compare_first_last
from core.fact_extraction import EvidenceRecord
from core.query_understanding import QueryUnderstandingResult


@dataclass
class StructuredAnswer:
    text: str
    mode: str
    evidence: List[EvidenceRecord] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)
    confidence: float = 0.0


def _choose_positive(records: List[EvidenceRecord]) -> Optional[EvidenceRecord]:
    for rec in records:
        if rec.polarity != 'negative':
            return rec
    return records[0] if records else None


def _choose_location(records: List[EvidenceRecord]) -> Optional[str]:
    for rec in records:
        if rec.directions:
            return rec.directions[0]
        if rec.locations:
            return rec.locations[0]
    return None


def _choose_method(records: List[EvidenceRecord]) -> Optional[str]:
    for rec in records:
        if rec.methods:
            return rec.methods[0]
    return None


def _build_metric_fact(intent: QueryUnderstandingResult, records: List[EvidenceRecord], stats: Dict[str, object]) -> Optional[StructuredAnswer]:
    indicators = set(getattr(getattr(intent, "contract", None), "indicators", ()) or ())
    if not indicators:
        return None
    for rec in records:
        if rec.indicator in indicators and rec.value:
            return StructuredAnswer(
                text=rec.sentence.strip(),
                mode="metric_fact",
                evidence=[rec],
                stats=stats,
                confidence=max(0.86, float(rec.confidence or 0.0)),
            )
    return None


def _fmt_top(values, limit: int = 3, unit: str = 'ед.') -> str:
    values = list(values or [])
    if not values:
        return ''
    return ', '.join(f'{name} — {count} {unit}' for name, count in values[:limit])


def _fmt_names(values, limit: int = 8) -> str:
    names: List[str] = []
    for item in list(values or []):
        if isinstance(item, (list, tuple)) and item:
            name = str(item[0]).strip()
        else:
            name = str(item).strip()
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return ', '.join(names)


def _wants_visible_numbers(intent: QueryUnderstandingResult) -> bool:
    """True только когда пользователь явно просит цифры/статистику.

    По умолчанию ChatKodi отвечает человеческим выводом, а не числовой
    простынёй. Запросы вида «проанализируй» дают краткую интерпретацию;
    подробные числа включаются только явным запросом на статистику/детали.
    """
    q = (getattr(intent, 'normalized_question', '') or '').lower()
    markers = (
        'сколько', 'количество', 'статист', 'таблиц', 'подроб', 'деталь',
        'все источники', 'все документы', 'все доказательства', 'покажи цифры',
        'по дням', 'таймлайн', 'хронолог', 'с какой периодич', 'как часто',
        'частот', 'чаще всего', 'в какие дни чаще', 'где чаще', 'чем чаще',
    )
    return any(m in q for m in markers)



def _stats_unit_label(stats: Dict[str, object]) -> str:
    return str((stats or {}).get('count_unit_label') or 'подтверждённых фактов')


def _stats_unit_short(stats: Dict[str, object]) -> str:
    return str((stats or {}).get('count_unit_short') or 'факт.')


def _record_date_key(rec: EvidenceRecord) -> str:
    dates = list(getattr(rec, 'event_dates', []) or [])
    return str(dates[0]) if dates else ''


def _target_count_label(count: int) -> str:
    value = abs(int(count))
    if value % 10 == 1 and value % 100 != 11:
        return "цель"
    if value % 10 in {2, 3, 4} and value % 100 not in {12, 13, 14}:
        return "цели"
    return "целей"


def _select_diverse_evidence(records: List[EvidenceRecord], limit: int = 8) -> List[EvidenceRecord]:
    """Выбирает примеры доказательств с покрытием разных документов/дат/целей.

    Важно: это именно примеры, а не весь набор записей для статистики. Раньше
    top-N мог состоять из 3-4 самых сильных документов, из-за чего пользователь
    видел неполный список источников, хотя расчёт уже учитывал больше дней.
    """
    out: List[EvidenceRecord] = []
    seen_keys = set()

    def add(rec: EvidenceRecord) -> None:
        if len(out) >= limit:
            return
        key = (str(getattr(rec, 'source', '') or ''), _record_date_key(rec), str(getattr(rec, 'target', '') or ''), str(getattr(rec, 'sentence', '') or '')[:80])
        if key in seen_keys:
            return
        seen_keys.add(key)
        out.append(rec)

    # 1) Сначала покрываем разные источники.
    seen_sources = set()
    for rec in records:
        source = str(getattr(rec, 'source', '') or '')
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)
        add(rec)

    # 2) Затем покрываем разные даты.
    seen_dates = {_record_date_key(rec) for rec in out if _record_date_key(rec)}
    for rec in records:
        d = _record_date_key(rec)
        if not d or d in seen_dates:
            continue
        seen_dates.add(d)
        add(rec)

    # 3) Затем разные цели.
    seen_targets = {str(getattr(rec, 'target', '') or '') for rec in out if getattr(rec, 'target', '')}
    for rec in records:
        target = str(getattr(rec, 'target', '') or '')
        if not target or target in seen_targets:
            continue
        seen_targets.add(target)
        add(rec)

    # 4) Добиваем лучшими по score/confidence.
    for rec in records:
        add(rec)
        if len(out) >= limit:
            break
    return out[:limit]

def _build_binary(intent: QueryUnderstandingResult, records: List[EvidenceRecord], stats: Dict[str, object]) -> StructuredAnswer:
    positive = _choose_positive(records)
    if not positive:
        return StructuredAnswer(text='В документах нет прямого подтверждения по этому вопросу.', mode='binary_fact', stats=stats, confidence=0.0)
    if positive.polarity == 'negative':
        return StructuredAnswer(text='Прямого положительного подтверждения не найдено.', mode='binary_fact', evidence=records[:2], stats=stats, confidence=0.25)
    text = 'Да, подтверждается.'
    if positive.target:
        text += f' Цель/объект: {positive.target}.'
    return StructuredAnswer(text=text, mode='binary_fact', evidence=[positive] + records[1:3], stats=stats, confidence=max(0.78, positive.confidence))


def _plain_names(values, limit: int = 5) -> List[str]:
    out: List[str] = []
    for item in list(values or [])[:limit]:
        if isinstance(item, (list, tuple)) and item:
            name = str(item[0])
        else:
            name = str(item)
        if name and name not in out:
            out.append(name)
    return out


def _build_location(intent: QueryUnderstandingResult, records: List[EvidenceRecord], stats: Dict[str, object]) -> Optional[StructuredAnswer]:
    places = _plain_names(stats.get('top_directions') or stats.get('top_locations') or [], 6)
    if not places:
        seen = []
        for rec in records:
            for item in list(rec.directions or []) + list(rec.locations or []):
                if item and item not in seen:
                    seen.append(item)
        places = seen[:6]
    if not places:
        return None
    obj = intent.primary_object
    subject = f'для {obj} ' if obj else ''
    if len(places) == 1:
        text = f'По найденным данным {subject}указано направление/место: {places[0]}.'
    else:
        text = f'По найденным данным {subject}указаны направления/места: {", ".join(places)}.'
    if intent.is_range_query and intent.start_date and intent.end_date:
        text += f' Период: {intent.start_date:%Y-%m-%d} — {intent.end_date:%Y-%m-%d}.'
    return StructuredAnswer(text=text, mode='location', evidence=records[:4], stats=stats, confidence=0.84)


def _build_method(intent: QueryUnderstandingResult, records: List[EvidenceRecord], stats: Dict[str, object]) -> Optional[StructuredAnswer]:
    method = _choose_method(records)
    if not method:
        return None
    bits = [f'Наиболее вероятный способ/средство: {method}.']
    if stats.get('top_method_target_pairs'):
        pair = stats['top_method_target_pairs'][0]
        bits.append(f'Наиболее сильная связка: {pair[0]} ({pair[1]} {_stats_unit_short(stats)}).')
    return StructuredAnswer(text=' '.join(bits), mode='method', evidence=records[:3], stats=stats, confidence=0.8)


def _build_target(intent: QueryUnderstandingResult, records: List[EvidenceRecord], stats: Dict[str, object]) -> Optional[StructuredAnswer]:
    """Собирает полный доказательный список обнаруженных или поражённых целей."""
    requested_type = str(getattr(intent, "event_type", "any") or "any")
    relation_label = {
        "observed": "обнаруженных",
        "struck": "поражённых",
        "any": "",
    }.get(requested_type, "")
    noun = f"{relation_label} целей".strip()
    if not records:
        return StructuredAnswer(
            text=f'Подтверждённых {noun} по заданным условиям не найдено.',
            mode='target',
            stats=stats,
            confidence=0.0,
        )

    ordered = sorted(
        records,
        key=lambda rec: (
            _record_date_key(rec),
            str(getattr(rec, "event_time", "") or ""),
            str(getattr(rec, "source", "") or ""),
            str(getattr(rec, "sentence", "") or ""),
        ),
    )
    count_label = noun if requested_type != "any" else _target_count_label(len(ordered))
    lines = [f'Найдено {len(ordered)} {count_label}:']
    multiple_dates = len({_record_date_key(rec) for rec in ordered if _record_date_key(rec)}) > 1
    for idx, rec in enumerate(ordered, 1):
        sentence = re.sub(r"\s+", " ", str(rec.sentence or "")).strip(" \t;")
        date_prefix = f"{_record_date_key(rec)}, " if multiple_dates and _record_date_key(rec) else ""
        lines.append(f"{idx}. {date_prefix}{sentence} [{rec.source}]")

    complete = all(
        bool(rec.target)
        and bool(rec.source)
        and (not intent.has_date_constraint or bool(rec.event_dates))
        and (requested_type == "any" or str(getattr(rec, "event_type", "any")) == requested_type)
        for rec in ordered
    )
    confidence = 0.94 if complete else 0.62
    return StructuredAnswer(
        text="\n".join(lines),
        mode='target',
        evidence=ordered[:10],
        stats=stats,
        confidence=confidence,
    )

def _build_count(intent: QueryUnderstandingResult, records: List[EvidenceRecord], stats: Dict[str, object]) -> StructuredAnswer:
    total = int(stats.get('total_records', len(records)))
    if getattr(intent, "count_unit", "") == "date":
        bits = [f'Подтверждено {total} календарных дней работы.']
    else:
        bits = [f'Найдено {total} {_stats_unit_label(stats)}.']
    conflicts = list(stats.get("conflicting_dates") or [])
    if conflicts:
        bits.append("Есть противоречивые сведения по датам: " + ", ".join(conflicts) + ".")
    if _wants_visible_numbers(intent):
        periodicity = stats.get('periodicity') or {}
        if periodicity:
            bits.append(f"Характер активности: {periodicity.get('label')}.")
        if stats.get('top_dates'):
            bits.append('Ключевые даты: ' + _fmt_top(stats['top_dates'], 4) + '.')
    return StructuredAnswer(text=' '.join(bits), mode='count', evidence=records[:4], stats=stats, confidence=0.78)

def _frequency_focus_line(intent: QueryUnderstandingResult, stats: Dict[str, object]) -> str:
    focus = intent.statistics_focus
    if focus == 'direction':
        top = stats.get('top_directions') or stats.get('top_locations') or []
        if top:
            return 'Чаще всего встречается направление/место: ' + _fmt_top(top, 3) + '.'
    if focus == 'method':
        top = stats.get('top_methods') or []
        if top:
            return 'Чаще всего применялся способ/средство: ' + _fmt_top(top, 3) + '.'
    if focus == 'target':
        top = stats.get('top_targets') or []
        if top:
            return 'Чаще всего встречались цели: ' + _fmt_top(top, 3) + '.'
    if focus == 'date':
        top = stats.get('top_dates') or []
        if top:
            return 'Пиковые даты: ' + _fmt_top(top, 4) + '.'
    return ''


def _build_frequency(intent: QueryUnderstandingResult, records: List[EvidenceRecord], stats: Dict[str, object]) -> Optional[StructuredAnswer]:
    if not records:
        return StructuredAnswer(text='Статистически значимых подтверждений не найдено.', mode='frequency', stats=stats, confidence=0.0)
    total = int(stats.get('total_records', len(records)))
    focus_line = _frequency_focus_line(intent, stats)
    numeric = _wants_visible_numbers(intent)
    bits: List[str] = []

    if focus_line:
        bits.append(focus_line)

    if numeric:
        bits.append(f'Найдено {total} {_stats_unit_label(stats)}.')
        periodicity = stats.get('periodicity') or {}
        if periodicity:
            bits.append(f"Периодичность: {periodicity.get('label')}.")
        compact_blocks = []
        if stats.get('top_dates'):
            compact_blocks.append('даты: ' + _fmt_top(stats['top_dates'], 3))
        if stats.get('top_directions'):
            compact_blocks.append('направления: ' + _fmt_top(stats['top_directions'], 2))
        elif stats.get('top_locations'):
            compact_blocks.append('места: ' + _fmt_top(stats['top_locations'], 2))
        if stats.get('top_methods'):
            compact_blocks.append('способы: ' + _fmt_top(stats['top_methods'], 2))
        if stats.get('top_targets'):
            compact_blocks.append('цели: ' + _fmt_top(stats['top_targets'], 2))
        if compact_blocks:
            bits.append('Ключевые срезы: ' + '; '.join(compact_blocks) + '.')
    else:
        named = []
        if stats.get('top_directions'):
            named.append('направления/места: ' + _fmt_names(stats['top_directions'], 3))
        elif stats.get('top_locations'):
            named.append('места: ' + _fmt_names(stats['top_locations'], 3))
        if stats.get('top_methods'):
            named.append('способы: ' + _fmt_names(stats['top_methods'], 3))
        if stats.get('top_targets'):
            named.append('цели: ' + _fmt_names(stats['top_targets'], 3))
        if named:
            bits.append('По найденным данным ключевые признаки: ' + '; '.join(named) + '.')
        else:
            bits.append('По найденным данным есть релевантные подтверждения, но выраженной статистической закономерности не выделено.')
        if total >= 2:
            bits.append('Для точных чисел можно задать вопрос со словом «статистика» или «подробно».')

    return StructuredAnswer(text=' '.join(bits), mode='frequency', evidence=_select_diverse_evidence(records, limit=6), stats=stats, confidence=0.80)


def _build_timeline(intent: QueryUnderstandingResult, records: List[EvidenceRecord], stats: Dict[str, object]) -> Optional[StructuredAnswer]:
    timeline = build_timeline_summary(records)
    if not timeline:
        return None
    bits = [f'Хронология по найденным {_stats_unit_label(stats)}:']
    for item in timeline[:8]:
        extras = []
        if item['methods']:
            extras.append('способы: ' + ', '.join(item['methods']))
        if item['targets']:
            extras.append('цели: ' + ', '.join(item['targets']))
        suffix = f" ({'; '.join(extras)})" if extras else ''
        bits.append(f"{item['date']}: {item['count']} {_stats_unit_short(stats)}{suffix}")
    periodicity = stats.get('periodicity') or {}
    if periodicity:
        bits.append(f"Итог: {periodicity.get('label')}.")
    return StructuredAnswer(text=' '.join(bits), mode='timeline', evidence=records[:6], stats=stats, confidence=0.8)


def _build_comparison(intent: QueryUnderstandingResult, records: List[EvidenceRecord], stats: Dict[str, object]) -> Optional[StructuredAnswer]:
    if not records:
        return None
    comparison = compare_first_last(records)
    parts = ['Сводное сравнение по найденным данным.']
    if comparison:
        parts.append(f"На старте периода ({comparison['first_date']}) найдено {comparison['first']['count']} {_stats_unit_short(stats)}, в конце ({comparison['last_date']}) — {comparison['last']['count']} {_stats_unit_short(stats)}.")
        if comparison['first']['top_methods']:
            parts.append('Ранние методы: ' + ', '.join(f"{name} ({count})" for name, count in comparison['first']['top_methods'][:2]) + '.')
        if comparison['last']['top_methods']:
            parts.append('Поздние методы: ' + ', '.join(f"{name} ({count})" for name, count in comparison['last']['top_methods'][:2]) + '.')
    dominant = build_pattern_summary(records)
    if dominant:
        parts.append('Доминирующие паттерны: ' + ', '.join(f"{item['pattern']} ({item['count']})" for item in dominant[:3]) + '.')
    return StructuredAnswer(text=' '.join(parts), mode='comparison', evidence=records[:6], stats=stats, confidence=0.76)


def _build_summary(intent: QueryUnderstandingResult, records: List[EvidenceRecord], stats: Dict[str, object], *, mode: str = 'summary') -> Optional[StructuredAnswer]:
    if not records:
        return StructuredAnswer(text='Подходящих данных для сводки не найдено.', mode=mode, stats=stats, confidence=0.0)
    total = int(stats.get('total_records', len(records)))
    numeric = _wants_visible_numbers(intent)
    parts: List[str] = []
    date_span = stats.get('date_span') or {}

    if numeric:
        parts.append(f'Краткий анализ по найденным данным: {total} {_stats_unit_label(stats)}.')
    else:
        parts.append('Краткий вывод по найденным данным.')
    if date_span:
        parts.append(f"Период: {date_span.get('start')} — {date_span.get('end')}.")

    for label, key in [('Действия', 'top_actions'), ('Способы', 'top_methods'), ('Направления/места', 'top_directions'), ('Цели', 'top_targets')]:
        values = stats.get(key) or []
        if values:
            if numeric:
                parts.append(f'{label}: ' + _fmt_top(values, 3, _stats_unit_short(stats)) + '.')
            else:
                parts.append(f'{label}: ' + _fmt_names(values, 3) + '.')

    if numeric:
        periodicity = stats.get('periodicity') or {}
        if periodicity:
            parts.append(f"Характер активности: {periodicity.get('label')}.")
    else:
        if total > 0:
            parts.append('Ответ намеренно сокращён: подробные цифры показываются только по запросу «подробно» или «статистика».')

    return StructuredAnswer(text=' '.join(parts), mode=mode, evidence=_select_diverse_evidence(records, limit=6), stats=stats, confidence=0.74)

def _build_activity_report(intent: QueryUnderstandingResult, records: List[EvidenceRecord], stats: Dict[str, object]) -> Optional[StructuredAnswer]:
    """Подробный отчёт по работе/активности сущности за период.

    Это универсальный режим «что делал X за период Y». В военных сводках X может
    быть рБПЛА, в технических логах — сервис/станок, в учебных материалах —
    процесс.  Мы не считаем простые упоминания: каждая строка ниже —
    дедуплицированное подтверждённое утверждение с источником.
    """
    if not records:
        return StructuredAnswer(
            text='Подходящих подтверждённых данных об активности за этот период не найдено.',
            mode='activity_report',
            stats=stats,
            confidence=0.0,
        )

    obj = intent.primary_object or 'объекта'
    period = ''
    if intent.start_date and intent.end_date:
        period = f' за период {intent.start_date:%Y-%m-%d} — {intent.end_date:%Y-%m-%d}'
    total = int(stats.get('total_records', len(records)))
    unit_label = _stats_unit_label(stats)

    by_date: Dict[str, List[EvidenceRecord]] = {}
    undated: List[EvidenceRecord] = []
    for rec in records:
        dates = list(getattr(rec, 'event_dates', []) or [])
        if not dates:
            undated.append(rec)
            continue
        for d in dates[:2]:
            by_date.setdefault(str(d), []).append(rec)

    parts: List[str] = [f'Подробная работа {obj}{period}: найдено {total} {unit_label}.']
    raw_facts = int(stats.get('raw_deduped_fact_records') or 0)
    if raw_facts and raw_facts > total:
        parts.append(f'Внутри этих эпизодов найдено {raw_facts} подтверждённых строк/фрагментов; статистика направлений и дней считается по эпизодам, а не по каждой строке.')

    top_dirs = stats.get('top_directions') or stats.get('top_locations') or []
    top_methods = stats.get('top_methods') or []
    top_targets = stats.get('top_targets') or []
    compact = []
    if top_dirs:
        compact.append('направления/места: ' + _fmt_names(top_dirs, 5))
    if top_methods:
        compact.append('способы/средства: ' + _fmt_names(top_methods, 5))
    if top_targets:
        compact.append('цели/объекты: ' + _fmt_names(top_targets, 5))
    if compact:
        parts.append('Итог по периоду: ' + '; '.join(compact) + '.')

    if by_date:
        parts.append('По дням:')
        for date in sorted(by_date.keys())[:20]:
            day_records = by_date[date]
            fragments = []
            seen = set()
            for rec in day_records:
                sentence = (rec.sentence or '').strip()
                if not sentence:
                    continue
                if len(sentence) > 220:
                    sentence = sentence[:220].rstrip() + '...'
                key = sentence.lower()
                if key in seen:
                    continue
                seen.add(key)
                extras = []
                if rec.directions:
                    extras.append('место/направление: ' + ', '.join(rec.directions[:2]))
                if rec.methods:
                    extras.append('способ: ' + ', '.join(rec.methods[:2]))
                if rec.target:
                    extras.append('цель: ' + rec.target)
                suffix = f" ({'; '.join(extras)})" if extras else ''
                fragments.append(f'— {sentence}{suffix}')
                if len(fragments) >= 3:
                    break
            if fragments:
                parts.append(f'{date}:\n' + '\n'.join(fragments))
    elif undated:
        parts.append('Даты в найденных фрагментах не восстановлены, поэтому ниже только подтверждённые фрагменты без группировки по дням.')

    if undated and len(by_date) < 3:
        samples = []
        for rec in undated[:4]:
            sentence = (rec.sentence or '').strip()
            if len(sentence) > 220:
                sentence = sentence[:220].rstrip() + '...'
            if sentence:
                samples.append('— ' + sentence)
        if samples:
            parts.append('Без явной даты:\n' + '\n'.join(samples))

    return StructuredAnswer(
        text='\n\n'.join(parts),
        mode='activity_report',
        evidence=_select_diverse_evidence(records, limit=12),
        stats=stats,
        confidence=0.86 if records else 0.0,
    )


def build_structured_answer(intent: QueryUnderstandingResult, records: List[EvidenceRecord], stats: Dict[str, object]) -> Optional[StructuredAnswer]:
    metric_answer = _build_metric_fact(intent, records, stats)
    if metric_answer is not None:
        return metric_answer
    if intent.question_type == 'binary_fact':
        return _build_binary(intent, records, stats)
    if intent.question_type == 'location':
        return _build_location(intent, records, stats)
    if intent.question_type == 'method':
        return _build_method(intent, records, stats)
    if intent.question_type == 'target':
        return _build_target(intent, records, stats)
    if intent.question_type == 'count':
        return _build_count(intent, records, stats)
    if intent.question_type == 'frequency':
        return _build_frequency(intent, records, stats)
    if intent.question_type == 'timeline':
        return _build_timeline(intent, records, stats)
    if intent.question_type == 'comparison':
        return _build_comparison(intent, records, stats)
    if intent.question_type == 'activity_report':
        return _build_activity_report(intent, records, stats)
    if intent.question_type == 'range_summary':
        return _build_summary(intent, records, stats, mode='range_summary')
    if intent.question_type == 'summary':
        return _build_summary(intent, records, stats)
    return None
