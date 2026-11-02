"""Извлечение опорных фактов и событий из найденных чанков."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple
import re

from core.date_range_parser import parse_date_constraints
from core.domain_lexicon import get_domain_lexicon
from core.domain_profile import get_domain_profile
from core.entity_resolution import entity_overlap_score, extract_structured_entities
from core.logging_utils import get_logger
from core.query_understanding import QueryUnderstandingResult, normalize_question

logger = get_logger(__name__)

@dataclass
class EvidenceRecord:
    doc_id: str
    source: str
    chunk_id: Any
    sentence: str
    full_text: str
    score: float
    date_match: bool = False
    object_match_count: int = 0
    entity_match_count: int = 0
    keyword_match_count: int = 0
    methods: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    directions: List[str] = field(default_factory=list)
    event_dates: List[str] = field(default_factory=list)
    quantities: List[int] = field(default_factory=list)
    target: str = ""
    action: str = ""
    event_type: str = "any"
    event_time: str = ""
    polarity: str = "positive"
    indicator: str = ""
    value: str = ""
    confidence: float = 0.0
    evidence_kind: str = "sentence"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexedEvent:
    doc_id: str
    source: str
    chunk_id: Any
    sentence: str
    normalized_sentence: str
    entities: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    directions: List[str] = field(default_factory=list)
    event_dates: List[str] = field(default_factory=list)
    quantities: List[int] = field(default_factory=list)
    target: str = ""
    action: str = ""
    event_type: str = "any"
    event_time: str = ""
    polarity: str = "positive"
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


_ABBR_DOT = "\uE000"
_EVENT_TIME_RE = re.compile(r"^\s*((?:[01]?\d|2[0-3])[.:][0-5]\d)\s*[—–-]\s*(.+)$", re.DOTALL)


def _event_type_from_text(text: str) -> str:
    return "any"


def _metadata_document_date(metadata: Dict[str, Any] | None) -> datetime | None:
    raw = str((metadata or {}).get("document_date") or "").strip()
    try:
        return datetime.strptime(raw, "%Y-%m-%d")
    except Exception:
        return None


def _protect_sentence_abbreviations(text: str) -> str:
    """Защищает типовые сокращения от ошибочного разбиения на предложения.

    Без этого строка вроде `до двух чел. в здании – поражены...`
    режется на `до двух чел.` и `в здании – поражены...`, после чего
    extractor теряет главный объект и отвечает только `в здании`.
    """
    value = text or ""

    def repl(match: re.Match) -> str:
        return match.group(0).replace(".", _ABBR_DOT)

    value = re.sub(r"(?i)\b(?:н\.п|т\.д|т\.п|и\.т\.д|и\.т\.п)\.", repl, value)
    value = re.sub(
        r"(?i)\b(?:чел|ед|шт|гр|расч|поз|позиц|рай|р-н|обл|окр|сев-зап|сев-вост|юго-зап|юго-вост|бр|дшп|одшбр|омсбр|мсбр|ппд|кп|оп|птур|бк)\.",
        repl,
        value,
    )
    return value


def _restore_sentence_abbreviations(text: str) -> str:
    return (text or "").replace(_ABBR_DOT, ".")


def _split_units(text: str) -> List[str]:
    raw = (text or "").replace("\r", "\n")
    units: List[str] = []
    for block in re.split(r"\n+", raw):
        block = block.strip(" \t-•")
        if not block:
            continue
        protected = _protect_sentence_abbreviations(block)
        parts = re.split(r"(?<=[\.!?;])\s+", protected)
        for part in parts:
            part = _restore_sentence_abbreviations(part).strip()
            is_identifier_header = bool(
                re.fullmatch(
                    r"[A-Za-zА-Яа-яЁё][\w/-]{0,30}\s*(?:№|#)\s*\d{1,8}(?:\s*\([^)\n]{2,100}\))?\s*:?",
                    part,
                )
            )
            if len(part) >= 10 or is_identifier_header:
                units.append(part)
    return units or ([raw.strip()] if raw.strip() else [])


def _dedupe_keep_order(values: List[str], limit: int = 6) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in values:
        low = item.lower().strip()
        if not low or low in seen:
            continue
        seen.add(low)
        out.append(item.strip())
        if len(out) >= limit:
            break
    return out



def _canonical_direction(value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip(" ,.;:-"))
    if not value:
        return ""
    low = value.lower().replace("ё", "е")
    if low.endswith("ском"):
        low = low[:-4] + "ское"
    elif low.endswith("ский"):
        low = low[:-4] + "ское"
    elif low.endswith("ской"):
        low = low[:-4] + "ское"
    elif low.endswith("ская"):
        low = low[:-4] + "ское"
    return low[:1].upper() + low[1:]


_DOMAIN = get_domain_lexicon()

_MAJOR_SECTION_RE = re.compile(
    # Разделы в корпусе часто выглядят как
    # `рБпЛА №11 (Изюмское направление):` или `шгр 90 (...):`.
    # Важно распознавать не только рБпЛА: любой новый заголовок должен сбрасывать
    # активный раздел, иначе строки ШГР/ДШБ ошибочно наследуют направление рБпЛА.
    rf"^\s*(?P<header>(?P<family>{_DOMAIN.actor_pattern})\s*(?:№|#)?\s*(?P<num>\d{{1,6}})[^:\n]{{0,120}})\s*:?\s*$",
    re.IGNORECASE,
)
_SECTION_DIRECTION_RE = re.compile(
    r"\(([А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z\- ]{2,60}?\s+направлен(?:ие|ии|ия|ию|ием)?)\)",
    re.IGNORECASE,
)


def _intent_numbers(intent: QueryUnderstandingResult) -> set[str]:
    nums = {str(x).lstrip("0") or "0" for x in (getattr(intent, 'identifiers', []) or []) if str(x).isdigit()}
    for ent in extract_structured_entities(getattr(intent, 'raw_question', '') or ''):
        if ent.number_norm:
            nums.add(ent.number_norm)
    return nums


def _intent_mentions_family(intent: QueryUnderstandingResult, family: str) -> bool:
    return _DOMAIN.mentions_family(getattr(intent, 'raw_question', '') or '', family)


def _parse_major_section(unit: str) -> dict:
    text = re.sub(r"\s+", " ", (unit or "").strip(" -•\t"))
    m = _MAJOR_SECTION_RE.match(text)
    if not m:
        return {}
    raw_num = str(m.group('num'))
    number = str(int(raw_num)) if raw_num.isdigit() else raw_num
    family = _DOMAIN.canonical_family(m.group('family') or '')
    direction = ""
    md = _SECTION_DIRECTION_RE.search(text)
    if md:
        direction = re.sub(r"\s+направлен(?:ие|ии|ия|ию|ием)?\b", "", md.group(1), flags=re.IGNORECASE).strip()
        direction = _canonical_direction(direction)
    return {
        'header': text,
        'family': family,
        'number': number,
        'direction': direction,
        'terms': _DOMAIN.reference_terms(family, number),
    }


def _section_matches_intent(section: dict, intent: QueryUnderstandingResult) -> bool:
    if not section:
        return False
    numbers = _intent_numbers(intent)
    if numbers and str(section.get('number')) not in numbers:
        return False
    if numbers and _intent_mentions_family(intent, 'рбпла'):
        return _DOMAIN.family_matches(str(section.get('family') or ''), 'рбпла')
    q = normalize_question(getattr(intent, 'raw_question', '') or '')
    return bool(numbers and f"№{section.get('number')}" in q)


def _split_units_with_sections(text: str, intent: QueryUnderstandingResult | None = None):
    units = _split_units(text)
    active_section: dict = {}
    pending_context: List[str] = []
    for unit in units:
        section = _parse_major_section(unit)
        if section:
            section["event_type"] = "any"
            section["context"] = " ".join(pending_context[-2:])
            active_section = section
            yield unit, active_section, bool(intent and _section_matches_intent(active_section, intent))
            continue
        if (
            unit.rstrip().endswith(":")
            and not _EVENT_TIME_RE.match(unit)
            and not re.search(r"(?:№|#)\s*\d{1,8}", unit)
        ):
            if active_section:
                pending_context = [unit.strip()]
                active_section = {}
            else:
                pending_context.append(unit.strip())
                pending_context = pending_context[-2:]
        yield unit, active_section, bool(intent and _section_matches_intent(active_section, intent))


def _record_matches_object_scope(rec: EvidenceRecord, intent: QueryUnderstandingResult) -> bool:
    """Проверяет, относится ли evidence именно к объекту из вопроса.

    Старое правило разрешало ``entity_match_count > 0`` как достаточное
    совпадение.  Для отчётов вида «работа рБпЛА №11 за период» это слишком
    широко: в диапазон начинали попадать соседние подразделения/направления
    только из-за похожих числовых сущностей или общего документа.  Поэтому для
    семейства БпЛА/рБпЛА + номер используем строгую проверку участника: номер
    должен стоять рядом с нужным семейством в самой строке или в метаданных
    активного раздела.
    """
    terms = [normalize_question(x) for x in (getattr(intent, 'object_terms', []) or []) if x]
    identifiers = [str(x).lstrip('0') or '0' for x in (getattr(intent, 'identifiers', []) or []) if str(x)]
    if not terms and not identifiers:
        return True

    norm = normalize_question(rec.sentence)
    meta = getattr(rec, 'metadata', {}) or {}
    section_family = normalize_question(str(meta.get('section_family') or ''))
    section_number = str(meta.get('section_number') or '').lstrip('0') or ''
    section_header = normalize_question(str(meta.get('section_header') or ''))
    section_terms = [normalize_question(str(x)) for x in (meta.get('section_terms') or []) if x]

    wants_air_uav = _intent_mentions_family(intent, 'рбпла')
    if identifiers and wants_air_uav:
        # Строгий scoped-режим: «№11» без «рБпЛА/БпЛА» рядом не подходит.
        for ident in identifiers:
            if _DOMAIN.family_matches(section_family, 'рбпла') and section_number == ident:
                return True
            if any(term and term in norm for term in terms):
                return True
            if any(term and (term in section_header or term in section_terms) for term in terms):
                return True
            if re.search(rf"\b(?:р\s*)?бпла\s*(?:№|#)?\s*0*{re.escape(ident)}\b", norm, flags=re.IGNORECASE):
                return True
        return False

    if any(t and t in norm for t in terms):
        return True
    if any(t and (t in section_header or t in section_terms) for t in terms):
        return True
    if rec.object_match_count > 0:
        return True
    # Для остальных точных ID entity-hit оставляем как fallback: например
    # «изделие-51» ≈ «изделием 51». Для activity_report он уже отсеян выше.
    if getattr(intent, 'question_type', '') != 'activity_report' and rec.entity_match_count > 0:
        return True
    return False

_TARGET_CONTEXT_ONLY_RE = re.compile(
    r"^(?:в|на|у|при|около|возле|из|с)\s+[а-яёa-z0-9\-\s]{2,80}$",
    re.IGNORECASE,
)
_QUANTITY_TARGET_RE = re.compile(
    r"\b(?:до\s+)?(?:одного|двух|трех|трёх|четырех|четырёх|пяти|шести|семи|восьми|девяти|десяти|\d{1,3})\s+(?:чел|человек|ед|единиц|расчет|расч[её]т|групп|гр)\b",
    re.IGNORECASE,
)


def _normalize_target_for_dedupe(value: str) -> str:
    text = normalize_question(value or "")
    text = re.sub(r"\b(?:до|примерно|около)\b", " ", text)
    text = re.sub(r"\b(?:одного|двух|трех|трёх|четырех|четырёх|пяти|шести|семи|восьми|девяти|десяти|\d{1,3})\b", " ", text)
    text = re.sub(r"\b(?:чел|человек|ед|единиц|расчет|расч[её]т|групп|гр)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _target_is_context_only(value: str) -> bool:
    text = re.sub(r"\s+", " ", value or "").strip(" ,.;:-—–")
    if not text:
        return False
    return bool(_TARGET_CONTEXT_ONLY_RE.match(text)) and not bool(_QUANTITY_TARGET_RE.search(text))


def _target_specificity(value: str) -> tuple[int, int, int, int]:
    """Оценка полноты цели для дедупликации статистики."""
    text = re.sub(r"\s+", " ", value or "").strip(" ,.;:-—–")
    if not text:
        return (0, 0, 0, 0)
    has_quantity = 1 if _QUANTITY_TARGET_RE.search(text) else 0
    context_only_penalty = 1 if _target_is_context_only(text) else 0
    token_count = len(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text))
    return (has_quantity, 0 if context_only_penalty else 1, token_count, len(text))


def _record_event_signature(rec: EvidenceRecord, *, ignore_target: bool = False) -> tuple:
    dates = tuple(sorted(str(x) for x in (getattr(rec, 'event_dates', []) or []) if x))
    methods = tuple(sorted(normalize_question(x) for x in (getattr(rec, 'methods', []) or []) if x))
    locations = tuple(sorted(normalize_question(x) for x in (getattr(rec, 'locations', []) or []) if x))
    directions = tuple(sorted(normalize_question(x) for x in (getattr(rec, 'directions', []) or []) if x))
    target = "" if ignore_target else _normalize_target_for_dedupe(str(getattr(rec, 'target', '') or ''))
    return (
        str(getattr(rec, 'source', '') or ''),
        dates,
        methods,
        locations,
        directions,
        normalize_question(str(getattr(rec, 'action', '') or '')),
        str(getattr(rec, 'event_type', '') or ''),
        str(getattr(rec, 'event_time', '') or ''),
        target,
    )


def _targets_overlap_for_dedupe(a: str, b: str) -> bool:
    na = _normalize_target_for_dedupe(a)
    nb = _normalize_target_for_dedupe(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return na in nb or nb in na


def _merge_record_prefer_richer(a: EvidenceRecord, b: EvidenceRecord) -> EvidenceRecord:
    spec_a = _target_specificity(str(getattr(a, 'target', '') or ''))
    spec_b = _target_specificity(str(getattr(b, 'target', '') or ''))
    if spec_b > spec_a or (spec_b == spec_a and (float(getattr(b, 'score', 0.0) or 0.0), float(getattr(b, 'confidence', 0.0) or 0.0)) > (float(getattr(a, 'score', 0.0) or 0.0), float(getattr(a, 'confidence', 0.0) or 0.0))):
        keep, other = b, a
    else:
        keep, other = a, b
    keep.methods = _dedupe_keep_order(list(getattr(keep, 'methods', []) or []) + list(getattr(other, 'methods', []) or []), limit=8)
    keep.locations = _dedupe_keep_order(list(getattr(keep, 'locations', []) or []) + list(getattr(other, 'locations', []) or []), limit=8)
    keep.directions = _dedupe_keep_order(list(getattr(keep, 'directions', []) or []) + list(getattr(other, 'directions', []) or []), limit=8)
    keep.event_dates = _dedupe_keep_order(list(getattr(keep, 'event_dates', []) or []) + list(getattr(other, 'event_dates', []) or []), limit=8)
    keep.quantities = list(dict.fromkeys(list(getattr(keep, 'quantities', []) or []) + list(getattr(other, 'quantities', []) or [])))[:8]
    if not getattr(keep, "event_time", ""):
        keep.event_time = str(getattr(other, "event_time", "") or "")
    if str(getattr(keep, "event_type", "any") or "any") == "any":
        keep.event_type = str(getattr(other, "event_type", "any") or "any")
    keep_meta = dict(getattr(keep, 'metadata', {}) or {})
    other_meta = dict(getattr(other, 'metadata', {}) or {})
    for key, value in other_meta.items():
        if key not in keep_meta or keep_meta.get(key) in (None, '', [], {}):
            keep_meta[key] = value
    keep.metadata = keep_meta
    keep.score = max(float(getattr(keep, 'score', 0.0) or 0.0), float(getattr(other, 'score', 0.0) or 0.0))
    keep.confidence = max(float(getattr(keep, 'confidence', 0.0) or 0.0), float(getattr(other, 'confidence', 0.0) or 0.0))
    return keep


def dedupe_evidence_records(records: List[EvidenceRecord]) -> List[EvidenceRecord]:
    """Убирает дубли одного физического события в статистике и ответах.

    Например, `до двух чел. в здании – поражены...` и обрезанный вариант
    `в здании – поражены...` считаются одним эпизодом; оставляется более
    информативная цель.
    """
    out: List[EvidenceRecord] = []
    for rec in records or []:
        merged = False
        base = _record_event_signature(rec, ignore_target=True)
        target = str(getattr(rec, 'target', '') or '')
        for idx, old in enumerate(out):
            if _record_event_signature(old, ignore_target=True) != base:
                continue
            old_target = str(getattr(old, 'target', '') or '')
            if target and old_target and _targets_overlap_for_dedupe(target, old_target):
                out[idx] = _merge_record_prefer_richer(old, rec)
                merged = True
                break
            if not target or not old_target:
                out[idx] = _merge_record_prefer_richer(old, rec)
                merged = True
                break
        if not merged:
            out.append(rec)
    out.sort(key=lambda x: (float(getattr(x, 'score', 0.0) or 0.0), float(getattr(x, 'confidence', 0.0) or 0.0)), reverse=True)
    return out

def _extract_methods(sentence: str) -> List[str]:
    s = sentence.strip()
    found: List[str] = []
    patterns = [
        r'применени(?:ем|я)\s+[«"]?([^"».,;:\n]{2,80})[»"]?',
        r'поражен\s+применени(?:ем|я)\s+[«"]?([^"».,;:\n]{2,80})[»"]?',
        r'(?<![A-Za-zА-Яа-яЁё])БпЛА\s+[«"]?([^"».,;:\n]{2,80})[»"]?',
        r'дрон\s+[«"]?([^"».,;:\n]{2,80})[»"]?',
        r'FPV[- ]?дрон(?:ом)?',
        r'(?:using|via|by means of|with the use of)\s+["“]?([^"”.,;:\n]{2,80})["”]?',
    ]
    for pat in patterns:
        for m in re.finditer(pat, s, flags=re.IGNORECASE):
            value = m.group(1).strip() if m.groups() else m.group(0).strip()
            value = re.sub(r"\s+", " ", value).strip(" «»\"',.;:-")
            if not value:
                continue
            low_value = normalize_question(value)
            if low_value.startswith("№") or "направлен" in low_value or low_value.startswith(("рбпла", "бпла №")):
                continue
            if low_value.startswith("бпла "):
                value = re.sub(r"(?i)^бпла\s+", "", value).strip(" «»\"',.;:-")
                low_value = normalize_question(value)
            found.append(value)
    for ent in extract_structured_entities(s):
        alpha_tokens = [tok for tok in ent.alpha_norm.split() if tok]
        if len(alpha_tokens) > 2:
            continue
        # Не считаем подразделения/номера расчётов и саму искомую единицу способом применения.
        # Иначе в статистике появляются странные "способы" вроде "рБпЛА №11".
        if alpha_tokens and alpha_tokens[0] in {
            "гр", "группа", "ат", "бтр", "птур", "рбпла", "бпла",
            "шгр", "дшб", "одшбр", "омсбр", "мсбр", "расчет", "расч",
        }:
            continue
        found.append(ent.raw.strip())
    return _dedupe_keep_order(found)


def _extract_locations(sentence: str) -> List[str]:
    found: List[str] = []
    # Локация часто стоит после запятой и может состоять из нескольких слов:
    # `..., МАЛАЯ ШАПКОВКА.`. Старый regex брал только `МАЛАЯ`.
    for m in re.finditer(r",\s*([А-ЯЁA-Z][А-ЯЁA-Z\-]*(?:\s+[А-ЯЁA-Z][А-ЯЁA-Z\-]*){0,4})(?=[\.,;:]|$)", sentence):
        value = re.sub(r"\s+", " ", m.group(1)).strip(" ,.;:")
        if value:
            found.append(value)
    for m in re.finditer(r"\b([А-ЯЁ][а-яё\-]+(?:ский|ское|ском|ской|ская))\s+направлен", sentence):
        found.append(_canonical_direction(m.group(1)))
    for m in re.finditer(
        r"\b(?:in|at|near|around|within)\s+([A-Z][A-Za-z\-]*(?:\s+[A-Z][A-Za-z\-]*){0,3})",
        sentence,
    ):
        found.append(re.sub(r"\s+", " ", m.group(1)).strip())
    return _dedupe_keep_order(found, limit=5)


def _extract_directions(sentence: str) -> List[str]:
    found: List[str] = []
    for m in re.finditer(r"\b([А-ЯЁ][а-яё\-]+(?:ский|ское|ском|ской|ская))\s+направлен", sentence):
        found.append(_canonical_direction(m.group(1)))
    for m in re.finditer(
        r"\b([A-Z][A-Za-z\-]*(?:\s+[A-Z][A-Za-z\-]*){0,2})\s+(?:direction|sector|area)\b",
        sentence,
    ):
        found.append(re.sub(r"\s+", " ", m.group(1)).strip())
    return _dedupe_keep_order(found, limit=5)


def _clean_target_phrase(value: str) -> str:
    """Возвращает человекочитаемую цель из левой части события.

    Важно не выкидывать количественную часть (`до двух чел.`), потому что
    фразы `до двух чел. в здании` и просто `в здании` несут разный смысл.
    """
    text = re.sub(r"\s+", " ", value or "").strip(" \t,.;:-—–")
    text = re.sub(r"(?i)^гр\s*(?:№|#)?\s*\d{1,8}\s*:\s*", "", text).strip(" \t,.;:-—–")
    text = re.sub(r"(?i)^группа\s*(?:№|#)?\s*\d{1,8}\s*:\s*", "", text).strip(" \t,.;:-—–")
    if len(text) > 140:
        text = text[-140:].strip(" \t,.;:-—–")
    return text


def _extract_target(sentence: str, *, has_section: bool = False) -> str:
    sentence = _restore_sentence_abbreviations(_protect_sentence_abbreviations(sentence or "")).strip()
    timed = _EVENT_TIME_RE.match(sentence)
    if timed:
        return _extract_observed_target(sentence)
    match = re.match(r"(?P<target>[^\n—–]{3,180}?)\s*[—–]\s*\S+", sentence) if has_section else None
    if match:
        return _clean_target_phrase(match.group("target"))
    return ""


def _extract_event_time(sentence: str) -> str:
    match = _EVENT_TIME_RE.match((sentence or "").strip())
    return match.group(1).replace(".", ":") if match else ""


def _extract_observed_target(sentence: str) -> str:
    match = _EVENT_TIME_RE.match((sentence or "").strip())
    if not match:
        return ""
    payload = re.sub(r"\s+", " ", match.group(2)).strip(" \t,.;:-—–")
    if len(payload) < 3:
        return ""
    return payload[:240]


def _extract_action(sentence: str) -> str:
    s = normalize_question(sentence)
    actions = get_domain_profile(sentence).actions_in(sentence)
    if actions:
        return actions[0]
    # Generic relationship verbs remain in the core; domain terminology lives
    # in DomainProfile above.
    for cue in ("deployed", "operated"):
        if cue in s:
            return cue
    return ""


def _extract_event_dates(sentence: str, reference_date: datetime | None = None) -> List[str]:
    parsed = parse_date_constraints(sentence, reference_date=reference_date)
    dates = parsed.get('dates') or []
    if dates:
        return [dt.strftime('%Y-%m-%d') for dt in dates if isinstance(dt, datetime)]
    start = parsed.get('start')
    end = parsed.get('end')
    if isinstance(start, datetime) and isinstance(end, datetime):
        if start == end:
            return [start.strftime('%Y-%m-%d')]
        return [start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')]
    return []


def _extract_quantities(sentence: str) -> List[int]:
    values: List[int] = []
    for m in re.finditer(r"\b(\d{1,3})\b", sentence):
        try:
            val = int(m.group(1))
        except Exception:
            continue
        if 0 <= val <= 500:
            values.append(val)
    out: List[int] = []
    seen = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out[:8]


def _detect_polarity(sentence: str) -> str:
    s = normalize_question(sentence)
    if any(token in s for token in [
        "не ", "не выяв", "не обнаруж", "не примен", "не использ", "успеха не имел",
        "безуспеш", "не подтвержд", "not ", "no evidence", "did not", "wasn't ",
        "weren't ", "without success", "unconfirmed",
    ]):
        return "negative"
    if get_domain_profile(sentence).actions_in(sentence) or any(token in s for token in ("deployed", "operated")):
        return "positive"
    return "neutral"


def _extract_indicator_value(sentence: str) -> tuple[str, str]:
    profile = get_domain_profile(sentence)
    indicators = profile.indicators_in(sentence)
    if not indicators:
        return "", ""
    indicator = indicators[0]
    aliases = profile.metric_aliases.get(indicator, ())
    for alias in sorted(aliases, key=len, reverse=True):
        match = re.search(
            rf"{re.escape(alias)}\b.{{0,36}}?(?P<value>[+-]?\d[\d\s]*(?:[.,]\d+)?\s*(?:тыс(?:яч[аи]?)?|млн|миллион(?:а|ов)?|млрд|миллиард(?:а|ов)?|thousand|million|billion)?\s*(?:руб(?:лей|ля|ль)?|₽|usd|eur|доллар(?:ов|а)?|евро|%)?)",
            sentence,
            flags=re.IGNORECASE,
        )
        if match:
            return indicator, re.sub(r"\s+", " ", match.group("value")).strip()
    return indicator, ""


def _sentence_positive_for_intent(intent: QueryUnderstandingResult, sentence: str) -> bool:
    q = intent.normalized_question
    s = normalize_question(sentence)
    usage_question = any(token in q for token in ["использовал", "применял", "использовалось", "применялось", "used", "using", "deployed"])
    strike_question = any(token in q for token in ["поражен", "уничтожен", "подавлен", "поврежден", "destroyed", "damaged", "suppressed"])
    location_question = intent.question_type == "location"
    method_question = intent.question_type == "method"
    if usage_question and any(cue in s for cue in ["применени", "использован", "бпла", "fpv", "used", "using", "deployed"]):
        return True
    if strike_question and any(cue in s for cue in ["поражен", "уничтожен", "поврежден", "подавлен", "destroyed", "damaged", "suppressed"]):
        return True
    if location_question and ("направлен" in s or "," in sentence or any(cue in s for cue in [" in ", " at ", " near ", "direction", "sector", "area"])):
        return True
    if method_question and any(cue in s for cue in ["применени", "бпла", "издели", "дрон", "using", " via ", "by means of", "device", "method"]):
        return True
    if getattr(intent, 'question_type', '') == 'activity_report' and any(cue in s for cue in ["работ", "примен", "обнаруж", "выяв", "пораж", "уничтож", "сопровожд", "коррект", "зафикс", "передал", "осуществ", "worked", "operated", "used", "deployed", "detected", "destroyed", "recorded", "transferred", "performed"]):
        return True
    return False


def extract_indexed_events(doc_id: str, source: str, chunk_id: Any, text: str, metadata: Dict[str, Any] | None = None) -> List[IndexedEvent]:
    events: List[IndexedEvent] = []
    base_meta = dict(metadata or {})
    document_date = _metadata_document_date(base_meta)
    chunk_start = int(base_meta.get("start_char") or 0) if str(base_meta.get("start_char") or "0").lstrip("-").isdigit() else 0
    search_pos = 0
    for unit, section, _section_match in _split_units_with_sections(text, None):
        local_start = text.find(unit, search_pos)
        if local_start < 0:
            local_start = text.find(unit)
        if local_start < 0:
            local_start = 0
        local_end = local_start + len(unit)
        search_pos = max(search_pos, local_end)
        sentence_start_char = chunk_start + local_start
        sentence_end_char = chunk_start + local_end

        norm = normalize_question(unit)
        if not norm:
            continue
        entities = [entity.raw for entity in extract_structured_entities(unit)]
        methods = _extract_methods(unit)
        locations = _extract_locations(unit)
        directions = _extract_directions(unit)
        if section.get('direction') and section.get('direction') not in directions:
            directions = [section['direction']] + directions
        event_type = "any"
        event_time = _extract_event_time(unit)
        dates = _extract_event_dates(unit, reference_date=document_date)
        quantities = _extract_quantities(unit)
        target = _extract_target(unit, has_section=bool(section))
        if not dates and document_date and (section or event_time or target):
            dates = [document_date.strftime("%Y-%m-%d")]
        action = _extract_action(unit)
        polarity = _detect_polarity(unit)
        confidence = min(
            1.0,
            0.28
            + (0.16 if entities else 0.0)
            + (0.16 if methods else 0.0)
            + (0.12 if dates else 0.0)
            + (0.10 if target else 0.0)
            + (0.08 if event_type != "any" else 0.0)
            + (0.06 if event_time else 0.0)
            + (0.08 if directions or locations else 0.0)
            + (0.06 if action else 0.0)
            + (0.08 if section else 0.0),
        )
        events.append(IndexedEvent(
            doc_id=str(doc_id or ""),
            source=str(source or ""),
            chunk_id=chunk_id,
            sentence=unit.strip(),
            normalized_sentence=norm,
            entities=_dedupe_keep_order(entities, limit=8),
            methods=methods,
            locations=locations,
            directions=directions,
            event_dates=dates,
            quantities=quantities,
            target=target,
            action=action,
            event_type=event_type,
            event_time=event_time,
            polarity=polarity,
            confidence=confidence,
            metadata={
                **base_meta,
                "kind": "section_event" if section else "sentence_event",
                "section_header": section.get('header', ''),
                "section_family": section.get('family', ''),
                "section_number": section.get('number', ''),
                "section_direction": section.get('direction', ''),
                "section_terms": section.get('terms', []),
                "section_event_type": section.get("event_type", "any"),
                "section_context": section.get("context", ""),
                "sentence_start_char": sentence_start_char,
                "sentence_end_char": sentence_end_char,
                "sentence_local_start": local_start,
                "sentence_local_end": local_end,
            },
        ))
    return events

def indexed_event_to_evidence(event: IndexedEvent, *, score: float = 1.0, date_match: bool = False, object_hits: int = 0, entity_hits: int = 0, keyword_hits: int = 0) -> EvidenceRecord:
    indicator, value = _extract_indicator_value(event.sentence)
    return EvidenceRecord(
        doc_id=event.doc_id,
        source=event.source,
        chunk_id=event.chunk_id,
        sentence=event.sentence,
        full_text=event.sentence,
        score=float(score),
        date_match=bool(date_match),
        object_match_count=int(object_hits),
        entity_match_count=int(entity_hits),
        keyword_match_count=int(keyword_hits),
        methods=list(event.methods),
        locations=list(event.locations),
        directions=list(event.directions),
        event_dates=list(event.event_dates),
        quantities=list(event.quantities),
        target=event.target,
        action=event.action,
        event_type=event.event_type,
        event_time=event.event_time,
        polarity=event.polarity,
        indicator=indicator,
        value=value,
        confidence=float(event.confidence),
        evidence_kind=str((event.metadata or {}).get("kind") or "event_store"),
        metadata=dict(event.metadata or {}),
    )


def infer_dates_from_source_for_intent(intent: QueryUnderstandingResult, source: str) -> List[str]:
    """Пробует восстановить дату события из имени документа в рамках периода запроса.

    Частый формат корпуса: документы называются `10.docx`, `11.docx`, `12.docx`,
    а месяц/год приходят из вопроса. Для статистики это критично: иначе заголовочные
    строки без явной даты либо теряются, либо загрязняют общий пул.
    """
    if not getattr(intent, 'has_date_constraint', False):
        return []
    source = str(source or '')
    base = source.replace('\\', '/').split('/')[-1]
    out: List[str] = []

    explicit = re.search(
        r"(?<!\d)(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})(?!\d)",
        base,
    )
    if explicit:
        day, month, year = map(int, explicit.groups())
        if year < 100:
            year += 2000
        try:
            out.append(datetime(year, month, day).strftime("%Y-%m-%d"))
        except (TypeError, ValueError) as exc:
            logger.debug("Ignoring invalid date inferred from source %s: %s", source, exc)

    if not out and getattr(intent, 'start_date', None) and getattr(intent, 'end_date', None):
        m = re.search(r'(?<!\d)(\d{1,2})(?!\d)', base)
        if m:
            day = int(m.group(1))
            start = intent.start_date
            end = intent.end_date
            if start.year == end.year and start.month == end.month:
                try:
                    dt = datetime(start.year, start.month, day)
                    if start <= dt <= end:
                        out.append(dt.strftime('%Y-%m-%d'))
                except (TypeError, ValueError) as exc:
                    logger.debug("Ignoring invalid day inferred from source %s: %s", source, exc)
    return list(dict.fromkeys(out))[:4]


def enrich_record_dates_from_intent(records: List[EvidenceRecord], intent: QueryUnderstandingResult) -> List[EvidenceRecord]:
    for rec in records:
        if not rec.event_dates:
            inferred = infer_dates_from_source_for_intent(intent, rec.source)
            if inferred:
                rec.event_dates = inferred
                rec.date_match = True
                rec.confidence = min(1.0, float(rec.confidence or 0.0) + 0.06)
    return records


def filter_records_to_intent_scope(records: List[EvidenceRecord], intent: QueryUnderstandingResult, allowed_doc_ids: set[str] | None = None) -> List[EvidenceRecord]:
    """Жёстко сужает evidence под вопрос, чтобы статистика не считалась по всему корпусу."""
    allowed_doc_ids = {str(x) for x in (allowed_doc_ids or set()) if x is not None and str(x)}
    scoped: List[EvidenceRecord] = []
    start_s = intent.start_date.strftime('%Y-%m-%d') if getattr(intent, 'start_date', None) else None
    end_s = intent.end_date.strftime('%Y-%m-%d') if getattr(intent, 'end_date', None) else None
    date_values = set()
    for dt in getattr(intent, 'dates', []) or []:
        try:
            date_values.add(dt.strftime('%Y-%m-%d'))
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug("Ignoring malformed parsed date in evidence scope: %s", exc)
    for rec in enrich_record_dates_from_intent(list(records or []), intent):
        if allowed_doc_ids and str(rec.doc_id) not in allowed_doc_ids:
            continue
        if getattr(intent, 'has_date_constraint', False):
            dates = list(rec.event_dates or [])
            if date_values:
                if not (set(dates) & date_values):
                    continue
            elif start_s and end_s:
                if not dates or not any(start_s <= d <= end_s for d in dates):
                    continue
        if not _record_matches_object_scope(rec, intent):
            continue
        required_indicators = set(getattr(getattr(intent, "contract", None), "indicators", ()) or ())
        if required_indicators:
            if not rec.indicator:
                rec.indicator, rec.value = _extract_indicator_value(rec.sentence)
            if rec.indicator not in required_indicators:
                continue
        required_actions = set(getattr(getattr(intent, "contract", None), "actions", ()) or ())
        if required_actions and getattr(intent, "question_type", "") in {"fact", "binary_fact", "count"}:
            found_actions = set(get_domain_profile(rec.sentence).actions_in(rec.sentence))
            if not found_actions.intersection(required_actions):
                continue
        required_polarity = str(getattr(getattr(intent, "contract", None), "polarity", "any") or "any")
        if required_polarity != "any" and rec.polarity != required_polarity:
            continue
        if getattr(intent, 'question_type', '') == 'binary_fact':
            # Бинарный вопрос должен подтверждаться той же строкой/единицей текста,
            # где есть искомый объект.  Так мы не отвечаем «Да» по соседнему
            # событию только из-за совпавшей даты или слова «поражён».
            qnorm = normalize_question(getattr(intent, 'raw_question', '') or '')
            snorm = normalize_question(getattr(rec, 'sentence', '') or '')
            strike_q = any(cue in qnorm for cue in ['поражен', 'поражена', 'поражены', 'уничтожен', 'уничтожена', 'уничтожены', 'поврежден', 'подавлен'])
            if strike_q and not any(cue in snorm for cue in ['поражен', 'поражена', 'поражены', 'уничтожен', 'уничтожена', 'уничтожены', 'поврежден', 'подавлен']):
                continue
            if strike_q and str(getattr(rec, 'polarity', '') or '') == 'negative':
                # Отрицательное упоминание оставляем только если вообще нет положительных
                # записей; это решается в build_binary, а здесь чистим явный мусор.
                continue
        if getattr(intent, 'question_type', '') == 'target':
            # Для вопросов "какие цели поражены ..." не считаем служебные/заголовочные
            # совпадения без явно выделенной цели. Иначе ответ превращается в общую
            # статистику по способу вместо списка целей.
            if not str(getattr(rec, 'target', '') or '').strip():
                continue
            if str(getattr(rec, 'polarity', '') or '') == 'negative':
                continue
        scoped.append(rec)
    return scoped


def build_evidence_records(intent: QueryUnderstandingResult, candidates: Iterable[Tuple[float, Dict[str, Any]]], limit: int = 80) -> List[EvidenceRecord]:
    results: List[EvidenceRecord] = []
    seen: set[tuple[str, str]] = set()
    object_terms = [normalize_question(x) for x in intent.object_terms]
    keywords = [normalize_question(x) for x in (intent.keywords + intent.answer_terms)]

    for base_score, meta in candidates:
        text = str((meta or {}).get("text") or "").strip()
        if not text:
            continue
        for unit, section, section_match in _split_units_with_sections(text, intent):
            norm = normalize_question(unit)
            if not norm:
                continue
            key = (str((meta or {}).get("doc_id")), norm)
            if key in seen:
                continue
            seen.add(key)
            object_hits = sum(1 for term in object_terms if term and term in norm)
            struct_score, struct_matches = entity_overlap_score(intent.raw_question, unit)
            entity_hits = len(struct_matches)
            kw_hits = sum(1 for kw in keywords if kw and kw in norm)
            if section_match:
                object_hits = max(object_hits, 1)
                entity_hits = max(entity_hits, 1)
                struct_score = max(float(struct_score), 0.82)
            score = float(base_score) + 0.55 * object_hits + 0.45 * float(struct_score) + 0.08 * kw_hits
            if section_match:
                score += 0.9
            if (meta or {}).get("date_match"):
                score += 0.28
            if _sentence_positive_for_intent(intent, unit):
                score += 0.85
            if (meta or {}).get("answer_rescue"):
                score += 0.18
            methods = _extract_methods(unit)
            locations = _extract_locations(unit)
            directions = _extract_directions(unit)
            if section_match and section.get('direction') and section.get('direction') not in directions:
                directions = [section['direction']] + directions
            document_date = _metadata_document_date(meta)
            event_type = "any"
            event_time = _extract_event_time(unit)
            event_dates = _extract_event_dates(unit, reference_date=document_date)
            quantities = _extract_quantities(unit)
            polarity = _detect_polarity(unit)
            target = _extract_target(unit, has_section=bool(section))
            if not event_dates and document_date and (section or event_time or target):
                event_dates = [document_date.strftime("%Y-%m-%d")]
            action = _extract_action(unit)
            indicator, value = _extract_indicator_value(unit)
            confidence = min(
                1.0,
                0.35
                + 0.12 * object_hits
                + 0.12 * entity_hits
                + 0.06 * kw_hits
                + (0.12 if (meta or {}).get('date_match') else 0.0)
                + (0.12 if methods else 0.0)
                + (0.10 if section_match else 0.0),
            )
            results.append(EvidenceRecord(
                doc_id=str((meta or {}).get("doc_id") or ""),
                source=str((meta or {}).get("source") or (meta or {}).get("doc_name") or (meta or {}).get("name") or ""),
                chunk_id=(meta or {}).get("chunk_id"),
                sentence=unit.strip(),
                full_text=text,
                score=score,
                date_match=bool((meta or {}).get("date_match")),
                object_match_count=int(object_hits),
                entity_match_count=int(entity_hits),
                keyword_match_count=int(kw_hits),
                methods=methods,
                locations=locations,
                directions=directions,
                event_dates=event_dates,
                quantities=quantities,
                target=target,
                action=action,
                event_type=event_type,
                event_time=event_time,
                polarity=polarity,
                indicator=indicator,
                value=value,
                confidence=confidence,
                evidence_kind="section_event" if section_match else "sentence",
                metadata={
                    **dict(meta or {}),
                    "kind": "section_event" if section_match else "sentence",
                    "section_header": section.get('header', ''),
                    "section_family": section.get('family', ''),
                    "section_number": section.get('number', ''),
                    "section_direction": section.get('direction', ''),
                    "section_terms": section.get('terms', []),
                    "section_event_type": section.get("event_type", "any"),
                    "section_context": section.get("context", ""),
                },
            ))
    results.sort(key=lambda x: (x.score, x.confidence), reverse=True)
    return results[:limit]
