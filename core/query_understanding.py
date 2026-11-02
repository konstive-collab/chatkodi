"""Разбор пользовательского вопроса для структурированного ответа ChatKodi 1.7."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import re

from core.date_range_parser import parse_date_constraints
from core.entity_resolution import extract_structured_entities
from core.knowledge_units import resolve_count_unit
from core.generation.router import detect_generation_kind
from core.domain_profile import get_domain_profile


@dataclass
class QueryAnswerContract:
    dates: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    indicators: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    polarity: str = "any"
    answer_type: str = "text"


@dataclass
class QueryUnderstandingResult:
    question_type: str
    normalized_question: str
    raw_question: str
    object_terms: List[str] = field(default_factory=list)
    identifiers: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    answer_terms: List[str] = field(default_factory=list)
    dates: List[datetime] = field(default_factory=list)
    date_mode: str = "none"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    wants_statistics: bool = False
    wants_exact_fact: bool = False
    wants_summary: bool = False
    wants_comparison: bool = False
    wants_timeline: bool = False
    wants_listing: bool = False
    wants_range_aggregation: bool = False
    statistics_focus: str = "general"
    prefer_structured_answer: bool = False
    llm_presentation_allowed: bool | None = None
    routing_source: str = "rules"
    routing_confidence: float = 1.0
    routing_notes: List[str] = field(default_factory=list)
    report_mode: str = "none"
    count_unit: str = "claim"
    count_unit_label: str = "подтверждённых фактов"
    count_unit_short: str = "факт."
    event_type: str = "any"
    generation_kind: str = "none"
    contract: QueryAnswerContract = field(default_factory=QueryAnswerContract)

    @property
    def has_date_constraint(self) -> bool:
        return self.date_mode != "none" and bool(self.start_date or self.end_date or self.dates)

    @property
    def is_range_query(self) -> bool:
        return self.date_mode in {"range", "discrete_dates"}

    @property
    def primary_object(self) -> str:
        return self.object_terms[0] if self.object_terms else ""

    @property
    def wants_algorithmic_answer(self) -> bool:
        return self.question_type in {"binary_fact", "location", "method", "target", "count", "frequency", "timeline", "comparison", "range_summary", "activity_report"}


# Compatibility for third-party integrations written against ChatKodi 1.7/1.8.
QueryIntent = QueryUnderstandingResult


def normalize_question(text: str) -> str:
    t = (text or "").lower().replace("ё", "е")
    t = re.sub(r"\s+", " ", t).strip()
    return t


_DATE_WORD_RE = r"(?:\d{1,2}\s+)?(?:январ[яеь]|феврал[яеь]|март[ае]?|апрел[яеь]|ма[йяе]|июн[яеь]|июл[яеь]|август[ае]?|сентябр[яеь]|октябр[яеь]|ноябр[яеь]|декабр[яеь])|\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?"
_EN_DATE_WORD_RE = r"(?:\d{1,2}\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+\d{1,2})?(?:,?\s+\d{4})?"


def _clean_object_candidate(value: str) -> str:
    """Очищает объект, извлечённый из вопроса вида «был ли поражён X 15 июня»."""
    text = normalize_question(value)
    text = re.sub(r"\b(?:на|в|во|за|с|по|от|до|при|около|у)\b.*$", "", text).strip()
    text = re.sub(r"\b(?:\d{1,2}|202\d|20\d{2}|года|год)\b.*$", "", text).strip()
    text = re.sub(_DATE_WORD_RE + r".*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(rf"\b(?:on|during|from|between)\s+{_EN_DATE_WORD_RE}.*$", "", text, flags=re.IGNORECASE).strip()
    text = text.strip(" ?!.,;:-—–\t")
    # Не считаем сам вопросный глагол объектом.
    text = re.sub(r"^(?:ли|был|была|были|было|поражен|поражена|поражены|уничтожен|уничтожена|уничтожены|поврежден|повреждена|повреждены|подавлен|подавлена|подавлены)\s+", "", text).strip()
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE).strip()
    return text


def _extract_binary_action_object(text: str) -> List[str]:
    """Достаёт обычный объект из бинарного вопроса без номера/ID.

    Пример: «Был ли поражен склад БК 15 июня 2025 года?» -> «склад бк».
    Без этого система видит только дату и глагол «поражён», после чего может
    ошибочно ответить по соседнему событию.
    """
    t = normalize_question(text)
    out: List[str] = []
    action = r"(?:поражен[аыо]?|уничтожен[аыо]?|поврежден[аыо]?|подавлен[аыо]?|обнаружен[аыо]?|зафиксирован[аыо]?)"
    for pat in (
        rf"(?:был|была|были|было)\s+ли\s+{action}\s+(?P<obj>.+?)(?=$|\?|\s+{_DATE_WORD_RE}|\s+20\d{{2}})",
        rf"{action}\s+ли\s+(?P<obj>.+?)(?=$|\?|\s+{_DATE_WORD_RE}|\s+20\d{{2}})",
        rf"{action}\s+(?P<obj>.+?)(?=$|\?|\s+{_DATE_WORD_RE}|\s+20\d{{2}})",
        rf"(?:was|were|is|are)\s+(?P<obj>.+?)\s+(?:destroyed|damaged|detected|deployed|activated|installed|recorded|confirmed)(?=$|\?|\s+(?:on|during)\s+{_EN_DATE_WORD_RE}|\s+in\s+20\d{{2}})",
    ):
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            obj = _clean_object_candidate(m.group('obj'))
            if 3 <= len(obj) <= 80 and not obj.isdigit():
                out.append(obj)
    return list(dict.fromkeys(out))[:4]


def extract_identifiers(text: str) -> List[str]:
    t = normalize_question(text)
    ids = set()
    ids.update(re.findall(r"(?:№|#)\s*([0-9]{1,12})", t))
    ids.update(re.findall(r"\b([0-9]{6,})\b", t))
    ids.update(re.findall(r"\b([a-zа-яё0-9]{2,24}[-_][a-zа-яё0-9]{1,24})\b", t, flags=re.IGNORECASE))
    return sorted(x for x in ids if x)


def _extract_named_organizations(text: str) -> List[str]:
    """Extract legal entity names used as the factual-answer object."""
    source = str(text or "")
    prefixes = r"(?:ООО|АО|ПАО|ЗАО|ИП|LLC|LTD|INC|CORP(?:ORATION)?)"
    found: List[str] = []
    patterns = (
        rf"\b(?P<prefix>{prefixes})\s*[«\"'](?P<name>[^»\"']{{1,80}})[»\"']",
        rf"\b(?P<prefix>{prefixes})\s+(?P<name>[A-Za-zА-ЯЁа-яё][A-Za-zА-ЯЁа-яё-]{{1,40}}(?:\s+[A-Za-zА-ЯЁа-яё][A-Za-zА-ЯЁа-яё-]{{1,40}}){{0,2}})",
    )
    trailing = {
        "за", "на", "по", "от", "до", "с", "в", "во", "и", "или",
        "on", "at", "for", "from", "to", "and", "or",
    }
    for pattern in patterns:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            name_tokens = match.group("name").split()
            while name_tokens and normalize_question(name_tokens[-1]) in trailing:
                name_tokens.pop()
            if not name_tokens:
                continue
            value = normalize_question(f"{match.group('prefix')} {' '.join(name_tokens)}")
            if value not in found:
                found.append(value)
    return found


def extract_object_terms(text: str) -> List[str]:
    t = normalize_question(text)
    out = set()
    for m in re.finditer(r"\b([a-zа-яё]{2,24}[-_/][a-zа-яё0-9]{1,24})\b", t, flags=re.IGNORECASE):
        out.add(m.group(1))
    for m in re.finditer(r"\b([a-zа-яё]{3,24}\d{1,8})\b", t, flags=re.IGNORECASE):
        out.add(m.group(1))
    for entity in extract_structured_entities(t):
        alpha_tokens = [tok for tok in entity.alpha_norm.split() if tok]
        if len(alpha_tokens) > 2:
            continue
        if any(tok in {"ли", "был", "было", "были", "какой", "какая", "какое", "какие", "сколько", "использовалось", "использовался", "применялось"} for tok in alpha_tokens):
            continue
        out.add(entity.raw)
        out.add(f"{entity.alpha_norm}-{entity.number_norm}")
        out.add(entity.canonical)

    # Универсальный fallback для обычных объектов без номера/ID.  Это особенно
    # важно для точных бинарных вопросов: «склад БК» не должен теряться только
    # потому, что он не похож на технический идентификатор.
    for obj in _extract_binary_action_object(text):
        out.add(obj)

    for organization in _extract_named_organizations(text):
        out.add(organization)

    for match in re.finditer(
        r"\b(специалист|сотрудник|инженер|эксперт|specialist|employee|engineer|expert)\s+([a-zа-яё-]{2,32}(?:\s+[a-zа-яё-]{2,32}){0,2})",
        t,
        flags=re.IGNORECASE,
    ):
        out.add(f"{match.group(1)} {match.group(2)}")

    cleaned = set()
    for value in out:
        value = re.sub(r"\s+", " ", str(value or "")).strip(" ?!.,;:-—–\t")
        if not value or re.search(r"(?:№|#)\s*$", value):
            continue
        cleaned.add(value)
    return sorted(cleaned)


def extract_keywords(text: str) -> List[str]:
    t = normalize_question(text)
    words = re.findall(r"[a-zа-я0-9№]+", t, flags=re.IGNORECASE)
    stop = {
        "и","в","во","на","по","к","ко","с","со","из","за","до","от","для","про","при","без","у","о","об","обо",
        "а","но","или","ли","же","что","это","этот","эта","эти","тот","та","те","там","тут","как","когда","где",
        "почему","зачем","над","под","между","через","бы","был","были","есть","нет","да","не","ни",
        "кто","кого","кому","кем","какой","какая","какое","какие","каком","каких","какого","какую","чей","чья","чье","чьи","чем","сколько",
        "a","an","the","and","or","but","in","on","at","to","from","for","of","with","without","by","as",
        "is","are","was","were","be","been","being","do","does","did","has","have","had","can","could","would","should",
        "what","which","who","whom","whose","where","when","why","how","this","that","these","those","there","here",
    }
    out = []
    for w in words:
        if len(w) < 3 and not w.isdigit() and not w.startswith("№"):
            continue
        if w in stop:
            continue
        out.append(w)
    return list(dict.fromkeys(out))[:40]


def detect_statistics_focus(q: str) -> str:
    q = normalize_question(q)
    if any(tok in q for tok in ["где чаще", "на каком направлении чаще", "направление чаще", "какое направление чаще", "where most often", "which location most", "which direction most"]):
        return "direction"
    if any(tok in q for tok in ["каким способом чаще", "чем чаще", "каким средством чаще", "как чаще примен", "which method most", "how most often", "most common method"]):
        return "method"
    if any(tok in q for tok in [
        "какие цели", "какие объекты", "какие цели чаще", "что чаще пораж", "какая цель чаще", "по каким целям чаще",
        "какие цели пораж", "какие объекты пораж", "что было пораж", "какие пораженные цели",
        "which targets", "which objects", "what targets", "what objects",
    ]):
        return "target"
    if any(tok in q for tok in ["в какие дни чаще", "по каким дням чаще", "когда чаще", "which days", "when most often"]):
        return "date"
    if any(tok in q for tok in ["сколько", "количество", "сколько раз", "how many", "how much", "number of", "count of"]):
        return "count"
    return "general"


def detect_event_type(q: str) -> str:
    # Relation semantics are resolved from document context by embeddings/LLM.
    # The deterministic parser intentionally does not key off query vocabulary.
    return "any"


def extract_answer_terms(text: str) -> List[str]:
    q = normalize_question(text)
    terms: List[str] = []
    groups = [
        (["где", "направлен", "место", "локац", "где чаще", "на каком направлении чаще"], ["направлен", "место", "локац", "район", "пункт", "чаще"]),
        (["чем", "каким образом", "каким способом", "каким средством", "поражен", "чем чаще"], ["применением", "бпла", "fpv", "изделием", "ударом", "птур", "способ", "средство"]),
        (["сколько", "колич"], ["количество", "сколько", "всего", "итого"]),
        (["как часто", "периодич", "частот", "чаще", "регулярно", "повторяем"], ["как часто", "периодич", "регулярно", "повторялось", "частота", "чаще"]),
        (["какие цели", "какие объекты", "что пораж", "цели пораж", "объекты пораж"], ["цель", "цели", "объект", "объекты", "поражен", "поражены"]),
        (["главное", "сводк", "итог", "что происходило", "за период", "проанализ", "анализ"], ["главное", "основное", "чаще всего", "ключевое", "период", "анализ"]),
        (["сравни", "отлич", "измен"], ["сравнение", "изменение", "разница"]),
        (["по дням", "хронолог", "таймлайн", "в какие дни"], ["по дням", "хронология", "timeline", "дни"]),
        (["where", "which location", "which direction"], ["location", "place", "direction", "area"]),
        (["how many", "how much", "number of", "count of"], ["count", "number", "total"]),
        (["how often", "frequency", "most often", "most common"], ["frequency", "regularity", "most often"]),
        (["compare", "comparison", "difference", "changed", "change between"], ["comparison", "change", "difference"]),
        (["timeline", "chronology", "by day", "which days"], ["timeline", "chronology", "days"]),
        (["summary", "summarize", "overview", "what happened", "analyze"], ["summary", "main", "key", "period", "analysis"]),
    ]
    for markers, expanded in groups:
        if any(m in q for m in markers):
            terms.extend(expanded)
    return list(dict.fromkeys(terms))[:20]


def _detect_question_type(q: str, *, date_mode: str = "none") -> str:
    # «Подробная работа X за период» — это не поиск одного факта и не простая
    # статистика. Это отчёт об активности сущности с группировкой по времени.
    if date_mode in {"range", "discrete_dates"} and any(token in q for token in ["подроб", "детальн", "полная", "полный", "развернут", "проанализ", "анализ", "detailed", "complete", "analyze", "analysis"]) and any(token in q for token in ["работ", "активн", "как работал", "что делал", "что выполнял", "действ", "activity", "worked", "did", "performed"]):
        return "activity_report"

    binary_starters = (
        "использовалось ли", "использовался ли", "использовалась ли", "использовались ли",
        "применялось ли", "применялся ли", "применялась ли", "применялись ли",
        "был ли", "была ли", "были ли", "было ли",
        "поражен ли", "поражена ли", "поражены ли", "уничтожен ли", "поврежден ли",
    )
    if "статист" in q or "statistics" in q:
        return "count"
    if any(token in q for token in ["сравни", "отлич", "изменил", "изменилось", "compare", "comparison", "difference", "changed", "change between"]):
        return "comparison"
    if any(token in q for token in ["какие цели", "какие объекты", "какие цели пораж", "какие объекты пораж", "что было пораж", "что поражено", "which targets", "which objects", "what targets", "what objects"]):
        return "target"
    if any(token in q for token in ["где чаще", "на каком направлении чаще", "чем чаще", "каким способом чаще", "каким средством чаще", "чаще всего", "с какой периодичностью", "насколько регулярно", "как часто", "периодич", "частот", "how often", "frequency", "most often", "most common"]):
        return "frequency"
    if any(token in q for token in ["по дням", "хронолог", "таймлайн", "в какие дни", "timeline", "chronology", "by day", "which days"]):
        return "timeline"
    if q.startswith(binary_starters):
        return "binary_fact"
    if re.match(r"^(?:is|are|was|were|did|does|has|have|can|could)\b", q):
        return "binary_fact"
    # Общая грамматическая форма вопроса с частицей «ли». Она позволяет
    # распознавать "есть ли информация", "зафиксировано ли..." и другие
    # формулировки без словаря предметных глаголов.
    if re.match(r"^(?:\S+\s+){0,3}ли(?:\s|$)", q):
        return "binary_fact"
    if q.endswith("?") and q.startswith(("использовался ", "использовалась ", "использовалось ", "использовались ", "применялся ", "применялась ", "применялось ", "применялись ", "поражен ", "поражена ", "поражены ", "уничтожен ", "уничтожена ", "уничтожены ")):
        return "binary_fact"
    if any(token in q for token in ["сколько", "какое количество", "сколько раз", "how many", "how much", "number of", "count of"]):
        return "count"
    if any(token in q for token in ["что происходило", "что было", "сводк", "главное", "итог", "за период", "за промежуток", "проанализ", "анализ", "what happened", "summary", "summarize", "overview", "analyze", "analysis"]):
        return "range_summary" if date_mode == "range" else "summary"
    if any(token in q for token in ["на каком направлен", "на каких направлен", "какие направления", "каких направлениях", "где ", "где?", "в каком районе", "where ", "where?", "which location", "which area", "which direction"]):
        if any(token in q for token in ["чаще", "регулярнее", "чаще всего"]):
            return "frequency"
        return "location"
    if any(token in q for token in ["чем ", "каким образом", "каким способом", "каким средством", "how ", "by what", "which method", "what method"]):
        if "чаще" in q:
            return "frequency"
        return "method"
    # Диапазон дат задаёт область поиска, но сам по себе не означает запрос
    # сводки. Тип ответа определяется смыслом вопроса выше.
    return "fact"


def analyze_question(question: str, reference_date: Optional[datetime] = None) -> QueryUnderstandingResult:
    normalized = normalize_question(question)
    parsed_dates = parse_date_constraints(question, reference_date=reference_date)
    qtype = _detect_question_type(normalized, date_mode=parsed_dates.get("mode", "none"))
    # Диапазон не всегда означает широкий analytics-ответ.
    # Например: "на каких направлениях работал X с 10 по 12" — это location over range,
    # а не range_summary на весь корпус. Но evidence всё равно надо строго ограничить периодом.
    wants_range_aggregation = parsed_dates.get("mode", "none") in {"range", "discrete_dates"} and qtype in {"frequency", "timeline", "comparison", "summary", "range_summary", "count", "location", "method", "target", "activity_report"}
    wants_statistics = qtype in {"count", "frequency", "summary", "comparison", "timeline", "range_summary", "target", "activity_report"}
    wants_exact_fact = qtype in {"binary_fact", "location", "method", "fact"}
    wants_summary = qtype in {"summary", "comparison", "timeline", "range_summary", "activity_report"}
    wants_comparison = qtype == "comparison"
    wants_timeline = qtype in {"timeline", "range_summary", "activity_report"}
    wants_listing = any(token in normalized for token in ["перечисли", "список", "все случаи", "какие ", "какие цели", "какие объекты", "list ", "list all", "which "])
    count_unit = resolve_count_unit(question, qtype, detect_statistics_focus(question))
    object_terms = extract_object_terms(question)
    indicators = get_domain_profile(question).indicators_in(question)
    actions = get_domain_profile(question).actions_in(question)
    contract_dates = tuple(
        dict.fromkeys(
            dt.strftime("%Y-%m-%d")
            for dt in list(parsed_dates.get("dates") or [])
            if isinstance(dt, datetime)
        )
    )
    if not contract_dates and isinstance(parsed_dates.get("start"), datetime) and parsed_dates.get("start") == parsed_dates.get("end"):
        contract_dates = (parsed_dates["start"].strftime("%Y-%m-%d"),)
    answer_type = (
        "number" if indicators
        else "boolean" if qtype == "binary_fact"
        else "count" if qtype == "count"
        else "location" if qtype == "location"
        else "method" if qtype == "method"
        else "list" if qtype in {"target", "timeline"}
        else "text"
    )
    polarity = "negative" if re.search(r"\b(?:не|not|no)\b", normalized) else "any"
    return QueryUnderstandingResult(
        question_type=qtype,
        normalized_question=normalized,
        raw_question=question,
        object_terms=object_terms,
        identifiers=extract_identifiers(question),
        keywords=extract_keywords(question),
        answer_terms=extract_answer_terms(question),
        date_mode=parsed_dates.get("mode", "none"),
        start_date=parsed_dates.get("start"),
        end_date=parsed_dates.get("end"),
        dates=list(parsed_dates.get("dates") or []),
        wants_statistics=wants_statistics,
        wants_exact_fact=wants_exact_fact,
        wants_summary=wants_summary,
        wants_comparison=wants_comparison,
        wants_timeline=wants_timeline,
        wants_listing=wants_listing,
        wants_range_aggregation=wants_range_aggregation,
        statistics_focus=detect_statistics_focus(question),
        report_mode="activity" if qtype == "activity_report" else "none",
        count_unit=count_unit.key,
        count_unit_label=count_unit.label,
        count_unit_short=count_unit.short,
        prefer_structured_answer=True if qtype == "activity_report" else False,
        event_type=detect_event_type(question),
        generation_kind=detect_generation_kind(question).value,
        contract=QueryAnswerContract(
            dates=contract_dates,
            objects=tuple(object_terms),
            indicators=indicators,
            actions=actions,
            polarity=polarity,
            answer_type=answer_type,
        ),
    )


__all__ = [
    "QueryAnswerContract",
    "QueryUnderstandingResult",
    "QueryIntent",
    "analyze_question",
    "normalize_question",
    "extract_identifiers",
    "extract_object_terms",
    "extract_keywords",
]
