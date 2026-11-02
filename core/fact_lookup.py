"""Deterministic fact layer: exact values, table rows and per-day aggregation.

The retrieval stage usually finds the right document; what used to fail is
everything after it:

* the value inside a long document (prefix truncation cut it off, or a
  "plan" figure stood next to the "actual" one);
* table rows that lost their header after chunking;
* questions over many documents ("how many days did X work in July", "on
  which days did vehicle A123BC go out") that no top-k context can answer.

This module answers those questions directly from the loaded documents, with
the exact source line as evidence, in milliseconds and without the LLM. It is
domain-agnostic: it only relies on dates, identifiers, person names, table
headers and the words of the question.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.date_range_parser import parse_date_constraints
from core.identifiers import extract_normalized_identifiers, normalize_identifier
from core.logging_utils import get_logger

logger = get_logger(__name__)

try:  # optional: pymorphy3 lemmatizer
    from core.lemmatizer import get_lemmatizer

    _LEMMATIZER = get_lemmatizer()
    if not getattr(_LEMMATIZER, "available", False):
        _LEMMATIZER = None
except Exception:  # pragma: no cover - lemmatizer is optional
    _LEMMATIZER = None


# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #

_QUESTION_WORDS = {
    "кто", "что", "какой", "какая", "какое", "какие", "каков", "какова", "каково", "каковы",
    "сколько", "когда", "где", "куда", "откуда", "почему", "зачем", "как", "чей", "чья",
    "чьё", "чьи", "ли", "какого", "каком", "каких", "какую", "какому", "каким",
}
_STOPWORDS = {
    "и", "в", "во", "на", "по", "к", "ко", "с", "со", "из", "за", "до", "от", "для", "про",
    "при", "без", "у", "о", "об", "а", "но", "или", "же", "это", "этот", "эта", "эти", "тот",
    "та", "те", "там", "тут", "бы", "был", "была", "было", "были", "быть", "есть", "не", "ни",
    "год", "года", "году", "г", "день", "дни", "дней", "дня", "дата", "даты", "число", "числа",
    "весь", "все", "всего", "всех", "мне", "нам", "пожалуйста", "подскажи", "скажи", "покажи",
    "найди", "найти", "указан", "указано", "указать", "документ", "документе", "документах",
    "отчет", "отчёт", "отчете", "отчёте", "отчетах", "раз", "период", "месяц", "неделя",
    "the", "a", "an", "of", "in", "on", "at", "for", "to", "is", "was", "were", "be",
    "он", "она", "оно", "они", "его", "ее", "её", "их", "им", "ему", "ей", "мы", "вы", "я", "ты",
}
_MONTH_STEMS = (
    "январ", "феврал", "март", "апрел", "мая", "май", "мае", "июн", "июл", "август",
    "сентябр", "октябр", "ноябр", "декабр",
)
# Words that turn a value into a *different* value («план по выручке»).
_CONTRAST_QUALIFIERS = {
    "план", "плановый", "планируемый", "прогноз", "прогнозный", "норматив", "норма", "лимит",
    "бюджет", "цель", "целевой", "ожидаемый", "прошлый", "предыдущий", "средний",
}
_ACTUAL_MARKERS = {"факт", "фактический", "фактически", "итог", "итого", "составить", "составлять"}
_NEGATION = re.compile(
    r"\b(?:отсутств\w*|в\s+отпуске|отпуск\w*|больничн\w*|не\s+вышел\w*|не\s+работал\w*|уволен\w*|прогул\w*|выходн(?:ой|ые))\b",
    flags=re.IGNORECASE,
)
_PERSON_CUES = (
    "сотрудник", "специалист", "водитель", "работник", "инженер", "менеджер", "кассир",
    "мастер", "оператор", "бухгалтер", "кладовщик", "электрик", "охранник", "механик",
    "гражданин", "гражданка", "господин", "г-н", "г-жа", "товарищ", "директор", "руководитель",
)
_SURNAME_SUFFIX = re.compile(
    r"(?:ов|ев|ёв|ин|ын|ова|ева|ёва|ина|ына|ский|цкий|ская|цкая|енко|ук|юк|их|ых|ко|ян|дзе|швили)$",
    flags=re.IGNORECASE,
)

_COUNT_DAYS = re.compile(
    r"\bсколько\b(?:\s+\w+){0,3}?\s+(?:рабочих\s+|календарных\s+)?(?:дн(?:ей|я)|сут(?:ок|ки)|смен\w*|выход\w*\s+на\s+работу|выезд\w*|рейс\w*)\b",
    flags=re.IGNORECASE,
)
_LIST_DATES = re.compile(
    # «до какого числа / к какому числу» is a deadline, not a list of days.
    r"\b(?:в\s+какие|какие|по\s+каким|в\s+каких)\s+(?:дни|даты|числа|днях)\b|\bв\s+какой\s+день\b|(?<!до\s)\bкакого\s+числа\b",
    flags=re.IGNORECASE,
)
_PLURAL_ANSWER = re.compile(r"\b(?:какие|перечисли\w*|список|кто\s+\w+ли)\b", flags=re.IGNORECASE)

_NUMBER = re.compile(
    r"(?<![\w.,])\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?(?![\d])|(?<![\w.,])\d+(?:[.,]\d+)?(?![\d])"
)
_UNIT = re.compile(
    r"\s*(?:руб(?:\.|л(?:ей|я|ь)?)?|₽|тыс\.?\s*руб\.?|млн\.?\s*руб\.?|тыс\.?|млн\.?|км|шт\.?|%|чел\.?|человек|кг|тонн?|т\.|л\.|ч\.|час(?:ов|а)?|мин\.?|дн(?:ей|я)?|ед\.?|\$|€)",
    flags=re.IGNORECASE,
)
_DATE_TOKEN = re.compile(
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+\d{4})?(?:\s*(?:г\.|года?))?|\b\d{1,2}:\d{2}\b",
    flags=re.IGNORECASE,
)
_DATE_RANGE_IN_QUESTION = re.compile(
    r"\b\d{1,2}\s*(?:по|до|-|–|—|и)\s*\d{1,2}(?:\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря))?"
    r"|\b(?:с|по|до|на|за)\s+\d{1,2}(?=\s+(?:по|до|и)\b|\s*$)|(?<![/\-№\d])\b(?:19|20)\d{2}\s*(?:г\.|года?\b|году\b)",
    flags=re.IGNORECASE,
)
_MONTH_YEAR = re.compile(
    r"\b(январ|феврал|март|апрел|ма[йяе]|июн|июл|август|сентябр|октябр|ноябр|декабр)[а-я]*\s+((?:19|20)\d{2})\b"
)
_MONTH_INDEX = {
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "май": 5, "мая": 5, "мае": 5, "июн": 6, "июл": 7,
    "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
}
_PERSON_IN_TEXT = re.compile(
    r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.?|\b[А-ЯЁ]\.\s?[А-ЯЁ]\.\s?[А-ЯЁ][а-яё]+|\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:вич|вна|чна|ична|ьич|ич)\b"
)


def _norm(text: str) -> str:
    return str(text or "").lower().replace("ё", "е")


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zа-я0-9]+", _norm(text))


_LEMMA_CACHE: Dict[str, str] = {}


def _lemma(word: str) -> str:
    word = _norm(word)
    cached = _LEMMA_CACHE.get(word)
    if cached is not None:
        return cached
    value = word
    if _LEMMATIZER is not None and re.fullmatch(r"[а-я]+", word):
        try:
            value = _norm(_LEMMATIZER.lemmatize_word(word))
        except Exception:
            value = word
    if len(_LEMMA_CACHE) < 200_000:
        _LEMMA_CACHE[word] = value
    return value


def _lemmas(text: str) -> set[str]:
    return {_lemma(token) for token in _tokens(text) if len(token) > 1}


def _stem_key(lemma: str) -> str:
    """Cheap stem for fuzzy matching when pymorphy is unavailable."""
    return lemma[:6] if len(lemma) > 6 else lemma


# --------------------------------------------------------------------------- #
# question frame
# --------------------------------------------------------------------------- #


@dataclass
class PersonRef:
    surname: str
    initials: str = ""
    display: str = ""

    def _surname_pattern(self) -> str:
        base = self.surname
        low = base.lower()
        if re.search(r"(?:ский|цкий)$", low):
            stem = base[:-2]
            endings = "(?:ий|ого|ому|им|ом|ая|ой|ую)"
        elif re.search(r"(?:ская|цкая)$", low):
            stem = base[:-2]
            endings = "(?:ая|ой|ую|ий|ого|ому|им|ом)"
        elif re.search(r"(?:ова|ева|ёва|ина|ына)$", low):
            stem = base[:-1]
            endings = "(?:|а|у|ым|е|ой|ом)"
        elif re.search(r"(?:ов|ев|ёв|ин|ын)$", low):
            stem = base
            endings = "(?:|а|у|ым|е|ой|ом)"
        elif re.search(r"(?:ову|ову|еву|ину|ыну)$", low):
            stem = base[:-1]
            endings = "(?:|а|у|ым|е|ой|ом)"
        elif re.search(r"(?:овым|евым|иным|ыным)$", low):
            stem = base[:-2]
            endings = "(?:|а|у|ым|е|ой|ом)"
        else:
            stem = base
            endings = "(?:|а|у|ом|ем|е|ой)"
        return re.escape(stem) + endings

    def pattern(self) -> re.Pattern[str]:
        surname = self._surname_pattern()
        if self.initials:
            first = self.initials[0]
            second = self.initials[1] if len(self.initials) > 1 else ""
            init_after = rf"\s+{first}\.\s?" + (rf"{second}\.?" if second else "")
            init_before = rf"{first}\.\s?" + (rf"{second}\.\s?" if second else "")
            full = rf"\s+{first}[а-яё]+" + (rf"\s+{second}[а-яё]+" if second else "")
            body = rf"(?:{surname}(?:{init_after}|{full})|{init_before}{surname})"
        else:
            body = surname
        return re.compile(rf"(?<![А-Яа-яЁё]){body}(?![а-яё])")

    def surname_only_pattern(self) -> re.Pattern[str]:
        return re.compile(rf"(?<![А-Яа-яЁё]){self._surname_pattern()}(?![а-яё])")


@dataclass
class QueryFrame:
    question: str
    kind: str
    start: Optional[date] = None
    end: Optional[date] = None
    identifiers: List[str] = field(default_factory=list)
    persons: List[PersonRef] = field(default_factory=list)
    terms: List[str] = field(default_factory=list)
    qualifiers: set[str] = field(default_factory=set)
    wants_person: bool = False
    wants_value: bool = False
    plural: bool = False
    # "sum" | "avg" | "max" | "min" over the values of a period, "" otherwise.
    aggregate: str = ""
    # «кто/какая машина чаще всех …»: rank entities by number of mentions.
    frequency: bool = False
    # Terms before aggregation words were removed (restored when the corpus
    # itself names the attribute that way, e.g. «средний чек»).
    raw_terms: List[str] = field(default_factory=list)
    raw_qualifiers: set[str] = field(default_factory=set)

    @property
    def has_entity(self) -> bool:
        return bool(self.identifiers or self.persons)

    @property
    def single_day(self) -> bool:
        return bool(self.start and self.end and self.start == self.end)

    def period_label(self) -> str:
        if not self.start:
            return "за весь период"
        if self.single_day:
            return self.start.strftime("%d.%m.%Y")
        if self.start.day == 1 and self.end and (self.end.month != (self.end.replace(day=28)).month or self.end.day >= 28):
            months = [
                "январе", "феврале", "марте", "апреле", "мае", "июне", "июле", "августе",
                "сентябре", "октябре", "ноябре", "декабре",
            ]
            if self.start.month == self.end.month and self.start.year == self.end.year:
                return f"в {months[self.start.month - 1]} {self.start.year}"
            if self.start.month == 1 and self.end.month == 12 and self.start.year == self.end.year:
                return f"в {self.start.year} году"
        return f"с {self.start:%d.%m.%Y} по {self.end:%d.%m.%Y}"


def extract_persons(question: str) -> List[PersonRef]:
    text = str(question or "")
    found: List[PersonRef] = []
    seen: set[Tuple[str, str]] = set()

    def add(surname: str, initials: str, display: str) -> None:
        key = (surname.lower(), initials)
        if key in seen or surname.lower() in _QUESTION_WORDS:
            return
        seen.add(key)
        found.append(PersonRef(surname=surname, initials=initials, display=display.strip()))

    for match in re.finditer(r"\b([А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?)\s+([А-ЯЁ])\.\s?([А-ЯЁ])?\.?", text):
        add(match.group(1), (match.group(2) or "") + (match.group(3) or ""), match.group(0))
    for match in re.finditer(r"\b([А-ЯЁ])\.\s?([А-ЯЁ])\.\s?([А-ЯЁ][а-яё]+)", text):
        add(match.group(3), match.group(1) + match.group(2), match.group(0))
    for match in re.finditer(r"\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+(?:вич|вна|чна|ична|ьич|ич))\b", text):
        add(match.group(1), match.group(2)[0] + match.group(3)[0], match.group(0))
    if not found:
        cue = "|".join(_PERSON_CUES)
        # Only the cue word is case-insensitive: the surname must be capitalised
        # («водитель Орлов», but not «водитель автомобиля»).
        for match in re.finditer(rf"\b(?i:{cue})\w*\s+([А-ЯЁ][а-яё]{{2,}})\b", text):
            add(match.group(1), "", match.group(1))
        for match in re.finditer(r"(?<=[а-яё,]\s)([А-ЯЁ][а-яё]{3,})\b", text):
            word = match.group(1)
            if _SURNAME_SUFFIX.search(word) and not word.lower().startswith(_MONTH_STEMS):
                add(word, "", word)
    return found


def _question_terms(question: str, persons: Sequence[PersonRef]) -> Tuple[List[str], set[str]]:
    text = _DATE_TOKEN.sub(" ", question)
    for person in persons:
        text = text.replace(person.display, " ")
    for match in re.finditer(r"[A-Za-zА-ЯЁа-яё0-9]*\d[A-Za-zА-ЯЁа-яё0-9 ]{0,12}", text):
        if extract_normalized_identifiers(match.group(0)):
            text = text.replace(match.group(0), " ")
    terms: List[str] = []
    qualifiers: set[str] = set()
    for token in _tokens(text):
        if token.isdigit() or len(token) < 3:
            continue
        if token in _QUESTION_WORDS or token in _STOPWORDS or token.startswith(_MONTH_STEMS):
            continue
        lemma = _lemma(token)
        if lemma in _STOPWORDS or lemma in _QUESTION_WORDS:
            continue
        if lemma in _CONTRAST_QUALIFIERS:
            qualifiers.add(lemma)
        if lemma not in terms:
            terms.append(lemma)
    return terms, qualifiers


def _as_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def parse_question(question: str) -> QueryFrame:
    parsed = parse_date_constraints(question)
    persons = extract_persons(question)
    raw_ids = extract_normalized_identifiers(question)
    # «автомобиль А138СК 77» yields both A138CK and A138CK77: keep the full one.
    identifiers = [item for item in raw_ids if not any(other != item and other.startswith(item) for other in raw_ids)]
    terms, qualifiers = _question_terms(question, persons)
    norm = _norm(question)
    if _COUNT_DAYS.search(norm):
        kind = "count_days"
    elif _LIST_DATES.search(norm):
        kind = "list_dates"
    else:
        kind = "lookup"
    start = _as_date(parsed.get("start"))
    end = _as_date(parsed.get("end")) or start
    if start is None:
        # «выручка за 04.08»: day.month without a year (the year is aligned
        # to the documents later, like «5 августа»).
        short_date = re.search(r"(?<![\d.,])(\d{1,2})[./](\d{2})(?![\d]|[./]\d)", norm)
        if short_date:
            day_value, month_value = int(short_date.group(1)), int(short_date.group(2))
            if 1 <= day_value <= 31 and 1 <= month_value <= 12:
                try:
                    start = end = date(datetime.now().year, month_value, day_value)
                except ValueError:
                    start = end = None
    if start is None:
        # «в какой день августа 2025», «выручка июля 2025»: a month in the
        # genitive with a year is a period the date parser does not catch.
        month_match = _MONTH_YEAR.search(norm)
        if month_match:
            month = _MONTH_INDEX[month_match.group(1)[:3]]
            year = int(month_match.group(2))
            start = date(year, month, 1)
            end = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1))
    frequency = detect_frequency(question)
    aggregate = "" if frequency else detect_aggregate(question)
    raw_terms, raw_qualifiers = list(terms), set(qualifiers)
    if aggregate or frequency:
        # «общая/средняя/самая большая выручка»: the aggregation words are not
        # part of the attribute and must not steer line matching («средний чек»).
        stripped = [term for term in terms if term not in _AGGREGATE_LEMMAS]
        if stripped:
            terms = stripped
            qualifiers = {item for item in qualifiers if item not in _AGGREGATE_LEMMAS}
    return QueryFrame(
        question=question,
        kind=kind,
        start=start,
        end=end,
        identifiers=identifiers,
        persons=persons,
        terms=terms,
        qualifiers=qualifiers,
        aggregate=aggregate,
        frequency=frequency,
        raw_terms=raw_terms,
        raw_qualifiers=raw_qualifiers,
        wants_person=bool(re.match(r"\s*(?:кто|кем|кого|кому|чей|чья|чьи)\b", norm)),
        wants_value=bool(
            re.search(r"\b(?:какая|какой|какое|каков\w*|сколько|чему\s+равн\w*|размер|сумм\w*|величин\w*)\b", norm)
            # keyword-style query «выручка 21.07.2025» asks for the value too
            or not any(token in _QUESTION_WORDS for token in _tokens(question))
        ),
        plural=bool(_PLURAL_ANSWER.search(norm)),
    )


# --------------------------------------------------------------------------- #
# corpus view
# --------------------------------------------------------------------------- #


@dataclass
class DocView:
    doc_id: str
    name: str
    day: Optional[date]
    lines: List[str]
    norm_lines: List[str]
    identifiers_by_line: List[set[str]] = field(default_factory=list)
    identifiers: set[str] = field(default_factory=set)
    _lemma_lines: Optional[List[set[str]]] = None

    def lemma_lines(self) -> List[set[str]]:
        if self._lemma_lines is None:
            self._lemma_lines = [_lemmas(line) for line in self.lines]
        return self._lemma_lines


_FILENAME_DATE = (
    re.compile(r"(?<!\d)(\d{1,2})[._-](\d{1,2})[._-](20\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})[._-](\d{1,2})[._-](\d{1,2})(?!\d)"),
)


def _doc_day(document: Any) -> Optional[date]:
    raw = str(((getattr(document, "metadata", None) or {}).get("document_date")) or "")
    if raw:
        try:
            return datetime.strptime(raw[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
    name = str(getattr(document, "filename", "") or getattr(document, "name", "") or "")
    for index, pattern in enumerate(_FILENAME_DATE):
        match = pattern.search(name)
        if not match:
            continue
        try:
            if index == 0:
                return date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            continue
    return None


def _split_lines(content: str) -> List[str]:
    lines = []
    for raw in str(content or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("--- "):
            continue
        lines.append(line)
    return lines


class CorpusView:
    def __init__(self, documents: Iterable[Any]) -> None:
        self.docs: List[DocView] = []
        for document in documents:
            content = str(getattr(document, "content", "") or "")
            lines = _split_lines(content)
            ids_by_line = [set(extract_normalized_identifiers(line)) if re.search(r"\d", line) else set() for line in lines]
            view = DocView(
                doc_id=str(getattr(document, "id", "")),
                name=str(getattr(document, "name", "") or getattr(document, "filename", "")),
                day=_doc_day(document),
                lines=lines,
                norm_lines=[_norm(line) for line in lines],
                identifiers_by_line=ids_by_line,
            )
            for ids in ids_by_line:
                view.identifiers.update(ids)
            self.docs.append(view)
        self.by_id = {doc.doc_id: doc for doc in self.docs}


_VIEW_CACHE: Dict[int, Tuple[Tuple[int, int], CorpusView]] = {}


def corpus_view(processor: Any) -> Optional[CorpusView]:
    documents = list((getattr(processor, "documents", None) or {}).values()) if isinstance(getattr(processor, "documents", None), dict) else list(getattr(processor, "documents", None) or [])
    if not documents:
        return None
    signature = (len(documents), sum(int(getattr(doc, "size_chars", 0) or 0) for doc in documents))
    cached = _VIEW_CACHE.get(id(processor))
    if cached and cached[0] == signature:
        return cached[1]
    view = CorpusView(documents)
    _VIEW_CACHE.clear()
    _VIEW_CACHE[id(processor)] = (signature, view)
    return view


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #


def _line_has_identifiers(doc: DocView, index: int, identifiers: Sequence[str]) -> bool:
    ids = doc.identifiers_by_line[index] if index < len(doc.identifiers_by_line) else set()
    return all(identifier in ids for identifier in identifiers)


def _person_hits(doc: DocView, person: PersonRef) -> List[int]:
    strict = person.pattern()
    hits = [index for index, line in enumerate(doc.lines) if strict.search(line)]
    if hits or not person.initials:
        return hits
    return []


def _term_score(frame: QueryFrame, lemma_line: set[str], norm_line: str) -> Tuple[float, int]:
    if not frame.terms:
        return 0.0, 0
    stems = {_stem_key(item) for item in lemma_line}
    matched = 0
    for term in frame.terms:
        if term in lemma_line or _stem_key(term) in stems or (len(term) >= 5 and term[:5] in norm_line):
            matched += 1
    return matched / len(frame.terms), matched


# A clock time «15.35» / «7.40» (no unit after it) is not a quantity.
_CLOCK_TIME = re.compile(
    r"(?<![\d.,])(?:[01]?\d|2[0-3])\.[0-5]\d(?![\d.,])(?!\s*(?:руб|₽|тыс|млн|км|кг|т\b|т\.|%|м\b|м\.|л\b|шт|ед|чел|мм|см|га|ч\b|час))",
    flags=re.IGNORECASE,
)
# A number that is part of a name or a number sign: «№22», «#5», «Изделие-51», «Молния-2».
_NUMBER_IN_NAME = re.compile(r"(?:№|#)\s*\d+|[A-Za-zА-ЯЁа-яё»\"]-\s?\d+")


def _mask(text: str, pattern: "re.Pattern[str]") -> str:
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def _numbers_outside_dates(line: str) -> List[Tuple[int, int, str]]:
    masked = _DATE_TOKEN.sub(lambda m: " " * len(m.group(0)), line)
    masked = _mask(_mask(masked, _CLOCK_TIME), _NUMBER_IN_NAME)
    for ident in re.finditer(r"[A-Za-zА-ЯЁа-яё]+\s?\d[\dA-Za-zА-ЯЁа-яё ]*", masked):
        if extract_normalized_identifiers(ident.group(0)):
            masked = masked[: ident.start()] + " " * (ident.end() - ident.start()) + masked[ident.end():]
    return [(m.start(), m.end(), m.group(0)) for m in _NUMBER.finditer(masked)]


# Units a question may ask for -> the unit as written after a number.
_ASKED_UNITS = (
    (re.compile(r"\bединиц|\bед\.", re.IGNORECASE), re.compile(r"\s*ед\b|\s*единиц", re.IGNORECASE)),
    (re.compile(r"\bчеловек|\bчел\.|\bлюдей", re.IGNORECASE), re.compile(r"\s*чел|\s*человек", re.IGNORECASE)),
    (re.compile(r"\bрубл|\bруб\.|₽", re.IGNORECASE), re.compile(r"\s*(?:руб|₽|тыс\.?\s*руб|млн\.?\s*руб)", re.IGNORECASE)),
    (re.compile(r"\bкилометр|\bкм\b", re.IGNORECASE), re.compile(r"\s*км\b", re.IGNORECASE)),
    (re.compile(r"\bтонн", re.IGNORECASE), re.compile(r"\s*(?:т\b|т\.|тонн)", re.IGNORECASE)),
    (re.compile(r"\bштук|\bшт\.", re.IGNORECASE), re.compile(r"\s*шт", re.IGNORECASE)),
    (re.compile(r"\bпроцент|%", re.IGNORECASE), re.compile(r"\s*%", re.IGNORECASE)),
)


def _asked_unit(question: str) -> Optional["re.Pattern[str]"]:
    for asked, written in _ASKED_UNITS:
        if asked.search(question or ""):
            return written
    return None


def _score_line(frame: QueryFrame, doc: DocView, index: int) -> float:
    line = doc.lines[index]
    norm_line = doc.norm_lines[index]
    lemma_line = doc.lemma_lines()[index]
    coverage, matched = _term_score(frame, lemma_line, norm_line)
    score = coverage
    if frame.terms and matched == 0:
        score -= 0.5
    contrast = (lemma_line & _CONTRAST_QUALIFIERS) - frame.qualifiers
    if contrast and not frame.qualifiers:
        score -= 0.6
    if frame.qualifiers and not (lemma_line & frame.qualifiers):
        score -= 0.4
    if not frame.qualifiers and lemma_line & _ACTUAL_MARKERS:
        score += 0.15
    if frame.wants_value and _numbers_outside_dates(line):
        score += 0.3
    if frame.wants_person and _PERSON_IN_TEXT.search(line):
        score += 0.3
    if ":" in line and "|" in line:
        score += 0.05  # keyed table row: compact and self-describing
    if len(line) > 600:
        score -= 0.2
    return score


@dataclass
class FactAnswer:
    text: str
    kind: str
    confidence: float
    sources: List[str]
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    value: str = ""


def _fmt_day(day: Optional[date]) -> str:
    return day.strftime("%d.%m.%Y") if day else "без даты"


def _in_period(day: Optional[date], frame: QueryFrame) -> bool:
    if not frame.start:
        return True
    if day is None:
        return False
    return frame.start <= day <= (frame.end or frame.start)


def _entity_label(frame: QueryFrame) -> str:
    labels = [person.display for person in frame.persons]
    for identifier in frame.identifiers:
        labels.append(identifier)
    return ", ".join(labels)


def _restore_identifier_spelling(frame: QueryFrame) -> str:
    question = frame.question
    spans = []
    for match in re.finditer(r"[A-Za-zА-ЯЁа-яё№#]*\s?\d[\dA-Za-zА-ЯЁа-яё ]{0,10}", question):
        piece = match.group(0).strip()
        if extract_normalized_identifiers(piece):
            spans.append(piece)
    return ", ".join(spans) or ", ".join(frame.identifiers)


def _entity_display(frame: QueryFrame) -> str:
    parts = [person.display for person in frame.persons]
    if frame.identifiers:
        parts.append(_restore_identifier_spelling(frame))
    return ", ".join(part for part in parts if part)


def _matching_lines(doc: DocView, frame: QueryFrame) -> List[int]:
    candidates = set(range(len(doc.lines)))
    if frame.identifiers:
        candidates &= {index for index in range(len(doc.lines)) if _line_has_identifiers(doc, index, frame.identifiers)}
    for person in frame.persons:
        candidates &= set(_person_hits(doc, person))
    return sorted(candidates)


def _aggregate_days(view: CorpusView, frame: QueryFrame) -> Optional[FactAnswer]:
    if not frame.has_entity:
        return None
    per_day: Dict[date, List[Tuple[DocView, str]]] = {}
    excluded: Dict[date, List[Tuple[DocView, str]]] = {}
    undated: List[Tuple[DocView, str]] = []
    for doc in view.docs:
        if frame.identifiers and not all(identifier in doc.identifiers for identifier in frame.identifiers):
            continue
        if doc.day is not None and not _in_period(doc.day, frame):
            continue
        lines = _matching_lines(doc, frame)
        if not lines:
            continue
        positive = [doc.lines[i] for i in lines if not _NEGATION.search(doc.lines[i])]
        if doc.day is None:
            undated.extend((doc, line) for line in positive[:1])
            continue
        if positive:
            per_day.setdefault(doc.day, []).append((doc, positive[0]))
        else:
            excluded.setdefault(doc.day, []).append((doc, doc.lines[lines[0]]))
    if not per_day and not excluded:
        return FactAnswer(
            text=(
                f"В документах {frame.period_label()} не найдено упоминаний «{_entity_display(frame)}». "
                "Проверьте написание (ФИО с инициалами или госномер) и период."
            ),
            kind=frame.kind,
            confidence=0.55,
            sources=[],
        )
    days = sorted(per_day)
    entity = _entity_display(frame)
    sources = []
    for day in days:
        for doc, _line in per_day[day]:
            if doc.name not in sources:
                sources.append(doc.name)
    evidence = [
        {"date": day.isoformat(), "source": per_day[day][0][0].name, "text": per_day[day][0][1]}
        for day in days
    ]
    lines_out: List[str] = []
    count_label = f"{len(days)} {_plural_days(len(days))}"
    if frame.kind == "count_days":
        lines_out.append(f"**{count_label}** — «{entity}» указан(а) в документах {frame.period_label()}.")
    else:
        lines_out.append(f"«{entity}» встречается в документах {frame.period_label()}: **{count_label}**.")
    lines_out.append("")
    show_rows = frame.kind == "list_dates" or len(days) <= 12
    if show_rows:
        for day in days:
            doc, line = per_day[day][0]
            snippet = line if len(line) <= 220 else line[:217] + "…"
            lines_out.append(f"- {_fmt_day(day)} — {snippet}  _(«{doc.name}»)_")
    else:
        lines_out.append("Даты: " + ", ".join(_fmt_day(day) for day in days))
    if excluded:
        lines_out.append("")
        lines_out.append(
            "Не засчитаны (упоминание с отметкой об отсутствии/отпуске): "
            + ", ".join(_fmt_day(day) for day in sorted(excluded))
        )
    if undated:
        lines_out.append("")
        lines_out.append(
            "Есть упоминания в документах без даты (в подсчёт не вошли): "
            + ", ".join(dict.fromkeys(doc.name for doc, _ in undated))
        )
    lines_out.append("")
    lines_out.append("Метод: один документ с датой = один день; учитываются только документы с этой сущностью в тексте.")
    lines_out.append(_sources_line(sources))
    return FactAnswer(
        text="\n".join(lines_out),
        kind=frame.kind,
        confidence=0.9,
        sources=sources,
        evidence=evidence,
        value=str(len(days)),
    )


def _plural_days(count: int) -> str:
    tail = count % 100
    if 11 <= tail <= 14:
        return "дней"
    if count % 10 == 1:
        return "день"
    if count % 10 in {2, 3, 4}:
        return "дня"
    return "дней"


def _sources_line(sources: Sequence[str], limit: int = 10) -> str:
    if not sources:
        return ""
    shown = ", ".join(sources[:limit])
    extra = f" и ещё {len(sources) - limit}" if len(sources) > limit else ""
    return f"📚 **Источники:** {shown}{extra}"


def _keyed_cells(line: str) -> List[Tuple[str, str]]:
    cells = []
    for part in line.split(" | "):
        if ": " in part:
            key, value = part.split(": ", 1)
            cells.append((key.strip(), value.strip()))
    return cells


def _extract_value(frame: QueryFrame, line: str) -> str:
    cells = _keyed_cells(line)
    if cells:
        best_key, best_value, best_score = "", "", 0.0
        for key, value in cells:
            key_lemmas = _lemmas(key)
            overlap = sum(1 for term in frame.terms if term in key_lemmas or _stem_key(term) in {_stem_key(k) for k in key_lemmas})
            if overlap > best_score:
                best_key, best_value, best_score = key, value, overlap
        if best_value and not any(identifier == normalize_identifier(best_value) for identifier in frame.identifiers):
            return f"{best_key}: {best_value}"
    if frame.wants_person:
        query_surnames = {person.surname.lower()[:5] for person in frame.persons}
        people = [m.group(0) for m in _PERSON_IN_TEXT.finditer(line) if m.group(0).split()[0].lower()[:5] not in query_surnames]
        if people:
            return ", ".join(dict.fromkeys(people))
    if frame.wants_value:
        # A number the user typed («Молния-2», «склад №3») names the object;
        # it cannot be the answer.
        # Numbers of dates («с 7 по 13 июня», «04.08») are not object names.
        question_without_dates = _DATE_RANGE_IN_QUESTION.sub(" ", _DATE_TOKEN.sub(" ", frame.question))
        asked = {re.sub(r"\D", "", item) for item in re.findall(r"\d+", question_without_dates)}
        numbers = [item for item in _numbers_outside_dates(line) if re.sub(r"\D", "", item[2]) not in asked]
        unit_pattern = _asked_unit(frame.question)
        if numbers and unit_pattern is not None:
            # «сколько единиц …»: the number written with that unit, not the first one.
            with_unit = [item for item in numbers if unit_pattern.match(line, item[1])]
            if with_unit:
                start, end, value = with_unit[0]
                unit = _UNIT.match(line, end)
                return (value + (unit.group(0) if unit else "")).strip()
            if frame.terms:
                return ""  # the asked quantity is not on this line
        if numbers:
            norm_line = _norm(line)
            anchor = 0
            for term in frame.terms:
                position = norm_line.find(term[:5])
                if position >= 0:
                    anchor = position
                    break
            after = [item for item in numbers if item[0] >= anchor] or numbers
            start, end, value = after[0]
            unit = _UNIT.match(line, end)
            return (value + (unit.group(0) if unit else "")).strip()
    return ""


def _lookup(view: CorpusView, frame: QueryFrame, ranked_doc_ids: Sequence[str]) -> Optional[FactAnswer]:
    if not frame.terms and not frame.wants_person:
        return None
    constrained = bool(frame.start or frame.has_entity)
    if not constrained:
        return None
    rank = {doc_id: position for position, doc_id in enumerate(ranked_doc_ids)}
    candidates: List[DocView] = []
    for doc in view.docs:
        if frame.identifiers and not all(identifier in doc.identifiers for identifier in frame.identifiers):
            continue
        if frame.start and doc.day is not None and not _in_period(doc.day, frame):
            continue
        if frame.start and doc.day is None and doc.doc_id not in rank:
            continue
        candidates.append(doc)
    if not candidates:
        return None
    # Undated documents only compete when no dated document matches the period.
    if frame.start and any(doc.day is not None for doc in candidates):
        candidates = [doc for doc in candidates if doc.day is not None]
    if frame.start and not frame.single_day:
        return None  # a period lookup is an aggregation the caller handles elsewhere

    scored: List[Tuple[float, DocView, int]] = []
    for doc in candidates:
        for index in _matching_lines(doc, frame):
            score = _score_line(frame, doc, index)
            score -= 0.02 * min(10, rank.get(doc.doc_id, 10))
            scored.append((score, doc, index))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_doc, best_index = scored[0]
    if best_score < 0.55:
        return None
    best_line = best_doc.lines[best_index]
    value = _extract_value(frame, best_line)
    if frame.wants_value and not value:
        return None
    # Two different documents competing for the same day with equal strength
    # mean the question is ambiguous: let the LLM/structured path decide.
    runner_up = next((item for item in scored[1:] if item[1].doc_id != best_doc.doc_id), None)
    if runner_up and runner_up[0] >= best_score - 0.05 and _extract_value(frame, runner_up[1].lines[runner_up[2]]) != value:
        return None

    selected = [(best_score, best_doc, best_index)]
    if frame.plural:
        selected += [item for item in scored[1:8] if item[0] >= max(0.55, best_score - 0.25) and item[1].doc_id == best_doc.doc_id]
    lines_out: List[str] = []
    if value:
        lines_out.append(f"**{value}**")
        lines_out.append("")
    day_note = f" (документ от {_fmt_day(best_doc.day)})" if best_doc.day else ""
    lines_out.append(f"Подтверждение из «{best_doc.name}»{day_note}:")
    for _score, doc, index in selected:
        lines_out.append(f"> {doc.lines[index]}")
    lines_out.append("")
    lines_out.append(_sources_line([best_doc.name]))
    confidence = max(0.0, min(0.97, 0.55 + best_score * 0.35))
    if not value:
        # Descriptive questions («что происходило», «на каких позициях») need
        # a synthesis of several lines: the lines go to the LLM as evidence,
        # but the fast path must not answer with one arbitrary sentence.
        confidence = min(confidence, 0.5)
    return FactAnswer(
        text="\n".join(lines_out),
        kind="lookup",
        confidence=confidence,
        sources=[best_doc.name],
        evidence=[{"date": best_doc.day.isoformat() if best_doc.day else "", "source": best_doc.name, "text": doc.lines[index]} for _s, doc, index in selected],
        value=value,
    )


def unasked_qualifiers(question: str, text: str) -> set[str]:
    """Contrast qualifiers («план», «прогноз»…) present in ``text`` but not asked for."""
    asked = {_lemma(token) for token in _tokens(question)}
    found = {_lemma(token) for token in _tokens(text)}
    return (found & _CONTRAST_QUALIFIERS) - asked


def _shift_year(value: date, year: int) -> Optional[date]:
    try:
        return value.replace(year=year)
    except ValueError:  # 29 February
        return None


def align_year_to_corpus(frame: QueryFrame, view: CorpusView) -> None:
    """«в июле», «5 августа» without a year mean the year the documents are from.

    The date parser defaults to the current year; reports are usually about a
    past period. Pick the latest year whose documents cover the period.
    """
    if not frame.start or re.search(r"\b(?:19|20)\d{2}\b", frame.question):
        return
    years = sorted({doc.day.year for doc in view.docs if doc.day}, reverse=True)
    for year in years:
        start = _shift_year(frame.start, year)
        end = _shift_year(frame.end or frame.start, year)
        if not start or not end:
            continue
        if any(doc.day and start <= doc.day <= end for doc in view.docs):
            frame.start, frame.end = start, end
            return


_STATISTICS_WORDS = re.compile(
    r"\b(?:статистик\w*|частот\w*|как\s+часто|динамик\w*|тенденц\w*|тренд\w*|периодичн\w*)\b",
    flags=re.IGNORECASE,
)
_SUM_WORDS = re.compile(r"\b(?:всего|итого|общ\w*|суммарн\w*|в\s+сумме|сумм\w*)\b", flags=re.IGNORECASE)
_AVG_WORDS = re.compile(r"\bсредн\w*\b", flags=re.IGNORECASE)
_MAX_WORDS = re.compile(
    r"\b(?:максимальн\w*|наибольш\w*|рекордн\w*|сам(?:ый|ая|ое|ые|ой|ую|ого|ом|ых|ым)\s+(?:больш\w*|высок\w*|крупн\w*|лучш\w*)|больше\s+всего|лучш\w*\s+(?:день|результат\w*))\b",
    flags=re.IGNORECASE,
)
_MIN_WORDS = re.compile(
    r"\b(?:минимальн\w*|наименьш\w*|сам(?:ый|ая|ое|ые|ой|ую|ого|ом|ых|ым)\s+(?:маленьк\w*|низк\w*|мал\w*|худш\w*)|меньше\s+всего|худш\w*\s+(?:день|результат\w*))\b",
    flags=re.IGNORECASE,
)
_FREQUENCY_WORDS = re.compile(
    r"\b(?:чаще\s+(?:всех|всего|других|остальных)|больше\s+всех|реже\s+(?:всех|всего)|сам(?:ый|ая|ое|ые|ой|ую|ого|ом|ых|ым)\s+част\w*|наибольш\w*\s+(?:числ\w*|количеств\w*))\b",
    flags=re.IGNORECASE,
)
# Lemmas that describe *how* to aggregate, not *what* attribute is asked.
_AGGREGATE_LEMMAS = {
    "общий", "суммарный", "итого", "итоговый", "всего", "сумма", "средний", "среднедневной", "дневной",
    "максимальный", "минимальный", "наибольший", "наименьший", "самый", "большой", "высокий", "крупный",
    "маленький", "низкий", "малый", "больший", "меньший", "дневный", "рекордный", "лучший", "хороший", "худший", "плохой", "частый", "часто",
    "чаще", "больше", "меньше", "много", "мало", "реже", "редко",
}
_ROLE_LEMMAS = {
    "водитель", "сотрудник", "специалист", "работник", "инженер", "менеджер", "кассир", "мастер",
    "оператор", "бухгалтер", "кладовщик", "электрик", "охранник", "механик", "директор", "руководитель",
    "экспедитор", "грузчик", "продавец", "курьер", "врач", "учитель",
}
_VEHICLE_WORDS = re.compile(
    r"\b(?:машин\w*|автомобил\w*|авто|автомашин\w*|госномер\w*|номер\w*|транспорт\w*|грузовик\w*|тс)\b",
    flags=re.IGNORECASE,
)
_VEHICLE_LEMMAS = {"машина", "автомобиль", "авто", "автомашина", "госномер", "номер", "транспорт", "грузовик", "тс"}
_PERSON_SHORT = r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.?"
# Verbs that say nothing about which lines to count («кто чаще всех работал
# кассиром» = every line that names a cashier).
_GENERIC_ACTIONS = {
    "работать", "быть", "являться", "находиться", "числиться", "присутствовать", "участвовать",
    "выходить", "значиться", "указать", "упоминаться", "встречаться", "стоять", "дежурить",
}
# Normalized Russian licence plate (Latin look-alikes): A123BC77 / A123BC777.
_PLATE_KEY = re.compile(r"[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}")
_LATIN_TO_CYRILLIC = str.maketrans("ABEKMHOPCTYX", "АВЕКМНОРСТУХ")


def _plate_display(key: str) -> str:
    match = re.fullmatch(r"([A-Z]\d{3}[A-Z]{2})(\d{2,3})", key)
    if not match:
        return key
    return f"{match.group(1).translate(_LATIN_TO_CYRILLIC)} {match.group(2)}"


_DAY_MENTION = re.compile(
    r"\b(\d{1,2})[./](\d{1,2})[./]((?:19|20)?\d{2})\b|\b(\d{1,2})(?:-?го)?\s+(январ|феврал|март|апрел|ма[йяе]|июн|июл|август|сентябр|октябр|ноябр|декабр)[а-я]*(?:\s+((?:19|20)\d{2}))?"
)
_COMPARE_WORDS = re.compile(r"\b(?:сравн\w*|разниц\w*|отлича\w*|по\s+сравнению|чем\s+\w+\s+отлича\w*|изменил\w*)\b")


def extract_day_mentions(question: str) -> List[Tuple[Optional[int], int, int]]:
    """(year or None, month, day) for every explicit day in the question."""
    found: List[Tuple[Optional[int], int, int]] = []
    for match in _DAY_MENTION.finditer(_norm(question)):
        if match.group(1):
            year = int(match.group(3))
            year = year + 2000 if year < 100 else year
            found.append((year, int(match.group(2)), int(match.group(1))))
        else:
            year = int(match.group(6)) if match.group(6) else None
            found.append((year, _MONTH_INDEX[match.group(5)[:3]], int(match.group(4))))
    explicit = [item[0] for item in found if item[0]]
    if explicit:
        found = [(item[0] or explicit[-1], item[1], item[2]) for item in found]
    return found


def detect_frequency(question: str) -> bool:
    norm = _norm(question)
    if _FREQUENCY_WORDS.search(norm):
        return True
    # «кто больше всего выезжал» ranks people; «в какой день выручка была
    # больше всего» is a maximum.
    return bool(
        re.search(r"\bбольше\s+всего\b", norm)
        and re.match(r"\s*(?:кто|как(?:ой|ая|ое|ие)\s+\w+|чей|чья)\b", norm)
        and not re.search(r"\bв\s+какой\s+день|какого\s+числа|\bкогда\b|\bдат\w*\b", norm)
    )


def detect_aggregate(question: str) -> str:
    norm = _norm(question)
    if _MAX_WORDS.search(norm):
        return "max"
    if _MIN_WORDS.search(norm):
        return "min"
    if _AVG_WORDS.search(norm):
        return "avg"
    if _SUM_WORDS.search(norm):
        return "sum"
    return ""


# Everyday words for the attributes reports are written with. A synonym is
# used only when the question's own word does not occur in the documents.
_TERM_SYNONYMS = {
    "заработать": ("выручка", "доход"),
    "заработок": ("выручка", "доход"),
    "зарабатывать": ("выручка", "доход"),
    "доход": ("выручка",),
    "оборот": ("выручка",),
    "касса": ("выручка",),
    "продать": ("выручка", "продажа"),
    "наторговать": ("выручка",),
    "выручить": ("выручка",),
    "проехать": ("пробег",),
    "накатать": ("пробег",),
    "километраж": ("пробег",),
    "отгрузить": ("отгрузка", "отгружено"),
    "покупатель": ("чек",),
    "клиент": ("чек",),
}
_TERM_PRESENCE: Dict[Tuple[int, str], bool] = {}


def _corpus_has_term(view: CorpusView, term: str) -> bool:
    key = (id(view), term)
    cached = _TERM_PRESENCE.get(key)
    if cached is not None:
        return cached
    probe = term[:5] if len(term) >= 5 else term
    found = any(probe in line for doc in view.docs for line in doc.norm_lines)
    if len(_TERM_PRESENCE) > 5000:
        _TERM_PRESENCE.clear()
    _TERM_PRESENCE[key] = found
    return found


def _apply_term_synonyms(frame: QueryFrame, view: CorpusView) -> None:
    replaced: List[str] = []
    for term in frame.terms:
        synonyms = _TERM_SYNONYMS.get(term)
        if synonyms and not _corpus_has_term(view, term):
            target = next((item for item in synonyms if _corpus_has_term(view, item)), None)
            if target:
                if target not in replaced:
                    replaced.append(target)
                continue
        if term not in replaced:
            replaced.append(term)
    frame.terms = replaced


def _restore_attribute_terms(frame: QueryFrame, view: CorpusView) -> None:
    """Keep «средний чек» / «общий пробег» when the documents use that name.

    The aggregation itself stays: the average of daily «средний чек» values is
    exactly what «средний чек за июль» asks for.
    """
    if frame.terms == frame.raw_terms:
        return
    match = re.search(r"\b(средн\w*|общ\w*|суммарн\w*|итогов\w*)\s+([а-яё]{3,})", _norm(frame.question))
    if not match:
        return
    left, right = match.group(1)[:5], match.group(2)[:4]
    if right.startswith(_MONTH_STEMS) or _lemma(match.group(2)) in _AGGREGATE_LEMMAS:
        return
    pattern = re.compile(rf"\b{re.escape(left)}\w*\s+{re.escape(right)}")
    for doc in view.docs:
        if any(pattern.search(line) for line in doc.norm_lines):
            frame.terms = list(frame.raw_terms)
            frame.qualifiers = set(frame.raw_qualifiers)
            return


def _plural(count: int, one: str, few: str, many: str) -> str:
    tail = count % 100
    if 11 <= tail <= 14:
        return many
    if count % 10 == 1:
        return one
    if count % 10 in {2, 3, 4}:
        return few
    return many


def _value_unit(value: str) -> str:
    raw = value.split(":", 1)[-1]
    match = _NUMBER.search(raw)
    if not match:
        return ""
    unit = _UNIT.match(raw, match.end())
    return unit.group(0).strip() if unit else ""


def _term_in_line(term: str, doc: DocView, index: int) -> bool:
    lemma_line = doc.lemma_lines()[index]
    if term in lemma_line or _stem_key(term) in {_stem_key(item) for item in lemma_line}:
        return True
    # verbs vs nouns: «выезжал» ~ «выезд», «работал» ~ «работа»
    return len(term) >= 5 and term[:4] in doc.norm_lines[index]


def _persons_for_role(line: str, roles: Sequence[str]) -> List[str]:
    found: List[str] = []
    for role in roles:
        stem = re.escape(role[:5])
        for match in re.finditer(rf"(?i:{stem}\w*)\s*[:\-—]?\s*({_PERSON_SHORT})", line):
            found.append(match.group(1))
        for match in re.finditer(rf"({_PERSON_SHORT})\s*\((?i:{stem}\w*)\)", line):
            found.append(match.group(1))
    return found


def _person_key(display: str) -> Tuple[str, str]:
    compact = re.sub(r"[\s.]", "", display)
    match = re.match(r"([А-ЯЁа-яё-]+?)([А-ЯЁ]{1,2})$", compact)
    if match:
        surname, initials = match.group(1), match.group(2)
        return surname.lower() + initials.lower(), f"{surname} {'.'.join(initials)}."
    return compact.lower(), display.strip()


def _rank_entities(view: CorpusView, frame: QueryFrame) -> Optional[FactAnswer]:
    """«Кто/какая машина чаще всех …»: count mentions per person or vehicle."""
    norm_q = _norm(frame.question)
    roles = [term for term in frame.terms if term in _ROLE_LEMMAS]
    want_vehicle = bool(_VEHICLE_WORDS.search(norm_q)) and not roles
    want_person = not want_vehicle and bool(roles or re.match(r"\s*(?:кто|кого|кому|кем)\b", norm_q) or frame.wants_person)
    if not (want_person or want_vehicle):
        return None
    actions = [
        term for term in frame.terms
        if term not in _ROLE_LEMMAS and term not in _VEHICLE_LEMMAS and term not in _GENERIC_ACTIONS
    ]
    counts: Dict[str, int] = {}
    days: Dict[str, set] = {}
    labels: Dict[str, str] = {}
    sources: List[str] = []
    for doc in view.docs:
        if frame.start and (doc.day is None or not _in_period(doc.day, frame)):
            continue
        for index, line in enumerate(doc.lines):
            if actions and not all(_term_in_line(term, doc, index) for term in actions):
                continue
            if roles and not any(_term_in_line(term, doc, index) for term in roles):
                continue
            keys: List[Tuple[str, str]] = []
            if want_vehicle:
                plates = [item for item in sorted(doc.identifiers_by_line[index]) if _PLATE_KEY.fullmatch(item)]
                for identifier in plates:
                    keys.append((identifier, _plate_display(identifier)))
            else:
                people = _persons_for_role(line, roles) if roles else [m.group(0) for m in _PERSON_IN_TEXT.finditer(line)]
                keys = [_person_key(person) for person in people]
            for key, label in dict(keys).items():
                counts[key] = counts.get(key, 0) + 1
                days.setdefault(key, set()).add(doc.day)
                labels.setdefault(key, label)
                if doc.name not in sources:
                    sources.append(doc.name)
    # vehicles: keep the longest identifier (A138CK77 over A138CK)
    if want_vehicle:
        for key in list(counts):
            if any(other != key and other.startswith(key) for other in counts):
                counts.pop(key)
    if not counts:
        return None
    ranking = sorted(counts, key=lambda key: (-counts[key], -len(days[key]), labels[key]))
    reverse = bool(re.search(r"\bреже\b|\bменьше\s+всех\b", norm_q))
    if reverse:
        ranking = list(reversed(ranking))
    top = ranking[0]
    period = f" {frame.period_label()}" if frame.start else ""
    lead = "Реже всех" if reverse else "Чаще всех"
    count = counts[top]
    n_days = len([d for d in days[top] if d])
    day_note = f" в {n_days} {_plural(n_days, 'день', 'днях', 'днях')}" if n_days else ""
    out = [f"**{lead}{period} — {labels[top]}: {count} {_plural(count, 'раз', 'раза', 'раз')}**{day_note}.", ""]
    for position, key in enumerate(ranking[:10], 1):
        key_days = len([d for d in days[key] if d])
        out.append(f"{position}. {labels[key]} — {counts[key]} ({key_days} дн.)")
    criteria = ", ".join(f"«{term}»" for term in (roles + actions)) or "сущность"
    out.append("")
    out.append(f"Метод: учтены строки документов, где есть {criteria}; одна строка = один раз.")
    out.append(_sources_line(sources))
    return FactAnswer(
        text="\n".join(out),
        kind="ranking",
        confidence=0.85,
        sources=sources,
        value=f"{labels[top]}: {count}",
    )


def _compare_days(view: CorpusView, frame: QueryFrame, ranked_doc_ids: Sequence[str]) -> Optional[FactAnswer]:
    """«Сравни выручку 1 июля и 1 августа»: the value for each day + difference."""
    mentions = extract_day_mentions(frame.question)
    if len(mentions) < 2 or not frame.terms:
        return None
    years = sorted({doc.day.year for doc in view.docs if doc.day}, reverse=True)
    days: List[date] = []
    for year, month, day in mentions[:4]:
        candidates = [year] if year else years
        picked = None
        for candidate in candidates:
            try:
                value = date(candidate, month, day)
            except ValueError:
                continue
            if year or any(doc.day == value for doc in view.docs):
                picked = value
                break
        if picked and picked not in days:
            days.append(picked)
    if len(days) < 2:
        return None
    results: List[Tuple[date, FactAnswer]] = []
    for day in days:
        single = replace(frame, start=day, end=day, kind="lookup", wants_value=True, plural=False)
        answer = _lookup(view, single, ranked_doc_ids)
        if answer is None or not answer.value:
            return None
        results.append((day, answer))
    subject = frame.terms[0] if frame.terms[0] not in {"сравнить", "разница"} else (frame.terms[1] if len(frame.terms) > 1 else "значение")
    out = [f"**{subject.capitalize()}: " + "; ".join(f"{_fmt_day(day)} — {answer.value}" for day, answer in results) + "**", ""]
    numbers = [_to_number(answer.value.split(":", 1)[-1]) for _day, answer in results]
    if len(results) == 2 and all(number is not None for number in numbers):
        first, second = numbers
        delta = second - first
        sign = "+" if delta > 0 else "−" if delta < 0 else "±"
        unit = _value_unit(results[1][1].value)
        unit_suffix = f" {unit}" if unit else ""
        percent = f" ({sign}{abs(delta) / abs(first) * 100:.1f}%)".replace(".", ",") if first else ""
        out.append(f"Разница: {sign}{_format_number(abs(delta))}{unit_suffix}{percent} — с {_fmt_day(results[0][0])} по {_fmt_day(results[1][0])}.")
        out.append("")
    sources: List[str] = []
    for day, answer in results:
        for item in answer.evidence[:1]:
            out.append(f"- {_fmt_day(day)}, «{item['source']}»: > {item['text']}")
        sources.extend(source for source in answer.sources if source not in sources)
    out.append("")
    out.append(_sources_line(sources))
    return FactAnswer(
        text="\n".join(out),
        kind="comparison",
        confidence=0.9,
        sources=sources,
        evidence=[item for _day, answer in results for item in answer.evidence[:1]],
        value="; ".join(answer.value for _day, answer in results),
    )


_PLAN_CHECK = re.compile(r"\b(?:выполн\w*|перевыполн\w*|недовыполн\w*|факт\w*|достиг\w*|сравн\w*)\b")
_PLAN_LEMMAS = {"план", "плановый", "планируемый", "норматив", "норма", "цель", "целевой", "бюджет", "прогноз", "прогнозный"}


def _plan_vs_fact(view: CorpusView, frame: QueryFrame, ranked_doc_ids: Sequence[str]) -> Optional[FactAnswer]:
    """«Какой был план на 4 августа и выполнили ли его?»: plan, actual, percent."""
    asked_plan = frame.qualifiers & _PLAN_LEMMAS
    if not asked_plan or not frame.single_day or not _PLAN_CHECK.search(_norm(frame.question)):
        return None
    helper_terms = {"выполнить", "выполнение", "перевыполнить", "достичь", "сравнить", "факт", "фактический"}
    base_terms = [term for term in frame.terms if term not in helper_terms]
    plan_frame = replace(frame, terms=[term for term in base_terms if term in _PLAN_LEMMAS] or base_terms, wants_value=True, plural=False)
    plan = _lookup(view, plan_frame, ranked_doc_ids)
    if plan is None or not plan.evidence:
        return None
    # «план продаж» vs «фактическая выручка»: name the actual value with the
    # attribute words of the plan line itself («План по выручке на день: …»).
    plan_line = str(plan.evidence[0].get("text") or "")
    label = plan_line.split(":", 1)[0] if ":" in plan_line else plan_line
    attribute_terms = [
        lemma for lemma in (_lemma(token) for token in _tokens(label))
        if len(lemma) >= 4 and lemma not in _PLAN_LEMMAS and lemma not in _STOPWORDS
        and lemma not in _QUESTION_WORDS and not lemma.startswith(_MONTH_STEMS) and not lemma.isdigit()
    ]
    actual_terms = attribute_terms or [term for term in base_terms if term not in _PLAN_LEMMAS]
    if not actual_terms:
        return None
    actual_frame = replace(frame, terms=actual_terms, qualifiers=set(), wants_value=True, plural=False)
    actual = _lookup(view, actual_frame, ranked_doc_ids)
    if plan is None or actual is None or not plan.value or not actual.value:
        return None
    plan_number = _to_number(plan.value.split(":", 1)[-1])
    actual_number = _to_number(actual.value.split(":", 1)[-1])
    if plan_number is None or actual_number is None or plan_number == 0 or plan_number == actual_number:
        return None
    percent = actual_number / plan_number * 100
    verdict = "план выполнен" if actual_number >= plan_number else "план не выполнен"
    unit = _value_unit(actual.value) or _value_unit(plan.value)
    unit_suffix = f" {unit}" if unit else ""
    delta = actual_number - plan_number
    sign = "+" if delta > 0 else "−"
    out = [
        f"**{_fmt_day(frame.start)}: {verdict} — {f'{percent:.1f}'.replace('.', ',')}%**",
        "",
        f"План: {_format_number(plan_number)}{unit_suffix}; факт: {_format_number(actual_number)}{unit_suffix} "
        f"(отклонение {sign}{_format_number(abs(delta))}{unit_suffix}).",
        "",
    ]
    for answer in (plan, actual):
        for item in answer.evidence[:1]:
            out.append(f"> {item['text']}")
    sources = list(dict.fromkeys(plan.sources + actual.sources))
    out.append("")
    out.append(_sources_line(sources))
    return FactAnswer(
        text="\n".join(out),
        kind="lookup",
        confidence=0.95,
        sources=sources,
        evidence=plan.evidence[:1] + actual.evidence[:1],
        value=f"{_format_number(actual_number)} / {_format_number(plan_number)}",
    )


def _corpus_wide_person(view: CorpusView, frame: QueryFrame) -> Optional[FactAnswer]:
    """«Кто составляет отчёты?» without a date: the person every matching line names."""
    if not frame.wants_person or frame.start or frame.has_entity or not frame.terms:
        return None
    counts: Counter = Counter()
    examples: Dict[str, Tuple[DocView, str]] = {}
    docs_per_person: Dict[str, set] = {}
    for doc in view.docs:
        for index, line in enumerate(doc.lines):
            coverage, matched = _term_score(frame, doc.lemma_lines()[index], doc.norm_lines[index])
            if matched == 0 or coverage < 0.5:
                continue
            for match in _PERSON_IN_TEXT.finditer(line):
                key, label = _person_key(match.group(0))
                counts[label] += 1
                examples.setdefault(label, (doc, line))
                docs_per_person.setdefault(label, set()).add(doc.doc_id)
    if not counts:
        return None
    (top, top_count), = counts.most_common(1)
    total = sum(counts.values())
    if len(docs_per_person[top]) < 2 or top_count / total < 0.6:
        return None
    doc, line = examples[top]
    n_docs = len(docs_per_person[top])
    out = [
        f"**{top}**",
        "",
        f"Упоминается так в {n_docs} {_plural(n_docs, 'документе', 'документах', 'документах')}, например «{doc.name}»:",
        f"> {line}",
        "",
        _sources_line([doc.name]),
    ]
    return FactAnswer(text="\n".join(out), kind="lookup", confidence=0.86, sources=[doc.name], value=top)


def _attribute_value(line: str, term: str) -> str:
    for key, value in _keyed_cells(line):
        key_lemmas = _lemmas(key)
        if term in key_lemmas or _stem_key(term) in {_stem_key(item) for item in key_lemmas}:
            return value.strip()
    stem = re.escape(term[:5] if len(term) > 5 else term)
    match = re.search(rf"(?i:(?<![а-яё]){stem}[а-яё]*)\s*[:\-—]?\s*([^,;|()]+)", line)
    if match:
        value = match.group(1).strip(" .")
        return value[:120]
    return ""


def _list_entity_values(view: CorpusView, frame: QueryFrame) -> Optional[FactAnswer]:
    """«Какие маршруты были у машины A123BC»: every value over all documents."""
    attributes = [term for term in frame.terms if term not in _VEHICLE_LEMMAS]
    if not attributes:
        return None
    attribute = attributes[0]
    values: Dict[str, Dict[str, Any]] = {}
    sources: List[str] = []
    total = 0
    for doc in view.docs:
        if frame.identifiers and not all(identifier in doc.identifiers for identifier in frame.identifiers):
            continue
        if frame.start and (doc.day is None or not _in_period(doc.day, frame)):
            continue
        for index in _matching_lines(doc, frame):
            value = _attribute_value(doc.lines[index], attribute)
            if not value or any(identifier == normalize_identifier(value) for identifier in frame.identifiers):
                continue
            key = _norm(value)
            bucket = values.setdefault(key, {"label": value, "count": 0, "days": []})
            bucket["count"] += 1
            if doc.day and doc.day not in bucket["days"]:
                bucket["days"].append(doc.day)
            total += 1
            if doc.name not in sources:
                sources.append(doc.name)
    if not values:
        return None
    entity = _entity_display(frame)
    period = f" {frame.period_label()}" if frame.start else ""
    n_values = len(values)
    out = [
        f"**{attribute.capitalize()} — «{entity}»{period}: {n_values} {_plural(n_values, 'вариант', 'варианта', 'вариантов')}, "
        f"{total} {_plural(total, 'упоминание', 'упоминания', 'упоминаний')}**",
        "",
    ]
    for bucket in sorted(values.values(), key=lambda item: (-item["count"], item["label"])):
        dates = ", ".join(_fmt_day(day) for day in sorted(bucket["days"])[:8])
        more = " …" if len(bucket["days"]) > 8 else ""
        count = bucket["count"]
        out.append(f"- {bucket['label']} — {count} {_plural(count, 'раз', 'раза', 'раз')}" + (f": {dates}{more}" if dates else ""))
    out.append("")
    out.append(f"Метод: просмотрены все строки документов с «{entity}», значение взято из поля «{attribute}».")
    out.append(_sources_line(sources))
    return FactAnswer(
        text="\n".join(out),
        kind="entity_values",
        confidence=0.85,
        sources=sources,
        value=str(n_values),
    )


def _to_number(raw: str) -> Optional[float]:
    match = _NUMBER.search(raw or "")
    if not match:
        return None
    try:
        return float(re.sub(r"[\s  ]", "", match.group(0)).replace(",", "."))
    except ValueError:
        return None


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}".replace(",", " ")
    return f"{value:,.2f}".replace(",", " ").replace(".", ",")


def _line_label(line: str) -> str:
    """The attribute caption of a line: its first words without numbers/punctuation.

    «Фактическая выручка за день составила 1 140 000 руб.» -> «фактическая выручка за день».
    """
    words = re.findall(r"[a-zа-яё]+", line.lower().replace("ё", "е"))
    return " ".join(words[:5])


def _same_attribute_rows(rows: list) -> list:
    """Keep only rows whose line is the same attribute in a series of reports.

    A period sum/maximum is meaningful over one repeated report line; lines
    that merely share a word with the question (one «размер» in a policy,
    another in a schedule) are not a series and must not be aggregated.
    """
    if len(rows) < 2:
        return rows
    labels = [set(_line_label(row[2]).split()) for row in rows]

    def similar(a: set, b: set) -> bool:
        # «отгружено фактически т продукции» ~ «дата показатель отгружено т значение»
        # (a table row with column captions): half of the shorter caption is shared.
        return bool(a and b) and len(a & b) / min(len(a), len(b)) >= 0.5

    groups = [[j for j, other in enumerate(labels) if similar(label, other)] for label in labels]
    best = max(groups, key=len)
    if len(best) < 2:
        return []
    return [rows[j] for j in best]


def _lookup_period(view: CorpusView, frame: QueryFrame) -> Optional[FactAnswer]:
    """Value of an attribute for every dated document of a period."""
    if not frame.terms or not frame.wants_value:
        return None
    best_per_day: Dict[date, Tuple[float, Tuple[date, DocView, str, str, Optional[float]]]] = {}
    for doc in view.docs:
        if doc.day is None or not _in_period(doc.day, frame):
            continue
        candidates: List[Tuple[float, int]] = []
        for index in _matching_lines(doc, frame):
            # A sum over the month must not mix in «плановая выручка» lines
            # (a plan document lists every day of the month too).
            if (doc.lemma_lines()[index] & _CONTRAST_QUALIFIERS) - frame.qualifiers:
                continue
            candidates.append((_score_line(frame, doc, index), index))
        # The best line that actually holds the asked value: a line that only
        # names the place (no quantity) must not hide the one with the number.
        best: Optional[Tuple[float, int]] = None
        value = ""
        for score, index in sorted(candidates, reverse=True):
            if score < 0.55:
                break
            value = _extract_value(frame, doc.lines[index])
            if value:
                best = (score, index)
                break
        if best is None:
            continue
        line = doc.lines[best[1]]
        row = (doc.day, doc, line, value, _to_number(value.split(":", 1)[-1]))
        previous = best_per_day.get(doc.day)
        if previous is None or best[0] > previous[0]:
            best_per_day[doc.day] = (best[0], row)
    rows = [row for _score, row in best_per_day.values()]
    rows = _same_attribute_rows(rows)
    if len(rows) < 2:
        return None
    rows.sort(key=lambda item: item[0])
    subject = frame.terms[0].capitalize()
    period = frame.period_label() if frame.start else "за весь период"
    numbers = [number for *_rest, number in rows if number is not None]
    numeric = bool(numbers) and len(numbers) == len(rows)
    unit = next((_value_unit(value) for _d, _doc, _l, value, _n in rows if _value_unit(value)), "")
    unit_suffix = f" {unit}" if unit else ""
    out: List[str] = []
    headline_value = str(len(rows))
    aggregate = frame.aggregate if numeric else ""
    if aggregate == "sum":
        total = sum(numbers)
        headline_value = _format_number(total)
        out.append(f"**{subject} {period}: всего {headline_value}{unit_suffix}**")
        out.append("")
        out.append(f"Сумма по {len(rows)} {_plural(len(rows), 'дню', 'дням', 'дням')}, за которые в документах есть значение.")
    elif aggregate == "avg":
        average = sum(numbers) / len(numbers)
        headline_value = _format_number(round(average))
        out.append(f"**{subject} {period}: в среднем {headline_value}{unit_suffix} за день**")
        out.append("")
        out.append(f"Среднее по {len(rows)} {_plural(len(rows), 'дню', 'дням', 'дням')} с данными (сумма за период — {_format_number(sum(numbers))}{unit_suffix})")
    elif aggregate in {"max", "min"}:
        pick = max if aggregate == "max" else min
        best = pick(rows, key=lambda item: item[4])
        label = "максимум" if aggregate == "max" else "минимум"
        headline_value = _format_number(best[4])
        out.append(f"**{subject} {period}: {label} {headline_value}{unit_suffix} — {_fmt_day(best[0])}**")
        out.append("")
        out.append(f"Источник: «{best[1].name}»")
        out.append(f"> {best[2]}")
    else:
        out.append(f"**{subject} {period}** — найдено значений: {len(rows)}")
    out.append("")
    out.append("Значения по дням:")
    for day, doc, _line, value, _number in rows:
        out.append(f"- {_fmt_day(day)}: {value}  _(«{doc.name}»)_")
    if numeric and not aggregate:
        out.append("")
        out.append(
            f"Сумма: {_format_number(sum(numbers))}{unit_suffix}; среднее за день: {_format_number(round(sum(numbers) / len(numbers)))}{unit_suffix}; "
            f"минимум: {_format_number(min(numbers))}; максимум: {_format_number(max(numbers))}"
        )
    out.append("")
    out.append("Метод: в каждом документе периода взята строка, лучше всего соответствующая вопросу; дни без такой строки не учитываются.")
    sources = list(dict.fromkeys(doc.name for _day, doc, *_ in rows))
    out.append(_sources_line(sources))
    return FactAnswer(
        text="\n".join(out),
        kind="period_aggregate" if aggregate else "period_values",
        confidence=0.88 if aggregate else 0.8,
        sources=sources,
        evidence=[{"date": day.isoformat(), "source": doc.name, "text": line} for day, doc, line, _v, _n in rows],
        value=headline_value,
    )


def answer_question(question: str, processor: Any, ranked_doc_ids: Sequence[str] = ()) -> Optional[FactAnswer]:
    """Return a verified deterministic answer, or ``None`` to defer to RAG."""
    try:
        view = corpus_view(processor)
        if view is None:
            return None
        frame = parse_question(question)
        align_year_to_corpus(frame, view)
        _restore_attribute_terms(frame, view)
        _apply_term_synonyms(frame, view)
        # «01.08.2025-15.08.2025» is a period (the date parser returns a range),
        # not two days to compare.
        is_range = bool(frame.start and frame.end and frame.start < frame.end)
        two_days = len(extract_day_mentions(question)) >= 2 and not is_range
        if not frame.has_entity and (_COMPARE_WORDS.search(_norm(question)) or two_days):
            frame.terms = [term for term in frame.terms if term not in {"сравнить", "разница", "отличаться", "изменить"}] or frame.terms
            compared = _compare_days(view, frame, ranked_doc_ids)
            if compared is not None:
                return compared
        if frame.frequency and not frame.has_entity:
            ranked = _rank_entities(view, frame)
            if ranked is not None:
                return ranked
        if frame.kind == "lookup" and frame.has_entity and not frame.single_day and re.match(r"\s*когда\b", _norm(question)):
            frame.kind = "list_dates"
        # «в какой день выручка была максимальной»: a maximum over the period,
        # not a list of days of an entity.
        if frame.aggregate in {"max", "min"} and not frame.has_entity and not frame.single_day and frame.kind in {"lookup", "list_dates"}:
            frame.kind = "lookup"
            frame.wants_value = True
            extreme = _lookup_period(view, frame)
            if extreme is not None:
                return extreme
        if frame.kind in {"count_days", "list_dates"}:
            return _aggregate_days(view, frame)
        if frame.has_entity and frame.plural and not frame.single_day:
            listed = _list_entity_values(view, frame)
            if listed is not None:
                return listed
        if not frame.has_entity and not frame.single_day and (frame.start or frame.aggregate in {"sum", "avg"}):
            if _STATISTICS_WORDS.search(question):
                return None  # event statistics belong to core.analytics
            return _lookup_period(view, frame)
        plan_check = _plan_vs_fact(view, frame, ranked_doc_ids)
        if plan_check is not None:
            return plan_check
        if frame.wants_person and not frame.start and not frame.has_entity:
            person = _corpus_wide_person(view, frame)
            if person is not None:
                return person
        return _lookup(view, frame, ranked_doc_ids)
    except Exception as exc:  # never break the chat because of the fast path
        logger.warning("Fact lookup failed: %s", exc)
        return None


def focused_context(question: str, processor: Any, ranked_doc_ids: Sequence[str], *, max_chars: int = 6000, max_docs: int = 4) -> str:
    """Context for the LLM made of the best-matching lines of the best documents.

    Unlike a document prefix, this keeps the lines that contain the question's
    words, dates, names and identifiers even when they sit at the end of a long
    document. Each line keeps its neighbours for local context.
    """
    try:
        view = corpus_view(processor)
        if view is None:
            return ""
        frame = parse_question(question)
        blocks: List[str] = []
        used = 0
        for doc_id in list(ranked_doc_ids)[: max_docs * 2]:
            doc = view.by_id.get(str(doc_id))
            if doc is None:
                continue
            indexes = _matching_lines(doc, frame) if frame.has_entity else list(range(len(doc.lines)))
            scored = sorted(((_score_line(frame, doc, i), i) for i in indexes), reverse=True)
            keep: set[int] = set()
            for score, index in scored[:6]:
                if score < 0.2 and keep:
                    break
                keep.update({index - 1, index, index + 1})
            keep = {i for i in keep if 0 <= i < len(doc.lines)}
            if not keep:
                continue
            header = f"📄 {doc.name}" + (f" (дата документа {_fmt_day(doc.day)})" if doc.day else "")
            body = "\n".join(doc.lines[i] for i in sorted(keep))
            block = f"{header}\n{body}"
            if used + len(block) > max_chars and blocks:
                break
            blocks.append(block[: max_chars - used])
            used += len(block)
            if len(blocks) >= max_docs:
                break
        return "\n\n".join(blocks)
    except Exception as exc:
        logger.warning("Focused context failed: %s", exc)
        return ""
