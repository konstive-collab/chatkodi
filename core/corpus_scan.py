"""Corpus scan: the lines of every document of a period that match a question.

For questions over many documents («сколько раз…», «в какие дни…», «как
менялось…», «всего за период…») retrieval of a few chunks cannot see every
day. The scan reads every document of the period, keeps the lines that contain
the question's content words, and answers counts and day lists itself; sums,
maxima and trends are left to the verified reasoned answer over the scan table.

Nothing here depends on a document format or a domain vocabulary: the terms
come from the question, dates from the documents' own text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple

from core.reasoned_answer import _NUMBER_WORDS, day_months

_MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
               "сентября", "октября", "ноября", "декабря"]
_STOP = {
    "сколько", "какие", "какой", "какая", "каких", "когда", "весь", "всего", "всех", "период", "периода",
    "были", "было", "был", "была", "быть", "раз", "дни", "день", "дней", "дням", "числа", "года", "году",
    "как", "менялось", "менялся", "изменялось", "изменение", "динамика", "больше", "меньше", "чем",
    "который", "которые", "это", "этот", "также", "или", "для", "при", "под", "над", "без", "после", "перед",
    "между", "сводке", "сводки", "отчету", "отчете", "документах", "документам", "данным", "показатель",
    "количество", "число", "сумма", "итого", "общее", "средний", "среднее", "максимальное", "минимальное",
    "видели", "было", "случаев", "упоминаний", "упоминается", "встречается",
}
_MONTH_STEMS = ("январ", "феврал", "март", "апрел", "мая", "май", "июн", "июл", "август", "сентябр",
                "октябр", "ноябр", "декабр")
PERIOD_QUESTION = re.compile(
    r"\bза\s+(?:весь\s+)?период\b|\bвсего\b|\bв\s+какие\s+дни\b|\bпо\s+каким\s+дням\b|\bсколько\s+раз\b"
    r"|\bкогда\b.*\b(?:больше|меньше)\s+всего\b|\bкак\s+(?:менял|изменял)\w*|\bдинамик\w*|\bпо\s+дням\b"
    r"|\bв\s+какой\s+день\b.*\b(?:больше|меньше|максим|миним)",
    re.IGNORECASE,
)
_COUNT_Q = re.compile(r"\bсколько\b", re.IGNORECASE)
_QUANTITY_Q = re.compile(
    r"\bсколько\s+(?:\w+\s+)?(?:человек|чел\b|людей|рубл|руб\b|денег|единиц|ед\b|тонн|км|кг|литр|процент|часов|метр)",
    re.IGNORECASE,
)
_TIMES_Q = re.compile(r"\bсколько\s+(?:\w+\s+)?(?:раз|случаев|эпизодов|упоминаний)\b", re.IGNORECASE)
_DAYS_Q = re.compile(r"\bв\s+какие\s+дни\b|\bпо\s+каким\s+дням\b|\bкогда\b(?!.*\b(?:больше|меньше)\b)", re.IGNORECASE)


@dataclass
class ScanRow:
    day: Optional[Tuple[int, int]]
    doc: str
    line: str


@dataclass
class ScanResult:
    terms: List[str]
    rows: List[ScanRow] = field(default_factory=list)
    documents: int = 0

    def table(self, limit: int = 300) -> str:
        out = []
        for row in self.rows[:limit]:
            label = _day_label(row.day)
            out.append(f"{label} ({row.doc}): {row.line}")
        return "\n".join(out)


def _day_label(day: Optional[Tuple[int, int]]) -> str:
    if not day:
        return "без даты"
    return f"{day[0]} {_MONTHS_GEN[day[1] - 1]}"


def _norm(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def is_series(documents: Iterable[Any], *, min_docs: int = 5) -> bool:
    """A series of periodic reports: many documents start with the same heading
    (numbers, dates and formatting aside)."""
    heads: dict = {}
    total = 0
    for doc in documents:
        content = doc.get("content") if isinstance(doc, dict) else getattr(doc, "content", "")
        first = next((line for line in str(content or "").splitlines() if line.strip()), "")
        key = re.sub(r"[^а-яёa-z]+", " ", _norm(first))
        key = " ".join(w for w in key.split() if not w.startswith(_MONTH_STEMS))
        total += 1
        if key:
            heads[key] = heads.get(key, 0) + 1
    best = max(heads.values()) if heads else 0
    return best >= max(min_docs, int(0.3 * total))


def name_terms(question: str) -> List[str]:
    """Names in the question (quoted, with digits, or Capitalized past the first word)."""
    out = [(_norm(a or b)).strip() for a, b in re.findall(r"«([^»]{2,40})»|\"([^\"]{2,40})\"", question)]
    words = re.findall(r"[A-Za-zА-ЯЁа-яё][\w-]*", re.sub(r"«[^»]*»|\"[^\"]*\"", " ", question))
    for index, word in enumerate(words):
        low = _norm(word)
        if low.startswith(_MONTH_STEMS) or low in _STOP:
            continue
        if re.search(r"\d", word) or (index > 0 and word[0].isupper()):
            out.append(low if re.search(r"\d|-", low) or len(low) <= 5 else low[: max(4, len(low) - 3)])
    return [term for term in out if term]


def question_terms(question: str) -> List[str]:
    """Content words and names of the question, as matching stems."""
    q = re.sub(r"\b\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\b", " ", question)
    names = re.findall(r"«([^»]{2,40})»|\"([^\"]{2,40})\"", q)
    terms: List[str] = []
    for pair in names:
        name = _norm(pair[0] or pair[1]).strip()
        if name:
            terms.append(name)
    q = re.sub(r"«[^»]*»|\"[^\"]*\"", " ", q)
    for token in re.findall(r"[A-Za-zА-ЯЁа-яё][\w-]*\d*|\d+[\w-]*", q):
        low = _norm(token)
        if low in _STOP or low.startswith(_MONTH_STEMS) or low.isdigit() or len(low) < 3:
            continue
        if re.fullmatch(r"\d{4}", low):
            continue
        stem = low if re.search(r"\d|-", low) or len(low) <= 5 else low[: max(4, len(low) - 3)]
        if stem not in terms:
            terms.append(stem)
    return terms


def _doc_day(content: str, name: str) -> Optional[Tuple[int, int]]:
    days = day_months(str(content or "")[:400] + " " + str(name or ""))
    return sorted(days)[0] if len(days) == 1 else (min(days) if days else None)


def _in_range(day: Optional[Tuple[int, int]], start: Optional[Tuple[int, int]], end: Optional[Tuple[int, int]]) -> bool:
    if start is None and end is None:
        return True
    if day is None:
        return False
    key = (day[1], day[0])
    return (start is None or key >= (start[1], start[0])) and (end is None or key <= (end[1], end[0]))


def scan(documents: Iterable[Any], question: str, start=None, end=None, *, max_rows: int = 400) -> ScanResult:
    terms = question_terms(question)
    lines: List[ScanRow] = []
    docs_in_range = 0
    for doc in documents:
        content = doc.get("content") if isinstance(doc, dict) else getattr(doc, "content", "")
        name = doc.get("name") if isinstance(doc, dict) else getattr(doc, "name", "")
        day = _doc_day(content, name)
        if not _in_range(day, start, end):
            continue
        docs_in_range += 1
        for raw in str(content or "").splitlines():
            line = raw.strip()
            if len(line) >= 4:
                lines.append(ScanRow(day, str(name or ""), line))
    if not terms or not lines:
        return ScanResult(terms=terms, documents=docs_in_range)
    norm_lines = [_norm(row.line) for row in lines]
    counts = {term: sum(1 for line in norm_lines if term in line) for term in terms}
    # A person/object named in the question that no document of the period
    # mentions: counting other lines would answer about something else.
    if any(not any(name in line for line in norm_lines) for name in name_terms(question)):
        return ScanResult(terms=terms, documents=docs_in_range)
    present = sorted((t for t in terms if counts[t] > 0), key=lambda t: counts[t])
    if not present:
        return ScanResult(terms=terms, documents=docs_in_range)
    # Greedy AND from the rarest term; a term that would empty the set is skipped.
    chosen: List[str] = []
    selected = list(range(len(lines)))
    for term in present:
        narrowed = [i for i in selected if term in norm_lines[i]]
        if narrowed:
            selected = narrowed
            chosen.append(term)
    rows = [lines[i] for i in selected][:max_rows]
    rows.sort(key=lambda r: ((r.day or (0, 0))[1], (r.day or (0, 0))[0], r.doc))
    return ScanResult(terms=chosen, rows=rows, documents=docs_in_range)


def _quantity_near(line: str, terms: Sequence[str]) -> Optional[int]:
    """«провел две контратаки» -> 2: the number written right before a term."""
    low = _norm(line)
    for term in terms:
        pos = low.find(term)
        if pos <= 0:
            continue
        before = re.findall(r"[0-9]+|[а-я]+", low[max(0, pos - 25):pos])
        for token in reversed(before[-2:]):
            if token.isdigit() and int(token) < 1000:
                return int(token)
            if token in _NUMBER_WORDS:
                return _NUMBER_WORDS[token]
    return None


def deterministic_answer(question: str, result: ScanResult) -> Optional[str]:
    """Counts and day lists computed here; None for questions the LLM must read."""
    if not result.rows:
        return None
    days = []
    for row in result.rows:
        label = _day_label(row.day)
        if label not in days:
            days.append(label)
    sources = ", ".join(dict.fromkeys(row.doc for row in result.rows))
    evidence = "\n".join(f"- {_day_label(r.day)} ({r.doc}): {r.line}" for r in result.rows[:60])
    more = f"\n… и ещё {len(result.rows) - 60}" if len(result.rows) > 60 else ""
    if _COUNT_Q.search(question) and not _QUANTITY_Q.search(question):
        quantities = [_quantity_near(row.line, result.terms) for row in result.rows]
        known = [q for q in quantities if q is not None]
        if len(known) >= max(1, len(result.rows) // 2):
            # Lines that state an amount carry the count; a line that only
            # mentions the subject (a conclusion) adds nothing.
            total = sum(known)
            how = "сумма количеств, указанных в строках документов"
        elif _TIMES_Q.search(question):
            total = len(result.rows)
            how = "число строк документов, где это упоминается"
        else:
            return None  # «сколько всего выручки» is a sum the LLM computes from the table
        return (
            f"**{total}** (за {len(days)} {_plural(len(days), 'день', 'дня', 'дней')}; {how}).\n\n"
            f"Найдено по всем документам периода ({result.documents}):\n{evidence}{more}\n\n📚 Источники: {sources}"
        )
    if _DAYS_Q.search(question):
        return (
            f"Дни: **{', '.join(days)}**.\n\nНайдено по всем документам периода ({result.documents}):\n"
            f"{evidence}{more}\n\n📚 Источники: {sources}"
        )
    return None


def _plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many
