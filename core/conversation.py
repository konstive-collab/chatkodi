"""Conversation layer in front of RAG: small talk, typos and follow-up questions.

Testers talk to ChatKodi like to a person:

* «Привет! Что ты умеешь?» used to go through retrieval and the LLM for two
  minutes and end with «подтверждение не найдено»;
* «скока выручки было 4 авгста» found nothing because the month did not parse;
* «а 15-го?» after «Какая выручка 4 августа 2025?» was answered without the
  previous question.

Everything here is deterministic and runs in microseconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Optional

from core.identifiers import extract_normalized_identifiers

# --------------------------------------------------------------------------- #
# small talk
# --------------------------------------------------------------------------- #

_GREETING = re.compile(
    r"^\s*(?:привет\w*|здравствуй\w*|добр(?:ый|ое|ого)\s+(?:день|вечер|утро|времени)|доброе\s+утро|хай|салют|hello|hi|hey)\b",
    flags=re.IGNORECASE,
)
_CAPABILITIES = re.compile(
    r"\b(?:что\s+(?:ты\s+)?(?:умеешь|можешь)|кто\s+ты|что\s+ты\s+такое|как\s+(?:тобой|с\s+тобой)\s+(?:пользоваться|работать)|"
    r"как\s+ты\s+работаешь|помоги\s+разобраться|что\s+ты\s+знаешь|help|помощь|справк\w*\s+по\s+программ\w*)\b",
    flags=re.IGNORECASE,
)
_THANKS = re.compile(r"^\s*(?:спасибо|благодар\w*|спс|сенкс|thanks|thank\s+you|отлично|супер|класс)\b[\s!.)]*$", flags=re.IGNORECASE)
_BYE = re.compile(r"^\s*(?:пока|до\s+свидания|увидимся|bye)\b[\s!.)]*$", flags=re.IGNORECASE)


def small_talk_answer(question: str, *, document_count: int = 0) -> Optional[str]:
    """Answer greetings and «what can you do» without search or LLM."""
    text = str(question or "").strip()
    if not text or len(text) > 160:
        return None
    if _THANKS.match(text):
        return "Пожалуйста! Спрашивайте ещё — я отвечу по вашим документам и покажу, откуда взят ответ."
    if _BYE.match(text):
        return "До встречи! Индекс документов сохранён, в следующий раз начнём сразу с вопросов."
    greeting = bool(_GREETING.match(text))
    capabilities = bool(_CAPABILITIES.search(text))
    if not (greeting or capabilities):
        return None
    # «Привет, какая выручка 4 августа?» is a real question with a greeting.
    rest = _GREETING.sub("", text).strip(" ,!.?-—")
    if greeting and not capabilities and len(rest.split()) > 2:
        return None
    # «нужна помощь по отчёту за август», «что ты знаешь о выручке за август
    # 2025?» name a subject after the capability phrase: that is a search.
    leftover = _CAPABILITIES.sub(" ", rest)
    if capabilities and len(re.findall(r"[а-яёa-z0-9]{3,}", leftover.lower())) >= 2:
        return None
    docs = (
        f"Сейчас в базе {document_count} {_plural(document_count, 'документ', 'документа', 'документов')}."
        if document_count
        else "Добавьте документы через меню «Документы» — и можно спрашивать."
    )
    hello = "Здравствуйте! " if greeting else ""
    return (
        f"{hello}Я ChatKodi — локальный ассистент по вашим документам, работаю без интернета: "
        f"данные никуда не передаются. {docs}\n\n"
        "**Что я умею:**\n"
        "- отвечать на вопросы по документам с цитатой и источником, считать за период, сравнивать даты, "
        "отвечать на уточнения вроде «а 15-го?»;\n"
        "- принимать файлы, фото и сканы (📎, перетаскивание, Ctrl+V): распознаю текст и таблицы, сохраню в Word или Excel;\n"
        "- проверять данные по вашим нормативам и составлять документы по образцам из библиотеки;\n"
        "- сравнивать редакции документов, пересказывать, переводить, писать письма и служебные записки;\n"
        "- слушать голосом (🎤) и отвечать голосом (🔊);\n"
        "- находить файлы на компьютере, прикладывать их к чату, переименовывать и раскладывать по папкам (с отменой);\n"
        "- готовить черновики писем в Outlook с вложениями, открывать файлы и программы, выполнять команды — "
        "без вопросов только то, что разрешено в настройках;\n"
        "- если задач несколько — составлю план и выполню по шагам; запоминаю, кто вы, между чатами.\n\n"
        "К каждому ответу по документам прикладываю источник — его можно открыть кнопкой «Источники ответа»."
    )


def _plural(count: int, one: str, few: str, many: str) -> str:
    tail = count % 100
    if 11 <= tail <= 14:
        return many
    if count % 10 == 1:
        return one
    if count % 10 in {2, 3, 4}:
        return few
    return many


# --------------------------------------------------------------------------- #
# typo-tolerant normalization
# --------------------------------------------------------------------------- #

_COLLOQUIAL = {
    "скока": "сколько", "скоко": "сколько", "сколко": "сколько", "скольк": "сколько", "сколька": "сколько",
    "чо": "что", "че": "что", "шо": "что", "чё": "что",
    "када": "когда", "кагда": "когда", "щас": "сейчас",
    "выручкa": "выручка", "выручьки": "выручки",
}
_MONTH_FORMS = [
    f"{stem}{ending}"
    for stem, endings in (
        ("январ", ("ь", "я", "е")), ("феврал", ("ь", "я", "е")), ("март", ("", "а", "е")),
        ("апрел", ("ь", "я", "е")), ("ма", ("й", "я", "е")), ("июн", ("ь", "я", "е")),
        ("июл", ("ь", "я", "е")), ("август", ("", "а", "е")), ("сентябр", ("ь", "я", "е")),
        ("октябр", ("ь", "я", "е")), ("ноябр", ("ь", "я", "е")), ("декабр", ("ь", "я", "е")),
    )
    for ending in endings
]
_MONTH_SET = set(_MONTH_FORMS)
# Real words one letter away from a month form: «марка» -> «марта» turned
# «Какая марка у автомобиля…» into a date question.
_NOT_MONTH_PREFIXES = ("марк", "маяк", "матк", "майн", "майя")


def normalize_query(question: str) -> str:
    """Fix colloquial spellings and misspelled month names, keep everything else."""

    def fix(match: re.Match[str]) -> str:
        word = match.group(0)
        low = word.lower().replace("ё", "е") if word.lower() != "чё" else "чё"
        if low in _COLLOQUIAL:
            replacement = _COLLOQUIAL[low]
            return replacement.capitalize() if word[:1].isupper() else replacement
        if (
            len(low) >= 4
            and low not in _MONTH_SET
            and not low.startswith(_NOT_MONTH_PREFIXES)
            and re.fullmatch(r"[а-я]+", low)
        ):
            close = get_close_matches(low, _MONTH_FORMS, n=1, cutoff=0.8)
            if close and close[0][0] == low[0]:
                return close[0]
        return word

    return re.sub(r"[А-Яа-яЁё]+", fix, str(question or ""))


# --------------------------------------------------------------------------- #
# follow-up questions
# --------------------------------------------------------------------------- #

_FOLLOWUP_START = re.compile(r"^\s*(?:а|и|ну\s+а|а\s+что|а\s+если|а\s+как|тогда|также|то\s+же\s+самое)\b", flags=re.IGNORECASE)
_MONTH_WORD = re.compile(
    r"\b(?:январ|феврал|март|апрел|ма[йяе]|июн|июл|август|сентябр|октябр|ноябр|декабр)[а-яё]*\b",
    flags=re.IGNORECASE,
)
_FULL_DATE = re.compile(
    r"\b\d{1,2}[./]\d{1,2}[./](?:\d{4}|\d{2})\b|\b\d{1,2}(?:-?го)?\s+(?:январ|феврал|март|апрел|ма[йяе]|июн|июл|август|сентябр|октябр|ноябр|декабр)[а-яё]*(?:\s+\d{4}(?:\s*(?:г\.|года?))?)?",
    flags=re.IGNORECASE,
)
_DAY_ONLY = re.compile(r"\b(\d{1,2})(?:-?(?:го|е|ое|ого))?(?:\s+числа)?\b", flags=re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_PERSON = re.compile(
    r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.?|\b[А-ЯЁ]\.\s?[А-ЯЁ]\.\s?[А-ЯЁ][а-яё]+"
)
_SURNAME_ALONE = re.compile(r"(?<=\bу\s)[А-ЯЁ][а-яё]{2,}|(?<=\bдля\s)[А-ЯЁ][а-яё]{2,}|(?<=\bпро\s)[А-ЯЁ][а-яё]{2,}")
_QUESTION_START = re.compile(
    r"^\s*(?:кто|что|какой|какая|какое|какие|каков\w*|сколько|когда|где|куда|почему|зачем|как|чей|чья|найди|покажи|перечисли|сравни|расскажи|перескажи|опиши)\b",
    flags=re.IGNORECASE,
)


_FOLLOWUP_FILLER = {"числа", "год", "года", "году", "для", "про", "тоже", "еще", "ещё", "так", "тогда", "это", "этот", "эта"}


def _plate_spans(text: str) -> list[str]:
    spans = []
    for match in re.finditer(r"[A-Za-zА-ЯЁа-яё]\s?\d{3}\s?[A-Za-zА-ЯЁа-яё]{2}\s?\d{2,3}", text):
        if extract_normalized_identifiers(match.group(0)):
            spans.append(match.group(0))
    return spans


@dataclass
class FollowUp:
    question: str
    resolved: bool = False
    note: str = ""


def resolve_followup(question: str, previous: Optional[str]) -> FollowUp:
    """Turn «а 15-го?» into a full question using the previous one."""
    text = str(question or "").strip()
    if not previous or not text:
        return FollowUp(text)
    words = re.findall(r"[\wА-Яа-яЁё.-]+", text)
    starts_like_followup = bool(_FOLLOWUP_START.match(text))
    short = len(words) <= 5
    if not starts_like_followup:
        # Without «а/и …» only a bare date/name/plate counts as a follow-up:
        # «выручка 04.08.25» is a complete keyword query of its own.
        if not short or _QUESTION_START.match(text):
            return FollowUp(text)
        stripped = _FULL_DATE.sub(" ", text)
        stripped = _PERSON.sub(" ", stripped)
        for plate in _plate_spans(stripped):
            stripped = stripped.replace(plate, " ")
        content = [
            word for word in re.findall(r"[а-яёa-z]{3,}", stripped.lower())
            if word not in _FOLLOWUP_FILLER and not _MONTH_WORD.fullmatch(word)
        ]
        if content:
            return FollowUp(text)

    body = _FOLLOWUP_START.sub("", text).strip(" ,?!.—-")
    base = str(previous).strip()
    result = base
    changed = False

    new_full = _FULL_DATE.search(body)
    old_full = _FULL_DATE.search(base)
    new_month = _MONTH_WORD.search(body)
    if new_full and old_full:
        replacement = new_full.group(0)
        if not _YEAR.search(replacement):
            old_year = _YEAR.search(old_full.group(0))
            if old_year and not re.match(r"\d{1,2}[./]", replacement):
                replacement = f"{replacement} {old_year.group(0)}"
        result = base[: old_full.start()] + replacement + base[old_full.end():]
        changed = True
    elif new_month and not _DAY_ONLY.search(body.replace(new_month.group(0), "")):
        old_month = _MONTH_WORD.search(base)
        if old_month:
            result = base[: old_month.start()] + new_month.group(0).lower() + base[old_month.end():]
            new_year = _YEAR.search(body)
            if new_year:
                result = _YEAR.sub(new_year.group(0), result, count=1)
            changed = True
    else:
        day = _DAY_ONLY.search(body)
        if day and old_full and not _YEAR.fullmatch(day.group(1)):
            old_day = re.match(r"\d{1,2}", old_full.group(0))
            if old_day:
                start = old_full.start() + old_day.start()
                result = base[:start] + day.group(1) + base[start + len(old_day.group(0)):]
                changed = True

    new_person = _PERSON.search(text) or _SURNAME_ALONE.search(text)
    old_person = _PERSON.search(base)
    if new_person and old_person:
        result = result.replace(old_person.group(0), new_person.group(0), 1)
        changed = True
    new_plates = _plate_spans(body)
    old_plates = _plate_spans(base)
    if new_plates and old_plates:
        result = result.replace(old_plates[0], new_plates[0], 1)
        changed = True

    if not changed:
        if not starts_like_followup or len(words) > 6:
            return FollowUp(text)
        # «а план?» — keep the previous question and add what changed.
        result = f"{base.rstrip(' ?')} — {body}?"
    if result.strip() == base.strip():
        return FollowUp(text)
    return FollowUp(result, resolved=True, note=f"_Понял вопрос как: «{result}»_\n\n")
