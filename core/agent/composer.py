"""Long documents written from scratch: methodical recommendations, instructions,
regulations, programmes, plans — and the parts of them a program does better.

A small model cannot write a good twenty-page document in one call. Here the
model first plans the sections (after the example of the same kind, when the
library has one), then writes them one by one, each with the library passages
that concern it. Lesson plans are a table whose minutes the program adds up and
fits to the duration asked for. Timetables and duty rosters are laid out by the
program from the conditions the model reads out of the request.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "sections": {
            "type": "array", "minItems": 3, "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {"heading": {"type": "string"}, "points": {"type": "array", "maxItems": 8, "items": {"type": "string"}}},
                "required": ["heading", "points"],
            },
        },
    },
    "required": ["title", "sections"],
}

OUTLINE_PROMPT = """Составь план документа по задаче пользователя. Вид документа: {kind}.
Задача: {task}
Условия пользователя: {message}
{example}
Верни название документа и 4–9 разделов; для каждого раздела — 2–6 пунктов: что именно в нём раскрыть
(конкретно по теме, не общие слова). Не делай разделов о поиске, источниках и о самом ассистенте.
{library}"""

OUTLINE_EXAMPLE = """В библиотеке есть ОБРАЗЕЦ документа этого вида — повтори его разделы и их порядок (заголовки того же смысла),
содержание бери по новой теме:
{example}
"""

SECTION_PROMPT = """Ты пишешь раздел документа «{title}» ({kind}). Условия пользователя: {message}

План всего документа:
{outline}

Напиши ТОЛЬКО раздел «{heading}». Раскрой пункты: {points}.
Пиши деловым русским языком, связными абзацами; списки — только для перечня действий, условий, критериев.
Конкретно: кто, что, когда, как делает, как проверяется. Не повторяй другие разделы, не пиши вступлений
и выводов ко всему документу. Факты, нормы, числа, даты и названия документов — только из МАТЕРИАЛОВ и условий
пользователя; своё предлагай словами «рекомендуется», «следует», без выдуманных чисел и ссылок.
Начни прямо с текста раздела, без заголовка.

МАТЕРИАЛЫ:
{materials}
"""

LESSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "header": {"type": "array", "maxItems": 10, "items": {"type": "string"}},
        "rows": {
            "type": "array", "minItems": 4, "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "part": {"type": "string", "enum": ["вводная", "основная", "заключительная"]},
                    "content": {"type": "string"},
                    "minutes": {"type": "number"},
                    "notes": {"type": "string"},
                },
                "required": ["part", "content", "minutes", "notes"],
            },
        },
    },
    "required": ["title", "header", "rows"],
}

LESSON_PROMPT = """Составь план-конспект занятия (урока, тренировки) по задаче.
Задача: {task}
Условия пользователя: {message}
Длительность занятия: {minutes} мин.

title — «План-конспект занятия по теме: …»;
header — строки вводных данных: «Тема: …», «Цель: …», «Задачи: …», «Группа (класс): …», «Место: …»,
«Оборудование: …» — только то, что известно из условий или материалов, остальное поле в квадратных скобках;
rows — ход занятия по порядку: вводная, основная и заключительная части; content — что делают (конкретно по теме),
minutes — время на пункт, notes — организационно-методические указания (как проводить, на что обратить внимание).
Время всех пунктов в сумме — {minutes} мин.
{library}"""

PARTS = {"вводная": "Вводная (подготовительная) часть", "основная": "Основная часть", "заключительная": "Заключительная часть"}

KIND_WORDS = [
    ("план-конспект", r"план[\s-]*конспект|конспект\s+(?:урока|занятия)|план\s+(?:урока|занятия|тренировк\w*|проведения\s+занятия)"),
    ("методические рекомендации", r"методическ\w+\s+(?:рекомендац|указан|пособ|разработк)|методичк"),
    ("инструкция", r"инструкци"),
    ("положение", r"положени[ея]\s+(?:о|об)\b"),
    ("регламент", r"регламент"),
    ("программа", r"программ\w*\s+(?:обучени|курса|занятий|подготовк|стажировк|мероприят)|учебн\w+\s+программ"),
    ("план мероприятий", r"план\w*\s+(?:мероприяти|работ\w*\s+на|действий|подготовки)"),
]


def document_kind(text: str) -> str:
    low = (text or "").lower().replace("ё", "е")
    for kind, pattern in KIND_WORDS:
        if re.search(pattern, low):
            return kind
    return "документ"


def lesson_minutes(text: str) -> int:
    """«на 45 минут», «урок 40 мин», «пара 90 минут», «2 часа», «академический час» → minutes."""
    low = (text or "").lower().replace("ё", "е")
    m = re.search(r"(\d{1,3})\s*(?:мин|минут)", low)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{1,2}(?:[.,]5)?)\s*(?:час|ч\b)", low)
    if m:
        return int(round(float(m.group(1).replace(",", ".")) * 60))
    if re.search(r"академическ\w+\s+час", low):
        return 45
    if re.search(r"\bпар[аыу]\b", low):
        return 90
    if re.search(r"\bурок", low):
        return 45
    return 45


@dataclass
class LessonRow:
    part: str
    content: str
    minutes: int
    notes: str


def fit_minutes(rows: List[LessonRow], total: int) -> Tuple[List[LessonRow], Optional[str]]:
    """Minutes add up to the duration: the main part takes up the difference
    (the model's sums are rarely right). A note says what was changed."""
    got = sum(r.minutes for r in rows)
    if got == total or not rows:
        return rows, None
    main = [r for r in rows if r.part == "основная"] or rows
    other = got - sum(r.minutes for r in main)
    room = total - other
    if room < len(main):
        # the model gave the warm-up and the end more than the whole lesson: scale everything
        scale = total / max(1, got)
        for r in rows:
            r.minutes = max(1, int(round(r.minutes * scale)))
        main = rows
        room = total - sum(r.minutes for r in rows if r not in main)
    weights = [max(1, r.minutes) for r in main]
    shares = [room * w / sum(weights) for w in weights]
    whole = [max(1, int(s)) for s in shares]
    # largest remainders get the leftover minutes so the sum is exact
    rest = room - sum(whole)
    order = sorted(range(len(main)), key=lambda i: shares[i] - int(shares[i]), reverse=True)
    for i in order[:max(0, rest)]:
        whole[i] += 1
    for r, m in zip(main, whole):
        r.minutes = m
    return rows, f"Время пунктов в сумме было {got} мин — распределено программой на {total} мин."


def lesson_markdown(title: str, header: Sequence[str], rows: List[LessonRow], total: int, note: Optional[str]) -> str:
    lines = [f"# {title.strip() or 'План-конспект занятия'}", ""]
    lines += [h.strip() for h in header if h.strip()] + [f"Продолжительность: {total} мин.", ""]
    lines += ["| № | Содержание | Время, мин | Организационно-методические указания |", "|---|---|---|---|"]
    number = 0
    for part in PARTS:
        part_rows = [r for r in rows if r.part == part]
        if not part_rows:
            continue
        number += 1
        lines.append(f"| **{number}** | **{PARTS[part]}** | **{sum(r.minutes for r in part_rows)}** |  |")
        for k, r in enumerate(part_rows, 1):
            cell = lambda s: re.sub(r"\s+", " ", s or "").replace("|", "/").strip()
            lines.append(f"| {number}.{k} | {cell(r.content)} | {r.minutes} | {cell(r.notes)} |")
    lines += ["", f"**Итого:** {sum(r.minutes for r in rows)} мин."]
    if note:
        lines += ["", f"_{note}_"]
    return "\n".join(lines)


def lesson_rows(payload: dict) -> List[LessonRow]:
    rows = []
    for raw in payload.get("rows") or []:
        if not isinstance(raw, dict) or not str(raw.get("content") or "").strip():
            continue
        part = raw.get("part") if raw.get("part") in PARTS else "основная"
        try:
            minutes = max(1, int(round(float(raw.get("minutes") or 1))))
        except (TypeError, ValueError):
            minutes = 1
        rows.append(LessonRow(part, str(raw["content"]).strip(), minutes, str(raw.get("notes") or "").strip()))
    order = list(PARTS)
    return sorted(rows, key=lambda r: order.index(r.part))  # stable: the model's order inside a part


# ------------------------------------------------------------ passages for a section

def _words(text: str) -> set:
    return {w[:6] for w in re.findall(r"[а-яa-z]{4,}", (text or "").lower().replace("ё", "е"))}


def passages_for(query: str, texts: Sequence[Tuple[str, str]], limit: int = 5000) -> str:
    """Paragraphs of the library texts that share the most words with a section's heading and points."""
    want = _words(query)
    scored = []
    for name, text in texts:
        for para in re.split(r"\n\s*\n|\n(?=\S)", text or ""):
            para = para.strip()
            if len(para) < 40:
                continue
            score = len(want & _words(para))
            if score:
                scored.append((score, name, para))
    scored.sort(key=lambda s: -s[0])
    out, used = [], 0
    for _score, name, para in scored:
        block = f"[{name}] {para[:1500]}"
        if used + len(block) > limit:
            break
        out.append(block)
        used += len(block)
    return "\n\n".join(out)


# ------------------------------------------------------------ timetables and rosters

SCHEDULE_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["занятия", "дежурства"]},
        "title": {"type": "string"},
        "days": {"type": "array", "items": {"type": "string"}},
        "slots": {"type": "array", "items": {"type": "string"}},
        "lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "group": {"type": "string"}, "subject": {"type": "string"},
                    "teacher": {"type": "string"}, "per_week": {"type": "integer"},
                },
                "required": ["group", "subject", "teacher", "per_week"],
            },
        },
        "people": {"type": "array", "items": {"type": "string"}},
        "date_from": {"type": "string"},
        "date_to": {"type": "string"},
        "per_day": {"type": "integer"},
        "skip_weekends": {"type": "boolean"},
    },
    "required": ["kind", "title", "days", "slots", "lessons", "people", "date_from", "date_to", "per_day", "skip_weekends"],
}

SCHEDULE_PROMPT = """Выпиши из задачи условия расписания — ничего не раскладывай сам, это сделает программа.
Задача: {message}
{data}
kind — «занятия» (расписание уроков, пар, тренировок по группам) или «дежурства» (график дежурств, смен,
очерёдность людей по датам).
Для занятий: days — дни недели по порядку («понедельник» …), slots — номера или время занятий в день
(«1 пара 09:00–10:30» или «1», «2», …), lessons — для каждой группы и предмета: group, subject, teacher
(или пустая строка), per_week — сколько раз в неделю. people, date_from, date_to — пусто, per_day — 0.
Для дежурств: people — все люди по списку, date_from и date_to — даты в виде ДД.ММ.ГГГГ, per_day — сколько
человек в день, skip_weekends — true, если выходные не дежурят. days, slots, lessons — пусто.
Если чего-то нет в задаче — пусто или 0, не выдумывай. Сегодня {today}.
"""

WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]


@dataclass
class Timetable:
    days: List[str]
    slots: List[str]
    groups: List[str]
    cells: Dict[Tuple[str, int, int], str] = field(default_factory=dict)  # (group, day, slot) -> «предмет (учитель)»
    unplaced: List[str] = field(default_factory=list)


def build_timetable(days: List[str], slots: List[str], lessons: List[dict]) -> Timetable:
    """Lessons spread over the week: a group has one lesson at a time, a teacher is in
    one place at a time, a subject is spread over different days, days are filled evenly."""
    days = [d for d in days if d.strip()] or WEEKDAYS[:5]
    slots = [s for s in slots if s.strip()] or [str(i) for i in range(1, 5)]
    groups = list(dict.fromkeys(str(l.get("group") or "").strip() or "Группа" for l in lessons))
    table = Timetable(days, slots, groups)
    teacher_busy: Dict[Tuple[str, int, int], str] = {}
    items = []
    for lesson in lessons:
        count = max(0, int(lesson.get("per_week") or 0))
        group = str(lesson.get("group") or "").strip() or "Группа"
        items += [(group, str(lesson.get("subject") or "").strip(), str(lesson.get("teacher") or "").strip())] * count
    # the most frequent subjects first: they are the hardest to spread
    weight = {}
    for item in items:
        weight[item] = weight.get(item, 0) + 1
    items.sort(key=lambda it: -weight[it])
    for group, subject, teacher in items:
        best = None
        for d in range(len(days)):
            same_day = sum(1 for (g, dd, _s), v in table.cells.items() if g == group and dd == d and v.startswith(subject))
            load = sum(1 for (g, dd, _s) in table.cells if g == group and dd == d)
            for s in range(len(slots)):
                if (group, d, s) in table.cells or (teacher and (teacher, d, s) in teacher_busy):
                    continue
                # fewer repeats of the subject that day, a lighter day, an earlier slot
                key = (same_day, load, s)
                if best is None or key < best[0]:
                    best = (key, d, s)
                break  # the first free slot of a day keeps lessons without gaps
        if best is None:
            table.unplaced.append(f"{group}: {subject}")
            continue
        _key, d, s = best
        table.cells[(group, d, s)] = subject + (f" ({teacher})" if teacher else "")
        if teacher:
            teacher_busy[(teacher, d, s)] = group
    return table


def timetable_markdown(title: str, table: Timetable) -> str:
    lines = [f"# {title.strip() or 'Расписание занятий'}", ""]
    for group in table.groups:
        if len(table.groups) > 1:
            lines += [f"## {group}", ""]
        lines.append("| Занятие | " + " | ".join(d.capitalize() for d in table.days) + " |")
        lines.append("|---|" + "---|" * len(table.days))
        for s, slot in enumerate(table.slots):
            lines.append(f"| {slot} | " + " | ".join(table.cells.get((group, d, s), "") for d in range(len(table.days))) + " |")
        lines.append("")
    if table.unplaced:
        lines.append("⚠ Не поместились (мало дней или занятий в день): " + ", ".join(table.unplaced) + ".")
    return "\n".join(lines).rstrip()


def parse_date(text: str) -> Optional[_dt.date]:
    m = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", text or "")
    if not m:
        return None
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    year += 2000 if year < 100 else 0
    try:
        return _dt.date(year, month, day)
    except ValueError:
        return None


def build_roster(people: List[str], start: _dt.date, end: _dt.date, per_day: int = 1, skip_weekends: bool = False) -> List[Tuple[_dt.date, List[str]]]:
    """People in turn over the dates: everyone the same number of times (±1), nobody twice in a row when avoidable."""
    people = [p.strip() for p in people if p.strip()]
    per_day = max(1, min(per_day or 1, len(people) or 1))
    out = []
    turn = 0
    day = start
    while day <= end and len(out) < 400:
        if not (skip_weekends and day.weekday() >= 5):
            out.append((day, [people[(turn + k) % len(people)] for k in range(per_day)]))
            turn += per_day
        day += _dt.timedelta(days=1)
    return out


def roster_markdown(title: str, roster: List[Tuple[_dt.date, List[str]]], people: List[str]) -> str:
    lines = [f"# {title.strip() or 'График дежурств'}", "", "| Дата | День недели | Дежурный |", "|---|---|---|"]
    for day, who in roster:
        lines.append(f"| {day:%d.%m.%Y} | {WEEKDAYS[day.weekday()]} | {', '.join(who)} |")
    counts = {p: sum(p in who for _d, who in roster) for p in people if p.strip()}
    lines += ["", "Число дежурств: " + ", ".join(f"{p} — {n}" for p, n in counts.items()) + "."]
    return "\n".join(lines)
