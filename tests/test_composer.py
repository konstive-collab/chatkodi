"""Long documents and what the program computes in them: lesson minutes, timetables, duty rosters."""

from __future__ import annotations

import datetime as dt

from core.agent import composer


def test_kind_and_duration_are_read_from_the_request():
    assert composer.document_kind("Напиши методичку по адаптации новых сотрудников") == "методические рекомендации"
    assert composer.document_kind("Составь план-конспект урока физкультуры") == "план-конспект"
    assert composer.document_kind("Разработай инструкцию по приёмке товара") == "инструкция"
    assert composer.document_kind("Напиши письмо подрядчику") == "документ"
    assert composer.lesson_minutes("урок на 40 минут") == 40
    assert composer.lesson_minutes("занятие 1,5 часа") == 90
    assert composer.lesson_minutes("план-конспект урока") == 45


def test_lesson_minutes_add_up_to_the_duration():
    rows = [composer.LessonRow("вводная", "Построение", 5, ""), composer.LessonRow("основная", "Прыжки", 20, ""),
            composer.LessonRow("основная", "Игра", 15, ""), composer.LessonRow("заключительная", "Итоги", 5, "")]
    fitted, note = composer.fit_minutes(rows, 40)
    assert sum(r.minutes for r in fitted) == 40 and "45" in note
    assert [r.minutes for r in fitted if r.part != "основная"] == [5, 5]
    text = composer.lesson_markdown("План-конспект", ["Тема: прыжки"], fitted, 40, note)
    assert "**Итого:** 40 мин." in text and "| **2** | **Основная часть** | **30** |  |" in text


def test_timetable_has_no_clashes_and_spreads_subjects():
    lessons = [
        {"group": "7А", "subject": "Математика", "teacher": "Иванова", "per_week": 4},
        {"group": "7А", "subject": "Русский", "teacher": "Петрова", "per_week": 3},
        {"group": "7Б", "subject": "Математика", "teacher": "Иванова", "per_week": 4},
        {"group": "7Б", "subject": "История", "teacher": "Козлов", "per_week": 2},
    ]
    table = composer.build_timetable(["понедельник", "вторник", "среда", "четверг", "пятница"], ["1", "2", "3"], lessons)
    assert not table.unplaced and len(table.cells) == 13
    busy = {}
    for (group, day, slot), value in table.cells.items():
        if "Иванова" in value:
            assert (day, slot) not in busy, "a teacher in two classes at once"
            busy[(day, slot)] = group
    maths_days = [day for (group, day, _s), v in table.cells.items() if group == "7А" and v.startswith("Математика")]
    assert len(set(maths_days)) == 4  # four lessons on four different days
    text = composer.timetable_markdown("Расписание", table)
    assert "## 7А" in text and "| Понедельник |" in text


def test_roster_takes_turns_evenly_and_skips_weekends():
    roster = composer.build_roster(["Иванов", "Петров", "Сидоров"], dt.date(2026, 11, 2), dt.date(2026, 11, 15), 1, True)
    assert len(roster) == 10 and all(day.weekday() < 5 for day, _who in roster)
    counts = [sum(p in who for _d, who in roster) for p in ("Иванов", "Петров", "Сидоров")]
    assert max(counts) - min(counts) <= 1
    assert all(a[1] != b[1] for a, b in zip(roster, roster[1:]))
    assert composer.parse_date("с 01.11.26") == dt.date(2026, 11, 1)


def test_requests_are_routed_to_compose_and_schedule(tmp_path):
    from core.agent.agent import Agent
    from core.agent.attachments import AttachmentStore

    class _Manager:
        llm_model = None
        hybrid_search = None

    agent = Agent(_Manager(), store=AttachmentStore(tmp_path))
    route = {"action": "write", "source": "library", "output": "chat", "language": "", "task": "т", "search_query": ""}
    assert agent._reroute_by_request(dict(route), "Напиши методические рекомендации по наставничеству")["action"] == "compose"
    assert agent._reroute_by_request(dict(route), "Составь график дежурств на ноябрь")["action"] == "schedule"
    assert agent._reroute_by_request(dict(route), "Напиши письмо подрядчику")["action"] == "write"
    assert agent._reroute_by_request(dict(route, action="answer"), "Что сказано в инструкции о перчатках?")["action"] == "answer"


def test_tables_after_headings_get_sheets_named_after_them(tmp_path):
    from openpyxl import load_workbook

    from core.agent.outputs import markdown_to_xlsx

    text = "# Расписание\n\n## 7А\n\n| Урок | Пн |\n|---|---|\n| 1 | Математика |\n\n## 7Б\n\n| Урок | Пн |\n|---|---|\n| 1 | История |"
    assert load_workbook(markdown_to_xlsx(text, tmp_path / "r.xlsx")).sheetnames == ["7А", "7Б"]
