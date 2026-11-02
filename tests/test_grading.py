"""Grades by a scale, honest partial answers, plans without repeats, Excel with formulas.

Two sheets with answer keys: the grade sheet of the vision corpus generator
(tools/vision_corpus.py — one scale for the mean of the scores) and the
fitness sheet of a user test (tools/fixtures/vedomost_fizra.jpg — a scale per
exercise, the grade is the rounded mean of the exercise grades). Here they
are the text the recognition returns; the photos run end to end in the eval.
"""

from __future__ import annotations

import pytest

from core.agent import grading
from core.agent.agent import Agent, AgentResult, _demands
from core.agent.attachments import Attachment, AttachmentStore
from core.agent.table_check import parse_markdown_tables
from tools.vision_corpus import GRADE_SHEET_KEY, GRADE_SHEET_ROWS, GRADE_SHEET_RULE


def sheet_text() -> str:
    lines = ["Ведомость, группа ТМ-21, «Материаловедение»", "",
             "| № | Студент | К1 | К2 | К3 | Ср. балл | Оценка |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {i} | " + " | ".join(row) + " |" for i, row in enumerate(GRADE_SHEET_ROWS, 1)]
    lines += ["", "Оценка по среднему баллу:", "85-100 — «5», 70-84 — «4»,", "50-69 — «3», менее 50 — «2».",
              GRADE_SHEET_RULE, "Преподаватель: Смирнова О. В."]
    return "\n".join(lines)


PAYLOAD = {
    "scales": [{"applies_to": "итог",
                "quote": "Оценка по среднему баллу: 85-100 — «5», 70-84 — «4», 50-69 — «3», менее 50 — «2».",
                "bands": [{"grade": "5", "min": "85", "max": "100"}, {"grade": "4", "min": "70", "max": "84"},
                          {"grade": "3", "min": "50", "max": "69"}, {"grade": "2", "min": "", "max": "50"}]}],
    "rounding": "math", "decimals": 0, "score_columns": ["К1", "К2", "К3"], "aggregate": "mean",
    "rule_quote": GRADE_SHEET_RULE, "result_column": "Ср. балл", "grade_column": "Оценка", "label_column": "Студент",
}

# the user's fitness sheet: three exercise scales on the sheet, the grade is their rounded mean
FIZRA_TEXT = """Ведомость сдачи нормативов — 9 «Б» класс, физкультура
Оценки за упражнения:
Бег 60 м: до 8,5 с — «5»; 8,6–9,0 — «4»; 9,1–9,6 — «3»; больше 9,6 — «2»
Подтягивания: 10 и более — «5»; 7–9 — «4»; 4–6 — «3»; меньше 4 — «2»
Прыжок в длину: 220 см и более — «5»; 200–219 — «4»; 180–199 — «3»; меньше 180 — «2»
Итоговая оценка = среднее трёх оценок, округляем до целого (0,5 — вверх).

| № | Фамилия | Бег 60 м, с | Подтяг., раз | Прыжок, см | Оценка |
|---|---|---|---|---|---|
| 1 | Иванов | 8,4 | 11 | 225 | 5 |
| 2 | Петрова | 8,9 | 8 | 205 | |
| 3 | Сидоров | 9,3 | 5 | 190 | 4 |
| 4 | Кузнецова | 8,6 | 10 | 215 | 4 |
| 5 | Морозов | 9,8 | 3 | 170 | |
| 6 | Лебедева | 8,5 | 7 | 200 | 5 |
| 7 | Новиков | 9,0 | 6 | | 3 |
| 8 | Орлова | 8,3 | 9 | 230 | |"""

# 2_КЛЮЧ_правильных_ответов.txt: name -> (right grade, what to do)
FIZRA_KEY = {
    "Иванов": ("5", "ok"), "Петрова": ("4", "filled"), "Сидоров": ("3", "fixed"), "Кузнецова": ("4", "ok"),
    "Морозов": ("2", "filled"), "Лебедева": ("4", "fixed"), "Новиков": (None, "missing"), "Орлова": ("5", "filled"),
}


def _bands(*bands):
    return [{"grade": g, "min": lo, "max": hi} for g, lo, hi in bands]


FIZRA_PAYLOAD = {
    "scales": [
        {"applies_to": "Бег 60 м, с", "quote": "Бег 60 м: до 8,5 с — «5»; 8,6–9,0 — «4»; 9,1–9,6 — «3»; больше 9,6 — «2»",
         "bands": _bands(("5", "", "8,5"), ("4", "8,6", "9,0"), ("3", "9,1", "9,6"), ("2", "9,6", ""))},
        {"applies_to": "Подтяг., раз", "quote": "Подтягивания: 10 и более — «5»; 7–9 — «4»; 4–6 — «3»; меньше 4 — «2»",
         "bands": _bands(("5", "10", ""), ("4", "7", "9"), ("3", "4", "6"), ("2", "", "4"))},
        {"applies_to": "Прыжок, см", "quote": "Прыжок в длину: 220 см и более — «5»; 200–219 — «4»; 180–199 — «3»; меньше 180 — «2»",
         "bands": _bands(("5", "220", ""), ("4", "200", "219"), ("3", "180", "199"), ("2", "", "180"))},
    ],
    "score_columns": ["Бег 60 м, с", "Подтяг., раз", "Прыжок, см"], "aggregate": "mean", "rounding": "math", "decimals": 0,
    "rule_quote": "Итоговая оценка = среднее трёх оценок, округляем до целого (0,5 — вверх)",
    "result_column": "", "grade_column": "Оценка", "label_column": "Фамилия",
}


class _NoLLM:
    def is_available(self):
        return False


class _Manager:
    llm_model = _NoLLM()
    hybrid_search = None

    def _document_count(self):
        return 0


def _graded(payload=PAYLOAD, text=None):
    text = text or sheet_text()
    table = parse_markdown_tables(text)[0]
    rules, why = grading.read_rules(payload, table, [("vedomost.jpg", text)])
    assert rules is not None, why
    layout, why = grading.read_layout(payload, table)
    assert layout is not None, why
    rules = grading.fit(rules, layout)
    results, why = grading.grade_rows(table, layout, rules)
    assert results, why
    return table, layout, rules, results


def test_sheet_is_graded_by_the_program_as_the_key_says():
    _table, _layout, rules, results = _graded()
    assert rules.sources == ["vedomost.jpg"] and rules.final is not None and not rules.components
    by_name = {r.label: r for r in results}
    assert set(by_name) == set(GRADE_SHEET_KEY)
    expected_status = {"ok": "ok", "average fixed": "fixed", "grade fixed": "fixed", "filled": "filled", "missing data": "missing"}
    for name, (average, grade, what) in GRADE_SHEET_KEY.items():
        r = by_name[name]
        assert r.status == expected_status[what], name
        if average is None:
            assert r.value is None and r.grade == "" and r.missing == ["К2"]
        else:
            assert r.value == average and r.grade == grade, name
    assert by_name["Васильев Д."].result_fixed and not by_name["Васильев Д."].grade_fixed
    # 84,67 rounds to 85: the rounding rule of the sheet decides the grade
    assert by_name["Жуков С."].calculation == "(84 + 85 + 85) / 3 = 84,67 → 85"


def test_fitness_sheet_with_a_scale_per_exercise_matches_the_users_key():
    _table, _layout, rules, results = _graded(FIZRA_PAYLOAD, FIZRA_TEXT)
    assert sorted(rules.components) == [2, 3, 4] and rules.final is None
    by_name = {r.label: r for r in results}
    for name, (grade, what) in FIZRA_KEY.items():
        r = by_name[name]
        assert r.status == what, name
        assert (r.grade or None) == grade, name
    # 8,5 is «до 8,5» (5); 9,6 would be «9,1–9,6» (3), not «больше 9,6»
    assert by_name["Лебедева"].calculation == "бег 60 м 8,5 → «5»; подтяг. 7 → «4»; прыжок 200 → «4»; (5 + 4 + 4) / 3 = 4,33 → 4"
    assert by_name["Новиков"].missing == ["Прыжок, см"]
    assert {r.status for r in results}.issuperset({"ok", "filled", "fixed", "missing"})


@pytest.mark.parametrize("value, grade", [(8.5, "5"), (8.6, "4"), (9.0, "4"), (9.6, "3"), (9.7, "2")])
def test_shared_boundaries_belong_to_the_written_range(value, grade):
    table = parse_markdown_tables(FIZRA_TEXT)[0]
    rules, _ = grading.read_rules(FIZRA_PAYLOAD, table, [("v", FIZRA_TEXT)])
    band, _ = grading.lookup(value, rules.components[2])
    assert band.grade == grade
    assert grading.lookup(8.55, rules.components[2]) == (None, "между диапазонами шкалы")


def test_open_bands_follow_the_scale_whatever_min_max_the_model_used():
    # the real model wrote «больше 9,6» as max and took «№» for the name column
    run = dict(FIZRA_PAYLOAD["scales"][0], bands=_bands(("5", "8,5", ""), ("4", "8,6", "9,0"), ("3", "9,1", "9,6"), ("2", "", "9,6")))
    payload = dict(FIZRA_PAYLOAD, scales=[run] + FIZRA_PAYLOAD["scales"][1:], label_column="№")
    _t, _l, _r, results = _graded(payload, FIZRA_TEXT)
    by_name = {r.label: r for r in results}
    assert by_name["Морозов"].grade == "2" and by_name["Иванов"].components[0] == "5"
    # from the library it wrote the «2» bands with no numbers at all: the open end of each scale
    blank = [dict(s, bands=s["bands"][:3] + [{"grade": "2", "min": "", "max": ""}]) for s in FIZRA_PAYLOAD["scales"]]
    _t, _l, _r, results = _graded(dict(FIZRA_PAYLOAD, scales=blank), FIZRA_TEXT)
    assert {r.label: r.grade for r in results}["Морозов"] == "2"
    assert [r.label for r in results if r.status == "missing"] == ["Новиков"]


def test_report_shows_calculations_and_does_not_guess():
    table, layout, rules, results = _graded()
    text = grading.report(table, layout, rules, results)
    assert "Дмитриев К.: (45 + 52 + 40) / 3 = 45,67 → 46 → «2» (на листе «3»)" in text
    assert "Жуков С.: (84 + 85 + 85) / 3 = 84,67 → 85 → «5»" in text
    assert "Егорова М.: нет значения: «К2»" in text
    assert "верно: 3 · исправлено: 3 · проставлено: 1 · нельзя посчитать: 1" in text
    fizra = grading.report(*_graded(FIZRA_PAYLOAD, FIZRA_TEXT))
    assert "Новиков: нет значения: «Прыжок, см» — итог посчитать нельзя; на листе стоит «3», проверить её нельзя" in fizra
    assert "| Сидоров | 9,3 | 5 | 190 | 3 · 3 · 3 | **3** | оценка исправлена (было «4») |" in fizra


def test_scale_must_be_quoted_from_the_text():
    text = sheet_text()
    table = parse_markdown_tables(text)[0]
    scale = PAYLOAD["scales"][0]
    invented = dict(PAYLOAD, scales=[dict(scale, bands=scale["bands"][:-1] + [{"grade": "2", "min": "", "max": "45"}])])
    assert grading.read_rules(invented, table, [("v", text)])[0] is None  # 45 is not a bound of the scale
    assert grading.read_rules(dict(PAYLOAD, scales=[dict(scale, quote="Шкала: 90–100 — отлично")]), table, [("v", text)])[0] is None
    assert grading.read_rules(dict(PAYLOAD, scales=[]), table, [("v", text)])[1] == "шкала не найдена"
    # one exercise scale missing from the text: that column cannot be graded honestly
    broken = dict(FIZRA_PAYLOAD, scales=FIZRA_PAYLOAD["scales"][:2] + [dict(FIZRA_PAYLOAD["scales"][2], quote="Прыжок: 250 и более — 5")])
    fizra_table = parse_markdown_tables(FIZRA_TEXT)[0]
    assert "не все шкалы прочитаны" in grading.read_rules(broken, fizra_table, [("v", FIZRA_TEXT)])[1]


def test_without_rounding_a_value_between_bands_is_marked_not_guessed():
    _t, _l, _r, results = _graded(dict(PAYLOAD, rounding="none"))
    zhukov = next(r for r in results if r.label == "Жуков С.")
    assert zhukov.status == "outside" and "между диапазонами" in zhukov.note and zhukov.grade == ""
    _t, _l, _r, results = _graded(dict(FIZRA_PAYLOAD, rounding="none", rule_quote=""), FIZRA_TEXT)
    assert next(r for r in results if r.label == "Лебедева").status == "outside"  # 4,33 and no rounding rule


def test_workbook_has_formulas_scale_sheet_and_highlights(tmp_path):
    from openpyxl import load_workbook

    table, layout, rules, results = _graded()
    path = grading.write_workbook(table, layout, rules, results, tmp_path / "v.xlsx")
    wb = load_workbook(path)
    ws = wb["Ведомость"]
    assert ws["F2"].value == '=IF(COUNT(C2:E2)<3,"нет данных",ROUND(AVERAGE(C2:E2),0))'
    assert ws["G2"].value == '=IF(ISNUMBER(F2),IF(AND(F2>=85,F2<=100),5,IF(AND(F2>=70,F2<=84),4,IF(AND(F2>=50,F2<=69),3,IF(F2<=50,2,"вне шкалы")))),"нет данных")'
    yellow = lambda cell: cell.fill.fgColor.rgb.endswith("FFF2CC")
    assert yellow(ws["F4"]) and "67" in ws["F4"].comment.text  # Васильев: average 67 → 64
    assert yellow(ws["G6"]) and "«3»" in ws["G6"].comment.text  # Дмитриев: grade 3 → 2
    assert ws["G8"].fill.fgColor.rgb.endswith("E2EFDA")  # Жуков: filled
    assert ws["D7"].fill.fgColor.rgb.endswith("FCE4D6")  # Егорова: К2 missing
    assert ws["H8"].value == "проставлено"
    assert wb["Шкала"]["A2"].value == "итог"


def test_fitness_workbook_grades_each_exercise_by_formula(tmp_path):
    from openpyxl import load_workbook

    table, layout, rules, results = _graded(FIZRA_PAYLOAD, FIZRA_TEXT)
    ws = load_workbook(grading.write_workbook(table, layout, rules, results, tmp_path / "f.xlsx"))["Ведомость"]
    assert [ws.cell(row=1, column=c).value for c in range(7, 11)] == ["Бег 60 м → оценка", "Подтяг. → оценка", "Прыжок → оценка", "Проверка"]
    assert ws["G2"].value.startswith('=IF(ISNUMBER(C2),IF(AND(C2>=8.6,C2<=9),4,IF(AND(C2>=9.1,C2<=9.6),3,IF(C2<=8.5,5,IF(C2>=9.6,2,')
    assert ws["F2"].value == '=IF(COUNT(G2:I2)<3,"нет данных",ROUND(AVERAGE(G2:I2),0))'
    assert ws["F4"].fill.fgColor.rgb.endswith("FFF2CC") and ws["F3"].fill.fgColor.rgb.endswith("E2EFDA")
    assert ws["E8"].fill.fgColor.rgb.endswith("FCE4D6")  # Новиков: no jump


def _agent(tmp_path, text=None):
    agent = Agent(_Manager(), store=AttachmentStore(tmp_path / "att"))
    item = Attachment(id="1", name="vedomost.jpg", path=tmp_path / "v.jpg", kind="image", text=text or sheet_text(),
                      recognized=True, read=True)
    agent.store.items.append(item)
    return agent, item


GRADE_ROUTE = {"action": "grade", "source": "attachments", "output": "xlsx", "language": "", "task": "оценки", "search_query": ""}


def test_grade_action_end_to_end_without_the_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATKODI_RUNTIME_DIR", str(tmp_path / "rt"))
    agent, item = _agent(tmp_path, FIZRA_TEXT)
    agent._json = lambda prompt, schema, max_tokens=0: FIZRA_PAYLOAD
    result = agent._do_grade("Где оценка не проставлена — рассчитай", GRADE_ROUTE, [item], lambda _t: None)
    assert result.action == "grade" and not result.incomplete
    assert [f.fmt for f in result.files] == ["xlsx"] and "Морозов: бег 60 м 9,8 → «2»" in result.text


def test_grade_without_a_scale_says_so_and_offers_no_dead_ends(tmp_path):
    text = sheet_text().split("Оценка по среднему")[0]
    agent, item = _agent(tmp_path, text)
    agent._json = lambda prompt, schema, max_tokens=0: dict(PAYLOAD, scales=[])
    result = agent._do_grade("выставь оценки", GRADE_ROUTE, [item], lambda _t: None)
    assert result.incomplete and "оценки не выставил" in result.text and "ни на листе, ни в библиотеке" in result.text
    assert "| Егорова М. |" in result.text  # the recognised table is still shown
    agent._llm_ready = lambda: True
    assert agent.suggest_next("выставь оценки", result) == []


def _route_raw(steps, action="transcribe"):
    base = {"source": "attachments", "output": "xlsx", "language": "", "search_query": ""}
    return {"results": ["таблица"] * len(steps), "action": action, "task": "t", **base,
            "steps": [dict(base, action=a, task=f"шаг {i}") for i, a in enumerate(steps)]}


def test_repeated_plan_steps_collapse_and_calculation_is_not_dropped(tmp_path):
    agent, _item = _agent(tmp_path)
    agent._json = lambda prompt, schema, max_tokens=0: _route_raw(["transcribe"] * 5)
    route = agent.route("Распознай ведомость, посчитай средний балл и оценки, проверь и исправь", [])
    assert route["action"] == "grade" and "steps" not in route
    route = agent.route("Где оценка не проставлена — рассчитай по критериям с листа и проставь", [])
    assert route["action"] == "grade"
    agent._json = lambda prompt, schema, max_tokens=0: _route_raw(["transcribe", "check"])
    route = agent.route("Переведи ведомость в Excel и проверь её по нашим нормативам", [])
    assert route["action"] == "grade" and route["fallback"] == "check"
    agent.store.items[0].text = "| Наименование | Цена |\n|---|---|\n| Бумага | 345 |\n| Ручка | 32 |"
    agent._json = lambda prompt, schema, max_tokens=0: _route_raw(["transcribe"] * 5)
    route = agent.route("Распознай таблицу в Excel и посчитай итог", [])
    assert [s["action"] for s in route["steps"]] == ["transcribe", "answer"]
    route = agent.route("Распознай таблицу в Excel", [])
    assert route["action"] == "transcribe" and "steps" not in route


def test_demands_are_read_from_the_request():
    assert _demands("Распознай ведомость, посчитай средний балл и оценки по шкале, проверь и исправь ошибки") >= {"grade", "calc", "check", "fix"}
    assert _demands("Какой средний балл у Григорьевой?") == set()
    assert _demands("Сделай из счёта таблицу Excel") == {"xlsx"}
    assert "grade" not in _demands("Проверь оценку стоимости в смете")  # an appraisal, not grades


def test_layout_matches_abbreviated_columns_and_keeps_surnames():
    text = sheet_text().replace("Алексеев П.", "Среднев П.")
    table = parse_markdown_tables(text)[0]
    layout, _ = grading.read_layout(dict(PAYLOAD, result_column="Средний балл"), table)
    assert layout.result == 5
    rules, _ = grading.read_rules(PAYLOAD, table, [("v", text)])
    assert "Среднев П." in [r.label for r in grading.grade_rows(table, layout, rules)[0]]
    fizra = parse_markdown_tables(FIZRA_TEXT)[0]
    assert grading._column(fizra.header, "Подтягивания") == 3 and grading._column(fizra.header, "Прыжок в длину") == 4


def test_partial_answer_is_said_plainly(tmp_path):
    agent, _item = _agent(tmp_path)
    done = AgentResult("таблица", action="transcribe")
    note = agent._unfinished("Распознай и посчитай итоговые оценки", done)
    assert note.startswith("⚠ Сделано не всё: перевёл файл в электронный вид, но") and "не выполнил расчёт" in note
    assert agent._unfinished("Распознай таблицу", done) == ""
    plan = AgentResult("план", action="plan", incomplete="норматив найден, но не разобран")
    plan.performed = ["transcribe"]
    assert "норматив найден" in agent._unfinished("Распознай и проверь по нормативам", plan)


def test_check_takes_rules_written_on_the_sheet(tmp_path):
    text = "| Образец | Прочность, МПа |\n|---|---|\n| 1 | 31,0 |\n\nПрочность каждого образца должна быть не менее 32,7 МПа."
    agent, item = _agent(tmp_path, text)
    agent._json = lambda prompt, schema, max_tokens=0: {"items": [{
        "parameter": "Прочность образца 1", "value": "31,0", "requirement": "не менее 32,7 МПа",
        "rule_quote": "Прочность каждого образца должна быть не менее 32,7 МПа", "status": "соответствует"}],
        "conclusion": ""}
    route = {"action": "check", "source": "attachments", "output": "chat", "language": "", "task": "проверь", "search_query": ""}
    result = agent._do_check("проверь", route, [item], lambda _t: None)
    assert "✗ не соответствует" in result.text and "Требования из приложенного файла" in result.text
    agent._json = lambda prompt, schema, max_tokens=0: {"items": [], "conclusion": ""}
    failed = agent._do_check("проверь", route, [item], lambda _t: None)
    assert failed.incomplete and "Нормативы найдены («vedomost.jpg» (приложенный файл))" in failed.text


@pytest.mark.parametrize("suggestion, ok", [
    ("Сравни с другими нормативами", False),
    ("Проверь по нормативам", False),  # empty library
    ("Переведи на английский", True),
])
def test_suggestions_only_what_can_be_done(tmp_path, suggestion, ok):
    agent, _item = _agent(tmp_path)
    assert agent._can_do(suggestion, AgentResult("x", action="transcribe")) is ok


def test_excel_totals_are_formulas_and_wrong_ones_are_marked(tmp_path):
    from openpyxl import load_workbook

    from core.agent.outputs import markdown_to_xlsx

    text = ("| № | Наименование | Кол-во | Цена | Сумма |\n|---|---|---|---|---|\n"
            "| 1 | Бумага | 20 | 345,00 | 6 900,00 |\n| 2 | Картридж | 3 | 7 450,00 | 22 350,00 |\n"
            "| 3 | Папка | 15 | 210,00 | 3 100,00 |\n| | Итого | | | 32 350,00 |\n\nОплатить до 23.10.2026.")
    ws = load_workbook(markdown_to_xlsx(text, tmp_path / "s.xlsx")).active
    assert ws["E2"].value == "=C2*D2" and ws["E4"].value == "=C4*D4"
    assert ws["E4"].fill.fgColor.rgb.endswith("FFF2CC") and "3 100,00" in ws["E4"].comment.text  # 15 × 210 = 3 150
    assert ws["E5"].value == "=SUM(E2:E4)" and ws["E5"].fill.fgColor.rgb.endswith("FFF2CC")
    assert ws["A7"].value == "Оплатить до 23.10.2026."


def test_mean_written_under_the_table_becomes_a_formula(tmp_path):
    from openpyxl import load_workbook

    from core.agent.outputs import markdown_to_xlsx

    text = ("Протокол № 145/26\n\n| № | Прочность, МПа |\n|---|---|\n| 1 | 35,2 |\n| 2 | 33,9 |\n| 3 | 36,1 |\n\n"
            "Средняя прочность серии: 35,1 МПа")
    ws = load_workbook(markdown_to_xlsx(text, tmp_path / "p.xlsx")).active
    assert ws["A6"].value == "Протокол № 145/26" and ws["B6"].value is None
    assert ws["A7"].value.startswith("Средняя прочность") and ws["B7"].value == "=AVERAGE(B2:B4)"


# ---------------------------------------------------------------- situations users are likely to meet


def _fizra_rows():
    table = parse_markdown_tables(FIZRA_TEXT)[0]
    return [table.header] + table.rows


def test_excel_and_csv_sheets_are_read_as_tables_with_their_empty_cells(tmp_path):
    from openpyxl import Workbook

    from core.agent.attachments import AttachmentStore, read_attachment

    wb = Workbook()
    ws = wb.active
    ws.title = "9Б"
    ws.append(["Ведомость сдачи нормативов"])
    ws.append([])
    for row in _fizra_rows():
        ws.append([to_cell(v) for v in row])
    source = tmp_path / "ведомость.xlsx"
    wb.save(source)
    item = read_attachment(AttachmentStore(tmp_path / "att").add(source), None)
    table = parse_markdown_tables(item.text)[0]
    assert table.header[:2] == ["№", "Фамилия"] and len(table.rows) == 8
    assert table.rows[6][1:] == ["Новиков", "9", "6", "", "3"] and table.rows[0][2] == "8,4"

    csv_path = tmp_path / "ведомость.csv"
    csv_path.write_text("\n".join(";".join(r) for r in _fizra_rows()), encoding="cp1251")
    item = read_attachment(AttachmentStore(tmp_path / "att2").add(csv_path), None)
    assert parse_markdown_tables(item.text)[0].rows[1][5] == ""  # Петрова: no grade


def to_cell(value: str):
    number = value.replace(",", ".")
    try:
        return int(number) if number.isdigit() else float(number)
    except ValueError:
        return value or None


def test_word_table_attachment_is_read_as_a_table(tmp_path):
    from docx import Document

    from core.agent.attachments import AttachmentStore, read_attachment

    doc = Document()
    doc.add_paragraph("Ведомость сдачи нормативов")
    rows = _fizra_rows()
    grid = doc.add_table(rows=len(rows), cols=len(rows[0]))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            grid.cell(r, c).text = value
    source = tmp_path / "ведомость.docx"
    doc.save(source)
    item = read_attachment(AttachmentStore(tmp_path / "att").add(source), None)
    assert parse_markdown_tables(item.text)[0].rows[4][1:] == ["Морозов", "9,8", "3", "170", ""]


def test_a_sheet_over_two_pages_is_one_table():
    first, second = FIZRA_TEXT.split("| 5 | Морозов")
    pages = first + "\n\n(страница 2)\n\n| 5 | Морозов" + second  # page 2 has no header row
    table, _name, _text = grading.pick_table([("scan.pdf", pages)])
    assert len(table.rows) == 8 and table.rows[4][1] == "Морозов"
    twice = FIZRA_TEXT + "\n\n" + FIZRA_TEXT.split("|---|---|---|---|---|---|\n")[0] + "|---|---|---|---|---|---|\n| 9 | Яковлев | 8,7 | 7 | 210 | |"
    assert len(grading.pick_table([("scan.pdf", twice)])[0].rows) == 9


def test_grades_in_words_match_digits():
    assert grading._grade_key("отл.") == grading._grade_key("5") == grading._grade_key("Отлично")
    assert grading._grade_key("неуд") == "2" and grading._grade_key("удовл.") == "3" and grading._grade_key("уд") == "3"
    assert grading._grade_key("незачёт") != grading._grade_key("зачёт")
    text = FIZRA_TEXT.replace("| 225 | 5 |", "| 225 | отл. |").replace("| 190 | 4 |", "| 190 | хор |")
    by_name = {r.label: r for r in _graded(FIZRA_PAYLOAD, text)[3]}
    assert by_name["Иванов"].status == "ok" and by_name["Сидоров"].grade_fixed


def test_times_and_absences_in_the_scores():
    text = ("Бег 1000 м: до 3:30 — «5»; 3:31–3:50 — «4»; 3:51–4:20 — «3»; больше 4:20 — «2».\n\n"
            "| № | Фамилия | Бег 1000 м | Оценка |\n|---|---|---|---|\n| 1 | Иванов | 3:28 | |\n| 2 | Петрова | 3:50,0 | 5 |\n"
            "| 3 | Сидоров | 4:35 | |\n| 4 | Кузнецова | н/я | 3 |")
    payload = {"scales": [{"applies_to": "итог", "quote": "Бег 1000 м: до 3:30 — «5»; 3:31–3:50 — «4»; 3:51–4:20 — «3»; больше 4:20 — «2»",
                           "bands": _bands(("5", "", "3:30"), ("4", "3:31", "3:50"), ("3", "3:51", "4:20"), ("2", "4:20", ""))}],
               "score_columns": ["Бег 1000 м"], "aggregate": "single", "rounding": "none", "decimals": 0, "rule_quote": "",
               "result_column": "", "grade_column": "Оценка", "label_column": "Фамилия"}
    table, layout, rules, results = _graded(payload, text)
    by_name = {r.label: r for r in results}
    assert by_name["Иванов"].grade == "5" and by_name["Петрова"].grade == "4" and by_name["Петрова"].grade_fixed
    assert by_name["Сидоров"].grade == "2"
    assert by_name["Кузнецова"].status == "missing" and "записано «н/я»" in by_name["Кузнецова"].note
    text_out = grading.report(table, layout, rules, results)
    assert "Петрова: 3:50,0 → «4» (на листе «5»)" in text_out and "Сидоров: 4:35 → «2»" in text_out
    invented = dict(payload, scales=[dict(payload["scales"][0], bands=_bands(("5", "", "3:25"), ("4", "3:31", "3:50")))])
    assert grading.read_rules(invented, table, [("v", text)])[0] is None  # 3:25 is not on the sheet


EXAM_TEXT = ("| № | Студент | Баллы | Оценка |\n|---|---|---|---|\n| 1 | Белов А. | 93 | отл. |\n| 2 | Гусев П. | 58 | удовл. |\n"
             "| 3 | Ершов В. | | неуд. |\n| 4 | Жданова К. | 90 | |")
EXAM_MESSAGE = "Проверь оценки по шкале: 90 баллов и выше — отл., 75–89 — хор., 60–74 — удовл., ниже 60 — неуд."


def test_scale_in_the_message_with_word_grades_and_a_computed_neighbour_bound():
    # as the real model read it: «ниже 60» became max 59, the scale was put on the «Баллы» column
    payload = {"scales": [{"applies_to": "Баллы", "quote": "90 баллов и выше — отл., 75–89 — хор., 60–74 — удовл., ниже 60 — неуд.",
                           "bands": _bands(("отл.", "90", ""), ("хор.", "75", "89"), ("удовл.", "60", "74"), ("неуд.", "", "59"))}],
               "score_columns": ["Баллы"], "aggregate": "single", "rounding": "none", "decimals": 0, "rule_quote": "",
               "result_column": "", "grade_column": "Оценка", "label_column": "Студент"}
    table = parse_markdown_tables(EXAM_TEXT)[0]
    rules, why = grading.read_rules(payload, table, [("v", EXAM_TEXT), ("сообщение", EXAM_MESSAGE)])
    assert rules is not None, why
    layout, _ = grading.read_layout(payload, table)
    rules = grading.fit(rules, layout)
    by_name = {r.label: r for r in grading.grade_rows(table, layout, rules)[0]}
    assert by_name["Белов А."].status == "ok" and by_name["Гусев П."].grade == "неуд." and by_name["Гусев П."].grade_fixed
    assert by_name["Жданова К."].grade == "отл." and by_name["Ершов В."].status == "missing"
    assert grading.lookup(59.5, rules.final)[0].grade == "неуд."  # «ниже 60», not a gap at 59–60


def test_a_scale_of_something_else_is_not_applied():
    run = ("| № | Фамилия | Время | Оценка |\n|---|---|---|---|\n| 1 | Антонов | 3:28 | 5 |\n| 2 | Быкова | 3:50 | 5 |\n"
           "| 3 | Власов | 4:35 | |\n| 4 | Дорохов | 3:51 | |")
    library = [("Нормативы.txt", FIZRA_TEXT)]
    table = parse_markdown_tables(run)[0]
    # the library's three exercise scales all put on «Время» (seen with the real model)
    to_time = dict(FIZRA_PAYLOAD, scales=[dict(s, applies_to="Время") for s in FIZRA_PAYLOAD["scales"]], score_columns=["Время"])
    rules, why = grading.read_rules(to_time, table, library)
    assert rules is None and "несколько разных шкал" in why
    one = dict(FIZRA_PAYLOAD, scales=[dict(FIZRA_PAYLOAD["scales"][0], applies_to="Время")], score_columns=["Время"])
    rules, why = grading.read_rules(one, table, library)
    assert rules is None and "не подходит к столбцу «Время»" in why  # 60 m seconds against 1000 m times
    # the library's scales for columns this table does not have are not a «result» scale
    exam_table = parse_markdown_tables(EXAM_TEXT)[0]
    assert grading.read_rules(FIZRA_PAYLOAD, exam_table, library)[0] is None


def test_save_in_word_keeps_the_answer_as_it_is(tmp_path):
    agent, _item = _agent(tmp_path)
    agent.history.append(("оценки", "**Оценки по шкале.** …"))
    agent._json = lambda prompt, schema, max_tokens=0: {
        "results": ["документ"], "action": "make_document", "source": "previous_answer", "output": "docx", "language": "",
        "task": "t", "search_query": "", "steps": []}
    route = agent.route("Сохрани это в Word", [])
    assert route["action"] == "transcribe" and route["source"] == "previous_answer" and route["output"] == "docx"
    assert agent.route("Оформи это по нашему образцу протокола в Word", [])["action"] == "make_document"


def test_values_given_in_the_chat_are_used_and_remembered(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATKODI_RUNTIME_DIR", str(tmp_path / "rt"))
    agent, item = _agent(tmp_path, FIZRA_TEXT)
    asked = []

    def model(prompt, schema, max_tokens=0):
        if schema is grading.CHANGES_SCHEMA:
            asked.append(prompt)
            return {"changes": [{"row": "Новиков", "column": "Прыжок", "value": "205"},
                                {"row": "Петрова", "column": "Бег 60 м, с", "value": "7,1"}]}  # 7,1 is not in the message
        return FIZRA_PAYLOAD

    agent._json = model
    first = agent._do_grade("У Новикова прыжок 205, пересчитай", GRADE_ROUTE, [item], lambda _t: None)
    assert asked and "Учтено из ваших сообщений: Новиков — «Прыжок, см»: 205 (на листе пусто)" in first.text
    assert "Новиков: бег 60 м 9,0 → «4»; подтяг. 6 → «3»; прыжок 205 → «4»; (4 + 3 + 4) / 3 = 3,67 → 4 → «4» (на листе «3»)" in first.text
    assert "Петрова: бег 60 м 8,9" in first.text  # the invented value was not taken
    asked.clear()
    second = agent._do_grade("Сохрани заново", GRADE_ROUTE, [item], lambda _t: None)
    assert not asked and "Новиков — «Прыжок, см»: 205" in second.text  # remembered without asking again


def test_after_grades_a_correction_goes_back_to_grading(tmp_path):
    agent, _item = _agent(tmp_path, FIZRA_TEXT)
    agent._last_action = "grade"
    agent._json = lambda prompt, schema, max_tokens=0: {
        "results": ["ответ"], "action": "answer", "source": "attachments", "output": "chat", "language": "",
        "task": "t", "search_query": "", "steps": []}
    assert agent.route("У Новикова прыжок 205, пересчитай", [])["action"] == "grade"
    assert agent.route("Почему у Лебедевой 4?", [])["action"] == "answer"
    # the model sent the question to a check (seen in a long run): still a question about the answer
    agent._json = lambda prompt, schema, max_tokens=0: {
        "results": ["проверка"], "action": "check", "source": "attachments", "output": "chat", "language": "",
        "task": "t", "search_query": "", "steps": []}
    route = agent.route("Почему у Лебедевой 4, а не 5?", [])
    assert route["action"] == "answer" and route["source"] == "previous_answer"


def test_translate_of_an_attached_memo_reads_the_file(tmp_path):
    agent, _item = _agent(tmp_path, "Служебная записка\nПрошу выделить 2 ноутбука.")
    agent._json = lambda prompt, schema, max_tokens=0: {
        "results": ["перевод"], "action": "translate", "source": "library", "output": "chat", "language": "немецкий",
        "task": "Перевести служебную записку на немецкий", "search_query": "", "steps": []}
    assert agent.route("Переведи служебную записку на немецкий", [])["source"] == "attachments"

