"""Separate chat norms and row-oriented Excel results must reach verified grading."""
from copy import deepcopy
from datetime import time, timedelta

import pytest
from docx import Document
from openpyxl import Workbook, load_workbook

from core.agent.agent import Agent
from core.agent.attachments import AttachmentStore, read_attachment, _cell
from core.agent import grading
from core.agent.table_check import parse_markdown_tables


class Manager:
    llm_model = None
    hybrid_search = None

    def _document_count(self):
        return 0


ROUTE = dict(action="grade", source="attachments", output="xlsx", language="",
             task="Рассчитай оценки по приложенным нормативам", search_query="")
RUN = "Бег 1000 м: до 3:40 — 5; 3:41–4:00 — 4; больше 4:00 — 3"
PULL = "Подтягивания: 10 и более — 5; 7–9 — 4; меньше 7 — 3"
PAYLOAD = dict(
    scales=[dict(applies_to="Бег 1000 м", quote=RUN, bands=[
        dict(grade="5", min="", max="3:40"), dict(grade="4", min="3:41", max="4:00"),
        dict(grade="3", min="4:00", max="")]),
        dict(applies_to="Подтягивания", quote=PULL, bands=[
            dict(grade="5", min="10", max=""), dict(grade="4", min="7", max="9"),
            dict(grade="3", min="", max="7")])],
    score_columns=["Результат"], aggregate="single", rounding="none", decimals=0,
    rule_quote="", result_column="", grade_column="Оценка", label_column="ФИО", exercise_column="Упражнение")


@pytest.mark.parametrize("upload", ["together", "norms_last", "results_last", "followup"])
def test_real_excel_and_word_uploads_are_used_together(tmp_path, monkeypatch, upload):
    monkeypatch.setenv("CHATKODI_RUNTIME_DIR", str(tmp_path / "runtime"))
    wb = Workbook()
    ws = wb.active
    ws.append(["ФИО", "Упражнение", "Результат", "Оценка"])
    ws.append(["Иванов Иван Иванович", "Бег 1000 м", time(0, 3, 50), None])
    ws.cell(2, 3).number_format = "m:ss"
    ws.append(["Петров Петр Петрович", "Подтягивания", 8, None])
    ws.append(["Сидоров Семен Семенович", "Плавание", 30, None])
    ws.append(["Орлов Олег Олегович", "Подтягивания", None, None])
    data = tmp_path / "results.xlsx"
    wb.save(data)
    doc = Document()
    doc.add_paragraph(RUN)
    doc.add_paragraph(PULL)
    # Larger than the results sheet: must never be picked as the data table.
    norms_table = doc.add_table(rows=1, cols=2)
    norms_table.cell(0, 0).text = "Показатель"
    norms_table.cell(0, 1).text = "Норма"
    for i in range(10):
        cells = norms_table.add_row().cells
        cells[0].text, cells[1].text = f"Пункт {i}", str(i)
    norms = tmp_path / "standards.docx"
    doc.save(norms)
    store = AttachmentStore(tmp_path / "att")
    results_item, norms_item = [read_attachment(store.add(p), None) for p in [data, norms]]
    assert results_item.ok and norms_item.ok
    assert "3:50" in results_item.text
    agent = Agent(Manager(), store=store)
    calls = []

    def extract(prompt, schema, max_tokens=0):
        calls.append(prompt)
        assert RUN in prompt and PULL in prompt
        assert "ФИО | Упражнение | Результат | Оценка" in prompt
        return deepcopy(PAYLOAD)

    agent._json = extract
    agent._library_docs = lambda *a, **kw: pytest.fail("Attached norms should be used before library fallback")
    new = {"together": [results_item, norms_item], "norms_last": [norms_item],
           "results_last": [results_item], "followup": []}[upload]
    result = agent._do_grade("Рассчитай оценки", ROUTE, new, lambda _: None)
    assert not result.incomplete and len(calls) == 1
    assert set(result.sources) == {"results.xlsx", "standards.docx"}
    assert "не найдена шкала" in result.text
    assert "Иванов Иван Иванович: Бег 1000 м: 3:50 → «4»" in result.text
    assert "Петров Петр Петрович: Подтягивания: 8 → «4»" in result.text
    book = load_workbook(result.files[0].path)
    sheet = book["Sheet"]
    assert "*86400" in sheet["D2"].value
    assert "*86400" not in sheet["D3"].value  # repetitions are not time fractions
    assert 'B2="Бег 1000 м"' in sheet["D2"].value
    assert sheet["D4"].value == '="нет шкалы"'
    assert "нет данных" in sheet["D5"].value
    assert "standards.docx" in str(list(book["Нормативы ChatKodi"].values))
    book.close()


def test_row_scales_still_require_verbatim_evidence():
    table = parse_markdown_tables("| ФИО | Упражнение | Результат | Оценка |\n|---|---|---|---|\n| Иванов | Подтягивания | 8 | | ")[0]
    payload = deepcopy(PAYLOAD)
    payload["scales"][1]["bands"][0]["min"] = "100"
    rules, why = grading.read_rules(payload, table, [("norms", RUN + "\n" + PULL)])
    assert rules is None and "не найдена" in why


def test_duplicate_exercise_norms_are_ambiguous():
    table = parse_markdown_tables("| ФИО | Упражнение | Результат |\n|---|---|---|\n| Иванов | Подтягивания | 8 | ")[0]
    payload = deepcopy(PAYLOAD)
    payload["scales"].append(deepcopy(payload["scales"][1]))
    rules, why = grading.read_rules(payload, table, [("norms", RUN + "\n" + PULL)])
    assert rules is None and "несколько разных шкал" in why


def test_excel_elapsed_time_is_read_as_minutes_and_seconds():
    assert _cell(time(0, 3, 50)) == "3:50"
    assert _cell(time(0, 0, 8, 400000)) == "0:08,4"
    assert _cell(timedelta(minutes=63, seconds=5)) == "63:05"
