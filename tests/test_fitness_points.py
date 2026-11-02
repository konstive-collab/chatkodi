"""Regression cases based on the user's exercise 1/2 points-table screenshots."""
from openpyxl import Workbook, load_workbook

from core.agent.agent import Agent
from core.agent.attachments import AttachmentStore, read_attachment, _fitness_header
from core.agent import fitness
from core.agent.table_check import parse_markdown_tables


class Manager:
    llm_model = None
    hybrid_search = None

    def _document_count(self):
        return 0


HEAD = ["Нагрудный номер", "Возрастная группа", "Категория", "Пол", "Весовая категория",
        "№ упражнения [1]", "Результат [1]", "Балл [1]", "№ упражнения [2]", "Результат [2]", "Балл [2]", "Сумма баллов", "Оценка физической подготовленности"]
ROWS = [
    ["2", "к", "3 курс", "м", "70", "1", "20", "", "2", "45", "", "", ""],
    ["3", "к", "3 курс", "м", "70", "1", "25", "", "2", "46", "", "", ""],
    ["4", "к", "3 курс", "м", "70", "1", "43", "", "2", "34", "", "", ""],
    ["10", "к", "3 курс", "м", "70", "", "", "", "", "", "", "", ""],
    ["23", "к", "3 курс", "м", "80+", "1", "50", "", "2", "24", "", "", ""],
    ["27", "к", "3 курс", "м", "70", "1", "43", "", "2", "54", "", "", ""],
    ["99", "к", "3 курс", "м", "70", "1", "51", "", "2", "45", "", "", ""],
]


def norms():
    # Complete discrete maps, including the non-linear part of exercise 1.
    first = {1: 1, 2: 2, 3: 4, 4: 6, 5: 8, 6: 10, 7: 12, 8: 14, 9: 16, 10: 18,
             11: 20, 12: 22, 13: 24, 14: 27, 15: 30}
    first.update({i: i * 2 for i in range(16, 51)})
    second = {i: i if i <= 36 else 36 + 3 * (i - 36) if i <= 40 else 40 + 3 * (i - 40) for i in range(2, 61)}
    # Screenshot: 37→37, 38→38, 39→39, 40→40; 41→43 ... 60→100.
    second.update({i: i for i in range(37, 41)})
    out = []
    for ex, mapping, height in [(1, first, 10), (2, second, 12)]:
        out += [f"Таблица перевода результата выполнения упражнения № {ex} в баллы",
                "| " + " | ".join(["Количество раз", "Баллы"] * 5) + " |", "|" + "---|" * 10]
        values = sorted(mapping, reverse=True)
        for i in range(height):
            cells = []
            for j in range(5):
                at = j * height + i
                cells += [str(values[at]), str(mapping[values[at]])] if at < len(values) else ["—", "—"]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def test_complete_side_by_side_lookup_tables():
    maps, sources, conflicts = fitness.standards([("нормативы.docx", norms())])
    assert len(maps["1"]) == 50 and len(maps["2"]) == 59
    assert maps["1"][14] == 27  # cannot approximate as 2 * repetitions
    assert maps["1"][43] == 86 and maps["2"][45] == 55
    assert sources == ["нормативы.docx"] and not conflicts


def test_multirow_merged_excel_headers_and_exact_points(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATKODI_RUNTIME_DIR", str(tmp_path / "runtime"))
    wb = Workbook()
    ws = wb.active
    for c, text in enumerate(HEAD[:5], 1):
        ws.cell(1, c, text)
        ws.merge_cells(start_row=1, end_row=3, start_column=c, end_column=c)
    ws.cell(1, 6, "Физические упражнения")
    ws.merge_cells(start_row=1, end_row=1, start_column=6, end_column=11)
    ws.cell(2, 6, "Общие физические упражнения")
    ws.merge_cells(start_row=2, end_row=2, start_column=6, end_column=11)
    for c, text in enumerate(["№ упражнения", "Результат", "Балл"] * 2, 6):
        ws.cell(3, c, text)
    for c, text in enumerate(HEAD[11:], 12):
        ws.cell(1, c, text)
        ws.merge_cells(start_row=1, end_row=3, start_column=c, end_column=c)
    ws.append(list(range(8, 21)))  # printed column numbers are not a person
    for row in ROWS:
        ws.append(row)
    data = tmp_path / "ведомость.xlsx"
    wb.save(data)
    standard_file = tmp_path / "нормативы.txt"
    standard_file.write_text(norms(), encoding="utf-8")
    store = AttachmentStore(tmp_path / "att")
    data_item = read_attachment(store.add(data), None)
    norms_item = read_attachment(store.add(standard_file), None)
    assert data_item.ok and norms_item.ok
    table = parse_markdown_tables(data_item.text)[0]
    assert table.header == HEAD and len(table.rows) == 7
    agent = Agent(Manager(), store=store)
    agent._json = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("No model should calculate these points"))
    route = dict(source="attachments", task="Рассчитай баллы", search_query="", output="xlsx")
    result = agent._do_grade("Рассчитай баллы", route, [norms_item], lambda _: None)
    assert not result.incomplete
    assert "№ 1: 20 → 40; № 2: 45 → 55 | 95 | ✓" in result.text
    assert "№ 1: 25 → 50; № 2: 46 → 58 | 108 | ✓" in result.text
    assert "№ 1: 43 → 86; № 2: 34 → 34 | 120 | ✓" in result.text
    assert "№ 1: 50 → 100; № 2: 24 → 24 | 124 | ✓" in result.text
    assert "№ 1: 43 → 86; № 2: 54 → 82 | 168 | ✓" in result.text
    assert "результат 51 отсутствует" in result.text
    assert "нет результатов упражнений" in result.text
    book = load_workbook(result.files[0].path)
    assert book["Нормативы"].max_row == 110
    assert "VLOOKUP" in book["Ведомость"]["H2"].value
    assert "FALSE" in book["Ведомость"]["H2"].value
    assert 'F2&""="1"' in book["Ведомость"]["H2"].value
    assert "COUNT(H2,K2)" in book["Ведомость"]["L2"].value
    assert book["Ведомость"]["M2"].value is None  # no invented final rating
    book.close()


def test_conflicting_tables_cannot_silently_choose_a_category(tmp_path):
    maps, used, conflicts = fitness.standards([("one", norms()), ("other", norms().replace("50 | 100", "50 | 99"))])
    assert "1" not in maps and "2" in maps and conflicts == ["1"]


def test_ordinary_table_header_is_unchanged():
    rows = [["ФИО", "Результат"], ["Иванов", "20"]]
    assert _fitness_header(rows) == rows


def test_points_request_cannot_be_routed_to_generic_answer(tmp_path):
    from core.agent.attachments import Attachment
    store = AttachmentStore(tmp_path / "att")
    text = "| " + " | ".join(HEAD) + " |\n|" + "---|" * len(HEAD) + "\n"
    text += "| " + " | ".join(ROWS[0]) + " |"
    store.items.append(Attachment("1", "sheet.xlsx", tmp_path / "sheet.xlsx", "document", text=text, read=True))
    agent = Agent(Manager(), store=store)
    agent._json = lambda *a, **kw: dict(results=["расчёт"], action="answer", task="баллы", source="attachments",
                                       output="chat", language="", search_query="", steps=[])
    for message in ["Рассчитай баллы", "Заполни баллы по нормативам", "Посчитай сумму баллов"]:
        assert agent.route(message, [])["action"] == "grade"
