"""Assistant pieces that do not need a model: table arithmetic, requirement
checks, number tracing, output files, routing guards and speech text."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.agent.agent import Agent, AgentResult, _strip_headers, requirement_holds
from core.agent.attachments import AttachmentStore, kind_of, read_attachment
from core.agent.outputs import docx_to_markdown, markdown_to_docx, markdown_to_xlsx
from core.agent.table_check import check_tables, parse_markdown_tables, to_number
from core.speech import speakable

INVOICE = """СЧЁТ № 318
| № | Наименование | Кол-во | Цена, руб. | Сумма, руб. |
|---|---|---|---|---|
| 1 | Бумага А4 | 20 пачек | 345,00 | 6 900,00 |
| 2 | Картридж | 3 шт. | 7 450,00 | 22 350,00 |
| 3 | Папка | 15 шт. | 210,00 | 3 150,00 |
| 4 | Ручка | 50 шт. | 32,00 | 1 600,00 |

Итого: 34 000,00 руб.
"""


def test_table_numbers_and_parse():
    assert to_number("22 350,00") == 22350.0
    assert to_number("20 пачек") == 20.0
    assert to_number("4,0–5,0") is None
    table = parse_markdown_tables(INVOICE)[0]
    assert table.header[1] == "Наименование" and len(table.rows) == 4


def test_table_check_confirms_consistent_invoice():
    notes = check_tables(INVOICE)
    assert any(n.startswith("✓") and "произведение" in n for n in notes)
    assert any(n.startswith("✓") and "34 000,00" in n for n in notes)
    assert not any(n.startswith("⚠") for n in notes)


def test_table_check_flags_misread_digit():
    notes = check_tables(INVOICE.replace("22 350,00", "22 850,00"))
    assert any("⚠" in n and "Картридж" in n for n in notes)
    assert any("⚠" in n and "Итого" in n for n in notes)


def test_table_check_mean_line():
    text = "| № | Прочность, МПа |\n|---|---|\n| 1 | 35,2 |\n| 2 | 33,9 |\n| 3 | 36,1 |\n\nСредняя прочность: 35,1 МПа"
    assert any("✓ 35,1" in n for n in check_tables(text))


@pytest.mark.parametrize("value, requirement, expected", [
    ("34,7 МПа", "не менее 32,7 МПа", True),
    ("31,0 МПа", "не менее 32,7 МПа", False),
    ("24,1°", "от 17 до 23 °C", False),
    ("18,5°", "от 17 до 23 °C", True),
    ("от 33,0 до 36,1 МПа", "не менее 28,0 МПа", True),
    ("34 000,00 руб.", "не более 30 000 руб.", False),
    ("11,2", "4,0–9,0", False),
    ("вкл. в 12:30", "при температуре выше 23 °C", None),  # time vs temperature: not comparable
    ("5 шт.", "не более 3 кг", None),  # different units
])
def test_requirement_holds(value, requirement, expected):
    assert requirement_holds(value, requirement) is expected


def test_strip_headers():
    assert _strip_headers("=== Файл «a.jpg» ===\nТекст") == "Текст"


class _NoLLM:
    def is_available(self):
        return False


class _Manager:
    llm_model = _NoLLM()
    hybrid_search = None

    def process_message(self, question, forced_mode=None):
        return f"RAG: {question}"


def test_trace_numbers_flags_example_and_unknown_values(tmp_path):
    agent = Agent(_Manager(), store=AttachmentStore(tmp_path))
    data = "Средняя прочность 34,7 МПа, протокол № 145/26 от 14.10.2026"
    body = "# Заключение\n1. Результаты\nСредняя прочность 34,7 МПа; прежний результат 41,5 МПа; итог 77,7.\nДата 14.10.2026, 01.01.2030"
    notes = agent._trace_numbers(body, data, "", example="значение 41,5")
    assert "41,5" in notes and "77,7" in notes and "01.01.2030" in notes
    assert "34,7" not in notes and "14.10.2026" not in notes


def test_trace_numbers_reads_thousands_as_one_number(tmp_path):
    agent = Agent(_Manager(), store=AttachmentStore(tmp_path))
    notes = agent._trace_numbers("Было 1 200 000 руб., стало 1 350 000 руб.", "1 200 000 и 1 350 000", "", "", quiet_ok=True)
    assert notes == ""


def test_date_at_the_end_of_a_sentence_is_a_date(tmp_path):
    agent = Agent(_Manager(), store=AttachmentStore(tmp_path))
    assert agent._trace_numbers("Приёмка переносится на 30.10.2026.", "сегодня 28.10.2026, пятница 30.10.2026", "", "", quiet_ok=True) == ""


def test_calendar_names_the_weekdays_ahead():
    import datetime

    from core.agent.agent import _calendar

    friday = datetime.date.today() + datetime.timedelta(days=(4 - datetime.date.today().weekday()) % 7 or 7)
    assert f"пятница {friday:%d.%m.%Y}" in _calendar()


def test_letter_is_signed_by_the_remembered_user(tmp_path, monkeypatch):
    from core.agent.memory import MemoryStore

    memory = MemoryStore(tmp_path / "memory.json")
    memory.add("Пользователь — Петрова Анна Сергеевна, инженер ПТО ООО «СтройЛаб».")
    monkeypatch.setattr("core.agent.memory.get_memory", lambda: memory)
    monkeypatch.setattr("core.agent.instructions.load_instructions", lambda: "")
    agent = Agent(_Manager(), store=AttachmentStore(tmp_path))
    prompts = []
    agent._generate = lambda prompt, **kw: prompts.append(prompt) or "Письмо.\nС уважением,\nПетрова А. С."
    route = {"action": "write", "source": "none", "output": "chat", "language": "", "task": "письмо", "search_query": ""}
    agent._do_write("Напиши письмо подрядчику", route, [], lambda s: None)
    assert "ОТПРАВИТЕЛЬ" in prompts[0] and "Петрова Анна Сергеевна" in prompts[0].split("ОТПРАВИТЕЛЬ")[1]


def test_trace_names_flags_requisites_copied_from_example():
    from core.agent.agent import _trace_names

    example = "Генеральному директору ООО «Север» Петрову А. А.\nУважаемый Андрей Алексеевич!\nПовторная приёмка."
    data = "АО «Северный терминал»\nУважаемый господин Шмидт!\nкоммерческий директор О. В. Гусева"
    body = "Генеральному директору ООО «Север» Петрову А. А.\nУважаемый Андрей Алексеевич!\nСогласны на проверку."
    note = _trace_names(body, data, example)
    assert "«Север»" in note and "Петрову" in note and "Андрей" in note
    good = "АО «Северный терминал»\nУважаемая О. В. Гусева!\nСогласны на проверку."
    assert _trace_names(good, data, example) == ""
    form = "| Показатель | Фактическое значение | Вывод |\n|---|---|---|\n| Минимальная прочность | 33,0 | соответствует |"
    assert _trace_names(form, "33,0", form) == ""  # a copied table form is not a name


def test_route_guards_without_files(tmp_path):
    agent = Agent(_Manager(), store=AttachmentStore(tmp_path))
    agent._json = lambda prompt, schema, max_tokens=0: {
        "action": "transcribe", "source": "attachments", "output": "chat", "language": "", "task": "t", "search_query": "",
    }
    route = agent.route("сделай таблицу", [])
    assert route["source"] == "library" and route["action"] == "answer"


def test_without_llm_answers_from_library(tmp_path):
    agent = Agent(_Manager(), store=AttachmentStore(tmp_path))
    result = agent.handle("Кто ответственный?")
    assert result.text == "RAG: Кто ответственный?"


def test_docx_roundtrip_keeps_headings_and_tables(tmp_path):
    md = "# Заключение № 1\n## Результаты\n| Показатель | Значение |\n|---|---|\n| Прочность | **34,7** |\n\n- пункт списка\nОбычный текст"
    path = markdown_to_docx(md, tmp_path / "out.docx")
    back = docx_to_markdown(path)
    assert "# Заключение № 1" in back and "## Результаты" in back
    assert "| Прочность | 34,7 |" in back and "Обычный текст" in back


def test_docx_uses_example_styles(tmp_path):
    from docx import Document
    from docx.shared import Pt

    example = Document()
    example.styles["Normal"].font.size = Pt(15)
    example.add_paragraph("Старый текст образца")
    example.save(str(tmp_path / "example.docx"))
    path = markdown_to_docx("Новый текст", tmp_path / "new.docx", base_docx=tmp_path / "example.docx")
    doc = Document(str(path))
    assert [p.text for p in doc.paragraphs if p.text] == ["Новый текст"]
    assert doc.styles["Normal"].font.size == Pt(15)


def test_xlsx_numbers_are_numbers(tmp_path):
    from openpyxl import load_workbook

    path = markdown_to_xlsx(INVOICE, tmp_path / "t.xlsx")
    ws = load_workbook(str(path)).active
    assert ws["E3"].value == 22350.0 and ws["C2"].value == "20 пачек"


def test_document_attachment_is_read(tmp_path):
    from docx import Document

    doc = Document()
    doc.add_paragraph("Прошу выделить 2 ноутбука")
    source = tmp_path / "zapiska.docx"
    doc.save(str(source))
    store = AttachmentStore(tmp_path / "att")
    item = read_attachment(store.add(source), llm=None)
    assert kind_of(source) == "document" and "ноутбука" in item.text and not item.error


def test_image_without_vision_reports_reason(tmp_path):
    from PIL import Image

    source = tmp_path / "photo.png"
    Image.new("RGB", (20, 20), "white").save(source)
    item = read_attachment(AttachmentStore(tmp_path / "att").add(source), llm=None)
    assert kind_of(source) == "image" and not item.ok and "зрения" in item.error


def test_command_checks():
    from core.agent.commands import CommandProposal, unwrap, validate_powershell

    assert CommandProposal(["Get-Date | Format-Table"], "").dangerous == []
    assert CommandProposal(["format D: /q"], "").dangerous
    assert CommandProposal(["Remove-Item C:\\temp -Recurse"], "").dangerous
    assert unwrap('powershell -Command "Get-PSDrive C"') == "Get-PSDrive C"
    good, unknown, broken = validate_powershell(["Get-PSDrive C", "Get-Item C:\\ | Format-String x", "Get-ChildItem | Where-Object {"])
    assert good == [] and any("Format-String" in p for p in unknown) and broken


def test_speech_codes_follow_the_documents():
    from core.speech import code_vocabulary, restore_codes

    vocab = code_vocabulary(["Датчик ДД-10, погрузчик ЕП-25, датчик ДД-10", "ТО-1"])
    assert vocab[0] == "ДД-10" and "ЕП-25" in vocab
    assert restore_codes("цена DD-10 и EP25", vocab) == "цена ДД-10 и ЕП-25"
    assert restore_codes("модель XZ-99", vocab) == "модель XZ-99"  # unknown code stays


def test_speakable_drops_tables_and_markdown():
    text = "**Итог: 34,7.**\n\n| a | b |\n|---|---|\n📎 Источник: x\nВсё соответствует нормам."
    assert speakable(text) == "Итог: 34,7. Всё соответствует нормам."


def test_this_invoice_means_the_attached_one(tmp_path):
    from core.agent.attachments import Attachment

    agent = Agent(_Manager(), store=AttachmentStore(tmp_path))
    agent.store.items.append(Attachment(id="1", name="schet_318.jpg", path=tmp_path / "s.jpg", kind="image",
                                        text="СЧЁТ НА ОПЛАТУ № 318\nИтого: 34 000,00 руб.", recognized=True, read=True))
    route = {"action": "answer", "source": "library", "output": "chat", "language": "", "task": "согласование", "search_query": "оплата"}
    assert agent._reroute_by_request(dict(route), "Можно оплатить этот счёт без согласования?")["source"] == "both"
    assert agent._reroute_by_request(dict(route), "А этот вопрос где регулируется?")["source"] == "library"
