"""Real PDFs, late-page tables, explicit source selection and OCR disagreements."""
import io
from pathlib import Path

from PIL import Image
from openpyxl import load_workbook
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table as PDFTable, TableStyle

from core.agent import fitness, vision
from core.agent.agent import Agent
from core.agent.attachments import Attachment, AttachmentStore, read_attachment
from core.agent.pdf_tables import read_pdf
from core.agent.norm_sources import select_named
from core.agent.table_check import parse_markdown_tables


TITLE = "Таблица перевода результата выполнения упражнения № 1 в баллы"
TABLE = "| Количество раз | Баллы |\n|---|---|\n| 20 | 40 |\n| 25 | 50 |"
SHEET = "| ФИО | № упражнения | Результат | Балл |\n|---|---|---|---|\n| Иванов | 1 | 20 | |"


def _pdf(path, *, blank_pages=0, scan=False, two_tables=False):
    pdfmetrics.registerFont(TTFont("OnestTest", str(Path(__file__).parents[1] / "assets/fonts/Onest-Regular.ttf")))
    doc = canvas.Canvas(str(path))
    for index in range(blank_pages):
        doc.setFont("OnestTest", 11)
        doc.drawString(30, 750, f"Раздел правил {index + 1}. " + "Справочный текст. " * 8)
        doc.showPage()
    if scan:
        from reportlab.lib.utils import ImageReader
        image = Image.new("RGB", (500, 500), "white")
        doc.drawImage(ImageReader(image), 20, 100, 500, 500)
    else:
        _draw_table(doc, TITLE, [["Количество раз", "Баллы"], ["20", "40"], ["25", "50"]], 720)
        if two_tables:
            _draw_table(doc, TITLE.replace("№ 1", "№ 2"), [["Количество раз", "Баллы"], ["45", "55"], ["46", "58"]], 460)
    doc.save()


def _draw_table(doc, title, rows, top):
    doc.setFont("OnestTest", 10)
    doc.drawString(30, top + 20, title)
    table = PDFTable(rows, colWidths=[140, 100], rowHeights=28)
    table.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "OnestTest"), ("GRID", (0, 0), (-1, -1), 1, "black")]))
    table.wrapOn(doc, 500, 800)
    table.drawOn(doc, 30, top - 28 * len(rows))


def test_table_on_page_42_is_read_with_page_reference(tmp_path):
    path = tmp_path / "НФП.pdf"
    _pdf(path, blank_pages=41)
    reading = read_pdf(path)
    maps, used, conflicts = fitness.standards([(path.name, reading.text)])
    assert reading.pages == 42 and not reading.issues and not reading.recognized
    assert maps["1"] == {20: 40, 25: 50}
    assert used == ["НФП.pdf, стр. 42"] and not conflicts


def test_spatial_order_keeps_two_tables_on_same_page_separate(tmp_path):
    path = tmp_path / "two.pdf"
    _pdf(path, two_tables=True)
    reading = read_pdf(path)
    maps, _, conflicts = fitness.standards([(path.name, reading.text)])
    assert maps["1"] == {20: 40, 25: 50}
    assert maps["2"] == {45: 55, 46: 58} and not conflicts


def test_mixed_pdf_scans_after_page_30_are_not_silently_skipped(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    _pdf(path, blank_pages=31, scan=True)
    calls = []
    monkeypatch.setattr(vision, "read_image", lambda *args: (calls.append(1) or TITLE + "\n" + TABLE, []))
    class Model:
        supports_vision = True
    reading = read_pdf(path, Model())
    assert len(calls) == 1 and reading.pages == 32 and reading.recognized
    maps, used, _ = fitness.standards([(path.name, reading.text)])
    assert maps["1"][20] == 40 and used == ["scan.pdf, стр. 32"]
    without_model = read_pdf(path)
    assert any("стр. 32" in issue for issue in without_model.issues)
    assert not fitness.standards([(path.name, without_model.text)])[0]


def test_uncertain_ocr_norm_values_do_not_enter_the_scale(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    _pdf(path, scan=True)
    doubt = vision.CellDoubt(2, 1, "40", "49", 0.0, row_label="20 40", column_name="Баллы")
    monkeypatch.setattr(vision, "read_image", lambda *args: (TITLE + "\n" + TABLE, [doubt]))
    class Model:
        supports_vision = True
    reading = read_pdf(path, Model())
    maps, _, _ = fitness.standards([(path.name, reading.text)])
    assert 20 not in maps["1"] and maps["1"][25] == 50 and reading.issues


def test_named_library_pdf_is_read_beyond_cached_search_snippet(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATKODI_RUNTIME_DIR", str(tmp_path / "runtime"))
    path = tmp_path / "НФП2023.pdf"
    _pdf(path, blank_pages=41)
    class Search:
        def hybrid_search(self, query, top_k=5):
            return {"relevant_docs": [dict(path=str(path), filename=path.name, content="Только начало большого PDF") ]}
    class Manager:
        llm_model = None
        hybrid_search = Search()
    store = AttachmentStore(tmp_path / "att")
    item = Attachment("1", "ведомость.xlsx", tmp_path / "v.xlsx", "document", text=SHEET, read=True)
    wrong = Attachment("2", "Другой.pdf", tmp_path / "missing.pdf", "document", text=TITLE + "\n" + TABLE.replace("40", "99"), read=True)
    store.items = [item, wrong]
    agent = Agent(Manager(), store=store)
    result = agent._do_grade("Поставь баллы по документу «НФП2023»", dict(source="attachments", task="баллы", search_query="НФП2023", output="xlsx"), [], lambda _: None)
    assert not result.incomplete
    assert "20 → 40" in result.text and "20 → 99" not in result.text
    assert "НФП2023.pdf, стр. 42" in result.sources
    assert len(agent._norm_pdf_cache) == 1


def test_unknown_explicit_document_never_uses_another_document():
    selected, issue = select_named([("other.pdf", TITLE + "\n" + TABLE)], "Рассчитай по документу «НФП2023»")
    assert selected == [] and issue


def test_numeric_verification_works_without_token_probabilities(monkeypatch):
    original = SHEET.replace("20", "25")
    monkeypatch.setattr(vision, "ask_image", lambda *a, **kw: SHEET)
    doubts = vision.verify_numbers(object(), b"", original)
    assert len(doubts) == 1 and doubts[0].value == "25" and doubts[0].alternative == "20"
    assert doubts[0].column == 2


def test_disputed_photo_cell_blocks_points_and_sum(tmp_path):
    table = parse_markdown_tables(SHEET)[0]
    doubt = vision.CellDoubt(2, 2, "20", "25", 0.0, row_label="Иванов 1", column_name="Результат")
    path = tmp_path / "result.xlsx"
    body, _, _ = fitness.calculate(table, [("НФП", TITLE + "\n" + TABLE)], path, doubts=[doubt])
    assert "неуверенное распознавание" in body and "не рассчитано" in body
    wb = load_workbook(path)
    assert wb["Ведомость"]["D2"].value == '="проверьте распознавание"'
    assert "25" in wb["Ведомость"]["D2"].comment.text
    wb.close()


def test_explicit_norm_name_takes_precedence_over_mentioned_results_file():
    selected, issue = select_named([("ведомость.xlsx", SHEET), ("other.pdf", TABLE)],
                                   "Заполни ведомость по документу «НФП2023»")
    assert selected == [] and issue
    selected, issue = select_named([("other.pdf", TABLE)], "Рассчитай по документу НФП2023")
    assert selected == [] and issue


def test_image_reading_without_logprobs_runs_independent_numeric_check(monkeypatch):
    monkeypatch.setattr(vision, "upright_reading", lambda *a: (0, SHEET, []))
    monkeypatch.setattr(vision, "ask_image", lambda *a, **kw: SHEET.replace("20", "25"))
    text, doubts = vision.read_image(object(), b"")
    assert text == SHEET and len(doubts) == 1 and doubts[0].alternative == "25"


def test_attachment_pdf_uses_structured_reader(tmp_path):
    path = tmp_path / "standards.pdf"
    _pdf(path)
    item = read_attachment(AttachmentStore(tmp_path / "att").add(path), None)
    maps, used, _ = fitness.standards([(item.name, item.text)])
    assert item.ok and not item.error and item.pages == 1
    assert maps["1"][20] == 40 and used == ["standards.pdf, стр. 1"]


def test_identity_disagreement_also_requires_verification(monkeypatch):
    monkeypatch.setattr(vision, "ask_image", lambda *a, **kw: SHEET.replace("Иванов", "Петров"))
    doubts = vision.verify_numbers(object(), b"", SHEET)
    assert len(doubts) == 1 and doubts[0].column == 0 and doubts[0].alternative == "Петров"


def test_results_with_units_stay_numeric_in_generated_workbook(tmp_path):
    table = parse_markdown_tables(SHEET.replace("20", "20 раз"))[0]
    path = tmp_path / "units.xlsx"
    body, _, _ = fitness.calculate(table, [("НФП", TITLE + "\n" + TABLE)], path)
    assert "20 раз → 40" in body
    book = load_workbook(path)
    assert book["Ведомость"]["C2"].value == 20
    book.close()


def test_vertical_upright_pdf_glyphs_are_read_in_their_actual_direction(tmp_path):
    pdfmetrics.registerFont(TTFont('OnestTest', str(Path(__file__).parents[1] / 'assets/fonts/Onest-Regular.ttf')))
    path = tmp_path / 'vertical.pdf'
    doc = canvas.Canvas(str(path)); doc.setFont('OnestTest', 9)
    doc.drawString(30, 720, TITLE)
    for x in [30,90,150]: doc.line(x,550,x,700)
    for y in [550,575,600,700]: doc.line(30,y,150,y)
    for word,x,start in [('Количество',43,608),('раз',59,635),('Баллы',112,625)]:
        for i,char in enumerate(word): doc.drawString(x,start+i*7,char)
    for y,result,score in [(583,'20','40'),(558,'25','50')]:
        doc.drawString(45,y,result);doc.drawString(110,y,score)
    doc.save()
    reading = read_pdf(path)
    maps,_,_=fitness.standards([(path.name,reading.text)])
    assert maps['1'] == {20:40,25:50}
