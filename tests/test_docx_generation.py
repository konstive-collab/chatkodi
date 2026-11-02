from pathlib import Path
import zipfile

from docx import Document

from core.generation.docx_engine import create_legal_draft


def test_placeholder_docx_is_cloned_validated_and_original_unchanged(tmp_path):
    template = tmp_path / "template.docx"
    document = Document()
    paragraph = document.add_paragraph()
    paragraph.add_run("Кому: {{АДРЕСАТ}}")
    paragraph.add_run("\nРапорт от {{ФИО}}")
    table = document.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "Место: {{ГОРОД}}"
    document.save(template)
    original = template.read_bytes()

    artifact = create_legal_draft(
        template,
        "{{АДРЕСАТ=Начальнику отдела}} {{ФИО=Тестов Тест Тестович}} {{ГОРОД=Тула}}",
        tmp_path / "drafts",
        lambda _prompt: (_ for _ in ()).throw(AssertionError("LLM must not be used")),
    )

    assert artifact.can_save, artifact.validation_errors
    assert template.read_bytes() == original
    result = Document(artifact.draft_path)
    text = "\n".join([p.text for p in result.paragraphs] + [cell.text for t in result.tables for row in t.rows for cell in row.cells])
    assert "Тестов Тест Тестович" in text
    assert "Тула" in text
    assert "{{" not in text
    assert zipfile.is_zipfile(artifact.draft_path)


def test_command_trip_keeps_first_form_and_removes_completed_examples(tmp_path):
    template = tmp_path / "reports.docx"
    document = Document()
    document.add_paragraph("Начальнику старого отдела Старову С. С.")
    document.add_paragraph("РАПОРТ")
    document.add_paragraph("Прошу командировать Иванова Ивана Ивановича в Москву с 01.01.2020 по 02.01.2020.")
    signature = document.add_table(rows=1, cols=2)
    signature.cell(0, 0).text = "20.12.2019"
    signature.cell(0, 1).text = "Иванов Иван Иванович"
    document.add_paragraph("РАПОРТ")
    document.add_paragraph("Прошу командировать Петрова Петра Петровича в Казань с 03.02.2021 по 04.02.2021.")
    old_signature = document.add_table(rows=1, cols=2)
    old_signature.cell(0, 0).text = "01.02.2021"
    old_signature.cell(0, 1).text = "Петров Петр Петрович"
    document.save(template)
    original = template.read_bytes()

    request = (
        "Составь рапорт на командировку для Тестова Теста Тестовича, преподавателя, "
        "в город Тула с 01.10.2026 по 03.10.2026 для участия в учебных занятиях. "
        "Адресат: начальнику учебного отдела. Автор: Тестов Тест Тестович, преподаватель, "
        "дата подписи 20.09.2026."
    )
    artifact = create_legal_draft(
        template,
        request,
        tmp_path / "drafts",
        lambda _prompt: (_ for _ in ()).throw(AssertionError("LLM must not be used")),
    )

    assert artifact.can_save, artifact.validation_errors
    assert template.read_bytes() == original
    result = Document(artifact.draft_path)
    text = "\n".join(
        [p.text for p in result.paragraphs]
        + [cell.text for table in result.tables for row in table.rows for cell in row.cells]
    )
    assert "Тестова Теста Тестовича" in text
    assert "Тула" in text
    assert "с 01.10.2026 по 03.10.2026 для участия в учебных занятиях" in text
    assert "20.09.2026" in text
    assert "Иванов" not in text
    assert "Петров" not in text
    assert "Москва" not in text
    assert "Казань" not in text
    assert len(result.tables) == 1
