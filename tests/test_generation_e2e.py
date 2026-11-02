from pathlib import Path

from docx import Document

from core.document_catalog import DocumentCatalog
from core.generation.service import GenerationService


class FakeLLM:
    def is_available(self):
        return True

    def generate_clean_answer(self, _prompt, **_kwargs):
        return "Основано на документах\nБег.\n\nДополнение модели\nПлавание."


class BrokenLLM(FakeLLM):
    def generate_clean_answer(self, _prompt, **_kwargs):
        return '{"grounded":"обрыв'


def _write_docx(path: Path, text: str):
    document = Document()
    document.add_paragraph(text)
    document.save(path)


def test_training_generation_uses_local_source_and_labels_addition(tmp_path):
    _write_docx(tmp_path / "Методика ФП.docx", "Физическая подготовка: бег и упражнения.")
    service = GenerationService(FakeLLM(), {"catalog_root": str(tmp_path)})
    service.catalog = DocumentCatalog(tmp_path)
    request = (
        "Составь план тренировок на месяц для развития выносливости: взрослый, "
        "средний уровень, 3 раза в неделю, 60 минут, стадион и турник, ограничений нет"
    )
    result = service.handle(request)
    assert result is not None
    assert "Основано на документах" in result.text
    assert "Дополнение модели" in result.text
    assert "Методика ФП.docx" in result.sources


def test_broken_sport_json_returns_usable_labeled_calendar(tmp_path):
    _write_docx(tmp_path / "Методика ФП.docx", "Физическая подготовка: бег и упражнения.")
    service = GenerationService(BrokenLLM(), {"catalog_root": str(tmp_path)})
    service.catalog = DocumentCatalog(tmp_path)
    result = service.handle(
        "Составь план тренировок на месяц для развития выносливости: взрослый, "
        "средний уровень, 3 раза в неделю, 60 минут, стадион, турник и бассейн, ограничений нет"
    )
    assert result is not None
    assert "Неделя 4" in result.text
    assert "создан программой, не является цитатой" in result.text
    assert "обрыв" not in result.text


def test_legal_request_creates_saveable_copy_from_placeholder_template(tmp_path):
    template = tmp_path / "Рапорт на командировку.docx"
    _write_docx(template, "Кому: {{АДРЕСАТ}}\nОт: {{ФИО}}\nГород: {{ГОРОД}}")
    original = template.read_bytes()
    service = GenerationService(FakeLLM(), {"catalog_root": str(tmp_path)})
    service.catalog = DocumentCatalog(tmp_path)
    request = (
        "Составь рапорт на командировку для Тестова: в город Тула с 01.10.2026 по 03.10.2026, "
        "цель для участия в занятиях, адресат начальник отдела, от автора в должности преподавателя. "
        "{{АДРЕСАТ=Начальнику отдела}} {{ФИО=Тестов Тест Тестович}} {{ГОРОД=Тула}}"
    )
    result = service.handle(request)
    assert result is not None
    assert result.artifact is not None
    assert result.artifact.can_save, result.artifact.validation_errors
    destination = tmp_path / "result.docx"
    service.save_artifact(result.artifact.artifact_id, destination)
    assert destination.is_file()
    assert template.read_bytes() == original
