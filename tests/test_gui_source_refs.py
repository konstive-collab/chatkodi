"""Source names parsed from an answer must match real file names."""

import pytest

pytest.importorskip("PyQt5")

from gui.main_window.messages import ChatKodiMessagesMixin  # noqa: E402


class _Window(ChatKodiMessagesMixin):
    pass


def test_underscores_in_file_names_survive_markdown_cleanup() -> None:
    text = (
        "**Выручка в августе 2025: всего 31 727 450 руб.**\n\n"
        "📚 **Источники:** report_0801.docx, Отчет филиала №46.txt и ещё 16\n"
        "📚 **Использованные документы:**\n"
        "1. **Ежедневный отчет 04.08.2025.txt** (релевантность: 1.00, 1,334 символов)"
    )
    names = [ref["name"] for ref in _Window()._extract_source_refs_from_message(text)]
    assert "report_0801.docx" in names
    assert "Отчет филиала №46.txt" in names
    assert "Ежедневный отчет 04.08.2025.txt" in names
