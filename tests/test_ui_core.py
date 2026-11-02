"""Logic behind the 2.0 window that runs without a screen: chats, settings,
paper paragraphs, Ctrl+K matching, quote variants."""

from __future__ import annotations

import os

from core.chat_history import ChatHistoryStore
from core.chat_sessions import ChatSessionStore, title_from_text


def test_title_from_text():
    assert title_from_text("проверь протокол по нормативам") == "Проверь протокол по нормативам"
    assert title_from_text("📎 C:\\inbox\\schet_318.jpg") == "schet_318.jpg"
    assert title_from_text("Сделай таблицу\n\n📎 C:\\inbox\\schet_318.jpg") == "Сделай таблицу"
    long = title_from_text("очень длинный вопрос " * 10)
    assert long.endswith("…") and len(long) <= 44


def test_chat_sessions_lifecycle(tmp_path):
    store = ChatSessionStore(tmp_path / "chats")
    first = store.create()
    store.history(first.id).append("user", "Какая выручка 4 августа?")
    store.touch(first.id, first_text="Какая выручка 4 августа?")
    second = store.create()
    store.touch(second.id, status="warn")
    assert store.chats[first.id].title == "Какая выручка 4 августа"
    assert store.ordered()[0].id == second.id  # most recent first
    store.set_pinned(first.id, True)
    assert store.ordered()[0].id == first.id  # pinned on top
    assert [c.id for c in store.ordered("выручка")] == [first.id]  # search in text
    store.rename(first.id, "Выручка")
    store.touch(first.id, first_text="другое")
    assert store.chats[first.id].title == "Выручка"  # a manual title stays
    reloaded = ChatSessionStore(tmp_path / "chats")
    assert reloaded.chats[second.id].status == "warn"
    reloaded.delete(second.id)
    assert second.id not in ChatSessionStore(tmp_path / "chats").chats


def test_legacy_history_becomes_first_chat(tmp_path):
    legacy = ChatHistoryStore(tmp_path / "chat_history.jsonl")
    legacy.append("user", "Сколько стоит датчик?")
    legacy.append("assistant", "7300")
    store = ChatSessionStore(tmp_path / "chats")
    info = store.import_legacy(legacy)
    assert info is not None and info.title.startswith("Сколько стоит датчик")
    assert len(store.history(info.id).load()) == 2
    assert store.import_legacy(legacy) is None  # only once


def test_settings_roundtrip_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATKODI_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("CHATKODI_FORCE_CPU", raising=False)
    monkeypatch.delenv("CHATKODI_DISABLE_VISION", raising=False)
    from core.agent import settings as module

    monkeypatch.setattr(module, "runtime_root", lambda: tmp_path)
    module.apply_to_environment(module.load_settings())  # nothing saved: nothing changes
    assert "CHATKODI_FORCE_CPU" not in os.environ
    s = module.load_settings()
    s.device, s.vision, s.context_tokens, s.answer_length = "cpu", False, 8192, "long"
    module.save_settings(s)
    loaded = module.load_settings()
    assert loaded.device == "cpu" and loaded.length_factor == 1.6
    module.apply_to_environment(loaded)
    assert os.environ["CHATKODI_FORCE_CPU"] == "1" and os.environ["CHATKODI_DISABLE_VISION"] == "1"
    assert os.environ["CHATKODI_LLM_CTX"] == "8192"


def test_paper_paragraphs_keep_tables_and_lists():
    from gui.components.paper import paragraphs

    md = "# Заключение\nДата: 26.10.2026\nОбъект: дом\n| a | b |\n|---|---|\n| 1 | 2 |\n- пункт 1\n- пункт 2"
    out = paragraphs(md)
    assert "Дата: 26.10.2026\n\nОбъект: дом" in out
    assert "| a | b |\n|---|---|\n| 1 | 2 |" in out
    assert "- пункт 1\n- пункт 2" in out


def test_palette_matching():
    from gui.components.command_palette import PaletteEntry, matches

    entry = PaletteEntry("Настройки", "Тема: тёмная", lambda: None, "", "оформление цвет")
    assert matches(entry, "тема темн") and matches(entry, "цвет")
    assert not matches(entry, "голос")


def test_quote_variants():
    from gui.components.source_pane import _variants

    variants = _variants("«Для бетона класса B30 средняя прочность серии должна быть не менее 39,2 МПа.»")
    assert variants[0].startswith("Для бетона класса B30")
    assert any(v.endswith("39,2 МПа") for v in variants)
