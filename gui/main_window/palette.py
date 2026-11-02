"""Ctrl+K command palette and the window's keyboard shortcuts."""

from __future__ import annotations

from pathlib import Path
from typing import List

from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut

from core.chat_mode import ChatMode
from gui.components.command_palette import CommandPalette, PaletteEntry, matches


class ChatKodiPaletteMixin:
    def _install_shortcuts(self) -> None:
        for keys, handler in (
            ("Ctrl+K", self._open_palette),
            ("Ctrl+N", self._new_chat),
            ("Ctrl+M", self._shortcut_voice),
            ("Ctrl+,", lambda: self._switch_workspace("settings")),
        ):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.activated.connect(handler)

    def _shortcut_voice(self) -> None:
        if hasattr(self, "voice_btn"):
            self.voice_btn.click()

    def _open_palette(self) -> None:
        palette = getattr(self, "_palette", None)
        if palette is None:
            palette = CommandPalette(self, self._palette_entries)
            self._palette = palette
        palette.open_centered()

    def _palette_actions(self) -> List[PaletteEntry]:
        voice_on = bool(getattr(self, "voice_reply_enabled", False))
        dialog_on = bool(getattr(self, "voice_dialog_enabled", False))
        entries = [
            PaletteEntry("Действия", "Новый чат", self._new_chat, "Ctrl+N", "создать начать"),
            PaletteEntry("Действия", "Приложить файл или фото", self._choose_attachments, "Ctrl+O", "скан pdf документ вложение"),
            PaletteEntry("Действия", "Сказать голосом", self._shortcut_voice, "Ctrl+M", "микрофон речь"),
            PaletteEntry("Действия", "Озвучивать ответы: " + ("выключить" if voice_on else "включить"),
                         lambda: self.speak_btn.setChecked(not self.speak_btn.isChecked()), "", "голос звук"),
            PaletteEntry("Действия", "Голосовой диалог: " + ("выключить" if dialog_on else "включить"),
                         lambda: self.voice_dialog_action.setChecked(not self.voice_dialog_action.isChecked()), "", "голос разговор"),
            PaletteEntry("Настройки", "Настройки модели и ассистента", lambda: self._switch_workspace("settings"), "Ctrl+,", "температура контекст видеокарта голос"),
            PaletteEntry("Настройки", "Личные инструкции", self._edit_personal_instructions, "", "подпись стиль"),
            PaletteEntry("Настройки", "Память ассистента", self._edit_assistant_memory, "", "запомнил факты забыть"),
            PaletteEntry("Настройки", "Папки для поиска файлов", self._edit_search_folders, "", "каталог директория найти"),
            PaletteEntry("Настройки", "Тема: классическая", lambda: self._set_theme("classic"), "", "оформление цвет"),
            PaletteEntry("Настройки", "Тема: тёмная", lambda: self._set_theme("dark"), "", "оформление цвет"),
            PaletteEntry("Настройки", "Тема: светлая", lambda: self._set_theme("light"), "", "оформление цвет"),
            PaletteEntry("Библиотека", "Документы библиотеки", lambda: self._switch_workspace("documents"), "", "файлы папка"),
            PaletteEntry("Библиотека", "Добавить документы в библиотеку", self._add_documents, "", "загрузить"),
            PaletteEntry("Библиотека", "Обновить индекс", self._update_index, "", "переиндексировать база"),
            PaletteEntry("Библиотека", "Открыть папку созданных файлов", self._open_outputs_folder, "", "результаты docx xlsx"),
            PaletteEntry("Чат", "Удалить текущий чат", lambda: self._delete_chat(self.current_chat_id), "", "очистить"),
        ]
        return entries

    def _palette_entries(self, query: str) -> List[PaletteEntry]:
        actions = self._palette_actions()
        if not query.strip():
            frequent = [e for e in actions if e.title in {"Новый чат", "Приложить файл или фото", "Сказать голосом", "Настройки модели и ассистента"}]
            chats = [
                PaletteEntry("Последние чаты", info.title, lambda cid=info.id: self._switch_chat(cid))
                for info in self.chat_sessions.ordered()[:5]
            ]
            return frequent + chats
        found = [e for e in actions if matches(e, query)]
        found += [
            PaletteEntry("Чаты", info.title, lambda cid=info.id: self._switch_chat(cid))
            for info in self.chat_sessions.ordered(query)[:6]
        ]
        processor = getattr(self, "doc_processor", None)
        documents = list(getattr(processor, "documents", None) or [])
        words = query.lower().replace("ё", "е").split()
        docs = [d for d in documents if all(w in str(getattr(d, "name", "")).lower().replace("ё", "е") for w in words)][:6]
        found += [
            PaletteEntry("Документы", str(d.name), lambda d=d: self._show_source_document_in_app(Path(d.path), str(d.name)))
            for d in docs
        ]
        return found
