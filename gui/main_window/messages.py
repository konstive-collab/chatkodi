"""Обработка пользовательских сообщений и streaming-ответов."""

from __future__ import annotations

import threading
import time
import re
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, QTimer, QRect, Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QTextCursor
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
    QSizePolicy,
    QWidget,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QLineEdit,
    QFileDialog,
)
from core.chat_mode import ChatMode
from core.logging_utils import get_logger
from core.theme_manager import theme_palette
from gui.components.motion import SmoothScroller, TypingIndicator, fade_in, pulse
from gui.components.rich_text import markdown_to_html, plain_to_html
from gui.workers.streaming import StreamingWorker

_MODE_TAGS = {ChatMode.RAG: "по документам", ChatMode.CHAT: "свободный чат", ChatMode.EXTRACT: "поиск фрагментов", ChatMode.AGENT: "ассистент"}

logger = get_logger(__name__)


class ChatKodiMessagesMixin:
    def _record_chat_message(self, role: str, text: str) -> None:
        clean = str(text or "").strip()
        if not clean:
            return
        mode = getattr(getattr(self, "current_mode", None), "value", "")
        entry = {"role": role, "text": clean, "mode": mode}
        manager = getattr(self, "chat_manager", None)
        if manager is not None:
            manager.chat_history.append(entry)
        if getattr(self, "chat_history_enabled", True):
            try:
                stored = self.chat_history_store.append(role, clean, mode)
                if stored is not None and manager is not None:
                    manager.chat_history[-1] = stored
            except OSError as exc:
                logger.warning("Failed to save chat history: %s", exc)
        if hasattr(self, "_chat_activity") and not getattr(self, "_restoring_chat_history", False):
            try:
                self._chat_activity(role, clean)
            except Exception as exc:
                logger.debug("Chat list update failed: %s", exc)

    def _restore_chat_history(self) -> bool:
        if not getattr(self, "chat_history_enabled", True):
            return False
        entries = self.chat_history_store.load()
        if not entries:
            return False
        manager = getattr(self, "chat_manager", None)
        if manager is not None:
            manager.chat_history = list(entries)
        self._restoring_chat_history = True
        try:
            for entry in entries:
                sender = "user" if entry.get("role") == "user" else "ai"
                self._add_message(str(entry.get("text") or ""), sender=sender)
        finally:
            self._restoring_chat_history = False
        return True

    def _export_chat_history(self) -> None:
        suggested = str(Path.home() / "ChatKodi_dialog.txt")
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт диалога",
            suggested,
            "Текстовый файл (*.txt);;JSON (*.json)",
        )
        if not target:
            return
        try:
            entries = list(getattr(getattr(self, "chat_manager", None), "chat_history", []) or [])
            saved = self.chat_history_store.export(target, entries=entries)
            self.status.showMessage(f"Диалог сохранён: {saved}")
        except OSError as exc:
            QMessageBox.critical(self, "Ошибка экспорта", f"Не удалось сохранить диалог: {exc}")

    def _clear_chat_history(self) -> None:
        reply = QMessageBox.question(
            self,
            "Очистить историю",
            "Удалить сохранённую историю диалога на этом компьютере?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            self.chat_history_store.clear()
        except OSError as exc:
            logger.warning("Failed to clear chat history: %s", exc)
            QMessageBox.critical(
                self,
                "Ошибка истории",
                f"Не удалось очистить сохранённую историю: {exc}",
            )
            return
        manager = getattr(self, "chat_manager", None)
        if manager is not None:
            manager.chat_history.clear()
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._show_welcome_message()
        self.status.showMessage("История диалога очищена")

    def _message_width_bounds(self, sender: str, text: str = "", is_welcome: bool = False) -> tuple[int, int]:
        viewport_width = 1180
        if hasattr(self, "chat_scroll") and self.chat_scroll is not None:
            try:
                viewport_width = max(760, self.chat_scroll.viewport().width() - 48)
            except Exception:
                viewport_width = 1180

        if is_welcome:
            max_width = min(860, int(viewport_width * 0.82))
            return min(max_width, 640), max_width
        # A bubble is as wide as its text (a short request is a short bubble),
        # long answers wrap at a comfortable reading width.
        if sender == "user":
            max_width = min(720, int(viewport_width * 0.66))
        else:
            max_width = min(880, int(viewport_width * 0.80))
        natural = self._natural_text_width(text) + 56 + 90  # padding, rich-text margins, the bubble's outer margins
        floor = 150 if sender == "user" else 220
        min_width = max(floor, min(max_width, natural))
        return min_width, max_width

    def _natural_text_width(self, text: str) -> int:
        """Width of the longest line of the message as it will be shown."""
        from PyQt5.QtGui import QFont, QFontMetrics

        from gui.components.fonts import ui_family

        clean = re.sub(r"\*\*|`|^#+\s*|^>\s*", "", text or "", flags=re.M)
        lines = [line for line in clean.splitlines() if line.strip()]
        if not lines:
            return 0
        if any(line.lstrip().startswith("|") for line in lines):
            return 10_000  # tables take the full width
        metrics = QFontMetrics(QFont(ui_family(), 10))
        return int(max(metrics.horizontalAdvance(line) for line in lines) * 1.12)

    def _refresh_message_widths(self) -> None:
        if not hasattr(self, "chat_layout"):
            return
        for i in range(self.chat_layout.count() - 1):
            item = self.chat_layout.itemAt(i)
            row = item.widget()
            if row is None:
                continue
            bubble = row.property("messageBubble")
            sender = row.property("messageSender") or "ai"
            welcome = bool(row.property("messageWelcome"))
            original_text = row.property("messageText") or ""
            if bubble is not None:
                min_width, max_width = self._message_width_bounds(sender, str(original_text), welcome)
                bubble.setMinimumWidth(min_width)
                bubble.setMaximumWidth(max_width)

    def _clean_source_quote(self, quote: str) -> str:
        """Нормализует короткую цитату источника для внутреннего поиска."""
        quote = quote or ""
        quote = re.sub(r"[`*_]+", "", quote)
        quote = quote.replace("…", "...")
        quote = re.sub(r"^цитата\s*[:—-]\s*", "", quote, flags=re.IGNORECASE).strip()
        quote = quote.strip(" \t\r\n.,;:—-")
        quote = re.sub(r"\s+", " ", quote)
        if len(quote) > 520:
            quote = quote[:520].rstrip() + "..."
        return quote

    def _extract_source_refs_from_message(self, text: str) -> list[dict[str, str]]:
        """Достаёт источники ответа и возможные цитаты рядом с ними.

        Возвращаем не только имя файла, но и короткую цитату после двоеточия,
        если она есть в ответе. Эта цитата нужна кнопке «Показать»: ChatKodi
        открывает документ во внутреннем просмотрщике и пытается сразу выделить
        тот фрагмент, на котором держался ответ.
        """
        raw = text or ""
        refs: list[dict[str, str]] = []
        by_key: dict[str, dict[str, str]] = {}
        ext = r"docx|doc|pdf|txt|md|markdown|csv|tsv|rtf|html|htm|xhtml|xlsx|xlsm|xls|pptx|ppt|odt|ods|odp|epub|json|jsonl|xml|yaml|yml|log"

        def add(value: str, quote: str = "") -> None:
            # Strip Markdown markers only around the name: «report_0801.docx»
            # must keep its underscore or the file cannot be found.
            value = (value or "").replace("**", "").replace("`", "")
            value = value.strip(" ,.;:-—–\t\r\n")
            value = re.sub(r"^[*_]+|[*_]+$", "", value)
            value = re.sub(r"\s*\((?:релевантность|relevance)[^)]*\)\s*$", "", value, flags=re.IGNORECASE)
            value = value.strip(" ,.;:-—–\t\r\n()«»\"'_*")
            value = re.sub(r"^📄\s*", "", value).strip()
            if not value or len(value) > 180:
                return
            key = value.replace('\\', '/').strip().lower()
            clean_quote = self._clean_source_quote(quote)
            if key in by_key:
                # Если первый формат дал только имя, а строка с доказательством
                # дала цитату позже — сохраняем более полезную версию.
                if clean_quote and not by_key[key].get("quote"):
                    by_key[key]["quote"] = clean_quote
                return
            item = {"name": value, "quote": clean_quote}
            by_key[key] = item
            refs.append(item)

        # Компактный формат: "📚 Источники: 10.docx, 11.docx и ещё 2".
        for m in re.finditer(r"(?:📚\s*)?Источники:\s*(.+)", raw, flags=re.IGNORECASE):
            line = m.group(1).split("\n", 1)[0]
            line = re.sub(r"\s+и\s+ещ[её]\s+\d+.*$", "", line, flags=re.IGNORECASE)
            for part in line.split(','):
                add(part)

        # Строки вида: "1. 15.docx: `цитата`" или "- 15.docx".
        line_re = re.compile(
            rf"^\s*(?:[-•*]|\d+[.)])\s+([^:\n]{{1,180}}\.(?:{ext}))(?:\s*:\s*(.*))?$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        for m in line_re.finditer(raw):
            add(m.group(1), m.group(2) or "")

        # Markdown-жирный: "1. **15.docx**: `цитата`".
        bold_re = re.compile(
            rf"^\s*\d+[.)]\s+\*\*([^*\n]{{1,180}}\.(?:{ext}))\*\*(?:\s*:\s*(.*))?$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        for m in bold_re.finditer(raw):
            add(m.group(1), m.group(2) or "")

        # Финальный мягкий проход по всем похожим file.ext в тексте. Лишнее
        # отсечётся в _resolve_source_path_for_button, если файла нет в корпусе.
        generic_re = re.compile(rf"(?<![\w.-])([A-Za-zА-Яа-яЁё0-9 _()№#.+\-\\/]+\.(?:{ext}))(?![\w.-])", flags=re.IGNORECASE)
        for m in generic_re.finditer(raw):
            add(Path(m.group(1).replace('\\', '/')).name)

        return refs[:24]

    def _extract_source_names_from_message(self, text: str) -> list[str]:
        """Совместимый helper: возвращает только имена документов."""
        return [ref["name"] for ref in self._extract_source_refs_from_message(text)]

    def _resolve_source_path_for_button(self, source_name: str) -> Path | None:
        """Находит физический файл документа по имени из ответа."""
        manager = getattr(self, "chat_manager", None)
        search = getattr(manager, "hybrid_search", None)
        processor = getattr(search, "doc_processor", None)
        if processor is None:
            return None
        query = (source_name or "").replace('\\', '/').strip().lower()
        if not query:
            return None
        docs = list(getattr(processor, "documents", []) or [])
        for doc in docs:
            candidates = [
                str(getattr(doc, "name", "") or ""),
                str(getattr(doc, "filename", "") or ""),
                str(getattr(doc, "path", "") or ""),
            ]
            norm_candidates = [c.replace('\\', '/').strip().lower() for c in candidates]
            if query in norm_candidates or any(c.endswith('/' + query) for c in norm_candidates):
                path = Path(str(getattr(doc, "path", "") or ""))
                return path if path.exists() else None
        # Мягкий fallback: source может быть относительным путём внутри docs_folder.
        docs_folder = getattr(processor, "docs_folder", None)
        if docs_folder is not None:
            path = Path(docs_folder) / source_name
            if path.exists():
                return path
        # 1.9 generation catalog can reference an external read-only corpus
        # which is intentionally not the active RAG folder.
        generation = getattr(manager, "generation_service", None)
        catalog = getattr(generation, "catalog", None)
        for record in list(getattr(catalog, "records", []) or []):
            candidates = {
                str(getattr(record, "name", "") or "").replace('\\', '/').lower(),
                str(getattr(record, "path", "") or "").replace('\\', '/').lower(),
            }
            if query in candidates or any(value.endswith('/' + query) for value in candidates):
                path = Path(str(getattr(record, "path", "") or ""))
                return path if path.exists() else None
        return None

    def _find_document_for_source_path(self, path: Path, source_name: str = ""):
        """Возвращает объект Document для внутреннего просмотра источника."""
        processor = getattr(self, "doc_processor", None)
        if processor is None:
            manager = getattr(self, "chat_manager", None)
            search = getattr(manager, "hybrid_search", None)
            processor = getattr(search, "doc_processor", None)
        if processor is None:
            return None

        docs = list(getattr(processor, "documents", []) or [])
        source_query = (source_name or "").replace('\\', '/').strip().lower()
        try:
            target = Path(path).resolve()
        except Exception:
            target = Path(path)

        for doc in docs:
            doc_path_raw = str(getattr(doc, "path", "") or "")
            try:
                doc_path = Path(doc_path_raw).resolve()
            except Exception:
                doc_path = Path(doc_path_raw)
            if doc_path == target:
                return doc

        if source_query:
            for doc in docs:
                candidates = [
                    str(getattr(doc, "name", "") or ""),
                    str(getattr(doc, "filename", "") or ""),
                    str(getattr(doc, "path", "") or ""),
                ]
                norm_candidates = [c.replace('\\', '/').strip().lower() for c in candidates]
                if source_query in norm_candidates or any(c.endswith('/' + source_query) for c in norm_candidates):
                    return doc
        return None

    def _open_source_path(self, path: Path) -> None:
        """Открывает уже разрешённый путь системным приложением."""
        if not path or not path.exists():
            QMessageBox.information(self, "Источник не найден", "Не удалось найти файл источника.")
            return
        ok = QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        if not ok:
            QMessageBox.warning(self, "Не удалось открыть", f"Система не открыла файл:\n{path}")

    def _open_source_document(self, source_name: str) -> None:
        """Открывает документ системным приложением Windows/Linux/macOS."""
        path = self._resolve_source_path_for_button(source_name)
        if not path:
            QMessageBox.information(self, "Источник не найден", f"Не удалось найти файл: {source_name}")
            return
        self._open_source_path(path)

    def _source_focus_terms(self, quote: str) -> list[str]:
        """Готовит варианты поиска цитаты в QTextEdit.

        Ответ может содержать цитату в упрощённом виде: без переносов, с
        многоточием или с обрезанным хвостом. Поэтому сначала пробуем точную
        строку, затем более короткие устойчивые куски.
        """
        clean = self._clean_source_quote(quote)
        if not clean:
            return []
        terms: list[str] = []

        def push(term: str) -> None:
            term = self._clean_source_quote(term)
            if len(term) < 12:
                return
            if term not in terms:
                terms.append(term)

        push(clean)
        for part in re.split(r"\.\.\.|…|\n", clean):
            push(part)
        if len(clean) > 160:
            push(clean[:160])
            push(clean[:100])
        words = re.findall(r"[A-Za-zА-Яа-яЁё0-9№#\-]+", clean)
        if len(words) >= 6:
            push(" ".join(words[:10]))
            push(" ".join(words[:7]))
        return terms[:8]

    def _focus_source_quote_in_preview(self, quote: str) -> bool:
        """Пытается выделить цитату во внутреннем просмотрщике документа."""
        edit = getattr(self, "doc_preview_text", None)
        if edit is None:
            return False
        cursor = edit.textCursor()
        cursor.movePosition(QTextCursor.Start)
        edit.setTextCursor(cursor)
        for term in self._source_focus_terms(quote):
            if edit.find(term):
                edit.setFocus(Qt.OtherFocusReason)
                return True
        cursor = edit.textCursor()
        cursor.movePosition(QTextCursor.Start)
        edit.setTextCursor(cursor)
        return False

    def _show_source_document_in_app(self, path: Path, source_name: str = "", quote: str = "") -> None:
        """Показывает источник внутри ChatKodi, не открывая внешнюю программу.

        Пользователь остаётся в приложении: открывается вкладка «Документы»,
        выбирается нужный файл, а если в ответе была цитата после имени файла —
        ChatKodi пытается найти и выделить её в предпросмотре.
        """
        if not path or not Path(path).exists():
            QMessageBox.information(self, "Источник не найден", "Не удалось найти файл источника.")
            return

        doc = self._find_document_for_source_path(Path(path), source_name)
        if doc is None:
            QMessageBox.information(
                self,
                "Источник не найден в индексе",
                "Файл существует, но не найден среди загруженных документов корпуса.\n"
                "Попробуйте обновить индекс.",
            )
            return

        self._switch_workspace("documents")
        self.current_document = doc

        # Выделяем документ в списке слева. Сигналы временно блокируем, чтобы
        # не делать двойной рендер: ниже всё равно явно обновляем предпросмотр.
        selected = False
        if hasattr(self, "documents_list"):
            self.documents_list.blockSignals(True)
            try:
                for row in range(self.documents_list.count()):
                    item = self.documents_list.item(row)
                    if item and item.data(Qt.UserRole) == getattr(doc, "id", None):
                        self.documents_list.setCurrentRow(row)
                        selected = True
                        break
            finally:
                self.documents_list.blockSignals(False)

        self._render_document_preview(doc)
        self._update_document_inspector(doc)
        if self.inspector_visible:
            self._set_inspector_context("document")

        found_quote = self._focus_source_quote_in_preview(quote)
        doc_name = getattr(doc, "filename", None) or getattr(doc, "name", "источник")
        if quote and found_quote:
            self.status.showMessage(f"Показан источник и выделен фрагмент: {doc_name}")
        elif quote:
            self.status.showMessage(f"Показан источник: {doc_name}. Точная цитата не найдена в предпросмотре.")
        else:
            self.status.showMessage(f"Показан источник: {doc_name}")

    def _show_source_side(self, path: Path, source_name: str = "", quote: str = "", answer_text: str = "") -> None:
        """The source opens beside the chat with the quoted places highlighted."""
        pane = getattr(self, "source_pane", None)
        doc = self._find_document_for_source_path(Path(path), source_name) if path else None
        if pane is None or doc is None:
            self._show_source_document_in_app(path, source_name, quote)
            return
        quotes = [quote] if quote else []
        # every «…» quote of the answer that comes from this document
        quotes += [q for q in re.findall(r"«([^«»]{12,400})»", answer_text or "") if q not in quotes]
        title = getattr(doc, "filename", None) or getattr(doc, "name", Path(path).name)
        found = pane.show_document(Path(path), str(title), str(getattr(doc, "content", "") or ""), quotes)
        total = max(1, self.main_splitter.width())
        pane.setMinimumWidth(360)
        sizes = self.main_splitter.sizes()
        self.main_splitter.setSizes([int(total * 0.6), int(total * 0.4), sizes[2] if len(sizes) > 2 else 0])
        if hasattr(self, "_refresh_message_widths"):
            QTimer.singleShot(50, self._refresh_message_widths)
        if hasattr(self, "_fit_chat_bar"):
            QTimer.singleShot(60, self._fit_chat_bar)
        self.status.showMessage(f"Источник: {title}" + (f" · подсвечено цитат: {found}" if found else ""))

    def _collect_resolved_sources(self, text: str) -> list[dict]:
        """Возвращает уникальные источники ответа с путями и цитатами."""
        resolved: list[dict] = []
        seen_paths: set[str] = set()
        for ref in self._extract_source_refs_from_message(text):
            name = ref.get("name", "")
            path = self._resolve_source_path_for_button(name)
            if not path:
                continue
            key = str(path.resolve()).lower() if path.exists() else str(path).lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            resolved.append({"name": name, "path": path, "quote": ref.get("quote", "")})
        return resolved[:12]

    def _toggle_source_dropdown(self, checked: bool, button: QPushButton, panel: QFrame, count: int) -> None:
        """Открывает/закрывает выпадающий список источников мягкой анимацией."""
        checked = bool(checked)
        button.setText(("▾" if checked else "▸") + f" Источники ответа • {count}")

        if not hasattr(self, "_source_dropdown_animations"):
            self._source_dropdown_animations = []

        # На слабых машинах это почти ничего не стоит: анимируем только высоту
        # маленького блока источников, не перерисовывая весь чат.
        panel.setVisible(True)
        target_height = max(1, panel.sizeHint().height()) if checked else 0
        start_height = panel.maximumHeight()
        if start_height <= 0 and checked:
            start_height = 0
        elif start_height <= 0:
            start_height = max(1, panel.sizeHint().height())

        anim = QPropertyAnimation(panel, b"maximumHeight", self)
        anim.setDuration(170)
        anim.setEasingCurve(QEasingCurve.OutCubic if checked else QEasingCurve.InCubic)
        anim.setStartValue(start_height)
        anim.setEndValue(target_height)

        def finish() -> None:
            if checked:
                panel.setMaximumHeight(16777215)
            else:
                panel.setVisible(False)
            try:
                self._source_dropdown_animations.remove(anim)
            except Exception:
                pass
            QTimer.singleShot(0, self._scroll_to_bottom)

        anim.finished.connect(finish)
        self._source_dropdown_animations.append(anim)
        anim.start()
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _toggle_context_ask_panel(self, checked: bool, button: QPushButton, panel: QFrame) -> None:
        """Мягко раскрывает поле уточняющего вопроса по данным сообщения."""
        checked = bool(checked)
        button.setText(("▾" if checked else "▸") + " Спросить по данным")

        if not hasattr(self, "_context_ask_animations"):
            self._context_ask_animations = []

        panel.setVisible(True)
        target_height = max(1, panel.sizeHint().height()) if checked else 0
        start_height = panel.maximumHeight()
        if start_height <= 0 and checked:
            start_height = 0
        elif start_height <= 0:
            start_height = max(1, panel.sizeHint().height())

        anim = QPropertyAnimation(panel, b"maximumHeight", self)
        anim.setDuration(165)
        anim.setEasingCurve(QEasingCurve.OutCubic if checked else QEasingCurve.InCubic)
        anim.setStartValue(start_height)
        anim.setEndValue(target_height)

        def finish() -> None:
            if checked:
                panel.setMaximumHeight(16777215)
                edit = panel.property("askInput")
                if edit is not None:
                    try:
                        edit.setFocus(Qt.OtherFocusReason)
                    except Exception:
                        pass
            else:
                panel.setVisible(False)
            try:
                self._context_ask_animations.remove(anim)
            except Exception:
                pass
            QTimer.singleShot(0, self._scroll_to_bottom)

        anim.finished.connect(finish)
        self._context_ask_animations.append(anim)
        anim.start()
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _add_context_ask_block(self, layout: QVBoxLayout, answer_text: str) -> None:
        """Добавляет к ответу компактный блок «Спросить по данным».

        Это не память диалога и не новый поиск по корпусу. Кнопка берёт именно
        текст текущего ответа как временный контекст, поэтому пользователь может
        уточнить уже найденную сводку без повторного срабатывания всей RAG-цепочки.
        """
        if not (answer_text or "").strip():
            return

        wrapper = QFrame()
        wrapper.setObjectName("contextAskDropdown")
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 2, 0, 0)
        wrapper_layout.setSpacing(6)

        toggle = QPushButton("▸ Спросить по данным", wrapper)
        toggle.setObjectName("contextAskToggleButton")
        toggle.setCheckable(True)
        toggle.setCursor(Qt.PointingHandCursor)
        toggle.setToolTip("Задать уточняющий вопрос только по данным этого ответа")
        wrapper_layout.addWidget(toggle)

        panel = QFrame(wrapper)
        panel.setObjectName("contextAskPanel")
        panel.setVisible(False)
        panel.setMaximumHeight(0)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 8, 10, 8)
        panel_layout.setSpacing(7)

        hint = QLabel("Уточнение пойдёт только по этому ответу — без нового поиска по базе.", panel)
        hint.setObjectName("contextAskHintLabel")
        hint.setWordWrap(True)
        panel_layout.addWidget(hint)

        row = QHBoxLayout()
        row.setSpacing(8)
        ask_input = QLineEdit(panel)
        ask_input.setObjectName("contextAskInput")
        ask_input.setPlaceholderText("Например: какие дни были самыми насыщенными?")
        row.addWidget(ask_input, 1)

        ask_btn = QPushButton("Спросить", panel)
        ask_btn.setObjectName("contextAskSubmitButton")
        ask_btn.setCursor(Qt.PointingHandCursor)
        row.addWidget(ask_btn, 0)
        panel_layout.addLayout(row)
        panel.setProperty("askInput", ask_input)
        wrapper_layout.addWidget(panel)

        def submit() -> None:
            question = ask_input.text().strip()
            if not question:
                ask_input.setFocus(Qt.OtherFocusReason)
                return
            ask_btn.setEnabled(False)
            ask_btn.setText("Думаю…")
            toggle.setChecked(False)
            self._toggle_context_ask_panel(False, toggle, panel)
            self._on_context_ask_submit(question, answer_text)

        ask_btn.clicked.connect(submit)
        ask_input.returnPressed.connect(submit)
        toggle.clicked.connect(lambda checked=False, b=toggle, p=panel: self._toggle_context_ask_panel(checked, b, p))
        layout.addWidget(wrapper)

    def _on_context_ask_submit(self, question: str, answer_text: str) -> None:
        """Запускает уточняющий вопрос по тексту выбранного ответа."""
        question = (question or "").strip()
        if not question:
            return

        source_question = getattr(self, "last_user_message", "")
        self.last_user_message = question
        self.last_error_text = ""
        self._switch_workspace("chat")
        self._add_message(f"💬 Вопрос по данным: {question}", sender="user")
        self.status.showMessage("Отвечаю по данным выбранного сообщения…")

        # Уточнение работает через LLM, но не через новый поиск. Важно: НЕ
        # грузим LLM в GUI-потоке. На CPU/8 GB RAM крупная модель может грузиться
        # минутами, поэтому подготовка/отказ происходят только в worker-потоке.
        if hasattr(self, "_ensure_llm_ready"):
            try:
                self._ensure_llm_ready(eager=False, reprobe=False)
            except Exception as exc:
                logger.warning("Context ask lazy LLM prepare failed: %s", exc)

        thread = threading.Thread(
            target=self._worker_process_context_question,
            args=(question, answer_text, source_question),
            daemon=True,
        )
        thread.start()

    def _worker_process_context_question(self, question: str, answer_text: str, source_question: str = "") -> None:
        try:
            time.sleep(0.1)
            if not self.chat_manager or not hasattr(self.chat_manager, "answer_from_context"):
                raise RuntimeError("Менеджер чата не поддерживает уточнения по данным сообщения")
            answer = self.chat_manager.answer_from_context(question, answer_text, source_question=source_question)
            self.response_ready.emit(answer, "context")
        except Exception as exc:
            self.response_error.emit(str(exc))

    def _add_source_buttons(self, layout: QVBoxLayout, text: str) -> None:
        """Добавляет органичный зелёный выпадающий блок источников под ответом.

        Старые отдельные синие кнопки быстро превращали ответ в визуальный шум.
        Теперь источники по умолчанию свернуты: пользователь видит аккуратную
        строку "Источники ответа", нажимает её и получает список документов с
        кнопками открытия. Это лучше масштабируется на 5–10 источников и не
        ломает ширину чат-пузыря.
        """
        sources = self._collect_resolved_sources(text)
        if not sources:
            return

        wrapper = QFrame()
        wrapper.setObjectName("sourceDropdown")
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 6, 0, 0)
        wrapper_layout.setSpacing(6)

        toggle = QPushButton(f"▸ Источники ответа • {len(sources)}", wrapper)
        toggle.setObjectName("sourceToggleButton")
        toggle.setCheckable(True)
        toggle.setCursor(Qt.PointingHandCursor)
        toggle.setToolTip("Показать документы, на которых основан ответ")
        wrapper_layout.addWidget(toggle)

        panel = QFrame(wrapper)
        panel.setObjectName("sourcePanel")
        panel.setVisible(False)
        panel.setMaximumHeight(0)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 8, 10, 8)
        panel_layout.setSpacing(7)

        for source in sources:
            name = source.get("name", "")
            path = source.get("path")
            quote = source.get("quote", "")
            if path is None:
                continue

            row = QFrame(panel)
            row.setObjectName("sourceDocRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(9, 7, 8, 7)
            row_layout.setSpacing(8)

            label_text = Path(name).name or Path(path).name
            doc_label = QLabel(f"📄 {label_text}", row)
            doc_label.setObjectName("sourceDocLabel")
            doc_label.setToolTip(str(path))
            doc_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            row_layout.addWidget(doc_label, 1)

            show_btn = QPushButton("Показать", row)
            show_btn.setObjectName("sourceShowButton")
            show_btn.setCursor(Qt.PointingHandCursor)
            show_btn.setToolTip("Показать документ внутри ChatKodi")
            show_btn.clicked.connect(lambda _checked=False, p=Path(path), n=name, q=quote: self._show_source_side(p, n, q, self.last_answer_text))
            row_layout.addWidget(show_btn, 0)

            open_btn = QPushButton("Открыть", row)
            open_btn.setObjectName("sourceOpenButton")
            open_btn.setCursor(Qt.PointingHandCursor)
            open_btn.setToolTip(f"Открыть во внешнем приложении: {path}")
            open_btn.clicked.connect(lambda _checked=False, p=Path(path): self._open_source_path(p))
            row_layout.addWidget(open_btn, 0)
            panel_layout.addWidget(row)

        hint = QLabel("Показать — открыть внутри ChatKodi. Открыть — запустить файл во внешней программе.", panel)
        hint.setObjectName("sourceHintLabel")
        panel_layout.addWidget(hint)
        wrapper_layout.addWidget(panel)

        toggle.clicked.connect(lambda checked=False, b=toggle, p=panel, c=len(sources): self._toggle_source_dropdown(checked, b, p, c))
        layout.addWidget(wrapper)

    def _add_explain_button(self, layout: QVBoxLayout) -> None:
        """Добавляет компактную кнопку «Почему так?» под ответом."""
        explain = QPushButton("Почему такой ответ?", None)
        explain.setObjectName("explainAnswerButton")
        explain.setCursor(Qt.PointingHandCursor)
        explain.setToolTip("Открыть правую панель с разбором поиска, источниками и альтернативами")
        explain.clicked.connect(lambda _checked=False: self._show_search_reason_panel() if hasattr(self, "_show_search_reason_panel") else None)
        layout.addWidget(explain)

    def _create_source_panel(self, parent: QWidget, sources: list[dict]) -> QFrame:
        """Создаёт раскрываемую панель источников без отдельной верхней кнопки.

        Верхняя кнопка теперь живёт в общей строке действий ответа, а панель
        раскрывается ниже этой строки. Так блоки «Источники», «Спросить по
        данным» и «Почему такой ответ?» не складываются вертикальной лестницей.
        """
        panel = QFrame(parent)
        panel.setObjectName("sourcePanel")
        panel.setVisible(False)
        panel.setMaximumHeight(0)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 8, 10, 8)
        panel_layout.setSpacing(7)

        for source in sources:
            name = source.get("name", "")
            path = source.get("path")
            quote = source.get("quote", "")
            if path is None:
                continue

            row = QFrame(panel)
            row.setObjectName("sourceDocRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(9, 7, 8, 7)
            row_layout.setSpacing(8)

            label_text = Path(name).name or Path(path).name
            doc_label = QLabel(f"📄 {label_text}", row)
            doc_label.setObjectName("sourceDocLabel")
            doc_label.setToolTip(str(path))
            doc_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            row_layout.addWidget(doc_label, 1)

            show_btn = QPushButton("Показать", row)
            show_btn.setObjectName("sourceShowButton")
            show_btn.setCursor(Qt.PointingHandCursor)
            show_btn.setToolTip("Показать документ внутри ChatKodi")
            show_btn.clicked.connect(lambda _checked=False, p=Path(path), n=name, q=quote: self._show_source_side(p, n, q, self.last_answer_text))
            row_layout.addWidget(show_btn, 0)

            open_btn = QPushButton("Открыть", row)
            open_btn.setObjectName("sourceOpenButton")
            open_btn.setCursor(Qt.PointingHandCursor)
            open_btn.setToolTip(f"Открыть во внешнем приложении: {path}")
            open_btn.clicked.connect(lambda _checked=False, p=Path(path): self._open_source_path(p))
            row_layout.addWidget(open_btn, 0)
            panel_layout.addWidget(row)

        hint = QLabel("Показать — открыть внутри ChatKodi. Открыть — запустить файл во внешней программе.", panel)
        hint.setObjectName("sourceHintLabel")
        panel_layout.addWidget(hint)
        return panel

    def _create_context_ask_panel(self, parent: QWidget, answer_text: str, toggle: QPushButton) -> QFrame:
        """Создаёт панель уточняющего вопроса по текущему ответу."""
        panel = QFrame(parent)
        panel.setObjectName("contextAskPanel")
        panel.setVisible(False)
        panel.setMaximumHeight(0)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 8, 10, 8)
        panel_layout.setSpacing(7)

        hint = QLabel("Уточнение пойдёт только по этому ответу — без нового поиска по базе.", panel)
        hint.setObjectName("contextAskHintLabel")
        hint.setWordWrap(True)
        panel_layout.addWidget(hint)

        row = QHBoxLayout()
        row.setSpacing(8)
        ask_input = QLineEdit(panel)
        ask_input.setObjectName("contextAskInput")
        ask_input.setPlaceholderText("Например: какие дни были самыми насыщенными?")
        row.addWidget(ask_input, 1)

        ask_btn = QPushButton("Спросить", panel)
        ask_btn.setObjectName("contextAskSubmitButton")
        ask_btn.setCursor(Qt.PointingHandCursor)
        row.addWidget(ask_btn, 0)
        panel_layout.addLayout(row)
        panel.setProperty("askInput", ask_input)

        def submit() -> None:
            question = ask_input.text().strip()
            if not question:
                ask_input.setFocus(Qt.OtherFocusReason)
                return
            ask_btn.setEnabled(False)
            ask_btn.setText("Думаю…")
            toggle.setChecked(False)
            self._toggle_context_ask_panel(False, toggle, panel)
            self._on_context_ask_submit(question, answer_text)

        ask_btn.clicked.connect(submit)
        ask_input.returnPressed.connect(submit)
        return panel

    def _add_response_actions(self, layout: QVBoxLayout, text: str) -> None:
        """Добавляет одну аккуратную строку действий под ответом.

        Вместо вертикальной пачки кнопок теперь есть компактная строка:
        «Источники ответа», «Спросить по данным», «Почему такой ответ?». Если
        какая-то кнопка раскрывает блок, сам блок появляется ниже строки.
        """
        answer_text = (text or "").strip()
        if not answer_text or getattr(self, "_skip_response_actions", False):
            return

        sources = self._collect_resolved_sources(answer_text)
        wrapper = QFrame()
        wrapper.setObjectName("responseActions")
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 6, 0, 0)
        wrapper_layout.setSpacing(6)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        wrapper_layout.addLayout(row)

        added_any = False
        generation_result = None
        manager = getattr(self, "chat_manager", None)
        if manager is not None and hasattr(manager, "get_generation_result_for_text"):
            try:
                generation_result = manager.get_generation_result_for_text(answer_text)
            except Exception:
                generation_result = None
        artifact = getattr(generation_result, "artifact", None)
        if artifact is not None and bool(getattr(artifact, "can_save", False)):
            save_btn = QPushButton("Сохранить как DOCX", wrapper)
            save_btn.setObjectName("saveGeneratedDocxButton")
            save_btn.setCursor(Qt.PointingHandCursor)
            save_btn.setToolTip("Сохранить проверенную копию, не изменяя исходный образец")
            save_btn.clicked.connect(
                lambda _checked=False, aid=artifact.artifact_id, name=artifact.suggested_name: self._save_generated_docx(aid, name)
            )
            row.addWidget(save_btn, 0)
            added_any = True
        source_panel = None
        if sources:
            source_toggle = QPushButton(f"▸ Источники ответа • {len(sources)}", wrapper)
            source_toggle.setObjectName("sourceToggleButton")
            source_toggle.setCheckable(True)
            source_toggle.setCursor(Qt.PointingHandCursor)
            source_toggle.setToolTip("Показать документы, на которых основан ответ")
            source_panel = self._create_source_panel(wrapper, sources)
            source_toggle.clicked.connect(lambda checked=False, b=source_toggle, p=source_panel, c=len(sources): self._toggle_source_dropdown(checked, b, p, c))
            row.addWidget(source_toggle, 0)
            added_any = True

        context_toggle = QPushButton("▸ Спросить по данным", wrapper)
        context_toggle.setObjectName("contextAskToggleButton")
        context_toggle.setCheckable(True)
        context_toggle.setCursor(Qt.PointingHandCursor)
        context_toggle.setToolTip("Задать уточняющий вопрос только по данным этого ответа")
        context_panel = self._create_context_ask_panel(wrapper, answer_text, context_toggle)
        context_toggle.clicked.connect(lambda checked=False, b=context_toggle, p=context_panel: self._toggle_context_ask_panel(checked, b, p))
        row.addWidget(context_toggle, 0)
        added_any = True

        explain = QPushButton("Почему такой ответ?", wrapper)
        explain.setObjectName("explainAnswerButton")
        explain.setCursor(Qt.PointingHandCursor)
        explain.setToolTip("Открыть правую панель с разбором поиска, источниками и альтернативами")
        explain.clicked.connect(lambda _checked=False: self._show_search_reason_panel() if hasattr(self, "_show_search_reason_panel") else None)
        row.addWidget(explain, 0)
        added_any = True
        row.addStretch(1)

        if source_panel is not None:
            wrapper_layout.addWidget(source_panel)
        wrapper_layout.addWidget(context_panel)

        if added_any:
            layout.addWidget(wrapper)

    def _save_generated_docx(self, artifact_id: str, suggested_name: str) -> None:
        manager = getattr(self, "chat_manager", None)
        if manager is None or not hasattr(manager, "save_generation_artifact"):
            QMessageBox.warning(self, "Сохранение недоступно", "Сервис создания документов не запущен.")
            return
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить созданный документ",
            suggested_name or "Новый_документ.docx",
            "Документ Word (*.docx)",
        )
        if not destination:
            return
        try:
            saved = manager.save_generation_artifact(artifact_id, destination)
            self.status.showMessage(f"Документ сохранён: {saved}")
            QMessageBox.information(self, "Документ сохранён", f"Создана новая копия:\n{saved}")
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось сохранить", str(exc))

    @staticmethod
    def _split_file_lines(text: str) -> tuple[str, list[Path]]:
        """«📎 C:\\...\\file.jpg» lines are files (cards), not text; «📎 Источник: …» stays text."""
        kept, files = [], []
        for line in (text or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("📎 "):
                candidate = Path(stripped[2:].strip())
                if candidate.is_absolute():
                    files.append(candidate)
                    continue
            kept.append(line)
        return "\n".join(kept).strip(), files

    def _add_message(self, text: str, sender: str = "user", is_welcome: bool = False) -> QFrame:
        text, files = self._split_file_lines(text) if not is_welcome else (text, [])
        bubble = self._add_message_bubble(text, sender, is_welcome)
        if files and hasattr(self, "_attach_cards"):
            self._attach_cards(bubble, files, sender)
        return bubble

    def _add_message_bubble(self, text: str, sender: str = "user", is_welcome: bool = False) -> QFrame:
        row = QWidget(self.chat_container)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)

        bubble = QFrame(row)
        if sender == "user":
            bubble.setProperty("userBubble", "true")
        else:
            bubble.setProperty("aiBubble", "true")
        if is_welcome:
            bubble.setProperty("welcomeBubble", "true")

        layout = QVBoxLayout(bubble)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)
        min_width, max_width = self._message_width_bounds(sender, text, is_welcome)
        bubble.setMinimumWidth(min_width)
        bubble.setMaximumWidth(max_width)
        bubble.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        author = QLabel(bubble)
        author.setProperty("msgAuthor", sender)
        author.setText("Вы" if sender == "user" else "ChatKodi")

        body = QLabel(bubble)
        body.setProperty("msgText", "true")
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        body.setTextFormat(Qt.RichText)
        body.setText(self._render_message_html(text, sender))
        body.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        if sender == "user" and not (text or "").strip():
            body.hide()  # a message of files only: the cards are the content

        meta = QLabel(bubble)
        meta.setProperty("msgMeta", "true")
        meta.setAlignment(Qt.AlignRight)
        stamp = datetime.now().strftime("%H:%M")
        if is_welcome:
            meta.setText("ChatKodi")
        elif sender == "ai":
            meta.setText(f"{_MODE_TAGS.get(self.current_mode, '')} · {stamp}")
        else:
            meta.setText(stamp)

        layout.addWidget(author)
        layout.addWidget(body)
        if sender == "ai" and not is_welcome and (text or "").strip():
            self._add_response_actions(layout, text)
            bubble.setProperty("responseActionsAdded", True)
        layout.addWidget(meta)

        row.setProperty("messageBubble", bubble)
        row.setProperty("messageBody", body)
        row.setProperty("messageSender", sender)
        row.setProperty("messageWelcome", is_welcome)
        row.setProperty("messageText", text)

        if sender == "user":
            row_layout.addStretch(1)
            row_layout.addWidget(bubble, 0, Qt.AlignRight)
        else:
            row_layout.addWidget(bubble, 0, Qt.AlignLeft)
            row_layout.addStretch(1)

        idx = self.chat_layout.count() - 1
        typing_row = getattr(self, "_typing_row", None)
        if typing_row is not None:
            # New bubbles go above the "typing…" indicator, never below it.
            typing_index = self.chat_layout.indexOf(typing_row)
            if typing_index >= 0:
                idx = typing_index
        self.chat_layout.insertWidget(idx, row)

        restoring = bool(getattr(self, "_restoring_chat_history", False))
        if sender == "ai" and not is_welcome and not restoring and self.streaming_enabled_action.isChecked():
            self.current_streaming_bubble = bubble
            self.current_streaming_body = body
            self._streaming_text = text or ""

        if not restoring:
            fade_in(row, 240)
            if sender == "ai" and not is_welcome and (text or "").strip() and getattr(self, "animate_answers", True):
                self._reveal_answer(bubble, body, text)
        self._scroll_to_bottom()
        return bubble

    def _reveal_answer(self, bubble: QFrame, body: QLabel, text: str) -> None:
        """Show a finished answer word by word; actions appear when it is done."""
        from gui.components.reveal import TextRevealer

        actions = bubble.findChild(QFrame, "responseActions")
        if actions is not None:
            actions.hide()
        accent = theme_palette(getattr(self, "current_theme", "classic")).get("accent", "#22c55e")
        revealer = TextRevealer(body, text, lambda part: self._render_message_html(part, "ai"), accent, parent=bubble)
        revealer.progressed.connect(lambda: self._scroll_to_bottom(animated=False))

        def done() -> None:
            try:
                if actions is not None:
                    actions.show()
                for extra in getattr(bubble, "_after_reveal", []):
                    extra()
                bubble._after_reveal = []
            except RuntimeError:
                pass
            self._scroll_to_bottom()

        revealer.finished.connect(done)
        bubble._revealer = revealer
        bubble._after_reveal = []
        revealer.start()

    def _render_message_html(self, text: str, sender: str) -> str:
        if sender == "user":
            return plain_to_html(text)
        return markdown_to_html(text, theme_palette(getattr(self, "current_theme", "classic")))

    def _rerender_messages(self) -> None:
        """Re-apply theme colours baked into rich-text bubbles."""
        if not hasattr(self, "chat_layout"):
            return
        for i in range(self.chat_layout.count()):
            row = self.chat_layout.itemAt(i).widget()
            if row is None:
                continue
            body = row.property("messageBody")
            if body is None or body is getattr(self, "current_streaming_body", None):
                continue
            try:
                body.setText(self._render_message_html(str(row.property("messageText") or ""), str(row.property("messageSender") or "ai")))
            except RuntimeError:
                continue

    def _show_typing_indicator(self) -> None:
        self._hide_typing_indicator()
        row = QWidget(self.chat_container)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        indicator = TypingIndicator(row, theme_palette(getattr(self, "current_theme", "classic")))
        row_layout.addWidget(indicator, 0, Qt.AlignLeft)
        row_layout.addStretch(1)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, row)
        self._typing_row = row
        self._typing_indicator = indicator
        fade_in(row, 200)
        self._scroll_to_bottom()

    def _hide_typing_indicator(self) -> None:
        row = getattr(self, "_typing_row", None)
        if row is None:
            return
        indicator = getattr(self, "_typing_indicator", None)
        if indicator is not None:
            indicator.stop()
        self.chat_layout.removeWidget(row)
        row.hide()
        row.deleteLater()
        self._typing_row = None
        self._typing_indicator = None

    def _update_streaming_message(self, new_text: str):
        if not getattr(self, "current_streaming_body", None):
            # The bubble appears with the first tokens; until then the
            # "typing…" indicator is shown.
            self._hide_typing_indicator()
            self._add_message("", sender="ai")
            if not getattr(self, "current_streaming_body", None):
                return
        self._streaming_text = getattr(self, "_streaming_text", "") + new_text
        accent = theme_palette(getattr(self, "current_theme", "classic")).get("accent", "#22c55e")
        html_text = plain_to_html(self._clean_plain_text(self._streaming_text))
        self.current_streaming_body.setText(f'{html_text}<span style="color:{accent};">&nbsp;▍</span>')
        row = self.current_streaming_body.parentWidget().parentWidget() if self.current_streaming_body.parentWidget() else None
        if row is not None:
            row.setProperty("messageText", self._streaming_text)
        self._scroll_to_bottom()

    def _finalize_streaming_message(self, full_text: str):
        if not getattr(self, "current_streaming_body", None):
            return
        self.current_streaming_body.setText(self._render_message_html(full_text, "ai"))
        bubble_row = self.current_streaming_body.parentWidget().parentWidget() if self.current_streaming_body.parentWidget() else None
        if bubble_row is not None:
            bubble_row.setProperty("messageText", full_text)
        # При streaming пузырь создаётся пустым, поэтому источники нельзя было
        # построить в _add_message. Добавляем их после получения полного ответа,
        # перед временем сообщения и только один раз.
        bubble = getattr(self, "current_streaming_bubble", None)
        if bubble is not None and not bool(bubble.property("responseActionsAdded")):
            bubble_layout = bubble.layout()
            if bubble_layout is not None:
                meta_item = bubble_layout.itemAt(bubble_layout.count() - 1)
                meta_widget = meta_item.widget() if meta_item is not None else None
                if meta_widget is not None:
                    bubble_layout.removeWidget(meta_widget)
                self._add_response_actions(bubble_layout, full_text)
                if meta_widget is not None:
                    bubble_layout.addWidget(meta_widget)
                bubble.setProperty("responseActionsAdded", True)
        self.current_streaming_bubble = None
        self.current_streaming_body = None

    def _scroll_to_bottom(self, animated: bool = True):
        if not hasattr(self, "chat_scroll"):
            return
        scroller = getattr(self, "_chat_scroller", None)
        if scroller is None:
            scroller = SmoothScroller(self.chat_scroll)
            self._chat_scroller = scroller
        scroller.scroll_to_bottom(animated=animated and not getattr(self, "_restoring_chat_history", False))

    def _on_streaming_token(self, token: str):
        if not self.is_streaming:
            return
        self.streaming_buffer += token
        if not self.streaming_timer.isActive():
            self.streaming_timer.start(80)

    def _flush_streaming_buffer(self):
        if self.streaming_buffer:
            self._update_streaming_message(self.streaming_buffer)
            self.streaming_buffer = ""
            self.streaming_indicator.setText("⌛")
        if not self.streaming_buffer:
            self.streaming_timer.stop()

    def _start_streaming(self):
        self.is_streaming = True
        self.streaming_buffer = ""
        self.streaming_indicator.setText("⌛")
        self.status.showMessage("Генерация ответа...")
        self.stop_btn.setEnabled(True)
        self.stop_btn.setVisible(True)
        # The answer bubble is created with the first tokens (chat) or with
        # the validated answer (documents); meanwhile «печатает…» is shown.
        self.current_streaming_bubble = None
        self.current_streaming_body = None
        self._streaming_text = ""

    def _stop_streaming(self, full_text: str = ""):
        self.is_streaming = False
        self.streaming_timer.stop()
        if self.streaming_buffer:
            self._update_streaming_message(self.streaming_buffer)
            self.streaming_buffer = ""
        self._hide_typing_indicator()
        if full_text:
            self.last_answer_text = full_text
            if getattr(self, "current_streaming_body", None) is None:
                self._add_message(full_text, sender="ai")
            self._finalize_streaming_message(full_text)
            self._record_chat_message("assistant", full_text)
            if hasattr(self, "_update_chat_inspector"):
                self._update_chat_inspector(answer=full_text)
        self.streaming_indicator.setText("")
        self.status.showMessage("Готово")
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Отправить")
        self.stop_btn.setVisible(False)

    def _release_streaming_worker(self, worker) -> None:
        if getattr(self, "current_streaming_worker", None) is worker:
            self.current_streaming_worker = None

    def _cancel_streaming(self) -> None:
        agent_worker = getattr(self, "_agent_worker", None)
        if agent_worker is not None and agent_worker.isRunning():
            self.stop_btn.setEnabled(False)
            self.status.showMessage("Останавливаю работу с файлами; сохранённые версии останутся доступны…")
            agent_worker.cancel()
            return
        worker = getattr(self, "current_streaming_worker", None)
        if worker is None or not worker.isRunning():
            return
        self.stop_btn.setEnabled(False)
        self.streaming_indicator.setText("⏹")
        self.status.showMessage("Останавливаю генерацию...")
        worker.cancel()

    def _on_streaming_cancelled(self, partial_text: str = "") -> None:
        self.is_streaming = False
        self.streaming_timer.stop()
        if self.streaming_buffer:
            self._update_streaming_message(self.streaming_buffer)
            self.streaming_buffer = ""
        self._hide_typing_indicator()
        visible = (partial_text or "").strip()
        if not visible and getattr(self, "current_streaming_body", None):
            visible = str(getattr(self, "_streaming_text", "") or "").strip()
        notice = "⏹ Генерация остановлена пользователем."
        final_text = f"{visible}\n\n{notice}" if visible else notice
        self.last_answer_text = final_text
        if getattr(self, "current_streaming_body", None) is None:
            self._add_message(final_text, sender="ai")
        self._finalize_streaming_message(final_text)
        self._record_chat_message("assistant", final_text)
        self.streaming_indicator.setText("")
        self.status.showMessage("Генерация остановлена")
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Отправить")
        self.stop_btn.setVisible(False)

    def _run_send_anim(self) -> None:
        pulse(self.send_btn)

    def _on_send(self) -> None:
        text = self.input_field.text().strip()
        if getattr(self, "_agent_worker", None) is not None and self._agent_worker.isRunning():
            return
        if self.current_mode == ChatMode.AGENT or (self._pending() and hasattr(self, "_send_to_assistant")):
            if text or self._pending():
                self._send_to_assistant(text)
            return
        if not text:
            return
        self.last_user_message = text
        self.last_error_text = ""
        self._switch_workspace("chat")
        self._add_message(text, sender="user")
        self._record_chat_message("user", text)
        self.input_field.clear()
        self._run_send_anim()
        if hasattr(self, "_update_chat_inspector"):
            self._update_chat_inspector()

        self.send_btn.setEnabled(False)
        self.send_btn.setText("Думаю…")
        if hasattr(self, "_sync_search_preferences_to_manager"):
            self._sync_search_preferences_to_manager()

        llm_ready = None
        if self.current_mode in (ChatMode.RAG, ChatMode.CHAT) and hasattr(self, "_ensure_llm_ready"):
            try:
                # RC5: RAG часто отдаёт структурный/direct answer без LLM.
                # Раньше здесь eager=True грузил Qwen3-4B на CPU до поиска и
                # убивал скорость. Теперь LLM заранее грузится только для чистого
                # Chat-режима; в RAG она останется ленивой и поднимется в фоне
                # только если прямого ответа действительно нет.
                eager_prepare = self.current_mode == ChatMode.CHAT
                self.status.showMessage("Подготавливаю чат…" if eager_prepare else "Ищу по документам…")
                llm_ready = bool(self._ensure_llm_ready(eager=eager_prepare, reprobe=True))
            except Exception as exc:
                logger.warning("LLM prepare failed before send: %s", exc)
                llm_ready = False
            logger.info(
                "Dispatch prepare: ui_mode=%s llm_ready=%s llm_present=%s eager=%s",
                self.current_mode.value,
                llm_ready,
                bool(self.chat_manager and getattr(self.chat_manager, "llm_model", None)),
                self.current_mode == ChatMode.CHAT,
            )

        llm_obj = getattr(self.chat_manager, "llm_model", None) if self.chat_manager else None
        effective_llm_ready = False
        if llm_obj is not None:
            try:
                # LazyLLMProxy.is_available() intentionally returns True before
                # the model is loaded. That is useful for capability checks, but
                # terrible for UI routing: streaming would start and then load a
                # multi-GB model on CPU. Для RAG считаем LLM готовой только если
                # она уже реально загружена; иначе пусть обычный worker сначала
                # попробует структурный/direct answer без блокировки интерфейса.
                if hasattr(llm_obj, "is_loaded") and not llm_obj.is_loaded():
                    effective_llm_ready = False
                else:
                    effective_llm_ready = bool(llm_obj.is_available())
            except Exception:
                effective_llm_ready = self.current_mode == ChatMode.CHAT
        use_streaming = (
            self.streaming_enabled_action.isChecked() and
            self.current_mode in [ChatMode.RAG, ChatMode.CHAT] and
            effective_llm_ready
        )
        logger.info(
            "Dispatch route: ui_mode=%s manager_mode=%s llm_ready=%s streaming=%s",
            self.current_mode.value,
            getattr(getattr(self, "chat_manager", None), "current_mode", self.current_mode).value if getattr(self, "chat_manager", None) else self.current_mode.value,
            effective_llm_ready,
            use_streaming,
        )

        self._show_typing_indicator()
        if use_streaming:
            self._start_streaming()
            worker = StreamingWorker(
                self.chat_manager,
                text,
                self.current_mode,
                self.use_full_documents
            )
            self.current_streaming_worker = worker
            worker.token_received.connect(self.streaming_token.emit)
            worker.generation_finished.connect(lambda txt: self._stop_streaming(txt))
            worker.generation_cancelled.connect(self._on_streaming_cancelled)
            worker.error_occurred.connect(lambda err: self._on_streaming_error(err))
            worker.finished.connect(lambda: self._release_streaming_worker(worker))
            worker.start()
        else:
            thread = threading.Thread(target=self._worker_process_message, args=(text, self.current_mode), daemon=True)
            thread.start()

    def _worker_process_message(self, question: str, mode: ChatMode):
        try:
            if self.chat_manager:
                self.chat_manager.set_mode(mode)
            answer = self.chat_manager.process_message(question, forced_mode=mode)
            self.response_ready.emit(answer, mode.value)
        except Exception as exc:
            self.response_error.emit(str(exc))

    def _on_streaming_error(self, error_msg: str):
        logger.error("Streaming failed: %s", error_msg)
        self.last_error_text = error_msg
        self._stop_streaming()
        self._add_message(f"❌ Ошибка: {error_msg}", sender="ai")
        self.current_streaming_bubble = None
        self.current_streaming_body = None
        if hasattr(self, "_update_chat_inspector"):
            self._update_chat_inspector(error=error_msg)
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Отправить")
        self.status.showMessage("Ошибка")

    def _on_response_ready(self, answer: str, mode: str):
        self._hide_typing_indicator()
        self.last_answer_text = answer
        self._add_message(answer, sender="ai")
        self.current_streaming_bubble = None
        self.current_streaming_body = None
        self._record_chat_message("assistant", answer)
        if hasattr(self, "_update_chat_inspector"):
            self._update_chat_inspector(answer=answer)
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Отправить")
        self.status.showMessage("Готово")

    def _on_response_error(self, message: str):
        self._hide_typing_indicator()
        self.last_error_text = message
        self._add_message(f"❌ Ошибка: {message}", sender="ai")
        self.current_streaming_bubble = None
        self.current_streaming_body = None
        if hasattr(self, "_update_chat_inspector"):
            self._update_chat_inspector(error=message)
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Отправить")
        self.status.showMessage("Ошибка")
