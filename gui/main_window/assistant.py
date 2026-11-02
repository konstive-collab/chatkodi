"""Assistant mode in the main window: attachments, files in answers,
confirmed console commands and personal instructions."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.agent.attachments import is_supported
from core.chat_mode import ChatMode
from core.logging_utils import get_logger
from gui.workers.agent import AgentWorker, CommandWorker

logger = get_logger(__name__)

_ICONS = {".docx": "📄", ".xlsx": "📊", ".txt": "📝", ".md": "📝", ".pdf": "📕"}
# results of actions, not answers from documents: no «Источники ответа» row under them
_NO_SOURCE_ACTIONS = {"files", "app", "action_result", "command", "command_result", "chat", "code"}
_KIND_ICONS = {"image": "🖼", "audio": "🎙", "document": "📄"}


class ChatKodiAssistantMixin:
    # ------------------------------------------------------------ state

    def _assistant(self):
        """The assistant of the current chat (own history and attachments)."""
        return self._current_agent()
    def _pending(self) -> List[Path]:
        if not hasattr(self, "_pending_attachments"):
            self._pending_attachments = []
        return self._pending_attachments

    # ------------------------------------------------------------ input row widgets

    def _build_attachment_widgets(self, input_layout: QVBoxLayout, entry_row: QHBoxLayout) -> None:
        self.attach_btn = QPushButton("", self.input_frame)
        self.attach_btn.setObjectName("attachButton")
        self.attach_btn.setToolTip("Приложить файлы, фото или сканы (также: перетащить в окно, Ctrl+V для картинки)")
        self.attach_btn.setCursor(Qt.PointingHandCursor)
        self.attach_btn.setFixedWidth(44)
        self.attach_btn.clicked.connect(self._choose_attachments)
        entry_row.insertWidget(0, self.attach_btn)

        self.attachment_bar = QFrame(self.input_frame)
        self.attachment_bar.setObjectName("attachmentBar")
        self.attachment_bar_layout = QHBoxLayout(self.attachment_bar)
        self.attachment_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.attachment_bar_layout.setSpacing(6)
        self.attachment_bar.setVisible(False)
        input_layout.insertWidget(0, self.attachment_bar)
        self.input_field.installEventFilter(self)
        self._refresh_assistant_icons()

    def _refresh_assistant_icons(self) -> None:
        """Vector icons in the colour of the current theme (called again on theme change)."""
        from PyQt5.QtCore import QSize

        from core.theme_manager import theme_palette
        from gui.components import icons

        color = theme_palette(getattr(self, "current_theme", "classic")).get("fg", "#e5e7eb")
        size = QSize(22, 22)
        if hasattr(self, "attach_btn"):
            self.attach_btn.setIcon(icons.paperclip(color))
            self.attach_btn.setIconSize(size)
        if hasattr(self, "voice_btn"):
            recording = self.voice_btn.isChecked()
            self.voice_btn.setIcon(icons.microphone("#ffffff" if recording else color))
            self.voice_btn.setIconSize(size)
        if hasattr(self, "speak_btn"):
            self.speak_btn.setIcon(icons.speaker(color, self.speak_btn.isChecked()))
            self.speak_btn.setIconSize(size)
        if hasattr(self, "voice_wave"):
            self.voice_wave.set_color(theme_palette(getattr(self, "current_theme", "classic")).get("accent", "#22c55e"))

    def _assistant_event_filter(self, obj, event) -> bool:
        """Ctrl+V of a picture or of copied files in the input field attaches them."""
        if obj is getattr(self, "input_field", None) and event.type() == QEvent.KeyPress:
            if event.matches(QKeySequence.Paste) and self._paste_image_from_clipboard():
                return True
        return False

    def _show_drop_overlay(self, paths) -> None:
        from core.theme_manager import theme_palette
        from gui.components.dropzone import DropOverlay
        from gui.components.fonts import ui_family

        page = self.page_stack.widget(self.workspace_pages["chat"])
        overlay = getattr(self, "_drop_overlay", None)
        if overlay is None:
            overlay = DropOverlay(page, theme_palette(self.current_theme), ui_family())
            self._drop_overlay = overlay
        overlay.set_palette(theme_palette(self.current_theme))
        names = ", ".join(p.name for p in paths[:3]) + (" …" if len(paths) > 3 else "")
        overlay.show_for(len(paths), names)

    def _hide_drop_overlay(self) -> None:
        overlay = getattr(self, "_drop_overlay", None)
        if overlay is not None:
            overlay.hide_soon()

    def _paste_image_from_clipboard(self) -> bool:
        mime = QApplication.clipboard().mimeData()
        if mime is None:
            return False
        if mime.hasUrls():
            files = [u.toLocalFile() for u in mime.urls() if u.isLocalFile() and is_supported(Path(u.toLocalFile()))]
            if files:
                self._attach_files(files)
                return True
        if mime.hasImage():
            image = QApplication.clipboard().image()
            if image.isNull():
                return False
            folder = Path(tempfile.gettempdir()) / "chatkodi_paste"
            folder.mkdir(exist_ok=True)
            path = folder / f"Изображение {time.strftime('%H-%M-%S')}.png"
            image.save(str(path), "PNG")
            self._attach_files([str(path)])
            return True
        return False

    def _choose_attachments(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Приложить файлы",
            str(Path.home()),
            "Документы и изображения (*.pdf *.docx *.doc *.rtf *.odt *.txt *.md *.xlsx *.xls *.csv *.pptx "
            "*.html *.py *.pyw *.js *.ts *.css *.sql *.eml *.json *.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.wav *.mp3 *.ogg *.m4a);;Все файлы (*.*)",
        )
        if files:
            self._attach_files(files)

    def _attach_files(self, paths) -> None:
        if self.current_mode != ChatMode.AGENT:
            self._set_mode(ChatMode.AGENT)
        pending = self._pending()
        added, skipped = 0, []
        for raw in paths:
            path = Path(raw)
            if not path.is_file() or not is_supported(path):
                skipped.append(path.name)
                continue
            if path not in pending:
                pending.append(path)
                added += 1
        self._refresh_attachment_bar()
        if skipped:
            self.status.showMessage("Не поддерживается: " + ", ".join(skipped[:4]))
        elif added:
            self.status.showMessage(f"Приложено файлов: {len(pending)}. Напишите задачу или просто нажмите «Отправить».")
        self.input_field.setFocus()

    def _remove_attachment(self, path: Path) -> None:
        pending = self._pending()
        if path in pending:
            pending.remove(path)
        self._refresh_attachment_bar()

    def _refresh_attachment_bar(self) -> None:
        layout = self.attachment_bar_layout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        pending = self._pending()
        for path in pending:
            from core.agent.attachments import kind_of

            chip = QFrame(self.attachment_bar)
            chip.setObjectName("attachmentChip")
            row = QHBoxLayout(chip)
            row.setContentsMargins(8, 2, 4, 2)
            row.setSpacing(4)
            label = QLabel(f"{_KIND_ICONS.get(kind_of(path), '📄')} {path.name}", chip)
            label.setToolTip(str(path))
            close = QPushButton("✕", chip)
            close.setObjectName("attachmentRemove")
            close.setFixedWidth(22)
            close.setCursor(Qt.PointingHandCursor)
            close.clicked.connect(lambda _c=False, p=path: self._remove_attachment(p))
            row.addWidget(label)
            row.addWidget(close)
            layout.addWidget(chip)
        layout.addStretch(1)
        self.attachment_bar.setVisible(bool(pending))

    # ------------------------------------------------------------ sending

    def _send_to_assistant(self, text: str) -> None:
        files = [str(p) for p in self._pending()]
        self._pending_attachments = []
        self._refresh_attachment_bar()
        # the history keeps full paths: the chat shows the same cards after a restart
        shown = "\n\n".join(part for part in (text, "\n".join(f"📎 {f}" for f in files)) if part)
        self.last_user_message = shown
        self.last_error_text = ""
        self._switch_workspace("chat")
        user_bubble = self._add_message(shown, sender="user")
        self._record_chat_message("user", shown)
        self.input_field.clear()
        self._run_send_anim()
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Думаю…")
        if hasattr(self, "_ensure_llm_ready"):
            try:
                self._ensure_llm_ready(eager=True, reprobe=True)
            except Exception as exc:
                logger.warning("LLM prepare failed before assistant turn: %s", exc)
        self._start_trail_bubble()
        worker = AgentWorker(self._assistant(), text, files)
        self._agent_worker = worker
        self._agent_turn = (text, user_bubble)
        worker.progress.connect(self._on_agent_progress)
        worker.finished_ok.connect(self._on_agent_result)
        worker.failed.connect(self._on_agent_failed)
        worker.start()

    # ------------------------------------------------------------ progress trail

    def _start_trail_bubble(self) -> None:
        from core.theme_manager import theme_palette
        from gui.components.fonts import ui_family
        from gui.components.progress import ProgressTrail

        self._hide_typing_indicator()
        animate = getattr(self, "animate_answers", True)
        self.animate_answers = False
        bubble = self._add_message_bubble("", sender="ai")
        self.animate_answers = animate
        self.current_streaming_bubble = None
        self.current_streaming_body = None
        trail = ProgressTrail(theme_palette(getattr(self, "current_theme", "classic")), bubble, ui_family())
        bubble.layout().insertWidget(1, trail)
        row = bubble.parentWidget()
        body = row.property("messageBody") if row is not None else None
        if body is not None:
            body.hide()
        self._trail = (bubble, trail)
        self._scroll_to_bottom()

    def _on_agent_progress(self, message: str) -> None:
        self.status.showMessage(message)
        trail = getattr(self, "_trail", None)
        if trail is not None:
            try:
                trail[1].step(message)
            except RuntimeError:
                pass

    def _on_agent_failed(self, message: str) -> None:
        trail = getattr(self, "_trail", None)
        self._trail = None
        if trail is not None:
            try:
                trail[1].finish()
                trail[0].parentWidget().deleteLater()
            except RuntimeError:
                pass
        self._on_response_error(message)

    # ------------------------------------------------------------ result

    def _on_agent_result(self, result) -> None:
        from core.agent.agent import attachment_outcome

        self._hide_typing_indicator()
        self.last_answer_text = result.text
        trail = getattr(self, "_trail", None)
        self._trail = None
        if trail is None:
            self._start_trail_bubble()
            trail, self._trail = self._trail, None
        bubble, progress = trail
        progress.finish(result.seconds)
        row = bubble.parentWidget()
        body = row.property("messageBody")
        row.setProperty("messageText", result.text)
        min_width, max_width = self._message_width_bounds("ai", result.text)
        if result.files or result.proposal is not None or getattr(result, "found", None):
            min_width = max(min_width, min(max_width, 560))  # file sheets and command rows need room
        bubble.setMinimumWidth(min_width)
        bubble.setMaximumWidth(max_width)
        body.show()
        # the answer is typed into the bubble; actions, files and hints follow
        layout = bubble.layout()
        meta = layout.itemAt(layout.count() - 1).widget()
        layout.removeWidget(meta)
        if result.action not in _NO_SOURCE_ACTIONS:
            self._add_response_actions(layout, result.text)  # sources and «why» belong to answers from documents
        layout.addWidget(meta)
        bubble.setProperty("responseActionsAdded", True)
        self._reveal_answer(bubble, body, result.text)
        extras = []
        for output in result.files:
            extras.append(self._output_widget(bubble, output))
        if getattr(result, "found", None):
            extras.append(self._found_files_row(bubble, result.found))
        if getattr(result, "undo_log", None):
            extras.append(self._undo_row(bubble, result.undo_log))
        if result.proposal is not None:
            extras.append(self._command_row(bubble, result.proposal))
        for widget in extras:
            widget.hide()
            self._insert_into_bubble(bubble, widget)
        bubble._after_reveal.extend(lambda w=w: w.show() for w in extras)

        # outcome of reading each attached file on its card
        turn_text, user_bubble = getattr(self, "_agent_turn", ("", None))
        cards = getattr(user_bubble, "_file_cards", {}) if user_bubble is not None else {}
        for item in result.attachments:
            card = cards.get(item.name)
            if card is not None:
                status, warn = attachment_outcome(item)
                try:
                    card.set_status(status, warn)
                except RuntimeError:
                    pass

        record = result.text + ("\n\n" + "\n".join(f"📎 {f.path}" for f in result.files) if result.files else "")
        self._record_chat_message("assistant", record)
        if hasattr(self, "_update_chat_inspector"):
            self._update_chat_inspector(answer=result.text)
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Отправить")
        self.status.showMessage(f"Готово за {result.seconds:.1f} с")
        if getattr(self, "voice_reply_enabled", False) and hasattr(self, "_speak_text"):
            self._speak_text(result.content or result.text)
        self._suggest_next_steps(bubble, turn_text, result)

    # ------------------------------------------------------------ next steps

    def _suggest_next_steps(self, bubble, message: str, result) -> None:
        """The model proposes 2–3 next steps; they appear under the answer a moment later."""
        import threading

        agent = self._assistant()

        def work():
            try:
                items = agent.suggest_next(message, result) if getattr(self, "suggestions_enabled", True) else []
            except Exception as exc:
                logger.debug("Next-step suggestions failed: %s", exc)
                items = []
            if items:
                self._suggestions_ready.emit(bubble, items)
            # after the answer, not before it: the model serves one request at a time
            try:
                agent.remember(message)
            except Exception as exc:
                logger.debug("Memory update failed: %s", exc)

        threading.Thread(target=work, daemon=True).start()

    def _show_suggestions(self, bubble, items) -> None:
        try:
            frame = QFrame(bubble)
        except RuntimeError:
            return
        frame.setObjectName("suggestionRow")
        row = QHBoxLayout(frame)
        row.setContentsMargins(0, 2, 0, 0)
        row.setSpacing(6)
        for text in items:
            chip = QPushButton(text, frame)
            chip.setObjectName("suggestionChip")
            chip.setCursor(Qt.PointingHandCursor)
            chip.clicked.connect(lambda _c=False, t=text, f=frame: (f.hide(), self._send_suggestion(t)))
            row.addWidget(chip)
        row.addStretch(1)

        def place():
            try:
                self._insert_into_bubble(bubble, frame)
                from gui.components.motion import fade_in

                fade_in(frame, 260)
                self._scroll_to_bottom()
            except RuntimeError:
                pass

        if getattr(bubble, "_after_reveal", None) and getattr(bubble, "_revealer", None) is not None and bubble._revealer.timer.isActive():
            bubble._after_reveal.append(place)
        else:
            place()

    def _send_suggestion(self, text: str) -> None:
        if self._busy():
            return
        self.input_field.setText(text)
        self._on_send()

    # ------------------------------------------------------------ cards

    def _attach_cards(self, bubble, files, sender: str) -> None:
        """File cards under a message: sent files (user) or created documents (assistant, restored chat)."""
        from gui.components.filecard import FileCard

        frame = QFrame(bubble)
        column = QVBoxLayout(frame)
        column.setContentsMargins(0, 4, 0, 0)
        column.setSpacing(6)
        cards = {}
        for path in files:
            if sender == "ai" and path.suffix.lower() in {".docx", ".txt", ".md"} and path.is_file():
                from core.agent.outputs import OutputFile, docx_to_markdown

                try:
                    markdown = docx_to_markdown(path) if path.suffix.lower() == ".docx" else path.read_text(encoding="utf-8")
                except Exception:
                    markdown = ""
                column.addWidget(self._output_widget(frame, OutputFile(path, "Документ", markdown)))
                continue
            card = FileCard(path, frame)
            card.opened.connect(lambda p: self._open_output(Path(p)))
            column.addWidget(card, 0, Qt.AlignRight if sender == "user" else Qt.AlignLeft)
            cards[path.name] = card
        bubble._file_cards = cards
        self._insert_into_bubble(bubble, frame)

    def _output_widget(self, parent, output) -> QWidget:
        """A created file: a sheet of paper (LMB save, RMB unfold with editing) or a card."""
        from gui.components.paper import DocumentViewer, PaperCard

        holder = QFrame(parent)
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 4, 0, 0)
        column.setSpacing(6)
        if not getattr(output, "markdown", ""):
            from gui.components.filecard import FileCard

            card = FileCard(output.path, holder)
            card.opened.connect(lambda _p, p=output.path: self._save_output_copy(p))
            card.setToolTip("Левый клик — сохранить файл")
            column.addWidget(card, 0, Qt.AlignLeft)
            return holder
        paper = PaperCard(output.name, output.label, output.markdown, holder)
        column.addWidget(paper, 0, Qt.AlignLeft)
        state = {"viewer": None}

        def save_edits(markdown: str) -> None:
            from core.agent.outputs import rewrite_output

            try:
                rewrite_output(output, markdown)
                paper.set_markdown(markdown)
                self.status.showMessage(f"Правки сохранены в «{output.name}»")
            except Exception as exc:
                QMessageBox.warning(self, "Не удалось сохранить правки", str(exc))

        def toggle() -> None:
            viewer = state["viewer"]
            if viewer is None:
                viewer = DocumentViewer(output.markdown, save_edits, lambda: self._save_output_copy(output.path), holder)
                state["viewer"] = viewer
                column.addWidget(viewer)
                from gui.components.motion import fade_in

                fade_in(viewer, 220)
            else:
                viewer.setVisible(not viewer.isVisible())
            self._scroll_to_bottom()

        paper.save_requested.connect(lambda: self._save_output_copy(output.path))
        paper.toggle_requested.connect(toggle)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        for caption, handler in (("Открыть в Word", self._open_output), ("В папке", self._show_output_in_folder), ("В библиотеку", self._add_output_to_library)):
            btn = QPushButton(caption, holder)
            btn.setObjectName("fileActionButton")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _c=False, p=output.path, h=handler: h(p))
            actions.addWidget(btn)
        actions.addStretch(1)
        column.addLayout(actions)
        return holder

    @staticmethod
    def _insert_into_bubble(bubble: QFrame, widget: QWidget) -> None:
        layout = bubble.layout()
        if layout is None:
            return
        # keep the time stamp as the last line of the bubble
        layout.insertWidget(max(0, layout.count() - 1), widget)

    # ------------------------------------------------------------ files in answers

    def _files_row(self, parent: QWidget, files) -> QFrame:
        frame = QFrame(parent)
        frame.setObjectName("filesRow")
        column = QVBoxLayout(frame)
        column.setContentsMargins(0, 4, 0, 0)
        column.setSpacing(4)
        for output in files:
            row = QHBoxLayout()
            row.setSpacing(6)
            icon = _ICONS.get(output.path.suffix.lower(), "📎")
            label = QLabel(f"{icon} <b>{output.name}</b> — {output.label}", frame)
            label.setTextFormat(Qt.RichText)
            row.addWidget(label, 1)
            for caption, handler in (
                ("Открыть", self._open_output),
                ("Сохранить как…", self._save_output_copy),
                ("В папке", self._show_output_in_folder),
                ("В библиотеку", self._add_output_to_library),
            ):
                btn = QPushButton(caption, frame)
                btn.setObjectName("fileActionButton")
                btn.setCursor(Qt.PointingHandCursor)
                btn.clicked.connect(lambda _c=False, p=output.path, h=handler: h(p))
                row.addWidget(btn, 0)
            column.addLayout(row)
        return frame

    def _open_output(self, path: Path) -> None:
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось открыть", str(exc))

    def _save_output_copy(self, path: Path) -> None:
        target, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", str(Path.home() / path.name), f"*{path.suffix}")
        if not target:
            return
        try:
            shutil.copy2(path, target)
            self.status.showMessage(f"Сохранено: {target}")
        except OSError as exc:
            QMessageBox.warning(self, "Не удалось сохранить", str(exc))

    def _add_output_to_library(self, path: Path) -> None:
        """Put a created file into the document library (it is indexed and searchable)."""
        copied, _skipped, errors = self._import_document_paths([str(path)])
        if errors:
            QMessageBox.warning(self, "Не удалось добавить", "\n".join(errors))
        elif copied:
            self.status.showMessage(f"«{path.name}» добавлен в библиотеку, индекс обновляется")
        else:
            self.status.showMessage(f"«{path.name}» уже есть в библиотеке")

    def _open_outputs_folder(self) -> None:
        from core.agent.outputs import outputs_dir

        self._open_output(outputs_dir().parent)

    def _show_output_in_folder(self, path: Path) -> None:
        try:
            subprocess.Popen(["explorer", "/select,", str(path)])
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось открыть папку", str(exc))

    def _found_files_row(self, parent: QWidget, paths) -> QFrame:
        """Files found on the computer: a card each (click opens) and «Приложить к чату»."""
        from gui.components.filecard import FileCard

        frame = QFrame(parent)
        frame.setObjectName("filesRow")
        column = QVBoxLayout(frame)
        column.setContentsMargins(0, 4, 0, 0)
        column.setSpacing(6)
        for path in [Path(p) for p in paths][:6]:
            row = QHBoxLayout()
            row.setSpacing(6)
            card = FileCard(path, frame)
            card.opened.connect(lambda p: self._open_output(Path(p)))
            card.setToolTip(str(path))
            row.addWidget(card, 0)
            for caption, handler in (("Приложить к чату", lambda p: self._attach_files([p])), ("В папке", self._show_output_in_folder)):
                btn = QPushButton(caption, frame)
                btn.setObjectName("fileActionButton")
                btn.setCursor(Qt.PointingHandCursor)
                btn.clicked.connect(lambda _c=False, p=path, h=handler: h(p))
                row.addWidget(btn, 0)
            row.addStretch(1)
            column.addLayout(row)
        return frame

    def _undo_row(self, parent: QWidget, log) -> QFrame:
        frame = QFrame(parent)
        frame.setObjectName("commandRow")
        row = QHBoxLayout(frame)
        row.setContentsMargins(0, 4, 0, 0)
        btn = QPushButton("↶ Отменить", frame)
        btn.setCursor(Qt.PointingHandCursor)

        def undo() -> None:
            btn.setEnabled(False)
            try:
                text = self._assistant().undo(Path(log))
            except Exception as exc:
                text = f"⚠ Не удалось отменить: {exc}"
            btn.setText("Отменено")
            self._skip_response_actions = True  # a report of an action, not an answer from documents
            try:
                self._add_message(text, "ai")
            finally:
                self._skip_response_actions = False
            self._record_chat_message("assistant", text)

        btn.clicked.connect(undo)
        row.addWidget(btn)
        row.addStretch(1)
        return frame

    # ------------------------------------------------------------ commands

    def _command_row(self, parent: QWidget, proposal) -> QFrame:
        frame = QFrame(parent)
        frame.setObjectName("commandRow")
        row = QHBoxLayout(frame)
        row.setContentsMargins(0, 4, 0, 0)
        caption = {"files": "▶ Выполнить", "app": "▶ Открыть"}.get(getattr(proposal, "kind", "powershell"), "▶ Выполнить")
        if getattr(proposal, "payload", {}).get("op") == "email":
            caption = "▶ Открыть черновик"
        run = QPushButton(caption, frame)
        run.setObjectName("commandRunButton")
        run.setCursor(Qt.PointingHandCursor)
        cancel = QPushButton("Отмена", frame)
        cancel.setCursor(Qt.PointingHandCursor)

        def disable(note: str) -> None:
            run.setEnabled(False)
            cancel.setEnabled(False)
            run.setText(note)

        run.clicked.connect(lambda: self._run_proposal(proposal, disable))
        cancel.clicked.connect(lambda: disable("Отменено"))
        row.addWidget(run)
        row.addWidget(cancel)
        row.addStretch(1)
        return frame

    def _run_proposal(self, proposal, disable) -> None:
        from core.agent.permissions import DANGER

        commands = "\n".join(proposal.commands)
        if proposal.dangerous or getattr(proposal, "risk", 0) >= DANGER:
            answer = QMessageBox.warning(
                self,
                "Подтвердите опасную команду",
                "Команда может удалить или изменить данные или настройки системы:\n\n"
                f"{commands}\n\nВыполнить?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        disable("Выполняется…")
        self._show_typing_indicator()
        worker = CommandWorker(self._assistant(), proposal)
        self._command_worker = worker
        worker.progress.connect(lambda message: self.status.showMessage(message))
        worker.finished_ok.connect(lambda result: (disable("Выполнено"), self._on_agent_result(result)))
        worker.failed.connect(self._on_response_error)
        worker.start()

    # ------------------------------------------------------------ personal instructions

    def _edit_personal_instructions(self) -> None:
        from core.agent.instructions import load_instructions, save_instructions

        dialog = QDialog(self)
        dialog.setWindowTitle("Личные инструкции")
        dialog.resize(640, 420)
        layout = QVBoxLayout(dialog)
        hint = QLabel(
            "Правила, которые ассистент соблюдает во всех ответах и документах: кто вы и ваша организация "
            "(для подписи писем), стиль, обязательные реквизиты, язык, формат дат.",
            dialog,
        )
        hint.setWordWrap(True)
        editor = QPlainTextEdit(dialog)
        editor.setPlainText(load_instructions())
        editor.setPlaceholderText(
            "Например:\nЯ — Петрова Анна Сергеевна, инженер ПТО ООО «СтройЛаб».\n"
            "Письма подписывай от моего имени.\nДаты пиши в формате ДД.ММ.ГГГГ.\nОтвечай кратко."
        )
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(hint)
        layout.addWidget(editor, 1)
        layout.addWidget(buttons)
        if dialog.exec_() == QDialog.Accepted:
            save_instructions(editor.toPlainText())
            self.status.showMessage("Личные инструкции сохранены")
