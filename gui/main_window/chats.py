"""Several chats in the sidebar: list, search, new / rename / pin / delete,
each chat with its own history, attachments and assistant memory."""

from __future__ import annotations

import time
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.chat_sessions import ChatSessionStore
from core.logging_utils import get_logger
from core.paths import runtime_root

logger = get_logger(__name__)


def _when(ts: float) -> str:
    moment = datetime.fromtimestamp(ts)
    today = datetime.now().date()
    if moment.date() == today:
        return moment.strftime("%H:%M")
    if (today - moment.date()).days == 1:
        return "вчера"
    return moment.strftime("%d.%m")


class ChatKodiChatsMixin:
    # ------------------------------------------------------------ state

    def _init_chat_sessions(self) -> None:
        self.chat_sessions = ChatSessionStore(runtime_root() / "chats")
        legacy = getattr(self, "chat_history_store", None)
        if legacy is not None and not self.chat_sessions.chats:
            self.chat_sessions.import_legacy(legacy)
        ordered = self.chat_sessions.ordered()
        info = ordered[0] if ordered else self.chat_sessions.create()
        self.current_chat_id = info.id
        self.chat_history_store = self.chat_sessions.history(info.id)
        self._agents = {}

    def _current_agent(self):
        from core.agent.agent import Agent
        from core.agent.attachments import AttachmentStore

        agent = self._agents.get(self.current_chat_id)
        manager = getattr(self, "chat_manager", None)
        if agent is None or agent.manager is not manager:
            transcriber = self._recognizer() if hasattr(self, "_recognizer") else None
            agent = Agent(
                manager,
                store=AttachmentStore(runtime_root() / "attachments" / self.current_chat_id),
                transcriber=transcriber if getattr(transcriber, "available", False) else None,
            )
            agent.allow_commands = bool(getattr(self, "allow_console_commands", True))
            # the assistant remembers the conversation of this chat
            entries = self.chat_history_store.load()
            pairs, user = [], None
            for entry in entries:
                if entry.get("role") == "user":
                    user = entry.get("text", "")
                elif user is not None:
                    pairs.append((user, entry.get("text", "")))
                    user = None
            agent.history = pairs[-8:]
            self._agents[self.current_chat_id] = agent
        return agent

    # ------------------------------------------------------------ sidebar

    def _build_chat_sidebar(self, layout: QVBoxLayout) -> None:
        self.new_chat_btn = QPushButton("+  Новый чат", self.sidebar_frame)
        self.new_chat_btn.setObjectName("newChatButton")
        self.new_chat_btn.setCursor(Qt.PointingHandCursor)
        self.new_chat_btn.setToolTip("Новый чат (Ctrl+N)")
        self.new_chat_btn.clicked.connect(self._new_chat)
        layout.addWidget(self.new_chat_btn)

        self.chat_search = QLineEdit(self.sidebar_frame)
        self.chat_search.setObjectName("chatSearch")
        self.chat_search.setPlaceholderText("Поиск по чатам")
        self.chat_search.setClearButtonEnabled(True)
        self.chat_search.textChanged.connect(lambda _t: self._refresh_chat_list())
        layout.addWidget(self.chat_search)

        self.chat_list = QListWidget(self.sidebar_frame)
        self.chat_list.setObjectName("chatList")
        self.chat_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.chat_list.customContextMenuRequested.connect(self._chat_context_menu)
        self.chat_list.itemClicked.connect(lambda item: self._switch_chat(item.data(Qt.UserRole)))
        self.chat_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.chat_list, 1)

    def _refresh_chat_list(self) -> None:
        if not hasattr(self, "chat_list"):
            return
        query = self.chat_search.text() if hasattr(self, "chat_search") else ""
        self.chat_list.clear()
        for info in self.chat_sessions.ordered(query):
            item = QListWidgetItem(self.chat_list)
            item.setData(Qt.UserRole, info.id)
            row = QWidget()
            row.setObjectName("chatRow")
            row.setProperty("current", info.id == self.current_chat_id)
            box = QHBoxLayout(row)
            box.setContentsMargins(10, 6, 8, 6)
            box.setSpacing(6)
            title = QLabel(row)
            title.setObjectName("chatRowTitle")
            title.setToolTip(info.title)
            available = max(60, self.chat_list.viewport().width() - 78 - (14 if info.status else 0))
            title.setText(title.fontMetrics().elidedText(("📌 " if info.pinned else "") + info.title, Qt.ElideRight, available))
            box.addWidget(title, 1)
            mark = {"ok": "✓", "warn": "!"}.get(info.status, "")
            if mark:
                badge = QLabel(mark, row)
                badge.setObjectName("chatRowMarkOk" if info.status == "ok" else "chatRowMarkWarn")
                badge.setToolTip("Всё проверено" if info.status == "ok" else "Есть несоответствия или замечания")
                box.addWidget(badge)
            when = QLabel(_when(info.updated), row)
            when.setObjectName("chatRowTime")
            box.addWidget(when)
            item.setSizeHint(row.sizeHint())
            self.chat_list.setItemWidget(item, row)
            if info.id == self.current_chat_id:
                item.setSelected(True)
                self.chat_list.setCurrentItem(item)
        title = self.chat_sessions.chats.get(self.current_chat_id)
        if hasattr(self, "chat_title_label") and title is not None:
            self.chat_title_label.setText(title.title)

    def _chat_context_menu(self, pos) -> None:
        item = self.chat_list.itemAt(pos)
        if item is None:
            return
        chat_id = item.data(Qt.UserRole)
        info = self.chat_sessions.chats.get(chat_id)
        if info is None:
            return
        menu = QMenu(self)
        rename = menu.addAction("Переименовать")
        pin = menu.addAction("Открепить" if info.pinned else "Закрепить сверху")
        menu.addSeparator()
        delete = menu.addAction("Удалить чат")
        chosen = menu.exec_(self.chat_list.viewport().mapToGlobal(pos))
        if chosen is rename:
            text, ok = QInputDialog.getText(self, "Переименовать чат", "Название:", text=info.title)
            if ok:
                self.chat_sessions.rename(chat_id, text)
        elif chosen is pin:
            self.chat_sessions.set_pinned(chat_id, not info.pinned)
        elif chosen is delete:
            self._delete_chat(chat_id)
            return
        self._refresh_chat_list()

    # ------------------------------------------------------------ actions

    def _busy(self) -> bool:
        worker = getattr(self, "_agent_worker", None)
        streaming = getattr(self, "current_streaming_worker", None)
        return bool((worker is not None and worker.isRunning()) or (streaming is not None and streaming.isRunning()))

    def _new_chat(self) -> None:
        if self._busy():
            self.status.showMessage("Дождитесь ответа в текущем чате")
            return
        if self.chat_sessions.is_empty(self.current_chat_id):
            self._switch_workspace("chat")
            self.input_field.setFocus()
            return
        info = self.chat_sessions.create()
        self._switch_chat(info.id)

    def _switch_chat(self, chat_id: str) -> None:
        if not chat_id or chat_id not in self.chat_sessions.chats:
            return
        if chat_id == self.current_chat_id and self.chat_layout.count() > 1:
            self._switch_workspace("chat")
            return
        if self._busy():
            self.status.showMessage("Дождитесь ответа в текущем чате")
            self._refresh_chat_list()
            return
        self.current_chat_id = chat_id
        self.chat_history_store = self.chat_sessions.history(chat_id)
        if hasattr(self, "_pending_attachments"):
            self._pending_attachments = []
            self._refresh_attachment_bar()
        self._clear_chat_view()
        if not self._restore_chat_history():
            self._show_welcome_message()
        self._switch_workspace("chat")
        self._refresh_chat_list()
        self.input_field.setFocus()

    def _clear_chat_view(self) -> None:
        manager = getattr(self, "chat_manager", None)
        if manager is not None:
            manager.chat_history = []
        self._hide_typing_indicator()
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _delete_chat(self, chat_id: str) -> None:
        info = self.chat_sessions.chats.get(chat_id)
        if info is None:
            return
        reply = QMessageBox.question(self, "Удалить чат", f"Удалить чат «{info.title}»?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self.chat_sessions.delete(chat_id)
        self._agents.pop(chat_id, None)
        if chat_id == self.current_chat_id:
            remaining = self.chat_sessions.ordered()
            target = remaining[0] if remaining else self.chat_sessions.create()
            self.current_chat_id = ""
            self._switch_chat(target.id)
        else:
            self._refresh_chat_list()

    def _chat_activity(self, role: str, text: str) -> None:
        """Called on every recorded message: title, time and status mark of the chat."""
        status = None
        if role == "assistant":
            status = "warn" if ("⚠" in text or "✗" in text) else ("ok" if "✓" in text else None)
        self.chat_sessions.touch(self.current_chat_id, first_text=text if role == "user" else "", status=status)
        self._refresh_chat_list()
