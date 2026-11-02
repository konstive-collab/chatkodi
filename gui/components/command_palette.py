"""Ctrl+K: one window for actions, chats and library documents.

Empty query — frequent actions and recent chats. Typing filters all three
sections (every word must match, «ё» = «е»). Arrows move, Enter runs, Esc
closes; shortcuts are shown on the right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout, QWidget


@dataclass
class PaletteEntry:
    section: str
    title: str
    run: Callable[[], None]
    hint: str = ""
    keywords: str = ""


def _norm(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def matches(entry: PaletteEntry, query: str) -> bool:
    haystack = _norm(f"{entry.title} {entry.keywords} {entry.section}")
    return all(word in haystack for word in _norm(query).split())


class CommandPalette(QDialog):
    def __init__(self, parent, provider: Callable[[str], List[PaletteEntry]]):
        super().__init__(parent, Qt.FramelessWindowHint | Qt.Dialog)
        self.setObjectName("commandPalette")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.provider = provider
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        card = QFrame(self)
        card.setObjectName("paletteCard")
        outer.addWidget(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(6)
        self.search = QLineEdit(card)
        self.search.setObjectName("paletteSearch")
        self.search.setPlaceholderText("Команда, чат или документ…")
        self.search.textChanged.connect(self._refresh)
        self.search.installEventFilter(self)
        layout.addWidget(self.search)
        self.results = QListWidget(card)
        self.results.setObjectName("paletteResults")
        self.results.itemActivated.connect(self._run_item)
        self.results.itemClicked.connect(self._run_item)
        layout.addWidget(self.results)
        footer = QLabel("↑↓ выбрать · Enter выполнить · Esc закрыть", card)
        footer.setObjectName("paletteFooter")
        layout.addWidget(footer)
        self.entries: List[PaletteEntry] = []
        self.resize(640, 460)
        self._refresh("")

    def open_centered(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            geo = parent.geometry()
            self.move(geo.x() + (geo.width() - self.width()) // 2, geo.y() + 90)
        self.search.clear()
        self._refresh("")
        self.show()
        self.raise_()
        self.activateWindow()
        self.search.setFocus()

    def _refresh(self, query: str) -> None:
        self.results.clear()
        self.entries = []
        section = None
        for entry in self.provider(query):
            if entry.section != section:
                section = entry.section
                head = QListWidgetItem()
                head.setFlags(Qt.NoItemFlags)
                head.setData(Qt.UserRole, None)
                self.results.addItem(head)
                widget = QLabel(section.upper())
                widget.setObjectName("paletteSection")
                head.setSizeHint(widget.sizeHint())
                self.results.setItemWidget(head, widget)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, len(self.entries))
            self.entries.append(entry)
            row = QWidget()
            box = QHBoxLayout(row)
            box.setContentsMargins(10, 6, 10, 6)
            title = QLabel(entry.title, row)
            title.setObjectName("paletteTitle")
            box.addWidget(title, 1)
            if entry.hint:
                hint = QLabel(entry.hint, row)
                hint.setObjectName("paletteHint")
                box.addWidget(hint)
            item.setSizeHint(row.sizeHint())
            self.results.addItem(item)
            self.results.setItemWidget(item, row)
        self._select(0, 1)

    def _select(self, start: int, step: int) -> None:
        count = self.results.count()
        index = start
        for _ in range(count):
            if 0 <= index < count and self.results.item(index).data(Qt.UserRole) is not None:
                self.results.setCurrentRow(index)
                return
            index += step
            if index >= count:
                index = 0
            if index < 0:
                index = count - 1

    def eventFilter(self, obj, event):  # noqa: N802 - Qt override
        if obj is self.search and event.type() == QEvent.KeyPress:
            key = event.key()
            if key in (Qt.Key_Down, Qt.Key_Up):
                step = 1 if key == Qt.Key_Down else -1
                self._select(self.results.currentRow() + step, step)
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter):
                item = self.results.currentItem()
                if item is not None:
                    self._run_item(item)
                return True
            if key == Qt.Key_Escape:
                self.hide()
                return True
        return super().eventFilter(obj, event)

    def _run_item(self, item) -> None:
        index = item.data(Qt.UserRole)
        if index is None:
            return
        self.hide()
        self.entries[index].run()

    def changeEvent(self, event):  # noqa: N802 - closes when focus leaves
        if event.type() == QEvent.ActivationChange and not self.isActiveWindow():
            self.hide()
        super().changeEvent(event)
