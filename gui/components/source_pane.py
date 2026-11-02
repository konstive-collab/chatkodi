"""The source document beside the chat, with the quoted place highlighted.

Opens to the right of the conversation (the chat stays visible): the
document text on a light sheet, every quote of the answer marked, the
first one scrolled into the middle. «Следующая» walks through the marks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QTextCharFormat, QTextCursor, QTextDocument
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout


def _variants(quote: str) -> List[str]:
    """The quote as is, then shorter stable pieces (answers may trim or join lines)."""
    clean = re.sub(r"\s+", " ", (quote or "").strip(" «»\"'.…"))
    out: List[str] = []

    def push(term: str) -> None:
        term = term.strip(" «»\"'.…,;:")
        if len(term) >= 12 and term not in out:
            out.append(term)

    push(clean)
    for part in re.split(r"\.\.\.|…|;|\n", clean):
        push(part)
    words = clean.split()
    if len(words) >= 7:
        push(" ".join(words[:8]))
        push(" ".join(words[-8:]))
    return out[:8]


class SourcePane(QFrame):
    closed = pyqtSignal()

    def __init__(self, parent=None, on_open_external=None):
        super().__init__(parent)
        self.setObjectName("sourcePane")
        self.on_open_external = on_open_external
        self.path = None
        self.marks: List[QTextCursor] = []
        self.index = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        head = QHBoxLayout()
        self.title = QLabel(self)
        self.title.setObjectName("sourcePaneTitle")
        self.title.setWordWrap(True)
        close = QPushButton("✕", self)
        close.setObjectName("sourcePaneClose")
        close.setFixedWidth(30)
        close.setCursor(Qt.PointingHandCursor)
        close.clicked.connect(lambda: (self.hide(), self.closed.emit()))
        head.addWidget(self.title, 1)
        head.addWidget(close, 0, Qt.AlignTop)
        layout.addLayout(head)
        bar = QHBoxLayout()
        self.counter = QLabel(self)
        self.counter.setObjectName("sourcePaneHint")
        self.next_btn = QPushButton("Следующая ↓", self)
        self.next_btn.setObjectName("fileActionButton")
        self.next_btn.clicked.connect(self._next)
        external = QPushButton("Открыть файл", self)
        external.setObjectName("fileActionButton")
        external.clicked.connect(lambda: self.on_open_external and self.path and self.on_open_external(self.path))
        bar.addWidget(self.counter, 1)
        bar.addWidget(self.next_btn)
        bar.addWidget(external)
        layout.addLayout(bar)
        self.view = QTextEdit(self)
        self.view.setObjectName("sourcePaneText")
        self.view.setReadOnly(True)
        layout.addWidget(self.view, 1)
        self.hide()

    def show_document(self, path: Path, title: str, text: str, quotes: List[str]) -> int:
        self.path = Path(path)
        self.title.setText(title)
        self.view.setPlainText(text or "")
        fmt = QTextCharFormat()
        fmt.setBackground(QColor("#fde68a"))
        fmt.setForeground(QColor("#1c1917"))
        document = self.view.document()
        self.marks, seen = [], set()
        for quote in quotes:
            for term in _variants(quote):
                cursor = document.find(term, 0)
                if cursor.isNull():
                    cursor = document.find(term, 0, QTextDocument.FindFlags())
                if not cursor.isNull():
                    if cursor.selectionStart() not in seen:
                        seen.add(cursor.selectionStart())
                        self.marks.append(cursor)
                    break
        self.marks.sort(key=lambda c: c.selectionStart())
        selections = []
        for cursor in self.marks:
            selection = QTextEdit.ExtraSelection()
            selection.cursor = cursor
            selection.format = fmt
            selections.append(selection)
        self.view.setExtraSelections(selections)
        self.index = 0
        self.show()
        self._focus()
        return len(self.marks)

    def _focus(self) -> None:
        count = len(self.marks)
        self.next_btn.setVisible(count > 1)
        if not count:
            self.counter.setText("Цитата не найдена в тексте — показан документ целиком")
            self.view.moveCursor(QTextCursor.Start)
            return
        self.counter.setText(f"Цитата {self.index + 1} из {count}")
        cursor = QTextCursor(self.marks[self.index])
        cursor.clearSelection()
        self.view.setTextCursor(cursor)
        # put the mark in the middle of the view
        rect = self.view.cursorRect()
        bar = self.view.verticalScrollBar()
        bar.setValue(bar.value() + rect.top() - self.view.viewport().height() // 2)

    def _next(self) -> None:
        if self.marks:
            self.index = (self.index + 1) % len(self.marks)
            self._focus()
