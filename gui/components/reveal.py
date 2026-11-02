"""Answers appear word by word: the finished, verified text is revealed in
small word groups with a soft caret, within ~0.6–2.4 s whatever the length."""

from __future__ import annotations

import re
from typing import Callable, Optional

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

_TOKEN = re.compile(r"\S+\s*|\s+")


class TextRevealer(QObject):
    progressed = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, label, text: str, render: Callable[[str], str], caret_color: str = "#22c55e", parent=None):
        super().__init__(parent or label)
        self.label = label
        self.text = text or ""
        self.render = render
        self.caret = f'<span style="color:{caret_color};">&nbsp;▍</span>'
        self.tokens = _TOKEN.findall(self.text)
        words = max(1, len(self.tokens))
        # 30 ms ticks; short answers ~0.6 s, long ones capped at ~2.4 s
        ticks = max(20, min(80, words // 2))
        self.per_tick = max(1, -(-words // ticks))
        self.shown = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    def start(self) -> None:
        if not self.tokens:
            self._done()
            return
        self.timer.start(30)

    def skip(self) -> None:
        self.shown = len(self.tokens)
        self._tick()

    def _tick(self) -> None:
        self.shown = min(len(self.tokens), self.shown + self.per_tick)
        partial = "".join(self.tokens[: self.shown])
        try:
            if self.shown >= len(self.tokens):
                self._done()
                return
            html = self.render(partial)
            # caret at the end of the rendered text
            html = html[:-6] + self.caret + html[-6:] if html.endswith("</div>") else html + self.caret
            self.label.setText(html)
            self.progressed.emit()
        except RuntimeError:  # the bubble was deleted (chat switched)
            self.timer.stop()

    def _done(self) -> None:
        self.timer.stop()
        try:
            self.label.setText(self.render(self.text))
        except RuntimeError:
            return
        self.progressed.emit()
        self.finished.emit()
