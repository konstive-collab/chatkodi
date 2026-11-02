"""«Ход работы» inside the answer bubble.

While the assistant works, the current step shimmers (a light sweep runs over
the text) and finished steps get a check mark. When the answer is ready the
trail collapses into one line «Ход работы · 4 шага · 7,4 с ▸» that expands to
the steps with their timings.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from PyQt5.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPen
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class ShimmerLabel(QWidget):
    """Text with a light sweep running over it (the step in progress)."""

    def __init__(self, text: str, base: str, light: str, parent=None, family: str = "Onest"):
        super().__init__(parent)
        self._text = text
        self.base = QColor(base)
        self.light = QColor(light)
        self.family = family
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)
        self.setMinimumHeight(20)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self._text = text
        self.updateGeometry()
        self.update()

    def stop(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.022) % 1.6
        self.update()

    def sizeHint(self):  # noqa: N802
        metrics = QFontMetrics(QFont(self.family, 10))
        return metrics.size(0, self._text + "   00,0 с")

    def paintEvent(self, _event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.TextAntialiasing)
        font = QFont(self.family, 10)
        font.setWeight(QFont.Medium)
        painter.setFont(font)
        width = max(1, self.width())
        center = (self._phase - 0.3) * width
        gradient = QLinearGradient(center - width * 0.25, 0, center + width * 0.25, 0)
        gradient.setColorAt(0.0, self.base)
        gradient.setColorAt(0.5, self.light)
        gradient.setColorAt(1.0, self.base)
        painter.setPen(QPen(gradient, 1))
        painter.drawText(QRectF(0, 0, width, self.height()), Qt.AlignLeft | Qt.AlignVCenter, self._text)


class ProgressTrail(QFrame):
    expanded_changed = pyqtSignal(bool)

    def __init__(self, palette: dict, parent=None, family: str = "Onest"):
        super().__init__(parent)
        self.setObjectName("progressTrail")
        self.palette_ = palette
        self.family = family
        self.started = time.perf_counter()
        self.steps: List[Tuple[str, float]] = []
        self.finished = False
        self.expanded = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(2)
        self.done_label = QLabel(self)
        self.done_label.setObjectName("progressDone")
        self.done_label.setTextFormat(Qt.RichText)
        self.done_label.hide()
        self.current = ShimmerLabel("Думаю…", palette.get("muted", "#94a3b8"), palette.get("title", "#ffffff"), self, family)
        self.summary = QLabel(self)
        self.summary.setObjectName("progressSummary")
        self.summary.setCursor(Qt.PointingHandCursor)
        self.summary.hide()
        self.summary.mousePressEvent = lambda _e: self.toggle()
        layout.addWidget(self.done_label)
        layout.addWidget(self.current)
        layout.addWidget(self.summary)
        self._clock = QTimer(self)
        self._clock.timeout.connect(self._refresh_current)
        self._clock.start(200)
        self._current_text = "Думаю…"

    def step(self, text: str) -> None:
        text = (text or "").strip().rstrip("…").rstrip(".") or "Работаю"
        if self.finished:
            return
        now = time.perf_counter() - self.started
        if self.steps and self.steps[-1][0] == text:
            return
        self.steps.append((text, now))
        self._current_text = text + "…"
        self._render_done()
        self._refresh_current()

    def _refresh_current(self) -> None:
        elapsed = time.perf_counter() - self.started
        self.current.setText(f"{self._current_text}  {elapsed:.1f} с".replace(".", ","))

    def _render_done(self) -> None:
        accent = self.palette_.get("accent", "#22c55e")
        muted = self.palette_.get("muted", "#94a3b8")
        done = self.steps[:-1] if not self.finished else self.steps
        if not done:
            self.done_label.hide()
            return
        lines = []
        for index, (text, at) in enumerate(done):
            nxt = done[index + 1][1] if index + 1 < len(done) else (self.total if self.finished else at)
            took = max(0.0, nxt - at)
            seconds = f"{took:.1f} с".replace(".", ",")
            lines.append(f'<span style="color:{accent};">✓</span> <span style="color:{muted};">{text} · {seconds}</span>')
        self.done_label.setText("<br>".join(lines))
        self.done_label.setVisible(not self.finished or self.expanded)

    def finish(self, total_seconds: Optional[float] = None) -> None:
        self.finished = True
        self.total = float(total_seconds if total_seconds is not None else time.perf_counter() - self.started)
        self._clock.stop()
        self.current.stop()
        self.current.hide()
        count = len(self.steps)
        word = "шаг" if count % 10 == 1 and count % 100 != 11 else ("шага" if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14 else "шагов")
        self._summary_text = f"Ход работы · {count} {word} · {self.total:.1f} с".replace(".", ",")
        self.summary.setText(self._summary_text + "  ▸")
        self.summary.setVisible(count > 0)
        self._render_done()

    def toggle(self) -> None:
        self.expanded = not self.expanded
        self.summary.setText(self._summary_text + ("  ▾" if self.expanded else "  ▸"))
        self.done_label.setVisible(self.expanded)
        self.expanded_changed.emit(self.expanded)
