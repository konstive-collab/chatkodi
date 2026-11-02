"""Voice wave: three smooth sine layers whose height follows the real
microphone level (no bars), tapered to the edges, gliding continuously."""

from __future__ import annotations

import math

from PyQt5.QtCore import QPointF, Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QWidget


class VoiceWave(QWidget):
    def __init__(self, color: str = "#22c55e", parent=None):
        super().__init__(parent)
        self.setObjectName("voiceWave")
        self.color = QColor(color)
        self.level = 0.0
        self.target = 0.0
        self.phase = 0.0
        self.setFixedHeight(40)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def set_color(self, color: str) -> None:
        self.color = QColor(color)

    def set_level(self, value: float) -> None:
        self.target = max(0.06, min(1.0, value))

    def start(self) -> None:
        self.level, self.target = 0.06, 0.06
        self._timer.start(16)
        self.show()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self.level += (self.target - self.level) * 0.16
        self.target *= 0.97  # falls back to calm between words
        self.phase += 0.11
        self.update()

    def paintEvent(self, _event):  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2
        for layer, (freq, speed, alpha, width) in enumerate(((1.6, 1.0, 230, 2.2), (2.3, -1.3, 120, 1.6), (3.1, 0.7, 70, 1.2))):
            color = QColor(self.color)
            color.setAlpha(alpha)
            pen = QPen(color, width)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            path = QPainterPath()
            steps = 120
            for i in range(steps + 1):
                x = w * i / steps
                t = i / steps
                taper = math.sin(math.pi * t) ** 2  # quiet at the edges
                amp = (h / 2 - 3) * (self.level ** 0.6) * taper * (1.0 - 0.18 * layer)
                y = mid + amp * math.sin(2 * math.pi * freq * t + self.phase * speed + layer)
                if i == 0:
                    path.moveTo(QPointF(x, y))
                else:
                    path.lineTo(QPointF(x, y))
            painter.drawPath(path)
