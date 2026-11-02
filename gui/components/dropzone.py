"""Overlay shown while files are dragged over the chat: a dashed frame and
«Отпустите — приложу к чату», fading in and out."""

from __future__ import annotations

from PyQt5.QtCore import QPropertyAnimation, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsOpacityEffect, QWidget


class DropOverlay(QWidget):
    def __init__(self, parent: QWidget, palette: dict, family: str = "Onest"):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.palette_ = palette
        self.family = family
        self.count = 0
        self.names = ""
        self._effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._effect)
        self._fade = QPropertyAnimation(self._effect, b"opacity", self)
        self._fade.setDuration(160)
        self.hide()

    def set_palette(self, palette: dict) -> None:
        self.palette_ = palette

    def show_for(self, count: int, names: str = "") -> None:
        self.count, self.names = count, names
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        if not self.isVisible():
            self._effect.setOpacity(0.0)
            self.show()
            self._fade.stop()
            self._fade.setStartValue(0.0)
            self._fade.setEndValue(1.0)
            self._fade.start()
        self.update()

    def hide_soon(self) -> None:
        self.hide()

    def paintEvent(self, _event):  # noqa: N802 - Qt override
        p = self.palette_
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bg = QColor(p.get("bg", "#020617"))
        bg.setAlpha(205)
        painter.fillRect(self.rect(), bg)
        accent = QColor(p.get("accent", "#22c55e"))
        tint = QColor(accent)
        tint.setAlpha(26)
        frame = QRectF(self.rect()).adjusted(18, 18, -18, -18)
        painter.setBrush(tint)
        pen = QPen(accent, 2.0, Qt.DashLine)
        pen.setDashPattern([6, 5])
        painter.setPen(pen)
        painter.drawRoundedRect(frame, 22, 22)
        painter.setPen(QColor(p.get("title", "#ffffff")))
        painter.setFont(QFont(self.family, 20, QFont.Bold))
        center = frame.center()
        painter.drawText(QRectF(frame.left(), center.y() - 40, frame.width(), 36), Qt.AlignCenter, "Отпустите — приложу к чату")
        painter.setPen(QColor(p.get("muted", "#94a3b8")))
        painter.setFont(QFont(self.family, 11))
        files = f"{self.count} файл" + ("" if self.count == 1 else ("а" if 2 <= self.count <= 4 else "ов"))
        detail = files + (f" · {self.names}" if self.names else "")
        painter.drawText(QRectF(frame.left() + 40, center.y() + 4, frame.width() - 80, 24), Qt.AlignCenter, detail)
