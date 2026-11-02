"""Small vector icons drawn in the theme colour (emoji glyphs of the system
font are monochrome and nearly invisible on dark themes)."""

from __future__ import annotations

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

_SIZE = 48  # drawn large, scaled down by Qt (crisp on HiDPI)


def _canvas():
    pixmap = QPixmap(_SIZE, _SIZE)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    return pixmap, painter


def _pen(color: str, width: float = 3.6) -> QPen:
    pen = QPen(QColor(color))
    pen.setWidthF(width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def paperclip(color: str) -> QIcon:
    pixmap, p = _canvas()
    p.setPen(_pen(color))
    path = QPainterPath()
    # outer loop down, round bottom, up, round top, inner line down
    path.moveTo(30, 16)
    path.lineTo(30, 33)
    path.arcTo(QRectF(18, 26, 12, 14), 0, -180)
    path.lineTo(18, 12)
    path.arcTo(QRectF(18, 5, 16, 14), 180, -180)
    path.lineTo(34, 34)
    path.arcTo(QRectF(14, 24, 20, 20), 0, -180)
    path.lineTo(14, 16)
    p.translate(2, 1)
    p.drawPath(path)
    p.end()
    return QIcon(pixmap)


def microphone(color: str) -> QIcon:
    pixmap, p = _canvas()
    p.setPen(_pen(color))
    p.setBrush(QColor(color))
    p.drawRoundedRect(QRectF(18, 6, 12, 22), 6, 6)
    p.setBrush(Qt.NoBrush)
    arc = QPainterPath()
    arc.moveTo(12, 22)
    arc.arcTo(QRectF(12, 12, 24, 22), 180, 180)
    p.drawPath(arc)
    p.drawLine(QPointF(24, 34), QPointF(24, 41))
    p.drawLine(QPointF(17, 42), QPointF(31, 42))
    p.end()
    return QIcon(pixmap)


def speaker(color: str, on: bool) -> QIcon:
    pixmap, p = _canvas()
    p.setPen(_pen(color))
    p.setBrush(QColor(color))
    body = QPainterPath()
    body.moveTo(7, 19)
    body.lineTo(15, 19)
    body.lineTo(25, 10)
    body.lineTo(25, 38)
    body.lineTo(15, 29)
    body.lineTo(7, 29)
    body.closeSubpath()
    p.drawPath(body)
    p.setBrush(Qt.NoBrush)
    if on:
        p.drawArc(QRectF(22, 16, 12, 16), -60 * 16, 120 * 16)
        p.drawArc(QRectF(22, 10, 20, 28), -60 * 16, 120 * 16)
    else:
        p.drawLine(QPointF(31, 18), QPointF(41, 30))
        p.drawLine(QPointF(41, 18), QPointF(31, 30))
    p.end()
    return QIcon(pixmap)


def document(color: str) -> QIcon:
    pixmap, p = _canvas()
    p.setPen(_pen(color))
    path = QPainterPath()
    path.moveTo(12, 5)
    path.lineTo(29, 5)
    path.lineTo(37, 13)
    path.lineTo(37, 43)
    path.lineTo(12, 43)
    path.closeSubpath()
    p.drawPath(path)
    for y in (20, 27, 34):
        p.drawLine(QPointF(18, y), QPointF(31, y))
    p.end()
    return QIcon(pixmap)


def database(color: str) -> QIcon:
    pixmap, p = _canvas()
    p.setPen(_pen(color))
    p.drawEllipse(QRectF(10, 5, 28, 10))
    p.drawLine(QPointF(10, 10), QPointF(10, 38))
    p.drawLine(QPointF(38, 10), QPointF(38, 38))
    p.drawArc(QRectF(10, 19, 28, 10), 180 * 16, 180 * 16)
    p.drawArc(QRectF(10, 33, 28, 10), 180 * 16, 180 * 16)
    p.end()
    return QIcon(pixmap)


def pulse(color: str) -> QIcon:
    pixmap, p = _canvas()
    p.setPen(_pen(color))
    path = QPainterPath()
    path.moveTo(5, 26)
    for x, y in ((14, 26), (19, 13), (26, 37), (31, 20), (35, 26), (43, 26)):
        path.lineTo(x, y)
    p.drawPath(path)
    p.end()
    return QIcon(pixmap)


def gear(color: str) -> QIcon:
    import math

    pixmap, p = _canvas()
    p.setPen(_pen(color))
    p.drawEllipse(QRectF(18, 18, 12, 12))
    for i in range(8):
        a = math.pi * 2 * i / 8
        p.drawLine(QPointF(24 + 11 * math.cos(a), 24 + 11 * math.sin(a)), QPointF(24 + 17 * math.cos(a), 24 + 17 * math.sin(a)))
    p.drawEllipse(QRectF(12, 12, 24, 24))
    p.end()
    return QIcon(pixmap)