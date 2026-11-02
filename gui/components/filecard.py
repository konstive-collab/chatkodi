"""A sent file as a card: miniature (the photo itself, the first PDF page or a
small sheet with the beginning of the text), name, kind and size; after the
assistant has read it, the outcome («распознано · арифметика ✓»)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

THUMB_W, THUMB_H = 54, 70
_KIND_NAMES = {"image": "Фото", "document": "Документ", "audio": "Аудио", "other": "Файл"}


def _human_size(size: int) -> str:
    for unit, step in (("ГБ", 1024 ** 3), ("МБ", 1024 ** 2), ("КБ", 1024)):
        if size >= step:
            return f"{size / step:.1f} {unit}".replace(".0 ", " ").replace(".", ",")
    return f"{size} Б"


def _text_sheet(lines, ext: str) -> QPixmap:
    pixmap = QPixmap(THUMB_W * 2, THUMB_H * 2)
    pixmap.fill(QColor("#fdfcf8"))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.TextAntialiasing)
    painter.setPen(QColor("#3a3833"))
    painter.setFont(QFont("Times New Roman", 7))
    y = 10
    for line in lines[:12]:
        painter.drawText(QRectF(8, y, THUMB_W * 2 - 16, 12), Qt.AlignLeft | Qt.TextSingleLine, line[:40])
        y += 10
    painter.setPen(QColor("#1f6b45"))
    painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
    painter.drawText(QRectF(0, THUMB_H * 2 - 22, THUMB_W * 2 - 8, 16), Qt.AlignRight, ext.upper())
    painter.end()
    return pixmap


def make_thumbnail(path: Path) -> QPixmap:
    from core.agent.attachments import kind_of

    kind = kind_of(path)
    ext = path.suffix.lower().lstrip(".")
    pixmap = QPixmap()
    try:
        if kind == "image":
            pixmap = QPixmap(str(path))
        elif ext == "pdf":
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(path))
            try:
                image = pdf[0].render(scale=0.6).to_pil()
            finally:
                pdf.close()
            from PIL.ImageQt import ImageQt

            pixmap = QPixmap.fromImage(ImageQt(image.convert("RGB")))
        elif kind == "document":
            from core.documents.processor import read_file_text

            lines = [l.strip() for l in read_file_text(path).splitlines() if l.strip()]
            pixmap = _text_sheet(lines, ext)
    except Exception:
        pixmap = QPixmap()
    if pixmap.isNull():
        pixmap = _text_sheet([], ext or "файл")
    return pixmap.scaled(THUMB_W, THUMB_H, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation).copy(0, 0, THUMB_W, THUMB_H)


class FileCard(QFrame):
    opened = pyqtSignal(str)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        from core.agent.attachments import kind_of

        self.path = Path(path)
        self.setObjectName("fileCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"{self.path}\nОткрыть файл")
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 6, 12, 6)
        row.setSpacing(10)
        thumb = QLabel(self)
        thumb.setObjectName("fileThumb")
        thumb.setFixedSize(THUMB_W, THUMB_H)
        thumb.setPixmap(make_thumbnail(self.path))
        row.addWidget(thumb)
        text = QVBoxLayout()
        text.setSpacing(1)
        name = QLabel(self)
        name.setObjectName("fileCardName")
        name.setText(name.fontMetrics().elidedText(self.path.name, Qt.ElideMiddle, 220))
        try:
            size = _human_size(self.path.stat().st_size)
        except OSError:
            size = ""
        meta = QLabel(f"{_KIND_NAMES.get(kind_of(self.path), 'Файл')} · {self.path.suffix.lstrip('.').upper()} · {size}", self)
        meta.setObjectName("fileCardMeta")
        self.status = QLabel("", self)
        self.status.setObjectName("fileCardStatus")
        self.status.hide()
        text.addStretch(1)
        text.addWidget(name)
        text.addWidget(meta)
        text.addWidget(self.status)
        text.addStretch(1)
        row.addLayout(text)

    def set_status(self, text: str, warn: bool = False) -> None:
        self.status.setText(text)
        self.status.setProperty("warn", warn)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.status.setVisible(bool(text))

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.LeftButton:
            self.opened.emit(str(self.path))
