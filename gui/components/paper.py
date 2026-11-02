"""Created documents as sheets of paper in the chat.

PaperCard — the first page in miniature: left click saves the file, right
click unfolds the DocumentViewer under it: every page side by side, and a
plain editor to correct the text before saving (the file is rebuilt in the
same format and with the same example styles).
"""

from __future__ import annotations

from typing import Callable, List, Optional

from PyQt5.QtCore import QRectF, QSizeF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap, QTextDocument
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

PAGE_W, PAGE_H = 794, 1123  # A4 at 96 dpi
_MARGIN = 64
_PAPER_CSS = (
    "body { color: #23221f; font-family: 'Times New Roman'; font-size: 19px; }"
    "h1 { font-size: 26px; } h2 { font-size: 22px; } h3 { font-size: 20px; }"
    "table { border-collapse: collapse; } td, th { border: 1px solid #8a857a; padding: 4px 6px; }"
)


def paragraphs(markdown: str) -> str:
    """Every line its own paragraph (as in the DOCX); tables and lists untouched.
    Plain Markdown would join «Дата: …» and «Объект: …» into one paragraph."""
    out = []
    for line in (markdown or "").splitlines():
        stripped = line.strip()
        prev = out[-1].strip() if out else ""
        block = stripped.startswith(("|", "- ", "* ", "• ")) or stripped[:3].rstrip(".)").isdigit()
        prev_block = prev.startswith(("|", "- ", "* ", "• "))
        if stripped and prev and not (block and prev_block):
            out.append("")
        out.append(line)
    return "\n".join(out)


def _document(markdown: str) -> QTextDocument:
    doc = QTextDocument()
    doc.setDefaultStyleSheet(_PAPER_CSS)
    doc.setDefaultFont(QFont("Times New Roman", 14))
    doc.setMarkdown(paragraphs(markdown))
    doc.setDocumentMargin(_MARGIN)
    doc.setPageSize(QSizeF(PAGE_W, PAGE_H))
    return doc


def render_pages(markdown: str, width: int, max_pages: int = 30) -> List[QPixmap]:
    """Pages of the document as pixmaps ``width`` px wide."""
    doc = _document(markdown)
    ratio = 2.0  # drawn at double density: crisp small print
    scale = width / PAGE_W  # painter works in logical pixels of the high-density pixmap
    pages = []
    for index in range(min(doc.pageCount(), max_pages)):
        pixmap = QPixmap(int(PAGE_W * scale * ratio), int(PAGE_H * scale * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(QColor("#fdfcf8"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.scale(scale, scale)
        painter.translate(0, -index * PAGE_H)
        doc.drawContents(painter, QRectF(0, index * PAGE_H, PAGE_W, PAGE_H))
        painter.end()
        pages.append(pixmap)
    return pages


class _Sheet(QLabel):
    clicked_left = pyqtSignal()
    clicked_right = pyqtSignal()

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.RightButton:
            self.clicked_right.emit()
        elif event.button() == Qt.LeftButton:
            self.clicked_left.emit()


class PaperCard(QFrame):
    """Miniature of the first page with the file name; LMB save, RMB unfold."""

    save_requested = pyqtSignal()
    toggle_requested = pyqtSignal()

    def __init__(self, title: str, label: str, markdown: str, parent=None):
        super().__init__(parent)
        self.setObjectName("paperCard")
        self.setCursor(Qt.PointingHandCursor)
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 8, 12, 8)
        row.setSpacing(12)
        self.sheet = _Sheet(self)
        self.sheet.setObjectName("paperSheet")
        self.set_markdown(markdown)
        self.sheet.clicked_left.connect(self.save_requested.emit)
        self.sheet.clicked_right.connect(self.toggle_requested.emit)
        self.sheet.setToolTip("Левый клик — скачать, правый — развернуть по листам")
        row.addWidget(self.sheet, 0, Qt.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.title = QLabel(title, self)
        self.title.setObjectName("paperTitle")
        self.title.setWordWrap(True)
        self.meta = QLabel(label, self)
        self.meta.setObjectName("paperMeta")
        hint = QLabel("левый клик — скачать · правый — развернуть и править", self)
        hint.setObjectName("paperHint")
        hint.setWordWrap(True)
        text.addWidget(self.title)
        text.addWidget(self.meta)
        text.addStretch(1)
        text.addWidget(hint)
        row.addLayout(text, 1)

    def set_markdown(self, markdown: str) -> None:
        pages = render_pages(markdown, 96, max_pages=1)
        if pages:
            self.sheet.setPixmap(pages[0])
        self.pages = None

    def mousePressEvent(self, event):  # noqa: N802 - the whole card reacts like the sheet
        if event.button() == Qt.RightButton:
            self.toggle_requested.emit()
        elif event.button() == Qt.LeftButton:
            self.save_requested.emit()


class DocumentViewer(QFrame):
    """All pages side by side; «Править» switches to a text editor."""

    def __init__(self, markdown: str, on_save: Callable[[str], None], on_download: Callable[[], None], parent=None):
        super().__init__(parent)
        self.setObjectName("documentViewer")
        self.markdown = markdown
        self.on_save = on_save
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        bar = QHBoxLayout()
        self.caption = QLabel(self)
        self.caption.setObjectName("paperMeta")
        self.edit_btn = QPushButton("Править текст", self)
        self.edit_btn.setObjectName("fileActionButton")
        self.edit_btn.clicked.connect(self._toggle_edit)
        self.save_btn = QPushButton("Сохранить правки", self)
        self.save_btn.setObjectName("commandRunButton")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.hide()
        download = QPushButton("Скачать", self)
        download.setObjectName("fileActionButton")
        download.clicked.connect(on_download)
        bar.addWidget(self.caption, 1)
        bar.addWidget(self.edit_btn)
        bar.addWidget(self.save_btn)
        bar.addWidget(download)
        layout.addLayout(bar)

        self.strip = QScrollArea(self)
        self.strip.setObjectName("pageStrip")
        self.strip.setWidgetResizable(True)
        self.strip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.strip.setFixedHeight(int(PAGE_H * 440 / PAGE_W) + 24)
        layout.addWidget(self.strip)

        self.editor = QTextEdit(self)
        self.editor.setObjectName("paperEditor")
        self.editor.setAcceptRichText(True)
        self.editor.setMinimumHeight(420)
        self.editor.hide()
        layout.addWidget(self.editor)
        self._render()

    def _render(self) -> None:
        holder = QWidget()
        holder.setObjectName("pageHolder")
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        pages = render_pages(self.markdown, 440)
        for index, pixmap in enumerate(pages, 1):
            sheet = QLabel(holder)
            sheet.setObjectName("paperSheet")
            sheet.setPixmap(pixmap)
            sheet.setToolTip(f"Лист {index}")
            row.addWidget(sheet)
        row.addStretch(1)
        self.strip.setWidget(holder)
        count = len(pages)
        self.caption.setText(f"{count} лист" + ("" if count == 1 else ("а" if 2 <= count <= 4 else "ов")))

    def _toggle_edit(self) -> None:
        editing = not self.editor.isVisible()
        if editing:
            self.editor.document().setDefaultStyleSheet(_PAPER_CSS)
            self.editor.setMarkdown(paragraphs(self.markdown))
        self.editor.setVisible(editing)
        self.strip.setVisible(not editing)
        self.save_btn.setVisible(editing)
        self.edit_btn.setText("Отменить правку" if editing else "Править текст")

    def _save(self) -> None:
        markdown = self.editor.toMarkdown()
        self.on_save(markdown)
        self.markdown = markdown
        self._render()
        self._toggle_edit()
