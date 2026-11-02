"""Files the user drops into the chat.

An attachment is copied into the session folder (the original is never
touched), read into text once — documents with the index readers, photos and
scans with the vision model — and then lives in the conversation context.
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from core.agent import vision
from core.agent.coding import SOURCE_EXTENSIONS, read_source
from core.agent.table_check import check_tables
from core.logging_utils import get_logger
from core.paths import runtime_root

logger = get_logger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".opus", ".wma"}
# A PDF page with less text than this is treated as a scan.
_SCAN_CHARS_PER_PAGE = 40


@dataclass
class Attachment:
    id: str
    name: str
    path: Path
    kind: str  # image | document | audio | other
    text: str = ""
    error: str = ""
    pages: int = 0
    seconds: float = 0.0
    checks: List[str] = field(default_factory=list)
    recognized: bool = False  # text came from the vision model, not from a text layer
    read: bool = False
    doubts: List[Any] = field(default_factory=list)  # vision.CellDoubt: numbers the model was unsure of

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())

    def describe(self) -> str:
        kinds = {"image": "фото/изображение", "document": "документ", "audio": "аудио", "other": "файл"}
        return f"{self.name} ({kinds.get(self.kind, 'файл')})"


def kind_of(path: Path) -> str:
    from core.documents.processor import SUPPORTED_EXTENSIONS

    ext = path.suffix.lower()
    if ext in vision.IMAGE_EXTENSIONS:
        return "image"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in SUPPORTED_EXTENSIONS or ext in SOURCE_EXTENSIONS:
        return "document"
    return "other"


def is_supported(path: Path) -> bool:
    return kind_of(Path(path)) != "other"


class AttachmentStore:
    """Attachments of one chat session."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or (runtime_root() / "attachments" / time.strftime("%Y%m%d_%H%M%S"))
        self.items: List[Attachment] = []

    def add(self, source: Path) -> Attachment:
        source = Path(source)
        self.root.mkdir(parents=True, exist_ok=True)
        att_id = uuid.uuid4().hex[:8]
        target = self.root / f"{att_id}_{source.name}"
        shutil.copy2(source, target)
        attachment = Attachment(id=att_id, name=source.name, path=target, kind=kind_of(source))
        self.items.append(attachment)
        return attachment

    def add_bytes(self, data: bytes, name: str) -> Attachment:
        self.root.mkdir(parents=True, exist_ok=True)
        att_id = uuid.uuid4().hex[:8]
        target = self.root / f"{att_id}_{name}"
        target.write_bytes(data)
        attachment = Attachment(id=att_id, name=name, path=target, kind=kind_of(target))
        self.items.append(attachment)
        return attachment

    def get(self, att_id: str) -> Optional[Attachment]:
        return next((a for a in self.items if a.id == att_id), None)

    def remove(self, att_id: str) -> None:
        self.items = [a for a in self.items if a.id != att_id]

    def clear(self) -> None:
        self.items = []


def read_attachment(attachment: Attachment, llm: Any, *, transcriber: Any = None,
                    progress: Optional[Callable[[str], None]] = None) -> Attachment:
    """Fill ``attachment.text`` (once)."""
    if attachment.read:
        return attachment
    started = time.perf_counter()
    try:
        if attachment.kind == "image":
            if not getattr(llm, "supports_vision", False):
                attachment.error = "Распознавание изображений недоступно: модель зрения не загружена."
            else:
                if progress:
                    progress(f"Распознаю изображение «{attachment.name}»…")
                attachment.text, cells = vision.read_image(llm, attachment.path.read_bytes(), progress)
                attachment.doubts = vision.doubtful(cells)
                attachment.recognized = True
                attachment.pages = 1
        elif attachment.kind == "audio":
            if transcriber is None:
                attachment.error = "Распознавание речи недоступно."
            else:
                if progress:
                    progress(f"Распознаю аудио «{attachment.name}»…")
                attachment.text = transcriber.transcribe_file(attachment.path)
                attachment.recognized = True
        elif attachment.kind == "document" and attachment.path.suffix.lower() in SOURCE_EXTENSIONS:
            # Source code must retain indentation, HTML tags and script bodies.
            attachment.text = read_source(attachment.path)
        elif attachment.kind == "document":
            from core.documents.processor import read_file_text

            if progress:
                progress(f"Читаю «{attachment.name}»…")
            if attachment.path.suffix.lower() == ".pdf":
                from core.agent.pdf_tables import read_pdf

                reading = read_pdf(attachment.path, llm, progress)
                text = reading.text
                attachment.pages = reading.pages
                attachment.recognized = reading.recognized
                attachment.error = "; ".join(reading.issues)
            else:
                # Keep the empty cells that scoring and Excel need.
                text = _tables_as_markdown(attachment.path) or read_file_text(attachment.path)
            attachment.text = text
            if not text.strip() and not attachment.error:
                attachment.error = "Не удалось извлечь текст из файла."
        else:
            attachment.error = "Этот тип файла не поддерживается."
    except Exception as exc:
        logger.exception("Attachment %s could not be read", attachment.name)
        attachment.error = f"Не удалось прочитать файл: {exc}"
    attachment.checks = check_tables(attachment.text) if attachment.text else []
    attachment.seconds = time.perf_counter() - started
    attachment.read = True
    logger.info(
        "Attachment read: name=%s kind=%s chars=%s recognized=%s seconds=%.1f error=%s",
        attachment.name, attachment.kind, len(attachment.text), attachment.recognized, attachment.seconds, attachment.error or "-",
    )
    return attachment


def _cell(value: Any) -> str:
    """An Excel value as a person reads it: 8.4 -> «8,4», 225.0 -> «225»."""
    if value is None:
        return ""
    if isinstance(value, (dt.time, dt.timedelta)):
        seconds = (value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
                   if isinstance(value, dt.time) else value.total_seconds())
        minutes, seconds = divmod(seconds, 60)
        shown = f"{seconds:06.3f}".rstrip("0").rstrip(".")
        return f"{int(minutes)}:{shown}".replace(".", ",")
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value).replace(".", ",")
    if hasattr(value, "strftime"):
        return value.strftime("%d.%m.%Y" if not getattr(value, "hour", 0) and not getattr(value, "minute", 0) else "%d.%m.%Y %H:%M")
    return re.sub(r"\s+", " ", str(value)).strip().replace("|", "/")


def _markdown_block(rows: List[List[str]]) -> List[str]:
    """Rows of one block: a Markdown table when it has a header and data, else lines."""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    if len(rows) < 2 or width < 2:
        return [" ".join(c for c in r if c) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    return lines + ["| " + " | ".join(r) + " |" for r in rows[1:]]


def _fitness_header(rows: List[List[str]]) -> List[List[str]]:
    """Flatten the multi-row/merged header of an exercise results sheet.

    Keep metadata from vertically merged cells and disambiguate repeated
    exercise/result/points groups. The numeric column-index row is not data.
    Ordinary worksheets and standards tables are left as they are.
    """
    for leaf, row in enumerate(rows[:12]):
        if sum(bool(re.fullmatch(r"Результат", c, re.I)) for c in row) < 2:
            continue
        if not any(re.search(r"№\s*упражнен", c, re.I) for c in row):
            continue
        width = max(map(len, rows))
        header = []
        counts = {}
        for c in range(width):
            text = next((rows[r][c] for r in range(leaf, -1, -1)
                         if c < len(rows[r]) and rows[r][c]), "")
            if re.search(r"№\s*упражнен|^Результат$|^Балл[ы]?$", text, re.I):
                key = text.lower()
                counts[key] = counts.get(key, 0) + 1
                text = f"{text} [{counts[key]}]"
            header.append(text)
        start = leaf + 1
        if start < len(rows) and all(not v or re.fullmatch(r"\d+", v) for v in rows[start]):
            start += 1
        return [header] + rows[start:]
    return rows


def _tables_as_markdown(path: Path) -> str:
    """Excel sheets, CSV files and Word documents with tables as text with
    Markdown tables; "" for other files (the index reader is used then)."""
    ext = path.suffix.lower()
    try:
        if ext in {".xlsx", ".xlsm", ".xls"}:
            if ext == ".xls":
                import xlrd

                book = xlrd.open_workbook(str(path))
                sheets = [(s.name, [s.row_values(r) for r in range(s.nrows)]) for s in book.sheets()]
            else:
                import openpyxl

                wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
                try:
                    sheets = [(ws.title, list(ws.iter_rows(values_only=True))) for ws in wb.worksheets]
                finally:
                    wb.close()
            out: List[str] = []
            for title, values in sheets:
                rows = [[_cell(v if v != "" else None) for v in row] for row in values]
                while rows and not any(rows[-1]):
                    rows.pop()
                if not rows:
                    continue
                # trailing empty columns are formatting, not data
                width = max((max((i + 1 for i, v in enumerate(r) if v), default=0) for r in rows), default=0)
                rows = [r[:width] for r in rows]
                out.append(f"## Лист «{title}»")
                block: List[List[str]] = []
                for row in rows + [[]]:
                    if any(row):
                        block.append(row)
                    elif block:
                        out += _markdown_block(_fitness_header(block)) + [""]
                        block = []
            return "\n".join(out).strip()
        if ext in {".csv", ".tsv"}:
            import csv

            raw = path.read_bytes()
            text = next((raw.decode(enc) for enc in ("utf-8-sig", "cp1251") if _decodes(raw, enc)), "")
            dialect = csv.Sniffer().sniff(text[:2000], delimiters=";,\t") if text.strip() else csv.excel
            rows = [[_cell(c) for c in row] for row in csv.reader(text.splitlines(), dialect)]
            rows = [r for r in rows if any(r)]
            return "\n".join(_markdown_block(rows)) if rows else ""
        if ext == ".docx":
            from docx import Document

            if not Document(str(path)).tables:
                return ""  # plain text documents keep the index reader (headers, footers)
            from core.agent.outputs import docx_to_markdown

            return docx_to_markdown(path)
    except Exception as exc:
        logger.warning("Tables of %s could not be read: %s", path.name, exc)
    return ""


def _decodes(raw: bytes, encoding: str) -> bool:
    try:
        raw.decode(encoding)
        return True
    except UnicodeDecodeError:
        return False


def _pdf_page_count(path: Path) -> int:
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            return len(pdf)
        finally:
            pdf.close()
    except Exception:
        return 0
