"""Загрузка документов и подготовка чанков для ChatKodi.

Этот модуль читает файлы из корпуса, нормализует текст и
создаёт чанки, которые затем индексируют поиск и RAG-пайплайн.
"""

import os
import re
import json
import html
import fnmatch
import csv
import io
import shutil
import subprocess
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import time
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from datetime import datetime

from core.date_range_parser import parse_date_constraints
from core.logging_utils import get_logger
from core.paths import project_root, resolve_project_path


logger = get_logger(__name__)

# Служебные каталоги никогда не должны попадать в корпус документов.
# Если пользователь случайно выбрал корень проекта, без этого индексатор начинает
# читать собственные логи/кэши/support-bundle и может породить рекурсивную кашу.
_ALWAYS_IGNORED_DIR_NAMES = {
    '.chatkodi', '.git', '__pycache__', '.pytest_cache', '.mypy_cache',
    '.ruff_cache', '.venv', 'venv', 'env', 'node_modules',
}
_ALWAYS_IGNORED_FILE_NAMES = {'.chatkodiignore'}
_APP_IGNORED_TOP_LEVEL = {
    'release', 'dist', 'build', 'models_cache', 'manuals', 'tools',
    'scripts', 'core', 'gui', 'plugins', 'runtime', 'logs',
}
_IGNORED_DIR_PREFIXES = ('_bundle_tmp_', 'chatkodi_support_bundle_')

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 is not installed. PDF files will be skipped.")

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx is not installed. DOCX files will be skipped.")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    BS4_AVAILABLE = False

try:
    from striprtf.striprtf import rtf_to_text
    STRIPRTF_AVAILABLE = True
except ImportError:
    rtf_to_text = None
    STRIPRTF_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    openpyxl = None
    OPENPYXL_AVAILABLE = False

try:
    import xlrd
    XLRD_AVAILABLE = True
except ImportError:
    xlrd = None
    XLRD_AVAILABLE = False

try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    Presentation = None
    PPTX_AVAILABLE = False

try:
    from odf.opendocument import load as odf_load
    from odf import text as odf_text
    from odf import table as odf_table
    ODF_AVAILABLE = True
except ImportError:
    odf_load = None
    odf_text = None
    odf_table = None
    ODF_AVAILABLE = False


TEXT_EXTENSIONS = {
    '.txt', '.md', '.markdown', '.csv', '.tsv', '.log', '.json', '.jsonl',
    '.yaml', '.yml', '.xml', '.ini', '.cfg', '.conf', '.py', '.js', '.ts',
    '.java', '.cs', '.cpp', '.c', '.h', '.hpp', '.go', '.rs', '.php', '.rb',
    '.sql', '.bat', '.cmd', '.ps1', '.sh', '.zsh', '.toml', '.env', '.properties',
    '.rst', '.tex', '.ipynb',
}
HTML_EXTENSIONS = {'.html', '.htm', '.xhtml'}
MESSAGE_EXTENSIONS = {'.eml'}
OFFICE_TEXT_EXTENSIONS = {'.pdf', '.docx', '.doc', '.rtf', '.odt', '.ods', '.odp', '.xlsx', '.xlsm', '.xltx', '.xltm', '.xls', '.pptx', '.ppt', '.epub'}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | HTML_EXTENSIONS | MESSAGE_EXTENSIONS | OFFICE_TEXT_EXTENSIONS


class _SimpleHTMLTextExtractor(HTMLParser):
    """Минимальный HTML→text fallback без внешних зависимостей."""

    BLOCK_TAGS = {'p', 'div', 'br', 'li', 'tr', 'table', 'section', 'article', 'header', 'footer', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in {'script', 'style', 'noscript'}:
            self._skip_depth += 1
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append('\n')

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {'script', 'style', 'noscript'} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append('\n')

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        value = html.unescape(data or '').strip()
        if value:
            self.parts.append(value + ' ')

    def get_text(self) -> str:
        text = ''.join(self.parts)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()


_NUMERIC_CELL = re.compile(r"^[\s\d.,:%/+\-–—₽$€№#]*$")


def _looks_like_header_row(values: List[str]) -> bool:
    cells = [value for value in values if value]
    if len(cells) < 2 or len(cells) != len(values):
        return False
    if any(_NUMERIC_CELL.match(cell) for cell in cells):
        return False
    return sum(len(cell) for cell in cells) / len(cells) <= 40


def format_table_rows(rows: List[List[str]]) -> List[str]:
    """Render table rows so every row is understandable on its own.

    With a header row, data rows become ``Госномер: А123ВС 77 | Водитель: …``.
    Chunking may separate a row from the header; keyed rows keep the meaning
    for BM25, embeddings, fact lookup and the LLM alike.
    """
    cleaned = [[re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row] for row in rows]
    for row in cleaned:
        while row and not row[-1]:
            row.pop()
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return []
    header = cleaned[0] if len(cleaned) >= 2 and _looks_like_header_row(cleaned[0]) else None
    if header is None:
        return [" | ".join(row) for row in cleaned]
    lines = [" | ".join(header)]
    for row in cleaned[1:]:
        if row == header:
            continue
        parts = []
        for index, value in enumerate(row):
            if not value:
                continue
            key = header[index] if index < len(header) else ""
            parts.append(f"{key}: {value}" if key else value)
        if parts:
            lines.append(" | ".join(parts))
    return lines


def read_file_text(path: "str | Path") -> str:
    """Text of one file in any supported format (the same readers as the index)."""
    file_path = Path(path)
    reader = DocumentProcessor.__new__(DocumentProcessor)  # readers use no instance state
    content, ok = reader._read_file_content(file_path)
    return reader._clean_document_content(content, file_path.name) if ok else ""


# Представляет полностью загруженный исходный файл и лёгкие метаданные,
# которые интерфейс и поиск могут переиспользовать без повторного чтения с диска.
@dataclass
class Document:
    """Структура для хранения целого документа."""
    id: str
    path: str
    name: str  # Относительный путь от папки docs
    filename: str  # Только имя файла
    content: str  # Полный текст документа
    size_chars: int
    size_kb: float
    created_time: float
    modified_time: float
    extension: str
    is_valid: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Гарантирует, что ``metadata`` всегда является словарём для остального кода."""
        if self.metadata is None:
            self.metadata = {}

    def get_summary(self) -> str:
        """Возвращает короткое человекочитаемое описание документа."""
        lines_count = self.content.count('\n') + 1
        return (f"{self.name} "
                f"({self.size_kb:.1f} KB, {self.size_chars:,} символов, "
                f"{lines_count} строк)")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует dataclass в обычный словарь для интерфейса и поиска."""
        return asdict(self)

    def get_excerpt(self, max_length: int = 500) -> str:
        """Возвращает начало документа, ограниченное значением ``max_length``."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."


# Отвечает за загрузку корпуса, нормализацию текста и формирование чанков.
class DocumentProcessor:
    """
    Процессор для работы с целыми документами.
    ВАЖНО: теперь создаёт реальные чанки по всему документу (для точности RAG/поиска).
    """

    def __init__(
        self,
        docs_folder: str = "docs",
        recursive: bool = True,
        max_file_size_mb: int = 100
    ):
        self.docs_folder = resolve_project_path(docs_folder)
        self.app_root = project_root().resolve()
        self.recursive = recursive
        self.max_file_size_mb = max_file_size_mb
        self.ignore_patterns = self._load_ignore_patterns()

        # Процессор хранит и целые документы, и производные чанки,
        # потому что приложение работает и на уровне документов, и на уровне фрагментов.

        # Хранилища
        self.documents: List[Document] = []
        self.documents_by_id: Dict[str, Document] = {}
        self.documents_by_name: Dict[str, Document] = {}

        # Чанки
        self.text_chunks: List[Dict[str, Any]] = []

        # Настройки чанкинга (баланс точность/контекст)
        self.chunk_max_chars = 1400
        self.chunk_overlap = 220
        self.min_chunk_chars = 120

        # Статистика
        self.stats = {
            "total_documents": 0,
            "total_size_mb": 0.0,
            "total_chars": 0,
            "by_extension": {},
            "load_time_seconds": 0,
            "failed_files": [],
            "scan_warnings": [],
            "skipped_large_files": [],
        }

        os.makedirs(self.docs_folder, exist_ok=True)
        logger.info(
            "Document processor initialized: folder=%s recursive=%s max_file_size_mb=%s ignore_patterns=%s",
            self.docs_folder,
            recursive,
            max_file_size_mb,
            len(self.ignore_patterns),
        )

    def _load_ignore_patterns(self) -> List[str]:
        """Load optional corpus-owned exclusions from ``.chatkodiignore``."""
        path = self.docs_folder / ".chatkodiignore"
        if not path.is_file():
            return []
        try:
            patterns = []
            for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                patterns.append(line.replace("\\", "/"))
            return patterns
        except Exception as exc:
            logger.warning("Could not read corpus ignore file %s: %s", path, exc)
            return []

    def _matches_corpus_ignore(self, path: Path) -> bool:
        if not self.ignore_patterns:
            return False
        try:
            relative = path.relative_to(self.docs_folder).as_posix()
        except Exception:
            return False
        name = path.name
        for pattern in self.ignore_patterns:
            normalized = pattern.lstrip("/")
            if normalized.endswith("/"):
                prefix = normalized.rstrip("/")
                if relative == prefix or relative.startswith(prefix + "/"):
                    return True
                continue
            if fnmatch.fnmatchcase(relative, normalized):
                return True
            if "/" not in normalized and fnmatch.fnmatchcase(name, normalized):
                return True
        return False

    def _is_ignored_path(self, path: Path) -> bool:
        """Проверяет, относится ли путь к служебной зоне ChatKodi/окружения.

        Правило применяется и к каталогам, и к файлам. Это защита от двух
        неприятных случаев: пользователь выбрал корень проекта как папку
        документов; support bundle лежит внутри .chatkodi/logs/support.
        """
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path.absolute()

        if resolved.name.lower() in _ALWAYS_IGNORED_FILE_NAMES:
            return True

        try:
            parts = list(resolved.relative_to(self.docs_folder).parts)
        except Exception:
            parts = list(resolved.parts)
        for part in parts:
            lowered = str(part).lower()
            if lowered in _ALWAYS_IGNORED_DIR_NAMES:
                return True
            if any(lowered.startswith(prefix) for prefix in _IGNORED_DIR_PREFIXES):
                return True

        # Repository/runtime folders are ignored only when the selected corpus
        # actually contains the ChatKodi installation. An external user folder
        # named "manuals", "logs" or "core" is valid document content.
        try:
            app_relative = resolved.relative_to(self.app_root)
            if app_relative.parts and str(app_relative.parts[0]).lower() in _APP_IGNORED_TOP_LEVEL:
                return True
        except (ValueError, OSError):
            pass
        return self._matches_corpus_ignore(path)

    def _collect_all_files(self) -> List[Path]:
        """Сканирует папку документов и оставляет только поддерживаемые читаемые файлы."""
        files = []
        max_size = self.max_file_size_mb * 1024 * 1024
        scan_warnings: list[dict[str, str]] = []
        skipped_large: list[dict[str, object]] = []

        def on_walk_error(exc: OSError) -> None:
            scan_warnings.append({
                "path": str(getattr(exc, "filename", "") or self.docs_folder),
                "reason": type(exc).__name__,
                "message": str(exc),
            })

        if self.recursive:
            for root, dirnames, filenames in os.walk(self.docs_folder, onerror=on_walk_error, followlinks=False):
                root_path = Path(root)
                # Важно менять dirnames на месте: os.walk тогда вообще не спускается
                # в .chatkodi/logs/models_cache и временные bundle-папки.
                dirnames[:] = [
                    d for d in dirnames
                    if not self._is_ignored_path(root_path / d)
                ]
                if self._is_ignored_path(root_path):
                    continue
                for filename in filenames:
                    file_path = Path(root) / filename
                    if self._is_supported_file(file_path):
                        if self._is_ignored_path(file_path):
                            continue
                        try:
                            if file_path.stat().st_size <= max_size:
                                files.append(file_path)
                            else:
                                size_mb = file_path.stat().st_size / (1024 * 1024)
                                logger.warning("Skipping large file %s (%.1f MB)", filename, size_mb)
                                skipped_large.append({"path": str(file_path), "size_mb": round(size_mb, 2)})
                        except Exception as exc:
                            scan_warnings.append({"path": str(file_path), "reason": type(exc).__name__, "message": str(exc)})
                            continue
        else:
            try:
                root_items = list(self.docs_folder.iterdir())
            except Exception as exc:
                root_items = []
                scan_warnings.append({"path": str(self.docs_folder), "reason": type(exc).__name__, "message": str(exc)})
            for item in root_items:
                if item.is_file() and not self._is_ignored_path(item) and self._is_supported_file(item):
                    try:
                        if item.stat().st_size <= max_size:
                            files.append(item)
                        else:
                            skipped_large.append({"path": str(item), "size_mb": round(item.stat().st_size / (1024 * 1024), 2)})
                    except Exception as exc:
                        scan_warnings.append({"path": str(item), "reason": type(exc).__name__, "message": str(exc)})
                        continue

        self.stats["scan_warnings"] = scan_warnings[:100]
        self.stats["skipped_large_files"] = skipped_large[:100]
        return sorted(files)

    def _is_supported_file(self, filename: str | Path) -> bool:
        """Возвращает ``True``, если имя файла соответствует поддерживаемому типу документа.

        Индексатор должен быть либеральным на входе: пользователь может
        добавить не только DOCX/PDF, но и HTML-выгрузки, таблицы Excel,
        презентации, старые DOC/RTF/ODT и обычные логи. Если формат
        поддерживается только best-effort, файл всё равно попадёт в очередь,
        а конкретный reader уже решит, получилось ли извлечь текст.
        """
        filename_str = str(filename)
        ext = os.path.splitext(filename_str)[1].lower()

        if ext in SUPPORTED_EXTENSIONS:
            return True

        if ext == '':
            try:
                with open(filename_str, 'rb') as f:
                    sample = f.read(2048)
                    if not sample:
                        return False
                    if sample.count(b"\x00") / max(1, len(sample)) > 0.08:
                        return False
                    for encoding in ("utf-8", "cp1251"):
                        try:
                            decoded = sample.decode(encoding)
                        except UnicodeDecodeError:
                            continue
                        printable = sum(char.isprintable() or char in "\t\r\n" for char in decoded)
                        if printable / max(1, len(decoded)) >= 0.90:
                            return True
            except Exception:
                pass

        return False

    def _generate_document_id(self, file_path: Path) -> str:
        """Строит стабильный идентификатор из пути и текущих метаданных файловой системы."""
        try:
            stat = file_path.stat()
            unique_str = f"{file_path.absolute()}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(unique_str.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.md5(str(file_path.absolute()).encode()).hexdigest()[:16]

    def _read_plain_text(self, file_path: Path) -> Tuple[str, bool]:
        """Читает текстовый файл с набором популярных кодировок."""
        encodings = ['utf-8-sig', 'utf-8', 'cp1251', 'windows-1251', 'koi8-r', 'iso-8859-5', 'latin-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read(), True
            except UnicodeDecodeError:
                continue
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='replace'), True

    def _read_delimited_text(self, file_path: Path) -> Tuple[str, bool]:
        """Read CSV/TSV without destroying quoted delimiters or embedded text."""
        raw, ok = self._read_plain_text(file_path)
        if not ok or not raw:
            return "", False
        sample = raw[:8192]
        delimiter = '\t' if file_path.suffix.lower() == '.tsv' else ','
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
            delimiter = dialect.delimiter
        except Exception:
            dialect = None
        try:
            reader = csv.reader(io.StringIO(raw), dialect=dialect) if dialect else csv.reader(io.StringIO(raw), delimiter=delimiter)
            rows = format_table_rows([list(row) for row in reader])
            return '\n'.join(rows), bool(rows)
        except Exception as exc:
            logger.warning("Delimited text parser failed for %s: %s", file_path.name, exc)
            return raw, bool(raw.strip())

    def _flatten_json_value(self, value: Any, *, prefix: str = "", depth: int = 0) -> List[str]:
        if depth > 20:
            return [f"{prefix}: [maximum nesting depth]" if prefix else "[maximum nesting depth]"]
        if isinstance(value, dict):
            lines: List[str] = []
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                lines.extend(self._flatten_json_value(item, prefix=child, depth=depth + 1))
            return lines
        if isinstance(value, list):
            lines = []
            for index, item in enumerate(value):
                child = f"{prefix}[{index}]" if prefix else f"[{index}]"
                lines.extend(self._flatten_json_value(item, prefix=child, depth=depth + 1))
            return lines
        rendered = "null" if value is None else str(value)
        return [f"{prefix}: {rendered}" if prefix else rendered]

    def _read_json_document(self, file_path: Path) -> Tuple[str, bool]:
        raw, ok = self._read_plain_text(file_path)
        if not ok or not raw.strip():
            return "", False
        try:
            if file_path.suffix.lower() == '.jsonl':
                parts: List[str] = []
                for line_number, raw_line in enumerate(raw.splitlines(), 1):
                    if not raw_line.strip():
                        continue
                    value = json.loads(raw_line)
                    parts.append(f"--- RECORD {line_number} ---")
                    parts.extend(self._flatten_json_value(value))
                return '\n'.join(parts), bool(parts)
            value = json.loads(raw)
            lines = self._flatten_json_value(value)
            return '\n'.join(lines), bool(lines)
        except Exception as exc:
            logger.warning("JSON parser failed for %s; indexing raw text: %s", file_path.name, exc)
            return raw, True

    def _read_eml(self, file_path: Path) -> Tuple[str, bool]:
        """Extract searchable headers and text bodies from RFC email files."""
        message = BytesParser(policy=policy.default).parsebytes(file_path.read_bytes())
        parts: List[str] = []
        for header in ("Subject", "From", "To", "Cc", "Date", "Message-ID"):
            value = message.get(header)
            if value:
                parts.append(f"{header}: {value}")

        bodies: List[str] = []
        candidates = message.walk() if message.is_multipart() else [message]
        for part in candidates:
            if part.is_multipart() or str(part.get_content_disposition() or '').lower() == 'attachment':
                continue
            content_type = part.get_content_type()
            if content_type not in {"text/plain", "text/html"}:
                continue
            try:
                payload = part.get_content()
            except Exception:
                payload_bytes = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                payload = payload_bytes.decode(charset, errors="replace")
            text = str(payload or "")
            if content_type == "text/html":
                text = self._html_to_text(text)
            if text.strip():
                bodies.append(text.strip())
        if bodies:
            parts.append("\n--- MESSAGE BODY ---\n" + "\n\n".join(bodies))
        return '\n'.join(parts), bool(parts)

    def _html_to_text(self, raw_html: str) -> str:
        """Превращает HTML в поисковый текст, сохраняя абзацы и заголовки."""
        if not raw_html:
            return ""
        if BS4_AVAILABLE and BeautifulSoup is not None:
            soup = BeautifulSoup(raw_html, 'html.parser')
            for tag in soup(['script', 'style', 'noscript']):
                tag.decompose()
            text = soup.get_text('\n')
        else:
            parser = _SimpleHTMLTextExtractor()
            parser.feed(raw_html)
            text = parser.get_text()
        text = html.unescape(text or '')
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()

    def _strip_xml_to_text(self, xml_text: str) -> str:
        """Грубый, но устойчивый XML→text fallback для ODT/EPUB/XHTML."""
        if not xml_text:
            return ""
        try:
            root = ET.fromstring(xml_text.encode('utf-8', errors='ignore') if isinstance(xml_text, str) else xml_text)
            parts: list[str] = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    parts.append(elem.text.strip())
                if elem.tail and elem.tail.strip():
                    parts.append(elem.tail.strip())
            return '\n'.join(parts)
        except Exception:
            value = re.sub(r'<[^>]+>', ' ', xml_text)
            return html.unescape(re.sub(r'\s+', ' ', value)).strip()

    def _read_rtf(self, file_path: Path) -> Tuple[str, bool]:
        raw, ok = self._read_plain_text(file_path)
        if not ok:
            return "", False
        if STRIPRTF_AVAILABLE and rtf_to_text is not None:
            try:
                return rtf_to_text(raw), True
            except Exception as exc:
                logger.warning("striprtf failed for %s: %s", file_path.name, exc)
        # Минимальный fallback: выкидываем управляющие команды RTF и оставляем текст.
        text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", raw)
        text = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", text)
        text = text.replace('{', ' ').replace('}', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip(), True

    def _read_docx(self, file_path: Path) -> Tuple[str, bool]:
        if not DOCX_AVAILABLE:
            return "", False
        if not zipfile.is_zipfile(file_path):
            logger.warning("Skipping invalid DOCX container %s", file_path.name)
            return "", False
        doc = docx.Document(file_path)
        try:
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except Exception:
            CT_Tbl = CT_P = Table = Paragraph = None

        parts: list[str] = []

        def add_paragraph(para) -> None:
            value = str(getattr(para, 'text', '') or '').strip()
            if not value:
                return
            style_name = str(getattr(getattr(para, 'style', None), 'name', '') or '')
            heading_match = re.search(r"(?:heading|заголовок)\s*(\d+)", style_name, flags=re.IGNORECASE)
            if heading_match:
                level = max(1, min(6, int(heading_match.group(1))))
                value = f"{'#' * level} {value}"
            elif re.search(r"(?:list|список)", style_name, flags=re.IGNORECASE):
                value = f"- {value}"
            parts.append(value)

        def add_table(table) -> None:
            rows = [[cell.text or '' for cell in row.cells] for row in table.rows]
            lines = format_table_rows(rows)
            if lines:
                parts.append("--- TABLE ---")
                parts.extend(lines)

        if CT_P is not None:
            for child in doc.element.body.iterchildren():
                if isinstance(child, CT_P):
                    add_paragraph(Paragraph(child, doc))
                elif isinstance(child, CT_Tbl):
                    add_table(Table(child, doc))
        else:
            for para in doc.paragraphs:
                add_paragraph(para)
            for table in doc.tables:
                add_table(table)

        # Headers and footers often contain a report date, title or scope that
        # is essential for retrieval but absent from ``doc.paragraphs``.
        header_footer_parts: list[str] = []
        seen = set()
        for section in doc.sections:
            for label, container in (("HEADER", section.header), ("FOOTER", section.footer)):
                values = [p.text.strip() for p in container.paragraphs if p.text.strip()]
                if not values:
                    continue
                block = '\n'.join(values)
                key = (label, block)
                if key in seen:
                    continue
                seen.add(key)
                header_footer_parts.append(f"--- {label} ---\n{block}")
        if header_footer_parts:
            parts = header_footer_parts + parts
        content = '\n'.join(parts)
        return content, bool(content.strip())

    def _read_legacy_doc(self, file_path: Path) -> Tuple[str, bool]:
        """Best-effort чтение старого Word .doc.

        Формат .doc бинарный. Надёжнее всего его читают внешние утилиты
        antiword/catdoc/textutil, если они установлены. Когда их нет, делаем
        безопасный fallback: вытягиваем длинные printable/UTF-16LE строки. Это
        не идеально, но лучше, чем молча игнорировать старый документ.
        """
        commands = []
        antiword = shutil.which('antiword')
        catdoc = shutil.which('catdoc')
        textutil = shutil.which('textutil')
        if antiword:
            commands.append([antiword, str(file_path)])
        if catdoc:
            commands.append([catdoc, str(file_path)])
        if textutil:
            commands.append([textutil, '-convert', 'txt', '-stdout', str(file_path)])
        for cmd in commands:
            try:
                completed = subprocess.run(cmd, capture_output=True, timeout=20)
                if completed.returncode == 0 and completed.stdout:
                    for enc in ('utf-8', 'cp1251', 'windows-1251', 'latin-1'):
                        try:
                            text = completed.stdout.decode(enc)
                            if len(text.strip()) > 20:
                                return text, True
                        except UnicodeDecodeError:
                            continue
            except Exception as exc:
                logger.debug("Legacy DOC reader failed %s for %s: %s", cmd[:1], file_path.name, exc)

        data = file_path.read_bytes()
        candidates: list[str] = []
        for enc in ('utf-16le', 'cp1251', 'windows-1251', 'latin-1'):
            try:
                decoded = data.decode(enc, errors='ignore')
            except Exception:
                continue
            # Вытаскиваем достаточно длинные текстовые островки.
            pieces = re.findall(r"[A-Za-zА-Яа-яЁё0-9№.,;:!?%()\[\]/\\\-–—+*=\"'\s]{8,}", decoded)
            cleaned = []
            for piece in pieces:
                piece = re.sub(r'\s+', ' ', piece).strip()
                if len(piece) >= 8 and sum(ch.isalnum() for ch in piece) >= 4:
                    cleaned.append(piece)
            text = '\n'.join(cleaned)
            if len(text.strip()) > max(60, len(data) // 200):
                candidates.append(text)
        if candidates:
            return max(candidates, key=len), True
        return "", False

    def _read_xlsx(self, file_path: Path) -> Tuple[str, bool]:
        if not OPENPYXL_AVAILABLE or openpyxl is None:
            logger.warning("openpyxl is not installed. XLSX files will be skipped: %s", file_path.name)
            return "", False
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        parts: list[str] = []
        for ws in wb.worksheets:
            parts.append(f"--- ЛИСТ: {ws.title} ---")
            # Блоки строк, разделённые пустыми строками, считаем отдельными
            # таблицами: у каждого блока может быть свой заголовок.
            block: list[list[str]] = []
            for row in ws.iter_rows(values_only=True):
                values = [str(v).strip() if v is not None else '' for v in row]
                if not any(values):
                    if block:
                        parts.extend(format_table_rows(block))
                        parts.append('')
                        block = []
                    continue
                block.append(values)
            if block:
                parts.extend(format_table_rows(block))
        return '\n'.join(parts), True

    def _read_xls(self, file_path: Path) -> Tuple[str, bool]:
        if XLRD_AVAILABLE and xlrd is not None:
            book = xlrd.open_workbook(str(file_path))
            parts: list[str] = []
            for sheet in book.sheets():
                parts.append(f"--- ЛИСТ: {sheet.name} ---")
                for r in range(sheet.nrows):
                    values = []
                    for c in range(sheet.ncols):
                        value = sheet.cell_value(r, c)
                        if value != '':
                            values.append(str(value).strip())
                    if values:
                        parts.append(' | '.join(values))
            return '\n'.join(parts), True
        return self._read_legacy_doc(file_path)

    def _read_pptx(self, file_path: Path) -> Tuple[str, bool]:
        if not PPTX_AVAILABLE or Presentation is None:
            logger.warning("python-pptx is not installed. PPTX files will be skipped: %s", file_path.name)
            return "", False
        prs = Presentation(str(file_path))
        parts: list[str] = []
        for idx, slide in enumerate(prs.slides, 1):
            slide_parts: list[str] = []
            for shape in slide.shapes:
                if hasattr(shape, 'text') and shape.text and shape.text.strip():
                    slide_parts.append(shape.text.strip())
                if getattr(shape, 'has_table', False):
                    for row in shape.table.rows:
                        cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if cells:
                            slide_parts.append(' | '.join(cells))
            if slide_parts:
                parts.append(f"--- СЛАЙД {idx} ---\n" + '\n'.join(slide_parts))
        return '\n\n'.join(parts), True

    def _read_odf(self, file_path: Path) -> Tuple[str, bool]:
        if ODF_AVAILABLE and odf_load is not None:
            doc = odf_load(str(file_path))
            parts: list[str] = []
            for elem in doc.getElementsByType(odf_text.P):
                value = ''.join(node.data for node in elem.childNodes if getattr(node, 'data', None)).strip()
                if value:
                    parts.append(value)
            for elem in doc.getElementsByType(odf_text.H):
                value = ''.join(node.data for node in elem.childNodes if getattr(node, 'data', None)).strip()
                if value:
                    parts.append(value)
            for table in doc.getElementsByType(odf_table.Table):
                for row in table.getElementsByType(odf_table.TableRow):
                    cells = []
                    for cell in row.getElementsByType(odf_table.TableCell):
                        cell_text = []
                        for p in cell.getElementsByType(odf_text.P):
                            value = ''.join(node.data for node in p.childNodes if getattr(node, 'data', None)).strip()
                            if value:
                                cell_text.append(value)
                        if cell_text:
                            cells.append(' '.join(cell_text))
                    if cells:
                        parts.append(' | '.join(cells))
            if parts:
                return '\n'.join(parts), True

        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path) as zf:
                names = [n for n in zf.namelist() if n.endswith('.xml') and ('content.xml' in n or 'styles.xml' not in n)]
                parts = []
                for name in names:
                    raw = zf.read(name).decode('utf-8', errors='ignore')
                    text = self._strip_xml_to_text(raw)
                    if text:
                        parts.append(text)
                return '\n\n'.join(parts), bool(parts)
        return "", False

    def _read_epub(self, file_path: Path) -> Tuple[str, bool]:
        if not zipfile.is_zipfile(file_path):
            return "", False
        parts: list[str] = []
        with zipfile.ZipFile(file_path) as zf:
            for name in zf.namelist():
                low = name.lower()
                if low.endswith(('.html', '.htm', '.xhtml', '.xml')) and 'meta-inf' not in low:
                    try:
                        raw = zf.read(name).decode('utf-8', errors='ignore')
                    except Exception:
                        continue
                    text = self._html_to_text(raw) if low.endswith(('.html', '.htm', '.xhtml')) else self._strip_xml_to_text(raw)
                    if text:
                        parts.append(f"--- {name} ---\n{text}")
        return '\n\n'.join(parts), bool(parts)

    def _read_file_content(self, file_path: Path) -> Tuple[str, bool]:
        """Читает один файл и возвращает ``(content, success)``."""
        ext = file_path.suffix.lower()

        try:
            if ext in {'.csv', '.tsv'}:
                return self._read_delimited_text(file_path)

            if ext in {'.json', '.jsonl', '.ipynb'}:
                return self._read_json_document(file_path)

            if ext in TEXT_EXTENSIONS or ext == '':
                return self._read_plain_text(file_path)

            if ext in HTML_EXTENSIONS:
                raw, ok = self._read_plain_text(file_path)
                return (self._html_to_text(raw), True) if ok else ("", False)

            if ext == '.eml':
                return self._read_eml(file_path)

            if ext == '.rtf':
                return self._read_rtf(file_path)

            if ext == '.pdf':
                if not PYPDF2_AVAILABLE:
                    return "", False
                content = []
                with open(file_path, 'rb') as f:
                    pdf = PdfReader(f)
                    for page_num, page in enumerate(pdf.pages):
                        try:
                            text = page.extract_text()
                            if text and text.strip():
                                content.append(f"--- Страница {page_num + 1} ---\n{text}")
                        except Exception:
                            continue
                return '\n\n'.join(content), True

            if ext == '.docx':
                return self._read_docx(file_path)

            if ext == '.doc':
                return self._read_legacy_doc(file_path)

            if ext in {'.xlsx', '.xlsm', '.xltx', '.xltm'}:
                return self._read_xlsx(file_path)

            if ext == '.xls':
                return self._read_xls(file_path)

            if ext == '.pptx':
                return self._read_pptx(file_path)

            if ext == '.ppt':
                return self._read_legacy_doc(file_path)

            if ext in {'.odt', '.ods', '.odp'}:
                return self._read_odf(file_path)

            if ext == '.epub':
                return self._read_epub(file_path)

        except zipfile.BadZipFile as e:
            logger.warning("Failed to read zip-based document %s: %s", file_path.name, e)
        except Exception as e:
            logger.warning("Failed to read file %s: %s", file_path.name, e)

        return "", False

    def _clean_document_content(self, content: str, filename: str) -> str:
        """Нормализует переводы строк и применяет небольшие очистки, зависящие от формата."""
        if not content:
            return ""

        if content.startswith('\ufeff'):
            content = content[1:]

        content = content.replace('\r\n', '\n').replace('\r', '\n')
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.rstrip()
            if line:
                cleaned_lines.append(line)
            else:
                cleaned_lines.append('')

        cleaned = '\n'.join(cleaned_lines)
        if filename.lower().endswith(('.doc', '.docx', '.odt', '.rtf')):
            cleaned = self._restore_structured_boundaries(cleaned)
        return cleaned

    def _restore_structured_boundaries(self, content: str) -> str:
        """Restore boundaries lost when an office file stores a report in one paragraph.

        The rules are layout-oriented: headings, punctuation and time-like list
        items. They intentionally do not depend on a document domain.
        """
        value = content or ""
        section_header = (
            r"[A-Za-zА-Яа-яЁё][\w/-]{1,30}\s*"
            r"(?:№|#)\s*\d{1,8}(?:\s*\([^)\n]{2,100}\))?\s*:"
        )
        value = re.sub(rf"(?<!\n)[ \t]+(?={section_header})", "\n", value)
        value = re.sub(rf"({section_header})[ \t]+(?=\S)", r"\1\n", value)
        value = re.sub(
            r"(?<=[;:])[ \t]+(?=(?:[01]?\d|2[0-3])[.:][0-5]\d\s*[—–-]\s*)",
            "\n",
            value,
        )
        # Office exports sometimes concatenate sentences and list items with no
        # whitespace at all ("...позиции.Следующий раздел" or "Title:item").
        # The two characters before the period must belong to a word: this
        # keeps initials («Иванов И.П.»), «г.Москва» and «т.к.Далее» intact.
        value = re.sub(r"(?<=[a-zа-яё0-9»)\]\"']{2}[.!?])(?=[A-ZА-ЯЁ])", "\n", value)
        value = re.sub(r"(?<=[;:])(?=(?:[A-ZА-ЯЁ]|[01]?\d[.:][0-5]\d\s*[—–-]))", "\n", value)
        return re.sub(r"\n{3,}", "\n\n", value)

    def _extract_primary_document_date(self, content: str) -> str:
        head = (content or "")[:3000]
        parsed = parse_date_constraints(head)
        dates = list(parsed.get("dates") or [])
        if dates:
            return dates[0].strftime("%Y-%m-%d")
        start = parsed.get("start")
        end = parsed.get("end")
        if isinstance(start, datetime) and isinstance(end, datetime) and start == end:
            return start.strftime("%Y-%m-%d")
        return ""

    def _create_document_object(self, file_path: Path, content: str) -> Optional[Document]:
        """Создаёт канонический объект ``Document`` для одного загруженного файла."""
        try:
            stat = file_path.stat()

            try:
                rel_path = file_path.relative_to(self.docs_folder)
            except ValueError:
                rel_path = file_path.name

            doc_id = self._generate_document_id(file_path)
            size_chars = len(content)
            size_kb = file_path.stat().st_size / 1024

            ext = file_path.suffix.lower()
            format_family = 'text'
            if ext in {'.doc', '.docx', '.odt', '.rtf'}:
                format_family = 'document'
            elif ext in {'.xlsx', '.xlsm', '.xltx', '.xltm', '.xls', '.ods'}:
                format_family = 'spreadsheet'
            elif ext in {'.ppt', '.pptx', '.odp'}:
                format_family = 'presentation'
            elif ext in {'.html', '.htm', '.xhtml', '.epub'}:
                format_family = 'web'
            elif ext == '.eml':
                format_family = 'message'
            elif ext == '.pdf':
                format_family = 'pdf'

            metadata = {
                'full_path': str(file_path.absolute()),
                'line_count': content.count('\n') + 1,
                'word_count': len(content.split()),
                'has_tables': '--- ТАБЛИЦЫ ---' in content or '--- TABLE ---' in content or '--- ЛИСТ:' in content,
                'has_pages': '--- Страница' in content,
                'format_family': format_family,
                'reader_extension': ext,
                'document_date': self._extract_primary_document_date(content),
            }

            return Document(
                id=doc_id,
                path=str(file_path.absolute()),
                name=str(rel_path),
                filename=file_path.name,
                content=content,
                size_chars=size_chars,
                size_kb=size_kb,
                created_time=stat.st_ctime,
                modified_time=stat.st_mtime,
                extension=file_path.suffix.lower(),
                is_valid=True,
                metadata=metadata
            )

        except Exception as e:
            logger.warning("Failed to create document object for %s: %s", file_path, e)
            return None

    def load_documents(self) -> bool:
        """Полностью перечитывает корпус с диска и перестраивает чанки в памяти."""
        start_time = time.time()
        files = self._collect_all_files()

        if not files:
            logger.info("No documents found in %s", self.docs_folder)
            return False

        self.documents.clear()
        self.documents_by_id.clear()
        self.documents_by_name.clear()
        self.text_chunks.clear()

        extension_stats = {}
        failed_files = []
        failed_details = []

        for i, file_path in enumerate(files, 1):
            try:
                raw_content, success = self._read_file_content(file_path)
                if not success or not raw_content.strip():
                    failed_files.append(file_path.name)
                    reason = "read_failed_or_empty"
                    if file_path.suffix.lower() == ".pdf" and success:
                        reason = "pdf_has_no_extractable_text_or_requires_ocr"
                    elif file_path.suffix.lower() in {".doc", ".ppt"}:
                        reason = "legacy_office_converter_unavailable_or_failed"
                    failed_details.append({"file": file_path.name, "reason": reason})
                    continue

                cleaned_content = self._clean_document_content(raw_content, file_path.name)
                if not cleaned_content.strip():
                    failed_files.append(file_path.name)
                    failed_details.append({"file": file_path.name, "reason": "empty_after_clean"})
                    continue

                doc = self._create_document_object(file_path, cleaned_content)
                if not doc:
                    failed_files.append(file_path.name)
                    failed_details.append({"file": file_path.name, "reason": "document_object_failed"})
                    continue

                self.documents.append(doc)
                self.documents_by_id[doc.id] = doc
                self.documents_by_name[doc.name] = doc

                ext = doc.extension
                extension_stats[ext] = extension_stats.get(ext, 0) + 1

            except Exception as e:
                logger.warning("Failed to load document %s: %s", file_path.name, e)
                failed_files.append(file_path.name)
                failed_details.append({"file": file_path.name, "reason": type(e).__name__, "message": str(e)})
                continue

        elapsed = time.time() - start_time
        total_chars = sum(doc.size_chars for doc in self.documents)
        total_size_mb = sum(doc.size_kb for doc in self.documents) / 1024

        self.stats.update({
            "total_documents": len(self.documents),
            "total_size_mb": total_size_mb,
            "total_chars": total_chars,
            "by_extension": extension_stats,
            "load_time_seconds": elapsed,
            "failed_files": failed_files,
            "failed_files_details": failed_details[:100]
        })

        logger.info(
            "Documents loaded: count=%s failed=%s total_size_mb=%.1f total_chars=%s elapsed=%.2f",
            len(self.documents),
            len(failed_files),
            total_size_mb,
            total_chars,
            elapsed,
        )
        if failed_details:
            preview = '; '.join(f"{item['file']}: {item.get('reason', 'unknown')}" for item in failed_details[:10])
            logger.warning("Document load failures preview: %s", preview)

        self._create_chunks()

        return len(self.documents) > 0

    def update_index(self, new_docs_folder: str = None, recursive: bool = None) -> bool:
        """Применяет изменения настроек папки и перестраивает индекс документов."""
        previous_state = {
            "docs_folder": self.docs_folder,
            "recursive": self.recursive,
            "ignore_patterns": list(self.ignore_patterns),
            "documents": list(self.documents),
            "documents_by_id": dict(self.documents_by_id),
            "documents_by_name": dict(self.documents_by_name),
            "text_chunks": list(self.text_chunks),
            "stats": dict(self.stats),
        }
        try:
            if new_docs_folder:
                self.docs_folder = Path(new_docs_folder).absolute()
                self.ignore_patterns = self._load_ignore_patterns()
            if recursive is not None:
                self.recursive = recursive

            self.documents.clear()
            self.documents_by_id.clear()
            self.documents_by_name.clear()
            self.text_chunks.clear()

            success = self.load_documents()
            if success:
                logger.info("Document index updated: %s documents", len(self.documents))
                return True

            self.docs_folder = previous_state["docs_folder"]
            self.recursive = previous_state["recursive"]
            self.ignore_patterns = previous_state["ignore_patterns"]
            self.documents = previous_state["documents"]
            self.documents_by_id = previous_state["documents_by_id"]
            self.documents_by_name = previous_state["documents_by_name"]
            self.text_chunks = previous_state["text_chunks"]
            self.stats = previous_state["stats"]
            logger.warning("Document index update produced no readable documents; previous index restored")
            return False

        except Exception as e:
            self.docs_folder = previous_state["docs_folder"]
            self.recursive = previous_state["recursive"]
            self.ignore_patterns = previous_state["ignore_patterns"]
            self.documents = previous_state["documents"]
            self.documents_by_id = previous_state["documents_by_id"]
            self.documents_by_name = previous_state["documents_by_name"]
            self.text_chunks = previous_state["text_chunks"]
            self.stats = previous_state["stats"]
            logger.exception("Failed to update document index")
            return False

    # -------------------- чанкинг -------------------- #

    def _char_to_line(self, text: str, char_pos: int) -> int:
        """Переводит offset символа в номер строки, начиная с 1."""
        if char_pos <= 0:
            return 1
        return (text or '').count('\n', 0, min(char_pos, len(text or ''))) + 1

    def _span_line_range(self, text: str, start_char: int, end_char: int) -> tuple[int, int]:
        """Возвращает диапазон строк для координат чанка/доказательства."""
        line_start = self._char_to_line(text, start_char)
        line_end = self._char_to_line(text, max(start_char, end_char - 1))
        return line_start, max(line_start, line_end)

    def _create_chunks(self):
        """Создаёт реальные чанки по всему документу (не 1 обрезок)."""
        self.text_chunks = []
        chunk_id = 0

        for doc in self.documents:
            chunks = self._split_text_into_chunks(
                doc.content,
                max_chars=self.chunk_max_chars,
                overlap=self.chunk_overlap,
                min_chars=self.min_chunk_chars
            )

            for idx_in_doc, (start_char, end_char, chunk_text) in enumerate(chunks):
                line_start, line_end = self._span_line_range(doc.content, start_char, end_char)
                section_path = self._section_context_before(doc.content, start_char)
                retrieval_text = chunk_text
                if section_path:
                    last_heading = section_path[-1]
                    if self._normalize_for_boundary(last_heading) not in self._normalize_for_boundary(chunk_text[:320]):
                        retrieval_text = f"{last_heading}\n{chunk_text}"
                self.text_chunks.append({
                    "text": retrieval_text,
                    "source_text": chunk_text,
                    "section_path": section_path,
                    "source": doc.name,
                    "doc_id": doc.id,
                    "chunk_id": chunk_id,
                    "chunk_index": idx_in_doc,
                    "start_char": start_char,
                    "end_char": end_char,
                    "line_start": line_start,
                    "line_end": line_end,
                    "span_key": f"{doc.id}:{start_char}:{end_char}",
                    "is_full_document": False,
                    "format_family": (doc.metadata or {}).get("format_family", "text"),
                    "document_date": (doc.metadata or {}).get("document_date", ""),
                })
                chunk_id += 1

        logger.info(
            "Text chunks built: count=%s max_chars=%s overlap=%s",
            len(self.text_chunks),
            self.chunk_max_chars,
            self.chunk_overlap,
        )

    @staticmethod
    def _normalize_for_boundary(value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip()).lower()

    def _looks_like_section_heading(self, value: str) -> bool:
        text = re.sub(r"\s+", " ", (value or "").strip())
        if not text or len(text) > 180:
            return False
        if text.startswith('#'):
            return True
        words = re.findall(r"[A-Za-zА-Яа-яЁё0-9№#]+", text)
        if not words or len(words) > 18:
            return False
        letters = ''.join(ch for ch in text if ch.isalpha())
        if (
            letters
            and len(letters) >= 4
            and not re.search(r"[,;!?]", text)
            and sum(ch.isupper() for ch in letters) / len(letters) >= 0.78
        ):
            return True
        if text.endswith(':') and len(words) <= 14:
            return True
        if re.match(r"^(?:\d+(?:\.\d+)*|[A-ZА-ЯЁ])(?:[.)]|\s+-)\s+\S", text):
            return True
        return False

    def _section_context_before(self, text: str, start_char: int, limit: int = 3) -> List[str]:
        """Return nearby structural headings without relying on domain words."""
        prefix_start = max(0, int(start_char) - 12000)
        prefix = (text or "")[prefix_start:max(0, int(start_char))]
        headings: List[str] = []
        for raw_line in prefix.splitlines():
            line = re.sub(r"\s+", " ", raw_line.strip())
            if self._looks_like_section_heading(line):
                headings.append(line)
        return headings[-max(1, int(limit)):]

    def _split_oversized_unit(self, text: str, absolute_start: int, max_chars: int) -> List[Tuple[int, int, str]]:
        """Split one long line at sentence/word boundaries, never mid-word when avoidable."""
        value = text or ""
        if len(value) <= max_chars:
            return [(absolute_start, absolute_start + len(value), value)]

        sentence_boundaries = [0]
        for match in re.finditer(r"(?<=[.!?;])(?:\s+|(?=[A-ZА-ЯЁ]))", value):
            sentence_boundaries.append(match.end())
        sentence_boundaries.append(len(value))

        spans: List[Tuple[int, int, str]] = []
        cursor = 0
        while cursor < len(value):
            target = min(len(value), cursor + max_chars)
            if target < len(value):
                candidates = [point for point in sentence_boundaries if cursor + max_chars // 2 <= point <= target]
                if candidates:
                    target = candidates[-1]
                else:
                    whitespace = max(value.rfind(' ', cursor + max_chars // 2, target), value.rfind('\t', cursor + max_chars // 2, target))
                    if whitespace > cursor:
                        target = whitespace
            if target <= cursor:
                target = min(len(value), cursor + max_chars)
            fragment = value[cursor:target].strip()
            if fragment:
                left_trim = len(value[cursor:target]) - len(value[cursor:target].lstrip())
                right_trimmed_end = target - (len(value[cursor:target]) - len(value[cursor:target].rstrip()))
                spans.append((absolute_start + cursor + left_trim, absolute_start + right_trimmed_end, fragment))
            cursor = target
            while cursor < len(value) and value[cursor].isspace():
                cursor += 1
        return spans

    def _split_text_into_chunks(
        self,
        text: str,
        max_chars: int = 1400,
        overlap: int = 220,
        min_chars: int = 120
    ) -> List[Tuple[int, int, str]]:
        """Split text by structural units with whole-unit overlap.

        Arbitrary fixed character windows used to cut words and detach child
        rows from headings. This implementation keeps line/sentence units and
        only uses a word-snapped hard split for a single oversized unit.
        """
        if not text:
            return []

        norm = text.replace('\r\n', '\n').replace('\r', '\n')
        units: List[Tuple[int, int, str]] = []
        for line_match in re.finditer(r"[^\n]+", norm):
            raw = line_match.group(0)
            stripped = raw.strip()
            if not stripped:
                continue
            left_trim = len(raw) - len(raw.lstrip())
            absolute_start = line_match.start() + left_trim
            units.extend(self._split_oversized_unit(stripped, absolute_start, max_chars))

        if not units:
            return []

        chunks: List[Tuple[int, int, str]] = []
        current: List[Tuple[int, int, str]] = []

        def current_length(items: List[Tuple[int, int, str]]) -> int:
            return sum(len(item[2]) for item in items) + max(0, len(items) - 1)

        def flush() -> None:
            nonlocal current
            if not current:
                return
            chunk_text = '\n'.join(item[2] for item in current).strip()
            if chunk_text:
                chunks.append((current[0][0], current[-1][1], chunk_text))

            tail: List[Tuple[int, int, str]] = []
            tail_size = 0
            if overlap > 0 and len(current) > 1:
                for item in reversed(current):
                    addition = len(item[2]) + (1 if tail else 0)
                    if tail and tail_size + addition > overlap:
                        break
                    if not tail and addition > overlap:
                        break
                    tail.insert(0, item)
                    tail_size += addition
                if len(tail) == len(current):
                    tail = tail[-1:] if len(tail) > 1 else []
            current = tail

        for unit in units:
            unit_text = unit[2]
            is_heading = self._looks_like_section_heading(unit_text)
            if is_heading and current and current_length(current) >= min_chars:
                flush()
                current = []

            projected = current_length(current) + (1 if current else 0) + len(unit_text)
            if current and projected > max_chars:
                flush()
                projected = current_length(current) + (1 if current else 0) + len(unit_text)
                if current and projected > max_chars:
                    current = []
            current.append(unit)

        flush()

        # Never discard a short final fact. Merge it when the combined chunk is
        # still reasonable; otherwise retain it as a searchable small chunk.
        if len(chunks) >= 2 and len(chunks[-1][2]) < min_chars:
            previous = chunks[-2]
            final = chunks[-1]
            merged_text = previous[2].rstrip() + '\n' + final[2].lstrip()
            if len(merged_text) <= max_chars + max(0, overlap):
                chunks[-2:] = [(previous[0], final[1], merged_text)]

        return chunks

    # -------------------- API -------------------- #

    def get_document(self, identifier: str) -> Optional[Document]:
        """Возвращает документ по id, точному имени или приблизительному совпадению пути."""
        if identifier in self.documents_by_id:
            return self.documents_by_id[identifier]
        if identifier in self.documents_by_name:
            return self.documents_by_name[identifier]

        for doc in self.documents:
            if identifier in doc.name or identifier in doc.filename:
                return doc
        return None

    def get_documents_by_ids(self, doc_ids: List[str]) -> List[Document]:
        """Возвращает все известные документы для запрошенных идентификаторов."""
        return [self.documents_by_id.get(doc_id) for doc_id in doc_ids
                if doc_id in self.documents_by_id]

    def get_document_info(self) -> str:
        """
        Нужно для GUI: gui/app.py вызывает этот метод в шапке.
        """
        docs = len(self.documents)
        chunks = len(self.text_chunks)
        total_chars = int(self.stats.get("total_chars", 0) or 0)
        total_mb = float(self.stats.get("total_size_mb", 0.0) or 0.0)

        if docs == 0:
            return "📄 Документы: 0 | 🧩 Чанки: 0"
        return f"📄 Документы: {docs} | 🧩 Чанки: {chunks} | 🔤 Символов: {total_chars:,} | 💾 Размер: {total_mb:.2f} MB"

    def get_document_names(self) -> List[str]:
        """Возвращает отображаемые имена документов в текущем порядке загрузки."""
        return [d.name for d in self.documents]

    def get_all_documents_context(
        self,
        max_total_chars: int = 15000,
        max_documents: int = 10
    ) -> str:
        """Строит грубую строку контекста из самых компактных загруженных документов."""
        if not self.documents:
            return ""

        sorted_docs = sorted(self.documents, key=lambda d: d.size_chars)

        context_parts = []
        total_chars = 0

        for doc in sorted_docs[:max_documents]:
            if total_chars + doc.size_chars > max_total_chars:
                remaining_chars = max_total_chars - total_chars
                if remaining_chars > 100:
                    excerpt = doc.content[:remaining_chars] + "..."
                    context_parts.append(f"📄 ДОКУМЕНТ: {doc.name}\n{excerpt}")
                break

            context_parts.append(f"📄 ДОКУМЕНТ: {doc.name}\n{doc.content}")
            total_chars += doc.size_chars

        separator = "\n\n" + "=" * 70 + "\n\n"
        return separator.join(context_parts)

    def get_relevant_documents_context(self, query: str, max_total_chars: int = 20000) -> str:
        """
        Простой fallback контекст по пересечению слов (на случай, если нужен).
        """
        if not self.documents:
            return ""

        query_words = set(re.findall(r'\w+', query.lower()))
        scored_docs = []
        for doc in self.documents:
            doc_words = set(re.findall(r'\w+', doc.content.lower()))
            common_words = query_words.intersection(doc_words)
            score = len(common_words)
            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(reverse=True, key=lambda x: x[0])

        context_parts = []
        total_chars = 0

        for score, doc in scored_docs[:5]:
            if total_chars + doc.size_chars > max_total_chars:
                remaining_chars = max_total_chars - total_chars
                if remaining_chars > 200:
                    excerpt = doc.content[:remaining_chars] + "..."
                    context_parts.append(f"📄 ДОКУМЕНТ ({score} совп.): {doc.name}\n{excerpt}")
                break

            context_parts.append(f"📄 ДОКУМЕНТ ({score} совп.): {doc.name}\n{doc.content}")
            total_chars += doc.size_chars

        separator = "\n\n" + "=" * 70 + "\n\n"
        return separator.join(context_parts)
