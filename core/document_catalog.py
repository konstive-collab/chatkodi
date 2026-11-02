"""Private local catalog for large, mixed document folders.

The catalog stores metadata only. Extracted document text is used in memory and
is never written to the catalog or application logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import stat
from typing import Dict, Iterable, List, Optional
import zipfile

from core.atomic_io import atomic_write_json
from core.logging_utils import get_logger

logger = get_logger(__name__)

_INDEXABLE = {".docx", ".doc", ".pdf", ".txt", ".md", ".rtf", ".odt"}
_SPORT_HINTS = (
    "физическая подготовка", "физической подготовки", "физическая культура",
    "трениров", "упражнен", "подтягив", "плаван", "гимнаст", "вынослив",
)
_LEGAL_HINTS = (
    "рапорт", "приказ", "заявлен", "ходатайств", "служебная записка",
    "объяснительн", "характеристик", "командиров",
)
_KNOWN_SPORT_NAMES = {
    "ответы фп.docx", "методика фп.docx", "сборник.docx", "word вариант 1.docx",
}


@dataclass
class CatalogRecord:
    path: str
    name: str
    extension: str
    size: int
    sha256: str
    valid: bool
    duplicate_of: str = ""
    temporary: bool = False
    hidden: bool = False
    topic: str = "general"
    document_type: str = "reference"
    paragraphs: int = 0
    tables: int = 0
    headers: int = 0
    footers: int = 0
    embedded_objects: int = 0
    external_relationships: int = 0
    source_eligible: bool = False
    template_eligible: bool = False
    error: str = ""


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xml_text(blob: bytes) -> str:
    text = blob.decode("utf-8", "ignore")
    text = re.sub(r"<w:tab[^>]*/>", "\t", text)
    text = re.sub(r"</w:p>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def extract_docx_text(path: Path, max_chars: int = 24000) -> str:
    if not zipfile.is_zipfile(path):
        return ""
    parts: List[str] = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        selected = ["word/document.xml"] + sorted(
            name for name in names if re.fullmatch(r"word/(?:header|footer)\d+\.xml", name)
        )
        for name in selected:
            if name not in names:
                continue
            info = archive.getinfo(name)
            if info.file_size > 50 * 1024 * 1024:
                continue
            parts.append(_xml_text(archive.read(name)))
            if sum(len(value) for value in parts) >= max_chars:
                break
    return "\n".join(parts)[:max_chars]


def _classify(name: str, text: str) -> tuple[str, str]:
    haystack = f"{name}\n{text}".lower().replace("ё", "е")
    sport_score = sum(haystack.count(item) for item in _SPORT_HINTS)
    legal_score = sum(haystack.count(item) for item in _LEGAL_HINTS)
    if name.lower() in _KNOWN_SPORT_NAMES:
        sport_score += 25
    if sport_score >= 3 and sport_score > legal_score:
        return "sport", "methodical"
    if legal_score:
        name_lower = name.lower().replace("ё", "е")
        beginning = text[:1200].lower().replace("ё", "е")
        looks_like_form = any(word in name_lower for word in _LEGAL_HINTS) or bool(
            re.search(r"(?:^|\n)\s*(?:рапорт|приказ|заявление|ходатайство|характеристика)\b", beginning)
        )
        doc_type = "legal_template" if looks_like_form else "legal_reference"
        return "legal", doc_type
    return "general", "reference"


class DocumentCatalog:
    def __init__(self, root: Path | str, catalog_path: Path | str | None = None):
        self.root = Path(root).expanduser().resolve()
        self.catalog_path = Path(catalog_path).resolve() if catalog_path else None
        self.records: List[CatalogRecord] = []

    def scan(self) -> List[CatalogRecord]:
        records: List[CatalogRecord] = []
        first_by_hash: Dict[tuple[int, str], str] = {}
        if not self.root.exists():
            self.records = []
            return []

        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            attributes = int(getattr(path.stat(), "st_file_attributes", 0) or 0)
            hidden = bool(attributes & getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0))
            temporary = path.name.startswith("~$") or ext in {".lnk", ".py", ".pyc"}
            if ext not in _INDEXABLE and not temporary:
                continue
            record = CatalogRecord(
                path=str(path), name=path.name, extension=ext, size=path.stat().st_size,
                sha256="", valid=False, temporary=temporary, hidden=hidden,
            )
            try:
                record.sha256 = _hash_file(path)
                duplicate_key = (record.size, record.sha256)
                record.duplicate_of = first_by_hash.get(duplicate_key, "")
                if not record.duplicate_of:
                    first_by_hash[duplicate_key] = str(path)

                text = ""
                if ext == ".docx":
                    if not zipfile.is_zipfile(path):
                        raise ValueError("invalid_docx_container")
                    with zipfile.ZipFile(path) as archive:
                        names = archive.namelist()
                        body = archive.read("word/document.xml")
                        record.paragraphs = len(re.findall(br"<w:p(?:\s|>)", body))
                        record.tables = len(re.findall(br"<w:tbl(?:\s|>)", body))
                        record.headers = sum(bool(re.fullmatch(r"word/header\d+\.xml", n)) for n in names)
                        record.footers = sum(bool(re.fullmatch(r"word/footer\d+\.xml", n)) for n in names)
                        record.embedded_objects = sum(n.startswith("word/embeddings/") for n in names)
                        for rel_name in (n for n in names if n.endswith(".rels")):
                            if archive.getinfo(rel_name).file_size <= 5 * 1024 * 1024:
                                record.external_relationships += archive.read(rel_name).count(b'TargetMode="External"')
                    text = extract_docx_text(path)
                elif ext in {".txt", ".md"}:
                    text = path.read_text(encoding="utf-8", errors="ignore")[:24000]
                record.topic, record.document_type = _classify(path.name, text)
                record.valid = not temporary
                record.source_eligible = record.valid and not record.hidden and not record.duplicate_of and ext in _INDEXABLE
                record.template_eligible = bool(
                    record.source_eligible
                    and ext == ".docx"
                    and record.document_type == "legal_template"
                    and not record.embedded_objects
                    and not record.external_relationships
                )
            except Exception as exc:
                record.error = type(exc).__name__
                record.valid = False
                record.source_eligible = False
                record.template_eligible = False
            records.append(record)

        self.records = records
        self._save_metadata()
        logger.info(
            "Document catalog ready: files=%s valid=%s sources=%s templates=%s",
            len(records), sum(item.valid for item in records),
            sum(item.source_eligible for item in records), sum(item.template_eligible for item in records),
        )
        return records

    def _save_metadata(self) -> None:
        if not self.catalog_path:
            return
        payload = {
            "root": str(self.root),
            "records": [asdict(item) for item in self.records],
            "summary": self.summary(),
        }
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.catalog_path, payload, backup=False)

    def summary(self) -> Dict[str, int]:
        return {
            "files": len(self.records),
            "visible_files": sum(not item.hidden for item in self.records),
            "valid": sum(item.valid for item in self.records),
            "invalid": sum(not item.valid for item in self.records),
            "invalid_docx": sum(item.extension == ".docx" and not item.valid and not item.temporary for item in self.records),
            "excluded_service_files": sum(item.temporary for item in self.records),
            "hidden_files": sum(item.hidden for item in self.records),
            "duplicates": sum(bool(item.duplicate_of) for item in self.records),
            "sources": sum(item.source_eligible for item in self.records),
            "templates": sum(item.template_eligible for item in self.records),
            "unsafe_docx": sum(bool(item.embedded_objects or item.external_relationships) for item in self.records),
        }

    def ensure_scanned(self) -> None:
        if not self.records:
            self.scan()

    def find_sources(self, topic: str, query: str = "", limit: int = 5) -> List[CatalogRecord]:
        self.ensure_scanned()
        tokens = {tok for tok in re.findall(r"[a-zа-яё0-9]{3,}", query.lower())}
        ranked = []
        for item in self.records:
            if not item.source_eligible or item.topic != topic:
                continue
            name_lower = item.name.lower()
            score = 20 if name_lower in _KNOWN_SPORT_NAMES else 0
            score += sum(3 for token in tokens if token in name_lower)
            score += 5 if item.document_type in {"methodical", "legal_template"} else 0
            score -= min(item.size // (25 * 1024 * 1024), 4)
            ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].name.lower()))
        return [item for _score, item in ranked[:limit]]

    def find_legal_templates(self, query: str, limit: int = 3) -> List[CatalogRecord]:
        self.ensure_scanned()
        q = query.lower().replace("ё", "е")
        wanted = [word for word in _LEGAL_HINTS if word in q]
        ranked = []
        for item in self.records:
            if not item.template_eligible:
                continue
            name = item.name.lower().replace("ё", "е")
            score = 10
            score += sum(15 for word in wanted if word in name)
            if "командиров" in q and "командиров" in name:
                score += 40
            if "образец" in name:
                score += 4
            ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].name.lower()))
        return [item for _score, item in ranked[:limit]]

    def read_text(self, record: CatalogRecord, max_chars: int = 18000) -> str:
        path = Path(record.path)
        if record.extension == ".docx":
            return extract_docx_text(path, max_chars=max_chars)
        if record.extension in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        return ""
