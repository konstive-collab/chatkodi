from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import tempfile
import uuid
import zipfile
from typing import Callable, Dict, Iterable, List, Tuple
from xml.etree import ElementTree as ET

from docx import Document
from docx.text.paragraph import Paragraph

from core.structured_output import extract_json_object

from .types import ArtifactDraft


@dataclass
class DocxBlock:
    block_id: str
    paragraph: Paragraph
    text: str


def _iter_blocks(document: Document) -> List[DocxBlock]:
    blocks: List[DocxBlock] = []

    def add(prefix: str, paragraphs: Iterable[Paragraph]) -> None:
        for index, paragraph in enumerate(paragraphs):
            text = " ".join((paragraph.text or "").split())
            if text:
                blocks.append(DocxBlock(f"{prefix}:p{index}", paragraph, text))

    add("body", document.paragraphs)
    for table_index, table in enumerate(document.tables):
        for row_index, row in enumerate(table.rows):
            for cell_index, cell in enumerate(row.cells):
                add(f"table{table_index}:r{row_index}:c{cell_index}", cell.paragraphs)
    for section_index, section in enumerate(document.sections):
        add(f"header{section_index}", section.header.paragraphs)
        add(f"footer{section_index}", section.footer.paragraphs)
        for table_index, table in enumerate(section.header.tables):
            for row_index, row in enumerate(table.rows):
                for cell_index, cell in enumerate(row.cells):
                    add(f"header{section_index}:table{table_index}:r{row_index}:c{cell_index}", cell.paragraphs)
        for table_index, table in enumerate(section.footer.tables):
            for row_index, row in enumerate(table.rows):
                for cell_index, cell in enumerate(row.cells):
                    add(f"footer{section_index}:table{table_index}:r{row_index}:c{cell_index}", cell.paragraphs)
    # Linked headers can expose the same XML paragraph through several sections.
    unique: List[DocxBlock] = []
    seen = set()
    for block in blocks:
        marker = id(block.paragraph._p)
        if marker not in seen:
            seen.add(marker)
            unique.append(block)
    return unique


def _set_paragraph_text(paragraph: Paragraph, value: str) -> None:
    value = str(value or "")
    if paragraph.runs:
        paragraph.runs[0].text = value
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(value)


def _extract_json_object(raw: str) -> Dict:
    return extract_json_object(raw) or {}


def _placeholder_replacements(blocks: List[DocxBlock], details: str) -> Dict[str, str]:
    """Support explicit {{field=value}} data without an LLM for deterministic templates."""
    values: Dict[str, str] = {}
    for key, value in re.findall(r"(?:\{\{|\[)([A-Za-zА-Яа-яЁё0-9_ -]{2,40})\s*=\s*([^}\]\n]+)(?:\}\}|\])", details):
        values[key.strip().lower()] = value.strip()
    replacements: Dict[str, str] = {}
    for block in blocks:
        changed = block.text
        for marker in re.findall(r"\{\{([^{}]{2,40})\}\}|\[([A-ZА-ЯЁ_][A-ZА-ЯЁ0-9_ -]{1,39})\]", changed):
            key = next((part for part in marker if part), "").strip().lower()
            if key in values:
                changed = re.sub(r"\{\{\s*" + re.escape(key) + r"\s*\}\}", values[key], changed, flags=re.IGNORECASE)
                changed = re.sub(r"\[\s*" + re.escape(key) + r"\s*\]", values[key], changed, flags=re.IGNORECASE)
        if changed != block.text:
            replacements[block.block_id] = changed
    return replacements


def _request_value(request: str, pattern: str) -> str:
    match = re.search(pattern, request, flags=re.IGNORECASE | re.DOTALL)
    return " ".join(match.group(1).strip(" ,.;:\n\t").split()) if match else ""


def _apply_command_trip_form(document: Document, request: str) -> int:
    """Fill the first command-trip report form without exposing old data to an LLM."""
    if "командиров" not in request.lower():
        return 0
    dates = re.findall(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b", request)
    destination = _request_value(request, r"(?:в\s+город|в\s+г\.)\s+([А-ЯЁа-яёA-Za-z -]+?)(?=\s+с\s+\d|[,.;\n])")
    person = _request_value(request, r"\bдля\s+(.+?)(?=,\s*(?:в\s+город|в\s+г\.|преподав|сотруд|военно|должност)|\s+в\s+город)")
    purpose = _request_value(
        request,
        r"((?:для\s+участия|для\s+проведения|в\s+связи\s+с)\s+.+?|цель\s*[:—-]?\s*.+?)"
        r"(?=\.\s*(?:Адресат|Автор)|;\s*(?:адресат|автор)|$)",
    )
    purpose = re.sub(r"^цель\s*[:—-]?\s*", "для ", purpose, flags=re.IGNORECASE)
    addressee = _request_value(request, r"адресат\s*[:—-]\s*(.+?)(?=\.\s*Автор|;\s*автор|$)")
    author = _request_value(request, r"автор\s*[:—-]\s*(.+?)(?=,?\s*дата\s+подписи|;|$)")
    signature_date = _request_value(request, r"дата\s+подписи\s*[:—-]?\s*(\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?)")
    if len(dates) >= 2 and not signature_date and len(dates) >= 3:
        signature_date = dates[-1]
    if not all((destination, person, len(dates) >= 2, addressee, author)):
        return 0

    headings = [p for p in document.paragraphs if re.fullmatch(r"\s*рапорт\s*", p.text or "", flags=re.IGNORECASE)]
    if not headings:
        return 0
    # Several source files contain 2-3 completed examples. Keep only the first
    # form in the copy; otherwise old personal data from later forms survives.
    if len(headings) > 1:
        body = document._body._element
        start = headings[1]._p
        deleting = False
        for child in list(body):
            if child is start:
                deleting = True
            if deleting and not child.tag.endswith("}sectPr"):
                body.remove(child)

    paragraphs = list(document.paragraphs)
    heading_index = next((index for index, paragraph in enumerate(paragraphs) if paragraph._p is headings[0]._p), -1)
    if heading_index < 0:
        return 0
    applied = 0
    # Address block is the last non-empty paragraph before the heading.
    for paragraph in reversed(paragraphs[:heading_index]):
        if paragraph.text.strip():
            _set_paragraph_text(paragraph, addressee)
            applied += 1
            break
    body_paragraph = None
    for paragraph in paragraphs[heading_index + 1:]:
        if "прошу" in paragraph.text.lower() or "командиров" in paragraph.text.lower():
            body_paragraph = paragraph
            break
    if body_paragraph is None:
        return 0
    main_text = (
        f"Прошу направить {person} в служебную командировку в город {destination} "
        f"в период с {dates[0]} по {dates[1]}"
        + (f" {purpose}" if purpose else "")
        + "."
    )
    _set_paragraph_text(body_paragraph, main_text)
    applied += 1
    # Remove remaining completed-example prose from the retained form.
    after_main = False
    for paragraph in list(document.paragraphs):
        if paragraph._p is body_paragraph._p:
            after_main = True
            continue
        if after_main and paragraph.text.strip() and not re.fullmatch(r"\s*рапорт\s*", paragraph.text, flags=re.IGNORECASE):
            _set_paragraph_text(paragraph, "")
            applied += 1
    if document.tables:
        table = document.tables[0]
        if table.rows and len(table.rows[0].cells) >= 2:
            left = table.rows[0].cells[0]
            right = table.rows[0].cells[1]
            left_nonempty = [p for p in left.paragraphs if p.text.strip()]
            right_nonempty = [p for p in right.paragraphs if p.text.strip()]
            target_left = left_nonempty[0] if left_nonempty else left.paragraphs[0]
            target_right = right_nonempty[0] if right_nonempty else right.paragraphs[0]
            _set_paragraph_text(target_left, signature_date or dates[0])
            _set_paragraph_text(target_right, author)
            for paragraph in left.paragraphs:
                if paragraph._p is not target_left._p:
                    _set_paragraph_text(paragraph, "")
            for paragraph in right.paragraphs:
                if paragraph._p is not target_right._p:
                    _set_paragraph_text(paragraph, "")
            applied += 2
    return applied


def build_patch_prompt(blocks: List[DocxBlock], request: str) -> str:
    block_payload = [{"id": block.block_id, "text": block.text} for block in blocks]
    return f"""Ты редактируешь КОПИЮ юридического DOCX по данным пользователя.
Верни ТОЛЬКО JSON вида {{"replacements": {{"block_id": "новый полный текст блока"}}}}.

Правила:
1. Меняй только блоки, необходимые для нового документа.
2. Удали из изменяемых блоков старые ФИО, даты, города и номера, заменив их данными пользователя.
3. Не выдумывай отсутствующие реквизиты и не оставляй старые персональные данные.
4. Не добавляй markdown и комментарии.
5. Идентификаторы блоков копируй точно.

Запрос и уточнённые данные:
{request}

Блоки шаблона:
{json.dumps(block_payload, ensure_ascii=False)}
"""


def _sensitive_candidates(text: str) -> List[str]:
    values = set(re.findall(r"\b\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\b", text))
    values.update(re.findall(r"\b(?:19|20)\d{2}\s*(?:г\.?|года)?\b", text, flags=re.IGNORECASE))
    values.update(re.findall(r"\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}\b", text))
    values.update(re.findall(r"\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.\b", text))
    return sorted((value.strip() for value in values if value.strip()), key=len, reverse=True)


def _validate(original_text: str, output_text: str, request: str) -> List[str]:
    errors: List[str] = []
    if re.search(r"\{\{[^{}]+\}\}|\[(?:УТОЧНИТЬ|[A-ZА-ЯЁ_][A-ZА-ЯЁ0-9_ -]{2,})\]", output_text):
        errors.append("В документе остались незаполненные поля.")
    normalized_request = request.lower().replace("ё", "е")
    stale = []
    for value in _sensitive_candidates(original_text):
        normalized = value.lower().replace("ё", "е")
        if value in output_text and normalized not in normalized_request:
            stale.append(value)
    if stale:
        errors.append("В копии могут оставаться реквизиты исходного документа; требуется дополнительное уточнение.")
    if len(output_text.strip()) < 40:
        errors.append("Получившийся документ не содержит достаточного текста.")
    return errors


def _sanitize_package(path: Path) -> None:
    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with tempfile.TemporaryDirectory(prefix="chatkodi_docx_") as tmp:
        rebuilt = Path(tmp) / "sanitized.docx"
        with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(rebuilt, "w", zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                name = info.filename
                if name.startswith("word/embeddings/") or name.endswith("vbaProject.bin"):
                    continue
                data = source.read(info)
                if name.endswith(".rels"):
                    try:
                        root = ET.fromstring(data)
                        changed = False
                        for relationship in list(root):
                            target_mode = relationship.attrib.get("TargetMode", "")
                            rel_target = relationship.attrib.get("Target", "").replace("\\", "/").lower()
                            rel_type = relationship.attrib.get("Type", "").lower()
                            if target_mode.lower() == "external" or "embeddings/" in rel_target or "oleobject" in rel_type:
                                root.remove(relationship)
                                changed = True
                        if changed:
                            ET.register_namespace("", rel_ns)
                            data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
                    except Exception:
                        pass
                target.writestr(info, data)
        shutil.copy2(rebuilt, path)


def create_legal_draft(
    template_path: Path,
    request: str,
    draft_dir: Path,
    generate_raw: Callable[[str], str],
) -> ArtifactDraft:
    document = Document(str(template_path))
    blocks = _iter_blocks(document)
    original_text = "\n".join(block.text for block in blocks)
    replacements = _placeholder_replacements(blocks, request)
    deterministic_applied = 0
    if not replacements:
        deterministic_applied = _apply_command_trip_form(document, request)
        if deterministic_applied:
            blocks = _iter_blocks(document)
    if not replacements and not deterministic_applied:
        payload = _extract_json_object(generate_raw(build_patch_prompt(blocks, request)))
        raw_replacements = payload.get("replacements") if isinstance(payload, dict) else None
        if isinstance(raw_replacements, dict):
            replacements = {str(key): str(value) for key, value in raw_replacements.items() if isinstance(value, (str, int, float))}

    by_id = {block.block_id: block for block in blocks}
    applied = deterministic_applied
    for block_id, value in replacements.items():
        block = by_id.get(block_id)
        if block is None or not str(value).strip():
            continue
        _set_paragraph_text(block.paragraph, str(value).strip())
        applied += 1

    draft_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = uuid.uuid4().hex
    draft_path = draft_dir / f"legal_{artifact_id}.docx"
    document.save(str(draft_path))
    _sanitize_package(draft_path)

    check_document = Document(str(draft_path))
    output_text = "\n".join(block.text for block in _iter_blocks(check_document))
    errors = []
    if not applied:
        errors.append("Модель не вернула проверяемые замены для выбранного шаблона.")
    errors.extend(_validate(original_text, output_text, request))
    return ArtifactDraft(
        artifact_id=artifact_id,
        kind="legal_docx",
        template_path=template_path,
        draft_path=draft_path,
        suggested_name=f"Новый_{template_path.stem}.docx",
        preview_text=output_text,
        can_save=not errors,
        validation_errors=list(dict.fromkeys(errors)),
    )
