"""Files the assistant hands back: DOCX, XLSX, PPTX, TXT/MD made from Markdown text.

DOCX can take its look (fonts, margins, headers/footers) from an example
document: the example is opened, its body is replaced, its styles stay.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from core.agent.table_check import parse_markdown_tables, to_number
from core.logging_utils import get_logger
from core.paths import runtime_root

logger = get_logger(__name__)

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


@dataclass
class OutputFile:
    path: Path
    label: str
    markdown: str = ""  # the text the file was made from (preview, edits)
    base_docx: Optional[Path] = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def fmt(self) -> str:
        return self.path.suffix.lower().lstrip(".")


def rewrite_output(output: "OutputFile", markdown: str) -> "OutputFile":
    """Save edited text into the same file (same format, same example styles)."""
    if output.fmt == "xlsx":
        markdown_to_xlsx(markdown, output.path)
    elif output.fmt == "pptx":
        from core.agent.slides import markdown_to_pptx

        markdown_to_pptx(markdown, output.path, base_pptx=output.base_docx)
    elif output.fmt in {"txt", "md"}:
        output.path.write_text(markdown, encoding="utf-8")
    else:
        markdown_to_docx(markdown, output.path, base_docx=output.base_docx)
    output.markdown = markdown
    return output


def outputs_dir() -> Path:
    folder = runtime_root() / "outputs" / time.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def safe_name(name: str, suffix: str) -> str:
    name = re.sub(r"\[[^\]]*\]|\*+", "", name or "")  # «№ [номер]» placeholders are not a name
    name = re.sub(r"№\s*(?=\s|$)", "", name)
    name = re.sub(r"\s+", " ", name)
    stem = _UNSAFE.sub("_", (name or "Документ").strip()).strip(" ._")[:80] or "Документ"
    return f"{stem}{suffix}"


def unique_path(folder: Path, file_name: str) -> Path:
    path = folder / file_name
    index = 2
    while path.exists():
        path = folder / f"{Path(file_name).stem} ({index}){Path(file_name).suffix}"
        index += 1
    return path


def _add_runs(paragraph, text: str) -> None:
    pos = 0
    for match in _BOLD.finditer(text):
        if match.start() > pos:
            paragraph.add_run(text[pos:match.start()])
        paragraph.add_run(match.group(1)).bold = True
        pos = match.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


def _style(doc, *names: str):
    for name in names:
        try:
            return doc.styles[name]
        except KeyError:
            continue
    return None


def _clear_body(doc) -> None:
    body = doc.element.body
    for child in list(body):
        if not child.tag.endswith("}sectPr"):
            body.remove(child)


def markdown_to_docx(markdown: str, path: Path, *, base_docx: Optional[Path] = None) -> Path:
    from docx import Document
    from docx.shared import Pt

    doc = None
    if base_docx is not None and base_docx.suffix.lower() == ".docx" and base_docx.is_file():
        try:
            doc = Document(str(base_docx))
            _clear_body(doc)
        except Exception as exc:
            logger.warning("Example DOCX could not be used as a base: %s", exc)
            doc = None
    if doc is None:
        doc = Document()
        normal = doc.styles["Normal"]
        normal.font.name = "Times New Roman"
        normal.font.size = Pt(12)

    lines = (markdown or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            tables = parse_markdown_tables("\n".join(block))
            if tables:
                table = tables[0]
                rows = [table.header] + table.rows + table.total_rows
                width = max(len(r) for r in rows)
                grid = doc.add_table(rows=len(rows), cols=width)
                style = _style(doc, "Table Grid", "Сетка таблицы")
                if style is not None:
                    grid.style = style
                for r, row in enumerate(rows):
                    for c in range(width):
                        cell = grid.cell(r, c)
                        cell.text = ""
                        _add_runs(cell.paragraphs[0], row[c] if c < len(row) else "")
                        if r == 0:
                            for run in cell.paragraphs[0].runs:
                                run.bold = True
                doc.add_paragraph()
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            style = _style(doc, f"Heading {level}", f"Заголовок {level}")
            para = doc.add_paragraph(style=style) if style is not None else doc.add_paragraph()
            _add_runs(para, heading.group(2).strip())
            if style is None:
                for run in para.runs:
                    run.bold = True
        elif re.match(r"^[-*•]\s+", stripped):
            para = doc.add_paragraph(style=_style(doc, "List Bullet", "Маркированный список"))
            _add_runs(para, re.sub(r"^[-*•]\s+", "", stripped))
        elif stripped:
            para = doc.add_paragraph()
            _add_runs(para, stripped)
        elif i and lines[i - 1].strip():
            pass  # paragraph spacing comes from the style
        i += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    return path


def docx_to_markdown(path: Path) -> str:
    """A DOCX as Markdown in document order (headings, paragraphs, tables):
    the model sees the example's real structure, not flattened lines."""
    from docx import Document
    from docx.table import Table as DocxTable
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    out: List[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name if para.style is not None else "").lower()
            level = re.search(r"(?:heading|заголовок)\s*(\d)", style)
            out.append(f"{'#' * int(level.group(1))} {text}" if level else ("# " + text if style == "title" else text))
        elif tag == "tbl":
            table = DocxTable(child, doc)
            rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
            if not rows:
                continue
            out.append("| " + " | ".join(rows[0]) + " |")
            out.append("|" + "---|" * len(rows[0]))
            out.extend("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join(out)


_PURE_NUMBER = re.compile(r"[-+]?[\d   ]+(?:[.,]\d+)?")
_MEAN_LABEL = re.compile(r"\b(?:средн\w*|average|mean)\b", re.IGNORECASE)


def _pure(text: str) -> Optional[float]:
    """A cell that is only a number («20 пачек» is not: it stays text)."""
    number = to_number(text)
    return number if number is not None and _PURE_NUMBER.fullmatch(text.strip()) else None


def _agrees(expected: float, text: str) -> bool:
    """The recognised value equals the formula's up to its own written precision."""
    written = to_number(text)
    if written is None:
        return False
    match = re.search(r"[.,](\d+)\s*$", text.strip())
    return abs(expected - written) <= 0.5 * 10 ** -(len(match.group(1)) if match else 0) + 1e-9


_TOTAL_LINE = re.compile(r"\b(?:итог\w*|всего|сумма)\b", re.IGNORECASE)


def _summary_column(line: str, body: List[List[str]]) -> Optional[tuple]:
    """(column, SUM | AVERAGE) when a line under the table — «Средняя прочность
    серии: 34,7 МПа», «Итого: 34 000,00» — is that statistic of one column."""
    func = "AVERAGE" if _MEAN_LABEL.search(line) else "SUM" if _TOTAL_LINE.search(line) else ""
    if not func or not body:
        return None
    from core.agent.table_check import _ANY_NUMBER

    numbers = [m.group(0) for m in _ANY_NUMBER.finditer(line)]
    width = max(len(row) for row in body)
    for raw in numbers[:1] if "%" not in line else numbers[-1:]:
        for c in range(width):
            values = [_pure(row[c]) if c < len(row) else None for row in body]
            if any(v is None for v in values):
                continue
            stat = sum(values) / len(values) if func == "AVERAGE" else sum(values)
            if _agrees(stat, raw):
                return c, func
    return None


def mark_doubts(sheet, rows: List[List[str]], first_row: int, doubts, column_offset: int = 0) -> None:
    """Cells the model read unsure of: orange, with its second choice in the note."""
    from openpyxl.comments import Comment
    from openpyxl.styles import PatternFill

    orange = PatternFill("solid", fgColor="FCD5B4")
    for d in doubts or ():
        for r, row in enumerate(rows):
            label = " ".join(c.strip() for c in row[:2] if c.strip())
            if label == d.row_label and d.column < len(row) and row[d.column].strip() == d.value:
                cell = sheet.cell(row=first_row + r, column=d.column + 1 + column_offset)
                cell.fill = orange
                cell.comment = Comment(f"Распознано неуверенно: {d.value}, возможно {d.alternative}. Сверьте с листом.", "ChatKodi")


def markdown_to_xlsx(markdown: str, path: Path, doubts=()) -> Path:
    """Every Markdown table becomes a sheet; numeric cells become numbers.
    Totals and means become SUM / AVERAGE formulas, «кол-во × цена = сумма»
    becomes a product formula; where the recognised value disagrees with the
    formula the cell is yellow and its note keeps what was recognised. Text
    around the tables (headings, «Итого: …» lines) goes under them."""
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    from core.agent.table_check import _product_relation

    yellow = PatternFill("solid", fgColor="FFF2CC")
    tables = parse_markdown_tables(markdown)
    wb = Workbook()
    ws = wb.active
    if not tables:
        ws.title = "Текст"
        for r, line in enumerate((markdown or "").splitlines(), 1):
            ws.cell(row=r, column=1, value=line)
    outside, seen_table = [], False  # (line, written after a table)
    for line in (markdown or "").splitlines():
        if line.strip().startswith("|"):
            seen_table = True
        elif line.strip():
            outside.append((line.strip(), seen_table))
    clean = lambda rows: [[_BOLD.sub(r"\1", v) for v in row] for row in rows]
    # a «## 7А» heading right before a table names its sheet (a timetable per group)
    names, heading, in_table = [], "", False
    for line in (markdown or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            if not in_table:
                names.append(heading)
                heading = ""
            in_table = True
            continue
        in_table = False
        if stripped.startswith("##"):
            heading = re.sub(r"[\[\]:*?/\\]", " ", stripped.lstrip("#")).strip()[:31]
    for index, table in enumerate(tables):
        sheet = ws if index == 0 else wb.create_sheet()
        name = names[index] if index < len(names) and names[index] and len(tables) > 1 else ""
        sheet.title = name if name and name not in wb.sheetnames else f"Таблица {index + 1}"
        for c, value in enumerate(table.header, 1):
            sheet.cell(row=1, column=c, value=_BOLD.sub(r"\1", value)).font = Font(bold=True)
        body = clean(table.rows)
        # «Среднее» rows at the end of the data are summaries like «Итого»
        means = []
        while len(body) > 1 and _MEAN_LABEL.search(" ".join(body[-1][:2])):
            means.insert(0, body.pop())
        data_first, data_last = 2, 1 + len(body)
        product = _product_relation(table) if not means else None
        if product and not all(_pure(row[product[0]]) is not None and _pure(row[product[1]]) is not None for row in body):
            product = None  # «20 пачек» × price: the formula would see text
        r = 1
        for row in body:
            r += 1
            for c, text in enumerate(row):
                number = _pure(text)
                cell = sheet.cell(row=r, column=c + 1, value=number if number is not None else text)
                if number is not None and re.search(r"[.,]\d{2}$", text.strip()):
                    cell.number_format = "#,##0.00"
            if product:
                i, j, k = product
                recognised = to_number(row[k]) if k < len(row) else None
                cell = sheet.cell(row=r, column=k + 1, value=f"={get_column_letter(i + 1)}{r}*{get_column_letter(j + 1)}{r}")
                cell.number_format = "#,##0.00"
                if recognised is None or not _agrees(_pure(row[i]) * _pure(row[j]), row[k]):
                    cell.fill = yellow
                    cell.comment = Comment(f"Распознано: {row[k] if k < len(row) else '—'}; по формуле {row[i]} × {row[j]}", "ChatKodi")
        mark_doubts(sheet, body, data_first, doubts)
        summaries = [(row, "SUM") for row in clean(table.total_rows)] + [(row, "AVERAGE") for row in means]
        for summary, func in summaries:
            r += 1
            for c, text in enumerate(summary):
                number = _pure(text)
                if product and c == product[2]:
                    # the column the sheet computes: its formula values, not the recognised ones
                    column = [_pure(row[product[0]]) * _pure(row[product[1]]) for row in body]
                else:
                    column = [_pure(row[c]) if c < len(row) else None for row in body]
                if number is not None and body and all(v is not None for v in column):
                    letter = get_column_letter(c + 1)
                    cell = sheet.cell(row=r, column=c + 1, value=f"={func}({letter}{data_first}:{letter}{data_last})")
                    expected = sum(column) / len(column) if func == "AVERAGE" else sum(column)
                    if not _agrees(expected, text):
                        cell.fill = yellow
                        cell.comment = Comment(f"Распознано: {text}; по столбцу получается {expected:g}", "ChatKodi")
                else:
                    cell = sheet.cell(row=r, column=c + 1, value=number if number is not None else text)
                cell.font = Font(bold=True)
        if index == len(tables) - 1 and outside:
            # headings and lines like «Итого: 34 000,00 руб.» stay with the table; a total or
            # a mean that is the sum / mean of a column becomes a formula under that column
            r += 1
            for line, after_table in outside:
                r += 1
                text = _BOLD.sub(r"\1", line).lstrip("#").strip()
                sheet.cell(row=r, column=1, value=text)
                column = _summary_column(text, body) if after_table else None
                if column is not None:
                    c, func = column
                    letter = get_column_letter(c + 1)
                    cell = sheet.cell(row=r, column=c + 1, value=f"={func}({letter}{data_first}:{letter}{data_last})")
                    cell.font = Font(bold=True)
        for column in sheet.columns:
            width = max(10 if str(cell.value or "").startswith("=") else len(str(cell.value or "")) for cell in column)
            sheet.column_dimensions[column[0].column_letter].width = min(60, max(8, width + 2))
    wb.calculation.fullCalcOnLoad = True
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(path))
    return path


def write_output(markdown: str, title: str, fmt: str, *, base_docx: Optional[Path] = None, doubts=()) -> OutputFile:
    fmt = (fmt or "docx").lower().lstrip(".")
    folder = outputs_dir()
    if fmt == "xlsx":
        path = markdown_to_xlsx(markdown, unique_path(folder, safe_name(title, ".xlsx")), doubts)
        return OutputFile(path, "Таблица Excel", markdown)
    if fmt == "pptx":
        from core.agent.slides import markdown_to_pptx

        # base_docx: the example the file takes its look from (a presentation here)
        path = markdown_to_pptx(markdown, unique_path(folder, safe_name(title, ".pptx")), base_pptx=base_docx)
        return OutputFile(path, "Презентация PowerPoint", markdown, base_docx)
    if fmt in {"txt", "md"}:
        path = unique_path(folder, safe_name(title, f".{fmt}"))
        path.write_text(markdown or "", encoding="utf-8")
        return OutputFile(path, "Текстовый файл", markdown)
    path = markdown_to_docx(markdown, unique_path(folder, safe_name(title, ".docx")), base_docx=base_docx)
    return OutputFile(path, "Документ Word", markdown, base_docx)


def list_outputs(limit: int = 50) -> List[Path]:
    root = runtime_root() / "outputs"
    if not root.is_dir():
        return []
    return sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
