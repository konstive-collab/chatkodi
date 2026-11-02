"""Presentations: Markdown slides to PPTX, in the look of an example presentation.

The model writes «# Название», a subtitle line, then «## Заголовок слайда» with
«- пункты» and «|» tables. An example .pptx gives the design: its theme, masters
and layouts stay, its slides are removed, new slides use the layouts of its
title and content slides. Without an example a plain 16:9 deck is made.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from core.agent.table_check import parse_markdown_tables
from core.logging_utils import get_logger

logger = get_logger(__name__)

MAX_BULLETS = 7  # more go to a continuation slide
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_BULLET = re.compile(r"^(\s*)(?:[-*•]|\d+[.)])\s+(.*)$")


@dataclass
class Slide:
    title: str
    bullets: List[tuple] = field(default_factory=list)  # (level, text)
    table: Optional[List[List[str]]] = None


def parse_slides(markdown: str):
    """(title, subtitle lines, slides) of a Markdown presentation."""
    title, subtitle, slides = "", [], []
    current: Optional[Slide] = None
    lines = (markdown or "").splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        stripped = raw.strip()
        i += 1
        if not stripped or re.fullmatch(r"-{3,}|\*{3,}", stripped):
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            text = heading.group(2).strip().strip("*")
            text = re.sub(r"^(?:слайд|slide)\s*\d+\s*[.:—–-]\s*", "", text, flags=re.IGNORECASE) or text
            if len(heading.group(1)) == 1 and not title and current is None:
                title = text
            else:
                current = Slide(text)
                slides.append(current)
            continue
        if stripped.startswith("|"):
            block = [raw]
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            tables = parse_markdown_tables("\n".join(block))
            if tables:
                if current is None:
                    current = Slide(title or "")
                    slides.append(current)
                table = tables[0]
                current.table = [table.header] + table.rows + table.total_rows
            continue
        bullet = _BULLET.match(raw)
        text, level = (bullet.group(2).strip(), min(2, len(bullet.group(1).replace("\t", "  ")) // 2)) if bullet else (stripped, 0)
        if current is None:
            subtitle.append(text)
        else:
            current.bullets.append((level, text))
    return title, subtitle, slides


def _key(text: str) -> str:
    return re.sub(r"[^а-яa-z0-9]+", " ", (text or "").lower().replace("ё", "е")).strip()


def drop_example_lines(markdown: str, example: str, data: str) -> str:
    """Lines copied from the example deck («Показатель 1», «Отдел, дата») that the data
    does not have: the example gives the form, never the content. Headings stay."""
    copied = {_key(re.sub(r"^\s*(?:[-*•]\s+|#+\s+)", "", line)) for line in (example or "").splitlines()}
    copied.discard("")
    data_key = _key(data)
    kept = []
    for line in (markdown or "").splitlines():
        stripped = line.strip()
        body = _key(re.sub(r"^\s*(?:[-*•]\s+)", "", stripped))
        if not stripped.startswith("#") and not stripped.startswith("|") and body in copied and body not in data_key:
            continue
        kept.append(line)
    return "\n".join(kept)


def _runs(paragraph, text: str, size=None) -> None:
    from pptx.util import Pt

    pos = 0
    pieces = []
    for match in _BOLD.finditer(text):
        if match.start() > pos:
            pieces.append((text[pos:match.start()], False))
        pieces.append((match.group(1), True))
        pos = match.end()
    if pos < len(text):
        pieces.append((text[pos:], False))
    for piece, bold in pieces or [("", False)]:
        run = paragraph.add_run()
        run.text = piece
        if bold:
            run.font.bold = True
        if size:
            run.font.size = Pt(size)


def _placeholder(slide, *types):
    from pptx.enum.shapes import PP_PLACEHOLDER

    wanted = {getattr(PP_PLACEHOLDER, t) for t in types}
    for shape in slide.placeholders:
        if shape.placeholder_format.type in wanted:
            return shape
    return None


def _body_of(slide):
    return _placeholder(slide, "BODY", "OBJECT", "SUBTITLE")


def _layout_has_body(layout) -> bool:
    from pptx.enum.shapes import PP_PLACEHOLDER

    return any(p.placeholder_format.type in {PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT} for p in layout.placeholders)


def _layouts(prs, from_example: bool):
    """Title and content layouts: those the example's own slides use, else the template's."""
    layouts = list(prs.slide_layouts)
    title = layouts[0]
    content = next((l for l in layouts[1:2] if _layout_has_body(l)), None) or next((l for l in layouts if _layout_has_body(l)), layouts[-1])
    if from_example and len(prs.slides):
        title = prs.slides[0].slide_layout
        used = [s.slide_layout for s in list(prs.slides)[1:] if _layout_has_body(s.slide_layout)]
        if used:
            content = max(used, key=used.count)  # the layout most slides of the example use
    return title, content


def _drop_slides(prs) -> None:
    ids = prs.slides._sldIdLst
    for slide_id in list(ids):
        prs.part.drop_rel(slide_id.rId)
        ids.remove(slide_id)


def _font_size(bullets) -> Optional[int]:
    chars = sum(len(text) for _level, text in bullets)
    if chars > 700:
        return 14
    if chars > 450:
        return 16
    if chars > 300:
        return 18
    return None  # the layout's own size


def _fill_bullets(shape, bullets) -> None:
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    size = _font_size(bullets)
    for index, (level, text) in enumerate(bullets):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.level = level
        _runs(paragraph, text, size)


def _add_table(slide, rows, left, top, width, height) -> None:
    from pptx.util import Pt

    width_cols = max(len(r) for r in rows)
    shape = slide.shapes.add_table(len(rows), width_cols, left, top, width, height)
    size = 14 if len(rows) <= 6 else 12 if len(rows) <= 10 else 10
    for r, row in enumerate(rows):
        for c in range(width_cols):
            cell = shape.table.cell(r, c)
            cell.text = ""
            _runs(cell.text_frame.paragraphs[0], row[c] if c < len(row) else "", size)
            if r == 0:
                for run in cell.text_frame.paragraphs[0].runs:
                    run.font.bold = True


def _widen(slide, factor: float) -> None:
    """Default template placeholders are laid out for 4:3; the deck is 16:9."""
    if abs(factor - 1) < 0.01:
        return
    for shape in slide.placeholders:
        # an inherited position is written whole: setting only «left» zeroes the rest
        left, top, width, height = shape.left, shape.top, shape.width, shape.height
        shape.left, shape.top, shape.width, shape.height = int(left * factor), top, int(width * factor), height


def markdown_to_pptx(markdown: str, path: Path, *, base_pptx: Optional[Path] = None) -> Path:
    from pptx import Presentation
    from pptx.util import Emu, Inches

    prs, from_example = None, False
    if base_pptx is not None and base_pptx.suffix.lower() == ".pptx" and base_pptx.is_file():
        try:
            prs = Presentation(str(base_pptx))
            from_example = True
        except Exception as exc:
            logger.warning("Example PPTX could not be used as a base: %s", exc)
    factor = 1.0
    if prs is None:
        prs = Presentation()
        factor = 13.333 / 10
        prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    title_layout, content_layout = _layouts(prs, from_example)
    if from_example:
        _drop_slides(prs)

    title, subtitle, slides = parse_slides(markdown)
    first = prs.slides.add_slide(title_layout)
    _widen(first, factor)
    if first.shapes.title is not None:
        first.shapes.title.text = title or (slides[0].title if slides else "Презентация")
    sub = _placeholder(first, "SUBTITLE", "BODY")
    if sub is not None:
        if subtitle:
            _fill_bullets(sub, [(0, line) for line in subtitle[:3]])
        else:
            sub._element.getparent().remove(sub._element)

    # long lists continue on the next slide
    pages: List[Slide] = []
    for slide in slides:
        bullets = slide.bullets
        chunks = [bullets[i:i + MAX_BULLETS] for i in range(0, len(bullets), MAX_BULLETS)] or [[]]
        for n, chunk in enumerate(chunks):
            pages.append(Slide(slide.title + (" (продолжение)" if n else ""), chunk, slide.table if n == len(chunks) - 1 else None))

    width, height = prs.slide_width, prs.slide_height
    for page in pages:
        slide = prs.slides.add_slide(content_layout)
        _widen(slide, factor)
        if slide.shapes.title is not None:
            slide.shapes.title.text = page.title
        body = _body_of(slide)
        if body is not None:
            left, top, box_w, box_h = body.left, body.top, body.width, body.height
        else:
            left, top, box_w, box_h = Emu(int(width * 0.06)), Emu(int(height * 0.22)), Emu(int(width * 0.88)), Emu(int(height * 0.7))
        if page.bullets:
            if body is None:
                body = slide.shapes.add_textbox(left, top, box_w, box_h)
            if page.table:
                body.height = Emu(int(box_h * 0.38))  # list above, table below
            _fill_bullets(body, page.bullets)
        elif body is not None:
            body._element.getparent().remove(body._element)
        if page.table:
            table_top = top + (int(box_h * 0.42) if page.bullets else 0)
            table_h = box_h - (int(box_h * 0.42) if page.bullets else 0)
            _add_table(slide, page.table, left, Emu(table_top), box_w, Emu(min(table_h, int(height * 0.06) * len(page.table))))
    path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(path))
    return path


def pptx_form(path: Path, limit: int = 4000) -> str:
    """The example as form only: slide titles and what kind of content each holds.
    Its own words («Показатель 1») are not shown, so the model has nothing to copy."""
    from pptx import Presentation

    prs = Presentation(str(path))
    parts = []
    for index, slide in enumerate(prs.slides, 1):
        title = slide.shapes.title.text.strip() if slide.shapes.title is not None and slide.shapes.title.has_text_frame else ""
        if index == 1:
            parts.append("# [название презентации]\n[подзаголовок]")
            continue
        kinds = []
        for shape in slide.shapes:
            if shape == slide.shapes.title:
                continue
            if getattr(shape, "has_table", False) and shape.has_table:
                kinds.append(f"(таблица: {len(shape.table.columns)} столбца, {len(shape.table.rows)} строк)")
            elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                count = sum(1 for p in shape.text_frame.paragraphs if p.text.strip())
                if count:
                    kinds.append(f"(список: {count} пункт{'а' if 1 < count < 5 else '' if count == 1 else 'ов'})")
        parts.append(f"## {title or 'Слайд ' + str(index)}\n" + ("\n".join(kinds) or "(текст)"))
    return "\n\n".join(parts)[:limit]


def pptx_outline(path: Path, limit: int = 6000) -> str:
    """An example presentation as Markdown slides: what each slide holds, in order."""
    from pptx import Presentation

    prs = Presentation(str(path))
    parts = []
    for index, slide in enumerate(prs.slides, 1):
        title = slide.shapes.title.text.strip() if slide.shapes.title is not None and slide.shapes.title.has_text_frame else ""
        # the first slide is the title page: «# Название» and its subtitle lines, as the model writes them
        first = index == 1
        lines = [f"# {title}" if first else f"## {title or 'Слайд ' + str(index)}"]
        for shape in slide.shapes:
            if shape == slide.shapes.title:
                continue
            if first and getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                lines += [p.strip() for p in shape.text_frame.text.splitlines() if p.strip()]
                continue
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [[cell.text.strip() for cell in row.cells] for row in shape.table.rows]
                if rows:
                    lines.append("| " + " | ".join(rows[0]) + " |")
                    lines.append("|" + "---|" * len(rows[0]))
                    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
            elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in paragraph.runs).strip()
                    if text:
                        lines.append("  " * paragraph.level + "- " + text)
        parts.append("\n".join(lines))
    return "\n\n".join(parts)[:limit]
