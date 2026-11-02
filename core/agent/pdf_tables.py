"""Page-by-page PDF reading with spatial tables and OCR for image pages.

Never truncate at a page count or flatten a points table into an unordered stream
of numbers. Reading is local; callers cache results for an attached document.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from core.agent import vision


@dataclass
class PDFReading:
    text: str = ""
    pages: int = 0
    recognized: bool = False
    issues: list[str] = field(default_factory=list)


def _vertical_cell(page, bbox, raw):
    """Some PDFs place upright glyphs vertically without rotation metadata.

    Recover reading direction from the actual glyph coordinates and content
    order, rather than reversing a mixed two-column text string.
    """
    if not raw or raw.count('\n') < 2 or bbox is None:
        return raw
    x0, top, x1, bottom = bbox
    chars = [c for c in page.chars if x0 <= (c['x0'] + c['x1']) / 2 <= x1
             and top <= (c['top'] + c['bottom']) / 2 <= bottom and c['text'].strip()]
    if len(chars) < 3:
        return raw
    columns = []
    for char in sorted(chars, key=lambda c: c['x0']):
        if not columns or abs(columns[-1][0] - char['x0']) > 1.5:
            columns.append((char['x0'], []))
        columns[-1][1].append(char)
    if len(columns) > 5 or max(len(c) for _, c in columns) < 3:
        return raw
    heights = [abs(c['bottom'] - c['top']) for c in chars]
    if max(c['top'] for c in chars) - min(c['top'] for c in chars) < 2 * min(heights):
        return raw
    words = []
    for _, column in columns:
        original = [c for c in chars if c in column]
        steps = [b['top'] - a['top'] for a, b in zip(original, original[1:]) if abs(b['top'] - a['top']) > 1]
        reverse = sum(steps) < 0
        words.append(''.join(c['text'] for c in sorted(column, key=lambda c: c['top'], reverse=reverse)))
    return ' '.join(words)


def _page_markdown(page) -> str:
    from core.agent.attachments import _cell, _markdown_block, _fitness_header

    tables = page.find_tables()
    # Text-aligned tables without borders get a second, supported extraction.
    if not tables:
        text = page.extract_text() or ""
        if "балл" in text.lower() and ("результат" in text.lower() or "количество" in text.lower()):
            tables = page.find_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text",
                                                       "min_words_vertical": 2})
    boxes = [t.bbox for t in tables]

    def outside(obj):
        return not any(obj.get("x0", -1) >= x0 - 1 and obj.get("x1", -1) <= x1 + 1
                       and obj.get("top", -1) >= top - 1 and obj.get("bottom", -1) <= bottom + 1
                       for x0, top, x1, bottom in boxes)

    events = []
    words = page.filter(outside).extract_words()
    lines = []
    for word in sorted(words, key=lambda w: (round(w["top"] / 3), w["x0"])):
        if not lines or abs(lines[-1][0] - word["top"]) > 3:
            lines.append([word["top"], []])
        lines[-1][1].append(word)
    for top, line in lines:
        events.append((top, min(w["x0"] for w in line), " ".join(w["text"] for w in sorted(line, key=lambda w: w["x0"]))))
    for table in tables:
        extracted = table.extract()
        rows = [[_cell(_vertical_cell(page, table.rows[ri].cells[ci], value))
                 for ci, value in enumerate(row)] for ri, row in enumerate(extracted)]
        rows = [row for row in rows if any(row)]
        prefix = []
        while len(rows) > 1 and sum(bool(c) for c in rows[0]) == 1:
            prefix.append(next(c for c in rows.pop(0) if c))
        if rows:
            events.append((table.bbox[1], table.bbox[0], "\n".join(prefix + _markdown_block(_fitness_header(rows)))))
    return "\n\n".join(text for _, _, text in sorted(events))


def read_pdf(path: Path, llm: Any = None, progress: Optional[Callable[[str], None]] = None) -> PDFReading:
    import io
    import pdfplumber

    reading = PDFReading()
    texts = []
    with pdfplumber.open(path) as pdf:
        reading.pages = len(pdf.pages)
        for index, page in enumerate(pdf.pages, 1):
            if progress:
                progress(f"Читаю PDF «{path.name}»: страница {index} из {reading.pages}…")
            try:
                text = _page_markdown(page)
                # A scan can have a text-layer page number or a short running title.
                large_image = any((image.get("x1", 0) - image.get("x0", 0)) *
                                  (image.get("bottom", 0) - image.get("top", 0)) > page.width * page.height * 0.35
                                  for image in page.images)
                needs_ocr = bool(page.images) and "|---" not in text and (
                    len((page.extract_text() or "").strip()) < 200 or large_image)
                if needs_ocr:
                    if getattr(llm, "supports_vision", False):
                        image = page.to_image(resolution=180).original
                        out = io.BytesIO()
                        image.save(out, "PNG")
                        text, doubts = vision.read_image(llm, out.getvalue(), progress)
                        reading.recognized = True
                        if doubts:
                            for cell in doubts:
                                text = vision._set_cell(text, cell, "[проверьте цифру]")
                            reading.issues.append(f"стр. {index}: неуверенные цифры исключены из нормативов")
                    else:
                        reading.issues.append(f"стр. {index}: для скана нужна модель зрения")
                        text += "\n[Скан не распознан: нужна модель зрения]"
                texts.append(f"--- Страница {index} ---\n{text}")
            except Exception as exc:
                # No silent missing pages. The caller can show the incomplete read.
                reading.issues.append(f"стр. {index}: не удалось прочитать ({type(exc).__name__})")
                texts.append(f"--- Страница {index} ---\n[Страница не прочитана]")
            finally:
                page.close()  # release page layout and image caches in large PDFs
    reading.text = "\n\n".join(texts)
    return reading
