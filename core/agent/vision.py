"""Photos and scans -> text with the local model's vision projector.

The model rewrites what is on the image (tables as Markdown); when there is
no text it describes the picture. PDF pages without a text layer are rendered
to images and read the same way.
"""

from __future__ import annotations

import base64
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.logging_utils import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
# Long side in pixels: enough for small print on an A4 photo, few image tokens.
MAX_SIDE = 2000

TRANSCRIBE_PROMPT = (
    "Перепиши весь текст с изображения дословно, сохраняя порядок строк. "
    "Таблицы оформи в Markdown (| … |), числа переписывай точно, как на изображении. "
    "Сохраняй пустые ячейки. Для повторяющихся групп столбцов повторяй заголовки, "
    "не объединяй результаты разных упражнений. Неуверенные цифры обозначай [?], не угадывай. "
    "Ничего не добавляй и не пропускай, не исправляй и не комментируй. "
    "Если текста на изображении нет — кратко и точно опиши, что изображено."
)


def image_payload(data: bytes, max_side: int = MAX_SIDE, upscale: bool = False) -> str:
    """JPEG data URL of an image, EXIF-rotated and limited to ``max_side`` (or enlarged to it)."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        scale = max_side / max(img.size)
        if scale < 1 or (upscale and scale > 1):
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def clean_transcription(text: str) -> str:
    text = (text or "").strip()
    # «```markdown … ```» around the whole answer
    text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def ask_image(llm: Any, data: bytes, prompt: str, *, max_tokens: int = 2500,
              probs: Optional[List[Dict[str, Any]]] = None, max_side: int = MAX_SIDE, upscale: bool = False) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_payload(data, max_side, upscale)}},
            {"type": "text", "text": prompt},
        ],
    }]
    parts: List[str] = []
    extra = {"token_logprobs": probs} if probs is not None else {}
    llm.generate_with_callback(
        messages,
        lambda token, final: parts.append(token) if token else None,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        context_limit=0,
        **extra,
    )
    return "".join(parts)


@dataclass
class CellDoubt:
    """A number of a recognised table the model was unsure of."""
    line: int  # line of the reading
    column: int
    value: str
    alternative: str  # the same cell with the model's second choice for the unsure digit
    confidence: float
    row_label: str = ""
    column_name: str = ""
    readings: List[str] = field(default_factory=list)  # the cell in every reading, when it was read again

    def describe(self) -> str:
        where = f"«{self.row_label}»" + (f", «{self.column_name}»" if self.column_name else "")
        if len(self.readings) > 1:
            return f"{where}: {self.value} (прочитано {len(self.readings)} раза: {' / '.join(self.readings)})"
        return f"{where}: {self.value} — возможно {self.alternative}"


def cell_confidence(probs: List[Dict[str, Any]]) -> Tuple[str, List[CellDoubt]]:
    """The reading and, for every number cell of its tables, how sure the model was of
    its least sure digit and what it would have written instead."""
    text, spans = "", []
    for item in probs:
        token = item.get("token") or ""
        spans.append((len(text), len(text) + len(token), item))
        text += token
    cells: List[CellDoubt] = []
    header: List[str] = []
    offset = 0
    for line_no, line in enumerate(text.split("\n")):
        start, offset = offset, offset + len(line) + 1
        if not line.strip().startswith("|"):
            header = []
            continue
        parts = line.split("|")
        if all(re.fullmatch(r"\s*:?-{2,}:?\s*", p) for p in parts[1:-1] if p.strip()):
            continue  # the |---| rule
        if not header:
            header = [p.strip() for p in parts[1:-1]]
            continue
        pos = start
        for col, part in enumerate(parts):
            cell_start, pos = pos, pos + len(part) + 1
            value = part.strip()
            # col 0 is before the first «|», col 1 is the row number / name
            if col <= 1 or not re.fullmatch(r"[-+]?\d[\d\s]*(?:[.,]\d+)?", value):
                continue
            worst, alternative = 1.0, value
            for begin, end, item in spans:
                if end <= cell_start or begin >= cell_start + len(part) or not re.search(r"\d", item.get("token") or ""):
                    continue
                p = math.exp(item.get("logprob", 0.0))
                if p < worst:
                    worst = p
                    chosen = item.get("token") or ""
                    alt = next((a.get("token") or "" for a in item.get("top_logprobs") or []
                                if (a.get("token") or "") != chosen and re.fullmatch(r"\s*[\d.,]+", a.get("token") or "")), "")
                    local = cell_start + len(part) - len(part.lstrip())
                    alternative = value
                    if alt:
                        cut = max(0, begin - local)
                        raw = part.strip()
                        alternative = (raw[:cut] + alt.strip() + raw[cut + len(chosen.strip()):]).strip()
            cells.append(CellDoubt(line_no, col - 1, value, alternative, worst,
                                   row_label=" ".join(p.strip() for p in parts[1:3] if p.strip()),
                                   column_name=header[col - 1] if col - 1 < len(header) else ""))
    return text, cells


# A sheet photographed sideways: printed text the model reads in any position, handwriting
# it does not («Федоров» becomes «38006»). Asking the model where the top is does not work
# (it answers «top» almost always), short readings are noisy; what works is reading the
# photo in the two side positions and keeping the reading with clearly more real words.
# Known Russian words per 100 characters of a reading: upright photos of the eval corpus
# score 2–10, the sideways handwritten sheet 0,8 — below this the side positions are tried.
READABLE = 1.2
# A number cell whose least sure digit is below this is shown as doubtful. On the
# handwriting benchmark (tools/eval_handwriting.py) it flags 7 of 10 wrong cells and
# 4 % of the right ones.
DOUBT_BELOW = 0.8


def known_words(text: str) -> int:
    from core.lemmatizer import RussianLemmatizer

    morph = RussianLemmatizer().morph
    if morph is None:
        return 0
    return sum(1 for w in re.findall(r"[А-Яа-яЁё]{3,}", text or "") if morph.word_is_known(w.lower()))


def readability(text: str) -> float:
    chars = len(re.sub(r"[\s|\-:]+", "", text or ""))
    return 100.0 * known_words(text) / chars if chars else 0.0


def rotated(data: bytes, angle: int) -> bytes:
    """The image turned counter-clockwise by ``angle`` degrees (after its EXIF turn)."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB").rotate(angle, expand=True)
        out = io.BytesIO()
        img.save(out, "JPEG", quality=92)
    return out.getvalue()


def _read_once(llm: Any, data: bytes) -> Tuple[str, List[CellDoubt]]:
    probs: List[Dict[str, Any]] = []
    raw = ask_image(llm, data, TRANSCRIBE_PROMPT, probs=probs, max_tokens=6000)
    cells = cell_confidence(probs)[1] if probs else []
    return clean_transcription(raw), cells


def upright_reading(llm: Any, data: bytes, progress: Optional[Callable[[str], None]] = None) -> Tuple[int, str, List[CellDoubt]]:
    """(the turn that makes the photo readable, its text, how sure the model was of every number)."""
    text, cells = _read_once(llm, data)
    if readability(text) >= READABLE:
        return 0, text, cells
    if progress:
        progress("Текст читается плохо — проверяю, не снят ли лист боком…")
    readings = {0: (text, cells)}
    for angle in (90, 270):  # counter-clockwise / clockwise
        readings[angle] = _read_once(llm, rotated(data, angle))
    known = {angle: known_words(reading[0]) for angle, reading in readings.items()}
    best = max((90, 270), key=lambda a: known[a])
    turn = best if known[best] >= max(known[0] + 3, 1.5 * known[0]) else 0
    logger.info("Sideways check: known words %s, turned by %s°", known, turn)
    return (turn,) + readings[turn]


def _contrast(data: bytes) -> bytes:
    from PIL import Image, ImageFilter, ImageOps

    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.autocontrast(ImageOps.exif_transpose(img).convert("L"), cutoff=2)
        img = img.filter(ImageFilter.UnsharpMask(2, 120, 3)).convert("RGB")
        out = io.BytesIO()
        img.save(out, "JPEG", quality=92)
    return out.getvalue()


def _cell_key(cell: CellDoubt) -> Tuple[str, int]:
    return (cell.row_label.split()[0] if cell.row_label.split() else "", cell.column)


def _set_cell(text: str, cell: CellDoubt, value: str) -> str:
    """The reading with one table cell replaced (found by its row and column)."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        parts = line.split("|")
        if not line.strip().startswith("|") or len(parts) <= cell.column + 1:
            continue
        label = " ".join(p.strip() for p in parts[1:3] if p.strip())
        if label == cell.row_label and parts[cell.column + 1].strip() == cell.value:
            parts[cell.column + 1] = parts[cell.column + 1].replace(cell.value, value, 1)
            lines[i] = "|".join(parts)
            break
    return "\n".join(lines)


def second_look(llm: Any, data: bytes, text: str, doubts: List[CellDoubt],
                progress: Optional[Callable[[str], None]] = None) -> Tuple[str, List[CellDoubt]]:
    """The photo read twice more (enlarged; contrast-enhanced) and every doubtful cell
    voted on. Two of three readings agreeing decide the value; three agreeing clear the
    doubt; no agreement keeps the first reading. On the handwriting benchmark this put
    12 of 14 doubtful cells right instead of 9, and every cell read alike three times was right."""
    if not doubts:
        return text, doubts
    if progress:
        progress(f"Перечитываю неуверенно распознанные цифры ({len(doubts)})…")
    others = []
    for kwargs, image in (({"max_side": 3000, "upscale": True}, data), ({}, _contrast(data))):
        probs: List[Dict[str, Any]] = []
        ask_image(llm, image, TRANSCRIBE_PROMPT, probs=probs, **kwargs)
        others.append({_cell_key(c): c for c in cell_confidence(probs)[1]})
    norm = lambda v: re.sub(r"\s+", "", (v or "").replace(".", ","))
    left: List[CellDoubt] = []
    for doubt in doubts:
        seen = [doubt.value] + [o[_cell_key(doubt)].value for o in others if _cell_key(doubt) in o]
        counts: Dict[str, int] = {}
        for value in seen:
            counts[norm(value)] = counts.get(norm(value), 0) + 1
        winner, top = max(counts.items(), key=lambda kv: kv[1])
        if top == len(seen) == 3:
            continue  # read alike three times: no longer a doubt
        doubt.readings = seen
        if top >= 2 and winner != norm(doubt.value):
            value = next(v for v in seen if norm(v) == winner)
            text = _set_cell(text, doubt, value)
            doubt.alternative, doubt.value = doubt.value, value
        left.append(doubt)
    logger.info("Second look: %s doubtful cells, %s left, values changed: %s", len(doubts), len(left),
                sum(1 for d in left if d.readings and norm(d.readings[0]) != norm(d.value)))
    return text, left


def verify_numbers(llm: Any, data: bytes, text: str, progress=None) -> List[CellDoubt]:
    """Check numeric cells independently even if the runtime supplies no logprobs.

    Disagreements remain unresolved; neither reading is silently preferred.
    A changed table shape marks every numeric cell for human verification.
    """
    from core.agent.table_check import parse_markdown_tables
    from core.agent.grading import value_of

    tables = parse_markdown_tables(text)
    if not tables or not any(value_of(c) is not None for t in tables for row in t.rows for c in row):
        return []
    if progress:
        progress("Повторно сверяю числа и структуру таблицы на увеличенном изображении…")
    other = clean_transcription(ask_image(llm, data, TRANSCRIBE_PROMPT, max_tokens=6000,
                                         max_side=3000, upscale=True))
    again = parse_markdown_tables(other)
    same_shape = len(tables) == len(again) and all(
        a.header == b.header and len(a.rows) == len(b.rows) for a, b in zip(tables, again))
    doubts = []
    for ti, table in enumerate(tables):
        for ri, row in enumerate(table.rows):
            for ci, value in enumerate(row):
                identity = ci < len(table.header) and bool(re.search(r"фио|фамили|full name", table.header[ci], re.I))
                if value_of(value) is None and not identity:
                    continue
                alternative = (again[ti].rows[ri][ci] if same_shape and ci < len(again[ti].rows[ri])
                               else "структура таблицы не совпала")
                different = (" ".join(value.casefold().split()) != " ".join(alternative.casefold().split())
                             if identity else value_of(value) != value_of(alternative))
                if not same_shape or different:
                    doubts.append(CellDoubt(-1, ci, value, alternative, 0.0,
                                            row_label=" ".join(c for c in row[:2] if c),
                                            column_name=table.header[ci] if ci < len(table.header) else "",
                                            readings=[value, alternative]))
    return doubts


def read_image(llm: Any, data: bytes, progress: Optional[Callable[[str], None]] = None) -> Tuple[str, List[CellDoubt]]:
    """The text of a photo and the numbers of its tables the model is still unsure of
    (with their doubtful ones read again)."""
    turn, text, cells = upright_reading(llm, data, progress)
    upright = rotated(data, turn) if turn else data
    initial = doubtful(cells)
    text, doubts = second_look(llm, upright, text, initial, progress)
    fitness_table = bool(re.search(r"№\s*упражнен|количество.*раз", text, re.I | re.S))
    if llm is not None and (not cells or (fitness_table and not initial)):
        doubts += verify_numbers(llm, upright, text, progress)
    return text, doubts


def transcribe_image(llm: Any, data: bytes, progress: Optional[Callable[[str], None]] = None) -> str:
    return read_image(llm, data, progress)[0]


def pdf_page_images(path: Path, *, max_pages: int = 30, scale: float = 2.2) -> List[bytes]:
    """PNG renders of PDF pages (for scans without a text layer)."""
    import pypdfium2 as pdfium

    pages: List[bytes] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for index in range(min(len(pdf), max_pages)):
            page = pdf[index]
            image = page.render(scale=scale).to_pil()
            out = io.BytesIO()
            image.save(out, "PNG")
            pages.append(out.getvalue())
    finally:
        pdf.close()
    return pages


def read_pages(llm: Any, pages: List[bytes], progress: Optional[Callable[[str], None]] = None) -> Tuple[str, List[CellDoubt]]:
    texts: List[str] = []
    doubts: List[CellDoubt] = []
    for index, data in enumerate(pages, 1):
        if progress:
            progress(f"Распознаю страницу {index} из {len(pages)}…")
        text, cells = read_image(llm, data, progress)
        texts.append(f"--- Страница {index} ---\n{text}")
        doubts += doubtful(cells)
    return "\n\n".join(texts), doubts


def transcribe_pages(llm: Any, pages: List[bytes], progress: Optional[Callable[[str], None]] = None) -> str:
    return read_pages(llm, pages, progress)[0]


def doubtful(cells: List[CellDoubt]) -> List[CellDoubt]:
    return [c for c in cells if c.confidence < DOUBT_BELOW]
