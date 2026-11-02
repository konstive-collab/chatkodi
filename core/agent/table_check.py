"""Arithmetic self-check of recognised tables.

A misread digit in a photo of an invoice or a test report breaks the
document's own arithmetic: quantity × price ≠ amount, the total is not the
column sum, the average is not the column mean. Checking that arithmetic
catches recognition errors without knowing anything about the document type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_NUM = r"[-+]?(?:\d{1,3}(?:[   ]\d{3})+|\d+)(?:[.,]\d+)?"
_CELL_NUMBER = re.compile(rf"^\s*({_NUM})\s*(?:[^\d\s][^\d]*)?$")
_ANY_NUMBER = re.compile(rf"(?<![\w.,]){_NUM}(?![\w])")
_TOTAL_WORD = re.compile(r"\b(?:итог\w*|всего|total|сумма\s+(?:по|всего)|в\s+сумме)\b", re.IGNORECASE)
_MEAN_WORD = re.compile(r"\b(?:средн\w*|average|mean)\b", re.IGNORECASE)


@dataclass
class Table:
    header: List[str]
    rows: List[List[str]]
    total_rows: List[List[str]] = field(default_factory=list)


def to_number(raw: str) -> Optional[float]:
    match = _CELL_NUMBER.match(raw or "")
    if not match:
        return None
    try:
        return float(re.sub(r"[   ]", "", match.group(1)).replace(",", "."))
    except ValueError:
        return None


def _decimals(raw: str) -> int:
    match = re.search(r"[.,](\d+)\s*$", re.sub(r"[^\d.,]+$", "", raw.strip()))
    return len(match.group(1)) if match else 0


def parse_markdown_tables(text: str) -> List[Table]:
    tables: List[Table] = []
    block: List[List[str]] = []

    def flush() -> None:
        if len(block) >= 2:
            header, body = block[0], [r for r in block[1:] if not all(re.fullmatch(r":?-{2,}:?", c.strip() or "--") for c in r)]
            table = Table(header=header, rows=[])
            for row in body:
                label = " ".join(row[:2])
                (table.total_rows if _TOTAL_WORD.search(label) else table.rows).append(row)
            if table.rows:
                tables.append(table)
        block.clear()

    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            block.append([c.strip() for c in stripped.strip("|").split("|")])
        else:
            flush()
    flush()
    return tables


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.011, 0.002 * max(abs(a), abs(b)))


def _numeric_columns(table: Table) -> List[Tuple[int, List[float]]]:
    columns = []
    width = max(len(r) for r in table.rows)
    for col in range(width):
        values = [to_number(r[col]) if col < len(r) else None for r in table.rows]
        present = [v for v in values if v is not None]
        if len(present) >= max(2, int(0.8 * len(table.rows))) and len(present) == len(values):
            columns.append((col, present))
    return columns


def _product_relation(table: Table) -> Optional[Tuple[int, int, int]]:
    """Columns (a, b, c) with a × b = c in most rows (quantity × price = amount)."""
    columns = [c for c, _ in _numeric_columns(table)]
    best, best_hits = None, 0
    for i in columns:
        for j in columns:
            if j <= i:
                continue
            for k in columns:
                if k in (i, j):
                    continue
                hits = sum(
                    1 for r in table.rows
                    if _close(to_number(r[i]) * to_number(r[j]), to_number(r[k]))
                )
                if hits > best_hits:
                    best, best_hits = (i, j, k), hits
    if best and best_hits >= max(2, (len(table.rows) + 1) // 2):
        return best
    return None


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}".replace(",", " ")
    return f"{value:,.2f}".replace(",", " ").replace(".", ",")


def check_tables(text: str) -> List[str]:
    """Human-readable notes: ✓ for checks that hold, ⚠ for contradictions."""
    notes: List[str] = []
    tables = parse_markdown_tables(text)
    if not tables:
        return notes
    all_columns: List[Tuple[str, List[float]]] = []
    for table in tables:
        relation = _product_relation(table)
        if relation:
            i, j, k = relation
            bad = [
                r for r in table.rows
                if not _close(to_number(r[i]) * to_number(r[j]), to_number(r[k]))
            ]
            name = table.header[k] if k < len(table.header) else f"столбец {k + 1}"
            if bad:
                for r in bad[:5]:
                    notes.append(
                        f"⚠ Строка «{' '.join(r[:2]).strip()}»: {r[i]} × {r[j]} ≠ {r[k]} — проверьте распознавание."
                    )
            else:
                notes.append(f"✓ «{name}» = произведение двух столбцов во всех {len(table.rows)} строках.")
        for col, values in _numeric_columns(table):
            all_columns.append((table.header[col] if col < len(table.header) else f"столбец {col + 1}", values))
        for row in table.total_rows:
            numbers = [(c, to_number(c)) for c in row if to_number(c) is not None]
            if numbers:
                notes.append(_check_summary(" ".join(row), numbers[-1][0], numbers[-1][1], all_columns, mean=False))

    # «Итого: 34 000,00», «Средняя прочность: 34,7» written under the table
    in_table = False
    for line in text.splitlines():
        if line.strip().startswith("|"):
            in_table = True
            continue
        if not in_table:
            continue
        mean = bool(_MEAN_WORD.search(line))
        if not (mean or _TOTAL_WORD.search(line)):
            continue
        found = [m.group(0) for m in _ANY_NUMBER.finditer(line)]
        if not found:
            continue
        raw = found[0] if re.search(r"%", line) is None else found[-1]
        value = to_number(raw)
        if value is None:
            continue
        note = _check_summary(line.strip(), raw, value, all_columns, mean=mean)
        if note:
            notes.append(note)
    return [n for n in notes if n]


def _check_summary(line: str, raw: str, value: float, columns: List[Tuple[str, List[float]]], *, mean: bool) -> str:
    decimals = _decimals(raw)
    for name, values in columns:
        stat = sum(values) / len(values) if mean else sum(values)
        if abs(round(stat, decimals) - value) <= 0.5 * 10 ** (-decimals) + 1e-9:
            what = "среднему" if mean else "сумме"
            return f"✓ {raw.strip()} равно {what} столбца «{name}»."
    if not mean and not re.search(r"\b(?:итог\w*|всего|total)\b", line, re.IGNORECASE):
        return ""
    what = "среднему" if mean else "сумме"
    return f"⚠ «{line[:80]}»: {raw.strip()} не равно {what} ни одного столбца — проверьте распознавание."
