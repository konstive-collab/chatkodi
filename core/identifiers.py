"""Exact identifiers (licence plates, document numbers, codes).

Plates are written in many ways: «А123ВС 77», «а 123 вс 77», «A123BC777» with
Latin look-alikes. Everything is reduced to one canonical upper-case Latin
form, so a question and a document match regardless of spelling.
"""

from __future__ import annotations

import re
from typing import List

_IDENTIFIER_TRANSLATION = str.maketrans(
    "АВЕКМНОРСТУХавекмнорстух",
    "ABEKMHOPCTYXABEKMHOPCTYX",
)

_VALUE = r"([A-Za-zА-ЯЁа-яё0-9][A-Za-zА-ЯЁа-яё0-9_/-]{0,23})"
_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        rf"(?:№|#)\s*{_VALUE}",
        rf"\b(?:госномер|государственный\s+номер|регистрационный\s+номер|номер)\s+(?:(?:автомобиля|машины|документа|техники|изделия|оборудования)\s+)?(?:№|#)?\s*{_VALUE}",
        rf"\b(?:автомобиль|машина|документ|техника|изделие|оборудование|vehicle|document|equipment)\s*(?:№|#|номер)?\s*{_VALUE}",
        r"\b[АВЕКМНОРСТУХABEKMHOPCTYX]\s*\d{3}\s*[АВЕКМНОРСТУХABEKMHOPCTYX]{2}\s*\d{2,3}\b",
        r"\b(?=[A-Za-zА-ЯЁа-яё0-9_/-]*[A-Za-zА-ЯЁа-яё])(?=[A-Za-zА-ЯЁа-яё0-9_/-]*\d)[A-Za-zА-ЯЁа-яё0-9]+(?:[-_/][A-Za-zА-ЯЁа-яё0-9]+)+\b",
    )
)
_PLATE = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9])[АВЕКМНОРСТУХABEKMHOPCTYX]\s*\d{3}\s*[АВЕКМНОРСТУХABEKMHOPCTYX]{2}\s*\d{2,3}(?![0-9])",
    flags=re.IGNORECASE,
)


def normalize_identifier(value: str) -> str:
    normalized = str(value or "").upper().translate(_IDENTIFIER_TRANSLATION)
    return re.sub(r"[^A-Z0-9]+", "", normalized)


def extract_normalized_identifiers(text: str) -> List[str]:
    """Extract exact identifiers without treating ordinary dates as IDs."""
    source = str(text or "")
    values: set[str] = set()
    for pattern in _PATTERNS:
        for match in pattern.finditer(source):
            candidate = match.group(1) if match.lastindex else match.group(0)
            normalized = normalize_identifier(candidate)
            if normalized and any(ch.isdigit() for ch in normalized):
                values.add(normalized)
    return sorted(values)


def find_plates(text: str) -> List[tuple[int, int, str]]:
    """Return ``(start, end, canonical)`` for every licence plate in ``text``."""
    return [
        (match.start(), match.end(), normalize_identifier(match.group(0)))
        for match in _PLATE.finditer(str(text or ""))
    ]


def is_plate(normalized: str) -> bool:
    return bool(re.fullmatch(r"[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}", normalized or ""))
