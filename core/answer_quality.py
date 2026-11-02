"""Quality guard for LLM answers in document-grounded modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import re


@dataclass
class AnswerQuality:
    ok: bool
    reason: str = "ok"
    flags: List[str] = field(default_factory=list)


_BAD_NO_INFO = [
    "нет информации", "не найдено информации", "не содержит информации", "не удалось найти",
    "отсутствует информация", "данных недостаточно", "в документах нет",
]
_BAD_REFUSAL = [
    "не могу предоставить", "не могу ответить", "конфиденциаль", "персональн", "законодательств",
    "я не имею доступа", "не располагаю информацией",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def inspect_llm_answer(answer: str, *, has_structured: bool = False, structured_confidence: float = 0.0, min_chars: int = 20) -> AnswerQuality:
    text = normalize(answer)
    flags: List[str] = []
    if len(text) < int(min_chars):
        flags.append("too_short")
    if any(x in text for x in _BAD_REFUSAL):
        flags.append("refusal_like")
    if any(x in text for x in _BAD_NO_INFO):
        flags.append("no_info_like")
    if has_structured and structured_confidence >= 0.55 and ("no_info_like" in flags or "refusal_like" in flags):
        return AnswerQuality(False, "llm_contradicts_structured_evidence", flags)
    if has_structured and structured_confidence >= 0.82 and "too_short" in flags:
        return AnswerQuality(False, "llm_weaker_than_structured_evidence", flags)
    if "refusal_like" in flags:
        return AnswerQuality(False, "bad_refusal", flags)
    if flags:
        return AnswerQuality(False, "+".join(flags), flags)
    return AnswerQuality(True, "ok", flags)
