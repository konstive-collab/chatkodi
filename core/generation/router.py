from __future__ import annotations

import re

from .types import GenerationKind


_LEGAL_DOCUMENT = re.compile(
    r"\b(?:состав(?:ь|ить|ьте)|подготов(?:ь|ить|ьте)|сдела(?:й|ть|йте)|созда(?:й|ть|йте)|оформ(?:и|ить|ите))\b"
    r".{0,80}\b(?:рапорт|приказ|заявлени[ея]|ходатайств[оа]|служебн(?:ая|ую)\s+записк[ау]|"
    r"объяснительн(?:ая|ую)|характеристик[ау]|справк[ау])\b",
    re.IGNORECASE | re.DOTALL,
)
_LEGAL_DOCUMENT_REVERSED = re.compile(
    r"\b(?:рапорт|приказ|заявлени[ея]|ходатайств[оа]|служебн(?:ая|ую)\s+записк[ау]|"
    r"объяснительн(?:ая|ую)|характеристик[ау]|справк[ау])\b.{0,80}"
    r"\b(?:состав(?:ь|ить|ьте)|подготов(?:ь|ить|ьте)|сдела(?:й|ть|йте)|созда(?:й|ть|йте)|оформ(?:и|ить|ите))\b",
    re.IGNORECASE | re.DOTALL,
)
_TRAINING_PLAN = re.compile(
    r"\b(?:состав(?:ь|ить|ьте)|подготов(?:ь|ить|ьте)|сдела(?:й|ть|йте)|созда(?:й|ть|йте)|разработа(?:й|ть|йте))\b"
    r".{0,100}\b(?:план|программ[ау]|расписани[ея])\b.{0,80}"
    r"\b(?:трениров\w*|физическ\w*\s+подготов\w*|заняти\w*)\b",
    re.IGNORECASE | re.DOTALL,
)
_TRAINING_PLAN_REVERSED = re.compile(
    r"\b(?:план|программ[ау]|расписани[ея])\b.{0,80}\b(?:трениров\w*|физическ\w*\s+подготов\w*)\b"
    r".{0,80}\b(?:состав(?:ь|ить|ьте)|подготов(?:ь|ить|ьте)|сдела(?:й|ть|йте)|созда(?:й|ть|йте)|разработа(?:й|ть|йте))\b",
    re.IGNORECASE | re.DOTALL,
)
_TRAINING_WANTED = re.compile(
    r"\b(?:мне\s+нуж(?:ен|на)|хочу|требуется)\b.{0,80}\b(?:план|программ[ау]|расписани[ея])\b"
    r".{0,80}\b(?:трениров\w*|физическ\w*\s+подготов\w*)\b",
    re.IGNORECASE | re.DOTALL,
)
_EXERCISE = re.compile(
    r"\b(?:придум(?:ай|ать|айте)|предлож(?:и|ить|ите)|подбер(?:и|ите|ать)|замен(?:и|ить|ите)|добав(?:ь|ить|ьте))\b"
    r".{0,100}\b(?:упражнен\w*|трениров\w*|плавани\w*|бег\w*|подтягив\w*)\b",
    re.IGNORECASE | re.DOTALL,
)


def detect_generation_kind(question: str) -> GenerationKind:
    """Detect only explicit creation requests; ordinary document Q&A stays unchanged."""
    text = " ".join((question or "").strip().split())
    if not text:
        return GenerationKind.NONE
    if _LEGAL_DOCUMENT.search(text) or _LEGAL_DOCUMENT_REVERSED.search(text):
        return GenerationKind.LEGAL_DOCUMENT
    if _TRAINING_PLAN.search(text) or _TRAINING_PLAN_REVERSED.search(text) or _TRAINING_WANTED.search(text):
        return GenerationKind.TRAINING_PLAN
    if _EXERCISE.search(text):
        return GenerationKind.EXERCISE_IDEATION
    return GenerationKind.NONE
