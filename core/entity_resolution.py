"""Universal structured entity extraction and matching.

The retrieval stack often deals with labels like ``изделие-51`` or
``станции 12`` where the *number* is the strongest signal and the alpha part
can appear in different grammatical forms.  This module normalizes such
mentions into a safe canonical representation without hard-coding concrete
business words.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Iterable, List, Optional, Sequence, Tuple

from core.lemmatizer import get_lemmatizer


_SEPARATORS = r"(?:\s*(?:[-_/]|№|#)\s*|\s+)"
_STRUCTURED_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(
        rf"(?<!\w)(?P<alpha>[A-Za-zА-Яа-яЁё]{{2,40}}(?:\s+[A-Za-zА-Яа-яЁё]{{2,40}}){{0,2}}){_SEPARATORS}(?P<number>\d{{1,8}})(?!\w)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\w)(?P<alpha>[A-Za-zА-Яа-яЁё]{2,40})(?P<number>\d{1,8})(?!\w)",
        re.IGNORECASE,
    ),
)

_MONTH_LIKE = {
    "январь", "января", "февраль", "февраля", "март", "марта", "апрель", "апреля",
    "май", "мая", "июнь", "июня", "июль", "июля", "август", "августа", "сентябрь",
    "сентября", "октябрь", "октября", "ноябрь", "ноября", "декабрь", "декабря",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
_STOP_ALPHA = {
    "дата", "дату", "дню", "день", "число", "год", "года", "месяц", "месяца", "page", "страница",
    "на", "в", "во", "по", "за", "с", "со", "к", "ко", "от", "до", "из", "при",
}


@dataclass(frozen=True)
class StructuredEntity:
    raw: str
    alpha_raw: str
    alpha_norm: str
    alpha_key: str
    number_raw: str
    number_norm: str

    @property
    def canonical(self) -> str:
        return f"{self.alpha_key}#{self.number_norm}"


@dataclass(frozen=True)
class EntityMention:
    entity: StructuredEntity
    start: int
    end: int


def _normalize_spaces(text: str) -> str:
    text = (text or "").strip().lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text)


def _fallback_russian_root(token: str) -> str:
    token = _normalize_spaces(token)
    if len(token) <= 4:
        return token
    endings = (
        "иями", "ями", "ами", "иями", "ов", "ев", "ями", "ами", "ием", "ием", "иях", "ях",
        "ах", "его", "ого", "ему", "ому", "ыми", "ими", "ый", "ий", "ой", "ая", "яя", "ое",
        "ее", "ую", "юю", "ом", "ем", "ам", "ям", "ах", "ях", "ия", "ья", "ие", "ье",
        "ию", "ью", "ия", "ья", "ий", "ый", "ой", "а", "я", "е", "и", "ы", "у", "ю",
        "о", "й", "ь", "м",
    )
    for end in endings:
        if len(token) - len(end) >= 4 and token.endswith(end):
            token = token[: -len(end)]
            break
    if len(token) > 6:
        return token[:6]
    return token


def _normalize_alpha_tokens(alpha_raw: str) -> Tuple[str, str]:
    alpha_raw = _normalize_spaces(alpha_raw)
    tokens = re.findall(r"[a-zа-я]+", alpha_raw, flags=re.IGNORECASE)
    if not tokens:
        return "", ""

    lemma = None
    try:
        lemma = get_lemmatizer()
    except Exception:
        lemma = None

    norm_tokens: List[str] = []
    key_tokens: List[str] = []
    for tok in tokens:
        if lemma and getattr(lemma, "available", False) and re.fullmatch(r"[а-я]+", tok):
            norm = lemma.lemmatize_word(tok)
        else:
            norm = tok
        norm = _normalize_spaces(norm)
        norm_tokens.append(norm)
        if re.fullmatch(r"[а-я]+", norm):
            key_tokens.append(_fallback_russian_root(norm))
        else:
            key_tokens.append(norm[:6] if len(norm) > 6 else norm)

    return " ".join(norm_tokens), " ".join(key_tokens)


def _normalize_number(number_raw: str) -> str:
    num = re.sub(r"\D+", "", number_raw or "")
    if not num:
        return ""
    return str(int(num)) if num.isdigit() else num


def _looks_like_date_or_noise(alpha_raw: str, number_raw: str) -> bool:
    alpha = _normalize_spaces(alpha_raw)
    if not alpha:
        return True
    alpha_words = [w for w in alpha.split(" ") if w]
    if all(w in _MONTH_LIKE or w in _STOP_ALPHA for w in alpha_words):
        return True
    if len(alpha_words) == 1 and alpha_words[0] in _STOP_ALPHA:
        return True
    if len(alpha_words) <= 2 and alpha_words[0] in _STOP_ALPHA:
        return True
    if number_raw.isdigit() and len(number_raw) == 4 and 1900 <= int(number_raw) <= 2100 and alpha in _MONTH_LIKE:
        return True
    return False


def build_structured_entity(alpha_raw: str, number_raw: str, raw: Optional[str] = None) -> Optional[StructuredEntity]:
    if _looks_like_date_or_noise(alpha_raw, number_raw):
        return None
    alpha_norm, alpha_key = _normalize_alpha_tokens(alpha_raw)
    number_norm = _normalize_number(number_raw)
    if not alpha_norm or not alpha_key or not number_norm:
        return None
    raw_value = raw if raw is not None else f"{alpha_raw}-{number_raw}"
    return StructuredEntity(
        raw=raw_value,
        alpha_raw=_normalize_spaces(alpha_raw),
        alpha_norm=alpha_norm,
        alpha_key=alpha_key,
        number_raw=str(number_raw),
        number_norm=number_norm,
    )


def extract_structured_entity_mentions(text: str) -> List[EntityMention]:
    text = text or ""
    if not text:
        return []

    mentions: List[EntityMention] = []
    seen = set()
    for pattern in _STRUCTURED_PATTERNS:
        for match in pattern.finditer(text):
            entity = build_structured_entity(
                match.group("alpha"),
                match.group("number"),
                raw=match.group(0),
            )
            if entity is None:
                continue
            key = (match.start(), match.end(), entity.canonical)
            if key in seen:
                continue
            seen.add(key)
            mentions.append(EntityMention(entity=entity, start=match.start(), end=match.end()))

    mentions.sort(key=lambda m: (m.start, m.end))
    return mentions


def extract_structured_entities(text: str) -> List[StructuredEntity]:
    entities: List[StructuredEntity] = []
    seen = set()
    for mention in extract_structured_entity_mentions(text):
        if mention.entity.canonical in seen:
            continue
        seen.add(mention.entity.canonical)
        entities.append(mention.entity)
    return entities


def alpha_similarity(left: StructuredEntity, right: StructuredEntity) -> float:
    if left.alpha_norm == right.alpha_norm:
        return 1.0
    if left.alpha_key == right.alpha_key:
        return 0.96

    left_keys = {tok for tok in left.alpha_key.split() if tok}
    right_keys = {tok for tok in right.alpha_key.split() if tok}
    inter = left_keys & right_keys
    if inter:
        coverage = len(inter) / max(1, min(len(left_keys), len(right_keys)))
        if coverage >= 1.0:
            return 0.95
        if coverage >= 0.5:
            return 0.88

    if left.alpha_key and right.alpha_key and (
        left.alpha_key.startswith(right.alpha_key) or right.alpha_key.startswith(left.alpha_key)
    ):
        if min(len(left.alpha_key), len(right.alpha_key)) >= 4:
            return 0.93
    ratio = SequenceMatcher(None, left.alpha_norm, right.alpha_norm).ratio()
    if ratio >= 0.90:
        return ratio
    key_ratio = SequenceMatcher(None, left.alpha_key, right.alpha_key).ratio()
    return max(ratio, key_ratio)


def entity_match_strength(left: StructuredEntity, right: StructuredEntity) -> float:
    if left.number_norm != right.number_norm:
        return 0.0
    sim = alpha_similarity(left, right)
    if sim >= 0.94:
        return 1.0
    if sim >= 0.84:
        return 0.85
    if sim >= 0.74:
        return 0.65
    return 0.0


def best_entity_matches(
    query_entities: Sequence[StructuredEntity],
    candidate_entities: Sequence[StructuredEntity],
) -> List[Tuple[StructuredEntity, StructuredEntity, float]]:
    matches: List[Tuple[StructuredEntity, StructuredEntity, float]] = []
    for q in query_entities:
        best: Optional[Tuple[StructuredEntity, float]] = None
        for cand in candidate_entities:
            strength = entity_match_strength(q, cand)
            if strength <= 0.0:
                continue
            if best is None or strength > best[1]:
                best = (cand, strength)
        if best is not None:
            matches.append((q, best[0], best[1]))
    return matches


def entity_overlap_score(query_text: str, candidate_text: str) -> Tuple[float, List[Tuple[StructuredEntity, StructuredEntity, float]]]:
    q_entities = extract_structured_entities(query_text)
    c_entities = extract_structured_entities(candidate_text)
    matches = best_entity_matches(q_entities, c_entities)
    if not matches:
        return 0.0, []
    total = sum(strength for _, _, strength in matches)
    score = min(1.0, 0.55 + 0.18 * total)
    return score, matches
