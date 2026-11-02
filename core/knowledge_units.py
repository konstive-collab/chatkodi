"""Универсальная модель знания для ChatKodi.

Этот модуль намеренно не привязан к конкретной предметной области.  В старых
итерациях часть логики называлась ``event``: это удобно для сводок, но плохо для
рецептов, договоров, лекций и технических инструкций.  Здесь описан общий язык:

* ``Mention`` — что буквально встретилось в тексте;
* ``Entity`` — нормализованная сущность, о которой говорит текст;
* ``Claim`` — утверждение/факт, который можно подтвердить источником;
* ``Qualifier`` — дата, количество, место, температура, условие и т.п.;
* ``CountUnit`` — явная единица счёта для статистики.

Пока слой используется как лёгкая надстройка над существующим ``EvidenceRecord``:
он помогает роутеру и статистике честно различать «упоминания», «факты»,
«цели», «документы», «шаги» и «эпизоды активности».  В следующих версиях сюда
можно перенести полноценное хранилище ``knowledge.db`` без переписывания GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re


def _norm(text: str) -> str:
    """Минимальная нормализация для правил выбора единицы счёта."""
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


@dataclass(frozen=True)
class SourceSpan:
    """Точный участок документа, из которого извлечён факт."""

    doc_id: str
    source: str
    chunk_id: Any = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    page: Optional[int] = None
    paragraph_index: Optional[int] = None


@dataclass(frozen=True)
class Mention:
    """Буквальное текстовое вхождение сущности или термина."""

    text: str
    normalized: str
    source_span: SourceSpan


@dataclass(frozen=True)
class Entity:
    """Нормализованная сущность: объект, ингредиент, болезнь, устройство и т.д."""

    name: str
    kind: str = "unknown"
    aliases: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Qualifier:
    """Уточнение факта: дата, количество, место, температура, условие."""

    kind: str
    value: str
    unit: str = ""


@dataclass(frozen=True)
class Claim:
    """Проверяемое утверждение, связанное с источником."""

    predicate: str
    subject: str = ""
    object: str = ""
    qualifiers: List[Qualifier] = field(default_factory=list)
    evidence: SourceSpan | None = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CountUnit:
    """Явная единица учёта для статистики.

    ``key`` хранится в диагностике и конфиге, ``label`` показывается человеку,
    ``short`` используется в компактных списках: «даты: 2025-06-10 — 2 эп.».
    """

    key: str
    label: str
    short: str
    description: str


COUNT_UNITS: Dict[str, CountUnit] = {
    "mention": CountUnit("mention", "упоминаний", "упом.", "буквальные вхождения в тексте"),
    "claim": CountUnit("claim", "подтверждённых фактов", "факт.", "дедуплицированные утверждения"),
    "activity_episode": CountUnit("activity_episode", "эпизодов активности", "эп.", "действия/состояния сущности за период"),
    "target": CountUnit("target", "уникальных целей/объектов", "целей", "нормализованные цели или объекты"),
    "source_document": CountUnit("source_document", "документов-источников", "док.", "уникальные документы"),
    "procedure_step": CountUnit("procedure_step", "шагов", "шаг.", "шаги инструкции или рецепта"),
    "ingredient": CountUnit("ingredient", "ингредиентов", "ингр.", "сущности-ингредиенты"),
    "method": CountUnit("method", "способов/средств", "спос.", "способы, инструменты, методы"),
    "location": CountUnit("location", "мест/направлений", "мест.", "места, направления, локации"),
    "date": CountUnit("date", "дат", "дат", "календарные даты"),
}


def get_count_unit(key: str | None) -> CountUnit:
    """Возвращает описание единицы счёта с безопасным fallback."""
    return COUNT_UNITS.get(str(key or "").strip(), COUNT_UNITS["claim"])


def resolve_count_unit(question: str, question_type: str, statistics_focus: str = "general") -> CountUnit:
    """Выбирает, что именно должна считать статистика.

    Правило специально простое и объяснимое: сначала уважаем явные слова
    пользователя («упоминания», «документы», «ингредиенты»), затем смотрим на
    тип запроса и статистический фокус.  Это защищает от старой ошибки, когда
    ChatKodi считал чанки, а в ответе звучало так, будто посчитаны реальные
    события.
    """
    q = _norm(question)
    qtype = str(question_type or "fact")
    focus = str(statistics_focus or "general")

    if any(x in q for x in ("упомин", "сколько раз встреч", "сколько раз указан")):
        return COUNT_UNITS["mention"]
    if any(x in q for x in ("документ", "источник", "файл")) and any(x in q for x in ("сколько", "количество", "найди")):
        return COUNT_UNITS["source_document"]
    if any(x in q for x in ("ингредиент", "продукт", "состав пирога", "для пирога")):
        return COUNT_UNITS["ingredient"]
    if any(x in q for x in ("шаг", "этап", "инструкц", "рецепт")) and any(x in q for x in ("сколько", "подроб", "перечисли", "по шагам")):
        return COUNT_UNITS["procedure_step"]
    if any(x in q for x in ("сколько дней", "сколько суток", "число дней", "количество дней", "how many days")):
        return COUNT_UNITS["date"]

    if qtype == "activity_report":
        return COUNT_UNITS["activity_episode"]
    if qtype == "target" or focus == "target":
        return COUNT_UNITS["target"]
    if focus == "method":
        return COUNT_UNITS["method"]
    if focus == "direction":
        return COUNT_UNITS["location"]
    if focus == "date":
        return COUNT_UNITS["date"]
    if qtype in {"frequency", "timeline", "range_summary", "summary", "comparison"}:
        return COUNT_UNITS["claim"]
    return COUNT_UNITS["claim"]
