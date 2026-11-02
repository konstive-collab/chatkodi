"""Пользовательские предпочтения поиска и объяснимости ответа.

Модуль намеренно небольшой: он хранит только мягкие настройки, которые
интерфейс может передать ядру без ломки RAG-пайплайна. Настройки не должны
подменять детерминированную логику поиска; они лишь регулируют ширину отбора
кандидатов и человекочитаемые подсказки в диагностике.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


QUERY_MODE_LABELS = {
    "auto": "Авто",
    "exact_fact": "Точный факт",
    "activity_report": "Подробный отчёт",
    "statistics": "Статистика",
    "document_search": "Поиск документов",
}

STRICTNESS_LABELS = {
    "strict": "Строго",
    "balanced": "Сбалансированно",
    "broad": "Шире",
}

SOURCE_SCOPE_LABELS = {
    "all": "Весь корпус",
    "selected": "Выбранные документы",
    "current_folder": "Текущая папка",
}


@dataclass
class SearchPreferences:
    """Мягкие настройки запроса, задаваемые пользователем из интерфейса."""

    query_mode: str = "auto"
    strictness: str = "balanced"
    source_scope: str = "all"

    @classmethod
    def from_config(cls, cfg: Dict[str, Any] | None) -> "SearchPreferences":
        ui = (cfg or {}).get("ui") or {}
        return cls(
            query_mode=normalize_query_mode(ui.get("query_mode", "auto")),
            strictness=normalize_strictness(ui.get("search_strictness", "balanced")),
            source_scope=normalize_source_scope(ui.get("source_scope", "all")),
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "query_mode": normalize_query_mode(self.query_mode),
            "search_strictness": normalize_strictness(self.strictness),
            "source_scope": normalize_source_scope(self.source_scope),
            "query_mode_label": QUERY_MODE_LABELS.get(normalize_query_mode(self.query_mode), "Авто"),
            "strictness_label": STRICTNESS_LABELS.get(normalize_strictness(self.strictness), "Сбалансированно"),
            "source_scope_label": SOURCE_SCOPE_LABELS.get(normalize_source_scope(self.source_scope), "Весь корпус"),
        }


def normalize_query_mode(value: Any) -> str:
    value = str(value or "auto").strip().lower()
    aliases = {
        "fact": "exact_fact",
        "exact": "exact_fact",
        "report": "activity_report",
        "activity": "activity_report",
        "stats": "statistics",
        "search": "document_search",
        "documents": "document_search",
    }
    value = aliases.get(value, value)
    return value if value in QUERY_MODE_LABELS else "auto"


def normalize_strictness(value: Any) -> str:
    value = str(value or "balanced").strip().lower()
    aliases = {"normal": "balanced", "medium": "balanced", "wide": "broad", "soft": "broad"}
    value = aliases.get(value, value)
    return value if value in STRICTNESS_LABELS else "balanced"


def normalize_source_scope(value: Any) -> str:
    value = str(value or "all").strip().lower()
    aliases = {"corpus": "all", "docs": "all", "current": "selected"}
    value = aliases.get(value, value)
    return value if value in SOURCE_SCOPE_LABELS else "all"
