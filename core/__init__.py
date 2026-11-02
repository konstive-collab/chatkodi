"""Публичные экспорты пакета core с ленивой загрузкой.

Это позволяет импортировать лёгкие служебные модули без мгновенной загрузки
эмбеддингов, search engine и других тяжёлых зависимостей.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AIModelLoader",
    "ChatManager",
    "ChatMode",
    "DocumentProcessor",
    "HybridSearch",
]


def __getattr__(name: str) -> Any:
    if name in {"ChatManager", "ChatMode"}:
        from .chat_mode import ChatManager, ChatMode
        return {"ChatManager": ChatManager, "ChatMode": ChatMode}[name]
    if name == "DocumentProcessor":
        from .document_processor import DocumentProcessor
        return DocumentProcessor
    if name == "AIModelLoader":
        from .models import AIModelLoader
        return AIModelLoader
    if name == "HybridSearch":
        from .search import HybridSearch
        return HybridSearch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
