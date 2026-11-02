"""Базовые типы режима чата."""

from enum import Enum


class ChatMode(Enum):
    """Режимы взаимодействия, доступные в интерфейсе."""

    RAG = "rag"
    CHAT = "chat"
    EXTRACT = "extract"
