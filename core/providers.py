"""Implementation-neutral provider contracts used by ChatKodi Core."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingProvider(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray: ...


@runtime_checkable
class VectorStore(Protocol):
    def needs_rebuild(self, total_items: int, *, item_ids: Sequence[int]) -> bool: ...
    def rebuild(self, vectors: np.ndarray, item_ids: Sequence[int]) -> bool: ...
    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[list[int], list[float]]: ...


@runtime_checkable
class LexicalRetriever(Protocol):
    def build(self, texts: Sequence[str], item_ids: Sequence[int]) -> None: ...
    def search(self, query: str, top_k: int) -> Mapping[int, float]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[tuple[int, float], int]],
        texts_by_id: Mapping[int, str],
    ) -> list[tuple[tuple[int, float], int]]: ...


@runtime_checkable
class LLMProvider(Protocol):
    def is_available(self) -> bool: ...
    def generate_clean_answer(
        self,
        question: str,
        context: str = "",
        rag_mode: bool = False,
        streaming_callback: Any = None,
        **kwargs: Any,
    ) -> Any: ...
    def generate_with_callback(self, *, messages: Sequence[Mapping[str, str]], callback: Any, **kwargs: Any) -> str: ...


__all__ = ["EmbeddingProvider", "VectorStore", "LexicalRetriever", "Reranker", "LLMProvider"]
