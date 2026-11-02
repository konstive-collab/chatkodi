"""Optional server adapters for TEI, Qdrant and OpenAI-compatible vLLM.

The adapters depend on injected clients, so importing ChatKodi Core never
imports server SDKs. Production deployments can pass official clients without
changing retrieval, evidence, or answer-building business logic.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


class TEIEmbeddingProvider:
    def __init__(self, client: Any):
        self.client = client

    def encode(self, texts: Sequence[str], **_kwargs: Any) -> np.ndarray:
        result = self.client.embed(inputs=list(texts))
        return np.asarray(result, dtype=np.float32)


class QdrantVectorStore:
    def __init__(
        self,
        client: Any,
        collection_name: str,
        *,
        namespace: str = "",
        expected_dimension: int | None = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.namespace = str(namespace or "")
        self.expected_dimension = int(expected_dimension) if expected_dimension else None

    @staticmethod
    def _item_digest(item_ids: Sequence[int]) -> str:
        payload = ",".join(str(int(item_id)) for item_id in item_ids).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _vector_dimension(info: Any) -> int | None:
        try:
            vectors = info.config.params.vectors
            if isinstance(vectors, Mapping):
                vectors = next(iter(vectors.values()))
            return int(vectors.size)
        except (AttributeError, StopIteration, TypeError, ValueError):
            return None

    def needs_rebuild(self, total_items: int, *, item_ids: Sequence[int]) -> bool:
        try:
            info = self.client.get_collection(self.collection_name)
        except Exception:
            return True
        count = int(getattr(info, "points_count", 0) or 0)
        if count != int(total_items):
            return True
        dimension = self._vector_dimension(info)
        if self.expected_dimension and dimension and dimension != self.expected_dimension:
            return True
        if not item_ids:
            return False
        try:
            page = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            points = page[0] if isinstance(page, tuple) else getattr(page, "points", page)
            first = list(points or [])[0]
            payload = getattr(first, "payload", None) or (first.get("payload", {}) if isinstance(first, dict) else {})
        except Exception:
            return True
        return (
            str(payload.get("chatkodi_namespace") or "") != self.namespace
            or str(payload.get("chatkodi_item_digest") or "") != self._item_digest(item_ids)
        )

    def rebuild(self, vectors: np.ndarray, item_ids: Sequence[int]) -> bool:
        try:
            from qdrant_client.http import models
        except ImportError as exc:
            raise RuntimeError("Install qdrant-client to rebuild a Qdrant collection") from exc
        matrix = np.asarray(vectors, dtype=np.float32)
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=int(matrix.shape[1]), distance=models.Distance.COSINE),
        )
        digest = self._item_digest(item_ids)
        payload = {
            "chatkodi_namespace": self.namespace,
            "chatkodi_item_digest": digest,
        }
        points = [
            models.PointStruct(id=int(item_id), vector=vector.tolist(), payload=payload)
            for item_id, vector in zip(item_ids, matrix)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return True

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[list[int], list[float]]:
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1).tolist()
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=int(top_k),
            with_payload=False,
        )
        points = list(getattr(response, "points", response) or [])
        return (
            [int(getattr(point, "id", point["id"] if isinstance(point, dict) else point)) for point in points],
            [float(getattr(point, "score", point.get("score", 0.0) if isinstance(point, dict) else 0.0)) for point in points],
        )


class VLLMProvider:
    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def is_available(self) -> bool:
        return self.client is not None and bool(self.model)

    def _token_stream(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            stream=True,
        )
        for chunk in stream:
            token = str(chunk.choices[0].delta.content or "")
            if token:
                yield token

    def generate_with_callback(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        callback: Any,
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **_kwargs: Any,
    ) -> str:
        pieces: list[str] = []
        for token in self._token_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            pieces.append(token)
            callback(token, False)
        callback("", True)
        return "".join(pieces)

    def generate_clean_answer(
        self,
        question: str,
        context: str = "",
        rag_mode: bool = False,
        streaming_callback: Any = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **_kwargs: Any,
    ):
        prompt = str(question or "")
        if rag_mode and context:
            prompt = (
                "Answer only from the supplied evidence. If it does not support the answer, say so.\n\n"
                f"EVIDENCE:\n{context}\n\nQUESTION:\n{question}"
            )
        messages = [{"role": "user", "content": prompt}]

        if streaming_callback is True:
            provider = self

            class StreamResult:
                def get_stream(self):
                    yield from provider._token_stream(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )

            return StreamResult()

        callback = streaming_callback if callable(streaming_callback) else (lambda _token, _final: None)
        return self.generate_with_callback(
            messages=messages,
            callback=callback,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )


__all__ = ["TEIEmbeddingProvider", "QdrantVectorStore", "VLLMProvider"]
