"""Embedding model for semantic search: a GGUF model in a local llama-server.

``AIModelLoader().embedding_model`` keeps the SentenceTransformer-style
interface the search code uses (``encode``, ``get_sentence_embedding_dimension``)
and adds ``encode_query``: Qwen3-Embedding expects an instruction before the
query, while documents are embedded as they are.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from core.gpu_info import primary_gpu
from core.logging_utils import get_logger
from core.llm.model_select import find_embedding_model
from core.llm.server import LlamaServer, LlamaServerError

logger = get_logger(__name__)

QUERY_INSTRUCTION = (
    "Instruct: Given a question about company documents, "
    "retrieve the document passages that answer it\nQuery: "
)
# One input must fit into one physical batch (tokens); ~2.5 chars per token
# is the worst case for Russian, so longer texts are cut before sending.
UBATCH_TOKENS = 4096
MAX_INPUT_CHARS = 9000


class GGUFEmbeddingModel:
    """Client of an embedding llama-server with a SentenceTransformer-like API."""

    def __init__(self, model_path: Path, server: LlamaServer) -> None:
        self.model_path = Path(model_path)
        self.model_name = self.model_path.stem
        self.server = server
        self.max_seq_length = UBATCH_TOKENS
        self._dimension: Optional[int] = None

    # ---------------- public API ---------------- #

    def encode(
        self,
        sentences: Union[str, Sequence[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **_: Any,
    ) -> np.ndarray:
        single = isinstance(sentences, str)
        texts = [sentences] if single else [str(t or "") for t in sentences]
        if not texts:
            return np.zeros((0, self.get_sentence_embedding_dimension()), dtype=np.float32)
        batch = max(1, int(batch_size or 32))
        vectors: List[List[float]] = []
        for start in range(0, len(texts), batch):
            vectors.extend(self._embed_batch(texts[start:start + batch]))
        arr = np.asarray(vectors, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-12)
        if self._dimension is None and arr.size:
            self._dimension = int(arr.shape[1])
        return arr[0] if single else arr

    def encode_query(self, queries: Union[str, Sequence[str]], **kwargs: Any) -> np.ndarray:
        if isinstance(queries, str):
            return self.encode(QUERY_INSTRUCTION + queries, **kwargs)
        return self.encode([QUERY_INSTRUCTION + q for q in queries], **kwargs)

    def get_sentence_embedding_dimension(self) -> int:
        if self._dimension is None:
            self.encode(["тест"])
        return int(self._dimension or 0)

    def fingerprint_payload(self) -> Dict[str, Any]:
        """Identity of the vectors for the embedding cache (survives folder moves)."""
        size = self.model_path.stat().st_size if self.model_path.exists() else 0
        return {
            "backend": "llama.cpp",
            "model_name": self.model_path.name,
            "model_bytes": size,
            "dimension": self.get_sentence_embedding_dimension(),
            "pooling": "last",
            "query_instruction": hashlib.sha256(QUERY_INSTRUCTION.encode("utf-8")).hexdigest()[:12],
        }

    def close(self) -> None:
        self.server.stop()

    # ---------------- internals ---------------- #

    def _post_with_restart(self, prepared: List[str]) -> Dict[str, Any]:
        """POST /v1/embeddings; a crashed server is restarted once and retried."""
        try:
            return self.server.post_json("/v1/embeddings", {"input": prepared})
        except OSError:
            if self.server.is_running():
                raise
            logger.warning("llama-server[embeddings] is not running; restarting")
            self.server.process = None
            self.server.start()
            return self.server.post_json("/v1/embeddings", {"input": prepared})

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        prepared = [(t if t.strip() else " ")[:MAX_INPUT_CHARS] for t in texts]
        try:
            data = self._post_with_restart(prepared)
        except LlamaServerError as exc:
            if len(prepared) == 1 and "too large" in str(exc).lower() and len(prepared[0]) > 500:
                # Denser text than estimated: retry with the first half.
                return self._embed_batch([prepared[0][: len(prepared[0]) // 2]])
            if len(prepared) > 1:
                return [vec for text in prepared for vec in self._embed_batch([text])]
            raise
        items = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
        if len(items) != len(prepared):
            raise LlamaServerError(f"embeddings: expected {len(prepared)} vectors, got {len(items)}")
        return [item["embedding"] for item in items]


def _embedding_server_args() -> List[str]:
    args = [
        "--embedding", "--pooling", "last",
        "-c", str(UBATCH_TOKENS * 2), "-b", str(UBATCH_TOKENS), "-ub", str(UBATCH_TOKENS),
        "-np", "2",
    ]
    on_gpu = primary_gpu() is not None and os.environ.get("CHATKODI_EMBEDDINGS_DEVICE", "").lower() != "cpu"
    args += ["-ngl", "99", "-fa", "on"] if on_gpu else ["-ngl", "0"]
    return args


class AIModelLoader:
    """Ищет GGUF embedding-модель и запускает для неё llama-server."""

    def __init__(self) -> None:
        self.embedding_model: Optional[GGUFEmbeddingModel] = None
        self.models_loaded = False
        self.model_path: Optional[Path] = None

    def _find_model_path(self) -> Path:
        path = find_embedding_model()
        if path is None:
            raise FileNotFoundError(
                "Не найдена локальная embedding-модель (*.gguf в models_cache/embeddings). "
                "Подготовьте окружение командой: python download_models.py"
            )
        return path

    def load_models(self) -> bool:
        """Запускает сервер эмбеддингов и делает проверочный запрос."""
        model_path = self._find_model_path()
        server = LlamaServer("embeddings", model_path, _embedding_server_args())
        try:
            logger.info("Loading embedding model from %s", model_path)
            server.start()
            model = GGUFEmbeddingModel(model_path, server)
            dim = model.get_sentence_embedding_dimension()
            self.embedding_model = model
            self.model_path = model_path
            self.models_loaded = True
            logger.info("Embedding model loaded: %s dim=%s in %.1f s", model_path.name, dim, server.load_seconds)
            return True
        except Exception as exc:
            server.stop()
            logger.exception("Failed to load embedding model from %s", model_path)
            raise RuntimeError(f"Не удалось загрузить embedding-модель: {exc}") from exc
