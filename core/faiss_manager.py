"""Опциональное управление индексом FAISS для семантического поиска по чанкам.

When FAISS is available the project keeps an approximate similarity
index alongside the raw embeddings. The rest of the search code can then
use one manager regardless of whether FAISS runs on CPU, GPU, or is not
installed at all.
"""

from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.gpu_accelerator import FAISS_AVAILABLE, get_gpu_accelerator
from core.logging_utils import get_logger
from core.paths import ensure_parent, resolve_project_path


logger = get_logger(__name__)

try:
    import faiss
except ImportError:
    faiss = None


class FaissManager:
    """Хранит индекс FAISS, сопоставление метаданных и вспомогательные методы сохранения."""

    def __init__(
        self,
        dimension: int = 384,
        index_path: str = "faiss_index.bin",
        meta_path: str = "faiss_meta.pkl",
        use_gpu: bool = True,
        namespace: str = "",
    ) -> None:
        self.dimension = int(dimension)
        self.expected_dimension = int(dimension)
        self.namespace = str(namespace or "")
        self.index_namespace = ""
        self.index_path = resolve_project_path(index_path)
        self.meta_path = resolve_project_path(meta_path)
        self.use_gpu = bool(use_gpu)

        self.accelerator = get_gpu_accelerator()
        self.index = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_to_idx: Dict[int, int] = {}
        self.idx_to_chunk: Dict[int, int] = {}
        self.last_modified = 0.0
        self.lock = threading.RLock()
        self.stats = {
            "build_count": 0,
            "search_count": 0,
            "add_count": 0,
            "total_vectors": 0,
            "last_build_time": 0.0,
        }

        self.load()

    def is_empty(self) -> bool:
        """Возвращает ``True``, если в индексе нет векторов для поиска."""
        try:
            return self.index is None or getattr(self.index, "ntotal", 0) == 0
        except Exception:
            return True

    def build_index(self, embeddings: np.ndarray, chunk_ids: List[int]) -> bool:
        """Полностью перестраивает индекс FAISS по текущим эмбеддингам чанков."""
        if not FAISS_AVAILABLE or faiss is None:
            logger.info("FAISS is unavailable; using numpy fallback")
            return False

        if not chunk_ids:
            logger.warning("FAISS build skipped because there are no chunk ids")
            return False

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.size == 0:
            logger.warning("FAISS build skipped because embeddings are empty")
            return False
        if len(vectors) != len(chunk_ids):
            logger.warning(
                "FAISS build skipped because embeddings and chunk ids have different sizes: %s != %s",
                len(vectors),
                len(chunk_ids),
            )
            return False

        actual_dimension = int(vectors.shape[1])
        if actual_dimension != self.dimension:
            logger.warning(
                "FAISS dimension mismatch detected; using actual embedding dimension %s instead of configured %s",
                actual_dimension,
                self.dimension,
            )
            self.dimension = actual_dimension

        with self.lock:
            try:
                start_time = time.time()
                self.embeddings = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0).copy()
                self.chunk_to_idx = {int(chunk_id): idx for idx, chunk_id in enumerate(chunk_ids)}
                self.idx_to_chunk = {idx: int(chunk_id) for idx, chunk_id in enumerate(chunk_ids)}

                self.index = self.accelerator.create_faiss_index(self.dimension, use_gpu=self.use_gpu)
                normalized = self.embeddings.copy()
                faiss.normalize_L2(normalized)
                self.index.add(normalized)
                self.index_namespace = self.namespace

                self.stats["build_count"] += 1
                self.stats["total_vectors"] = len(chunk_ids)
                self.stats["last_build_time"] = time.time() - start_time
                self.last_modified = time.time()
                self.save()
                logger.info("FAISS index built: %s vectors", len(chunk_ids))
                return True
            except Exception as exc:
                logger.warning("Failed to build FAISS index: %s", exc)
                self.index = None
                return False

    def add_embeddings(self, embeddings: np.ndarray, chunk_ids: List[int]) -> bool:
        """Добавляет новые векторы в существующий индекс без полной перестройки."""
        if self.index is None:
            return self.build_index(embeddings, chunk_ids)

        if not FAISS_AVAILABLE or faiss is None:
            return False

        with self.lock:
            try:
                new_embeddings = np.asarray(embeddings, dtype=np.float32)
                start_idx = len(self.embeddings) if self.embeddings is not None else 0
                self.embeddings = (
                    new_embeddings.copy()
                    if self.embeddings is None
                    else np.vstack([self.embeddings, new_embeddings])
                )

                for offset, chunk_id in enumerate(chunk_ids):
                    idx = start_idx + offset
                    self.chunk_to_idx[int(chunk_id)] = idx
                    self.idx_to_chunk[idx] = int(chunk_id)

                normalized = new_embeddings.copy()
                faiss.normalize_L2(normalized)
                self.index.add(normalized)

                self.stats["add_count"] += len(chunk_ids)
                self.stats["total_vectors"] = len(self.embeddings)
                self.last_modified = time.time()
                self.save()
                return True
            except Exception as exc:
                logger.warning("Failed to append vectors to FAISS index: %s", exc)
                return False

    def remove_chunks(self, chunk_ids: List[int]) -> bool:
        """Удаляет привязки чанков и помечает сохранённые метаданные как изменённые.

        The current implementation updates the Python-side mapping only.
        A full rebuild remains the authoritative way to physically compact
        the FAISS index after many removals.
        """
        removed = 0
        with self.lock:
            for chunk_id in chunk_ids:
                idx = self.chunk_to_idx.pop(int(chunk_id), None)
                if idx is None:
                    continue
                self.idx_to_chunk.pop(idx, None)
                removed += 1

            if removed:
                self.last_modified = time.time()
                self.save()
            return removed > 0

    def search(self, query_embeddings: np.ndarray, top_k: int = 10) -> Tuple[List[int], List[float]]:
        """Выполняет поиск по индексу и переводит строковые id FAISS обратно в chunk id."""
        if self.index is None or getattr(self.index, "ntotal", 0) == 0:
            return [], []

        with self.lock:
            try:
                distances, indices = self.accelerator.search_faiss(self.index, query_embeddings, top_k)
                self.stats["search_count"] += 1

                batch_ids: List[List[int]] = []
                batch_scores: List[List[float]] = []

                for row_indices, row_scores in zip(indices, distances):
                    row_ids: List[int] = []
                    row_score_values: List[float] = []
                    for idx, score in zip(row_indices, row_scores):
                        mapped = self.idx_to_chunk.get(int(idx))
                        if mapped is None:
                            continue
                        row_ids.append(mapped)
                        row_score_values.append(float(score))
                    batch_ids.append(row_ids)
                    batch_scores.append(row_score_values)

                if np.asarray(query_embeddings).ndim == 1 or np.asarray(query_embeddings).shape[0] == 1:
                    return batch_ids[0] if batch_ids else [], batch_scores[0] if batch_scores else []
                return batch_ids, batch_scores
            except Exception as exc:
                logger.warning("FAISS search failed: %s", exc)
                return [], []

    def search_by_chunk_ids(self, chunk_ids: List[int], top_k: int = 10) -> Tuple[List[int], List[float]]:
        """Находит соседей для уже проиндексированных чанков."""
        if self.embeddings is None:
            return [], []

        query_embeddings = []
        source_ids = []
        for chunk_id in chunk_ids:
            idx = self.chunk_to_idx.get(int(chunk_id))
            if idx is None:
                continue
            query_embeddings.append(self.embeddings[idx])
            source_ids.append(int(chunk_id))

        if not query_embeddings:
            return [], []

        all_ids, all_scores = self.search(np.asarray(query_embeddings, dtype=np.float32), top_k + 1)
        filtered_ids = []
        filtered_scores = []
        for current_id, result_ids, result_scores in zip(source_ids, all_ids, all_scores):
            ids_row = []
            scores_row = []
            for found_id, found_score in zip(result_ids, result_scores):
                if found_id == current_id:
                    continue
                ids_row.append(found_id)
                scores_row.append(found_score)
                if len(ids_row) >= top_k:
                    break
            filtered_ids.append(ids_row)
            filtered_scores.append(scores_row)
        return filtered_ids, filtered_scores

    def save(self) -> bool:
        """Сохраняет индекс FAISS и вспомогательные метаданные на диск."""
        if self.index is None or not FAISS_AVAILABLE or faiss is None:
            return False

        try:
            ensure_parent(self.index_path)
            ensure_parent(self.meta_path)

            index_to_save = self.index
            if self.use_gpu and self.accelerator.faiss_gpu_available:
                try:
                    index_to_save = faiss.index_gpu_to_cpu(self.index)
                except Exception:
                    index_to_save = self.index

            faiss.write_index(index_to_save, str(self.index_path))
            meta = {
                "dimension": self.dimension,
                "namespace": self.index_namespace or self.namespace,
                "chunk_to_idx": self.chunk_to_idx,
                "idx_to_chunk": self.idx_to_chunk,
                "last_modified": self.last_modified,
                "stats": self.stats,
            }
            with self.meta_path.open("wb") as handle:
                pickle.dump(meta, handle)
            return True
        except Exception as exc:
            logger.warning("Failed to save FAISS index: %s", exc)
            return False

    def load(self) -> bool:
        """Загружает сохранённый индекс FAISS, если он совместим с текущим состоянием проекта."""
        if not self.index_path.exists() or not self.meta_path.exists():
            return False
        if not FAISS_AVAILABLE or faiss is None:
            return False

        try:
            with self.meta_path.open("rb") as handle:
                meta = pickle.load(handle)

            self.index = faiss.read_index(str(self.index_path))
            if self.use_gpu and self.accelerator.faiss_gpu_available:
                try:
                    resources = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(resources, 0, self.index)
                except Exception as exc:
                    logger.warning("Failed to move FAISS index to GPU: %s", exc)

            self.dimension = int(meta.get("dimension", self.dimension))
            self.index_namespace = str(meta.get("namespace") or "")
            self.chunk_to_idx = {int(k): int(v) for k, v in meta.get("chunk_to_idx", {}).items()}
            self.idx_to_chunk = {int(k): int(v) for k, v in meta.get("idx_to_chunk", {}).items()}
            self.last_modified = float(meta.get("last_modified", 0.0))
            self.stats.update(meta.get("stats", {}))
            self.embeddings = None
            logger.info("Loaded FAISS index from %s", self.index_path)
            return True
        except Exception as exc:
            logger.warning("Failed to load FAISS index, rebuilding it later: %s", exc)
            try:
                self.index_path.unlink(missing_ok=True)
                self.meta_path.unlink(missing_ok=True)
            except Exception:
                return False
            return False

    def needs_rebuild(
        self,
        total_chunks: int,
        force: bool = False,
        chunk_ids: Optional[List[int]] = None,
    ) -> bool:
        """Определяет, рассинхронизирован ли сохранённый индекс относительно текущих чанков."""
        if force:
            return True
        if self.index is None:
            return True
        expected_dimension = int(getattr(self, "expected_dimension", self.dimension))
        if int(self.dimension) != expected_dimension:
            return True
        namespace = str(getattr(self, "namespace", "") or "")
        if namespace and str(getattr(self, "index_namespace", "") or "") != namespace:
            return True
        if getattr(self.index, "ntotal", 0) != total_chunks:
            return True
        if len(self.chunk_to_idx) != total_chunks:
            return True
        if chunk_ids is not None:
            expected = [int(chunk_id) for chunk_id in chunk_ids]
            actual = [self.idx_to_chunk.get(idx) for idx in range(total_chunks)]
            if actual != expected:
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает диагностическую информацию о состоянии FAISS."""
        return {
            **self.stats,
            "index_exists": self.index is not None,
            "total_vectors": getattr(self.index, "ntotal", 0) if self.index is not None else 0,
            "mapped_chunks": len(self.chunk_to_idx),
            "dimension": self.dimension,
            "use_gpu": self.use_gpu,
            "last_modified": self.last_modified,
            "index_size_mb": self.index_path.stat().st_size / (1024**2) if self.index_path.exists() else 0.0,
        }

    def print_stats(self) -> None:
        """Пишет в лог короткую человекочитаемую сводку по состоянию FAISS."""
        stats = self.get_stats()
        logger.info(
            "FAISS stats: vectors=%s mapped_chunks=%s searches=%s",
            stats["total_vectors"],
            stats["mapped_chunks"],
            stats["search_count"],
        )
