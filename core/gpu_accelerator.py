"""Вспомогательные runtime-утилиты для опционального ускорения через CUDA и FAISS.

The rest of the codebase should not need to know whether the current
machine has only CPU, CUDA, FAISS-CPU, or FAISS-GPU available. This
module centralizes those checks and provides a single adapter object.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.gpu_info import primary_gpu
from core.logging_utils import get_logger


logger = get_logger(__name__)

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    logger.info("FAISS is not installed; numpy fallback will be used")


class GPUAccelerator:
    """Singleton-обёртка над опциональными возможностями CUDA и FAISS."""

    _instance: Optional["GPUAccelerator"] = None

    def __new__(cls) -> "GPUAccelerator":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        """Один раз проверяет возможности хоста и кэширует доступные варианты ускорения."""
        gpu = primary_gpu()
        self.cuda_available = gpu is not None
        self.faiss_gpu_available = False
        self.gpu_name = "CPU"
        self.gpu_memory_gb = 0.0
        self.recommended_batch_size = 32
        self.stats = {
            "gpu_operations": 0,
            "cpu_operations": 0,
            "total_time_saved": 0.0,
            "embeddings_computed": 0,
            "searches_performed": 0,
        }

        if gpu is not None:
            self.gpu_name = gpu.name
            self.gpu_memory_gb = gpu.total_gb
            if self.gpu_memory_gb >= 16:
                self.recommended_batch_size = 256
            elif self.gpu_memory_gb >= 8:
                self.recommended_batch_size = 128
            elif self.gpu_memory_gb >= 4:
                self.recommended_batch_size = 64

        if FAISS_AVAILABLE and self.cuda_available and faiss is not None:
            try:
                self.faiss_gpu_available = bool(
                    hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0
                )
            except Exception:
                self.faiss_gpu_available = False

        self.current_mode = self._determine_best_mode()
        logger.info("Acceleration mode: %s", self.current_mode)

    def _determine_best_mode(self) -> str:
        """Выбирает наиболее полезный режим ускорения, доступный на текущей машине."""
        if self.faiss_gpu_available:
            return "FAISS-GPU"
        if self.cuda_available:
            return "CUDA"
        if FAISS_AVAILABLE:
            return "FAISS-CPU"
        return "CPU"

    def _estimate_speedup(self) -> float:
        """Возвращает грубую оценку ускорения, используемую только для диагностики."""
        if self.current_mode == "FAISS-GPU":
            return 50.0
        if self.current_mode == "CUDA":
            return 20.0
        if self.current_mode == "FAISS-CPU":
            return 5.0
        return 1.0

    def get_device(self) -> str:
        """Возвращает строку устройства torch, которую нужно использовать для выполнения модели."""
        return "cuda" if self.cuda_available else "cpu"

    def optimize_model(self, model: Any) -> Any:
        """Модели работают в llama-server, переносить на GPU нечего."""
        return model

    def create_faiss_index(self, dimension: int, use_gpu: Optional[bool] = None):
        """Создаёт индекс FAISS inner-product на CPU или GPU."""
        if not FAISS_AVAILABLE or faiss is None:
            raise ImportError("FAISS is not installed")

        index = faiss.IndexFlatIP(int(dimension))
        should_use_gpu = self.faiss_gpu_available if use_gpu is None else bool(use_gpu)
        if not should_use_gpu or not self.faiss_gpu_available:
            return index

        try:
            resources = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(resources, 0, index)
        except Exception as exc:
            logger.warning("Failed to create FAISS GPU index, using CPU: %s", exc)
            return index

    def compute_embeddings_batch(self, model: Any, texts: list, batch_size: Optional[int] = None) -> np.ndarray:
        """Кодирует список текстов пакетами и возвращает матрицу float32."""
        batch_size = batch_size or self.recommended_batch_size
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            embeddings.append(np.asarray(encoded, dtype=np.float32))

        if not embeddings:
            return np.empty((0, 0), dtype=np.float32)

        result = np.vstack(embeddings)
        self.stats["embeddings_computed"] += len(texts)
        if self.cuda_available:
            self.stats["gpu_operations"] += 1
        else:
            self.stats["cpu_operations"] += 1
        return result

    def search_faiss(self, index, query_embeddings: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Нормализует векторы запроса и запускает поиск ближайших соседей через FAISS."""
        if not FAISS_AVAILABLE or faiss is None:
            raise ImportError("FAISS is not installed")

        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        queries = queries.copy()

        start_time = time.time()
        faiss.normalize_L2(queries)
        distances, indices = index.search(queries, int(top_k))
        elapsed_ms = (time.time() - start_time) * 1000

        self.stats["searches_performed"] += 1
        if self.cuda_available:
            self.stats["gpu_operations"] += 1
        else:
            self.stats["cpu_operations"] += 1
        if elapsed_ms < 1.0:
            self.stats["total_time_saved"] += 10.0

        return distances, indices

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает машинно-читаемый снимок статистики использования ускорителей."""
        stats = dict(self.stats)
        stats.update(
            {
                "current_mode": self.current_mode,
                "cuda_available": self.cuda_available,
                "faiss_gpu_available": self.faiss_gpu_available,
                "gpu_name": self.gpu_name,
                "gpu_memory_gb": self.gpu_memory_gb,
                "estimated_speedup": self._estimate_speedup(),
            }
        )
        return stats

    def print_stats(self) -> None:
        """Пишет в лог короткую сводку о текущем состоянии ускорения."""
        stats = self.get_stats()
        logger.info(
            "Acceleration stats: mode=%s searches=%s embeddings=%s",
            stats["current_mode"],
            stats["searches_performed"],
            stats["embeddings_computed"],
        )


_gpu_accelerator: Optional[GPUAccelerator] = None


def get_gpu_accelerator() -> GPUAccelerator:
    """Возвращает глобальный singleton ускорителя для текущего процесса."""
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator()
    return _gpu_accelerator
