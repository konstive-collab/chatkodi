"""Persistent, model-aware embedding cache."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.logging_utils import get_logger
from core.paths import ensure_parent, resolve_project_path


logger = get_logger(__name__)
CACHE_SCHEMA_VERSION = 2


def embedding_model_fingerprint(model: Any) -> str:
    """Return a stable namespace for vectors produced by ``model``."""
    if hasattr(model, "fingerprint_payload"):
        encoded = json.dumps(model.fingerprint_payload(), ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
        return f"embedding-v{CACHE_SCHEMA_VERSION}-{hashlib.sha256(encoded).hexdigest()[:20]}"
    payload: Dict[str, Any] = {
        "class": f"{type(model).__module__}.{type(model).__name__}",
    }
    try:
        payload["dimension"] = int(model.get_sentence_embedding_dimension())
    except Exception:
        payload["dimension"] = None
    try:
        payload["max_seq_length"] = int(getattr(model, "max_seq_length", 0) or 0)
    except Exception:
        payload["max_seq_length"] = 0
    try:
        first_module = model._first_module()  # type: ignore[attr-defined]
        config = getattr(getattr(first_module, "auto_model", None), "config", None)
        payload.update({
            "model_type": getattr(config, "model_type", ""),
            "hidden_size": getattr(config, "hidden_size", None),
            "architectures": list(getattr(config, "architectures", None) or []),
            # A moved portable folder should keep the same namespace.
            "model_name": Path(str(getattr(config, "_name_or_path", "") or "")).name,
        })
    except Exception:
        pass
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    return f"embedding-v{CACHE_SCHEMA_VERSION}-{hashlib.sha256(encoded).hexdigest()[:20]}"


class EmbeddingCache:
    """Store vectors by hashed text keys in an atomic NPZ/JSON pair."""

    def __init__(
        self,
        cache_path: str = "embedding_cache.npz",
        meta_path: str = "embedding_cache_meta.json",
        auto_save_interval: int = 64,
        min_save_interval: float = 3.0,
        namespace: str = "default",
    ) -> None:
        self.cache_path = resolve_project_path(cache_path)
        self.meta_path = resolve_project_path(meta_path)
        self.auto_save_interval = int(auto_save_interval)
        self.min_save_interval = float(min_save_interval)
        self.namespace = str(namespace or "default")

        self._embeddings: Dict[str, np.ndarray] = {}
        self._pending_save: Dict[str, np.ndarray] = {}
        self._dirty = False
        self._load_count = 0
        self._save_count = 0
        self._last_save_time = 0.0
        self._last_save_size = 0
        self._lock = threading.RLock()

        self._load()

    def _text_key(self, text: str) -> str:
        payload = (self.namespace + "\0" + str(text or "")).encode("utf-8", errors="surrogatepass")
        return hashlib.sha256(payload).hexdigest()

    def _load(self) -> None:
        if not self.cache_path.exists() or not self.meta_path.exists():
            return
        try:
            payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or int(payload.get("schema_version", 0) or 0) != CACHE_SCHEMA_VERSION:
                logger.info("Ignoring legacy embedding cache without a model namespace")
                self._discard_files()
                return
            stored_namespace = str(payload.get("namespace") or "")
            if stored_namespace != self.namespace:
                logger.info(
                    "Embedding model changed; rebuilding cache: stored=%s current=%s",
                    stored_namespace,
                    self.namespace,
                )
                self._discard_files()
                return
            entries = payload.get("entries") or {}
            if not isinstance(entries, dict):
                raise ValueError("embedding cache entries must be an object")

            loaded = 0
            skipped = 0
            with np.load(self.cache_path, allow_pickle=False) as npz_file:
                for key, idx in entries.items():
                    array_key = f"arr_{idx}"
                    if array_key not in npz_file:
                        skipped += 1
                        continue
                    self._embeddings[str(key)] = np.asarray(npz_file[array_key], dtype=np.float32)
                    loaded += 1
            self._load_count = loaded
            self._last_save_size = loaded
            if loaded:
                logger.info("Loaded %s embedding vectors from cache", loaded)
            if skipped:
                logger.warning("Skipped %s damaged embedding cache entries", skipped)
        except Exception as exc:
            logger.warning("Embedding cache is corrupted and will be rebuilt: %s", exc)
            self._embeddings.clear()
            self._pending_save.clear()
            self._discard_files()

    def _safe_remove(self, path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except Exception as exc:
            logger.warning("Failed to remove cache file %s: %s", path, exc)

    def _discard_files(self) -> None:
        self._safe_remove(self.cache_path)
        self._safe_remove(self.meta_path)

    def save(self, force: bool = False) -> bool:
        with self._lock:
            return self._save_locked(force=force)

    def _save_locked(self, force: bool = False) -> bool:
        if not self._dirty and not force:
            return False
        if self._pending_save:
            self._embeddings.update(self._pending_save)
            self._pending_save.clear()
        if not self._embeddings:
            return False

        current_time = time.time()
        if not force and current_time - self._last_save_time < self.min_save_interval:
            return False
        current_size = len(self._embeddings)
        if not force and current_size - self._last_save_size < self.auto_save_interval:
            return False

        cache_tmp = self.cache_path.with_name(self.cache_path.name + ".tmp")
        meta_tmp = self.meta_path.with_name(self.meta_path.name + ".tmp")
        try:
            ensure_parent(self.cache_path)
            ensure_parent(self.meta_path)
            keys = list(self._embeddings.keys())
            arrays = [np.asarray(self._embeddings[key], dtype=np.float32) for key in keys]
            metadata = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "namespace": self.namespace,
                "entries": {key: idx for idx, key in enumerate(keys)},
            }

            with cache_tmp.open("wb") as handle:
                np.savez_compressed(handle, *arrays)
                handle.flush()
                os.fsync(handle.fileno())
            meta_tmp.write_text(
                json.dumps(metadata, ensure_ascii=True, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(cache_tmp, self.cache_path)
            os.replace(meta_tmp, self.meta_path)

            self._dirty = False
            self._save_count += 1
            self._last_save_time = current_time
            self._last_save_size = current_size
            logger.info("Saved embedding cache: %s vectors", current_size)
            return True
        except Exception as exc:
            logger.warning("Failed to save embedding cache: %s", exc)
            self._safe_remove(cache_tmp)
            self._safe_remove(meta_tmp)
            return False

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._text_key(text)
        with self._lock:
            if key in self._embeddings:
                return self._embeddings[key]
            return self._pending_save.get(key)

    def set(self, text: str, embedding: np.ndarray, defer_save: bool = False) -> None:
        key = self._text_key(text)
        with self._lock:
            if key in self._embeddings or key in self._pending_save:
                return
            self._pending_save[key] = np.asarray(embedding, dtype=np.float32)
            self._dirty = True
            if not defer_save and len(self._pending_save) >= self.auto_save_interval:
                self._save_locked()

    def get_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        return [self.get(text) for text in texts]

    def set_batch(self, texts: List[str], embeddings: List[np.ndarray], defer_save: bool = False) -> None:
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding, defer_save=defer_save)

    def compute_missing(
        self,
        texts: List[str],
        compute_func: Callable[[List[str]], List[np.ndarray]],
        batch_size: int = 32,
        show_progress: bool = True,
        defer_saves: bool = True,
    ) -> List[np.ndarray]:
        results = self.get_batch(texts)
        missing_indices = [idx for idx, embedding in enumerate(results) if embedding is None]
        if not missing_indices:
            if show_progress:
                logger.info("Embedding cache hit for %s texts", len(texts))
            return [np.asarray(embedding, dtype=np.float32) for embedding in results if embedding is not None]

        missing_texts = [texts[idx] for idx in missing_indices]
        if show_progress:
            logger.info("Computing %s missing embeddings (cached=%s)", len(missing_texts), len(texts) - len(missing_texts))
        computed: List[np.ndarray] = []
        for start in range(0, len(missing_texts), batch_size):
            batch = missing_texts[start:start + batch_size]
            batch_embeddings = compute_func(batch)
            computed.extend(batch_embeddings)
            self.set_batch(batch, batch_embeddings, defer_save=defer_saves)
        for idx, embedding in zip(missing_indices, computed):
            results[idx] = np.asarray(embedding, dtype=np.float32)
        if defer_saves:
            self.save(force=True)
        return [np.asarray(embedding, dtype=np.float32) for embedding in results if embedding is not None]

    def get_stats(self) -> Dict[str, Any]:
        size_mb = self.cache_path.stat().st_size / (1024 * 1024) if self.cache_path.exists() else 0.0
        pending_bytes = sum(embedding.nbytes for embedding in self._pending_save.values())
        return {
            "cached_embeddings": len(self._embeddings),
            "pending_embeddings": len(self._pending_save),
            "total_embeddings": len(self._embeddings) + len(self._pending_save),
            "cache_size_mb": size_mb,
            "pending_size_mb": pending_bytes / (1024 * 1024),
            "load_count": self._load_count,
            "save_count": self._save_count,
            "last_save_time": self._last_save_time,
            "schema_version": CACHE_SCHEMA_VERSION,
            "namespace": self.namespace,
        }

    def clear_pending(self) -> None:
        with self._lock:
            self._pending_save.clear()

    def __del__(self) -> None:
        try:
            if not sys.is_finalizing() and self._dirty:
                self.save(force=True)
        except Exception:
            return
