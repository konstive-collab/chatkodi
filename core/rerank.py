"""
core/rerank.py

Dedup + Rerank инструменты для HybridSearch.

Задачи:
- убрать "почти одинаковые" чанки (дубли в индексе, копии текста, повторы заголовков)
- улучшить порядок кандидатов после первичного скоринга
- не ломать производительность (лимиты сравнения, работа на topN)
"""

from __future__ import annotations

import re
import math
import threading
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

from core.logging_utils import get_logger
from core.paths import resolve_project_path

logger = get_logger(__name__)


def _normalize_for_dedup(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("ё", "е")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _shingles(text: str, k: int) -> set:
    if not text:
        return set()
    words = text.split()
    if len(words) <= k:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i+k]) for i in range(0, len(words) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / max(1, union)


@dataclass
class RerankConfig:
    enabled: bool = True

    dedup_enabled: bool = True
    dedup_shingle_size: int = 6
    dedup_threshold: float = 0.92
    dedup_max_compare: int = 300

    rerank_topn: int = 220
    anchor_boost: float = 0.10
    date_boost: float = 0.12
    id_boost: float = 0.15
    entity_boost: float = 0.06

    prefer_diversity: bool = True
    diversity_doc_penalty: float = 0.10
    max_chunks_per_doc_in_top: int = 4

    learned_enabled: bool = False
    learned_model_path: str = ""
    learned_topn: int = 40
    learned_weight: float = 0.35


def load_rerank_config(cfg_dict: Dict[str, Any]) -> RerankConfig:
    d = cfg_dict or {}
    return RerankConfig(
        enabled=bool(d.get("enabled", True)),

        dedup_enabled=bool(d.get("dedup_enabled", True)),
        dedup_shingle_size=int(d.get("dedup_shingle_size", 6)),
        dedup_threshold=float(d.get("dedup_threshold", 0.92)),
        dedup_max_compare=int(d.get("dedup_max_compare", 300)),

        rerank_topn=int(d.get("rerank_topn", 220)),
        anchor_boost=float(d.get("anchor_boost", 0.10)),
        date_boost=float(d.get("date_boost", 0.12)),
        id_boost=float(d.get("id_boost", 0.15)),
        entity_boost=float(d.get("entity_boost", 0.06)),

        prefer_diversity=bool(d.get("prefer_diversity", True)),
        diversity_doc_penalty=float(d.get("diversity_doc_penalty", 0.10)),
        max_chunks_per_doc_in_top=int(d.get("max_chunks_per_doc_in_top", 4)),
        learned_enabled=bool(d.get("learned_enabled", False)),
        learned_model_path=str(d.get("learned_model_path", "") or ""),
        learned_topn=int(d.get("learned_topn", 40)),
        learned_weight=float(d.get("learned_weight", 0.35)),
    )


class CrossEncoderReranker:
    """Lazy, offline-only learned reranker for the shortlist."""

    def __init__(self, cfg: RerankConfig, model_factory=None) -> None:
        self.cfg = cfg
        self._model_factory = model_factory
        self._model = None
        self._load_attempted = False
        self._lock = threading.RLock()

    def _load(self):
        if self._load_attempted:
            return self._model
        with self._lock:
            if self._load_attempted:
                return self._model
            self._load_attempted = True
            if not self.cfg.learned_enabled or not self.cfg.learned_model_path:
                return None
            model_path = resolve_project_path(self.cfg.learned_model_path)
            if not model_path.exists():
                logger.warning("Learned reranker model not found: %s", model_path)
                return None
            try:
                if self._model_factory is not None:
                    self._model = self._model_factory(str(model_path))
                else:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(str(model_path), local_files_only=True)
            except Exception as exc:
                logger.warning("Learned reranker unavailable: %s", exc)
                self._model = None
            return self._model

    @staticmethod
    def _probability(value: float) -> float:
        if 0.0 <= value <= 1.0:
            return value
        if value >= 40:
            return 1.0
        if value <= -40:
            return 0.0
        return 1.0 / (1.0 + math.exp(-value))

    def rerank(
        self,
        query: str,
        scored_chunks: List[Tuple[Tuple[int, float], int]],
        chunk_texts_by_id: Dict[int, str],
    ) -> List[Tuple[Tuple[int, float], int]]:
        model = self._load()
        if model is None or not scored_chunks:
            return scored_chunks
        topn = max(1, min(int(self.cfg.learned_topn), len(scored_chunks)))
        head = scored_chunks[:topn]
        pairs = [(query, chunk_texts_by_id.get(int(cid), "")) for _, cid in head]
        try:
            predictions = list(model.predict(pairs, show_progress_bar=False))
        except TypeError:
            predictions = list(model.predict(pairs))
        if len(predictions) != len(head):
            raise RuntimeError("Learned reranker returned an unexpected score count")
        weight = max(0.0, min(1.0, float(self.cfg.learned_weight)))
        rescored = []
        for ((date_bucket, original_score), cid), prediction in zip(head, predictions):
            if hasattr(prediction, "item"):
                prediction = prediction.item()
            learned_score = self._probability(float(prediction))
            blended = (1.0 - weight) * float(original_score) + weight * learned_score
            rescored.append(((int(date_bucket), max(0.0, min(1.0, blended))), int(cid)))
        rescored.sort(reverse=True, key=lambda item: item[0])
        return rescored + scored_chunks[topn:]


def dedup_scored_chunks(
    scored_chunks: List[Tuple[Tuple[int, float], int]],
    chunk_texts_by_id: Dict[int, str],
    cfg: RerankConfig
) -> List[Tuple[Tuple[int, float], int]]:
    """
    Убирает почти-дубликаты чанков среди top-K (ограничение по cfg.dedup_max_compare)

    scored_chunks: [((date_bucket, score), chunk_id), ...] уже отсортированы по убыванию.
    """
    if not cfg.dedup_enabled:
        return scored_chunks

    max_cmp = max(50, int(cfg.dedup_max_compare))
    k = max(2, int(cfg.dedup_shingle_size))
    thr = float(cfg.dedup_threshold)

    keep: List[Tuple[Tuple[int, float], int]] = []
    kept_shingles: List[set] = []

    # ограничиваем область дедупа сверху
    head = scored_chunks[:max_cmp]
    tail = scored_chunks[max_cmp:]

    for item in head:
        _, cid = item
        txt = chunk_texts_by_id.get(int(cid), "")
        norm = _normalize_for_dedup(txt)
        sh = _shingles(norm, k)

        is_dup = False
        for prev_sh in kept_shingles:
            sim = jaccard(sh, prev_sh)
            if sim >= thr:
                is_dup = True
                break

        if not is_dup:
            keep.append(item)
            kept_shingles.append(sh)

    # хвост без дедупа
    keep.extend(tail)
    return keep


def rerank_scored_chunks(
    scored_chunks: List[Tuple[Tuple[int, float], int]],
    chunk_meta_extra_by_id: Dict[int, Dict[str, Any]],
    chunk_meta_by_id: Dict[int, Dict[str, Any]],
    cfg: RerankConfig,
    has_date_constraint: bool
) -> List[Tuple[Tuple[int, float], int]]:
    """
    Лёгкий rerank:
    - усиливаем score за date_match/id_match/entity signals (из extra)
    - по желанию уменьшаем доминацию одного документа в top части
    """
    if not cfg.enabled:
        return scored_chunks

    topn = max(20, int(cfg.rerank_topn))
    head = scored_chunks[:topn]
    tail = scored_chunks[topn:]

    rescored: List[Tuple[Tuple[int, float], int]] = []
    doc_counts = defaultdict(int)

    for (date_bucket, score), cid in head:
        extra = chunk_meta_extra_by_id.get(int(cid), {})
        meta = chunk_meta_by_id.get(int(cid), {})
        doc_id = meta.get("doc_id", "unknown")

        new_score = float(score)

        # базовые усилители
        if has_date_constraint and extra.get("date_match"):
            new_score += float(cfg.date_boost)
        if extra.get("id_match"):
            new_score += float(cfg.id_boost)

        # matched_ids (если несколько совпадений — бонус)
        matched_ids = extra.get("matched_ids") or []
        if matched_ids:
            new_score += min(float(cfg.anchor_boost), 0.03 * len(matched_ids))

        # entities overlap (если хранится)
        if extra.get("entity_match_count", 0):
            new_score += min(float(cfg.entity_boost), 0.02 * int(extra.get("entity_match_count", 0)))

        # разнообразие: если один документ уже много раз в топе — наказываем
        if cfg.prefer_diversity:
            doc_counts[doc_id] += 1
            if doc_counts[doc_id] > int(cfg.max_chunks_per_doc_in_top):
                new_score -= float(cfg.diversity_doc_penalty)

        # clamp 0..1
        new_score = max(0.0, min(1.0, new_score))
        rescored.append(((int(date_bucket), float(new_score)), int(cid)))

    rescored.sort(reverse=True, key=lambda x: x[0])
    return rescored + tail
