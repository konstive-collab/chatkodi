"""Гибридный пайплайн поиска, используемый в ChatKodi.

Этот модуль объединяет лексический поиск, семантический поиск,
эвристический rerank и опциональное ускорение через FAISS. Его результат
используется и в извлекающем поиске, и в документно-ориентированном RAG.
"""

import re
import time
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import replace
from datetime import datetime

import numpy as np
from rank_bm25 import BM25Okapi

from core.embedding_cache import EmbeddingCache, embedding_model_fingerprint
from core.document_processor import DocumentProcessor
from core.date_range_parser import MONTHS_RU, parse_date_range, parse_date_constraints
from core.logging_utils import get_logger
from core.diagnostics import QueryDiagnostics
from core.telemetry import TelemetryLogger

from core.entity_resolution import extract_structured_entities, entity_overlap_score
from core.event_store import EventStore
from core.system_profile import detect_system_profile


logger = get_logger(__name__)


def _faiss_installed() -> bool:
    try:
        from core.gpu_accelerator import FAISS_AVAILABLE

        return bool(FAISS_AVAILABLE)
    except Exception:
        return False

# -------------------- опциональные модули (безопасный импорт) -------------------- #

# конфиг (опционально)
try:
    from core.config import get_config  # type: ignore
except Exception:
    get_config = None  # резервный сценарий обрабатывается ниже

# сборщик контекста v2 (опционально)
try:
    from core.context_builder import load_context_v2_config, build_context_v2  # type: ignore
    _CTXV2_OK = True
except Exception:
    _CTXV2_OK = False
    load_context_v2_config = None
    build_context_v2 = None

# rerank + дедупликация (опционально)
try:
    from core.rerank import CrossEncoderReranker, load_rerank_config, dedup_scored_chunks, rerank_scored_chunks  # type: ignore
    _RERANK_OK = True
except Exception:
    _RERANK_OK = False
    load_rerank_config = None
    dedup_scored_chunks = None
    rerank_scored_chunks = None
    CrossEncoderReranker = None

# роутер интентов (опционально)
try:
    from core.intent_router import IntentRouter  # type: ignore
    _INTENT_OK = True
except Exception:
    _INTENT_OK = False
    IntentRouter = None

# лемматизатор (опционально)
try:
    from core.lemmatizer import get_lemmatizer  # type: ignore
    _LEMMATIZER = get_lemmatizer()
except Exception:
    _LEMMATIZER = None

# FAISS (опционально)
try:
    from core.faiss_manager import FaissManager  # type: ignore
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False
    FaissManager = None


# -------------------- стоп-слова -------------------- #

_STOPWORDS = {
    # русский
    "и", "в", "во", "на", "по", "к", "ко", "с", "со", "из", "за", "до", "от",
    "для", "про", "при", "без", "у", "о", "об", "обо", "а", "но", "или", "ли",
    "же", "что", "это", "этот", "эта", "эти", "тот", "та", "те", "там", "тут",
    "как", "когда", "где", "почему", "зачем", "над", "под", "между", "через",
    "бы", "был", "были", "есть", "нет", "да", "не", "ни", "лишь",
    "кто", "кого", "кому", "кем", "какой", "какая", "какое", "какие", "каком",
    "каких", "какого", "какую", "чей", "чья", "чье", "чьи", "чем", "сколько",
    # английский
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "from", "by",
    "with", "without", "about", "into", "over", "under", "between", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "it", "as", "not", "no", "yes",
    "who", "whom", "whose", "what", "which", "where", "when", "why", "how", "many", "much",
}

from core.identifiers import (  # noqa: E402  (kept under the historical names)
    _IDENTIFIER_TRANSLATION,
    extract_normalized_identifiers as _extract_normalized_identifiers,
    normalize_identifier as _normalize_identifier,
)


# -------------------- утилиты -------------------- #

def _safe_clamp01(x: float) -> float:
    """Ограничивает значение score диапазоном ``0..1`` включительно."""
    return max(0.0, min(1.0, float(x)))


def _minmax_norm(values: List[float]) -> List[float]:
    """Нормализует список оценок в диапазон ``0..1`` с корректной обработкой плоских диапазонов."""
    if not values:
        return []
    vmin = float(min(values))
    vmax = float(max(values))
    if vmax - vmin < 1e-12:
        return [0.0 for _ in values]
    return [(float(v) - vmin) / (vmax - vmin) for v in values]


def _passes_entity_signal_filter(
    meta: Dict[str, Any],
    doc_id: str,
    *,
    has_any_entity_hits: bool,
    query_entities: List[str],
    doc_has_date: Dict[str, bool],
    is_neighbor_of_support: bool,
) -> bool:
    """Return whether a chunk may survive the entity-signal filter.

    When at least one candidate contains a requested entity, unrelated chunks
    must not leak into the final result set. A dated chunk from the same
    document remains useful as supporting evidence, as does its immediate
    neighbour.
    """
    if not has_any_entity_hits or not query_entities:
        return True
    if int(meta.get("entity_match_count", 0) or 0) > 0:
        return True
    return bool(
        doc_has_date.get(doc_id)
        and (meta.get("date_match") or is_neighbor_of_support)
    )


# -------------------- HybridSearch -------------------- #

class HybridSearch:
    """Строит и использует гибридный лексико-семантический индекс проекта."""

    def __init__(
        self,
        embedding_model,
        doc_processor: DocumentProcessor,
        bm25_weight: float = 0.35,
        semantic_weight: float = 0.65,
        use_document_level: bool = True,
        faiss_index_path: str = "faiss_index.bin",
        faiss_meta_path: str = "faiss_meta.pkl",
        vector_store: Optional[Any] = None,
        lexical_retriever: Optional[Any] = None,
        reranker_provider: Optional[Any] = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.doc_processor = doc_processor
        self.use_document_level = use_document_level
        self.vector_store = vector_store
        self.lexical_retriever = lexical_retriever
        self.reranker_provider = reranker_provider

        self.bm25_weight = float(bm25_weight)
        self.semantic_weight = float(semantic_weight)

        self.embedding_cache = EmbeddingCache(namespace=embedding_model_fingerprint(embedding_model))

        # данные конфига (опционально)
        self.config = None
        self._cfg_data: Dict[str, Any] = {}
        if get_config is not None:
            try:
                self.config = get_config()
                if hasattr(self.config, "data") and isinstance(self.config.data, dict):
                    self._cfg_data = self.config.data
            except Exception:
                self._cfg_data = {}

        self._retr_cfg = self._cfg_data.get("retrieval", {}) or {}
        self.bm25_weight = float(self._retr_cfg.get("bm25_weight", self.bm25_weight))
        self.semantic_weight = float(
            self._retr_cfg.get("semantic_weight", self.semantic_weight)
        )
        weight_total = self.bm25_weight + self.semantic_weight
        if weight_total <= 0:
            logger.warning("Invalid retrieval weights; restoring 0.35/0.65 defaults")
            self.bm25_weight, self.semantic_weight = 0.35, 0.65
        elif abs(weight_total - 1.0) > 1e-9:
            self.bm25_weight /= weight_total
            self.semantic_weight /= weight_total
        self._ctx_cfg = self._cfg_data.get("context", {}) or {}
        try:
            self._system_profile = detect_system_profile(self._cfg_data)
        except Exception:
            self._system_profile = None
        self._ctx2_cfg = None
        if _CTXV2_OK and load_context_v2_config is not None:
            self._ctx2_cfg = load_context_v2_config(self._cfg_data.get("context_v2", {}) or {})
        self._rer_cfg = None
        self._learned_reranker = None
        if _RERANK_OK and load_rerank_config is not None:
            self._rer_cfg = load_rerank_config(self._cfg_data.get("rerank", {}) or {})
            if CrossEncoderReranker is not None:
                self._learned_reranker = CrossEncoderReranker(self._rer_cfg)

        self._diagnostics = QueryDiagnostics.from_config_data(self._cfg_data)
        self._telemetry = TelemetryLogger.from_config_data(self._cfg_data)

        self._intent_router = None
        if _INTENT_OK and IntentRouter is not None:
            try:
                self._intent_router = IntentRouter(self._cfg_data.get("intents", {}) or {})
            except Exception:
                self._intent_router = None

        # индексы
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_corpus: List[List[str]] = []
        self._chunk_id_to_pos: Dict[int, int] = {}
        self._pos_to_chunk_id: Dict[int, int] = {}
        self._query_reference_cache: Dict[str, Optional[datetime]] = {}
        self._query_date_parse_cache: Dict[str, Dict[str, Any]] = {}

        self._chunk_texts: List[str] = []
        self._chunk_meta: List[Dict[str, Any]] = []
        self._chunk_embeddings: Optional[np.ndarray] = None
        self._chunk_embeddings_norm: Optional[np.ndarray] = None
        self._doc_chunk_positions: Dict[str, List[int]] = defaultdict(list)
        self._identifier_to_chunk_ids: Dict[str, set[int]] = defaultdict(set)

        # кэши
        self._chunk_dates_cache: Dict[int, List[datetime]] = {}
        self._chunk_entities_cache: Dict[int, List[str]] = {}
        self._failed_signal_warnings: set[str] = set()

        # FAISS
        self.faiss: Optional["FaissManager"] = None
        self._faiss_enabled = _FAISS_OK
        self.event_store: Optional[EventStore] = None

        self.stats = {
            "total_searches": 0,
            "avg_search_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_candidates": 0.0,
        }

        self._initialize_indexes(faiss_index_path=faiss_index_path, faiss_meta_path=faiss_meta_path)

    # -------------------- текст -------------------- #

    def _normalize_text(self, text: str) -> str:
        """Нормализует текст для токен-ориентированных эвристик."""
        t = (text or "").lower().replace("ё", "е")
        t = re.sub(r"№\s+", "№", t)
        return t

    def _tokenize(self, text: str) -> List[str]:
        """Токенизирует текст для BM25 и лёгких эвристик перекрытия."""
        t = self._normalize_text(text)
        raw = re.findall(r"[a-zа-я0-9№]+", t, flags=re.IGNORECASE)
        raw = [w for w in raw if (len(w) > 2 or w.isdigit() or w.startswith("№"))]

        if _LEMMATIZER and getattr(_LEMMATIZER, "available", False):
            out = []
            for w in raw:
                if re.fullmatch(r"[а-я]+", w):
                    out.append(_LEMMATIZER.lemmatize_word(w))
                else:
                    out.append(w)
            return out

        return raw

    # -------------------- индексы -------------------- #

    def _guess_embedding_dim(self) -> int:
        """Определяет размерность векторов по embedding-модели."""
        if hasattr(self.embedding_model, "get_sentence_embedding_dimension"):
            try:
                return int(self.embedding_model.get_sentence_embedding_dimension())
            except Exception as exc:
                logger.debug("Embedding model did not expose its dimension: %s", exc)
        emb = self.embedding_model.encode(["test"], show_progress_bar=False)
        emb = np.asarray(emb)
        return int(emb.shape[-1])

    def _initialize_indexes(self, faiss_index_path: str, faiss_meta_path: str):
        """Подготавливает BM25, эмбеддинги и опциональные индексы FAISS."""
        if not self.doc_processor.documents:
            logger.info("No documents available for indexing")
            return

        if not self.doc_processor.text_chunks:
            self.doc_processor._create_chunks()

        logger.info("Initializing search indexes")

        self._chunk_texts = []
        self._chunk_meta = []
        self._chunk_id_to_pos.clear()
        self._pos_to_chunk_id.clear()
        self._chunk_dates_cache.clear()
        self._chunk_entities_cache.clear()
        self._doc_chunk_positions.clear()
        self._identifier_to_chunk_ids.clear()

        for pos, ch in enumerate(self.doc_processor.text_chunks):
            chunk_id = int(ch["chunk_id"])
            self._chunk_id_to_pos[chunk_id] = pos
            self._pos_to_chunk_id[pos] = chunk_id
            self._chunk_texts.append(ch["text"])
            self._chunk_meta.append(ch)
            self._doc_chunk_positions[str(ch.get("doc_id"))].append(pos)
            for identifier in _extract_normalized_identifiers(ch["text"]):
                self._identifier_to_chunk_ids[identifier].add(chunk_id)

        self.bm25_corpus = [self._tokenize(t) for t in self._chunk_texts]
        if self.bm25_corpus:
            self.bm25_index = BM25Okapi(self.bm25_corpus)
            logger.info("BM25 index ready: %s chunks", len(self.bm25_corpus))
        else:
            self.bm25_index = None
            logger.warning("BM25 corpus is empty")
        if self.lexical_retriever is not None:
            self.lexical_retriever.build(
                self._chunk_texts,
                [int(ch["chunk_id"]) for ch in self._chunk_meta],
            )

        self._build_embeddings_and_faiss(faiss_index_path, faiss_meta_path)
        self._initialize_event_store()

    def _build_embeddings_and_faiss(self, faiss_index_path: str, faiss_meta_path: str):
        """Загружает кэшированные эмбеддинги или вычисляет недостающие для всех чанков."""
        if not self._chunk_texts:
            self._chunk_embeddings = None
            return

        logger.info("Preparing chunk embeddings")

        def _compute(batch_texts: List[str]) -> np.ndarray:
            # На CPU/8GB большие batch-и sentence-transformers могут дать
            # резкий пик RAM. На GPU оставляем шире, на CPU режем безопаснее.
            try:
                prof = self._system_profile
                if prof is not None and not prof.has_cuda and prof.total_ram_gb and prof.total_ram_gb <= 8.5:
                    safe_batch = 16
                elif prof is not None and not prof.has_cuda:
                    safe_batch = 24
                else:
                    safe_batch = 64
            except Exception:
                safe_batch = 32
            embs = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=False,
                batch_size=min(safe_batch, len(batch_texts))
            )
            return np.asarray(embs, dtype=np.float32)

        cached = self.embedding_cache.get_batch(self._chunk_texts)
        missing_idx = [i for i, e in enumerate(cached) if e is None]

        if missing_idx:
            self.stats["cache_misses"] += len(missing_idx)
            batch_size = 64
            for start in range(0, len(missing_idx), batch_size):
                ids = missing_idx[start:start + batch_size]
                texts = [self._chunk_texts[i] for i in ids]
                embs = _compute(texts)
                for i, emb in zip(ids, embs):
                    self.embedding_cache.set(self._chunk_texts[i], emb, defer_save=True)
                    cached[i] = emb
            self.embedding_cache.save(force=True)
        else:
            self.stats["cache_hits"] += len(self._chunk_texts)

        self._chunk_embeddings = np.asarray(cached, dtype=np.float32)
        if self._chunk_embeddings.ndim == 1 and self._chunk_embeddings.size > 0:
            self._chunk_embeddings = self._chunk_embeddings.reshape(1, -1)
        if self._chunk_embeddings.size > 0:
            self._chunk_embeddings = np.nan_to_num(self._chunk_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                norms = np.linalg.norm(self._chunk_embeddings, axis=1, keepdims=True) + 1e-9
                self._chunk_embeddings_norm = (self._chunk_embeddings / norms).astype(np.float32, copy=False)
            except Exception:
                self._chunk_embeddings_norm = None
        else:
            self._chunk_embeddings_norm = None

        # FAISS (опционально)
        if self.vector_store is not None:
            chunk_ids = [int(ch["chunk_id"]) for ch in self._chunk_meta]
            try:
                if self.vector_store.needs_rebuild(len(chunk_ids), item_ids=chunk_ids):
                    self.vector_store.rebuild(self._chunk_embeddings, chunk_ids)
                logger.info("External vector store is ready: %s vectors", len(chunk_ids))
                return
            except Exception as exc:
                # Внешнее хранилище (Qdrant и т.п.) недоступно: поиск обязан
                # продолжить работу на локальных эмбеддингах, а не падать.
                logger.warning("External vector store unavailable, falling back to local index: %s", exc)
                self.vector_store = None
        if self._faiss_enabled and FaissManager is not None:
            try:
                if self._chunk_embeddings.size == 0 or self._chunk_embeddings.ndim != 2:
                    logger.warning("FAISS build skipped because chunk embeddings are empty or malformed")
                    self.faiss = None
                    return
                dim = int(self._chunk_embeddings.shape[1]) if self._chunk_embeddings.shape[1] else self._guess_embedding_dim()
                self.faiss = FaissManager(
                    dimension=dim,
                    index_path=faiss_index_path,
                    meta_path=faiss_meta_path,
                    use_gpu=True,
                    namespace=self.embedding_cache.namespace,
                )
                chunk_ids = [int(ch["chunk_id"]) for ch in self._chunk_meta]
                if self.faiss.needs_rebuild(
                    len(chunk_ids),
                    chunk_ids=chunk_ids,
                ):
                    ok = self.faiss.build_index(self._chunk_embeddings, chunk_ids)
                else:
                    ok = True
                    logger.info("Reusing compatible FAISS index: %s vectors", len(chunk_ids))
                if ok:
                    logger.info("FAISS index is ready")
                elif not _faiss_installed():
                    # Not an error: without the faiss package numpy search is
                    # the intended path (exact and fast enough for this corpus size).
                    logger.info("FAISS is not installed; vector search uses numpy")
                    self.faiss = None
                else:
                    logger.warning("FAISS index build failed, falling back to numpy")
                    self.faiss = None
            except Exception as e:
                logger.warning("FAISS error, falling back to numpy: %s", e)
                self.faiss = None

    def _initialize_event_store(self) -> None:
        """Индексирует события по чанкам в лёгкую SQLite-базу для structured-first режима."""
        try:
            ev_cfg = (self._cfg_data.get("event_store") or {})
            self.event_store = EventStore(
                max_sync_chunks=int(ev_cfg.get("max_sync_chunks", 0) or 0),
                max_sync_seconds=float(ev_cfg.get("max_sync_seconds", 0) or 0.0),
            )
            sync_meta = self.event_store.sync_chunks(self._chunk_meta)
            logger.info(
                "Event store ready: chunks=%s inserted=%s updated=%s unchanged=%s removed=%s",
                sync_meta.get('chunks_total', 0),
                sync_meta.get('inserted', 0),
                sync_meta.get('updated', 0),
                sync_meta.get('unchanged', 0),
                sync_meta.get('removed', 0),
            )
        except Exception as e:
            logger.warning("Event store initialization failed: %s", e)
            self.event_store = None

    def get_event_store_stats(self) -> Dict[str, Any]:
        if self.event_store is None:
            return {'available': False}
        try:
            stats = self.event_store.get_stats()
            stats['available'] = True
            return stats
        except Exception as e:
            return {'available': False, 'error': str(e)}

    # -------------------- якоря: идентификаторы -------------------- #

    def _extract_identifiers(self, text: str) -> List[str]:
        """Извлекает жёсткие идентификаторы вроде #11, №11 и латинских кодов."""
        return _extract_normalized_identifiers(text)

    def _exact_identifier_candidates(self, query: str) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for identifier in self._extract_identifiers(query):
            for chunk_id in self._identifier_to_chunk_ids.get(identifier, ()):
                out[int(chunk_id)] = 1.0
        return out

    def _bonus_by_identifiers(self, query: str, chunk_text: str) -> Tuple[float, bool, List[str]]:
        q_ids = self._extract_identifiers(query)
        if not q_ids:
            return 0.0, False, []
        c_ids = self._extract_identifiers(chunk_text)
        if not c_ids:
            return 0.0, False, []
        inter = sorted(set(q_ids) & set(c_ids))
        if not inter:
            return 0.0, False, []
        bonus = min(0.35, 0.12 + 0.06 * len(inter))
        return bonus, True, inter

    def _extract_object_like_terms(self, text: str) -> List[str]:
        """Выделяет якорные объектные обозначения вроде `изделие-51`, `робот №11`, `станция 3`."""
        t = self._normalize_text(text)
        out = set()

        # кириллица/латиница + номер через дефис/слеш/подчеркивание
        for m in re.finditer(r"\b([a-zа-яё]{2,24}[-_/][a-zа-яё0-9]{1,24})\b", t, flags=re.IGNORECASE):
            out.add(m.group(1))

        # еще один мягкий вариант: слово-число, чтобы изделие51 тоже не терялось
        for m in re.finditer(r"\b([a-zа-яё]{3,24}\d{1,8})\b", t, flags=re.IGNORECASE):
            out.add(m.group(1))

        for entity in extract_structured_entities(t):
            out.add(entity.raw)
            out.add(f"{entity.alpha_norm}-{entity.number_norm}")

        return sorted(out)

    def _extract_query_keywords(self, text: str) -> List[str]:
        toks = self._tokenize(text)
        out: List[str] = []
        seen = set()
        for tok in toks:
            if not tok:
                continue
            if tok in _STOPWORDS:
                continue
            if tok.isdigit() and len(tok) <= 4:
                # короткие числа почти всегда часть даты и только засоряют match'и шапки
                continue
            if len(tok) < 3 and not tok.isdigit() and not tok.startswith("№"):
                continue
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
        return out[:48]

    def _safe_extract_query_keywords(self, text: str) -> List[str]:
        try:
            return self._extract_query_keywords(text)
        except Exception as exc:
            logger.warning("Keyword extraction failed; continuing with other retrieval signals: %s", exc)
            return []

    def _safe_signal(self, name: str, callback, default):
        """Keep one failed heuristic from disabling all other retrieval signals."""
        try:
            return callback()
        except Exception as exc:
            warned = getattr(self, "_failed_signal_warnings", None)
            if warned is None:
                warned = set()
                self._failed_signal_warnings = warned
            if name not in warned:
                warned.add(name)
                logger.warning("%s retrieval signal failed; using a neutral score: %s", name, exc)
            return default

    def _object_term_bridge_score(self, query: str, chunk_text: str) -> Tuple[float, int]:
        q_objs = self._extract_object_like_terms(query)
        if not q_objs:
            return 0.0, 0

        c_norm = self._normalize_text(chunk_text)
        c_compact = re.sub(r"[-_/\s]+", "", c_norm)
        c_tokens = set(re.findall(r"[a-zа-яё0-9]+", c_norm, flags=re.IGNORECASE))

        hits = 0
        for obj in q_objs:
            obj_norm = self._normalize_text(obj)
            compact = re.sub(r"[-_/\s]+", "", obj_norm)
            matched = False

            variants = {obj_norm}
            variants.add(re.sub(r"[-_/]+", " ", obj_norm))
            variants.add(re.sub(r"\s+", " ", obj_norm))

            for variant in variants:
                if variant and variant in c_norm:
                    matched = True
                    break

            if not matched and compact and compact in c_compact:
                matched = True

            if not matched:
                parts = re.findall(r"[a-zа-яё0-9]+", obj_norm, flags=re.IGNORECASE)
                if parts:
                    alpha = {p for p in parts if not p.isdigit() and len(p) >= 3}
                    digits = {p for p in parts if p.isdigit()}
                    alpha_ok = not alpha or bool(alpha & c_tokens)
                    digit_ok = not digits or bool(digits & c_tokens)
                    if alpha_ok and digit_ok:
                        matched = True

            if matched:
                hits += 1

        if not hits:
            return 0.0, 0

        bonus = min(0.34, 0.16 + 0.08 * hits)
        return bonus, hits

    def _keyword_overlap_no_dates(
        self,
        query: str,
        chunk_text: str,
        query_keywords: Optional[List[str]] = None,
    ) -> Tuple[float, int]:
        q_kw = list(query_keywords) if query_keywords is not None else self._safe_extract_query_keywords(query)
        if not q_kw:
            return 0.0, 0

        c_tokens = set(self._safe_extract_query_keywords(chunk_text))
        inter = [tok for tok in q_kw if tok in c_tokens]
        if not inter:
            return 0.0, 0

        bonus = min(0.18, 0.05 + 0.025 * len(inter))
        return bonus, len(inter)

    def _fuzzy_keyword_score(
        self,
        query: str,
        chunk_text: str,
        query_keywords: Optional[List[str]] = None,
    ) -> Tuple[float, int]:
        """Small typo-tolerant signal; it never replaces BM25 or exact IDs."""
        q_tokens = list(query_keywords) if query_keywords is not None else self._safe_extract_query_keywords(query)
        q_tokens = [token for token in q_tokens if len(token) >= 5][:12]
        if not q_tokens:
            return 0.0, 0
        chunk_tokens = self._safe_extract_query_keywords(chunk_text)[:96]
        if not chunk_tokens:
            return 0.0, 0
        chunk_set = set(chunk_tokens)
        hits = 0
        for query_token in q_tokens:
            if query_token in chunk_set:
                continue
            comparable = [token for token in chunk_tokens if abs(len(token) - len(query_token)) <= 2]
            if any(SequenceMatcher(None, query_token, token).ratio() >= 0.84 for token in comparable):
                hits += 1
        return (min(0.12, 0.035 + 0.025 * hits), hits) if hits else (0.0, 0)

    def _score_answer_chunk_in_doc(self, query: str, chunk_text: str, chunk_id: int, *, doc_has_date: bool = False) -> Tuple[float, Dict[str, Any]]:
        query_keywords = self._safe_extract_query_keywords(query)
        id_bonus, id_match, matched_ids = self._safe_signal(
            "exact_identifier", lambda: self._bonus_by_identifiers(query, chunk_text), (0.0, False, [])
        )
        obj_bonus, object_term_hits = self._safe_signal(
            "object", lambda: self._object_term_bridge_score(query, chunk_text), (0.0, 0)
        )
        ent_bonus, ent_match_cnt = self._safe_signal(
            "entities", lambda: self._entities_overlap_score(query, chunk_text, chunk_id), (0.0, 0)
        )
        kw_bonus, kw_hits = self._safe_signal(
            "keywords", lambda: self._keyword_overlap_no_dates(query, chunk_text, query_keywords), (0.0, 0)
        )
        fuzzy_bonus, fuzzy_hits = self._safe_signal(
            "fuzzy", lambda: self._fuzzy_keyword_score(query, chunk_text, query_keywords), (0.0, 0)
        )
        dates = self._safe_signal("dates", lambda: self._get_chunk_dates(chunk_id, chunk_text), [])
        date_match = bool(dates)

        text_norm = self._normalize_text(chunk_text)
        non_empty_lines = [ln for ln in chunk_text.splitlines() if ln.strip()]
        looks_like_header = (
            bool(date_match)
            and len(non_empty_lines) <= 7
            and len(text_norm) <= 900
            and not id_match
            and object_term_hits == 0
            and ent_match_cnt == 0
            and kw_hits <= 1
        )

        score = 0.0
        score += id_bonus * 1.75
        score += obj_bonus * 1.55
        score += max(0.0, ent_bonus) * 1.20
        score += kw_bonus
        score += fuzzy_bonus

        if doc_has_date and (id_match or object_term_hits > 0 or ent_match_cnt > 0 or kw_hits >= 2):
            score += 0.18
        if looks_like_header:
            score -= 0.35
        if date_match and not (id_match or object_term_hits > 0 or ent_match_cnt > 0 or kw_hits >= 2):
            score -= 0.12

        return score, {
            "id_match": bool(id_match),
            "matched_ids": matched_ids,
            "entity_match_count": int(ent_match_cnt),
            "entity_match": bool(ent_match_cnt > 0),
            "object_term_hits": int(object_term_hits),
            "anchor_match": bool(id_match or object_term_hits > 0),
            "keyword_match_count": int(kw_hits),
            "fuzzy_match_count": int(fuzzy_hits),
            "date_match": bool(date_match),
            "doc_date_bridge": bool(doc_has_date and not date_match and (id_match or object_term_hits > 0 or ent_match_cnt > 0 or kw_hits >= 2)),
            "looks_like_header": bool(looks_like_header),
        }

    def _find_best_answer_chunk_for_doc(
        self,
        query: str,
        doc_id: str,
        *,
        doc_has_date: bool = False,
        exclude_chunk_ids: Optional[set] = None,
    ) -> Optional[Tuple[float, Dict[str, Any]]]:
        positions = self._doc_chunk_positions.get(str(doc_id), [])
        if not positions:
            return None

        exclude_chunk_ids = exclude_chunk_ids or set()
        best: Optional[Tuple[float, Dict[str, Any]]] = None

        for pos in positions:
            meta_src = self._chunk_meta[pos]
            cid = int(meta_src.get("chunk_id"))
            if cid in exclude_chunk_ids:
                continue
            text = self._chunk_texts[pos]
            score, extra = self._score_answer_chunk_in_doc(query, text, cid, doc_has_date=doc_has_date)
            if score < 0.26:
                continue
            meta = dict(meta_src)
            meta["text"] = text
            meta["score"] = float(score)
            meta.update(extra)
            meta["answer_rescue"] = True

            cand = (float(score), meta)
            if best is None:
                best = cand
                continue

            prev_meta = best[1]
            prev_key = (int(prev_meta.get("anchor_match", False)), int(prev_meta.get("entity_match_count", 0) > 0), int(prev_meta.get("keyword_match_count", 0)), best[0])
            cand_key = (int(meta.get("anchor_match", False)), int(meta.get("entity_match_count", 0) > 0), int(meta.get("keyword_match_count", 0)), float(score))
            if cand_key > prev_key:
                best = cand

        return best

    def _soft_token_overlap_score(self, query: str, chunk_text: str) -> Tuple[float, int]:
        """Мягкий score для объектных токенов, чтобы `изделие-51` и `изделие 51` матчились даже если это не один и тот же entity."""
        def soft_tokens(s: str) -> List[str]:
            s = self._normalize_text(s)
            toks = re.findall(r"[a-zа-яё0-9]+", s, flags=re.IGNORECASE)
            return [t for t in toks if len(t) >= 2 and t not in _STOPWORDS]

        q_obj = self._extract_object_like_terms(query)
        if not q_obj:
            return 0.0, 0

        q_tokens = set()
        for obj in q_obj:
            q_tokens.update(soft_tokens(obj))
        if not q_tokens:
            return 0.0, 0

        c_tokens = set(soft_tokens(chunk_text))
        inter = q_tokens & c_tokens

        # Нужен хотя бы один буквенный и один числовой токен, если в запросе есть числа
        has_digit_in_query = any(tok.isdigit() for tok in q_tokens)
        has_digit_match = any(tok.isdigit() for tok in inter)
        has_alpha_match = any(not tok.isdigit() for tok in inter)

        if has_digit_in_query:
            if has_digit_match and has_alpha_match:
                return 0.22, len(inter)
            if has_digit_match:
                return 0.08, len(inter)
            return 0.0, 0

        if inter:
            return min(0.16, 0.08 + 0.03 * len(inter)), len(inter)
        return 0.0, 0

    # -------------------- dates -------------------- #

    def _extract_dates_from_text(self, text: str, reference_date: Optional[datetime] = None) -> List[datetime]:
        t = self._normalize_text(text)
        out: List[datetime] = []

        def _add(y: int, mo: int, d: int):
            if 1 <= d <= 31 and 1 <= mo <= 12 and 1900 <= y <= 2100:
                try:
                    out.append(datetime(y, mo, d))
                except (TypeError, ValueError) as exc:
                    logger.debug("Ignoring invalid date components %s-%s-%s: %s", y, mo, d, exc)

        for m in re.findall(r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})\b", t):
            d, mo, y = int(m[0]), int(m[1]), int(m[2])
            if y < 100:
                y += 2000
            _add(y, mo, d)

        for m in re.findall(r"\b(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\b", t):
            y, mo, d = int(m[0]), int(m[1]), int(m[2])
            _add(y, mo, d)

        month_map = {
            'январь': 1, 'января': 1,
            'февраль': 2, 'февраля': 2,
            'март': 3, 'марта': 3,
            'апрель': 4, 'апреля': 4,
            'май': 5, 'мая': 5,
            'июнь': 6, 'июня': 6,
            'июль': 7, 'июля': 7,
            'август': 8, 'августа': 8,
            'сентябрь': 9, 'сентября': 9,
            'октябрь': 10, 'октября': 10,
            'ноябрь': 11, 'ноября': 11,
            'декабрь': 12, 'декабря': 12,
        }
        for m in re.findall(r"\b(\d{1,2})\s+(январь|января|февраль|февраля|март|марта|апрель|апреля|май|мая|июнь|июня|июль|июля|август|августа|сентябрь|сентября|октябрь|октября|ноябрь|ноября|декабрь|декабря)(?:\s+(\d{4}))?(?:\s*г(?:ода?)?)?\b", t):
            d = int(m[0])
            mo = month_map.get(m[1])
            y = int(m[2]) if m[2] else (reference_date.year if reference_date else datetime.now().year)
            if mo:
                _add(y, mo, d)

        if reference_date:
            _add(reference_date.year, reference_date.month, reference_date.day)
        return sorted({datetime(dt.year, dt.month, dt.day) for dt in out}, key=lambda x: x.toordinal())

    def _get_chunk_dates(self, chunk_id: int, chunk_text: str) -> List[datetime]:
        if chunk_id in self._chunk_dates_cache:
            return self._chunk_dates_cache[chunk_id]
        reference_date = None
        pos = self._chunk_id_to_pos.get(chunk_id)
        if pos is not None:
            raw = str((self._chunk_meta[pos] or {}).get("document_date") or "")
            try:
                reference_date = datetime.strptime(raw, "%Y-%m-%d")
            except Exception:
                reference_date = None
        dates = self._extract_dates_from_text(chunk_text, reference_date=reference_date)
        self._chunk_dates_cache[chunk_id] = dates
        return dates

    def infer_query_reference_date(self, query: str) -> Optional[datetime]:
        """Infer an omitted year from dated documents in the active corpus.

        Explicit years and relative expressions keep their normal calendar
        semantics. For an absolute day/month without a year, each corpus year
        is evaluated and the year covering the most documents is selected.
        """
        normalized = str(query or "").lower().replace("ё", "е")
        if normalized in self._query_reference_cache:
            return self._query_reference_cache[normalized]
        if re.search(r"\b(?:19|20)\d{2}\b", normalized):
            self._query_reference_cache[normalized] = None
            return None
        if not any(re.search(rf"\b{re.escape(month)}\b", normalized) for month in MONTHS_RU):
            self._query_reference_cache[normalized] = None
            return None

        dated_docs: Dict[str, datetime] = {}
        for meta in self._chunk_meta or []:
            raw = str((meta or {}).get("document_date") or "").strip()
            doc_id = str((meta or {}).get("doc_id") or raw)
            if not raw or doc_id in dated_docs:
                continue
            try:
                dated_docs[doc_id] = datetime.strptime(raw, "%Y-%m-%d")
            except Exception:
                continue
        if not dated_docs:
            self._query_reference_cache[normalized] = None
            return None

        years = sorted({dt.year for dt in dated_docs.values()})
        ranked: List[Tuple[int, int, int]] = []
        for year in years:
            parsed = parse_date_constraints(query, reference_date=datetime(year, 1, 1))
            start = parsed.get("start")
            end = parsed.get("end")
            dates = set(parsed.get("dates") or [])
            if parsed.get("mode") == "none":
                continue
            matches = 0
            for doc_date in dated_docs.values():
                day = datetime(doc_date.year, doc_date.month, doc_date.day)
                if dates and day in dates:
                    matches += 1
                elif not dates and (not start or day >= start) and (not end or day <= end):
                    matches += 1
            year_frequency = sum(1 for dt in dated_docs.values() if dt.year == year)
            ranked.append((matches, year_frequency, year))

        if not ranked:
            self._query_reference_cache[normalized] = None
            return None
        matches, _frequency, selected_year = max(ranked)
        if matches <= 0 and len(years) > 1:
            self._query_reference_cache[normalized] = None
            return None
        reference = datetime(selected_year, 1, 1)
        self._query_reference_cache[normalized] = reference
        logger.info(
            "Inferred omitted query year from corpus: year=%s matching_docs=%s corpus_years=%s",
            selected_year,
            matches,
            years,
        )
        return reference

    def _parse_query_date_constraints(self, query: str) -> Dict[str, Any]:
        cache_key = str(query or "").lower().replace("ё", "е")
        cached = self._query_date_parse_cache.get(cache_key)
        if cached is not None:
            return cached
        reference = self.infer_query_reference_date(query)
        parsed = parse_date_constraints(query, reference_date=reference)
        self._query_date_parse_cache[cache_key] = parsed
        return parsed

    def _date_range_score(self, query: str, chunk_text: str, chunk_id: int, date_strict: bool = False) -> Tuple[float, bool, int, float]:
        """
        Возвращает:
            (adjustment, date_match_flag, matched_requested_dates, coverage_ratio)
        """
        parsed = self._parse_query_date_constraints(query)
        mode = parsed.get("mode", "none")
        start = parsed.get("start")
        end = parsed.get("end")
        requested_dates = parsed.get("dates") or []

        if mode == "none" or (not start and not end and not requested_dates):
            return 0.0, False, 0, 0.0

        dates = self._get_chunk_dates(chunk_id, chunk_text)
        if not dates:
            return (-0.90 if date_strict else -0.70), False, 0, 0.0

        normalized_dates = sorted({datetime(dt.year, dt.month, dt.day) for dt in dates}, key=lambda x: x.toordinal())

        def in_range(dt: datetime) -> bool:
            if start and dt < start:
                return False
            if end and dt > end:
                return False
            return True

        if mode == "discrete_dates":
            req = {datetime(dt.year, dt.month, dt.day) for dt in requested_dates}
            got = {dt for dt in normalized_dates if dt in req}
            matched = len(got)
            coverage = matched / max(1, len(req))
            if matched > 0:
                return 0.45 + min(0.35, 0.35 * coverage), True, matched, coverage
            return (-0.95 if date_strict else -0.75), False, 0, 0.0

        if mode == "single":
            target = start or end
            matched = 1 if target and any(dt == target for dt in normalized_dates) else 0
            if matched:
                return 0.70, True, 1, 1.0
            return (-0.95 if date_strict else -0.75), False, 0, 0.0

        matches = [dt for dt in normalized_dates if in_range(dt)]
        matched = len(matches)
        if matched:
            coverage = 1.0
            if start and end:
                req_days = max(1, (end - start).days + 1)
                coverage = min(1.0, matched / req_days)
            return 0.42 + min(0.28, 0.28 * coverage), True, matched, coverage

        return (-0.90 if date_strict else -0.70), False, 0, 0.0

    # -------------------- универсальные сущности -------------------- #

    def _is_stopword(self, w: str) -> bool:
        return w in _STOPWORDS

    def _normalize_entity(self, s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        s = s.replace("ё", "е")
        return s.lower()

    def _extract_keyphrases(self, text: str, max_phrases: int = 12) -> List[str]:
        tl = self._normalize_text(text)
        words = re.findall(r"[a-zа-я0-9]+", tl, flags=re.IGNORECASE)
        words = [w for w in words if len(w) >= 3 and not self._is_stopword(w)]

        phrases = set()
        for n in (2, 3, 4, 5):
            for i in range(0, max(0, len(words) - n + 1)):
                ng = words[i:i + n]
                if sum(len(x) for x in ng) < 10:
                    continue
                phrases.add(" ".join(ng))

        ranked = sorted(phrases, key=lambda x: (len(x), x), reverse=True)
        return ranked[:max_phrases]

    def _extract_entities_universal(self, text: str) -> List[str]:
        if not text:
            return []

        entities = set()

        # строки в кавычках
        for m in re.finditer(r"[«\"\']([^»\"\']{2,80})[»\"\']", text):
            val = m.group(1).strip()
            if val:
                entities.add(self._normalize_entity(val))

        tl = self._normalize_text(text)

        # коды с разделителями
        for m in re.finditer(r"\b[a-zа-я]{1,10}[a-zа-я0-9]{0,10}[-_/#][a-zа-я0-9][a-zа-я0-9\-_/#]{1,30}\b", tl):
            entities.add(self._normalize_entity(m.group(0)))

        # буквенно-цифровые сущности с цифрами
        for m in re.finditer(r"\b[a-zа-я]{1,12}\d{1,12}[a-zа-я0-9]{0,8}\b", tl):
            entities.add(self._normalize_entity(m.group(0)))

        # слова через дефис
        for m in re.finditer(r"\b[a-zа-я]{2,30}-[a-zа-я0-9]{2,30}\b", tl):
            entities.add(self._normalize_entity(m.group(0)))

        # фразы с заглавных букв (2..5 слов)
        cap_phr = re.findall(
            r"\b(?:[A-ZА-ЯЁ][a-zа-яё]+)(?:\s+(?:[A-ZА-ЯЁ][a-zа-яё]+)){1,4}\b",
            text
        )
        for p in cap_phr:
            p = p.strip()
            if len(p) >= 4:
                entities.add(self._normalize_entity(p))

        # ключевые фразы
        for kp in self._extract_keyphrases(text, max_phrases=12):
            entities.add(self._normalize_entity(kp))

        cleaned = []
        for e in entities:
            e = e.strip(" -_/#")
            if len(e) < 3:
                continue
            if e in _STOPWORDS:
                continue
            cleaned.append(e)

        return sorted(set(cleaned))

    def _get_chunk_entities(self, chunk_id: int, chunk_text: str) -> List[str]:
        if chunk_id in self._chunk_entities_cache:
            return self._chunk_entities_cache[chunk_id]
        ents = self._extract_entities_universal(chunk_text)
        self._chunk_entities_cache[chunk_id] = ents
        return ents

    def _entities_overlap_score(self, query: str, chunk_text: str, chunk_id: int) -> Tuple[float, int]:
        struct_score, struct_matches = entity_overlap_score(query, chunk_text)
        if struct_matches:
            return max(0.18, min(0.34, 0.18 + 0.08 * len(struct_matches) + 0.06 * struct_score)), len(struct_matches)

        q_ents = self._extract_entities_universal(query)
        c_ents = self._get_chunk_entities(chunk_id, chunk_text)

        if not q_ents and not self._extract_object_like_terms(query):
            return 0.0, 0

        if not c_ents:
            soft_bonus, soft_cnt = self._soft_token_overlap_score(query, chunk_text)
            if soft_bonus > 0:
                return soft_bonus, soft_cnt
            return -0.04, 0

        qset = set(q_ents)
        cset = set(c_ents)
        inter = qset.intersection(cset)

        if inter:
            cnt = len(inter)
            return min(0.30, 0.14 + 0.06 * cnt), cnt

        def soft_norm(s: str) -> str:
            return re.sub(r"[-_/#\s]+", "", s)

        qsoft = [soft_norm(x) for x in qset]
        csoft = [soft_norm(x) for x in cset]

        for qs in qsoft:
            if len(qs) < 5:
                continue
            for cs in csoft:
                if len(cs) < 5:
                    continue
                if qs in cs or cs in qs:
                    return 0.12, 1

        soft_bonus, soft_cnt = self._soft_token_overlap_score(query, chunk_text)
        if soft_bonus > 0:
            return max(soft_bonus, 0.12), max(soft_cnt, 1)

        return 0.0, 0

    # -------------------- normalization -------------------- #

    def _normalize_scores_robust_p95(self, scores: np.ndarray) -> np.ndarray:
        """
        Твой стабильный нормализатор BM25: делим на p95.
        """
        if scores.size == 0:
            return scores
        s = scores.astype(np.float32)
        s = np.maximum(s, 0.0)
        if not np.any(s):
            return np.zeros_like(s)
        p95 = float(np.percentile(s, 95))
        if p95 <= 1e-9:
            # Fewer than 5% of chunks contain the query terms: the most
            # specific queries. Dividing by p95 (= 0) used to zero them all
            # and BM25 returned no candidates; scale by the best match instead.
            p95 = float(s.max())
        return np.clip(s / p95, 0.0, 1.0)

    # -------------------- raw candidates -------------------- #

    def _bm25_candidates_raw(self, query: str, top_k: int = 350) -> Dict[int, float]:
        if self.lexical_retriever is not None:
            return {int(cid): float(score) for cid, score in self.lexical_retriever.search(query, top_k).items()}
        if not self.bm25_index:
            return {}
        toks = [t for t in self._tokenize(query) if t not in _STOPWORDS]
        if not toks:
            return {}
        scores = np.asarray(self.bm25_index.get_scores(toks), dtype=np.float32)
        norm = self._normalize_scores_robust_p95(scores)
        k = min(top_k, len(norm))
        if k <= 0:
            return {}
        idx = np.argpartition(-norm, k - 1)[:k]
        out: Dict[int, float] = {}
        for i in idx:
            cid = self._pos_to_chunk_id.get(int(i))
            if cid is None:
                continue
            val = float(norm[int(i)])
            if val <= 0.0:
                continue
            out[int(cid)] = val
        return out

    def _semantic_candidates_raw(self, query: str, top_k: int = 350) -> Dict[int, float]:
        if self._chunk_embeddings is None or len(self._chunk_embeddings) == 0:
            return {}

        # Instruction-tuned models (Qwen3-Embedding) embed queries differently from passages.
        encode_query = getattr(self.embedding_model, "encode_query", None)
        if callable(encode_query):
            q_emb = encode_query([query], show_progress_bar=False)
        else:
            q_emb = self.embedding_model.encode([query], show_progress_bar=False)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)

        if self.vector_store is not None:
            try:
                ids, scores = self.vector_store.search(q_emb, top_k=top_k)
                return {int(cid): float((float(score) + 1.0) / 2.0) for cid, score in zip(ids, scores)}
            except Exception as exc:
                logger.warning("External vector store search failed, using local embeddings: %s", exc)

        if self.faiss is not None:
            ids, scores = self.faiss.search(q_emb, top_k=top_k)
            out = {}
            for cid, sc in zip(ids, scores):
                # Похожесть FAISS может быть близка к cosine; переводим её в диапазон 0..1
                out[int(cid)] = float((float(sc) + 1.0) / 2.0)
            return out

        # cosine через numpy. Нормализованную матрицу держим заранее:
        # это быстрее на CPU и не создаёт новую копию всего индекса на каждый запрос.
        qe = q_emb[0]
        qe = qe / (np.linalg.norm(qe) + 1e-9)
        embsn = self._chunk_embeddings_norm
        if embsn is None:
            embs = self._chunk_embeddings
            embsn = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        sims = embsn @ qe  # диапазон [-1..1]
        k = min(top_k, len(sims))
        if k <= 0:
            return {}
        idx = np.argpartition(-sims, k - 1)[:k]
        out: Dict[int, float] = {}
        for i in idx:
            cid = self._pos_to_chunk_id.get(int(i))
            if cid is None:
                continue
            out[int(cid)] = float((float(sims[int(i)]) + 1.0) / 2.0)
        return out

    # -------------------- адаптивный retrieval -------------------- #

    def _adaptive_collect_candidates(
        self,
        question: str,
        k_base_override: Optional[int] = None,
        k_max_override: Optional[int] = None
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Динамически растит k, пока:
        - топ стабильный (пересечение >= threshold)
        - мало новых кандидатов
        - улучшение best score меньше eps
        - или истёк бюджет времени
        """
        adaptive_enabled = bool(self._retr_cfg.get("adaptive_enabled", True))
        if not adaptive_enabled:
            bm_k = int(self._retr_cfg.get("bm25_k", 480))
            se_k = int(self._retr_cfg.get("semantic_k", 480))
            try:
                bm = self._bm25_candidates_raw(question, top_k=bm_k)
            except Exception as exc:
                logger.warning("BM25 retrieval failed; continuing without it: %s", exc)
                bm = {}
            try:
                semantic = self._semantic_candidates_raw(question, top_k=se_k)
            except Exception as exc:
                logger.warning("Semantic retrieval failed; continuing without it: %s", exc)
                semantic = {}
            return bm, semantic

        k_base = int(k_base_override or self._retr_cfg.get("k_base", 80))
        k_step = int(self._retr_cfg.get("k_step", 80))
        k_max = int(k_max_override or self._retr_cfg.get("k_max", 480))

        time_budget_ms = float(self._retr_cfg.get("time_budget_ms", 280))
        min_new = int(self._retr_cfg.get("min_new_candidates", 25))
        topn = int(self._retr_cfg.get("topn_stability", 20))
        overlap_thr = float(self._retr_cfg.get("stability_overlap", 0.9))
        improve_eps = float(self._retr_cfg.get("score_improve_eps", 0.01))

        start = time.time()
        prev_top = None
        prev_best = 0.0
        prev_total = 0

        bm_best: Dict[int, float] = {}
        se_best: Dict[int, float] = {}

        # BM25 scores and the query embedding do not depend on the adaptive k.
        # Compute both expensive signals once, then expose progressively larger
        # prefixes to the existing stability/early-stop loop.
        try:
            bm_all = self._bm25_candidates_raw(question, top_k=k_max)
        except Exception as exc:
            logger.warning("BM25 retrieval failed; continuing without it: %s", exc)
            bm_all = {}
        try:
            se_all = self._semantic_candidates_raw(question, top_k=k_max)
        except Exception as exc:
            logger.warning("Semantic retrieval failed; continuing without it: %s", exc)
            se_all = {}
        bm_ranked = sorted(bm_all.items(), key=lambda item: item[1], reverse=True)
        se_ranked = sorted(se_all.items(), key=lambda item: item[1], reverse=True)

        k = max(10, k_base)
        while True:
            if (time.time() - start) * 1000.0 > time_budget_ms and k > k_base:
                break

            current_k = min(k, k_max)
            bm = dict(bm_ranked[:current_k])
            se = dict(se_ranked[:current_k])

            for cid, sc in bm.items():
                prev = bm_best.get(cid)
                if prev is None or sc > prev:
                    bm_best[cid] = sc

            for cid, sc in se.items():
                prev = se_best.get(cid)
                if prev is None or sc > prev:
                    se_best[cid] = sc

            candidate_ids = set(bm_best.keys()) | set(se_best.keys())
            total = len(candidate_ids)

            # быстрый скоринг для проверки стабильности (глобальный minmax по пулу)
            bm_vals = [bm_best.get(cid, 0.0) for cid in candidate_ids]
            se_vals = [se_best.get(cid, 0.0) for cid in candidate_ids]
            bm_norm = dict(zip(candidate_ids, _minmax_norm(bm_vals)))
            se_norm = dict(zip(candidate_ids, _minmax_norm(se_vals)))

            quick = []
            for cid in candidate_ids:
                base = bm_norm.get(cid, 0.0) * self.bm25_weight + se_norm.get(cid, 0.0) * self.semantic_weight
                quick.append((float(base), int(cid)))
            quick.sort(reverse=True, key=lambda x: x[0])

            top_ids = [cid for _, cid in quick[: max(1, topn)]]
            best = float(quick[0][0]) if quick else 0.0

            new_cnt = total - prev_total
            few_new = (prev_total > 0 and new_cnt < min_new)

            stable = False
            if prev_top is not None and top_ids:
                inter = len(set(prev_top) & set(top_ids))
                stable = (inter / max(1, len(set(top_ids)))) >= overlap_thr

            no_improve = (prev_best > 0 and (best - prev_best) < improve_eps)

            if prev_top is not None and (few_new and stable and no_improve):
                break

            prev_top = top_ids
            prev_best = best
            prev_total = total

            if k >= k_max:
                break
            k = min(k + k_step, k_max)

        return bm_best, se_best

    # -------------------- контекст (legacy) -------------------- #

    def _build_chunks_context_legacy(
        self,
        chunk_results: List[Tuple[float, Dict[str, Any]]],
        max_docs: int = 5,
        max_chars: int = 26000
    ) -> str:
        """Собирает читаемый блок контекста из лучших чанков, сгруппированных по документам."""
        if not chunk_results:
            return ""

        by_doc: Dict[str, List[Tuple[float, Dict[str, Any]]]] = defaultdict(list)
        for sc, ch in chunk_results:
            by_doc[ch["doc_id"]].append((sc, ch))

        doc_order = sorted(by_doc.keys(), key=lambda d: max(x[0] for x in by_doc[d]), reverse=True)

        parts = []
        total = 0
        docs_used = 0

        for doc_id in doc_order:
            if docs_used >= max_docs:
                break

            doc = self.doc_processor.get_document(doc_id)
            if not doc:
                continue

            chs = sorted(by_doc[doc_id], key=lambda x: x[0], reverse=True)[:3]

            header = (
                f"📄 ДОКУМЕНТ: {doc.name}\n"
                f"📅 Изменен: {time.ctime(doc.modified_time)}\n"
                f"📏 Размер: {doc.size_chars:,} символов\n"
                + "-" * 70 + "\n"
            )
            if total + len(header) > max_chars:
                break

            parts.append(header)
            total += len(header)

            for sc, ch in chs:
                block = (
                    f"[chunk score={sc:.3f}] "
                    f"(chunk_index={ch.get('chunk_index')}, chars={ch.get('start_char')}-{ch.get('end_char')})\n"
                    f"{ch['text']}\n\n"
                )
                if total + len(block) > max_chars:
                    break
                parts.append(block)
                total += len(block)

            docs_used += 1
            if total >= max_chars:
                break

        return "\n".join(parts).strip()

    # -------------------- основной поиск -------------------- #

    def hybrid_search(
        self,
        question: str,
        top_k: int = 5,
        return_full_documents: bool = True
    ) -> Dict[str, Any]:
        """Запускает полный гибридный пайплайн поиска для одного пользовательского запроса."""
        if not self.doc_processor.documents or not self._chunk_meta:
            payload = {"results": [], "results_all": [], "documents_context": "", "relevant_docs": [], "relevant_docs_all": []}
            try:
                self._telemetry.log_search(question, {"top_k": int(top_k), "return_full_documents": bool(return_full_documents), "reason": "no_documents_or_chunks", "candidate_count": 0, "documents_found": 0})
            except Exception as exc:
                logger.debug("Unable to write empty-search telemetry: %s", exc)
            return payload

        start_time = time.time()
        self.stats["total_searches"] += 1

        # 1. Опциональный роутинг интента может сузить или расширить лимиты
        # retrieval/контекста ещё до запуска дорогого скоринга.

        # профиль интента (опционально)
        profile_name = "default"
        k_base_override = None
        k_max_override = None
        date_strict = False
        ctx_docs_override = None
        ctx_chars_override = None

        if self._intent_router is not None:
            try:
                profile = self._intent_router.route(question)
                profile_name = getattr(profile, "name", "default") or "default"
                date_strict = bool(getattr(profile, "date_strict", False))
                kb = getattr(profile, "retrieval_k_base", None)
                km = getattr(profile, "retrieval_k_max", None)
                k_base_override = int(kb) if kb else None
                k_max_override = int(km) if km else None
                ctx_docs_override = int(getattr(profile, "context_max_docs", 0) or 0) or None
                ctx_chars_override = int(getattr(profile, "context_max_chars", 0) or 0) or None
            except Exception as exc:
                logger.warning("Intent routing failed; using default retrieval profile: %s", exc)

        # 2. Собираем лексических и семантических кандидатов, затем объединяем
        # их идентификаторы в единый пул для последующих эвристик.
        bm_raw, se_raw = self._adaptive_collect_candidates(
            question,
            k_base_override=k_base_override,
            k_max_override=k_max_override
        )

        exact_raw = self._safe_signal(
            "exact_identifier_index",
            lambda: self._exact_identifier_candidates(question),
            {},
        )
        candidate_ids = set(bm_raw.keys()) | set(se_raw.keys()) | set(exact_raw.keys())
        if not candidate_ids:
            payload = {"results": [], "results_all": [], "documents_context": "", "relevant_docs": [], "relevant_docs_all": [], "intent_profile": profile_name}
            try:
                self._telemetry.log_search(question, {"top_k": int(top_k), "return_full_documents": bool(return_full_documents), "reason": "no_candidates", "candidate_count": 0, "documents_found": 0, "intent_profile": profile_name})
            except Exception as exc:
                logger.debug("Unable to write no-candidate telemetry: %s", exc)
            return payload

        self.stats["avg_candidates"] = (
            (self.stats["avg_candidates"] * (self.stats["total_searches"] - 1)) + len(candidate_ids)
        ) / max(1, self.stats["total_searches"])

        # 3. BM25 и semantic живут в разных шкалах, поэтому сначала нормализуем
        # их, а уже потом смешиваем и добавляем эвристики по датам/сущностям/якорям.
        bm_vals = [bm_raw.get(cid, 0.0) for cid in candidate_ids]
        se_vals = [se_raw.get(cid, 0.0) for cid in candidate_ids]
        bm_norm = dict(zip(candidate_ids, _minmax_norm(bm_vals)))
        se_norm = dict(zip(candidate_ids, _minmax_norm(se_vals)))

        parsed_dates = self._safe_signal(
            "query_dates",
            lambda: self._parse_query_date_constraints(question),
            {"mode": "none", "start": None, "end": None, "dates": []},
        )
        q_start, q_end = parsed_dates.get("start"), parsed_dates.get("end")
        has_date_constraint = parsed_dates.get("mode") != "none" and bool(q_start or q_end or parsed_dates.get("dates"))

        query_identifiers = self._safe_signal("query_identifiers", lambda: self._extract_identifiers(question), [])
        query_object_terms = self._safe_signal("query_objects", lambda: self._extract_object_like_terms(question), [])
        query_entities = self._safe_signal("query_entities", lambda: self._extract_entities_universal(question), [])
        query_keywords = self._safe_extract_query_keywords(question)
        has_anchor_signal = bool(query_identifiers or query_object_terms)

        # оцениваем чанки
        chunk_texts_by_id: Dict[int, str] = {}
        chunk_meta_by_id: Dict[int, Dict[str, Any]] = {}
        chunk_meta_extra_by_id: Dict[int, Dict[str, Any]] = {}

        # We keep date bucket for date-first sorting:
        #   (date_bucket, score) where date_bucket=1 if matches date constraints (when date requested)
        chunk_scored: List[Tuple[Tuple[int, float], int]] = []

        for cid in candidate_ids:
            pos = self._chunk_id_to_pos.get(cid)
            if pos is None:
                continue

            text = self._chunk_texts[pos]
            meta_src = self._chunk_meta[pos]

            chunk_texts_by_id[int(cid)] = text
            chunk_meta_by_id[int(cid)] = meta_src

            base = bm_norm.get(cid, 0.0) * self.bm25_weight + se_norm.get(cid, 0.0) * self.semantic_weight

            id_bonus, id_match, matched_ids = self._safe_signal(
                "exact_identifier", lambda: self._bonus_by_identifiers(question, text), (0.0, False, [])
            )
            obj_bonus, object_term_hits = self._safe_signal(
                "object", lambda: self._object_term_bridge_score(question, text), (0.0, 0)
            )
            kw_bonus, keyword_match_count = self._safe_signal(
                "keywords", lambda: self._keyword_overlap_no_dates(question, text, query_keywords), (0.0, 0)
            )
            fuzzy_bonus, fuzzy_match_count = self._safe_signal(
                "fuzzy", lambda: self._fuzzy_keyword_score(question, text, query_keywords), (0.0, 0)
            )
            date_adj, date_match, matched_requested_dates, date_coverage = self._safe_signal(
                "dates",
                lambda: self._date_range_score(question, text, cid, date_strict=date_strict),
                (0.0, False, 0, 0.0),
            )
            ent_bonus, ent_match_cnt = self._safe_signal(
                "entities", lambda: self._entities_overlap_score(question, text, cid), (0.0, 0)
            )

            # если в запросе есть ограничение по дате — немного усиливаем влияние сущностей (старое поведение)
            if has_date_constraint:
                ent_bonus = min(0.32, ent_bonus + 0.02)

            score = float(base + id_bonus + date_adj + ent_bonus + 0.70 * obj_bonus + 0.45 * kw_bonus + fuzzy_bonus)
            score = _safe_clamp01(score)

            date_bucket = 1 if (has_date_constraint and date_match) else 0
            chunk_scored.append(((date_bucket, score), int(cid)))

            chunk_meta_extra_by_id[int(cid)] = {
                "bm25_raw_score": float(bm_raw.get(cid, 0.0)),
                "semantic_raw_score": float(se_raw.get(cid, 0.0)),
                "date_match": bool(date_match),
                "id_match": bool(id_match),
                "matched_ids": matched_ids,
                "entity_match_count": int(ent_match_cnt),
                "entity_match": bool(ent_match_cnt > 0),
                "object_term_hits": int(object_term_hits),
                "anchor_match": bool(id_match or object_term_hits > 0),
                "keyword_match_count": int(keyword_match_count),
                "fuzzy_match_count": int(fuzzy_match_count),
                "query_term_coverage": float(keyword_match_count / max(1, len(query_keywords))),
                "matched_requested_dates": int(matched_requested_dates),
                "date_coverage": float(date_coverage),
            }

        chunk_scored.sort(reverse=True, key=lambda x: x[0])

        # дедупликация + rerank (опционально)
        if _RERANK_OK and self._rer_cfg is not None and dedup_scored_chunks is not None:
            try:
                if getattr(self._rer_cfg, "dedup_enabled", False):
                    chunk_scored = dedup_scored_chunks(chunk_scored, chunk_texts_by_id, self._rer_cfg)
            except Exception as exc:
                logger.warning("Chunk deduplication failed; continuing without it: %s", exc)

        if _RERANK_OK and self._rer_cfg is not None and rerank_scored_chunks is not None:
            try:
                if getattr(self._rer_cfg, "enabled", False):
                    chunk_scored = rerank_scored_chunks(
                        chunk_scored,
                        chunk_meta_extra_by_id,
                        chunk_meta_by_id,
                        self._rer_cfg,
                        has_date_constraint=has_date_constraint
                    )
            except Exception as exc:
                logger.warning("Heuristic reranking failed; continuing with retrieval scores: %s", exc)

        if self._learned_reranker is not None:
            try:
                chunk_scored = self._learned_reranker.rerank(
                    question,
                    chunk_scored,
                    chunk_texts_by_id,
                )
            except Exception as exc:
                logger.warning("Learned reranking failed; continuing with heuristic scores: %s", exc)

        if self.reranker_provider is not None:
            try:
                chunk_scored = self.reranker_provider.rerank(
                    question,
                    chunk_scored,
                    chunk_texts_by_id,
                )
            except Exception as exc:
                logger.warning("Provider reranking failed; continuing with prior scores: %s", exc)

        # агрегируем результаты до документов с объединением сигналов из нескольких чанков
        doc_best: Dict[str, float] = defaultdict(float)
        doc_best_date: Dict[str, float] = defaultdict(float)
        doc_best_id: Dict[str, float] = defaultdict(float)
        doc_best_object: Dict[str, float] = defaultdict(float)
        doc_best_entity: Dict[str, float] = defaultdict(float)
        doc_best_semantic_raw: Dict[str, float] = defaultdict(float)
        doc_best_bm25_raw: Dict[str, float] = defaultdict(float)
        doc_best_query_coverage: Dict[str, float] = defaultdict(float)
        doc_counts: Dict[str, int] = defaultdict(int)
        doc_counts_date: Dict[str, int] = defaultdict(int)
        doc_counts_id: Dict[str, int] = defaultdict(int)
        doc_counts_object: Dict[str, int] = defaultdict(int)
        doc_counts_entity: Dict[str, int] = defaultdict(int)
        doc_support_chunk_ids: Dict[str, set] = defaultdict(set)

        for (date_bucket, sc), cid in chunk_scored[:1400]:
            pos = self._chunk_id_to_pos.get(cid)
            if pos is None:
                continue
            meta = self._chunk_meta[pos]
            doc_id = meta["doc_id"]
            extra = chunk_meta_extra_by_id.get(int(cid), {})

            doc_best_semantic_raw[doc_id] = max(doc_best_semantic_raw[doc_id], float(extra.get("semantic_raw_score", 0.0) or 0.0))
            doc_best_bm25_raw[doc_id] = max(doc_best_bm25_raw[doc_id], float(extra.get("bm25_raw_score", 0.0) or 0.0))
            doc_best_query_coverage[doc_id] = max(doc_best_query_coverage[doc_id], float(extra.get("query_term_coverage", 0.0) or 0.0))

            doc_counts[doc_id] += 1
            doc_best[doc_id] = max(doc_best[doc_id], float(sc))

            if has_date_constraint and date_bucket == 1:
                doc_counts_date[doc_id] += 1
                doc_best_date[doc_id] = max(doc_best_date[doc_id], float(sc))
                doc_support_chunk_ids[doc_id].add(int(cid))

            if extra.get("id_match"):
                doc_counts_id[doc_id] += 1
                doc_best_id[doc_id] = max(doc_best_id[doc_id], float(sc))
                doc_support_chunk_ids[doc_id].add(int(cid))

            if extra.get("object_term_hits", 0) > 0:
                doc_counts_object[doc_id] += 1
                doc_best_object[doc_id] = max(doc_best_object[doc_id], float(sc))
                doc_support_chunk_ids[doc_id].add(int(cid))

            if extra.get("entity_match_count", 0) > 0:
                doc_counts_entity[doc_id] += 1
                doc_best_entity[doc_id] = max(doc_best_entity[doc_id], float(sc))
                doc_support_chunk_ids[doc_id].add(int(cid))

        doc_scores: Dict[str, float] = {}
        doc_has_date: Dict[str, bool] = {}
        doc_has_anchor: Dict[str, bool] = {}
        doc_has_entity: Dict[str, bool] = {}
        doc_cross_signal: Dict[str, bool] = {}

        for doc_id in doc_best.keys():
            has_doc_date = bool(doc_id in doc_best_date)
            has_doc_anchor = bool(doc_id in doc_best_id or doc_id in doc_best_object)
            has_doc_entity = bool(doc_id in doc_best_entity)

            doc_has_date[doc_id] = has_doc_date
            doc_has_anchor[doc_id] = has_doc_anchor
            doc_has_entity[doc_id] = has_doc_entity

            if has_date_constraint:
                if has_doc_date:
                    coverage_bonus = min(0.12, 0.02 * doc_counts_date.get(doc_id, 0))
                    score = min(1.0, doc_best_date[doc_id] + 0.08 + coverage_bonus)
                else:
                    score = max(0.0, doc_best[doc_id] - (0.65 if date_strict else 0.45))
            else:
                score = doc_best[doc_id]

            if has_anchor_signal and has_doc_anchor:
                anchor_cnt = max(doc_counts_id.get(doc_id, 0), doc_counts_object.get(doc_id, 0))
                score = min(1.0, score + 0.18 + min(0.08, 0.02 * anchor_cnt))

            if query_entities and has_doc_entity:
                score = min(1.0, score + 0.08 + min(0.06, 0.015 * doc_counts_entity.get(doc_id, 0)))

            cross = False
            if has_date_constraint:
                if has_anchor_signal and has_doc_date and has_doc_anchor:
                    score = min(1.0, score + 0.22)
                    cross = True
                elif (not has_anchor_signal) and query_entities and has_doc_date and has_doc_entity:
                    score = min(1.0, score + 0.14)
                    cross = True
                elif has_anchor_signal and query_entities and has_doc_date and has_doc_entity:
                    score = min(1.0, score + 0.06)

            doc_cross_signal[doc_id] = cross

            cnt = doc_counts.get(doc_id, 0)
            if cnt >= 3:
                score = min(1.0, score + 0.05)
            if cnt >= 6:
                score = min(1.0, score + 0.05)

            doc_scores[doc_id] = score

        doc_ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # build chunk results (diversified)
        has_any_date_hits = bool(has_date_constraint and any(extra.get("date_match") for extra in chunk_meta_extra_by_id.values()))
        has_any_anchor_hits = bool(any(extra.get("anchor_match") for extra in chunk_meta_extra_by_id.values()))
        has_any_entity_hits = bool(any(extra.get("entity_match_count", 0) > 0 for extra in chunk_meta_extra_by_id.values()))
        preferred_docs = {doc_id for doc_id, _ in doc_ranked if doc_cross_signal.get(doc_id)}
        if not preferred_docs and has_date_constraint and has_anchor_signal:
            preferred_docs = {doc_id for doc_id, _ in doc_ranked if doc_has_date.get(doc_id) and doc_has_anchor.get(doc_id)}
        results_all: List[Tuple[float, Dict[str, Any]]] = []
        docs_seen = defaultdict(int)

        per_doc_cap = int(self._retr_cfg.get("per_doc_chunk_cap", 3))
        max_results_cap = int(self._retr_cfg.get("max_results_cap", 80))

        for (date_bucket, sc), cid in chunk_scored:
            pos = self._chunk_id_to_pos.get(cid)
            if pos is None:
                continue

            meta = dict(self._chunk_meta[pos])
            meta["text"] = self._chunk_texts[pos]
            meta["score"] = float(sc)

            extra = chunk_meta_extra_by_id.get(int(cid), {})
            meta.update(extra)

            doc_id = meta["doc_id"]

            # Если найден документ, где дата и объект/сущность подтверждаются РАЗНЫМИ чанками,
            # не требуем от каждого отдельного чанка совпасть по обоим признакам одновременно.
            if preferred_docs and doc_id not in preferred_docs:
                continue

            support_ids = doc_support_chunk_ids.get(doc_id, set())
            chunk_index = int(meta.get("chunk_index", -9999))
            is_support_chunk = int(cid) in support_ids
            is_neighbor_of_support = False
            if support_ids:
                for support_cid in support_ids:
                    support_pos = self._chunk_id_to_pos.get(int(support_cid))
                    if support_pos is None:
                        continue
                    support_meta = self._chunk_meta[support_pos]
                    if support_meta.get("doc_id") != doc_id:
                        continue
                    if abs(int(support_meta.get("chunk_index", -9999)) - chunk_index) <= 1:
                        is_neighbor_of_support = True
                        break

            if preferred_docs:
                contributes = bool(meta.get("date_match") or meta.get("anchor_match") or meta.get("entity_match_count", 0) > 0 or meta.get("keyword_match_count", 0) >= 2 or is_support_chunk or is_neighbor_of_support)
                if not contributes:
                    continue
            else:
                if has_any_date_hits and has_date_constraint and not meta.get("date_match"):
                    # если у документа уже есть объектное совпадение, позволяем взять соседний объектный чанк
                    if not (doc_has_anchor.get(doc_id) and (meta.get("anchor_match") or meta.get("entity_match_count", 0) > 0 or meta.get("keyword_match_count", 0) >= 2 or is_neighbor_of_support)):
                        continue
                if has_any_anchor_hits and has_anchor_signal and not meta.get("anchor_match"):
                    # если у документа уже есть датовое попадание — оставляем датовый supporting chunk
                    if not (doc_has_date.get(doc_id) and (meta.get("date_match") or is_neighbor_of_support)):
                        continue
                if not _passes_entity_signal_filter(
                    meta,
                    doc_id,
                    has_any_entity_hits=has_any_entity_hits,
                    query_entities=query_entities,
                    doc_has_date=doc_has_date,
                    is_neighbor_of_support=is_neighbor_of_support,
                ):
                    continue

            if docs_seen[doc_id] >= per_doc_cap and len(results_all) < top_k:
                continue
            docs_seen[doc_id] += 1

            meta["doc_cross_signal"] = bool(doc_cross_signal.get(doc_id))
            meta["doc_has_date"] = bool(doc_has_date.get(doc_id))
            meta["doc_has_anchor"] = bool(doc_has_anchor.get(doc_id))
            meta["doc_has_entity"] = bool(doc_has_entity.get(doc_id))
            meta["is_support_chunk"] = bool(is_support_chunk)
            meta["is_neighbor_of_support"] = bool(is_neighbor_of_support)

            results_all.append((float(sc), meta))
            if len(results_all) >= max(30, top_k * 6, max_results_cap):
                break

        # узкий rescue-pass: если документ уже определён, но в пуле нет ответного чанка,
        # добираем лучший object/entity chunk по всему документу без требования локальной даты.
        rescue_docs = int(self._retr_cfg.get("answer_rescue_docs", 4) or 4)
        existing_chunk_ids = {
            int(meta.get("chunk_id"))
            for _, meta in results_all
            if meta.get("chunk_id") is not None
        }
        answer_docs_present = defaultdict(bool)
        for _, meta in results_all:
            d_id = str(meta.get("doc_id"))
            if meta.get("anchor_match") or meta.get("entity_match_count", 0) > 0 or meta.get("keyword_match_count", 0) >= 2:
                answer_docs_present[d_id] = True

        rescue_targets = doc_ranked[:max(1, rescue_docs)]
        if has_date_constraint and any(doc_has_date.get(doc_id) for doc_id, _ in doc_ranked):
            rescue_targets = [(doc_id, sc) for doc_id, sc in rescue_targets if doc_has_date.get(doc_id)] or rescue_targets

        for doc_id, _doc_sc in rescue_targets:
            doc_id = str(doc_id)
            if answer_docs_present.get(doc_id):
                continue
            rescued = self._find_best_answer_chunk_for_doc(
                question,
                doc_id,
                doc_has_date=bool(doc_has_date.get(doc_id)),
                exclude_chunk_ids=existing_chunk_ids,
            )
            if not rescued:
                continue
            rescued_score, rescued_meta = rescued
            rescued_meta["doc_cross_signal"] = bool(doc_cross_signal.get(doc_id))
            rescued_meta["doc_has_date"] = bool(doc_has_date.get(doc_id))
            rescued_meta["doc_has_anchor"] = bool(doc_has_anchor.get(doc_id))
            rescued_meta["doc_has_entity"] = bool(doc_has_entity.get(doc_id))
            rescued_meta["is_support_chunk"] = False
            rescued_meta["is_neighbor_of_support"] = False
            results_all.append((float(rescued_score), rescued_meta))
            existing_chunk_ids.add(int(rescued_meta.get("chunk_id")))
            answer_docs_present[doc_id] = True

        if results_all:
            # Если нашли answer-chunk, не держим второй чисто датовый header для того же документа.
            filtered_results: List[Tuple[float, Dict[str, Any]]] = []
            docs_with_answer = {
                str(meta.get("doc_id"))
                for _, meta in results_all
                if meta.get("answer_rescue") or meta.get("anchor_match") or meta.get("entity_match_count", 0) > 0 or meta.get("keyword_match_count", 0) >= 2
            }
            date_only_seen = set()
            for sc, meta in sorted(results_all, key=lambda x: float(x[0]), reverse=True):
                d_id = str(meta.get("doc_id"))
                date_only = bool(meta.get("date_match") and not meta.get("anchor_match") and meta.get("entity_match_count", 0) <= 0 and meta.get("keyword_match_count", 0) < 2)
                if d_id in docs_with_answer and date_only:
                    if d_id in date_only_seen:
                        continue
                    # совсем не показываем date-only header, если для документа уже есть ответный фрагмент
                    continue
                filtered_results.append((float(sc), meta))
            results_all = filtered_results

        # сборка контекста
        documents_context = ""
        if return_full_documents:
            # используем ctxv2, если он доступен
            if _CTXV2_OK and self._ctx2_cfg is not None and build_context_v2 is not None:
                try:
                    ctx_cfg = self._ctx2_cfg
                    if ctx_docs_override or ctx_chars_override:
                        try:
                            ctx_cfg = replace(
                                self._ctx2_cfg,
                                max_docs=int(ctx_docs_override or self._ctx2_cfg.max_docs),
                                max_chars=int(ctx_chars_override or self._ctx2_cfg.max_chars)
                            )
                        except Exception:
                            # very safe fallback
                            ctx_cfg.max_docs = int(ctx_docs_override or ctx_cfg.max_docs)
                            ctx_cfg.max_chars = int(ctx_chars_override or ctx_cfg.max_chars)

                    documents_context = build_context_v2(
                        results_all,
                        self.doc_processor,
                        cfg=ctx_cfg,
                        chunk_id_to_pos=self._chunk_id_to_pos,
                        pos_to_chunk_id=self._pos_to_chunk_id,
                        chunk_texts=self._chunk_texts,
                        chunk_meta_list=self._chunk_meta,
                    )
                except Exception:
                    # иначе откатываемся к legacy-варианту
                    documents_context = self._build_chunks_context_legacy(
                        results_all,
                        max_docs=int(self._ctx_cfg.get("max_docs", 5)),
                        max_chars=int(self._ctx_cfg.get("max_chars", 26000)),
                    )
            else:
                documents_context = self._build_chunks_context_legacy(
                    results_all,
                    max_docs=int(self._ctx_cfg.get("max_docs", 5)),
                    max_chars=int(self._ctx_cfg.get("max_chars", 26000)),
                )

        # релевантные документы (полный список + обрезанный)
        relevant_docs_all: List[Tuple[float, Dict[str, Any]]] = []
        # для интерфейса показываем больше, чем top_k, чтобы выдача была стабильнее
        take_docs = max(20, top_k)
        ranked_docs_source = doc_ranked
        if has_date_constraint and any(doc_id in doc_best_date for doc_id, _ in doc_ranked):
            ranked_docs_source = [(doc_id, sc) for doc_id, sc in ranked_docs_source if doc_id in doc_best_date]
        if has_any_anchor_hits and any((doc_id in doc_best_id or doc_id in doc_best_object) for doc_id, _ in ranked_docs_source):
            ranked_docs_source = [(doc_id, sc) for doc_id, sc in ranked_docs_source if (doc_id in doc_best_id or doc_id in doc_best_object)]
        for doc_id, sc in ranked_docs_source[:take_docs]:
            doc = self.doc_processor.get_document(doc_id)
            if not doc:
                continue
            doc_dict = doc.to_dict()
            doc_dict["doc_id"] = doc_id
            doc_dict["date_match"] = bool(doc_id in doc_best_date)
            doc_dict["date_match_count"] = int(doc_counts_date.get(doc_id, 0))
            doc_dict["id_match"] = bool(doc_id in doc_best_id or doc_id in doc_best_object)
            doc_dict["id_match_count"] = int(max(doc_counts_id.get(doc_id, 0), doc_counts_object.get(doc_id, 0)))
            doc_dict["semantic_raw_score"] = float(doc_best_semantic_raw.get(doc_id, 0.0))
            doc_dict["bm25_raw_score"] = float(doc_best_bm25_raw.get(doc_id, 0.0))
            doc_dict["query_term_coverage"] = float(doc_best_query_coverage.get(doc_id, 0.0))
            relevant_docs_all.append((float(sc), doc_dict))

        elapsed = time.time() - start_time
        self.stats["avg_search_time"] = (
            (self.stats["avg_search_time"] * (self.stats["total_searches"] - 1)) + elapsed
        ) / max(1, self.stats["total_searches"])

        result_payload = {
            "results": results_all[:top_k],
            "results_all": results_all,
            "documents_context": documents_context,
            "relevant_docs": relevant_docs_all[:top_k],
            "relevant_docs_all": relevant_docs_all,
            "total_documents_found": len(relevant_docs_all),
            "search_mode": "chunk_hybrid_adaptive_date_first_entities_dedup_rerank_ctxv2_intent",
            "intent_profile": profile_name
        }

        telemetry_payload = {
            "top_k": int(top_k),
            "return_full_documents": bool(return_full_documents),
            "elapsed_ms": round(elapsed * 1000.0, 2),
            "intent_profile": profile_name,
            "candidate_count": len(candidate_ids),
            "date_constraint": has_date_constraint,
            "query_identifiers": query_identifiers,
            "query_object_terms": query_object_terms,
            "query_entities": query_entities,
            "query_keywords": query_keywords,
            "documents_found": len(relevant_docs_all),
            "top_documents": [
                {
                    "score": float(score),
                    "doc_id": doc.get("doc_id"),
                    "name": doc.get("name"),
                    "date_match": bool(doc.get("date_match")),
                    "id_match": bool(doc.get("id_match")),
                }
                for score, doc in relevant_docs_all[: self._diagnostics.cfg.max_trace_items]
            ],
            "top_chunks": [
                {
                    "score": float(score),
                    "chunk_id": meta.get("chunk_id"),
                    "doc_id": meta.get("doc_id"),
                    "date_match": bool(meta.get("date_match")),
                    "id_match": bool(meta.get("id_match")),
                    "object_term_hits": int(meta.get("object_term_hits", 0) or 0),
                    "entity_match_count": int(meta.get("entity_match_count", 0) or 0),
                    "preview": self._diagnostics.preview(meta.get("text", "")),
                }
                for score, meta in results_all[: self._diagnostics.cfg.max_trace_items]
            ],
        }
        try:
            self._diagnostics.write_search_trace(question, telemetry_payload)
        except Exception as exc:
            logger.warning("Unable to write search diagnostics: %s", exc)
        try:
            self._telemetry.log_search(question, telemetry_payload)
        except Exception as exc:
            logger.debug("Unable to write search telemetry: %s", exc)

        return result_payload

    # -------------------- прочее -------------------- #

    def quick_search(self, question: str) -> List[str]:
        """Возвращает компактную текстовую сводку по лучшим найденным документам."""
        result = self.hybrid_search(question, top_k=5, return_full_documents=False)
        docs = result.get("relevant_docs_all", []) or result.get("relevant_docs", [])
        if not docs:
            return ["Нет релевантных документов"]
        out = ["🔍 НАЙДЕННЫЕ ДОКУМЕНТЫ (v1.2):"]
        for i, (score, doc_dict) in enumerate(docs[:10], 1):
            out.append(f"{i}. {doc_dict.get('name', 'Неизвестно')} (релевантность: {float(score):.3f})")
        return out

    def save_cache(self, force: bool = True):
        """Сохраняет кэш эмбеддингов.

        Для маленьких корпусов сохраняем принудительно, иначе кэш может никогда
        не появиться на диске из-за слишком крупного порога auto-save.
        """
        self.embedding_cache.save(force=force)
        logger.info("Embedding cache saved (force=%s)", force)

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает верхнеуровневую статистику поиска и индекса для диагностики."""
        return {
            "total_searches": self.stats["total_searches"],
            "avg_search_time": self.stats["avg_search_time"],
            "avg_candidates": self.stats.get("avg_candidates", 0.0),
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "total_documents": len(self.doc_processor.documents),
            "total_chunks": len(self._chunk_meta),
            "bm25_weight": self.bm25_weight,
            "semantic_weight": self.semantic_weight,
            "faiss_enabled": self.faiss is not None
        }

    def rebuild_indexes(self):
        """Полностью перестраивает поисковые индексы после изменения набора документов."""
        logger.info("Rebuilding search indexes")
        self._query_reference_cache.clear()
        self._query_date_parse_cache.clear()
        self._initialize_indexes("faiss_index.bin", "faiss_meta.pkl")
        logger.info("Search indexes rebuilt")


class AnswerGenerator(HybridSearch):
    """Тонкая совместимая обёртка над ``HybridSearch`` для RAG-вывода."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_rag_answer(self, question: str) -> Tuple[str, List]:
        """Возвращает сгенерированный контекст и лучшие чанки для вопроса."""
        search_result = self.hybrid_search(question, return_full_documents=True)
        context = search_result.get("documents_context", "")
        results = []
        for score, chunk in search_result.get("results", []):
            results.append((score, chunk))
        return context, results
