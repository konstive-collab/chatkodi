from __future__ import annotations

import json
import threading
import numpy as np
import pytest
from queue import Queue
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

from core.config import DEFAULT_CONFIG
from core.domain_lexicon import DomainLexicon
from core.domain_profile import BUILTIN_DOMAIN_PROFILES, get_domain_profile
from core.chat.manager import ChatManager
from core.chat_history import ChatHistoryStore
from core.chat_mode import ChatMode
from core.faiss_manager import FaissManager
from core.llm import runtime as runtime_module
from core.llm.runtime import LocalLLM, classify_llm_unavailability, get_last_llm_load_status
from core.system_profile import safe_cpu_model_size_gb
from core import memory_guard
from core.intent_router import IntentRouter, profile_name_for_intent
from core.model_integrity import (
    create_model_hash_manifest,
    inspect_hf_model_payload,
    verify_model_hash_manifest,
)
from core.search_engine.hybrid_search import HybridSearch, _passes_entity_signal_filter
from core.search_engine import hybrid_search as hybrid_search_module
from core.fact_extraction import _detect_polarity, _extract_action, _parse_major_section
from core.structured_output import extract_json_object
from core.query_understanding import analyze_question, QueryUnderstandingResult, QueryIntent
from core.query_planner import QueryPlanner
from core.rerank import CrossEncoderReranker, RerankConfig
from core.retrieval_signals import is_anchor_chunk
from core.providers import EmbeddingProvider, VectorStore, LexicalRetriever, Reranker, LLMProvider
from integrations.server_providers import TEIEmbeddingProvider, QdrantVectorStore, VLLMProvider
from gui.main_window.documents import ChatKodiDocumentsMixin
from gui.workers.streaming import StreamingWorker
from tools.model_manager.downloader import SmartModelDownloader
from tools.run_benchmark import _load_cases


def test_entity_signal_filter_rejects_unrelated_chunk() -> None:
    assert not _passes_entity_signal_filter(
        {"entity_match_count": 0, "date_match": False},
        "doc-1",
        has_any_entity_hits=True,
        query_entities=["alpha"],
        doc_has_date={"doc-1": False},
        is_neighbor_of_support=False,
    )


def test_entity_signal_filter_keeps_dated_support_chunk() -> None:
    assert _passes_entity_signal_filter(
        {"entity_match_count": 0, "date_match": True},
        "doc-1",
        has_any_entity_hits=True,
        query_entities=["alpha"],
        doc_has_date={"doc-1": True},
        is_neighbor_of_support=False,
    )


def test_anchor_signal_has_one_shared_definition() -> None:
    assert is_anchor_chunk({"answer_rescue": True})
    assert is_anchor_chunk({"structured_entity_match_count": 1})
    assert not is_anchor_chunk({"keyword_match_count": 3})


def test_hybrid_search_returns_cleanly_without_documents() -> None:
    search = object.__new__(HybridSearch)
    search.doc_processor = SimpleNamespace(documents={})
    search._chunk_meta = []
    search._telemetry = SimpleNamespace(log_search=lambda *_args, **_kwargs: None)

    payload = search.hybrid_search("question")

    assert payload["results"] == []
    assert payload["relevant_docs"] == []


def test_hybrid_search_returns_cleanly_without_candidates() -> None:
    search = object.__new__(HybridSearch)
    search.doc_processor = SimpleNamespace(documents={"doc": object()})
    search._chunk_meta = [{"id": 1}]
    search._telemetry = SimpleNamespace(log_search=lambda *_args, **_kwargs: None)
    search._intent_router = None
    search._adaptive_collect_candidates = lambda *_args, **_kwargs: ({}, {})
    search.stats = {"total_searches": 0, "avg_candidates": 0.0}

    payload = search.hybrid_search("question")

    assert payload["results"] == []
    assert payload["intent_profile"] == "default"


def test_adaptive_retrieval_computes_each_signal_only_once() -> None:
    search = object.__new__(HybridSearch)
    search._retr_cfg = {
        "adaptive_enabled": True,
        "k_base": 10,
        "k_step": 10,
        "k_max": 30,
        "time_budget_ms": 10_000,
        "min_new_candidates": 0,
        "topn_stability": 5,
        "stability_overlap": 2.0,
        "score_improve_eps": -1.0,
    }
    search.bm25_weight = 0.35
    search.semantic_weight = 0.65
    calls = {"bm25": 0, "semantic": 0}

    def bm25(_query: str, top_k: int) -> dict[int, float]:
        calls["bm25"] += 1
        return {idx: 1.0 - (idx / 100.0) for idx in range(top_k)}

    def semantic(_query: str, top_k: int) -> dict[int, float]:
        calls["semantic"] += 1
        return {idx: 1.0 - (idx / 100.0) for idx in range(top_k)}

    search._bm25_candidates_raw = bm25
    search._semantic_candidates_raw = semantic

    bm, semantic = search._adaptive_collect_candidates("test")

    assert len(bm) == 30
    assert len(semantic) == 30
    assert calls == {"bm25": 1, "semantic": 1}


def test_keyword_extractor_failure_isolated_from_other_retrieval_signals() -> None:
    search = object.__new__(HybridSearch)
    search._tokenize = lambda _text: (_ for _ in ()).throw(RuntimeError("extractor failed"))

    assert search._safe_extract_query_keywords("query") == []


def test_bm25_and_semantic_fail_independently() -> None:
    search = object.__new__(HybridSearch)
    search._retr_cfg = {"adaptive_enabled": False, "bm25_k": 10, "semantic_k": 10}
    search._bm25_candidates_raw = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bm25 failed"))
    search._semantic_candidates_raw = lambda *_args, **_kwargs: {7: 0.9}

    bm25, semantic = search._adaptive_collect_candidates("query")

    assert bm25 == {}
    assert semantic == {7: 0.9}


def test_exact_identifier_index_distinguishes_11_from_111_and_keeps_date_filter() -> None:
    search = object.__new__(HybridSearch)
    search._identifier_to_chunk_ids = {
        "11": {1, 3},
        "111": {2},
        "A123BC77": {4},
    }

    assert search._exact_identifier_candidates("автомобиль №11, 5 августа 2025 года") == {1: 1.0, 3: 1.0}
    assert search._exact_identifier_candidates("автомобиль №111") == {2: 1.0}
    assert search._exact_identifier_candidates("госномер А123ВС77") == {4: 1.0}


def test_exact_identifier_candidates_intersect_with_requested_date() -> None:
    search = object.__new__(HybridSearch)
    search._identifier_to_chunk_ids = {"11": {1, 2}}
    search._query_date_parse_cache = {}
    search._query_reference_cache = {}
    search._chunk_meta = []
    search._chunk_dates_cache = {
        1: [datetime(2025, 8, 5)],
        2: [datetime(2025, 8, 6)],
    }

    candidates = search._exact_identifier_candidates("автомобиль №11 5 августа 2025 года")
    matching = {
        chunk_id
        for chunk_id in candidates
        if search._date_range_score(
            "автомобиль №11 5 августа 2025 года",
            "",
            chunk_id,
            date_strict=True,
        )[1]
    }

    assert matching == {1}


def test_retrieval_heuristic_signal_failure_returns_neutral_default() -> None:
    search = object.__new__(HybridSearch)
    search._failed_signal_warnings = set()

    result = search._safe_signal(
        "dates",
        lambda: (_ for _ in ()).throw(RuntimeError("date parser failed")),
        (0.0, False, 0, 0.0),
    )

    assert result == (0.0, False, 0, 0.0)
    assert search._failed_signal_warnings == {"dates"}


def test_default_config_contains_runtime_sections_and_portable_catalog() -> None:
    assert DEFAULT_CONFIG["context_v2"]["enabled"] is True
    assert DEFAULT_CONFIG["rerank"]["enabled"] is True
    assert DEFAULT_CONFIG["intents"]["profiles"]["fact_lookup"]
    assert DEFAULT_CONFIG["generation"]["catalog_root"] == "documents"
    assert DEFAULT_CONFIG["retrieval"]["bm25_weight"] == 0.35
    assert DEFAULT_CONFIG["retrieval"]["semantic_weight"] == 0.65


def test_shipped_config_exposes_critical_sections_without_developer_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    shipped = json.loads((root / "chatkodi_config.json").read_text(encoding="utf-8"))

    for section in ("context_v2", "rerank", "intents", "history", "domain_lexicon"):
        assert isinstance(shipped.get(section), dict) and shipped[section]
    assert shipped["generation"]["catalog_root"] == "documents"
    assert "chatkodi_v1.6" not in json.dumps(shipped, ensure_ascii=False)


def test_faiss_rebuild_check_validates_chunk_identity() -> None:
    manager = object.__new__(FaissManager)
    manager.dimension = 384
    manager.expected_dimension = 384
    manager.namespace = "model-a"
    manager.index_namespace = "model-a"
    manager.index = SimpleNamespace(ntotal=2)
    manager.chunk_to_idx = {10: 0, 20: 1}
    manager.idx_to_chunk = {0: 10, 1: 20}

    assert not manager.needs_rebuild(2, chunk_ids=[10, 20])
    assert manager.needs_rebuild(2, chunk_ids=[20, 10])
    assert manager.needs_rebuild(2, chunk_ids=[10, 30])


def test_faiss_rebuild_check_validates_embedding_model_namespace() -> None:
    manager = object.__new__(FaissManager)
    manager.dimension = 384
    manager.expected_dimension = 384
    manager.namespace = "model-new"
    manager.index_namespace = "model-old"
    manager.index = SimpleNamespace(ntotal=1)
    manager.chunk_to_idx = {10: 0}
    manager.idx_to_chunk = {0: 10}

    assert manager.needs_rebuild(1, chunk_ids=[10])


def test_faiss_save_load_roundtrip_reuses_matching_index(tmp_path, monkeypatch) -> None:
    class FakeIndex:
        def __init__(self, total=0):
            self.ntotal = total

        def add(self, vectors):
            self.ntotal += len(vectors)

    class FakeFaiss:
        @staticmethod
        def normalize_L2(_vectors):
            return None

        @staticmethod
        def write_index(index, path):
            Path(path).write_text(str(index.ntotal), encoding="ascii")

        @staticmethod
        def read_index(path):
            return FakeIndex(int(Path(path).read_text(encoding="ascii")))

    accelerator = SimpleNamespace(
        faiss_gpu_available=False,
        create_faiss_index=lambda _dimension, use_gpu=True: FakeIndex(),
    )
    monkeypatch.setattr("core.faiss_manager.FAISS_AVAILABLE", True)
    monkeypatch.setattr("core.faiss_manager.faiss", FakeFaiss)
    monkeypatch.setattr("core.faiss_manager.get_gpu_accelerator", lambda: accelerator)
    index_path = tmp_path / "index.bin"
    meta_path = tmp_path / "meta.pkl"

    created = FaissManager(2, str(index_path), str(meta_path), use_gpu=False, namespace="emb-v1")
    assert created.build_index(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), [10, 20])

    loaded = FaissManager(2, str(index_path), str(meta_path), use_gpu=False, namespace="emb-v1")

    assert loaded.index is not None
    assert not loaded.needs_rebuild(2, chunk_ids=[10, 20])
    assert loaded.needs_rebuild(2, chunk_ids=[20, 10])


def test_hybrid_search_reuses_compatible_faiss_without_rebuild(monkeypatch) -> None:
    built = []

    class FakeManager:
        def __init__(self, **_kwargs): pass
        def needs_rebuild(self, total_chunks, chunk_ids=None):
            assert total_chunks == 1
            assert chunk_ids == [7]
            return False
        def build_index(self, embeddings, chunk_ids):
            built.append((embeddings, chunk_ids))
            return True

    monkeypatch.setattr(hybrid_search_module, "FaissManager", FakeManager)
    search = object.__new__(HybridSearch)
    search._chunk_texts = ["text"]
    search._chunk_embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    search._chunk_meta = [{"chunk_id": 7}]
    search._faiss_enabled = True
    search.embedding_cache = SimpleNamespace(
        namespace="embedding-a",
        get_batch=lambda _texts: [np.asarray([1.0, 0.0], dtype=np.float32)],
    )
    search.stats = {"cache_hits": 0, "cache_misses": 0}
    search._system_profile = None
    search.vector_store = None
    search.faiss = None

    search._build_embeddings_and_faiss("index.bin", "meta.pkl")

    assert isinstance(search.faiss, FakeManager)
    assert built == []


def test_unavailable_external_vector_store_falls_back_to_local_embeddings() -> None:
    class BrokenVectorStore:
        def needs_rebuild(self, *_args, **_kwargs):
            raise ConnectionError("qdrant unavailable")

    search = object.__new__(HybridSearch)
    search._chunk_texts = ["text"]
    search._chunk_meta = [{"chunk_id": 7}]
    search._faiss_enabled = False
    search.embedding_cache = SimpleNamespace(
        namespace="embedding-a",
        get_batch=lambda _texts: [np.asarray([1.0, 0.0], dtype=np.float32)],
    )
    search.stats = {"cache_hits": 0, "cache_misses": 0}
    search._system_profile = None
    search.vector_store = BrokenVectorStore()
    search.faiss = None

    search._build_embeddings_and_faiss("index.bin", "meta.pkl")

    assert search.vector_store is None
    assert search._chunk_embeddings.tolist() == [[1.0, 0.0]]


def test_structured_output_parser_ignores_wrappers_and_trailing_braces() -> None:
    payload = extract_json_object(
        'Explanation before {"answer": "ok", "nested": {"x": 1}} after {bad}'
    )
    assert payload == {"answer": "ok", "nested": {"x": 1}}


def test_retrieval_profile_uses_shared_query_intent() -> None:
    intent = analyze_question("Сравни показатели за 10 и 11 июня 2025 года")
    assert intent.question_type == "comparison"
    assert profile_name_for_intent(intent) == "compare"
    assert IntentRouter(DEFAULT_CONFIG["intents"]).route(intent.raw_question).name == "compare"


def test_intent_components_return_one_canonical_understanding_result() -> None:
    direct = analyze_question("Какая выручка 5 августа 2025 года?")
    planned = QueryPlanner(planner_cfg={"enabled": False}).analyze(direct.raw_question)
    manager = ChatManager()
    managed = manager.analyze_question(direct.raw_question)

    assert isinstance(direct, QueryUnderstandingResult)
    assert isinstance(planned, QueryUnderstandingResult)
    assert isinstance(managed, QueryUnderstandingResult)
    assert QueryIntent is QueryUnderstandingResult
    assert direct.contract == planned.contract == managed.contract


def test_query_intents_cover_common_english_questions() -> None:
    assert analyze_question("Compare Alpha and Beta").question_type == "comparison"
    assert analyze_question("How many incidents are documented?").question_type == "count"
    assert analyze_question("Where was Service Alpha deployed?").question_type == "location"


def test_domain_actor_vocabulary_is_configurable_and_shared() -> None:
    lexicon = DomainLexicon.from_config(
        {
            "actor_aliases": ["service", "служба"],
            "family_equivalents": {"service": ["service", "служба"]},
        }
    )
    assert lexicon.mentions_family("Служба №42", "service")
    assert lexicon.family_matches("служба", "service")
    assert _parse_major_section("Service 42:")["number"] == "42"


def test_domain_profiles_cover_builtins_and_custom_vocabulary() -> None:
    assert {"general", "military_reports", "finance", "technical"}.issubset(BUILTIN_DOMAIN_PROFILES)
    assert get_domain_profile("Выручка составила 10 млн").indicators_in("Выручка составила 10 млн") == ("revenue",)
    custom = get_domain_profile(
        config={
            "name": "custom",
            "custom": {"metric_aliases": {"temperature": ["температура"]}},
        }
    )
    assert custom.indicators_in("Температура 20 C") == ("temperature",)


def test_fact_actions_and_polarity_come_from_domain_profile() -> None:
    assert analyze_question("Был ли поражен объект Альфа?").contract.actions == ("struck",)
    assert _extract_action("Система была активирована.") == "activated"
    assert _detect_polarity("Система была активирована.") == "positive"


def test_provider_contracts_are_structural_and_server_adapters_stream() -> None:
    class FakeEmbedding:
        def encode(self, texts, **_kwargs):
            return np.zeros((len(texts), 2), dtype=np.float32)

    class FakeVector:
        def needs_rebuild(self, total_items, *, item_ids): return False
        def rebuild(self, vectors, item_ids): return True
        def search(self, query_vector, top_k): return ([1], [0.9])

    class FakeLexical:
        def build(self, texts, item_ids): return None
        def search(self, query, top_k): return {1: 1.0}

    class FakeReranker:
        def rerank(self, query, candidates, texts_by_id): return list(candidates)

    class FakeLLM:
        def is_available(self): return True
        def generate_clean_answer(self, question, **kwargs): return question
        def generate_with_callback(self, *, messages, callback, **kwargs): return ""

    assert isinstance(FakeEmbedding(), EmbeddingProvider)
    assert isinstance(FakeVector(), VectorStore)
    assert isinstance(FakeLexical(), LexicalRetriever)
    assert isinstance(FakeReranker(), Reranker)
    assert isinstance(FakeLLM(), LLMProvider)

    class FakeTEI:
        def embed(self, *, inputs): return [[1.0, 2.0] for _ in inputs]

    assert TEIEmbeddingProvider(FakeTEI()).encode(["a"]).shape == (1, 2)

    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="one "))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="two"))]),
    ]
    completions = SimpleNamespace(create=lambda **_kwargs: iter(chunks))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    emitted = []
    result = VLLMProvider(client, "model").generate_with_callback(
        messages=[{"role": "user", "content": "q"}],
        callback=lambda token, final: emitted.append((token, final)),
    )
    assert result == "one two"
    assert emitted[-1] == ("", True)

    answer = VLLMProvider(client, "model").generate_clean_answer(
        question="q",
        context="evidence",
        rag_mode=True,
        streaming_callback=False,
    )
    assert answer == "one two"


def test_qdrant_adapter_invalidates_same_size_collection_with_other_items() -> None:
    info = SimpleNamespace(
        points_count=2,
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=2))),
    )
    stored = SimpleNamespace(payload={"chatkodi_namespace": "emb-v1", "chatkodi_item_digest": "wrong"})
    client = SimpleNamespace(
        get_collection=lambda _name: info,
        scroll=lambda **_kwargs: ([stored], None),
    )
    store = QdrantVectorStore(client, "chunks", namespace="emb-v1", expected_dimension=2)

    assert store.needs_rebuild(2, item_ids=[10, 20])


def test_token_budget_guard_uses_actual_collected_context(monkeypatch) -> None:
    profile = SimpleNamespace(
        has_cuda=False,
        gpu_vram_gb=0.0,
        total_ram_gb=16.0,
        context_char_limit=24000,
    )
    monkeypatch.setattr(memory_guard, "detect_system_profile", lambda: profile)
    # build_release.ps1 sets these for its smoke steps; they raise the budget.
    monkeypatch.delenv("CHATKODI_FORCE_LLM", raising=False)
    monkeypatch.delenv("CHATKODI_ALLOW_HEAVY_CPU_LLM", raising=False)
    plan = memory_guard.plan_rag_execution(
        question_type="summary",
        relevant_docs=1,
        chunk_results=2,
        total_chunks=20,
        requested_context_chars=20000,
    )

    memory_guard.apply_actual_context_guard(plan, actual_context_chars=1000, question_chars=100)
    assert plan.llm_allowed
    assert plan.estimated_prompt_tokens < 1000

    # A heavy context is trimmed to the hardware budget; the LLM stays allowed.
    memory_guard.apply_actual_context_guard(plan, actual_context_chars=12000, question_chars=100)
    assert plan.llm_allowed
    assert plan.context_trimmed
    assert plan.reason == "token_budget_trim"
    assert plan.estimated_prompt_tokens <= plan.prompt_token_budget
    assert plan.max_context_chars < 12000


def test_large_corpus_does_not_forbid_llm_on_small_gpu(monkeypatch) -> None:
    profile = SimpleNamespace(has_cuda=True, gpu_vram_gb=6.0, total_ram_gb=15.6, context_char_limit=24000)
    monkeypatch.setattr(memory_guard, "detect_system_profile", lambda: profile)
    monkeypatch.delenv("CHATKODI_FORCE_LLM", raising=False)
    monkeypatch.delenv("CHATKODI_ALLOW_HEAVY_CPU_LLM", raising=False)
    plan = memory_guard.plan_rag_execution(
        question_type="fact",
        relevant_docs=40,
        chunk_results=120,
        total_chunks=5000,
        requested_context_chars=12000,
    )
    memory_guard.apply_actual_context_guard(plan, actual_context_chars=1004, question_chars=40)
    assert plan.llm_allowed
    assert plan.reason != "token_budget_guard"


def test_windows_pagefile_error_has_specific_llm_unavailability_reason() -> None:
    error = OSError("The paging file is too small for this operation to complete")
    error.winerror = 1455

    reason, guidance = classify_llm_unavailability(error)

    assert reason == "insufficient_commit_memory"
    assert "pagefile" in guidance.lower()


def test_cpu_model_auto_limit_uses_current_ram_and_commit_headroom() -> None:
    healthy = safe_cpu_model_size_gb(32.0, available_ram_gb=20.0, available_commit_gb=24.0)
    pressured = safe_cpu_model_size_gb(32.0, available_ram_gb=2.0, available_commit_gb=3.0)

    assert healthy == 12.0
    assert pressured == pytest.approx(1.65)
    assert pressured < healthy


def test_llm_disabled_state_is_distinct_from_load_failure(monkeypatch) -> None:
    monkeypatch.setenv("CHATKODI_DISABLE_LLM", "1")

    assert runtime_module.load_llm_model() is None
    status = get_last_llm_load_status()
    assert status["available"] is False
    assert status["reason"] == "user_disabled"


def test_learned_reranker_reorders_only_configured_shortlist(tmp_path) -> None:
    model_dir = tmp_path / "reranker"
    model_dir.mkdir()

    class FakeCrossEncoder:
        def predict(self, pairs, show_progress_bar=False):
            assert pairs == [("query", "first"), ("query", "second")]
            return [0.05, 0.95]

    cfg = RerankConfig(
        learned_enabled=True,
        learned_model_path=str(model_dir),
        learned_topn=2,
        learned_weight=0.8,
    )
    reranker = CrossEncoderReranker(cfg, model_factory=lambda _path: FakeCrossEncoder())
    scored = [((1, 0.9), 1), ((1, 0.8), 2), ((0, 0.99), 3)]

    result = reranker.rerank("query", scored, {1: "first", 2: "second", 3: "tail"})

    assert [cid for _, cid in result] == [2, 1, 3]


def test_chat_manager_exposes_only_matching_cached_search_result() -> None:
    manager = ChatManager()
    manager._last_search_question = "question one"
    manager._last_search_result = {"results": [(1.0, {"doc_id": "a"})]}
    assert manager.get_last_search_result("question one")["results"]
    assert manager.get_last_search_result("question two") == {}


def test_model_hash_manifest_detects_same_size_tampering(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    weights = model / "model.safetensors"
    weights.write_bytes(b"a" * 2048)

    create_model_hash_manifest(model)
    assert verify_model_hash_manifest(model)["ok"] is True

    weights.write_bytes(b"b" * 2048)
    report = verify_model_hash_manifest(model)
    assert report["ok"] is False
    assert any("SHA-256 mismatch" in error for error in report["errors"])


def test_partial_model_is_never_reported_as_available(tmp_path) -> None:
    model = tmp_path / "partial-model"
    model.mkdir()
    (model / "offline_config.json").write_text(
        '{"partial_download": true}',
        encoding="utf-8",
    )
    downloader = object.__new__(SmartModelDownloader)
    valid, reason = downloader._check_model_integrity(str(model), "llm")
    assert valid is False
    assert "Частичная загрузка" in reason


def test_metadata_only_model_is_never_reported_as_available(tmp_path) -> None:
    model = tmp_path / "metadata-only-model"
    model.mkdir()
    metadata = '{"model_type": "test", "padding": "' + ("x" * 100) + '"}'
    (model / "config.json").write_text(metadata, encoding="utf-8")
    (model / "tokenizer.json").write_text(metadata, encoding="utf-8")
    downloader = object.__new__(SmartModelDownloader)

    valid, reason = downloader._check_model_integrity(str(model), "llm")

    assert valid is False
    assert "model weight files are missing" in reason


def test_malformed_hash_manifest_fails_without_raising(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"w" * 2048)
    (model / ".chatkodi-sha256.json").write_text(
        '{"algorithm":"sha256","files":{"model.safetensors":{"size":"bad","sha256":"bad"}}}',
        encoding="utf-8",
    )

    report = verify_model_hash_manifest(model)

    assert report["ok"] is False
    assert any("invalid hash record" in error for error in report["errors"])


def test_model_payload_requires_real_tokenizer(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"w" * 2048)

    report = inspect_hf_model_payload(model, require_tokenizer=True)

    assert report["ok"] is False
    assert "missing tokenizer files" in report["errors"]


def test_document_import_chooses_non_destructive_destination(tmp_path) -> None:
    source = tmp_path / "source" / "report.docx"
    source.parent.mkdir()
    source.write_bytes(b"new")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "report.docx").write_bytes(b"existing")

    destination = ChatKodiDocumentsMixin._available_destination(docs_dir, source)

    assert destination.name == "report (2).docx"
    assert (docs_dir / "report.docx").read_bytes() == b"existing"


def test_chat_history_is_bounded_and_exports_utf8(tmp_path) -> None:
    store = ChatHistoryStore(tmp_path / "history.jsonl", max_entries=10)
    for index in range(12):
        store.append("user", f"Вопрос {index}", "rag")

    entries = store.load()
    exported = store.export(tmp_path / "dialog.txt", entries)

    assert len(entries) == 10
    assert entries[0]["text"] == "Вопрос 2"
    assert "Вопрос 11" in exported.read_text(encoding="utf-8")


def test_search_accuracy_dataset_is_loadable_by_benchmark() -> None:
    cases_path = Path(__file__).resolve().parents[1] / "search_accuracy_cases.json"
    cases = _load_cases(cases_path)

    assert len(cases) >= 10
    assert all(str(case.get("question") or "").strip() for case in cases)
    assertion_keys = ("must_contain", "must_not_contain", "expected_sources", "forbidden_sources", "expected_event_count")
    assert all(any(key in case for key in assertion_keys) for case in cases)


def test_streaming_worker_cancel_reaches_llm() -> None:
    class FakeLlm:
        def __init__(self):
            self.cancelled = False

        def cancel_generation(self):
            self.cancelled = True

    llm = FakeLlm()
    manager = SimpleNamespace(llm_model=llm)
    worker = StreamingWorker(manager, "question", ChatMode.CHAT)

    worker.cancel()

    assert worker._cancel_requested.is_set()
    assert llm.cancelled is True


def test_rag_worker_does_not_replay_completed_answer_as_fake_tokens() -> None:
    class FakeManager:
        llm_model = None

        def set_mode(self, _mode):
            return None

        def process_message(self, _question, forced_mode=None):
            assert forced_mode == ChatMode.RAG
            return "validated final answer"

    worker = StreamingWorker(FakeManager(), "question", ChatMode.RAG)
    tokens = []
    finished = []
    worker.token_received.connect(tokens.append)
    worker.generation_finished.connect(finished.append)

    worker.run()

    assert tokens == []
    assert finished == ["validated final answer"]


def test_chat_worker_forwards_runtime_stream_chunks() -> None:
    class Stream:
        def get_stream(self):
            yield "first "
            yield "second"

    class FakeLlm:
        def generate_clean_answer(self, **kwargs):
            assert kwargs["streaming_callback"] is True
            return Stream()

    manager = SimpleNamespace(llm_model=FakeLlm(), set_mode=lambda _mode: None)
    worker = StreamingWorker(manager, "question", ChatMode.CHAT)
    tokens = []
    finished = []
    worker.token_received.connect(tokens.append)
    worker.generation_finished.connect(finished.append)

    worker.run()

    assert tokens == ["first ", "second"]
    assert finished == ["first second"]


class _FakeSseStream:
    """Stands in for core.llm.server.SseStream: yields OpenAI-style chunks."""

    def __init__(self, pieces, callback_seen=None):
        self.pieces = pieces
        self.callback_seen = callback_seen
        self.closed = False

    def __iter__(self):
        for index, piece in enumerate(self.pieces):
            if index and self.callback_seen is not None:
                assert self.callback_seen.wait(timeout=2), "callback did not run during generation"
            yield {"choices": [{"delta": {"content": piece}}]}

    def close(self):
        self.closed = True


class _FakeServer:
    def __init__(self, stream=None):
        self._stream = stream
        self.requests = []

    def is_running(self):
        return True

    def stream(self, path, payload):
        self.requests.append((path, payload))
        assert self._stream is not None, "no request expected"
        return self._stream


def _bare_llm(server) -> LocalLLM:
    llm = object.__new__(LocalLLM)
    llm.model_loaded = True
    llm.server = server
    llm.device = "cpu"
    llm._generation_cancel_event = threading.Event()
    llm._external_generation_cancel_event = None
    llm._generation_lock = threading.RLock()
    llm.stats = {
        "total_queries": 0,
        "successful": 0,
        "failed": 0,
        "avg_response_time": 0.0,
    }
    return llm


def test_generate_with_callback_streams_before_model_finishes() -> None:
    callback_seen = threading.Event()
    stream = _FakeSseStream(["first ", "second"], callback_seen)
    server = _FakeServer(stream)
    llm = _bare_llm(server)

    chunks = []

    def callback(text: str, is_final: bool) -> None:
        chunks.append((text, is_final))
        if text == "first ":
            callback_seen.set()

    result = llm.generate_with_callback(
        [{"role": "user", "content": "hello"}],
        callback,
    )

    assert result == "first second"
    assert chunks == [("first ", False), ("second", False), ("", True)]
    assert stream.closed
    path, payload = server.requests[0]
    assert path == "/v1/chat/completions"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_request_cancel_before_model_start_is_not_cleared() -> None:
    external_cancel = threading.Event()
    external_cancel.set()
    server = _FakeServer(stream=None)
    llm = _bare_llm(server)
    llm.attach_generation_cancel_event(external_cancel)

    chunks = []
    result = llm.generate_with_callback(
        [{"role": "user", "content": "hello"}],
        lambda text, is_final: chunks.append((text, is_final)),
    )

    assert result == ""
    assert chunks == [("", True)]
    assert server.requests == []
    assert external_cancel.is_set()
    llm.detach_generation_cancel_event(external_cancel)
    assert llm._external_generation_cancel_event is None


def test_fit_messages_keeps_question_when_documents_are_trimmed() -> None:
    from core.llm.runtime import _fit_messages

    question = "ВОПРОС: какая выручка?"
    messages = [{"role": "user", "content": "А" * 5000 + "\n" + question}]
    fitted = _fit_messages(messages, 2000)

    assert len(fitted[0]["content"]) <= 2000 + 3
    assert fitted[0]["content"].endswith(question)


def test_llm_refusal_names_memory_policy_not_unavailability() -> None:
    manager = object.__new__(ChatManager)
    manager.llm_model = SimpleNamespace(is_available=lambda: True)
    message = manager._llm_unavailable_reason(
        llm_ready=True,
        llm_allowed_by_plan=False,
        memory_plan={"reason": "cpu_low_ram"},
        intent=SimpleNamespace(llm_presentation_allowed=None),
    )
    assert "LLM загружена" in message
    assert "cpu_low_ram" in message
    assert "недоступна" not in message
