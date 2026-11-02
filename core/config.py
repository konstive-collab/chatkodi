"""
core/config.py

Единая конфигурация ChatKodi.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from core.logging_utils import get_logger
from core.paths import resolve_project_path
from core.system_profile import apply_profile_overrides, detect_system_profile


logger = get_logger(__name__)


DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "2.0.0",
    "retrieval": {
        "adaptive_enabled": True,
        "k_base": 60,
        "k_step": 40,
        "k_max": 320,
        "time_budget_ms": 180,
        "min_new_candidates": 25,
        "topn_stability": 20,
        "stability_overlap": 0.9,
        "score_improve_eps": 0.01,
        "bm25_k": 320,
        "semantic_k": 320,
        "bm25_weight": 0.35,
        "semantic_weight": 0.65,
        "per_doc_chunk_cap": 3,
        "max_results_cap": 60,
        "answer_rescue_docs": 4,
    },
    "context": {
        "max_docs": 4,
        "max_chars": 18000,
        "chunks_per_doc": 2,
    },
    "context_v2": {
        "enabled": True,
        "max_docs": 6,
        "max_chars": 32000,
        "window_before": 1,
        "window_after": 1,
        "max_chunks_per_doc": 5,
        "prefer_anchor_chunks": True,
        "allow_full_doc_fallback": True,
        "full_doc_max_chars": 7000,
    },
    "rerank": {
        "enabled": True,
        "dedup_enabled": True,
        "dedup_shingle_size": 6,
        "dedup_threshold": 0.92,
        "dedup_max_compare": 300,
        "rerank_topn": 220,
        "anchor_boost": 0.1,
        "date_boost": 0.12,
        "id_boost": 0.15,
        "entity_boost": 0.06,
        "prefer_diversity": True,
        "diversity_doc_penalty": 0.1,
        "max_chunks_per_doc_in_top": 4,
        "learned_enabled": False,
        "learned_model_path": "",
        "learned_topn": 40,
        "learned_weight": 0.35,
    },
    "intents": {
        "enabled": True,
        "profiles": {
            "fact_lookup": {
                "description": "Короткий факт/значение/ответ в 1-2 предложения",
                "retrieval_k_base": 60,
                "retrieval_k_max": 320,
                "context_max_docs": 4,
                "context_max_chars": 18000,
                "date_strict": False,
            },
            "date_fact": {
                "description": "Факт на конкретную дату/в диапазоне",
                "retrieval_k_base": 90,
                "retrieval_k_max": 520,
                "context_max_docs": 6,
                "context_max_chars": 28000,
                "date_strict": True,
            },
            "id_fact": {
                "description": "Факт про конкретный ID/номер",
                "retrieval_k_base": 110,
                "retrieval_k_max": 600,
                "context_max_docs": 7,
                "context_max_chars": 32000,
                "date_strict": False,
            },
            "summary": {
                "description": "Сводка или пересказ",
                "retrieval_k_base": 130,
                "retrieval_k_max": 700,
                "context_max_docs": 8,
                "context_max_chars": 38000,
                "date_strict": False,
            },
            "compare": {
                "description": "Сравнение дат или объектов",
                "retrieval_k_base": 150,
                "retrieval_k_max": 800,
                "context_max_docs": 9,
                "context_max_chars": 42000,
                "date_strict": True,
            },
        },
    },
    "ui": {
        "preview_chars": 700,
        "extract_docs_min": 1,
        "extract_docs_max": 20,
        "extract_docs_rel_threshold": 0.2,
        "extract_docs_abs_threshold": 0.12,
        "extract_docs_request_k": 30,
        "extract_chunks_per_doc": 2,
        "extract_chunk_rel_threshold": 0.55,
        "extract_chunk_abs_threshold": 0.22,
        "theme": "classic",
        "animate_ui": True,
    },
    "history": {
        "enabled": True,
        "path": "chat_history.jsonl",
        "max_entries": 200,
    },
    "domain_lexicon": {
        "actor_aliases": [
            "рбпла", "бпла", "шгр", "дшб", "одшбр", "омсбр", "мсбр", "гр",
            "группа", "расчет", "расчёт", "подразделение",
            "uav", "drone", "group", "team", "crew", "unit", "department", "service", "device", "system"
        ],
        "family_equivalents": {
            "рбпла": ["рбпла", "бпла", "uav"]
        },
    },
    "domain_profile": {
        "name": "military_reports",
        "custom": {},
    },
    "llm_runtime": {
        "rag_fast_max_tokens": 220,
        "rag_full_max_tokens": 360,
        "chat_max_tokens": 320,
        "low_resource_context_chars": 9000,
        "balanced_context_chars": 12000,
        "high_context_chars": 16000,
        "rag_allow_lazy_load": False,
        "chat_allow_lazy_load": False,
    },
    "generation": {
        "enabled": True,
        "catalog_root": "documents",
        "allow_creative_exercises": True,
        "label_creative_additions": True,
        "legal_template_formats": [".docx"],
        "block_unresolved_legal_fields": True,
        "legal_patch_max_tokens": 1200,
        "legal_context_limit": 24000,
        "sport_max_tokens": 560,
        "sport_context_limit": 14000,
    },
    "explainable": {
        "enabled": True,
        "max_evidence_items": 12,
    },
    "structured_answers": {
        "enabled": True,
        "max_docs": 8,
        "max_evidence_items": 3,
        "prefer_algorithmic_for_facts": True
    },
    "grounded_reasoning": {
        "enabled": True,
        "strict_validation": True,
        "max_evidence_items": 18,
        "max_evidence_chars": 18000,
        "max_item_chars": 2200,
        "max_tokens": 520,
        "retry_invalid_json": True
    },
    "memory_guard": {
        "enabled": True,
        "force_chunk_mode_on_large_corpus": True,
        "force_structured_for_fact_questions": True
    },
    "event_store": {
        "max_sync_chunks": 1200,
        "max_sync_seconds": 10
    },
    "query_planner": {
        "enabled": True,
        "use_llm_router": False,
        "min_confidence": 0.72,
        "max_tokens": 160,
        "context_limit": 1800,
        "temperature": 0.05,
        "use_on_simple_questions": False,
        "timeout_seconds": 2.5,
        "cache_size": 128
    },
    "features": {
        "plugin_loader_enabled": False,
        "diagnostics_enabled": True,
        "healthcheck_cli_enabled": True,
        "telemetry_enabled": False,
        "benchmark_cli_enabled": True,
    },
    "diagnostics": {
        "enabled": True,
        "trace_search": True,
        "trace_extract": True,
        "trace_dir": ".chatkodi/logs/diagnostics",
        "max_preview_chars": 240,
        "max_trace_items": 12,
    },
    "plugins": {
        "enabled": False,
        "directory": "plugins",
        "auto_discover": True,
    },
    "telemetry": {
        "enabled": False,
        "log_dir": ".chatkodi/logs/telemetry",
        "benchmark_dir": ".chatkodi/logs/benchmarks",
        "events_file": "events.jsonl",
        "capture_question": True,
        "capture_response_preview": True,
        "capture_context_preview": False,
        "max_preview_chars": 500,
        "include_environment": True,
        "session_separate_file": True
    },
    "runtime_profiles": {
        "auto_detect": True,
        "force_profile": "",
        "auto_apply_limits": True
    },
    "resilience": {
        "enabled": True,
        "safe_mode_enabled": True,
        "continue_on_optional_stage_failure": True,
        "llm_startup_mode": "auto",
        "record_incidents": True,
        "incident_dir": ".chatkodi/logs/recovery",
        "auto_disable_llm_after_failure": True,
        "auto_apply_runtime_hints": True,
        "max_llm_failures": 2
    },
}


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Рекурсивно объединяет словарь ``src`` со словарём ``dst``."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


@dataclass
class ChatKodiConfig:
    """Объект с итоговой конфигурацией проекта после объединения дефолтов и файла пользователя."""

    data: Dict[str, Any] = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_CONFIG)))
    path: Path = field(default_factory=lambda: resolve_project_path(os.environ.get("CHATKODI_CONFIG", "chatkodi_config.json")))

    def load(self) -> "ChatKodiConfig":
        """Загружает файл конфигурации и накладывает его поверх встроенных значений по умолчанию."""
        if not self.path.exists():
            return self
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            merged = json.loads(json.dumps(DEFAULT_CONFIG))
            _deep_update(merged, raw)
            profile = detect_system_profile(merged)
            self.data = apply_profile_overrides(merged, profile)
        except Exception as exc:
            logger.warning(
                "Failed to load config %s; using built-in defaults: %s",
                self.path,
                exc,
            )
            merged = json.loads(json.dumps(DEFAULT_CONFIG))
            profile = detect_system_profile(merged)
            self.data = apply_profile_overrides(merged, profile)
        return self

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Возвращает одно значение из указанного верхнеуровневого раздела конфига."""
        return (self.data.get(section) or {}).get(key, default)

    def section(self, name: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Возвращает целый раздел конфигурации как словарь."""
        value = self.data.get(name)
        if isinstance(value, dict):
            return value
        return dict(default or {})

    def flag(self, section: str, key: str, default: bool = False) -> bool:
        """Возвращает булевый флаг из конфига с безопасным преобразованием типов."""
        value = self.get(section, key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
        return bool(value)


_CONFIG_SINGLETON: ChatKodiConfig | None = None


def get_config() -> ChatKodiConfig:
    """Возвращает лениво создаваемый глобальный экземпляр конфигурации для текущего процесса."""
    global _CONFIG_SINGLETON
    if _CONFIG_SINGLETON is None:
        _CONFIG_SINGLETON = ChatKodiConfig().load()
    return _CONFIG_SINGLETON
