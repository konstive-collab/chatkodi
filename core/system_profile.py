"""Автоопределение аппаратного профиля и безопасные runtime-подсказки.

Модуль ничего не знает о GUI или RAG-логике. Он только определяет
примерный класс машины и предлагает консервативные настройки, которые
можно безопасно применить без перелома поведения программы.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from dataclasses import dataclass, asdict
from typing import Any, Mapping


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on", "да"}


def _user_forced_llm() -> bool:
    return _env_flag("CHATKODI_FORCE_LLM") or _env_flag("CHATKODI_ALLOW_HEAVY_CPU_LLM")


@dataclass(frozen=True)
class SystemProfile:
    profile_name: str
    total_ram_gb: float
    cpu_cores: int
    machine: str
    platform_name: str
    has_cuda: bool
    gpu_name: str
    gpu_vram_gb: float
    llm_startup_mode: str
    context_char_limit: int
    prefer_lightweight_llm: bool
    notes: tuple[str, ...]
    available_ram_gb: float = 0.0
    available_commit_gb: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload

    @property
    def summary_line(self) -> str:
        parts = [
            f"profile={self.profile_name}",
            f"ram={self.total_ram_gb:.1f}GB",
            f"cpu={self.cpu_cores}",
            f"cuda={self.has_cuda}",
        ]
        if self.gpu_name:
            parts.append(f"gpu={self.gpu_name}")
        if self.gpu_vram_gb > 0:
            parts.append(f"vram={self.gpu_vram_gb:.1f}GB")
        if self.available_ram_gb > 0:
            parts.append(f"ram_free={self.available_ram_gb:.1f}GB")
        if self.available_commit_gb > 0:
            parts.append(f"commit_free={self.available_commit_gb:.1f}GB")
        parts.append(f"llm={self.llm_startup_mode}")
        parts.append(f"context={self.context_char_limit}")
        return " | ".join(parts)


def _get_total_ram_gb() -> float:
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return float(stat.ullTotalPhys) / (1024 ** 3)
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return float(pages * page_size) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _get_available_memory_gb() -> tuple[float, float]:
    """Return available physical RAM and commit/pagefile capacity."""
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return (
                    float(stat.ullAvailPhys) / (1024 ** 3),
                    float(stat.ullAvailPageFile) / (1024 ** 3),
                )
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        available_ram = float(memory.available) / (1024 ** 3)
        return available_ram, available_ram + float(swap.free) / (1024 ** 3)
    except Exception:
        return 0.0, 0.0


def safe_cpu_model_size_gb(
    total_ram_gb: float,
    available_ram_gb: float = 0.0,
    available_commit_gb: float = 0.0,
) -> float:
    """Conservative maximum on-disk model size for automatic CPU loading."""
    total = float(total_ram_gb or 0.0)
    if total <= 0:
        limit = 2.8
    elif total < 10:
        limit = 2.8
    elif total < 14:
        limit = 4.2
    elif total < 20:
        limit = 7.2
    else:
        limit = 12.0

    # Loading transiently needs more memory than the weights on disk. Free
    # physical RAM protects interactivity; free commit protects Windows from
    # WinError 1455 even when a pagefile exists but is nearly exhausted.
    if available_ram_gb > 0:
        limit = min(limit, max(0.4, float(available_ram_gb) * 0.90))
    if available_commit_gb > 0:
        limit = min(limit, max(0.4, float(available_commit_gb) * 0.55))
    return max(0.4, limit)


def _detect_cuda() -> tuple[bool, str, float]:
    try:
        from core.gpu_info import primary_gpu

        gpu = primary_gpu()
        if gpu is None:
            return False, "", 0.0
        return True, gpu.name, float(gpu.total_gb)
    except Exception:
        return False, "", 0.0


def _pick_profile(total_ram_gb: float, cpu_cores: int, has_cuda: bool, gpu_vram_gb: float) -> SystemProfile:
    notes: list[str] = []
    machine = platform.machine() or "unknown"
    platform_name = platform.platform()

    if has_cuda and gpu_vram_gb >= 10 and total_ram_gb >= 14:
        notes.extend([
            "GPU-профиль с ранней загрузкой LLM",
            "Контекст можно держать заметно шире",
        ])
        return SystemProfile(
            profile_name="desktop_full",
            total_ram_gb=total_ram_gb,
            cpu_cores=cpu_cores,
            machine=machine,
            platform_name=platform_name,
            has_cuda=True,
            gpu_name="",
            gpu_vram_gb=gpu_vram_gb,
            llm_startup_mode="eager",
            context_char_limit=22000,
            prefer_lightweight_llm=False,
            notes=tuple(notes),
        )

    if has_cuda:
        notes.extend([
            "GPU найден, но профиль оставлен умеренным",
            "Генерацию лучше поднимать осторожно на длинном контексте",
        ])
        return SystemProfile(
            profile_name="gpu_fast",
            total_ram_gb=total_ram_gb,
            cpu_cores=cpu_cores,
            machine=machine,
            platform_name=platform_name,
            has_cuda=True,
            gpu_name="",
            gpu_vram_gb=gpu_vram_gb,
            llm_startup_mode="eager",
            context_char_limit=16000,
            prefer_lightweight_llm=True,
            notes=tuple(notes),
        )

    if total_ram_gb and total_ram_gb <= 8.5:
        notes.extend([
            "Очень ограниченная память — лучше не грузить LLM на старте",
            "Приоритет: retrieval-first и мягкая деградация",
        ])
        return SystemProfile(
            profile_name="academic_safe",
            total_ram_gb=total_ram_gb,
            cpu_cores=cpu_cores,
            machine=machine,
            platform_name=platform_name,
            has_cuda=False,
            gpu_name="",
            gpu_vram_gb=0.0,
            llm_startup_mode="lazy",
            context_char_limit=6000,
            prefer_lightweight_llm=True,
            notes=tuple(notes),
        )

    if total_ram_gb and total_ram_gb <= 14:
        notes.extend([
            "CPU-only профиль со щадящими лимитами",
            "LLM оставлена ленивой, чтобы не тормозить старт",
        ])
        return SystemProfile(
            profile_name="balanced_cpu",
            total_ram_gb=total_ram_gb,
            cpu_cores=cpu_cores,
            machine=machine,
            platform_name=platform_name,
            has_cuda=False,
            gpu_name="",
            gpu_vram_gb=0.0,
            llm_startup_mode="lazy",
            context_char_limit=9000,
            prefer_lightweight_llm=True,
            notes=tuple(notes),
        )

    notes.extend([
        "Универсальный CPU-профиль",
        "LLM загружается по запросу, чтобы приложение гарантированно открылось",
    ])
    return SystemProfile(
        profile_name="desktop_cpu",
        total_ram_gb=total_ram_gb,
        cpu_cores=cpu_cores,
        machine=machine,
        platform_name=platform_name,
        has_cuda=False,
        gpu_name="",
        gpu_vram_gb=0.0,
        llm_startup_mode="lazy",
        context_char_limit=12000,
        prefer_lightweight_llm=True,
        notes=tuple(notes),
    )


def _apply_forced_profile(base: SystemProfile, forced: str | None) -> SystemProfile:
    forced = (forced or "").strip().lower()
    if not forced:
        return base
    if forced == base.profile_name:
        return base

    mapping = {
        "academic_safe": dict(profile_name="academic_safe", llm_startup_mode="lazy", context_char_limit=min(base.context_char_limit, 6000), prefer_lightweight_llm=True),
        "balanced_cpu": dict(profile_name="balanced_cpu", llm_startup_mode="lazy", context_char_limit=min(base.context_char_limit, 9000), prefer_lightweight_llm=True),
        "desktop_full": dict(profile_name="desktop_full", llm_startup_mode="eager", context_char_limit=max(base.context_char_limit, 22000), prefer_lightweight_llm=False),
        "gpu_fast": dict(profile_name="gpu_fast", llm_startup_mode="eager", context_char_limit=max(base.context_char_limit, 16000), prefer_lightweight_llm=True),
        "desktop_cpu": dict(profile_name="desktop_cpu", llm_startup_mode="lazy", context_char_limit=max(base.context_char_limit, 12000), prefer_lightweight_llm=True),
    }
    patch = mapping.get(forced)
    if not patch:
        return base
    return SystemProfile(
        profile_name=patch["profile_name"],
        total_ram_gb=base.total_ram_gb,
        cpu_cores=base.cpu_cores,
        machine=base.machine,
        platform_name=base.platform_name,
        has_cuda=base.has_cuda,
        gpu_name=base.gpu_name,
        gpu_vram_gb=base.gpu_vram_gb,
        llm_startup_mode=patch["llm_startup_mode"],
        context_char_limit=int(patch["context_char_limit"]),
        prefer_lightweight_llm=bool(patch["prefer_lightweight_llm"]),
        notes=tuple(list(base.notes) + [f"forced_profile={forced}"]),
        available_ram_gb=base.available_ram_gb,
        available_commit_gb=base.available_commit_gb,
    )


def detect_system_profile(config_data: Mapping[str, Any] | None = None) -> SystemProfile:
    config_data = config_data or {}
    runtime_cfg = config_data.get("runtime_profiles", {}) if isinstance(config_data, Mapping) else {}
    forced = os.environ.get("CHATKODI_PROFILE") or runtime_cfg.get("force_profile")

    total_ram_gb = _get_total_ram_gb()
    available_ram_gb, available_commit_gb = _get_available_memory_gb()
    cpu_cores = max(1, os.cpu_count() or 1)
    has_cuda, gpu_name, gpu_vram_gb = _detect_cuda()
    base = _pick_profile(total_ram_gb, cpu_cores, has_cuda, gpu_vram_gb)
    base = SystemProfile(
        profile_name=base.profile_name,
        total_ram_gb=base.total_ram_gb,
        cpu_cores=base.cpu_cores,
        machine=base.machine,
        platform_name=base.platform_name,
        has_cuda=base.has_cuda,
        gpu_name=gpu_name,
        gpu_vram_gb=base.gpu_vram_gb,
        llm_startup_mode=base.llm_startup_mode,
        context_char_limit=base.context_char_limit,
        prefer_lightweight_llm=base.prefer_lightweight_llm,
        notes=base.notes,
        available_ram_gb=available_ram_gb,
        available_commit_gb=available_commit_gb,
    )
    return _apply_forced_profile(base, str(forced or ""))


def apply_runtime_hints(profile: SystemProfile, config_data: Mapping[str, Any] | None = None) -> dict[str, str]:
    config_data = config_data or {}
    resil_cfg = config_data.get("resilience", {}) if isinstance(config_data, Mapping) else {}
    auto_hints = bool(resil_cfg.get("auto_apply_runtime_hints", True))
    applied: dict[str, str] = {}
    if not auto_hints:
        return applied

    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        applied["TOKENIZERS_PARALLELISM"] = "false"

    if not profile.has_cuda:
        threads = str(max(1, min(4, profile.cpu_cores)))
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            if key not in os.environ:
                os.environ[key] = threads
                applied[key] = threads

    return applied


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _scale_section_numbers(section: dict[str, Any], caps: dict[str, int]) -> None:
    for key, cap in caps.items():
        if key in section:
            section[key] = min(_safe_int(section.get(key), cap), int(cap))


def apply_profile_overrides(config_data: Mapping[str, Any], profile: SystemProfile) -> dict[str, Any]:
    """Возвращает копию конфига с адаптивными лимитами под CPU/GPU.

    Важная цель 1.7.7: одна сборка должна безопасно работать и на CPU-only
    машине с 8 GB RAM, и на GPU-ноутбуке/ПК. Поэтому здесь режутся не только
    размеры prompt-а, но и ширина retrieval, event-store sync и LLM-router.
    """
    import copy

    data = copy.deepcopy(dict(config_data))
    runtime_cfg = data.setdefault("runtime_profiles", {})
    if not bool(runtime_cfg.get("auto_apply_limits", True)):
        data["runtime_profile_resolved"] = profile.as_dict()
        return data

    is_cpu_only = not bool(profile.has_cuda)
    is_tiny_cpu = is_cpu_only and profile.total_ram_gb > 0 and profile.total_ram_gb <= 8.5
    is_low_ram = profile.total_ram_gb > 0 and profile.total_ram_gb <= 16.5
    is_low_vram = bool(profile.has_cuda and profile.gpu_vram_gb > 0 and profile.gpu_vram_gb <= 8.1)
    force_llm = _user_forced_llm()

    ctx = data.setdefault("context", {})
    ctx2 = data.setdefault("context_v2", {})
    llm = data.setdefault("llm_runtime", {})
    retrieval = data.setdefault("retrieval", {})
    rerank = data.setdefault("rerank", {})
    ui = data.setdefault("ui", {})
    planner = data.setdefault("query_planner", {})
    evidence = data.setdefault("evidence_budget", {})
    analytics = data.setdefault("analytics", {})
    memory_guard = data.setdefault("memory_guard", {})
    event_store = data.setdefault("event_store", {})

    resolved_limit = int(profile.context_char_limit)
    current_ctx = _safe_int(ctx.get("max_chars", resolved_limit), resolved_limit)
    ctx["max_chars"] = min(current_ctx, resolved_limit) if current_ctx > 0 else resolved_limit

    low_resource = max(2200, min(resolved_limit, 5200 if is_tiny_cpu else 7000))
    balanced = max(low_resource + 800, min(resolved_limit, 8500 if is_low_ram else 10000))
    high = max(balanced, resolved_limit)

    llm["low_resource_context_chars"] = min(_safe_int(llm.get("low_resource_context_chars", low_resource), low_resource), low_resource)
    llm["balanced_context_chars"] = min(_safe_int(llm.get("balanced_context_chars", balanced), balanced), balanced)
    llm["high_context_chars"] = min(_safe_int(llm.get("high_context_chars", high), high), high)

    # CPU-only 8 GB: режим "не падаем". Поиск и structured answers живут,
    # а генерация/LLM-router не грузят локальную LLM автоматически.
    if is_tiny_cpu:
        ctx.update({"max_docs": min(_safe_int(ctx.get("max_docs", 2), 2), 2), "max_chars": min(_safe_int(ctx.get("max_chars", 3500), 3500), 3500), "chunks_per_doc": 1})
        ctx2.update({
            "max_docs": min(_safe_int(ctx2.get("max_docs", 2), 2), 2),
            "max_chars": min(_safe_int(ctx2.get("max_chars", 4500), 4500), 4500),
            "max_chunks_per_doc": min(_safe_int(ctx2.get("max_chunks_per_doc", 2), 2), 2),
            "allow_full_doc_fallback": False,
            "full_doc_max_chars": min(_safe_int(ctx2.get("full_doc_max_chars", 2500), 2500), 2500),
        })
        retrieval.update({
            "adaptive_enabled": True,
            "k_base": min(_safe_int(retrieval.get("k_base", 40), 40), 40),
            "k_step": min(_safe_int(retrieval.get("k_step", 30), 30), 30),
            "k_max": min(_safe_int(retrieval.get("k_max", 180), 180), 180),
            "time_budget_ms": min(_safe_int(retrieval.get("time_budget_ms", 120), 120), 120),
            "bm25_k": min(_safe_int(retrieval.get("bm25_k", 180), 180), 180),
            "semantic_k": min(_safe_int(retrieval.get("semantic_k", 120), 120), 120),
            "max_results_cap": min(_safe_int(retrieval.get("max_results_cap", 35), 35), 35),
            "per_doc_chunk_cap": min(_safe_int(retrieval.get("per_doc_chunk_cap", 2), 2), 2),
        })
        _scale_section_numbers(rerank, {"rerank_topn": 90, "dedup_max_compare": 160, "max_chunks_per_doc_in_top": 2})
        ui["extract_docs_request_k"] = min(_safe_int(ui.get("extract_docs_request_k", 20), 20), 20)
        ui["extract_docs_max"] = min(_safe_int(ui.get("extract_docs_max", 12), 12), 12)
        ui["extract_chunks_per_doc"] = min(_safe_int(ui.get("extract_chunks_per_doc", 1), 1), 1)
        planner["use_llm_router"] = False
        planner["enabled"] = bool(planner.get("enabled", True))
        llm["rag_allow_lazy_load"] = False
        llm["chat_allow_lazy_load"] = bool(llm.get("chat_allow_lazy_load", False))
        memory_guard["cpu_low_ram_structured_only"] = True
        memory_guard["rag_llm_allowed_on_tiny_cpu"] = False
        event_store.update({"max_sync_chunks": min(_safe_int(event_store.get("max_sync_chunks", 900), 900), 900), "max_sync_seconds": min(_safe_int(event_store.get("max_sync_seconds", 8), 8), 8)})
        evidence.update({
            "range_max_docs": min(_safe_int(evidence.get("range_max_docs", 6), 6), 6),
            "range_chunks_per_doc": min(_safe_int(evidence.get("range_chunks_per_doc", 3), 3), 3),
            "stats_max_docs": min(_safe_int(evidence.get("stats_max_docs", 5), 5), 5),
            "stats_chunks_per_doc": min(_safe_int(evidence.get("stats_chunks_per_doc", 3), 3), 3),
            "exact_max_docs": min(_safe_int(evidence.get("exact_max_docs", 3), 3), 3),
            "exact_chunks_per_doc": min(_safe_int(evidence.get("exact_chunks_per_doc", 2), 2), 2),
        })
    elif is_cpu_only or is_low_ram or is_low_vram:
        # Умеренный профиль: не душим GPU/CPU полностью, но не даём RAG съесть всё.
        ctx["max_docs"] = min(_safe_int(ctx.get("max_docs", 4), 4), 4)
        ctx["max_chars"] = min(_safe_int(ctx.get("max_chars", 9000), 9000), 9000)
        ctx["chunks_per_doc"] = min(_safe_int(ctx.get("chunks_per_doc", 2), 2), 2)
        ctx2["max_docs"] = min(_safe_int(ctx2.get("max_docs", 4), 4), 4)
        ctx2["max_chars"] = min(_safe_int(ctx2.get("max_chars", 11000), 11000), 11000)
        ctx2["max_chunks_per_doc"] = min(_safe_int(ctx2.get("max_chunks_per_doc", 3), 3), 3)
        retrieval["k_base"] = min(_safe_int(retrieval.get("k_base", 60), 60), 60)
        retrieval["k_step"] = min(_safe_int(retrieval.get("k_step", 50), 50), 50)
        retrieval["k_max"] = min(_safe_int(retrieval.get("k_max", 320), 320), 320)
        retrieval["bm25_k"] = min(_safe_int(retrieval.get("bm25_k", 320), 320), 320)
        retrieval["semantic_k"] = min(_safe_int(retrieval.get("semantic_k", 260), 260), 260)
        retrieval["max_results_cap"] = min(_safe_int(retrieval.get("max_results_cap", 60), 60), 60)
        _scale_section_numbers(rerank, {"rerank_topn": 160, "dedup_max_compare": 240, "max_chunks_per_doc_in_top": 3})
        if is_cpu_only:
            planner["use_llm_router"] = False
            llm["rag_allow_lazy_load"] = False
        event_store.update({"max_sync_chunks": min(_safe_int(event_store.get("max_sync_chunks", 1800), 1800), 1800), "max_sync_seconds": min(_safe_int(event_store.get("max_sync_seconds", 15), 15), 15)})

    if force_llm:
        # Ручное желание пользователя важнее autosafe: даже на слабом CPU
        # ChatKodi должен дать возможность запустить LLM. Но лимиты поиска и
        # контекста остаются экономными, чтобы приложение не падало без причины.
        llm["rag_allow_lazy_load"] = True
        llm["chat_allow_lazy_load"] = True
        llm["force_user_requested_llm"] = True
        memory_guard["cpu_low_ram_structured_only"] = False
        memory_guard["rag_llm_allowed_on_tiny_cpu"] = True
        planner["use_llm_router"] = False

    analytics.setdefault("event_store_enabled", True)
    analytics.setdefault("structured_first", True)
    data["runtime_profile_resolved"] = profile.as_dict()
    return data
