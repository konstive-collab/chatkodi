"""Оценивает безопасный режим RAG на текущем железе и объёме корпуса."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import os

from core.system_profile import detect_system_profile


@dataclass
class MemoryPlan:
    allow_full_documents: bool
    max_context_chars: int
    max_chunks_for_context: int
    force_structured_answer: bool
    llm_allowed: bool
    mode: str
    estimated_prompt_tokens: int
    reason: str = ""
    notes: List[str] = field(default_factory=list)
    prompt_token_budget: int = 0
    context_trimmed: bool = False


def estimate_prompt_tokens(chars: int) -> int:
    if chars <= 0:
        return 0
    return max(1, int(chars / 3.6))


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on", "да"}


def user_forced_llm() -> bool:
    """Возвращает True, когда пользователь явно разрешил LLM даже на слабом ПК.

    Это не отменяет экономию контекста: на 8 GB RAM мы всё равно режем prompt,
    но больше не говорим пользователю «нельзя». Если он осознанно включил LLM
    в лаунчере или передал env-флаг, ChatKodi обязан хотя бы попытаться загрузить
    модель в lazy-режиме.
    """
    return _env_flag("CHATKODI_FORCE_LLM") or _env_flag("CHATKODI_ALLOW_HEAVY_CPU_LLM")


def plan_rag_execution(*, question_type: str, relevant_docs: int, chunk_results: int, total_chunks: int, requested_context_chars: int = 12000) -> MemoryPlan:
    profile = detect_system_profile()
    forced_llm = user_forced_llm()
    low_vram = profile.has_cuda and profile.gpu_vram_gb > 0 and profile.gpu_vram_gb <= 8.1
    cpu_only = not bool(profile.has_cuda)
    tiny_cpu = cpu_only and profile.total_ram_gb > 0 and profile.total_ram_gb <= 8.5
    low_ram = profile.total_ram_gb > 0 and profile.total_ram_gb <= 16.5
    huge_corpus = total_chunks >= 1200 or relevant_docs >= 24
    medium_corpus = total_chunks >= 400 or relevant_docs >= 10 or chunk_results >= 80

    allow_full = True
    llm_allowed = True
    max_chars = min(int(profile.context_char_limit), int(requested_context_chars or profile.context_char_limit))
    max_chunks = 5
    structured_types = {"binary_fact", "location", "method", "count", "frequency", "timeline", "comparison", "range_summary"}
    force_structured = question_type in structured_types
    reason = "default"
    mode = "balanced"
    notes: List[str] = []

    if tiny_cpu:
        allow_full = False
        max_chars = min(max_chars, 3500)
        max_chunks = 2
        force_structured = True
        llm_allowed = False
        reason = "tiny_cpu_structured_only"
        mode = "safe_cpu"
        notes.append("CPU-only 8GB: RAG работает в structured-only режиме без автозагрузки LLM.")
    elif cpu_only and low_ram:
        allow_full = False
        max_chars = min(max_chars, 6000)
        max_chunks = 3
        force_structured = True if question_type in structured_types else force_structured
        llm_allowed = False if question_type in structured_types or medium_corpus else True
        reason = "cpu_low_ram"
        mode = "safe_cpu"
        notes.append("CPU-only/low RAM: контекст и LLM ограничены, приоритет structured-answer.")
    elif low_vram or low_ram:
        allow_full = False
        max_chars = min(max_chars, 9000 if low_vram else 8000)
        max_chunks = 4
        reason = "low_resource_profile"
        mode = "safe"
        notes.append("Снижен размер контекста из-за ограничений по железу.")

    if medium_corpus and not tiny_cpu:
        allow_full = False
        max_chars = min(max_chars, 8000)
        max_chunks = min(max_chunks, 4)
        reason = "medium_corpus" if reason == "default" else reason
        mode = "balanced" if mode == "balanced" else mode
        notes.append("Включен chunk-first режим для среднего корпуса.")

    if huge_corpus:
        allow_full = False
        max_chars = min(max_chars, 6000 if (tiny_cpu or low_ram or low_vram) else 8000)
        max_chunks = min(max_chunks, 3)
        force_structured = True
        reason = "huge_corpus" if reason == "default" else reason
        mode = "safe"
        notes.append("Для большого корпуса включен structured-first ответ.")

    if question_type in {"binary_fact", "location", "method"}:
        force_structured = True
        notes.append("Для фактологического вопроса приоритетен алгоритмический ответ.")

    if question_type == "summary" and not huge_corpus and not tiny_cpu:
        force_structured = False

    # The requested limit is not the prompt. The final estimate is filled by
    # apply_actual_context_guard after retrieval has assembled the real text.
    estimated_tokens = 0
    # Размер корпуса не влияет на размер промпта: контекст уже ограничен
    # max_context_chars и финальным бюджетом токенов. Поэтому большой корпус
    # больше не запрещает LLM — он только включает chunk-first и structured-first.
    if tiny_cpu:
        llm_allowed = False
    if forced_llm:
        # Важное продуктовое правило: слабый ПК не должен получать LLM "насильно"
        # в Авто-режиме, но если пользователь прямо включил LLM, мы не имеем
        # права блокировать её memory guard'ом. Вместо запрета оставляем
        # экономные лимиты контекста и предупреждаем в notes.
        llm_allowed = True
        notes.append("Пользователь явно разрешил LLM на слабом устройстве; контекст оставлен урезанным для стабильности.")

    return MemoryPlan(
        allow_full_documents=allow_full,
        max_context_chars=max(1800 if tiny_cpu else 3000, int(max_chars)),
        max_chunks_for_context=max(1 if tiny_cpu else 2, int(max_chunks)),
        force_structured_answer=bool(force_structured),
        llm_allowed=bool(llm_allowed),
        mode=mode,
        estimated_prompt_tokens=int(estimated_tokens),
        reason=reason,
        notes=notes,
    )


def prompt_token_budget(profile=None) -> int:
    """How many prompt tokens the current hardware can process in reasonable time.

    The budget trims the context; it never forbids the LLM. A 6-8 GB GPU with
    partial CPU offload still prefills ~3.5k tokens in seconds, CPU-only
    machines get less so the first token arrives in under a minute.
    """
    profile = profile or detect_system_profile()
    if user_forced_llm():
        return 6000
    vram = float(getattr(profile, "gpu_vram_gb", 0.0) or 0.0)
    ram = float(getattr(profile, "total_ram_gb", 0.0) or 0.0)
    if getattr(profile, "has_cuda", False):
        # llama.cpp on a GPU reads the prompt at ~3.5k tokens/s even on 8 GB
        # (the old 3.6k budget was for torch with CPU offload).
        return 7000
    if 0 < ram <= 8.5:
        return 1600
    if 0 < ram <= 16.5:
        return 2400
    return 4000


def apply_actual_context_guard(
    plan: MemoryPlan,
    *,
    actual_context_chars: int,
    question_chars: int = 0,
    prompt_overhead_chars: int = 800,
) -> MemoryPlan:
    """Finalize the token budget from the context that will really be sent.

    Earlier versions switched the LLM off when the estimate exceeded 2200
    tokens, even with the model loaded and a 1 KB context ("token_budget_guard").
    Now the plan keeps the LLM allowed and lowers ``max_context_chars`` so the
    caller trims the context to the budget instead.
    """
    question_chars = max(0, int(question_chars))
    overhead = max(0, int(prompt_overhead_chars))
    total_chars = max(0, int(actual_context_chars)) + question_chars + overhead
    plan.estimated_prompt_tokens = estimate_prompt_tokens(total_chars)
    budget = prompt_token_budget()
    plan.prompt_token_budget = budget
    if plan.estimated_prompt_tokens > budget:
        allowed_chars = max(1200, int(budget * 3.6) - question_chars - overhead)
        plan.max_context_chars = min(int(plan.max_context_chars or allowed_chars), allowed_chars)
        plan.context_trimmed = True
        plan.reason = "token_budget_trim"
        plan.estimated_prompt_tokens = estimate_prompt_tokens(plan.max_context_chars + question_chars + overhead)
        note = (
            f"Контекст сокращён до ~{budget} токенов под возможности железа; "
            "в промпт оставлены самые релевантные фрагменты."
        )
        if note not in plan.notes:
            plan.notes.append(note)
    return plan
