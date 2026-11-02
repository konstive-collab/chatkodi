"""Совместимый фасад локальной LLM.

В ветке 1.8 часть кода ещё импортирует ``core.llm.model``, а основная
реализация исторически живёт в ``core.llm.runtime``. Чтобы не держать две
расходящиеся версии загрузчика, фасад просто переэкспортирует рабочую
реализацию. Это убирает скрытую проблему с несуществующим
``LocalLLMRuntimeMixin`` и делает поведение единым для всех входов.
"""

from .runtime import LocalLLM, StreamingCallback, load_llm_model

__all__ = ["LocalLLM", "StreamingCallback", "load_llm_model"]
