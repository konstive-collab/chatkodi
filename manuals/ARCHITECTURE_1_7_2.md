# ChatKodi 1.7.2 Architecture

## Центральная идея

Facts first. LLM last.

1. Вопрос разбирается на `QueryIntent`.
2. Retrieval собирает документы, чанки и события.
3. `EventStore` хранит on-disk события по чанкам и позволяет отвечать без тяжёлой генерации.
4. `StructuredAnswer` строится алгоритмически.
5. LLM используется только как слой переписывания/сводки, когда это безопасно по памяти.

## Главные контуры

- `core/query_understanding.py` — тип вопроса, объекты, даты, ключи.
- `core/fact_extraction.py` — извлечение событий из текста и evidence-records.
- `core/event_store.py` — SQLite-хранилище событий.
- `core/analytics/*` — статистика, timeline, паттерны, сравнение периодов.
- `core/answer_builders/structured.py` — answer builder по типу вопроса.
- `core/memory_guard.py` — защита от переполнения памяти.
- `core/chat/manager.py` — orchestration structured-first + RAG.

## Почему это полезно

- На 100+ документах в LLM уходит только узкий evidence/context.
- Ответы на фактологические вопросы не зависят от свободной генерации.
- Support bundle теперь содержит больше данных о structured-режиме и event store.
