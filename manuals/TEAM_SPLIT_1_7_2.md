# Team Split for ChatKodi 1.7.2

## Слой 1 — Query / Answers
Владелец: архитектор ядра

- `core/query_understanding.py`
- `core/answer_builders/*`
- `core/chat/manager.py`

## Слой 2 — Retrieval / Indexing / Cache
- `core/search_engine/*`
- `core/embedding_cache.py`
- `core/event_store.py`
- `core/documents/*`

## Слой 3 — Analytics
- `core/analytics/*`
- benchmark-кейсы

## Слой 4 — Tools / Platform
- `tools/bootstrap.py`
- `tools/doctor.py`
- `tools/support_bundle.py`

## Контракты
- `QueryIntent`
- `EvidenceRecord`
- `StructuredAnswer`
