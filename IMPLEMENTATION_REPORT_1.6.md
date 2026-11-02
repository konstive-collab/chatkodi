# ChatKodi 1.6 — отчёт по внедрению

Дата сборки: 2026-04-08

## Что реализовано в этой доработке

### Retrieval Core 2.0
- добавлен `RetrievalSession` (`core/query_session.py`) для кеширования данных одного запроса;
- добавлен `LexicalPrefilterIndex` (`core/prefilter_index.py`) и интеграция shortlist-prefilter в `core/search.py`;
- добавлена каноническая нормализация якорей и идентификаторов (`core/anchor_normalizer.py`);
- исправлен regression-risk: добавлен совместимый метод `_normalize_identifier()` в `HybridSearch`;
- `BM25` и semantic retrieval теперь умеют работать по shortlist и переиспользовать готовый embedding запроса;
- добавлена отдельная fusion-логика с поддержкой `weighted`, `rrf`, `weighted_rrf`.

### Chunking / corpus cleaning
- `core/document_processor.py` переписан под structural/sentence-aware chunking;
- добавлено удаление повторяющихся headers/footers;
- чанки теперь несут block metadata: `block_type`, `page_num`, `section_title`, `is_header_like`, `is_table_like`;
- добавлен fallback для коротких документов, чтобы они не терялись из поиска.

### RAG orchestration
- `core/chat_mode.py` переведён на выбор стратегии:
  - `extractive_only`
  - `chunk_rag`
  - `full_doc_rag`
- добавлен fast-path через `core/extractive_answer.py` для простых фактологических ответов.

### LLM runtime
- `core/local_llm.py` получил token-aware trimming контекста;
- `generate_with_callback()` переведён на настоящий streaming через `TextIteratorStreamer` с безопасным fallback для сред без `transformers`.

### Eval / tests / diagnostics
- добавлен `core/eval_runner.py`;
- добавлен набор unit + smoke tests в `tests/`;
- в `hybrid_search()` теперь возвращается `prefilter_info`.

## Изменённые и добавленные файлы

### Новые
- `core/anchor_normalizer.py`
- `core/query_session.py`
- `core/prefilter_index.py`
- `core/extractive_answer.py`
- `core/eval_runner.py`
- `tests/test_anchor_normalizer.py`
- `tests/test_prefilter_index.py`
- `tests/test_document_processor.py`
- `tests/test_extractive_answer.py`
- `tests/test_search_pipeline.py`
- `tests/test_local_llm_trim.py`

### Изменённые
- `core/search.py`
- `core/document_processor.py`
- `core/chat_mode.py`
- `core/local_llm.py`
- `core/config.py`
- `core/__init__.py`

## Что прогнано

Команда:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

Результат:
- 8 тестов
- 8 passed
- статус: OK

Подробный лог: `../chatkodi_v1.6_TEST_LOG.txt`

## Что сделано частично, а не полностью по roadmap

Эта сборка закрывает ключевые архитектурные пункты roadmap, но **не закрывает на 100% весь документ целиком**.

### Реализовано частично
- `RRF / weighted_rrf` добавлены на уровне fusion-логики, но без отдельного большого eval-бенчмарка на реальном корпусе;
- `EvalRunner` есть, но полноценный production eval-set на 120–200 вопросов пока не собран автоматически;
- `true streaming` внедрён на уровне runtime, но без проверки в реальном GUI под каждой моделью;
- `intent-aware routing` сделан в orchestration-слое, но не было глубокого рефакторинга `intent_router.py`.

### Не закрыто полностью в этой итерации
- полноценный cross-encoder reranker;
- sentence-window second pass как отдельный модуль rerank-пайплайна;
- UI debug widgets / evidence widgets;
- migration-ready adapter для Qdrant;
- полный benchmark before/after на большом реальном корпусе.

## Практический итог

Это уже не косметический патч 1.5, а рабочий апгрейд ядра 1.6:
- меньше повторных retrieval-пересчётов;
- лучше поиск по `изделие-51 / изделием 51 / объект №51`;
- меньше шансов потерять короткий документ;
- лучше candidate selection через prefilter;
- RAG меньше гоняет LLM там, где можно ответить extractive fast-path'ом;
- контекст режется по токенам, а не только по символам.
