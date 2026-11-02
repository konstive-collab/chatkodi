# ChatKodi 1.9.0 — отчёт стабилизации

Дата проверки: 2026-09-03. Итоговый статус: stable.

## Подтверждённые дефекты и первопричины

1. Adaptive retrieval повторно вызывал BM25 и повторно строил query embedding при каждом увеличении `k`. Причина: оба дорогих вызова находились внутри цикла. Теперь каждый сигнал вычисляется один раз, а цикл работает с растущими срезами уже рассчитанного рейтинга.
2. Entity filter раньше пропускал нерелевантный chunk из-за `pass` вместо исключения кандидата. Исправление уже присутствовало в rc2; добавлены и сохранены regression-тесты, повторного изменения кода не потребовалось.
3. В запросе не было отдельного контракта показателя. Поэтому выручка, прибыль и соседние даты могли конкурировать как обычные слова. Добавлен `QueryAnswerContract`: даты, объект, показатель, действие, polarity и тип ответа.
4. Ошибка keyword extractor могла оборвать поиск. BM25, semantic, exact ID, date/entity и keyword/fuzzy теперь деградируют независимо; сбой одного retriever не обнуляет остальные.
5. Exact ID проверялся во время прохода по кандидатам и не покрывал русские автомобильные номера. Добавлен индекс `normalized_identifier -> chunk_ids`, границы `11`/`111` не смешиваются, визуально эквивалентные кириллические/латинские символы номера нормализуются.
6. «Сколько дней работал специалист X?» считало факты, а не уникальные календарные дни. `EventStore` остаётся источником структурированных событий; итог считается программно по уникальным датам, дубликаты документов не увеличивают число дней, полное имя отделяет однофамильцев, конфликт positive/negative сохраняется в диагностике и ответе.
7. EvidenceReasoner проверял существование точной цитаты, но не соответствие вопросу. Теперь цитируемое evidence дополнительно проверяется по Query/Answer Contract: показатель, дата, объект и действие.
8. Query understanding назывался `QueryIntent` в нескольких слоях. Канонический runtime-тип теперь `QueryUnderstandingResult`; старое имя оставлено только как совместимый alias.
9. Предметная лексика была встроена в общий модуль. Добавлен `DomainProfile` с профилями `general`, `military_reports`, `finance`, `technical`, `custom`; новые показатели и действия разрешаются через профиль.
10. Token budget оценивался по настроенному максимуму до сборки контекста. Теперь guard принимает фактическую длину уже собранного context и только после этого разрешает или запрещает LLM.
11. Причины недоступности LocalLLM не различались. Добавлены коды `user_disabled`, `model_not_found`, `model_incomplete`, `dependency_missing`, `insufficient_ram`, `insufficient_vram`, `insufficient_commit_memory`, `load_failed`; Windows WinError 1455 получает отдельную рекомендацию по commit/pagefile.
12. FAISS мог быть переиспользован только по числу векторов. Проверяются также порядок chunk IDs, embedding namespace и размерность; совместимый индекс не перестраивается.
13. Старый streaming на rc1 буферизовал результат. В rc2 уже внедрён `TextIteratorStreamer`; тест подтверждает получение первого токена до завершения model.generate и сохранение ранней отмены запроса.

## Архитектурная граница

В `core/providers.py` определены независимые контракты:

- `EmbeddingProvider`;
- `VectorStore`;
- `LexicalRetriever`;
- `Reranker`;
- `LLMProvider`.

Desktop продолжает использовать локальный Transformers/SentenceTransformers, BM25, FAISS и LocalLLM. `integrations/server_providers.py` содержит изолированные адаптеры TEI, Qdrant и OpenAI-compatible vLLM. Retrieval, evidence filtering, EventStore aggregation и answer building остаются общими и не импортируют server SDK.

## Regression-тесты этой итерации

- один расчёт BM25/semantic на adaptive query;
- безопасная деградация keyword extractor;
- независимый отказ BM25 при работающем semantic;
- exact ID `11` против `111`, ID вместе с датой и кириллический госномер;
- Query/Answer Contract для выручки и даты;
- выручка против прибыли и соседней даты;
- проверка question contract в EvidenceReasoner;
- уникальные дни специалиста, дубликаты, однофамилец и противоречие;
- получение multi-document facts из EventStore;
- единый `QueryUnderstandingResult` в parser/planner/ChatManager;
- builtin/custom DomainProfile;
- provider protocols и настоящий stream адаптера vLLM;
- token guard по фактическому context;
- отдельная классификация WinError 1455;
- отличие явно отключённой LLM от ошибки загрузки;
- переиспользование совместимого FAISS без rebuild.

До этой итерации: 73 passed. После: 89 passed.

## Замеры

Синтетический corpus 50 000 chunks, один точный ID:

- прежний полный scan: 2122.974 ms;
- однократное построение ID index: 1444.696 ms;
- средний indexed lookup по 1000 запросам: 0.016915 ms;
- найден ровно один chunk.

Adaptive retrieval с тремя шагами `k=10/20/30` и одинаковой задержкой retriever:

- прежняя схема: 6 дорогих signal calls, 62.24 ms;
- новая схема: 2 дорогих signal calls, 21.08 ms;
- синтетическое ускорение: 2.95x.

Это микрозамеры алгоритмической части; скорость на пользовательском корпусе зависит от модели, диска, FAISS/Qdrant и размера документов.

## Изменённые области

- Search/index: `core/search_engine/hybrid_search.py`, `core/faiss_manager.py`, `core/rerank.py`, `core/retrieval_signals.py`.
- Understanding/evidence: `core/query_understanding.py`, `core/query_planner.py`, `core/intent_router.py`, `core/fact_extraction.py`, `core/evidence_reasoner.py`, `core/answer_builders/structured.py`.
- Structured multi-document: `core/event_store.py`, `core/analytics/statistics_engine.py`, `core/knowledge_units.py`.
- Domain/providers: `core/domain_profile.py`, `core/domain_lexicon.py`, `core/providers.py`, `integrations/server_providers.py`.
- LLM/runtime: `core/llm/runtime.py`, `core/memory_guard.py`, `core/model_integrity.py`, `gui/workers/streaming.py`.
- Desktop/product/packaging: GUI modules, configs, launcher, installer, requirements, documentation and CI files listed by Git diff.
- Tests: `tests/test_core_regressions.py`, `tests/test_search_accuracy.py`, `tests/test_evidence_reasoner.py`.

## Dead code

Удалённые legacy chat/LLM/search mixins и `gui.main_window.app_impl` проверены поиском импортов: runtime-ссылок нет. Оставшаяся отрицательная ссылка в PyInstaller spec была только исключением уже удалённого модуля и тоже удалена. `compileall` для `core`, `gui`, `integrations`, `tools` проходит.

## Что ещё не подтверждено

- Live server E2E с реально запущенными TEI, Qdrant и vLLM не выполнялся; адаптеры проверены контрактными fake-client тестами.
- В текущем test environment нет нативного FAISS, поэтому rebuild/load проверены state/regression-тестами, а поиск выполняется через штатный numpy fallback.
- Замеры выполнены на синтетическом наборе, не на закрытом пользовательском corpus.
- В старом military extractor остаются исторические специализированные правила. Их массовое перемещение не выполнялось без отдельного полного набора доменных fixtures; новая лексика уже проходит только через DomainProfile.
