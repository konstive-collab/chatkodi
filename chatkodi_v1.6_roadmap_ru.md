# ChatKodi 1.6 — подробный roadmap полной прокачки

## 1. Куда реально упирается версия 1.5

По аудиту кода текущая версия 1.5 уже не «сырая»: в проекте есть hybrid retrieval, BM25, semantic retrieval, intent-router, FAISS-слой, chunk-level rerank, extract/RAG/chat режимы и локальная LLM. Но архитектура пока не дожата до production-quality для точного и быстрого поиска по корпоративному/техническому корпусу.

### Сильные стороны 1.5
- уже есть гибридный retrieval, а не только векторы;
- есть intent-профили (`fact_lookup`, `date_fact`, `id_fact`, `summary`, `compare`);
- есть date-aware scoring и anchor/entity boosting;
- есть FAISS fallback и кэш эмбеддингов;
- есть разделение на extractive и RAG-режимы.

### Главные слабые места 1.5
1. **Повторные пересчёты retrieval на одном и том же запросе**  
   В `core/search.py` adaptive loop несколько раз вызывает BM25/semantic retrieval для того же вопроса. Особенно дорого повторно считается embedding запроса и cosine по корпусу.

2. **Неполная нормализация якорных сущностей**  
   Обозначения вида `изделие-51`, `изделием-51`, `изделие 51`, `объект №51` не сводятся в один канонический ключ.

3. **Chunking 1.0 слишком символьный**  
   `document_processor.py` режет текст по символам и делает overlap хвостом строки. Это тянет мусор, шапки, куски таблиц и полуслов в соседние чанки.

4. **RAG слишком часто идёт через full-doc режим**  
   Для коротких факт-вопросов это дороже и хуже, чем точечный chunk-RAG.

5. **Streaming LLM не настоящий**  
   В `local_llm.py` сначала делается `model.generate(...)`, а потом уже по готовому тензору текст дорезается в callback. UI получает «псевдостриминг».

6. **Контекст режется по символам, а не по токенам**  
   Это ухудшает плотность полезного контекста и может резать важные границы фрагментов.

7. **Неполный prefilter кандидатов**  
   Система сразу запускает дорогой retrieval, хотя часть запросов можно резко ускорить дешёвым lexical prefilter по anchor/date/id.

8. **Нехватка измерений качества**  
   Есть runtime stats, но нет полноценного eval-набора: Recall@k, MRR, NDCG, groundedness, hit-rate по датам/ID.

---

## 2. Что взять у конкурентов и почему

### Cursor / Instant Grep
Полезная идея не в «быстром regex», а в том, чтобы **сначала очень дёшево сузить кандидатов**, а уже потом запускать дорогую проверку. Cursor описывает trigram / sparse n-gram индекс, локальное хранение индекса и фазу candidate selection до финальной проверки по содержимому. Для ChatKodi это хорошо переносится как prefilter перед BM25/semantic retrieval. citeturn0search0

### Haystack
Haystack прямо подчёркивает, что BM25 особенно хорош для **точных keyword/entity/ID match** и часто трудно обгоняется embedding-подходами на out-of-domain данных. Это полностью совпадает с твоим кейсом: номера, коды, обозначения, даты, служебные маркеры. citeturn1search0turn1search1

### Sentence Transformers
Рекомендуемая схема для retrieval-heavy QA — **Retrieve → Re-rank**: сначала дешёвый retrieval большого shortlist, потом более точный reranker только по кандидатам. Это подтверждает, что тебе не нужно «усиливать всё сразу» — лучше добавить второй точный проход по shortlist. citeturn1search3turn1search6

### Elastic / hybrid search
Elastic рекомендует hybrid retrieval как сочетание lexical + semantic и отдельно советует fusion через RRF. Для ChatKodi это полезно как более чистая схема объединения BM25 и semantic сигналов вместо ручных весов только в одном месте пайплайна. citeturn0search2turn0search6

### Qdrant / современные vector DB
Qdrant уже даёт native hybrid dense+sparse, metadata filters, multivector и reranking primitives. Это значит, что переезд на vector DB не обязателен для 1.6, но архитектуру 1.6 стоит строить так, чтобы 1.7+ можно было без боли перенести в Qdrant или похожий движок. citeturn0search5turn0search8

### RAG как концепция
Оригинальный RAG — это не «обязательно vector DB», а связка параметрической памяти модели и внешней непараметрической памяти. То есть текущий hybrid-подход ChatKodi концептуально нормален; проблема сейчас не в отсутствии модной БД, а в качестве retrieval pipeline. citeturn2search0turn2search4

---

## 3. Целевая архитектура 1.6

### Цель версии
Сделать ChatKodi 1.6 системой, которая:
- быстрее отвечает на факт-вопросы;
- лучше находит точные фрагменты по ID/датам/объектам;
- подаёт в LLM меньше мусора и более плотный контекст;
- имеет измеримое качество retrieval/generation;
- готова к будущему переходу на native hybrid backend.

### Целевая схема пайплайна
1. **Intent routing**  
2. **Canonical query normalization**  
3. **Fast lexical prefilter**  
4. **BM25 + semantic retrieval only on shortlist**  
5. **Fusion (RRF / weighted hybrid)**  
6. **Chunk rerank / sentence rerank**  
7. **Fast-path extractive answer if confidence is high**  
8. **LLM only when really needed**  
9. **Token-aware context builder**  
10. **True streaming in UI**

---

## 4. Полный roadmap 1.6 по этапам

# Этап 0 — заморозка 1.5 и измерение базы

## Задачи
- Зафиксировать baseline по качеству и скорости.
- Собрать eval-набор минимум из 120–200 вопросов:
  - fact lookup;
  - date fact;
  - id/object fact;
  - summary;
  - compare.
- Для каждого вопроса хранить:
  - expected doc_id;
  - expected chunk span;
  - expected short answer;
  - тип вопроса;
  - сложность.

## Метрики baseline
- `P50/P95 retrieval latency`
- `P50/P95 answer latency`
- `Recall@10 / Recall@50`
- `MRR@10`
- `NDCG@10`
- `Top-1 anchor hit rate`
- `Top-1 date hit rate`
- `LLM grounded answer rate`
- `No-answer precision`

## DoD
- Есть повторяемый eval script.
- Есть CSV/JSON отчёт baseline.

---

# Этап 1 — Retrieval Core 2.0 (самый высокий приоритет)

## 1.1. `core/search.py` — RetrievalSession и кэш запроса

### Проблема
Adaptive retrieval повторно считает одно и то же.

### Изменения
Добавить объект уровня одного запроса:
- `RetrievalSession(question, normalized_question, query_embedding, tokenized_query, canonical_anchors, date_constraints)`

Внутри сессии хранить:
- query embedding;
- токены запроса;
- canonical anchors;
- извлечённые идентификаторы;
- даты/диапазоны;
- результаты prefilter;
- результаты BM25/semantic на max_k.

### Что поменять
- `_adaptive_collect_candidates()` больше не должен на каждом шаге заново кодировать запрос.
- `_semantic_candidates_raw()` должен уметь принимать готовый `query_embedding`.
- `_bm25_candidates_raw()` должен уметь работать по shortlist positions, а не по всему корпусу.
- Один раз считать retrieval на `k_max`, а внутри adaptive loop только расширять видимый top.

### Ожидаемый эффект
- -20%…-45% к latency retrieval на одном запросе.
- Меньше CPU/GPU waste.

## 1.2. `core/search.py` — Lexical Prefilter Index

### Проблема
Дорогой retrieval запускается по слишком широкому корпусу.

### Изменения
Добавить новый слой:
- `LexicalPrefilterIndex`

Хранить на chunk-level:
- нормализованный текст;
- набор trigram / hashed n-gram;
- canonical anchors;
- identifiers;
- numbers;
- dates;
- короткий signature для header/table-like фрагментов.

### Новый сценарий
- Если запрос содержит `ID`, `№`, якорный объект, дату, редкий токен → сначала prefilter.
- Prefilter отдаёт shortlist, например 150–400 chunk ids.
- BM25 и semantic retrieval работают уже по shortlist.

### Режимы
- `prefilter_mode = off | auto | force`
- `prefilter_trigger_on_ids = true`
- `prefilter_trigger_on_dates = true`
- `prefilter_trigger_on_rare_terms = true`

### Ожидаемый эффект
- Сильное ускорение id/date/object запросов.
- Меньше ложных семантических попаданий.

## 1.3. `core/search.py` — Canonical Anchors 1.0

### Проблема
`изделие-51` и `изделием-51` живут как разные сущности.

### Изменения
Вынести отдельный модуль, например:
- `core/anchor_normalizer.py`

Функции:
- `normalize_object_anchor(text) -> canonical_key`
- `extract_canonical_anchors(text) -> list[str]`

### Правила нормализации
- привести к нижнему регистру;
- `ё -> е`;
- убрать лишние пробелы, `№`, служебные знаки;
- словесную часть лемматизировать;
- числовую часть оставить как есть;
- варианты `изделие 51`, `изделием-51`, `изделия №51` сводить к `изделие#51`.

### Где использовать
- query preprocessing;
- prefilter index;
- `_object_term_bridge_score()`;
- `_bonus_by_identifiers()`;
- final rerank explanations.

### Ожидаемый эффект
- Сильный рост Top-1 hit rate по якорным объектам.

## 1.4. Fusion 2.0

### Проблема
Текущий hybrid score рабочий, но жёстко зашит и не очень прозрачен.

### Изменения
Варианты:
- базовый режим: current weighted fusion;
- новый режим: `RRF`;
- экспериментальный: `weighted_rrf`.

### Почему
Elastic рекомендует RRF как устойчивый способ объединять lexical и semantic выдачу. citeturn0search2turn0search6

### Что добавить
В конфиг:
```json
"fusion": {
  "mode": "rrf",
  "rrf_k": 60,
  "bm25_weight": 0.45,
  "semantic_weight": 0.55
}
```

---

# Этап 2 — Chunking 2.0 и cleaner corpus

## 2.1. `core/document_processor.py` — Structural Chunking

### Проблема
Сейчас chunking слишком символьный.

### Изменения
Новая схема:
1. выделить блоки документа:
   - page headers/footers;
   - заголовки;
   - абзацы;
   - списки;
   - таблицы.
2. внутри больших блоков резать по предложениям;
3. overlap делать не хвостом строки, а последними 1–2 предложениями;
4. хранить block metadata:
   - `block_type`
   - `page_num`
   - `section_title`
   - `is_header_like`
   - `is_table_like`

### Новые параметры
```json
"chunking": {
  "strategy": "structural_v2",
  "target_chars": 1100,
  "hard_max_chars": 1500,
  "min_chars": 180,
  "sentence_overlap": 2,
  "drop_repeated_headers": true,
  "separate_tables": true
}
```

### Ожидаемый эффект
- Чанки становятся семантически плотнее.
- Меньше мусорных шапок и «половинок».
- Лучше extraction и RAG.

## 2.2. Cleaner pipeline

### Добавить
- удаление повторяющихся page headers/footers;
- пометка таблиц и служебных секций;
- нормализация OCR-like артефактов;
- отдельная логика для DOCX-таблиц и PDF page markers.

### Важный эффект
Если document cleaner не починен, retrieval всегда будет бороться с мусором downstream.

---

# Этап 3 — Rerank / answer extraction 2.0

## 3.1. `core/search.py` — Sentence-level second pass

### Проблема
Нужная фраза часто сидит внутри длинного чанка.

### Изменения
После выбора top documents:
- взять top 3–5 docs;
- разрезать их top chunks на предложения/мини-фрагменты;
- сделать second-pass scoring по sentence windows.

### Режимы
- для `fact_lookup`, `date_fact`, `id_fact` — включать обязательно;
- для `summary` — выключать или ограничивать.

### Эффект
- Рост точности на «когда / кто / какой номер / какая дата / какой статус».

## 3.2. `core/rerank.py` — Cross-encoder hook (optional)

### Изменения
Добавить интерфейс reranker’а:
- `heuristic`
- `cross_encoder_local`
- `off`

Минимальный стартовый вариант:
- локальный small cross-encoder только на shortlist 20–40 кандидатов.

### Почему
Retrieve→Rerank — уже стандартная практика для QA / IR систем. citeturn1search3turn1search6

## 3.3. Fast-path без LLM

### Проблема
LLM вызывается даже там, где answer already in chunk.

### Изменения
Если:
- Top-1 chunk score высокий,
- есть `id_match` или `canonical_anchor_match`,
- или strong date hit,
- и ответ можно вытащить extractive шаблоном,

то:
- сначала пробуем `extractive_answer_builder`;
- LLM только если confidence низкий.

### Эффект
- Сильное ускорение простых факт-вопросов.
- Меньше галлюцинаций.

---

# Этап 4 — RAG Orchestrator 2.0

## 4.1. `core/chat_mode.py` — Intent-aware generation mode

### Проблема
`use_full_documents=True` слишком часто применяется по умолчанию.

### Изменения
Новая логика:
- `fact_lookup` → chunk-RAG
- `date_fact` → chunk-RAG + sentence windows
- `id_fact` → chunk-RAG + anchor priority
- `summary` → full-doc / expanded context
- `compare` → paired evidence blocks

### Что добавить
Метод:
- `_select_generation_strategy(intent_profile, search_result)`

Возвращает:
- `extractive_only`
- `chunk_rag`
- `full_doc_rag`
- `compare_rag`

## 4.2. Context Builder 2.1

### Изменения
Контекст строить не просто как список чанков, а как evidence packet:
- основной ответный фрагмент;
- соседние supporting chunks;
- section title;
- страница/источник;
- confidence markers.

### Для compare
Собирать две стороны сравнения отдельно:
- `evidence_a`
- `evidence_b`
- `delta hints`

## 4.3. No-answer policy

### Добавить
Если retrieval weak:
- не отправлять мусор в LLM;
- возвращать честное `в документах нет надёжного подтверждения`;
- показывать 2–3 лучших источника как кандидатов.

---

# Этап 5 — LLM Runtime 2.0

## 5.1. `core/local_llm.py` — True streaming

### Проблема
Сейчас streaming псевдостриминговый.

### Изменения
Перейти на `TextIteratorStreamer` или эквивалентный поток токенов из `transformers`, чтобы UI получал ответ во время генерации, а не после полной генерации.

### Эффект
- Лучшая воспринимаемая скорость.
- Чище UX.

## 5.2. Token-aware context trimming

### Проблема
Контекст сейчас режется по символам.

### Изменения
- резать по токенам tokenizer’а;
- учитывать reserved budget:
  - system/prompt tokens;
  - answer budget;
  - safety margin.

### Новый бюджет
```json
"llm_runtime": {
  "max_input_tokens": 12000,
  "reserved_output_tokens": 320,
  "safety_margin_tokens": 256
}
```

## 5.3. Runtime profiles через конфиг

### Изменения
Вынести в конфиг:
- quantization mode;
- dtype;
- device_map;
- max_new_tokens per mode;
- repetition penalty;
- stream mode;
- flash attention toggle.

### Пример
```json
"llm": {
  "quantization": "8bit",
  "dtype": "float16",
  "use_flash_attention": true,
  "profiles": {
    "chunk_rag": {"max_new_tokens": 220, "temperature": 0.2},
    "full_doc_rag": {"max_new_tokens": 360, "temperature": 0.2},
    "chat": {"max_new_tokens": 320, "temperature": 0.5}
  }
}
```

## 5.4. Optional structured answering

### Добавить
Для RAG-ответов режим structured output:
- short_answer
- evidence
- sources
- confidence

Это поможет cleaner post-processing и тестированию groundedness.

---

# Этап 6 — Конфиг 1.6 и расширение модулей

## Новый конфиг
Добавить секции:
- `chunking`
- `prefilter`
- `fusion`
- `anchors`
- `llm`
- `eval`
- `telemetry`

## Новые модули
- `core/anchor_normalizer.py`
- `core/prefilter_index.py`
- `core/eval_runner.py`
- `core/extractive_answer.py`
- `core/query_session.py`

## Рефакторинг существующих модулей
- `search.py` — облегчить и разделить на retrieval/fusion/post-rank
- `document_processor.py` — вынести structural parsing helpers
- `chat_mode.py` — сделать router стратегии генерации
- `local_llm.py` — отделить runtime/generation/postprocess

---

# Этап 7 — UI / UX 1.6

## Что показать пользователю
- почему найден именно этот фрагмент;
- какие якоря/даты/ID совпали;
- confidence retrieval;
- режим ответа: `extractive`, `chunk-rag`, `full-doc-rag`;
- реальный streaming;
- кнопка «показать доказательства».

## Полезные элементы
- badge с типом вопроса (`ID`, `DATE`, `SUMMARY`, `COMPARE`);
- toggle «точный режим / широкий режим»;
- диагностика: сколько кандидатов отфильтровано prefilter’ом.

---

## 5. Приоритеты задач

### P0 — must have для 1.6
1. Query/session cache в retrieval
2. Canonical anchors
3. Lexical prefilter
4. Structural chunking 2.0
5. Intent-aware generation mode
6. Token-aware trimming
7. True streaming
8. Eval harness

### P1 — strongly recommended
1. RRF fusion
2. Sentence-level second pass
3. Fast-path without LLM
4. Cleaner repeated header/footer removal
5. Better no-answer policy

### P2 — можно после 1.6.0, но желательно в 1.6.x
1. Cross-encoder local reranker
2. Optional sparse encoder / SPLADE-like experiments
3. Exportable search diagnostics
4. Migration-ready adapter for Qdrant

---

## 6. Что именно менять по файлам

## `core/search.py`
### Изменить
- `_adaptive_collect_candidates`
- `_semantic_candidates_raw`
- `_bm25_candidates_raw`
- `_object_term_bridge_score`
- `_extract_object_like_terms`
- retrieval fusion block
- candidate scoring pipeline

### Добавить
- `RetrievalSession`
- shortlist-aware retrieval
- prefilter integration
- canonical anchor matching
- RRF fusion mode
- sentence rerank hook

## `core/document_processor.py`
### Изменить
- `_clean_document_content`
- `_create_chunks`
- `_split_text_into_chunks`

### Добавить
- structural block parser
- repeated header/footer detector
- sentence-aware overlap
- block metadata

## `core/chat_mode.py`
### Изменить
- `process_message`
- `_rag_response_full_documents`
- extract/RAG routing

### Добавить
- `_select_generation_strategy`
- `extractive_only` path
- evidence packet builder
- compare-mode context assembly

## `core/local_llm.py`
### Изменить
- `generate_with_callback`
- `generate_streaming`
- `_trim_context`
- `_get_runtime_profile`

### Добавить
- tokenizer-aware budgeter
- true streamer
- configurable runtime profiles
- structured output mode

## `core/intent_router.py`
### Изменить
- лучше различать `date+id`, `compare without explicit date`, `summary vs extractive fact`

## `core/rerank.py`
### Добавить
- cross-encoder adapter
- fusion-aware rerank inputs
- sentence-window rerank

## `core/faiss_manager.py`
### Изменить
- использовать загрузку существующего индекса там, где rebuild не нужен
- подготовить API для shortlist retrieval и persistent normalized vectors

---

## 7. Метрики успеха для релиза 1.6

### Retrieval
- Recall@10: +10–20% на id/date/object queries
- MRR@10: +12%+
- Top-1 anchor hit rate: +20%+
- Top-1 date hit rate: +15%+

### Speed
- P50 retrieval latency: -25%+
- P95 retrieval latency: -30%+
- Fast factual answer latency: -35%+
- perceived first-token latency: -50%+ за счёт true streaming

### Answer quality
- grounded answer rate: +10%+
- hallucination rate: -20%+
- no-answer precision: +15%+

---

## 8. Стратегия тестирования

## Unit tests
- normalizer for anchors;
- date parser / constraints;
- chunk builder;
- prefilter candidate selection;
- token budget trimming.

## Integration tests
- docs -> chunks -> index -> query -> answer;
- id/date/summary/compare flows;
- no-answer path;
- full-doc vs chunk-rag path.

## Regression pack
Набор «плохих кейсов»:
- `изделие-51` vs `изделием-51`;
- repeated page headers;
- дата только в соседнем чанке;
- нужная сущность в таблице;
- вопрос, на который ответа нет.

## Performance tests
- corpus 100 docs / 1k docs / 10k chunks;
- cold start;
- warm cache;
- CPU mode;
- low VRAM mode;
- full-doc RAG vs chunk-RAG.

---

## 9. План релиза 1.6

## 1.6.0-alpha
- RetrievalSession
- canonical anchors
- prefilter prototype
- eval harness

## 1.6.0-beta1
- chunking 2.0
- intent-aware generation routing
- token-aware trimming
- true streaming

## 1.6.0-beta2
- sentence rerank
- fast-path no-LLM
- RRF fusion
- better no-answer policy

## 1.6.0-rc
- performance polish
- config cleanup
- migration compatibility layer
- UI evidence/debug widgets

## 1.6.0 stable
- docs
- benchmarks
- packaged config presets
- release notes with before/after numbers

---

## 10. Что не надо делать в 1.6

1. Не надо сразу полностью переписывать систему под Qdrant/Chroma/Milvus.  
   Для 1.6 выгоднее усилить текущий retrieval core.

2. Не надо бесконечно увеличивать `k_max`.  
   Это лечит симптомы, а не причину.

3. Не надо сразу грузить тяжёлый reranker на каждый запрос.  
   Сначала нужен хороший shortlist.

4. Не надо держать full-doc RAG как дефолт для всех вопросов.  
   Это бьёт и по скорости, и по точности.

---

## 11. Самый практичный порядок работ

Если делать 1.6 максимально рационально, то порядок такой:

### Первая волна
1. Eval harness
2. RetrievalSession cache
3. Canonical anchors
4. Lexical prefilter

### Вторая волна
5. Chunking 2.0
6. Intent-aware generation routing
7. Token-aware context trimming
8. True streaming

### Третья волна
9. Sentence-level rerank
10. Fast-path without LLM
11. RRF fusion
12. UI evidence/debug

---

## 12. Итог

### Версия 1.6 должна стать не «ещё одной версией с подкруткой весов», а именно архитектурным апгрейдом:
- retrieval становится двухступенчатым и дешёвым на входе;
- якорные сущности нормализуются канонически;
- chunking перестаёт тащить мусор;
- LLM вызывается реже и умнее;
- контекст режется по токенам, а не по символам;
- ответы начинают быть быстрее, точнее и честнее.

### Главная ставка 1.6
Не «больше векторов», а **лучше candidate selection + лучше evidence selection + меньше ненужной генерации**.
