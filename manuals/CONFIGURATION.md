# Настройка программы

У ChatKodi есть три уровня конфигурации:

1. базовый конфиг проекта — `chatkodi_config.json`
2. пользовательские runtime-настройки — `.chatkodi/chatkodi_settings.json`
3. зашитые в код дефолты — `gui/app.py`, `gui/styles.py`, `core/document_processor.py`, `core/search.py`, `core/chat_mode.py`, `core/local_llm.py`

Ниже — практическая карта, где именно что менять.

## 1. Главный конфиг: `chatkodi_config.json`

Это основной файл, где без правки Python-кода настраиваются:

- retrieval
- лимиты контекста
- rerank
- intent-профили
- часть параметров выдачи интерфейса

Если нужен пример с подробными русскими комментариями, смотрите файл [chatkodi_config.example.ru.jsonc](chatkodi_config.example.ru.jsonc).

## 2. Как выбрать свой конфиг

По умолчанию используется `chatkodi_config.json`, но путь можно переопределить через переменную среды `CHATKODI_CONFIG`.

### PowerShell

```powershell
$env:CHATKODI_CONFIG="C:\path\to\my_config.json"
python main.py
```

### bash

```bash
export CHATKODI_CONFIG=/path/to/my_config.json
python main.py
```

## 3. Раздел `retrieval`

Этот блок управляет первичным поиском.

Основные параметры:

- `adaptive_enabled` — включён ли адаптивный добор кандидатов
- `k_base` — стартовый размер пула кандидатов
- `k_step` — насколько расширять пул на каждой итерации
- `k_max` — верхний предел пула
- `time_budget_ms` — лимит времени на адаптивный retrieval
- `bm25_k` — сколько кандидатов брать из BM25
- `semantic_k` — сколько кандидатов брать из semantic-поиска
- `per_doc_chunk_cap` — максимум чанков от одного документа
- `max_results_cap` — жёсткий потолок результатов
- `answer_rescue_docs` — сколько документов участвует в rescue-pass

Когда это менять:

- если поиск слишком узкий — увеличивайте `k_base`, `k_max`, `bm25_k`, `semantic_k`
- если поиск медленный — уменьшайте эти значения
- если один документ забивает всю выдачу — уменьшайте `per_doc_chunk_cap`

## 4. Раздел `context`

Это базовые лимиты контекста для RAG.

- `max_docs` — максимум документов в контексте
- `max_chars` — общий лимит символов
- `chunks_per_doc` — сколько чанков брать от документа

## 5. Раздел `context_v2`

Это более продвинутая логика сборки контекста.

- `enabled` — включение новой схемы
- `window_before` / `window_after` — сколько соседних чанков брать до и после якорного
- `max_chunks_per_doc` — максимум чанков на документ
- `prefer_anchor_chunks` — приоритет чанков с сильными сигналами
- `allow_full_doc_fallback` — можно ли откатываться к целому документу
- `full_doc_max_chars` — лимит символов для fallback на весь документ

## 6. Раздел `rerank`

Управляет переоценкой результатов.

- `dedup_enabled` — включить удаление почти одинаковых фрагментов
- `dedup_shingle_size` — размер шингла для дедупликации
- `dedup_threshold` — порог схожести
- `anchor_boost` — усиление чанков с якорными признаками
- `date_boost` — усиление чанков с датным совпадением
- `id_boost` — усиление чанков с совпадением по идентификатору
- `entity_boost` — усиление чанков с сущностями
- `prefer_diversity` — поощрять разнообразие документов
- `max_chunks_per_doc_in_top` — максимум чанков одного документа в топе

## 7. Раздел `intents`

Позволяет отдельно настраивать профили под тип вопроса.

Доступные профили в текущей поставке:

- `fact_lookup`
- `date_fact`
- `id_fact`
- `summary`
- `compare`

Для каждого профиля можно задавать свой retrieval и контекст.

## 8. Раздел `ui`

Это не стили, а параметры выдачи в интерфейсе.

- `preview_chars` — длина preview-фрагмента
- `extract_docs_min` / `extract_docs_max` — границы числа документов в `Extract`
- `extract_docs_rel_threshold` — относительный порог отсечения
- `extract_docs_abs_threshold` — абсолютный порог отсечения
- `extract_docs_request_k` — сколько результатов запрашивать для режима `Extract`
- `extract_chunks_per_doc` — сколько чанков показывать на документ

## 9. Пользовательские настройки: `.chatkodi/chatkodi_settings.json`

Этот файл создаётся во время работы и хранит пользовательские параметры.

Обычно там лежат:

- `docs_folder`
- `docs_recursive`
- `theme`

За чтение и запись отвечают методы `_load_settings()` и `_save_settings()` в `gui/app.py`.

## 10. Что менять в коде, если JSON не хватает

### Папка документов по умолчанию

Файл: `gui/app.py`

Меняйте значение:

```python
self.docs_folder = "docs"
```

### Рекурсивный обход подпапок

Файл: `gui/app.py`

```python
self.docs_recursive = True
```

### Тема по умолчанию

Файл: `gui/app.py`

```python
self.current_theme = "dark"
```

### Цвета, рамки, кнопки, пузырьки сообщений

Файл: `gui/styles.py`

Там меняются:

- фон окна
- верхняя панель
- цвета текста
- границы и рамки
- кнопка отправки
- поле ввода
- пузырьки пользователя и ассистента
- scrollbar

### Цвета экрана загрузки, спиннера и консоли

Файл: `gui/app.py`

Ищите классы:

- `LoadingSpinner`
- `ConsoleOutput`
- `LoadingScreen`

### Размеры чанков и overlap

Файл: `core/document_processor.py`

```python
self.chunk_max_chars = 1400
self.chunk_overlap = 220
self.min_chunk_chars = 120
```

### Максимальный размер файла

Файлы:

- `gui/app.py` — при создании `DocumentProcessor`
- `core/document_processor.py` — параметр `max_file_size_mb`

### Веса BM25 и semantic

Файл: `core/search.py`

Параметры конструктора `HybridSearch`:

```python
bm25_weight=0.35
semantic_weight=0.65
```

### Поведение режима `Extract`

Файл: `core/chat_mode.py`

Настройки по умолчанию:

```python
self.default_preview_chars = 800
self.default_docs_min = 1
self.default_docs_max = 30
self.default_docs_rel_threshold = 0.60
self.default_docs_abs_threshold = 0.18
self.default_chunks_per_doc = 1
self.default_docs_request_k = 40
```

### Параметры генерации LLM

Файл: `core/local_llm.py`

Ищите:

- `_get_runtime_profile()`
- `generate_with_callback()`
- `generate_clean_answer()`

Там регулируются:

- `max_tokens`
- `temperature`
- `top_p`
- `repetition_penalty`
- контекстные лимиты для CPU/GPU

### Пути к моделям

Файлы:

- `download_models.py`
- `core/models.py`
- `core/local_llm.py`

Менять лучше согласованно во всех трёх местах.

## 11. Где править квантование

С версии 2.0 модели — GGUF-файлы для llama.cpp, и квантование задаётся выбором файла:
`Qwen3.5-4B-Q4_K_M.gguf` (по умолчанию), `...-Q5_K_M.gguf`, `...-Q8_0.gguf` и т. д.
Положите нужный файл в `models_cache/llms/` или укажите его в `CHATKODI_MODEL_PATH`.

- Запуск сервера модели и параметры генерации: `core/llm/runtime.py` (`_server_args`, `_get_runtime_profile`).
- Выбор модели под видеопамять: `core/llm/model_select.py`.
- Сервер эмбеддингов: `core/models.py`.
- Размер контекста LLM: переменная `CHATKODI_LLM_CTX` (по умолчанию 16384 токена).

## 12. Быстрый совет по безопасной настройке

Сначала меняйте только JSON и стили, а уже потом — поведение поиска и LLM. Такой порядок снижает шанс случайно сломать проект.
