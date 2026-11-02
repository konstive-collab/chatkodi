# Настройка программы

У ChatKodi есть три уровня конфигурации:

1. базовый конфиг проекта — `chatkodi_config.json`
2. пользовательские runtime-настройки — `chatkodi_settings.json`
3. зашитые в код дефолты — `core/config.py`, `gui/main_window/bootstrap.py`,
   `gui/styles.py`, `core/documents/processor.py`,
   `core/search_engine/hybrid_search.py`, `core/chat/manager.py`, `core/llm/runtime.py`

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

Для опционального обучаемого второго прохода используются:

- `learned_enabled` — включает локальный cross-encoder
- `learned_model_path` — путь к уже скачанной модели; сеть не используется
- `learned_topn` — размер короткого списка для модели
- `learned_weight` — доля оценки cross-encoder в итоговом балле

По умолчанию обучаемый reranker выключен, поэтому слабые компьютеры сохраняют
прежний быстрый эвристический режим.

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

### История диалога

Секция `history` управляет локальной историей: `enabled`, путь `path` и предел
`max_entries`. Данные остаются на компьютере пользователя; из меню «Файл» их
можно экспортировать в TXT/JSON или удалить после подтверждения.

### Словарь предметной области

Секция `domain_lexicon` содержит `actor_aliases` и группы
`family_equivalents`. Так новые обозначения акторов/объектов добавляются в
конфиге один раз и одинаково используются извлечением фактов, хранилищем и
статистикой — без правки трёх разных Python-файлов.

## 9. Пользовательские настройки: `chatkodi_settings.json`

Этот файл создаётся во время работы и хранит пользовательские параметры.

Обычно там лежат:

- `docs_folder`
- `docs_recursive`
- `theme`

За чтение и запись отвечают методы `_load_settings()` и `_save_settings()` в
`gui/main_window/settings.py`.

## 10. Что менять в коде, если JSON не хватает

### Папка документов по умолчанию

Файл: `gui/main_window/bootstrap.py`

Меняйте значение:

```python
self.docs_folder = "docs"
```

### Рекурсивный обход подпапок

Файл: `gui/main_window/bootstrap.py`

```python
self.docs_recursive = True
```

### Тема по умолчанию

Файлы: `core/config.py` и `gui/main_window/bootstrap.py`

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

Файлы: `gui/components/loading.py` и `gui/components/console.py`

Ищите классы:

- `LoadingSpinner`
- `ConsoleOutput`
- `LoadingScreen`

### Размеры чанков и overlap

Файл: `core/documents/processor.py`

```python
self.chunk_max_chars = 1400
self.chunk_overlap = 220
self.min_chunk_chars = 120
```

### Максимальный размер файла

Файлы:

- `gui/main_window/bootstrap.py` — при создании `DocumentProcessor`
- `core/documents/processor.py` — параметр `max_file_size_mb`

### Веса BM25 и semantic

Настраиваются в секции `retrieval` пользовательского конфига:

```json
{
  "retrieval": {
    "bm25_weight": 0.35,
    "semantic_weight": 0.65
  }
}
```

Если сумма весов отличается от 1, ChatKodi нормализует её автоматически.

### Поведение режима `Extract`

Файл: `core/chat/manager.py`

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

Файл: `core/llm/runtime.py`

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
- `core/llm/runtime.py`

Менять лучше согласованно во всех трёх местах.

## 11. Где править квантование

В текущей версии проекта нет отдельного JSON-поля вроде `quantization = 4bit/8bit`. Поэтому настройка квантования делается косвенно:

1. через выбор конкретной локальной модели в `download_models.py`
2. через ручное размещение уже квантованной модели в `models_cache/llms/`
3. через изменение логики загрузки в `core/llm/runtime.py`

Что важно понимать:

- проект загружает модель через `AutoModelForCausalLM.from_pretrained(...)`
- для CUDA используется `torch.bfloat16` (на видеокартах без bf16 — `torch.float16`)
- для CPU используется безопасный режим `dtype="auto"`
- **int8 на видеокарте (`core/llm/quantize.py`)**: если модель в bf16 не помещается
  в видеопамять, веса линейных слоёв декодера при загрузке переводятся в int8
  (масштаб на выходной канал), и модель целиком остаётся на GPU. Qwen3-4B на
  RTX 4060 8 ГБ: ~4,1 ГБ видеопамяти вместо выгрузки 6 из 36 слоёв на CPU,
  генерация ~12 ток/с вместо ~3,6 ток/с. Дополнительные пакеты не нужны.
- переменная `CHATKODI_LLM_INT8`: `auto` (по умолчанию) — int8 только когда bf16
  не помещается; `1` — всегда int8 на CUDA; `0` — старое поведение (выгрузка слоёв на CPU)
- `bitsandbytes`/4bit по-прежнему не используются

Если нужна другая схема квантования, дорабатывать надо `core/llm/quantize.py`,
`LocalLLM._should_use_int8` в `core/llm/runtime.py` и список моделей в `download_models.py`.

Анимации интерфейса (появление сообщений, плавная прокрутка, переходы страниц)
отключаются переменной `CHATKODI_DISABLE_UI_ANIMATIONS=1`.

## 12. Быстрый совет по безопасной настройке

Сначала меняйте только JSON и стили, а уже потом — поведение поиска и LLM. Такой порядок снижает шанс случайно сломать проект.
