# Работа с моделями

ChatKodi использует два типа локальных моделей:

- embedding-модель — обязательна
- LLM — опциональна

## 1. Что обязательно для старта

Без embedding-модели приложение не сможет построить индекс и не запустится.

Без LLM программа запускается, но будет доступен только поиск и извлечение фрагментов без генерации ответа.

## 2. Какая embedding-модель используется

По текущей логике проекта целевой вариант — семейство:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

## 3. Как выбирается LLM

Скрипт `download_models.py`:

- анализирует RAM, CPU и наличие CUDA
- определяет класс устройства (`low`, `medium`, `high`)
- подбирает подходящую модель из списка
- скачивает её в `models_cache/llms/`

Если нужен ручной выбор, используйте:

```bash
python download_models.py --manual
```

## 4. Где лежат модели

По умолчанию используется папка:

```text
models_cache/
```

Типичная структура:

```text
models_cache/
├─ embeddings/
├─ llms/
├─ models_meta.json
├─ model_choice.json
└─ download_success.txt
```

## 5. Полезные команды

```bash
python download_models.py
python download_models.py --check
python download_models.py --list
python download_models.py --offline
python download_models.py --force
python download_models.py --cleanup
python download_models.py --repair
python download_models.py --manual
```

## 6. CPU, GPU и типы загрузки

Файл `core/local_llm.py` определяет устройство автоматически.

Текущая логика:

- если доступна CUDA — используется `device_map="auto"` и `torch.float16`
- если CUDA нет — используется CPU и `torch.float32`

Это значит, что в текущей версии уже есть разделение по устройствам, но нет отдельной полноценной подсистемы настраиваемого квантования.

## 7. Что с квантованием

Это важно: отдельного параметра вроде `quantization: 4bit` в текущем проекте нет.

На практике сейчас возможны три сценария:

### Вариант 1. Просто выбрать более лёгкую модель

Это самый безопасный путь. Лёгкая модель даст меньшую нагрузку и обычно заменяет необходимость отдельного квантования в рамках текущей реализации.

### Вариант 2. Положить уже подготовленную квантованную модель вручную

Если у вас есть локальная модель в совместимом формате и она корректно загружается через `transformers`, её можно положить в `models_cache/llms/<имя_папки>`.

После этого проверьте:

```bash
python download_models.py --check
```

### Вариант 3. Доработать загрузчик модели

Если вам нужна реальная настройка `4bit/8bit`, дорабатывать надо `core/local_llm.py`. Именно там вызывается `AutoModelForCausalLM.from_pretrained(...)`.

## 8. Где менять модельную логику

### Каталог моделей

- `download_models.py`
- `core/models.py`
- `core/local_llm.py`

### Автовыбор подходящей LLM

- `download_models.py`
- логика `SystemProfiler` и список рекомендованных моделей

### Параметры генерации

- `core/local_llm.py`
- `_get_runtime_profile()`
- `generate_with_callback()`

### Поведение под CPU/GPU

- `core/local_llm.py`
- `core/gpu_accelerator.py`

## 9. Как понять, что модель не видится

Проверьте:

1. существует ли `models_cache/`
2. есть ли внутри `embeddings/` и `llms/`
3. видна ли модель через `python download_models.py --list`
4. совпадают ли пути в `download_models.py`, `core/models.py` и `core/local_llm.py`
5. нет ли ситуации, когда LLM случайно определяется как embedding-модель

## 10. Практический совет

Если нужна понятная и стабильная работа, сначала подберите правильный размер модели и только потом уже лезьте в изменение загрузчика. Для большинства задач это даёт больше пользы, чем агрессивная доработка квантования.
