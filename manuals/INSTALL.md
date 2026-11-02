# Установка и запуск ChatKodi

Этот файл описывает полный путь: от чистой папки проекта до первого запуска приложения и базовой проверки, что всё работает.

## 1. Требования

Нужно заранее подготовить:

- Python `3.10` или новее
- доступ к интернету для первого скачивания моделей
- свободное место под `.chatkodi/models_cache/`
- желательно `pip` последней версии

Рекомендуемый вариант — Python `3.11`.

## 2. Установка зависимостей

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Опциональные зависимости

Они не обязательны для первого старта, но заметно улучшают работу:

```bash
pip install faiss-cpu pymorphy3
```

- `faiss-cpu` ускоряет семантический поиск
- `pymorphy3` улучшает лемматизацию русского текста

## 4. Скачивание моделей

Подготовьте локальные модели одной командой:

```bash
python download_models.py
```

Скрипт:

- анализирует железо
- выбирает подходящую LLM из списка доступных
- скачивает обязательную embedding-модель
- сохраняет модели в `.chatkodi/models_cache/`
- пишет служебные метаданные в эту же папку

Полезные команды проверки:

```bash
python download_models.py --check
python download_models.py --list
python download_models.py --offline
python download_models.py --repair
python download_models.py --cleanup
```

## 5. Первый запуск

После установки зависимостей и скачивания моделей:

```bash
python main.py
```

На старте приложение:

1. Проверяет наличие embedding-модели
2. Пытается загрузить LLM
3. Загружает документы
4. Строит чанки и индексы
5. Открывает главное окно

## 6. Что делать, если моделей нет

### Нет embedding-модели

Приложение не запустится. Нужно выполнить:

```bash
python download_models.py
```

### Нет LLM

Приложение запустится, но без генерации ответа. Поиск и извлечение фрагментов останутся доступными.

## 7. Смена папки документов

### Через интерфейс

В приложении можно выбрать другую папку. После выбора программа:

- запомнит путь
- сохранит его в `.chatkodi/chatkodi_settings.json`
- переиндексирует документы

### Через код

Если хотите поменять дефолт для всех запусков, измените в `gui/app.py` значение:

```python
self.docs_folder = "docs"
```

## 8. Что создаётся во время работы

Во время локальной работы могут появляться:

- `.chatkodi/models_cache/`
- `.chatkodi/logs/`
- `.chatkodi/chatkodi_settings.json`
- кэш эмбеддингов
- индексы документов и FAISS

Это нормальное поведение.

## 9. Разработка

Для локальной разработки можно дополнительно поставить dev-зависимости:

```bash
pip install -r requirements-dev.txt
```

## 10. Если старт не удался

Проверьте по порядку:

1. Активировано ли виртуальное окружение
2. Установлены ли зависимости из `requirements.txt`
3. Удалось ли скачать модели через `python download_models.py`
4. Есть ли в `.chatkodi/models_cache/embeddings/` embedding-модель
5. Есть ли в `.chatkodi/models_cache/llms/` локальная LLM
6. Нет ли ошибок чтения документов в папке `docs/`

## 11. Самый короткий сценарий установки с нуля

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install faiss-cpu pymorphy3
python download_models.py
python main.py
```

## 12. Windows-скрипты из архива

Если вы не хотите вводить команды вручную, используйте готовые bat-файлы из `scripts/windows/`:

- `scripts/windows/00_install_windows.bat`
- `scripts/windows/01_download_models.bat`
- `scripts/windows/02_healthcheck.bat`
- `scripts/windows/03_start_chatkodi.bat`
- `scripts/windows/04_create_support_bundle.bat`
