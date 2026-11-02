# ChatKodi 1.9 — переносимая кодовая база

Эта папка предназначена для передачи разработчикам и встраивания ChatKodi в
сайт, внутренний портал или другое приложение. В комплект не входят модели,
личные документы, индексы, логи и пользовательские настройки.

## Быстрый запуск на Windows

1. Установите Python 3.11 x64 и Git.
2. Запустите `SETUP_CODEBASE_WINDOWS.bat`.
3. Поместите собственные документы в отдельную папку.
4. Укажите эту папку как `generation.catalog_root` в `chatkodi_config.json`.
5. Для desktop-версии запустите `START_LAUNCHER.bat`.

Модели загружаются отдельно командой `.venv\Scripts\python.exe download_models.py`.
Они намеренно не вложены в передаваемый комплект: размер моделей составляет
несколько гигабайт, а вариант PyTorch зависит от оборудования сервера.

## Встраивание в сайт

Пример HTTP-обёртки находится в `integrations/fastapi_example.py`.

```bat
.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.venv\Scripts\python.exe -m uvicorn integrations.fastapi_example:app --host 127.0.0.1 --port 8080
```

Проверка: `GET http://127.0.0.1:8080/health`.

Основной запрос: `POST /api/chat` с JSON:

```json
{"session_id":"user-42","message":"Составь рапорт на командировку"}
```

Повторные сообщения должны иметь тот же `session_id`: так сохраняется сценарий
уточнений. Если ответ содержит проверенный `artifact.id`, файл забирается через
`POST /api/artifacts/save`.

Функция `build_llm()` в примере намеренно возвращает `None`. Разработчик сайта
должен подключить разрешённую локальную модель или собственный совместимый
провайдер. Без модели ChatKodi честно сообщает, что генерация недоступна.

## Карта адаптации

- `core/generation/router.py` — распознавание сценариев.
- `core/generation/service.py` — диалог, уточнения и генерация.
- `core/generation/docx_engine.py` — безопасное создание DOCX.
- `core/document_catalog.py` — каталогизация и классификация корпуса.
- `core/chat/manager.py` — прежний RAG и общий чат.
- `integrations/fastapi_example.py` — пример интеграции сайта.
- `gui/` — desktop-интерфейс; для сайта он не обязателен.

## Важные требования безопасности

- Не публикуйте папку документов через web-сервер.
- Не записывайте ФИО и реквизиты документов в access/error-логи.
- Разделяйте пользовательские сессии и права доступа к корпусам.
- Отдавайте DOCX только после `can_save=true`.
- Не разрешайте пользователю передавать произвольный путь к файлу на сервере.
- Для нескольких процессов замените память сессий на Redis или БД.

## Проверка изменений

`.venv` хранит абсолютный путь к Python, из которого создан, поэтому окружение,
скопированное с другого компьютера (или из полной копии проекта), там не
запускается («No Python at …»). Рабочее окружение в корне проекта создаёт или
пересоздаёт скрипт (он же ставит `requirements-dev.txt`, `-Test` сразу запускает тесты):

```bat
powershell -ExecutionPolicy Bypass -File tools\dev_env.ps1 -Test
```

```bat
.venv\Scripts\python.exe -m compileall -q core gui integrations
.venv\Scripts\python.exe -m pytest -q
```

Расширенный набор контрольных запросов подключён к тому же benchmark CLI. Для
него нужен соответствующий проверочный корпус документов:

```bat
.venv\Scripts\python.exe benchmark.py --cases search_accuracy_cases.json --docs-folder <папка-с-контрольными-документами>
```

`benchmark_cases.sample.json` остаётся компактным автономным набором, а
`search_accuracy_cases.json` — расширенным E2E-набором.

### Проверка через окно программы (как у тестировщика)

`tools/gui_e2e.py` запускает настоящее окно ChatKodi с реальными моделями,
печатает вопросы в поле ввода, нажимает «Отправить», ждёт ответ и проверяет
видимый текст; скриншоты провалов сохраняются в `<out>/shots`. Банк из ~300
сценариев (опечатки, короткие даты, итоги за месяц, рейтинги, сравнения,
уточнения «а 15-го?», вопросы вне документов, приветствия, пересказы) строит
`tools/scenario_bank.py`, эталоны считаются из генератора корпуса:

```bat
.venv\Scripts\python.exe -m tools.synthetic_corpus --out runtime\synthetic --days 62
.venv\Scripts\python.exe -m tools.scenario_bank --out runtime\synthetic\scenarios.json
.venv\Scripts\python.exe -m tools.gui_e2e --docs runtime\synthetic\docs --questions runtime\synthetic\scenarios.json --out runtime\gui_e2e --shots fail
```

`--ui-tour` дополнительно снимает все страницы, темы и размеры окна,
`--only revenue_typo,followup` гоняет отдельные категории.

Версия комплекта: `1.9.0`.
