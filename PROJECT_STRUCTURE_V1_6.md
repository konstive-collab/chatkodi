# ChatKodi v1.6 — историческая структура

Этот файл оставлен только как отметка о переходе к модульной структуре в 1.6.
Он больше не является картой актуального дерева: в 1.9.0 недостижимые
копии `helpers/selection/responses`, старые поисковые стадии и дубли LLM были
удалены. Текущая карта находится в `PROJECT_STRUCTURE.md`.

Актуальные канонические реализации:

```text
core/
  chat/manager.py
  llm/runtime.py
  search_engine/hybrid_search.py
  documents/processor.py

gui/
  components/
  workers/
  main_window/

tools/
  model_manager/downloader.py
```

Совместимые корневые точки входа (`core/chat_mode.py`, `core/search.py`,
`core/local_llm.py`, `gui/app.py`, `download_models.py`) остаются тонкими
реэкспортами и не должны использоваться как место для новых исправлений.
