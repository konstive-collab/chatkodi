# ChatKodi 1.7.1-beta

## Что усилили

- structured-first ответы для фактологических и статистических запросов
- интеграция structured answer в RAG и streaming-поток
- расширенная аналитика: top dates, timeline, dominant patterns, pairs method→target
- memory guard 2.0: режимы safe / balanced, отключение LLM на опасных по памяти сценариях
- исправлены повреждённые regex-границы `\b` в разборе дат, идентификаторов и якорей
- extract-режим теперь может показывать структурированную выжимку перед списком фрагментов
- doctor / bootstrap / support bundle стали полезнее для диагностики и внедрения

## Что нового по инструментам

- `tools/doctor.py` теперь даёт блокирующие проблемы и рекомендации
- `tools/bootstrap.py` умеет запускать doctor и установку зависимостей
- `tools/support_bundle.py` кладёт в архив doctor report, healthcheck и свежие benchmark/diagnostic артефакты

## Важное

Это всё ещё beta-ветка 1.7, а не финальный релиз. Её уже можно нагружать реальными сценариями, но перед внешней сдачей нужен прогон на golden dataset и большой hardening retrieval/evidence-логики.
