# Быстрый старт ChatKodi 1.6.0

## Windows: самый короткий путь

1. Откройте папку проекта.
2. Запустите `scripts/windows/00_install_windows.bat`.
3. После установки зависимостей запустите `scripts/windows/01_download_models.bat`.
4. Затем проверьте окружение через `scripts/windows/02_healthcheck.bat`.
5. Запустите программу через `scripts/windows/03_start_chatkodi.bat`.

## Если нужна диагностика

- `scripts/windows/04_create_support_bundle.bat` собирает архив с логами и настройками

## Где лежат логи

- `.chatkodi/logs/chatkodi.log`
- `.chatkodi/logs/telemetry/`
- `.chatkodi/logs/diagnostics/`
- `.chatkodi/logs/recovery/`

## Что положить в программу

Ваши документы можно класть в папку `docs/` или выбрать другую директорию через интерфейс.
