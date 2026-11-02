# Плагины ChatKodi

Этот каталог зарезервирован под будущие расширения.

По умолчанию загрузчик плагинов выключен, поэтому наличие файлов в этой папке
никак не влияет на работу приложения. Чтобы включить автозагрузку, установите
в `chatkodi_config.json`:

```json
{
  "features": {
    "plugin_loader_enabled": true
  },
  "plugins": {
    "enabled": true,
    "directory": "plugins",
    "auto_discover": true
  }
}
```
