"""Плагинная инфраструктура ChatKodi.

Она по умолчанию выключена и нужна как безопасная заготовка для v1.6+.
"""

from .registry import ChatKodiPluginSpec, PluginRegistry, load_plugins_from_config

__all__ = ["ChatKodiPluginSpec", "PluginRegistry", "load_plugins_from_config"]
