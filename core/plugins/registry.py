"""Лёгкий и безопасный загрузчик плагинов."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, List, Mapping

from core.feature_flags import FeatureFlags
from core.logging_utils import get_logger
from core.paths import resolve_project_path

logger = get_logger(__name__)


@dataclass
class ChatKodiPluginSpec:
    name: str
    version: str = "0.1.0"
    description: str = ""
    module_path: str = ""


class PluginRegistry:
    def __init__(self) -> None:
        self.plugins: List[ChatKodiPluginSpec] = []
        self.errors: List[str] = []

    def register(self, spec: ChatKodiPluginSpec) -> None:
        self.plugins.append(spec)

    def register_from_module(self, module: ModuleType, module_path: Path) -> None:
        plugin_name = getattr(module, "PLUGIN_NAME", module_path.stem)
        plugin_version = getattr(module, "PLUGIN_VERSION", "0.1.0")
        plugin_description = getattr(module, "PLUGIN_DESCRIPTION", "")
        spec = ChatKodiPluginSpec(
            name=str(plugin_name),
            version=str(plugin_version),
            description=str(plugin_description),
            module_path=str(module_path),
        )
        if hasattr(module, "register") and callable(module.register):
            module.register(self)
        else:
            self.register(spec)

    def discover(self, directory: str | Path) -> "PluginRegistry":
        folder = resolve_project_path(directory)
        if not folder.exists() or not folder.is_dir():
            logger.info("Plugin directory does not exist: %s", folder)
            return self

        for path in sorted(folder.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"chatkodi_plugin_{path.stem}", path)
                if not spec or not spec.loader:
                    raise RuntimeError("plugin spec is not loadable")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.register_from_module(module, path)
                logger.info("Plugin loaded: %s", path.name)
            except Exception as exc:
                msg = f"{path.name}: {exc}"
                self.errors.append(msg)
                logger.warning("Plugin load failed: %s", msg)
        return self


def load_plugins_from_config(config_data: Mapping[str, Any] | None = None) -> PluginRegistry:
    config_data = config_data or {}
    flags = FeatureFlags.from_config_data(config_data)
    plugins_cfg = config_data.get("plugins", {}) if isinstance(config_data, Mapping) else {}
    registry = PluginRegistry()
    if not flags.plugin_loader_enabled:
        return registry
    if not bool(plugins_cfg.get("enabled", False)):
        return registry
    if bool(plugins_cfg.get("auto_discover", True)):
        registry.discover(plugins_cfg.get("directory", "plugins"))
    return registry
