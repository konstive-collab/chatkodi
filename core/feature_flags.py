"""Безопасные feature-flags для будущих расширений проекта."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FeatureFlags:
    plugin_loader_enabled: bool = False
    diagnostics_enabled: bool = False
    healthcheck_cli_enabled: bool = True
    telemetry_enabled: bool = False
    benchmark_cli_enabled: bool = True
    resilience_enabled: bool = True
    auto_profile_enabled: bool = True

    @classmethod
    def from_config_data(cls, data: Mapping[str, Any] | None) -> "FeatureFlags":
        features = (data or {}).get("features", {}) if isinstance(data, Mapping) else {}
        return cls(
            plugin_loader_enabled=bool(features.get("plugin_loader_enabled", False)),
            diagnostics_enabled=bool(features.get("diagnostics_enabled", False)),
            healthcheck_cli_enabled=bool(features.get("healthcheck_cli_enabled", True)),
            telemetry_enabled=bool(features.get("telemetry_enabled", False)),
            benchmark_cli_enabled=bool(features.get("benchmark_cli_enabled", True)),
            resilience_enabled=bool(features.get("resilience_enabled", True)),
            auto_profile_enabled=bool(features.get("auto_profile_enabled", True)),
        )

    def is_enabled(self, name: str, default: bool = False) -> bool:
        return bool(getattr(self, name, default))
