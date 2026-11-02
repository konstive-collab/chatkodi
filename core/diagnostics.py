"""Опциональная диагностическая трассировка поискового пайплайна.

Модуль по умолчанию ничего не пишет на диск. Он начинает работать только
если это явно включено в конфиге. Все операции обёрнуты в безопасные
проверки, чтобы диагностика не влияла на рабочий сценарий.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from core.paths import ensure_parent


@dataclass(frozen=True)
class DiagnosticConfig:
    enabled: bool = False
    trace_search: bool = False
    trace_extract: bool = False
    trace_dir: str = ".chatkodi/logs/diagnostics"
    max_preview_chars: int = 240
    max_trace_items: int = 12


class QueryDiagnostics:
    """Пишет компактные JSON-трейсы для отладки retrieval и extract-режима."""

    def __init__(self, cfg: DiagnosticConfig | None = None):
        self.cfg = cfg or DiagnosticConfig()

    @classmethod
    def from_config_data(cls, data: Mapping[str, Any] | None) -> "QueryDiagnostics":
        data = data or {}
        features = data.get("features", {}) if isinstance(data, Mapping) else {}
        diag = data.get("diagnostics", {}) if isinstance(data, Mapping) else {}
        enabled = bool(diag.get("enabled", False) or features.get("diagnostics_enabled", False))
        cfg = DiagnosticConfig(
            enabled=enabled,
            trace_search=bool(diag.get("trace_search", False)),
            trace_extract=bool(diag.get("trace_extract", False)),
            trace_dir=str(diag.get("trace_dir", ".chatkodi/logs/diagnostics")),
            max_preview_chars=int(diag.get("max_preview_chars", 240) or 240),
            max_trace_items=int(diag.get("max_trace_items", 12) or 12),
        )
        return cls(cfg)

    def preview(self, text: str, max_chars: int | None = None) -> str:
        """Возвращает укороченный однострочный preview без риска раздувать trace."""
        max_chars = int(max_chars or self.cfg.max_preview_chars)
        compact = " ".join((text or "").split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max(0, max_chars - 3)] + "..."

    def write_search_trace(self, question: str, payload: Mapping[str, Any]) -> Path | None:
        if not (self.cfg.enabled and self.cfg.trace_search):
            return None
        return self._write("search", question, payload)

    def write_extract_trace(self, question: str, payload: Mapping[str, Any]) -> Path | None:
        if not (self.cfg.enabled and self.cfg.trace_extract):
            return None
        return self._write("extract", question, payload)

    def _write(self, kind: str, question: str, payload: Mapping[str, Any]) -> Path | None:
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            qhash = hashlib.sha1((question or "").encode("utf-8")).hexdigest()[:10]
            path = ensure_parent(Path(self.cfg.trace_dir) / f"{kind}_{stamp}_{qhash}.json")
            body = {
                "kind": kind,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "payload": self._sanitize(payload),
            }
            path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
            return path
        except Exception:
            return None

    def _sanitize(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(k): self._sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize(v) for v in list(value)[: self.cfg.max_trace_items * 3]]
        return repr(value)
