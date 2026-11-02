"""Структурированное JSONL-логирование запросов, retrieval и тестовых прогонов.

Модуль работает в best-effort режиме: если запись на диск недоступна,
основной сценарий приложения не ломается.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from core.paths import ensure_parent


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = False
    log_dir: str = '.chatkodi/logs/telemetry'
    benchmark_dir: str = '.chatkodi/logs/benchmarks'
    events_file: str = 'events.jsonl'
    capture_question: bool = True
    capture_response_preview: bool = True
    capture_context_preview: bool = False
    max_preview_chars: int = 500
    include_environment: bool = True
    session_separate_file: bool = True


class TelemetryLogger:
    """Пишет структурированные события в JSONL для последующего разбора."""

    def __init__(self, cfg: TelemetryConfig | None = None):
        self.cfg = cfg or TelemetryConfig()
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        self._lock = threading.Lock()
        self._env_written = False

    @classmethod
    def from_config_data(cls, data: Mapping[str, Any] | None) -> 'TelemetryLogger':
        data = data or {}
        features = data.get('features', {}) if isinstance(data, Mapping) else {}
        telemetry = data.get('telemetry', {}) if isinstance(data, Mapping) else {}
        enabled = bool(telemetry.get('enabled', False) or features.get('telemetry_enabled', False))
        cfg = TelemetryConfig(
            enabled=enabled,
            log_dir=str(telemetry.get('log_dir', '.chatkodi/logs/telemetry')),
            benchmark_dir=str(telemetry.get('benchmark_dir', '.chatkodi/logs/benchmarks')),
            events_file=str(telemetry.get('events_file', 'events.jsonl')),
            capture_question=bool(telemetry.get('capture_question', True)),
            capture_response_preview=bool(telemetry.get('capture_response_preview', True)),
            capture_context_preview=bool(telemetry.get('capture_context_preview', False)),
            max_preview_chars=int(telemetry.get('max_preview_chars', 500) or 500),
            include_environment=bool(telemetry.get('include_environment', True)),
            session_separate_file=bool(telemetry.get('session_separate_file', True)),
        )
        return cls(cfg)

    def preview(self, text: str | None, max_chars: int | None = None) -> str:
        text = text or ''
        compact = ' '.join(text.split())
        limit = int(max_chars or self.cfg.max_preview_chars)
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)] + '...'

    def _sanitize(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(k): self._sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize(v) for v in list(value)]
        return repr(value)

    def _event_path(self) -> Path:
        return ensure_parent(Path(self.cfg.log_dir) / self.cfg.events_file)

    def _session_path(self) -> Path:
        return ensure_parent(Path(self.cfg.log_dir) / f'session_{self.session_id}.jsonl')

    def _write_line(self, path: Path, payload: Mapping[str, Any]) -> None:
        with path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def _ensure_environment_snapshot(self) -> None:
        if not self.cfg.enabled or not self.cfg.include_environment or self._env_written:
            return
        self._env_written = True
        self.event('environment', {
            'python': sys.version,
            'python_executable': sys.executable,
            'platform': platform.platform(),
            'machine': platform.machine(),
            'pid': os.getpid(),
            'cwd': os.getcwd(),
        })

    def event(self, kind: str, payload: Mapping[str, Any] | None = None) -> Path | None:
        if not self.cfg.enabled:
            return None
        payload = payload or {}
        body = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'session_id': self.session_id,
            'kind': kind,
            'payload': self._sanitize(payload),
        }
        try:
            with self._lock:
                path = self._event_path()
                self._write_line(path, body)
                if self.cfg.session_separate_file:
                    self._write_line(self._session_path(), body)
            return path
        except Exception:
            return None

    def log_query_received(self, request_id: str, mode: str, question: str) -> None:
        self._ensure_environment_snapshot()
        payload = {
            'request_id': request_id,
            'mode': mode,
        }
        if self.cfg.capture_question:
            payload['question'] = question
            payload['question_preview'] = self.preview(question)
        self.event('query_received', payload)

    def log_query_finished(self, request_id: str, mode: str, question: str, response: str, elapsed_ms: float, extra: Mapping[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            'request_id': request_id,
            'mode': mode,
            'elapsed_ms': round(float(elapsed_ms), 2),
            'response_chars': len(response or ''),
        }
        if self.cfg.capture_question:
            payload['question'] = question
            payload['question_preview'] = self.preview(question)
        if self.cfg.capture_response_preview:
            payload['response_preview'] = self.preview(response)
        if extra:
            payload.update(self._sanitize(dict(extra)))
        self.event('query_finished', payload)

    def log_query_failed(self, request_id: str, mode: str, question: str, elapsed_ms: float, error: str) -> None:
        payload = {
            'request_id': request_id,
            'mode': mode,
            'elapsed_ms': round(float(elapsed_ms), 2),
            'error': error,
        }
        if self.cfg.capture_question:
            payload['question'] = question
            payload['question_preview'] = self.preview(question)
        self.event('query_failed', payload)

    def log_search(self, question: str, payload: Mapping[str, Any]) -> None:
        body = dict(payload)
        if self.cfg.capture_question:
            body['question'] = question
            body['question_preview'] = self.preview(question)
        self.event('hybrid_search', body)

    def log_extract(self, question: str, payload: Mapping[str, Any]) -> None:
        body = dict(payload)
        if self.cfg.capture_question:
            body['question'] = question
            body['question_preview'] = self.preview(question)
        self.event('extract_result', body)

    def log_rag(self, question: str, payload: Mapping[str, Any]) -> None:
        body = dict(payload)
        if self.cfg.capture_question:
            body['question'] = question
            body['question_preview'] = self.preview(question)
        self.event('rag_result', body)

    def log_benchmark_run(self, payload: Mapping[str, Any]) -> Path | None:
        if not self.cfg.enabled:
            return None
        try:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = ensure_parent(Path(self.cfg.benchmark_dir) / f'benchmark_{stamp}.json')
            body = {
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'session_id': self.session_id,
                'payload': self._sanitize(payload),
            }
            path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding='utf-8')
            self.event('benchmark_run', {'path': str(path), 'summary': payload.get('summary', {}), 'cases': payload.get('cases_count')})
            return path
        except Exception:
            return None
