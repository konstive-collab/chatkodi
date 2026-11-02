"""Local, bounded conversation history and transcript export."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from core.logging_utils import get_logger
from core.paths import resolve_project_path

logger = get_logger(__name__)


class ChatHistoryStore:
    def __init__(self, path: str | Path = "chat_history.jsonl", max_entries: int = 200) -> None:
        self.path = resolve_project_path(path)
        self.max_entries = max(10, int(max_entries))
        self._lock = threading.RLock()

    def load(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.path.exists():
                return []
            entries: list[dict[str, Any]] = []
            try:
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed chat-history record in %s", self.path)
                        continue
                    if item.get("role") in {"user", "assistant"} and str(item.get("text") or "").strip():
                        entries.append(item)
            except OSError as exc:
                logger.warning("Unable to read chat history %s: %s", self.path, exc)
                return []
            return entries[-self.max_entries :]

    def append(self, role: str, text: str, mode: str = "") -> dict[str, Any] | None:
        clean = str(text or "").strip()
        if role not in {"user", "assistant"} or not clean:
            return None
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "mode": str(mode or ""),
            "text": clean,
        }
        with self._lock:
            entries = self.load()
            entries.append(entry)
            self._write(entries[-self.max_entries :])
        return entry

    def clear(self) -> None:
        with self._lock:
            self._write([])

    def export(self, destination: str | Path, entries: Iterable[dict[str, Any]] | None = None) -> Path:
        target = Path(destination).expanduser().resolve()
        records = list(entries) if entries is not None else self.load()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".json":
            payload = json.dumps(records, ensure_ascii=False, indent=2)
        else:
            lines = []
            for item in records:
                label = "Вы" if item.get("role") == "user" else "ChatKodi"
                timestamp = str(item.get("timestamp") or "")
                lines.append(f"[{timestamp}] {label}\n{item.get('text', '')}")
            payload = "\n\n".join(lines).rstrip() + ("\n" if lines else "")
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(payload, encoding="utf-8")
        temp.replace(target)
        return target

    def _write(self, entries: Iterable[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in entries)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(payload, encoding="utf-8")
        temp.replace(self.path)
