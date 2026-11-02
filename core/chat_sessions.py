"""Several chats: each one is a ChatHistoryStore file plus a line in index.json
(title, times, pinned, status mark)."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from core.chat_history import ChatHistoryStore
from core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ChatInfo:
    id: str
    title: str = "Новый чат"
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    pinned: bool = False
    status: str = ""  # "" | "ok" (all verified) | "warn" (⚠ / несоответствия)
    auto_title: bool = True


def title_from_text(text: str, limit: int = 42) -> str:
    """A short chat name from the first request: «Проверь протокол испытаний…»."""
    words, files = [], []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("📎"):
            files.append(Path(stripped.lstrip("📎 ").strip()).name)
        elif stripped:
            words.append(stripped)
    if not words and files:
        name = ", ".join(files)  # a file name keeps its own spelling
        return name if len(name) <= limit else name[: limit - 1] + "…"
    clean = re.sub(r"\s+", " ", " ".join(words)).strip(" .,!?:;—-")
    if not clean:
        return "Новый чат"
    if len(clean) <= limit:
        return clean[:1].upper() + clean[1:]
    cut = clean[:limit]
    cut = cut[: cut.rfind(" ")] if " " in cut[limit // 2:] else cut
    return cut[:1].upper() + cut[1:] + "…"


class ChatSessionStore:
    def __init__(self, root: Path, max_entries: int = 400):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self.chats: Dict[str, ChatInfo] = {}
        self._load()

    # ---------------------------------------------------------------- index

    def _load(self) -> None:
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            for item in data.get("chats", []):
                info = ChatInfo(**{k: v for k, v in item.items() if k in ChatInfo.__dataclass_fields__})
                self.chats[info.id] = info
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Chat index is damaged, starting a new one: %s", exc)

    def _save(self) -> None:
        with self._lock:
            payload = {"chats": [asdict(c) for c in self.chats.values()]}
            tmp = self.index_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.index_path)

    # ---------------------------------------------------------------- chats

    def ordered(self, query: str = "") -> List[ChatInfo]:
        items = list(self.chats.values())
        if query.strip():
            q = query.lower().replace("ё", "е")
            items = [c for c in items if q in c.title.lower().replace("ё", "е") or q in self._text(c.id)]
        return sorted(items, key=lambda c: (not c.pinned, -c.updated))

    def _text(self, chat_id: str) -> str:
        try:
            return self.history_path(chat_id).read_text(encoding="utf-8").lower().replace("ё", "е")
        except OSError:
            return ""

    def history_path(self, chat_id: str) -> Path:
        return self.root / f"{chat_id}.jsonl"

    def history(self, chat_id: str) -> ChatHistoryStore:
        return ChatHistoryStore(self.history_path(chat_id), max_entries=self.max_entries)

    def create(self, title: str = "Новый чат") -> ChatInfo:
        info = ChatInfo(id=time.strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:6], title=title)
        self.chats[info.id] = info
        self._save()
        return info

    def touch(self, chat_id: str, *, first_text: str = "", status: Optional[str] = None) -> None:
        info = self.chats.get(chat_id)
        if info is None:
            return
        # the clock ticks in ~16 ms on Windows: the chat touched last must still sort first
        info.updated = max([time.time()] + [c.updated + 0.001 for c in self.chats.values() if c is not info])
        if first_text and info.auto_title and info.title == "Новый чат":
            info.title = title_from_text(first_text)
        if status is not None:
            info.status = status
        self._save()

    def rename(self, chat_id: str, title: str) -> None:
        info = self.chats.get(chat_id)
        if info and title.strip():
            info.title = title.strip()[:80]
            info.auto_title = False
            self._save()

    def set_pinned(self, chat_id: str, pinned: bool) -> None:
        info = self.chats.get(chat_id)
        if info:
            info.pinned = pinned
            self._save()

    def delete(self, chat_id: str) -> None:
        self.chats.pop(chat_id, None)
        try:
            self.history_path(chat_id).unlink()
        except OSError:
            pass
        self._save()

    def is_empty(self, chat_id: str) -> bool:
        return not self.history(chat_id).load()

    def import_legacy(self, legacy: ChatHistoryStore) -> Optional[ChatInfo]:
        """The single history of 2.1 becomes the first chat."""
        entries = legacy.load()
        if not entries or self.chats:
            return None
        first = next((e.get("text", "") for e in entries if e.get("role") == "user"), "")
        info = self.create(title_from_text(first) if first else "Предыдущий диалог")
        store = self.history(info.id)
        store._write(entries)
        return info
