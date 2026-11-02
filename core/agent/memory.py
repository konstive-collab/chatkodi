"""Memory between chats: stable facts about the user and their work
(organisation, position, regular addressees, preferences), kept in
runtime/memory.json, shown and editable in the settings, added to every chat.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import List

from core.paths import runtime_root

MAX_FACTS = 40

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
}

EXTRACT_PROMPT = """Выпиши из сообщения пользователя устойчивые факты, которые пригодятся в будущих разговорах:
кто он (ФИО, должность), его организация и отдел, постоянные адресаты и контрагенты, предпочтения в оформлении,
просьбы «запомни …». Каждый факт — самостоятельное предложение от третьего лица («Пользователь — Иванов Пётр
Ильич, инженер ПТО ООО «Ромашка»»). Сохраняй все детали дословно: ФИО полностью, должность, организацию, правила
подписи и оформления — ничего не сокращай и не пропускай. Разовые задачи, вопросы и содержимое документов не выписывай.
Если таких фактов нет — пустой список.

Уже известно:
{known}

Сообщение: {message}
"""


@dataclass
class Fact:
    text: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created: float = field(default_factory=time.time)


class MemoryStore:
    def __init__(self, path=None):
        self.path = path or (runtime_root() / "memory.json")
        self._lock = threading.Lock()
        self.facts: List[Fact] = []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.facts = [Fact(**item) for item in data.get("facts", [])]
        except Exception:
            self.facts = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"facts": [asdict(f) for f in self.facts]}, ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def _key(text: str) -> str:
        return re.sub(r"[^а-яa-z0-9]+", " ", text.lower().replace("ё", "е")).strip()

    def add(self, text: str) -> bool:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if len(text) < 8:
            return False
        with self._lock:
            key = self._key(text)
            if any(self._key(f.text) == key or key in self._key(f.text) for f in self.facts):
                return False
            self.facts.append(Fact(text[:300]))
            self.facts = self.facts[-MAX_FACTS:]
            self._save()
            return True

    def remove(self, fact_id: str) -> None:
        with self._lock:
            self.facts = [f for f in self.facts if f.id != fact_id]
            self._save()

    def clear(self) -> None:
        with self._lock:
            self.facts = []
            self._save()

    def prompt_block(self) -> str:
        if not self.facts:
            return ""
        return "Что известно о пользователе из прошлых разговоров (учитывай, если уместно):\n" + "\n".join(f"- {f.text}" for f in self.facts)

_SHARED = None


def get_memory() -> MemoryStore:
    """One store for all chats of this run (facts added in one chat are seen in the others)."""
    global _SHARED
    if _SHARED is None:
        _SHARED = MemoryStore()
    return _SHARED
