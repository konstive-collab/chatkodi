"""Personal instructions: the user's standing rules for every answer
(style, mandatory requisites, language, signature). Stored as plain text."""

from __future__ import annotations

from pathlib import Path

from core.paths import runtime_root

MAX_CHARS = 4000


def instructions_path() -> Path:
    return runtime_root() / "personal_instructions.md"


def load_instructions() -> str:
    try:
        return instructions_path().read_text(encoding="utf-8").strip()[:MAX_CHARS]
    except OSError:
        return ""


def save_instructions(text: str) -> None:
    path = instructions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((text or "").strip()[:MAX_CHARS] + "\n", encoding="utf-8")


def system_message(text: str) -> str:
    if not text.strip():
        return ""
    return (
        "Личные инструкции пользователя — соблюдай их во всех ответах, "
        "если они не противоречат документам и фактам:\n" + text.strip()
    )
