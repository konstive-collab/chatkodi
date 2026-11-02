"""Atomic UTF-8 JSON persistence for small runtime configuration files."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, payload: Any, *, backup: bool = True) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    backup_path = target.with_name(target.name + ".bak")
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if backup and target.is_file():
            shutil.copy2(target, backup_path)
        os.replace(temporary, target)
        return target
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass


def read_json_with_backup(path: str | Path, *, default: Any = None) -> Any:
    target = Path(path)
    for candidate in (target, target.with_name(target.name + ".bak")):
        try:
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
    return default
