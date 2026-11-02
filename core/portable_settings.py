"""Helpers for portable runtime settings that can survive folder moves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.paths import is_portable_runtime, project_root, resolve_project_path

SETTINGS_APP_ROOT_KEY = "app_root"


def current_settings_app_root() -> str:
    """Return the current application root value stored with portable settings."""
    return str(project_root())


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except Exception:
        return path.absolute()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
    except Exception:
        return False


def normalize_docs_folder_setting(
    raw_value: Any,
    *,
    stored_app_root: Any = None,
    default: str = "docs",
) -> str:
    """Normalize a saved docs folder path for the current runtime.

    Portable builds should not inherit an absolute docs path from another copy of
    the app. A user-selected external folder is still respected when it was saved
    by the same portable app root.
    """
    value = str(raw_value or "").strip()
    if not value:
        return default

    if not is_portable_runtime():
        return value

    current_root = _safe_resolve(project_root())
    try:
        saved_root = _safe_resolve(Path(str(stored_app_root))) if stored_app_root else None
    except Exception:
        saved_root = None

    candidate = Path(value).expanduser()
    resolved_candidate = _safe_resolve(candidate if candidate.is_absolute() else resolve_project_path(candidate))
    saved_by_current_copy = bool(saved_root and saved_root == current_root)
    outside_current_root = not _is_under(resolved_candidate, current_root)

    if candidate.is_absolute():
        if not resolved_candidate.exists():
            return default
        if outside_current_root and not saved_by_current_copy:
            return default

    return value
