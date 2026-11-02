"""Path helpers for project files and local runtime artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

RUNTIME_DIR_ENV = "CHATKODI_RUNTIME_DIR"
DEFAULT_RUNTIME_DIRNAME = ".chatkodi"
PRODUCTION_ENV = "CHATKODI_PRODUCTION"
APP_ROOT_ENV = "CHATKODI_APP_ROOT"
PORTABLE_ENV = "CHATKODI_PORTABLE"
PORTABLE_MARKER = "portable.flag"
PORTABLE_RUNTIME_DIRNAME = "runtime"
PRODUCTION_RUNTIME_DIRNAME = "ChatKodi"
TRUST_PATH_OVERRIDES_ENV = "CHATKODI_TRUST_PATH_OVERRIDES"
EXTRA_MODEL_CACHE_DIRS_ENV = "CHATKODI_EXTRA_MODEL_CACHE_DIRS"
_RUNTIME_DIR_ALIASES = {".chatkodi", ".ChatKodi", ".CHATKODI"}
_RUNTIME_RELATIVE_DIRS = {
    "background_processing",
    "knowledge",
    "logs",
    "models_cache",
}
_RUNTIME_RELATIVE_FILES = {
    "chatkodi_settings.json",
    "device_config.json",
    "documents_index.json",
    "embedding_cache.npz",
    "embedding_cache_meta.json",
    "faiss_index.bin",
    "faiss_meta.pkl",
    "llm_analysis_cache.json",
    "synonyms_cache.pkl",
    "events.db",
    "events_meta.json",
    "query_history.jsonl",
    "chat_history.jsonl",
}


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _safe_resolve(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except Exception:
        return path.expanduser().absolute()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except (ValueError, OSError):
        return False


def project_root() -> Path:
    """Return the source or installed application root directory.

    A frozen onedir build is frequently copied to another disk or computer.
    In that mode the executable directory is authoritative: a stale global
    ``CHATKODI_APP_ROOT`` must not send the app back to a previous location.
    Non-local overrides remain possible for advanced layouts by explicitly
    setting ``CHATKODI_TRUST_PATH_OVERRIDES=1``.
    """
    configured = os.environ.get(APP_ROOT_ENV)
    if getattr(sys, "frozen", False):
        executable_root = _safe_resolve(Path(sys.executable).parent)
        if configured:
            configured_root = _safe_resolve(Path(configured))
            if configured_root == executable_root or _env_flag(TRUST_PATH_OVERRIDES_ENV):
                return configured_root
        return executable_root
    if configured:
        return _safe_resolve(Path(configured))
    return Path(__file__).resolve().parent.parent


def is_production_runtime() -> bool:
    """Return True when running from an installed/frozen production build."""
    flag = str(os.environ.get(PRODUCTION_ENV, "")).strip().lower()
    return getattr(sys, "frozen", False) or flag in {"1", "true", "yes", "on"}


def is_portable_runtime() -> bool:
    """Return True for a portable zip layout extracted beside the executables."""
    flag = str(os.environ.get(PORTABLE_ENV, "")).strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        if not getattr(sys, "frozen", False) or _env_flag(TRUST_PATH_OVERRIDES_ENV):
            return True
        # A copied installed build can inherit a stale machine-level flag.
        # Frozen portable mode therefore also requires its local marker.
        return (project_root() / PORTABLE_MARKER).is_file()
    try:
        return (project_root() / PORTABLE_MARKER).exists()
    except Exception:
        return False


def default_production_runtime_root() -> Path:
    """Return the default writable runtime root for installed Windows builds."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return (Path(base) / PRODUCTION_RUNTIME_DIRNAME).resolve()
    return (Path.home() / f".{PRODUCTION_RUNTIME_DIRNAME.lower()}").resolve()


def runtime_root() -> Path:
    """Return the base directory for local runtime artifacts."""
    configured_env = os.environ.get(RUNTIME_DIR_ENV)
    if is_portable_runtime():
        portable_default = project_root() / PORTABLE_RUNTIME_DIRNAME
        if configured_env:
            requested = _safe_resolve(Path(configured_env))
            if _is_under(requested, project_root()) or _env_flag(TRUST_PATH_OVERRIDES_ENV):
                configured = requested
            else:
                configured = portable_default
        else:
            configured = portable_default
    elif configured_env:
        configured = Path(configured_env)
    elif is_production_runtime():
        configured = default_production_runtime_root()
    else:
        configured = Path(DEFAULT_RUNTIME_DIRNAME)
    if configured.is_absolute():
        return configured.resolve()
    return (project_root() / configured).resolve()


def _uses_runtime_root(path: Path) -> bool:
    """Return True when a relative path should live under the runtime folder."""
    if not path.parts:
        return False
    head = path.parts[0]
    if is_production_runtime() and not is_portable_runtime() and head in {"docs", "documents"}:
        return True
    return head in _RUNTIME_DIR_ALIASES or head in _RUNTIME_RELATIVE_DIRS or head in _RUNTIME_RELATIVE_FILES


def resolve_project_path(path: PathLike) -> Path:
    """Resolve absolute paths and anchor relative paths to the proper root.

    Backward compatibility note: older builds sometimes passed paths as
    ``.chatkodi/logs/...`` while newer code passes ``logs/...``.  Both forms
    must resolve to the same runtime directory, never to
    ``.chatkodi/.chatkodi/...``.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in _RUNTIME_DIR_ALIASES:
        p = Path(*p.parts[1:]) if len(p.parts) > 1 else Path()
        return (runtime_root() / p).resolve()
    base = runtime_root() if _uses_runtime_root(p) else project_root()
    return (base / p).resolve()




def runtime_model_cache_dir() -> Path:
    """Return the canonical runtime models cache directory."""
    return (runtime_root() / "models_cache").resolve()


def legacy_model_cache_dir() -> Path:
    """Return the legacy project-root models cache directory."""
    return (project_root() / "models_cache").resolve()


def model_cache_search_dirs() -> list[Path]:
    """Return all known model cache directories, preserving priority order."""
    result: list[Path] = []
    for candidate in (runtime_model_cache_dir(), legacy_model_cache_dir()):
        if candidate not in result:
            result.append(candidate)
    # A clean source checkout may deliberately share a verified, read-only
    # model payload with a sibling checkout during diagnostics/builds.
    # Frozen applications ignore this override unless path overrides were
    # explicitly trusted, so a stale machine variable cannot redirect a release.
    extra = os.environ.get(EXTRA_MODEL_CACHE_DIRS_ENV, "")
    if extra and (not getattr(sys, "frozen", False) or _env_flag(TRUST_PATH_OVERRIDES_ENV)):
        for raw in extra.split(os.pathsep):
            raw = raw.strip()
            if not raw:
                continue
            candidate = _safe_resolve(Path(raw))
            if candidate not in result:
                result.append(candidate)
    return result


def _cache_dir_has_model_content(path: Path) -> bool:
    """Return True when the cache directory contains usable model payloads.

    We treat the directory as useful only when it contains real embedding/LLM
    subfolders or model files. This avoids preferring an empty runtime cache
    over an older legacy cache that still stores the actual models.
    """
    if not path.exists() or not path.is_dir():
        return False

    direct_markers = {
        "config.json",
        "generation_config.json",
        "modules.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer.model",
    }

    try:
        for item in path.iterdir():
            if item.name in {"llms", "embeddings"} and item.is_dir():
                for nested in item.iterdir():
                    if nested.is_dir():
                        return True
            if item.is_file() and item.name in direct_markers:
                return True
            if item.is_dir():
                for marker in direct_markers:
                    if (item / marker).exists():
                        return True
    except Exception:
        return False

    return False


def resolve_model_cache_dir(path: PathLike | None = None, *, prefer_existing: bool = True) -> Path:
    """Resolve the models cache directory with backward compatibility.

    If an explicit path is given, it is resolved normally. Otherwise the runtime
    cache is preferred, but an existing legacy ``models_cache`` in the project
    root is reused to preserve compatibility with older releases.
    """
    if path:
        return resolve_project_path(path)

    runtime_dir = runtime_model_cache_dir()
    legacy_dir = legacy_model_cache_dir()

    if prefer_existing:
        runtime_has_content = _cache_dir_has_model_content(runtime_dir)
        legacy_has_content = _cache_dir_has_model_content(legacy_dir)

        if runtime_has_content:
            return runtime_dir
        if legacy_has_content:
            return legacy_dir
        if runtime_dir.exists():
            return runtime_dir
        if legacy_dir.exists():
            return legacy_dir
    return runtime_dir

def ensure_parent(path: PathLike) -> Path:
    """Resolve a path and create its parent directory when needed."""
    p = resolve_project_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
