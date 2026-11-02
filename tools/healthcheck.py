#!/usr/bin/env python3
"""Быстрая самопроверка проекта без запуска GUI."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import contextlib
import io
import json
import os
import sys
from typing import Any, Dict

from core.config import get_config
from core.feature_flags import FeatureFlags
from core.logging_utils import setup_logging
from core.resilience import ResiliencePolicy
from core.system_profile import detect_system_profile
from core.paths import is_production_runtime, project_root, resolve_project_path

logger = setup_logging()


def _check_python() -> Dict[str, Any]:
    version = sys.version_info
    return {
        "ok": version >= (3, 10),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "is_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
    }


def _check_project_files() -> Dict[str, Any]:
    root = project_root()
    if is_production_runtime():
        required = [
            root / "ChatKodiApp.exe",
            root / "ChatKodiLauncher.exe",
            root / "manuals",
        ]
    else:
        required = [
            root / "main.py",
            root / "core" / "search.py",
            root / "core" / "chat_mode.py",
            root / "gui" / "app.py",
        ]
    missing = [str(p.relative_to(root)) for p in required if not p.exists()]
    return {"ok": not missing, "root": str(root), "missing": missing}


def _check_docs() -> Dict[str, Any]:
    from core.documents.processor import DocumentProcessor, SUPPORTED_EXTENSIONS

    docs_dir = resolve_project_path("docs")
    files = []
    scan_warnings = []
    if docs_dir.exists():
        processor = DocumentProcessor(str(docs_dir), recursive=True)
        files = [str(path.relative_to(docs_dir)) for path in processor._collect_all_files()]
        scan_warnings = list(processor.stats.get("scan_warnings") or [])
    return {
        "ok": docs_dir.exists() or is_production_runtime(),
        "path": str(docs_dir),
        "count": len(files),
        "examples": files[:10],
        "supported_extensions": len(SUPPORTED_EXTENSIONS),
        "scan_warnings": scan_warnings[:10],
    }


def _check_models() -> Dict[str, Any]:
    try:
        from download_models import check_models_availability
        with contextlib.redirect_stdout(io.StringIO()):
            result = check_models_availability()
        payload = {
            "ok": bool(result.get("embeddings")),
            "embeddings": bool(result.get("embeddings")),
            "embeddings_path": result.get("embeddings_path"),
            "llm": bool(result.get("llm")),
            "llm_path": result.get("llm_path"),
            "llm_name": result.get("llm_name"),
            "llm_size": result.get("llm_size"),
            "has_cuda": bool(result.get("has_cuda", False)),
        }
        try:
            from core.llm.server import find_llama_server
            from core.model_integrity import inspect_gguf_model

            server = find_llama_server()
            payload["llama_server"] = str(server) if server else None
            embedding_path = payload.get("embeddings_path")
            llm_path = payload.get("llm_path")
            if embedding_path:
                payload["embeddings_integrity"] = inspect_gguf_model(embedding_path)
                payload["embeddings"] = bool(payload["embeddings_integrity"].get("ok"))
            if llm_path:
                payload["llm_integrity"] = inspect_gguf_model(llm_path)
                payload["llm"] = bool(payload["llm_integrity"].get("ok"))
            # Without llama-server neither search nor answers can start.
            payload["ok"] = bool(payload.get("embeddings")) and server is not None
        except Exception as exc:
            payload["integrity_error"] = str(exc)
        return payload
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_runtime_profile(cfg_data: Dict[str, Any]) -> Dict[str, Any]:
    profile = detect_system_profile(cfg_data)
    policy = ResiliencePolicy.from_config_data(cfg_data)
    return {
        "ok": True,
        "profile_name": profile.profile_name,
        "summary": profile.summary_line,
        "has_cuda": profile.has_cuda,
        "gpu_vram_gb": profile.gpu_vram_gb,
        "total_ram_gb": profile.total_ram_gb,
        "llm_startup_mode": profile.llm_startup_mode,
        "context_char_limit": profile.context_char_limit,
        "safe_mode_enabled": policy.safe_mode_enabled,
    }


def run_healthcheck() -> Dict[str, Any]:
    cfg = get_config()
    flags = FeatureFlags.from_config_data(cfg.data)
    from core.document_dependencies import check_document_readers

    payload = {
        "document_readers": check_document_readers(),
        "project_version": cfg.data.get("version", "unknown"),
        "python": _check_python(),
        "project": _check_project_files(),
        "documents": _check_docs(),
        "models": _check_models(),
        "runtime_profile": _check_runtime_profile(cfg.data),
        "features": {
            "plugin_loader_enabled": flags.plugin_loader_enabled,
            "diagnostics_enabled": flags.diagnostics_enabled,
            "healthcheck_cli_enabled": flags.healthcheck_cli_enabled,
            "resilience_enabled": flags.resilience_enabled,
            "auto_profile_enabled": flags.auto_profile_enabled,
        },
    }
    payload["ok"] = all(
        bool(section.get("ok", True))
        for key, section in payload.items()
        if isinstance(section, dict) and key not in {"features"}
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка окружения ChatKodi без запуска GUI")
    parser.add_argument("--json", action="store_true", help="вывести результат в JSON")
    args = parser.parse_args()
    report = run_healthcheck()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"ChatKodi healthcheck | version={report['project_version']}")
        print(f"Python: {report['python']['python']} | venv={report['python']['is_venv']}")
        print(f"Project root: {report['project']['root']}")
        print(f"Documents: {report['documents']['count']} files")
        print(f"Embeddings: {report['models'].get('embeddings')} | LLM: {report['models'].get('llm')}")
        print(f"Runtime profile: {report['runtime_profile']['profile_name']} | llm_startup={report['runtime_profile']['llm_startup_mode']}")
        if report['project']['missing']:
            print("Missing files:", ", ".join(report['project']['missing']))
        if report['models'].get('error'):
            print("Model check error:", report['models']['error'])
        print("OK" if report['ok'] else "ISSUES_FOUND")
    return 0 if report['ok'] else 1


if __name__ == "__main__":
    raise SystemExit(main())
