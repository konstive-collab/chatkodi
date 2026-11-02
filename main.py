#!/usr/bin/env python3
"""Точка входа desktop-приложения ChatKodi.

Этот модуль держит стартовую последовательность в одном месте:
- подготавливает окружение Qt;
- импортирует GUI только после настройки среды;
- при необходимости безопасно подгружает плагины;
- запускает цикл событий Qt и пишет ошибки старта в лог.
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path

def _configure_production_environment() -> None:
    if not getattr(sys, "frozen", False):
        return
    app_root = str(Path(sys.executable).resolve().parent)
    os.environ["CHATKODI_PRODUCTION"] = "1"
    os.environ["CHATKODI_APP_ROOT"] = app_root
    if (Path(app_root) / "portable.flag").exists():
        os.environ["CHATKODI_PORTABLE"] = "1"
        os.environ["CHATKODI_RUNTIME_DIR"] = str(Path(app_root) / "runtime")
    else:
        os.environ.pop("CHATKODI_PORTABLE", None)


def _startup_crash_log_path() -> Path:
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
    else:
        root = Path(__file__).resolve().parent
    if (root / "portable.flag").is_file():
        runtime = root / "runtime"
    elif sys.platform == "win32" and getattr(sys, "frozen", False):
        runtime = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or Path.home()) / "ChatKodi"
    else:
        runtime = root / ".chatkodi"
    path = runtime / "logs" / "app_startup_crash.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _report_startup_crash(exc: BaseException, *, show_dialog: bool = True) -> None:
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        path = _startup_crash_log_path()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] app startup failed\n{details}\n")
    except Exception:
        path = None

    if show_dialog and sys.platform == "win32" and getattr(sys, "frozen", False):
        try:
            import ctypes

            suffix = f"\n\nДиагностический лог: {path}" if path else ""
            ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
                None,
                f"ChatKodi не смог запуститься.\n\n{exc}{suffix}",
                "Ошибка запуска ChatKodi",
                0x10,
            )
        except Exception:
            pass


_configure_production_environment()

try:
    from core.config import get_config
    from core.system_profile import apply_runtime_hints, detect_system_profile
    from core.logging_utils import setup_logging
except BaseException as import_error:
    _report_startup_crash(import_error, show_dialog=not any(arg.startswith("--") for arg in sys.argv[1:]))
    raise


logger = setup_logging()


def _configure_qt_environment() -> None:
    """Настраивает переменные среды Qt, необходимые для Windows-сборки приложения."""
    if sys.platform != "win32":
        return

    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
        for candidate in (
            base / "_internal" / "PyQt5" / "Qt5" / "plugins",
            base / "_internal" / "PyQt5" / "Qt" / "plugins",
            base / "PyQt5" / "Qt5" / "plugins",
            base / "PyQt5" / "Qt" / "plugins",
            base / "_internal" / "qt_plugins",
            base / "qt_plugins",
        ):
            if candidate.exists():
                os.environ["QT_PLUGIN_PATH"] = str(candidate)
                break
        os.environ["QT_QPA_PLATFORM"] = "windows"
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
        return

    qt_plugin_path = os.path.join(
        os.path.dirname(sys.executable),
        "Lib",
        "site-packages",
        "PyQt5",
        "Qt",
        "plugins",
    )
    if os.path.exists(qt_plugin_path):
        os.environ["QT_PLUGIN_PATH"] = qt_plugin_path
    else:
        venv_qt_plugins = os.path.join(
            os.path.dirname(os.path.dirname(sys.executable)),
            "Lib",
            "site-packages",
            "PyQt5",
            "Qt",
            "plugins",
        )
        if os.path.exists(venv_qt_plugins):
            os.environ["QT_PLUGIN_PATH"] = venv_qt_plugins

    os.environ["QT_QPA_PLATFORM"] = "windows"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"


def _bootstrap_plugins() -> None:
    """Безопасно пытается подгрузить плагины, если пользователь их включил."""
    try:
        from core.plugins import load_plugins_from_config

        cfg = get_config()
        registry = load_plugins_from_config(cfg.data)
        if registry.plugins:
            logger.info("Plugins loaded: %s", ", ".join(p.name for p in registry.plugins))
        for err in registry.errors:
            logger.warning("Plugin error: %s", err)
    except Exception as exc:
        logger.warning("Plugin bootstrap skipped: %s", exc)


def _run_doctor_json() -> int:
    from tools.doctor import build_doctor_report

    report = build_doctor_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("doctor", {}).get("status") == "ok" else 1


def _run_llm_smoke() -> int:
    from core.local_llm import load_llm_model
    from core.evidence_reasoner import GroundingEvidence, reason_over_evidence

    payload = {"ok": False}
    started = time.perf_counter()
    llm = load_llm_model(lightweight=True)
    payload["load_seconds"] = round(time.perf_counter() - started, 2)
    payload["loaded"] = bool(llm and llm.is_available())
    if llm:
        payload["model"] = llm.get_model_info()
    else:
        from core.llm.runtime import describe_llm_status, get_last_llm_load_status

        payload["status"] = get_last_llm_load_status()
        payload["message"] = describe_llm_status()
    if llm and llm.is_available():
        gen_started = time.perf_counter()
        answer = llm.generate_clean_answer(
            "Ответь одним числом: 2+2?",
            rag_mode=False,
            use_cache=False,
        )
        payload["generate_seconds"] = round(time.perf_counter() - gen_started, 2)
        payload["answer"] = str(answer)
        arithmetic_ok = "4" in str(answer)

        grounded_started = time.perf_counter()
        evidence = [
            GroundingEvidence(
                evidence_id="E1",
                source="synthetic_policy.txt",
                text="Регламент ORION: резервное копирование выполняется каждый четверг в 21:15.",
            ),
            GroundingEvidence(
                evidence_id="E2",
                source="synthetic_notes.txt",
                text="В заметке описано обновление интерфейса по понедельникам.",
            ),
        ]
        grounded = reason_over_evidence(
            llm,
            "Когда выполняется резервное копирование по регламенту ORION?",
            evidence,
            max_tokens=420,
            retry_invalid_json=True,
        )
        payload["grounded_seconds"] = round(time.perf_counter() - grounded_started, 2)
        payload["grounded"] = {
            "valid": grounded.valid,
            "status": grounded.status,
            "answer": grounded.answer,
            "confidence": grounded.confidence,
            "validation_error": grounded.validation_error,
            "claims": [
                {
                    "text": claim.text,
                    "evidence_ids": claim.evidence_ids,
                    "quotes": claim.quotes,
                }
                for claim in grounded.claims
            ],
        }
        grounded_text = " ".join(
            [grounded.answer]
            + [claim.text for claim in grounded.claims]
            + [quote for claim in grounded.claims for quote in claim.quotes]
        ).lower()
        grounded_ok = bool(
            grounded.valid
            and grounded.status == "supported"
            and "e1" in {item.lower() for claim in grounded.claims for item in claim.evidence_ids}
            and "четверг" in grounded_text
            and "21:15" in grounded_text
        )
        payload["ok"] = bool(arithmetic_ok and grounded_ok)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("ok") else 1


def _run_search_benchmark() -> int:
    from tools.run_benchmark import run_benchmark

    flag_index = sys.argv.index("--search-benchmark")
    try:
        cases_path = Path(sys.argv[flag_index + 1])
    except IndexError:
        print(json.dumps({"ok": False, "error": "--search-benchmark requires a cases JSON path"}, ensure_ascii=False))
        return 2

    docs_folder = "docs"
    if "--docs-folder" in sys.argv:
        docs_index = sys.argv.index("--docs-folder")
        try:
            docs_folder = sys.argv[docs_index + 1]
        except IndexError:
            print(json.dumps({"ok": False, "error": "--docs-folder requires a path"}, ensure_ascii=False))
            return 2

    payload = run_benchmark(cases_path, docs_folder=docs_folder, load_llm="--with-llm" in sys.argv)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("summary", {}).get("cases_failed", 1) == 0 else 1


def _run_docs_diagnose() -> int:
    from core.document_processor import DocumentProcessor
    from core.paths import project_root, runtime_root, resolve_project_path

    docs_folder = "docs"
    if "--docs-folder" in sys.argv:
        docs_index = sys.argv.index("--docs-folder")
        try:
            docs_folder = sys.argv[docs_index + 1]
        except IndexError:
            print(json.dumps({"ok": False, "error": "--docs-folder requires a path"}, ensure_ascii=False))
            return 2

    processor = DocumentProcessor(docs_folder=docs_folder)
    ok = processor.load_documents()
    payload = {
        "ok": bool(ok),
        "app_root": str(project_root()),
        "runtime_root": str(runtime_root()),
        "requested_docs_folder": docs_folder,
        "resolved_docs_folder": str(resolve_project_path(docs_folder)),
        "processor_docs_folder": str(processor.docs_folder),
        "exists": processor.docs_folder.exists(),
        "recursive": processor.recursive,
        "documents": len(processor.documents),
        "chunks": len(processor.text_chunks),
        "stats": processor.stats,
        "document_names": processor.get_document_names()[:50],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def _run_portable_verify() -> int:
    from core.paths import project_root
    from tools.portable_verify import verify_and_log

    report = verify_and_log(project_root(), full="--full" in sys.argv)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


def _run_generation_catalog() -> int:
    from core.document_catalog import DocumentCatalog
    from core.paths import resolve_project_path, runtime_root

    cfg = get_config().data
    generation_cfg = dict(cfg.get("generation") or {})
    requested = generation_cfg.get("catalog_root") or "documents"
    if "--catalog-documents" in sys.argv:
        index = sys.argv.index("--catalog-documents")
        if index + 1 < len(sys.argv) and not sys.argv[index + 1].startswith("--"):
            requested = sys.argv[index + 1]
    root = Path(requested)
    if not root.is_absolute():
        root = resolve_project_path(root)
    catalog = DocumentCatalog(root, runtime_root() / "document_catalog.json")
    catalog.scan()
    print(json.dumps({"ok": True, "root": str(root), "summary": catalog.summary()}, ensure_ascii=False, indent=2))
    return 0


def _verify_direct_portable_start() -> None:
    if not getattr(sys, "frozen", False):
        return
    root = Path(sys.executable).resolve().parent
    if not (root / "portable.flag").is_file():
        return
    if os.environ.get("CHATKODI_PORTABLE_VERIFIED"):
        return
    if str(os.environ.get("CHATKODI_SKIP_PORTABLE_VERIFY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    from tools.portable_verify import format_blocking_message, startup_preflight, write_preflight_log

    preflight = startup_preflight(root)
    log_path = write_preflight_log(root, preflight)
    if preflight.get("warnings"):
        logger.warning(
            "Portable preflight: %s non-blocking issue(s), details in %s",
            len(preflight["warnings"]),
            log_path,
        )
    if preflight.get("llm_degraded"):
        # LLM payload is damaged: start in search-only mode instead of crashing.
        os.environ["CHATKODI_LLM_PAYLOAD_DAMAGED"] = "1"
    if not preflight.get("ok"):
        raise RuntimeError(format_blocking_message(preflight))
    os.environ["CHATKODI_PORTABLE_VERIFIED"] = str(preflight.get("build_id") or "1")


def main() -> int:
    """Запускает GUI и возвращает код завершения процесса."""
    _configure_production_environment()
    if any(arg in sys.argv for arg in ("--doctor-json", "--llm-smoke", "--search-benchmark", "--docs-diagnose", "--verify-portable", "--catalog-documents")):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if "--doctor-json" in sys.argv:
        return _run_doctor_json()
    if "--llm-smoke" in sys.argv:
        return _run_llm_smoke()
    if "--search-benchmark" in sys.argv:
        return _run_search_benchmark()
    if "--docs-diagnose" in sys.argv:
        return _run_docs_diagnose()
    if "--verify-portable" in sys.argv:
        return _run_portable_verify()
    if "--catalog-documents" in sys.argv:
        return _run_generation_catalog()

    try:
        _verify_direct_portable_start()
    except Exception as exc:
        logger.exception("Portable preflight failed")
        _report_startup_crash(exc, show_dialog=True)
        return 1

    cfg = get_config()
    profile = detect_system_profile(cfg.data)
    applied_hints = apply_runtime_hints(profile, cfg.data)
    logger.info("Runtime profile selected: %s", profile.summary_line)
    if applied_hints:
        logger.info("Applied runtime hints: %s", ", ".join(f"{k}={v}" for k, v in applied_hints.items()))
    _configure_qt_environment()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _bootstrap_plugins()

    try:
        from PyQt5.QtWidgets import QApplication
        from gui.app import ChatKodiApp

        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        from gui.components.fonts import apply_app_font

        apply_app_font(app)
        try:
            from core.agent.settings import apply_to_environment, load_settings

            apply_to_environment(load_settings())  # context, device, vision before the model starts
        except Exception as exc:
            logger.warning("Assistant settings were not applied: %s", exc)
        window = ChatKodiApp()
        return app.exec_()
    except Exception as exc:
        logger.exception("Application startup failed")
        _report_startup_crash(exc, show_dialog=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
