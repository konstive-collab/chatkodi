#!/usr/bin/env python3
"""Надёжная точка входа графического лаунчера ChatKodi."""

from __future__ import annotations

import os
import json
import sys
import time
import traceback
from pathlib import Path


def _launcher_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _crash_log_path() -> Path:
    root = _launcher_root()
    if (root / "portable.flag").is_file():
        runtime = root / "runtime"
    elif sys.platform == "win32":
        runtime = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or Path.home()) / "ChatKodi"
    else:
        runtime = Path.home() / ".chatkodi"
    path = runtime / "logs" / "launcher_crash.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _report_launcher_crash(exc: BaseException) -> None:
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        path = _crash_log_path()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] launcher startup failed\n{details}\n")
    except Exception:
        path = None

    if sys.platform == "win32":
        try:
            import ctypes

            suffix = f"\n\nДиагностический лог: {path}" if path else ""
            ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
                None,
                f"ChatKodi Launcher не смог запуститься.\n\n{exc}{suffix}",
                "Ошибка запуска ChatKodi",
                0x10,
            )
        except Exception:
            pass


def _verify_portable_payload() -> None:
    root = _launcher_root()
    if not getattr(sys, "frozen", False) or not (root / "portable.flag").is_file():
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
    write_preflight_log(root, preflight)
    if preflight.get("llm_degraded"):
        os.environ["CHATKODI_LLM_PAYLOAD_DAMAGED"] = "1"
    if not preflight.get("ok"):
        raise RuntimeError(format_blocking_message(preflight))
    os.environ["CHATKODI_PORTABLE_VERIFIED"] = str(preflight.get("build_id") or "1")


def main() -> int:
    try:
        _verify_portable_payload()
        if "--self-test-json" in sys.argv:
            try:
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
            except Exception:
                pass
            from PyQt5.QtCore import QT_VERSION_STR
            from tools.doctor import build_doctor_report
            from tools.launcher.gui import run_launcher  # noqa: F401

            doctor = build_doctor_report()
            payload = {
                "ok": doctor.get("doctor", {}).get("status") == "ok",
                "launcher_root": str(_launcher_root()),
                "frozen": bool(getattr(sys, "frozen", False)),
                "qt_version": QT_VERSION_STR,
                "doctor": doctor,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if payload["ok"] else 1
        from tools.launcher.gui import run_launcher

        return int(run_launcher())
    except BaseException as exc:
        _report_launcher_crash(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
