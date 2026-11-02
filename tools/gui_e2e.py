"""End-to-end test through the real ChatKodi window, the way a tester uses it.

Starts the same window as ``main.py`` (config, runtime hints, Qt environment,
``ChatKodiApp`` with the loading screen and real models), types every question
into the input field with real key events, clicks «Отправить», waits until the
UI says the answer is done, grades the visible answer and saves screenshots.

    python -m tools.gui_e2e --docs runtime/synthetic/docs \
        --questions runtime/synthetic/scenarios.json --out runtime/gui_e2e --shots fail

All state lives in ``--out/runtime``; the user's settings and history are not
touched. ``--ui-tour`` also screenshots every page, theme and window size.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Drive the real ChatKodi window with scripted questions")
    ap.add_argument("--docs", required=True, help="documents folder the window should index")
    ap.add_argument("--questions", required=True, help="scenarios.json / cases.json or a JSON list")
    ap.add_argument("--out", required=True, help="output folder (results, screenshots, runtime)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-category", type=int, default=0, help="at most N cases of each category (compact window check)")
    ap.add_argument("--only", default="", help="comma-separated categories")
    ap.add_argument("--shots", default="fail", choices=["all", "fail", "none"])
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per answer")
    ap.add_argument("--ui-tour", action="store_true")
    ap.add_argument("--size", default="1440x900")
    ap.add_argument("--theme", default="")
    return ap.parse_args()


def load_cases(path: Path, only: str, limit: int, per_category: int = 0) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases", data) if isinstance(data, dict) else data
    if per_category:
        # The scenario bank repeats one template (the same question for other
        # dates): a window check needs a few of each kind, not 170 revenue days.
        counts: Dict[str, int] = {}
        selected = []
        for index, case in enumerate(cases):
            category = str(case.get("category"))
            if counts.get(category, 0) >= per_category:
                continue
            if case.get("depends_on_previous") and index > 0 and (not selected or selected[-1] is not cases[index - 1]):
                selected.append(cases[index - 1])
            selected.append(case)
            counts[category] = counts.get(category, 0) + 1
        cases = selected
    if only:
        wanted = {item.strip() for item in only.split(",") if item.strip()}
        # keep follow-up chains intact: a follow-up needs its base question
        selected = []
        for index, case in enumerate(cases):
            if case.get("category") in wanted:
                if case.get("depends_on_previous") and index > 0 and (not selected or selected[-1] is not cases[index - 1]):
                    selected.append(cases[index - 1])
                selected.append(case)
        cases = selected
    if limit:
        cases = cases[:limit]
    return cases


def grade(case: Dict[str, Any], answer: str) -> Tuple[Optional[bool], Dict[str, Any]]:
    """Grade an answer; ``None`` when the case has no expectations."""
    keys = ("expect_numbers", "expect_text", "expect_dates", "expect_text_any", "forbid_text")
    if not any(key in case for key in keys):
        return None, {}
    details: Dict[str, Any] = {}
    ok = True
    if any(key in case for key in ("expect_numbers", "expect_text", "expect_dates")):
        from tools.eval_synthetic import check_case

        ok, details = check_case(case, answer)
    lowered = answer.lower().replace("ё", "е")
    if case.get("expect_text_any"):
        hit = any(item.lower().replace("ё", "е") in lowered for item in case["expect_text_any"])
        details["expect_text_any_hit"] = hit
        ok = ok and hit
    if case.get("forbid_text"):
        found = [item for item in case["forbid_text"] if item.lower() in lowered]
        details["forbidden_text_found"] = found
        ok = ok and not found
    return ok, details


def main() -> int:
    args = parse_args()
    out = Path(args.out).resolve()
    shots = out / "shots"
    runtime = out / "runtime"
    shots.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    os.environ["CHATKODI_RUNTIME_DIR"] = str(runtime)
    sibling = ROOT.parent / "chatkodi_v1.6"
    if sibling.exists():
        os.environ.setdefault(
            "CHATKODI_EXTRA_MODEL_CACHE_DIRS",
            os.pathsep.join([str(sibling / ".chatkodi" / "models_cache"), str(sibling / "models_cache")]),
        )
        model = sibling / ".chatkodi" / "models_cache" / "llms" / "Qwen3-4B-Instruct-2507"
        if model.exists():
            os.environ.setdefault("CHATKODI_MODEL_PATH", str(model))
    settings: Dict[str, Any] = {"docs_folder": str(Path(args.docs).resolve())}
    if args.theme:
        settings["theme"] = args.theme
    (runtime / "chatkodi_settings.json").write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8")

    cases = load_cases(Path(args.questions), args.only, args.limit, args.per_category)
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import main as ck_main
    from core.config import get_config
    from core.system_profile import apply_runtime_hints, detect_system_profile

    cfg = get_config()
    apply_runtime_hints(detect_system_profile(cfg.data), cfg.data)
    ck_main._configure_qt_environment()
    ck_main._bootstrap_plugins()

    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    from gui.components.fonts import apply_app_font

    apply_app_font(app)
    from core.chat_mode import ChatMode
    from gui.app import ChatKodiApp

    started_at = time.perf_counter()
    events: List[Dict[str, Any]] = []

    def note(kind: str, **data: Any) -> None:
        item = {"t": round(time.perf_counter() - started_at, 2), "kind": kind, **data}
        events.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    def wait_until(predicate, timeout: float) -> bool:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            QTest.qWait(50)
            try:
                if predicate():
                    return True
            except Exception:
                pass
        return False

    def shot(widget, name: str) -> str:
        if widget is None:
            return ""
        path = shots / f"{name}.png"
        try:
            widget.grab().save(str(path))
        except Exception as exc:
            note("shot_failed", name=name, error=str(exc))
            return ""
        return str(path)

    def type_text(widget, text: str) -> None:
        # QTest.keyClicks aborts the process (Qt assert, exit 127) on
        # non-Latin characters: send key events that carry Unicode text.
        for char in text:
            QApplication.sendEvent(widget, QKeyEvent(QEvent.KeyPress, Qt.Key_unknown, Qt.NoModifier, char))
            QApplication.sendEvent(widget, QKeyEvent(QEvent.KeyRelease, Qt.Key_unknown, Qt.NoModifier, char))

    window = ChatKodiApp()
    note("window_created")
    ready = False
    loading_shots = 0
    next_shot = time.perf_counter() + 0.5
    deadline = time.perf_counter() + 900
    while time.perf_counter() < deadline:
        QTest.qWait(100)
        if window.isVisible() and hasattr(window, "input_field"):
            ready = True
            break
        loading = getattr(window, "loading_screen", None)
        if args.shots != "none" and loading is not None and loading.isVisible() and time.perf_counter() >= next_shot and loading_shots < 4:
            shot(loading, f"00_loading_{loading_shots}")
            loading_shots += 1
            next_shot = time.perf_counter() + 5
    startup_seconds = round(time.perf_counter() - started_at, 1)
    note("startup", ready=ready, seconds=startup_seconds)
    if not ready:
        (out / "summary.json").write_text(json.dumps({"ready": False, "startup_seconds": startup_seconds}, indent=2), encoding="utf-8")
        return 2

    width, height = (int(value) for value in args.size.lower().split("x"))
    window.resize(width, height)
    QTest.qWait(600)
    if args.shots != "none":
        shot(window, "01_ready")

    if args.ui_tour:
        for page in ("documents", "index", "diagnostics", "settings", "chat"):
            try:
                window._switch_workspace(page)
                QTest.qWait(500)
                shot(window, f"02_page_{page}")
            except Exception as exc:
                note("tour_error", page=page, error=repr(exc))
        for theme in ("dark", "light", "classic"):
            try:
                window._set_theme(theme)
                QTest.qWait(400)
                shot(window, f"03_theme_{theme}")
            except Exception as exc:
                note("tour_error", theme=theme, error=repr(exc))
        for w, h in ((1024, 680), (1280, 800), (1920, 1040)):
            window.resize(w, h)
            QTest.qWait(500)
            shot(window, f"04_size_{w}x{h}")
        window.resize(width, height)
        QTest.qWait(300)

    modes = {"rag": ChatMode.RAG, "chat": ChatMode.CHAT, "search": ChatMode.EXTRACT, "agent": ChatMode.AGENT}
    results: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        question = str(case.get("question") or "")
        mode_name = str(case.get("mode") or "").lower()
        if mode_name in modes:
            window._set_mode(modes[mode_name])
        window.last_answer_text = ""
        window.last_error_text = ""
        window.input_field.setFocus()
        window.input_field.clear()
        attach = [str(Path(args.questions).parent / name) for name in case.get("attach") or []]
        if attach:
            window._attach_files(attach)
            QTest.qWait(200)
        if len(question) < 400:
            type_text(window.input_field, question)
        else:
            window.input_field.setText(question)
        QTest.qWait(20)
        begin = time.perf_counter()
        QTest.mouseClick(window.send_btn, Qt.LeftButton)
        QTest.qWait(100)
        done = wait_until(
            lambda: window.send_btn.isEnabled() and window.send_btn.text() == "Отправить" and not getattr(window, "is_streaming", False),
            args.timeout,
        )
        seconds = round(time.perf_counter() - begin, 2)
        QTest.qWait(200)
        answer = str(getattr(window, "last_answer_text", "") or "")
        error = str(getattr(window, "last_error_text", "") or "")
        ok, details = grade(case, answer)
        if not done:
            ok, details["timeout"] = False, True
        if error:
            ok = False
        status = "ok" if ok else ("fail" if ok is False else "na")
        path = ""
        if args.shots == "all" or (args.shots == "fail" and ok is False):
            path = shot(window, f"q{index:03d}_{status}")
        row = {
            "index": index,
            "id": case.get("id"),
            "category": case.get("category"),
            "question": question,
            "done": done,
            "seconds": seconds,
            "ok": ok,
            "details": details,
            "answer": answer[:4000],
            "error": error[:2000],
            "shot": path,
        }
        results.append(row)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        note("answer", index=index, category=row["category"], ok=ok, seconds=seconds, head=re.sub(r"\s+", " ", answer[:120]))

    graded = [row for row in results if row["ok"] is not None]
    by_category: Dict[str, Dict[str, int]] = {}
    for row in graded:
        bucket = by_category.setdefault(str(row["category"]), {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(bool(row["ok"]))
    times = sorted(row["seconds"] for row in results)
    summary = {
        "ready": True,
        "startup_seconds": startup_seconds,
        "questions": len(results),
        "graded": len(graded),
        "passed": sum(1 for row in graded if row["ok"]),
        "pass_rate": round(sum(1 for row in graded if row["ok"]) / max(1, len(graded)), 3),
        "timeouts": sum(1 for row in results if not row["done"]),
        "errors": sum(1 for row in results if row["error"]),
        "median_seconds": times[len(times) // 2] if times else None,
        "p90_seconds": times[int(len(times) * 0.9)] if times else None,
        "max_seconds": times[-1] if times else None,
        "by_category": by_category,
        "failed_ids": [row["id"] for row in graded if not row["ok"]][:200],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    window.close()
    app.quit()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise SystemExit(3)
