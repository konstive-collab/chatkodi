"""Visual tour of the 2.0 window with screenshots: progress trail while the
assistant works, file cards, paper sheets (unfolded, edit mode), Ctrl+K,
drop overlay, the source beside the chat, settings, chat list, voice wave,
the three themes.

    python -m tools.gui_tour --docs <library> --inbox <folder with photos> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--inbox", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="1440x900")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    shots = out / "shots"
    runtime = out / "runtime"
    shots.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["CHATKODI_RUNTIME_DIR"] = str(runtime)
    (runtime / "chatkodi_settings.json").write_text(json.dumps({"docs_folder": str(Path(args.docs).resolve())}), encoding="utf-8")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    import main as ck_main
    from core.config import get_config
    from core.system_profile import apply_runtime_hints, detect_system_profile

    apply_runtime_hints(detect_system_profile(get_config().data), get_config().data)
    ck_main._configure_qt_environment()

    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    from gui.components.fonts import apply_app_font

    apply_app_font(app)
    from gui.app import ChatKodiApp

    def shot(widget, name):
        widget.grab().save(str(shots / f"{name}.png"))
        print("shot", name, flush=True)

    def wait(predicate, timeout=300):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            QTest.qWait(50)
            try:
                if predicate():
                    return True
            except Exception:
                pass
        return False

    window = ChatKodiApp()
    wait(lambda: window.isVisible() and hasattr(window, "input_field"), 600)
    w, h = (int(v) for v in args.size.split("x"))
    window.resize(w, h)
    QTest.qWait(800)
    shot(window, "01_start")
    window.resource_toggle.setChecked(True)
    QTest.qWait(2200)
    shot(window, "01b_resources_open")
    window.resource_toggle.setChecked(False)
    QTest.qWait(400)

    idle = lambda: window.send_btn.isEnabled() and window.send_btn.text() == "Отправить"
    inbox = Path(args.inbox)

    # 1. a photo without words: trail while working, then the result
    window._attach_files([str(inbox / "protokol_ispytaniy_betona.jpg")])
    QTest.qWait(300)
    shot(window, "02_attachment_chip")
    window._on_send()
    QTest.qWait(2500)
    shot(window, "03_trail_working")
    wait(idle)
    QTest.qWait(3500)  # reveal + suggestions
    shot(window, "04_photo_result")

    # 2. document by example: paper sheet, unfolded, edit mode
    window.input_field.setText("Оформи заключение о прочности по нашему образцу")
    window._on_send()
    wait(idle)
    QTest.qWait(3500)
    shot(window, "05_paper_card")
    from gui.components.paper import DocumentViewer, PaperCard

    cards = window.chat_container.findChildren(PaperCard)
    if cards:
        cards[-1].toggle_requested.emit()
        QTest.qWait(700)
        window._scroll_to_bottom(animated=False)
        QTest.qWait(400)
        shot(window, "06_paper_unfolded")
        viewers = window.chat_container.findChildren(DocumentViewer)
        if viewers:
            viewers[-1]._toggle_edit()
            QTest.qWait(500)
            window._scroll_to_bottom(animated=False)
            QTest.qWait(300)
            shot(window, "07_paper_edit")
            viewers[-1]._toggle_edit()

    # 3. library question, then its source beside the chat
    window.input_field.setText("Какая средняя прочность требуется для бетона класса B30?")
    window._on_send()
    wait(idle)
    QTest.qWait(3000)
    shot(window, "08_library_answer")
    docs = [d for d in window.doc_processor.documents if "Положение о контроле" in str(d.name)]
    if docs:
        window._show_source_side(Path(docs[0].path), str(docs[0].name), "средняя прочность серии должна быть не менее 39,2 МПа", window.last_answer_text)
        QTest.qWait(700)
        shot(window, "09_source_beside_chat")
        window.source_pane.hide()

    # 4. Ctrl+K, drop overlay, voice wave
    window._open_palette()
    QTest.qWait(400)
    shot(window._palette, "10_palette_empty")
    window._palette.search.setText("тема")
    QTest.qWait(300)
    shot(window._palette, "11_palette_query")
    window._palette.hide()
    window._show_drop_overlay([inbox / "schet_318.jpg", inbox / "analiz_krovi.jpg"])
    QTest.qWait(400)
    shot(window, "12_drop_overlay")
    window._hide_drop_overlay()
    window.voice_wave.start()
    window.input_field.setText("Проверь протокол по нашим нормативам")
    for level in (0.3, 0.8, 0.5, 0.9, 0.6):
        window.voice_wave.set_level(level)
        QTest.qWait(120)
    shot(window, "13_voice_wave")
    window.voice_wave.stop()
    window.input_field.clear()

    # 5. second chat, list, settings, themes
    window._new_chat()
    QTest.qWait(400)
    window.input_field.setText("Сколько стоит датчик давления ДД-10?")
    window._on_send()
    wait(idle)
    QTest.qWait(2500)
    shot(window, "14_second_chat")

    # 6. files on the computer: found cards, a rename waiting for «Выполнить», then «Отменить»
    from PyQt5.QtWidgets import QPushButton

    def ask(text, pause=2500):
        window.input_field.setText(text)
        window._on_send()
        wait(idle)
        QTest.qWait(pause)

    ask("Найди положение о командировках")
    shot(window, "14b_found_files")
    ask("Переименуй служебную записку об остановке линии в Записка_остановка_линии")
    shot(window, "14c_rename_waiting")
    runs = [b for b in window.chat_container.findChildren(QPushButton, "commandRunButton") if b.isEnabled()]
    if runs:
        runs[-1].click()
        wait(lambda: runs[-1].text() == "Выполнено", 60)
        QTest.qWait(2500)
        shot(window, "14d_renamed_with_undo")
        undo = [b for b in window.chat_container.findChildren(QPushButton) if b.text() == "↶ Отменить" and b.isEnabled()]
        if undo:
            undo[-1].click()
            QTest.qWait(1500)
            window._scroll_to_bottom(animated=False)
            QTest.qWait(300)
            shot(window, "14e_undone")

    window._switch_workspace("settings")
    QTest.qWait(500)
    shot(window, "15_settings")
    window._switch_workspace("chat")
    for theme in ("dark", "light", "classic"):
        window._set_theme(theme)
        QTest.qWait(500)
        shot(window, f"16_theme_{theme}")
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
