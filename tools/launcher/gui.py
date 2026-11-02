#!/usr/bin/env python3
"""Дружелюбный лаунчер ChatKodi 1.8.

Главный принцип лаунчера: обычный пользователь нажимает 1–2 кнопки и
получает рабочий запуск. Ручные настройки остаются, но живут во вкладках
«Модели» и «Ручная настройка», не мешая основному сценарию. Начиная
с rc7 лаунчер разделяет минимальную установку самого лаунчера и полную
подготовку ChatKodi: стартовый скрипт ставит только GUI-оболочку, а всё
тяжёлое доустанавливается уже здесь по кнопке.
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
    os.environ["CHATKODI_PRODUCTION"] = "1"
    os.environ["CHATKODI_APP_ROOT"] = str(ROOT)
    if (ROOT / "portable.flag").exists():
        os.environ["CHATKODI_PORTABLE"] = "1"
        os.environ["CHATKODI_RUNTIME_DIR"] = str(ROOT / "runtime")
else:
    ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.atomic_io import atomic_write_json, read_json_with_backup
from core.paths import is_production_runtime, model_cache_search_dirs, runtime_root
from core.theme_manager import THEME_LABELS, normalize_theme, theme_label, theme_palette
from tools.doctor import build_doctor_report


class LauncherCliFallback:
    """Текстовый fallback, когда PyQt5 ещё не установлен."""

    def run(self) -> int:
        print("ChatKodi Launcher fallback")
        print("PyQt5 не найден, поэтому графический лаунчер не запустился.")
        print("Для полной подготовки выполните:")
        print(r"  powershell -ExecutionPolicy Bypass -File scripts\windows\install_all.ps1")
        print("Для запуска после установки:")
        print(r"  START_LAUNCHER.bat")
        try:
            report = build_doctor_report()
            print(json.dumps(report.get("doctor", {}), ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"Doctor недоступен: {exc}")
        return 1


def _python_executable() -> str:
    venv_python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _is_production() -> bool:
    return is_production_runtime()


def _chatkodi_app_executable() -> Path | None:
    if not _is_production():
        return None
    exe_name = "ChatKodiApp.exe" if os.name == "nt" else "ChatKodiApp"
    candidate = ROOT / exe_name
    return candidate if candidate.exists() else None


def _doctor_command() -> list[str]:
    app_exe = _chatkodi_app_executable()
    if app_exe:
        return [str(app_exe), "--doctor-json"]
    if _is_production():
        exe_name = "ChatKodiApp.exe" if os.name == "nt" else "ChatKodiApp"
        return [str(ROOT / exe_name), "--doctor-json"]
    return [_python_executable(), "-m", "tools.doctor", "--json"]


def _run_process(command: list[str], *, cwd: Path = ROOT, env: dict | None = None) -> tuple[int, str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
    )
    return proc.returncode, proc.stdout or ""


def _run_sequence(commands: Iterable[list[str]], *, env: dict | None = None) -> tuple[int, str]:
    output: list[str] = []
    last_code = 0
    for command in commands:
        output.append("> " + " ".join(command))
        code, text = _run_process(command, env=env)
        last_code = code
        if text:
            output.append(text.rstrip())
        output.append(f"[exit code] {code}")
        if code != 0:
            break
    return last_code, "\n".join(output) + "\n"




def _python_can_import(module: str, *, py: str | None = None) -> bool:
    """Быстрая проверка, установлен ли модуль в текущем/venv Python."""
    try:
        code, _ = _run_process([py or _python_executable(), "-c", f"import {module}"])
        return code == 0
    except Exception:
        return False


def _has_nvidia_smi() -> bool:
    try:
        proc = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4)
        return proc.returncode == 0
    except Exception:
        return False


def _dir_size_gb(path: Path) -> float:
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        return 0.0
    return total / (1024 ** 3)


def _discover_installed_llms() -> list[dict[str, str]]:
    """GGUF models in models_cache/llms of every known model cache."""
    from core.llm.model_select import describe_model, list_gguf

    found: list[dict[str, str]] = []
    for path in list_gguf("llms"):
        info = describe_model(path)
        found.append({"name": str(info["name"]), "path": str(path), "size": f"{info['total_size_gb']:.1f} GB"})
    found.sort(key=lambda item: item["name"].lower())
    return found


def _settings_path() -> Path:
    return runtime_root() / "launcher_settings.json"


def _load_launcher_settings() -> dict:
    try:
        path = _settings_path()
        payload = read_json_with_backup(path, default={})
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _save_launcher_settings(payload: dict) -> None:
    path = _settings_path()
    atomic_write_json(path, payload)


def _manual_file_text(name: str) -> str:
    path = ROOT / "manuals" / name
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"# {name}\n\nНе удалось открыть manual: {exc}"


def _mini_markdown_to_html(markdown: str) -> str:
    """Небольшой безопасный markdown-renderer для лаунчера.

    Не тянем внешние зависимости ради просмотра manuals: поддерживаем заголовки,
    списки, кодовые блоки и обычные абзацы. Для пользовательской навигации этого
    достаточно, а лаунчер не становится хрупким до установки библиотек.
    """
    lines = (markdown or "").splitlines()
    out: list[str] = []
    in_code = False
    in_list = False
    code_buf: list[str] = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_code:
                close_list()
                in_code = True
                code_buf = []
            else:
                out.append("<pre><code>" + html.escape("\n".join(code_buf)) + "</code></pre>")
                in_code = False
            continue
        if in_code:
            code_buf.append(line)
            continue
        if not stripped:
            close_list()
            continue
        if stripped.startswith("#"):
            close_list()
            level = min(len(stripped) - len(stripped.lstrip("#")), 3)
            text = stripped[level:].strip()
            out.append(f"<h{level}>{html.escape(text)}</h{level}>")
            continue
        if stripped.startswith(('- ', '* ', '• ')):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + html.escape(stripped[2:].strip()) + "</li>")
            continue
        close_list()
        safe = html.escape(stripped)
        safe = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)
        out.append("<p>" + safe + "</p>")
    close_list()
    if in_code:
        out.append("<pre><code>" + html.escape("\n".join(code_buf)) + "</code></pre>")
    return "\n".join(out)


def _launcher_palette(theme: str = "classic") -> dict[str, str]:
    """Палитра лаунчера из единого theme_manager.

    Лаунчеру нужны два дополнительных цвета для вкладок, но базовые фон/текст/
    акцент он берёт из того же источника, что экран загрузки и основное окно.
    """
    p = theme_palette(theme)
    p.setdefault("tab", p["panel"])
    p.setdefault("tab_selected", p["panel_2"])
    return p

def _manual_page_html(title: str, files: list[str], intro: str = "", theme: str = "classic") -> str:
    chunks = []
    if intro:
        chunks.append(f"<div class='intro'>{html.escape(intro)}</div>")
    for idx, filename in enumerate(files):
        text = _manual_file_text(filename)
        if idx:
            chunks.append("<hr>")
        chunks.append(_mini_markdown_to_html(text))
    body = "\n".join(chunks)
    palette = _launcher_palette(theme)
    return f"""
    <html><head><style>
        body {{ background: {palette['sunken']}; color: {palette['fg']}; font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.58; padding: 22px 26px; }}
        h1 {{ color: {palette['title']}; font-size: 25px; margin: 0 0 14px; }}
        h2 {{ color: {palette['accent_soft']}; font-size: 19px; margin-top: 22px; }}
        h3 {{ color: {palette['accent_soft']}; font-size: 16px; margin-top: 18px; }}
        p, li {{ color: {palette['fg']}; font-size: 14px; }}
        .intro {{ background: {palette['accent_bg']}; border: 1px solid {palette['accent_border']}; border-radius: 14px; padding: 12px 14px; color: {palette['accent_soft']}; margin-bottom: 18px; }}
        code {{ background: {palette['code_bg']}; color: {palette['accent_soft']}; border-radius: 6px; padding: 1px 5px; }}
        pre {{ background: {palette['bg']}; border: 1px solid {palette['border']}; border-radius: 14px; padding: 12px; overflow: auto; }}
        pre code {{ background: transparent; padding: 0; color: {palette['accent_soft']}; }}
        hr {{ border: none; border-top: 1px solid {palette['border']}; margin: 26px 0; }}
        ul {{ margin-top: 6px; }}
    </style></head><body><h1>{html.escape(title)}</h1>{body}</body></html>
    """


def run_launcher() -> int:
    try:
        from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
        from PyQt5.QtGui import QFont
        from PyQt5.QtWidgets import (
            QApplication,
            QComboBox,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QPlainTextEdit,
            QTabWidget,
            QCheckBox,
            QTextBrowser,
            QVBoxLayout,
            QWidget,
        )
    except Exception:
        return LauncherCliFallback().run()

    class CommandWorker(QThread):
        finished_with_output = pyqtSignal(int, str)

        def __init__(self, fn: Callable[[], tuple[int, str]], parent=None):
            super().__init__(parent)
            self.fn = fn

        def run(self) -> None:
            try:
                code, output = self.fn()
            except Exception as exc:
                code, output = 1, str(exc)
            self.finished_with_output.emit(code, output)

    class LauncherWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("ChatKodi Launcher 2.0.0")
            self.setMinimumSize(900, 620)
            screen = QApplication.primaryScreen()
            if screen is not None:
                available = screen.availableGeometry()
                self.resize(
                    min(1240, max(900, int(available.width() * 0.88))),
                    min(820, max(620, int(available.height() * 0.88))),
                )
            else:
                self.resize(1120, 760)
            self.worker: CommandWorker | None = None
            self._app_process: subprocess.Popen | None = None
            self.last_report: dict = {}
            self.settings = _load_launcher_settings()
            self.theme = normalize_theme(self.settings.get("theme") or os.environ.get("CHATKODI_THEME"), default="classic")
            self._build_ui()
            self._apply_style()
            self.refresh_doctor(silent=True)

        def _build_ui(self) -> None:
            central = QWidget(self)
            central.setObjectName("launcherRoot")
            root = QVBoxLayout(central)
            root.setContentsMargins(22, 22, 22, 22)
            root.setSpacing(16)
            self.setCentralWidget(central)

            hero = QFrame(self)
            hero.setObjectName("heroCard")
            hero_layout = QHBoxLayout(hero)
            hero_layout.setContentsMargins(20, 18, 20, 18)
            hero_layout.setSpacing(18)

            title_box = QVBoxLayout()
            title_box.setSpacing(6)
            title = QLabel("ChatKodi Launcher")
            title.setObjectName("title")
            title.setFont(QFont("Segoe UI", 24, QFont.Bold))
            subtitle = QLabel("Запуск без шаманства: сначала ставится лёгкий лаунчер, а потом он сам готовит окружение, выбирает CPU/GPU, модели и профиль. Ручные настройки есть, но обычный путь — Авто.")
            subtitle.setObjectName("subtitle")
            subtitle.setWordWrap(True)
            title_box.addWidget(title)
            title_box.addWidget(subtitle)

            self.ready_badge = QLabel("Проверяю систему…")
            self.ready_badge.setObjectName("readyBadge")
            self.ready_badge.setAlignment(Qt.AlignCenter)
            self.ready_badge.setMinimumWidth(230)

            hero_layout.addLayout(title_box, 1)
            hero_layout.addWidget(self.ready_badge, 0)
            root.addWidget(hero)

            self.tabs = QTabWidget(self)
            self.tabs.setObjectName("launcherTabs")
            root.addWidget(self.tabs, 1)

            self._build_start_tab()
            self._build_models_tab()
            self._build_install_tab()
            self._build_settings_tab()
            self._build_manuals_tab()
            self._build_logs_tab()

        def _card(self, title: str, text: str = "") -> QFrame:
            frame = QFrame(self)
            frame.setObjectName("card")
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(15, 14, 15, 14)
            layout.setSpacing(7)
            label = QLabel(title)
            label.setObjectName("cardTitle")
            body = QLabel(text)
            body.setObjectName("cardText")
            body.setWordWrap(True)
            layout.addWidget(label)
            layout.addWidget(body)
            frame.body_label = body  # type: ignore[attr-defined]
            return frame

        def _build_start_tab(self) -> None:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(14)

            main_card = QFrame(page)
            main_card.setObjectName("primaryCard")
            main_layout = QVBoxLayout(main_card)
            main_layout.setContentsMargins(18, 18, 18, 18)
            main_layout.setSpacing(12)
            lead = QLabel("Быстрый старт")
            lead.setObjectName("largeCardTitle")
            desc = QLabel("Обычный сценарий: нажмите «Подготовить и запустить». Лаунчер сам доустановит зависимости, проверит модели и перейдёт в ChatKodi. Отдельные кнопки ниже нужны только если хочется контролировать процесс.")
            desc.setObjectName("cardText")
            desc.setWordWrap(True)
            main_layout.addWidget(lead)
            main_layout.addWidget(desc)
            actions = QHBoxLayout()
            actions.setSpacing(10)
            self.prepare_and_start_btn = QPushButton("Подготовить и запустить")
            self.prepare_and_start_btn.setObjectName("primaryButton")
            self.prepare_and_start_btn.clicked.connect(self.auto_prepare_and_start)
            self.auto_setup_btn = QPushButton("Только подготовить")
            self.auto_setup_btn.clicked.connect(self.auto_prepare)
            self.start_btn = QPushButton("Запустить ChatKodi")
            self.start_btn.clicked.connect(self.start_chatkodi)
            refresh_btn = QPushButton("Проверить ещё раз")
            refresh_btn.clicked.connect(self.refresh_doctor)
            actions.addWidget(self.prepare_and_start_btn)
            actions.addWidget(self.auto_setup_btn)
            actions.addWidget(self.start_btn)
            actions.addWidget(refresh_btn)
            actions.addStretch(1)
            main_layout.addLayout(actions)
            layout.addWidget(main_card)

            grid = QGridLayout()
            grid.setSpacing(12)
            self.python_card = self._card("Python", "—")
            self.models_card = self._card("Модели", "—")
            self.runtime_card = self._card("Профиль", "—")
            self.docs_card = self._card("Документы", "—")
            grid.addWidget(self.python_card, 0, 0)
            grid.addWidget(self.models_card, 0, 1)
            grid.addWidget(self.runtime_card, 1, 0)
            grid.addWidget(self.docs_card, 1, 1)
            layout.addLayout(grid)

            utility_row = QHBoxLayout()
            for text, handler in [
                ("Мануалы", self.open_manuals),
                ("Support bundle", self.create_support_bundle),
            ]:
                btn = QPushButton(text)
                btn.clicked.connect(handler)
                utility_row.addWidget(btn)
            utility_row.addStretch(1)
            layout.addLayout(utility_row)
            layout.addStretch(1)
            self.tabs.addTab(page, "Старт")

        def _build_models_tab(self) -> None:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(12)
            hint = QLabel("По умолчанию используется «Авто»: ChatKodi берёт скачанную модель безопасно для устройства. Если ПК слабый, но LLM всё равно нужна, включите ручное разрешение ниже — программа не будет блокировать загрузку, только предупредит, что это может быть медленно.")
            hint.setObjectName("cardText")
            hint.setWordWrap(True)
            layout.addWidget(hint)

            self.model_choice = QComboBox()
            self.model_choice.setObjectName("wideCombo")
            self.refresh_model_list()
            layout.addWidget(QLabel("Модель LLM:"))
            layout.addWidget(self.model_choice)

            self.force_llm_checkbox = QCheckBox("Разрешить LLM даже на слабом ПК — медленно, но запускать")
            self.force_llm_checkbox.setObjectName("forceLlmCheck")
            self.force_llm_checkbox.setToolTip("Используйте, если нужна генерация/«Спросить по данным», даже когда Auto считает модель тяжёлой для CPU или 8 GB RAM.")
            self.force_llm_checkbox.setChecked(bool(self.settings.get("force_llm", False)))
            self.force_llm_checkbox.stateChanged.connect(lambda *_: self.save_launch_profile(show_dialog=False))
            layout.addWidget(self.force_llm_checkbox)

            row = QHBoxLayout()
            download_btn = QPushButton("Скачать / проверить модели автоматически")
            if _is_production():
                download_btn.setText("Check production payload")
                download_btn.clicked.connect(lambda: self.run_command(_doctor_command()))
            else:
                download_btn.clicked.connect(lambda: self.run_command([_python_executable(), "download_models.py"]))
            refresh_btn = QPushButton("Обновить список моделей")
            refresh_btn.clicked.connect(self.refresh_model_list)
            row.addWidget(download_btn)
            row.addWidget(refresh_btn)
            row.addStretch(1)
            layout.addLayout(row)

            self.model_text = QPlainTextEdit()
            self.model_text.setReadOnly(True)
            self.model_text.setPlainText("После проверки здесь появится состояние моделей. Оставьте «Авто», если не хотите вручную выбирать LLM.")
            layout.addWidget(self.model_text, 1)
            self.tabs.addTab(page, "Модели")

        def _build_install_tab(self) -> None:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(12)
            hint = QLabel("Эта вкладка для ручного ремонта. В обычной ситуации достаточно кнопки «Подготовить автоматически» на вкладке «Старт».")
            hint.setObjectName("cardText")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            if _is_production():
                prod_hint = QLabel("Production build: dependencies and models are installed with setup. This tab only runs diagnostics.")
                prod_hint.setObjectName("cardText")
                prod_hint.setWordWrap(True)
                layout.addWidget(prod_hint)
                btn = QPushButton("Run production diagnostics")
                btn.clicked.connect(lambda: self.run_command(_doctor_command()))
                layout.addWidget(btn)
                layout.addStretch(1)
                self.tabs.addTab(page, "Diagnostics")
                return

            from core.document_dependencies import READERS
            for text, cmd in [
                ("Восстановить компоненты документов", [_python_executable(), "-m", "pip", "install", *[package for _, package in READERS.values()]]),
                ("Создать .venv и поставить базовые зависимости", [sys.executable, "-m", "tools.bootstrap", "--create-venv", "--install-requirements"]),
                ("Скачать рекомендуемые модели", [_python_executable(), "download_models.py"]),
                ("Проверить систему", [_python_executable(), "-m", "tools.doctor"]),
            ]:
                btn = QPushButton(text)
                btn.clicked.connect(lambda _=False, c=cmd: self.run_command(c))
                layout.addWidget(btn)
            layout.addStretch(1)
            self.tabs.addTab(page, "Ручная установка")

        def _build_settings_tab(self) -> None:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(12)
            hint = QLabel("Авто — основной режим. Он не навязывает тяжёлую модель слабому ПК и не душит GPU-машину. Ручные профили нужны для тестов и тонкой настройки.")
            hint.setObjectName("cardText")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            self.launch_profile = QComboBox()
            self.launch_profile.setObjectName("wideCombo")
            profiles = [
                ("Авто — рекомендовано", "auto"),
                ("Экономный CPU / слабый ноутбук", "academic_safe"),
                ("Сбалансированный CPU", "balanced_cpu"),
                ("GPU быстрый", "gpu_fast"),
                ("Мощный ПК / широкий контекст", "desktop_full"),
                ("Только поиск, без LLM", "search_only"),
            ]
            for label, value in profiles:
                self.launch_profile.addItem(label, value)
            saved_profile = self.settings.get("launch_profile", "auto")
            self._set_combo_data(self.launch_profile, saved_profile)
            self.launch_profile.currentIndexChanged.connect(lambda *_: self.save_launch_profile(show_dialog=False))
            layout.addWidget(QLabel("Профиль запуска:"))
            layout.addWidget(self.launch_profile)

            self.theme_choice = QComboBox()
            self.theme_choice.setObjectName("wideCombo")
            for value, label in [("classic", "Классическая — как старый экран загрузки"), ("dark", "Тёмная — спокойная зелёная"), ("light", "Светлая — дневная")]:
                self.theme_choice.addItem(label, value)
            self._set_combo_data(self.theme_choice, self.theme)
            self.theme_choice.currentIndexChanged.connect(self._on_theme_changed)
            layout.addWidget(QLabel("Тема интерфейса:"))
            layout.addWidget(self.theme_choice)

            row = QHBoxLayout()
            save_btn = QPushButton("Сохранить настройки лаунчера")
            save_btn.clicked.connect(self.save_launch_profile)
            auto_btn = QPushButton("Вернуть всё в Авто")
            auto_btn.clicked.connect(self.reset_to_auto)
            row.addWidget(save_btn)
            row.addWidget(auto_btn)
            row.addStretch(1)
            layout.addLayout(row)
            layout.addStretch(1)
            self.tabs.addTab(page, "Ручная настройка")

        def _build_manuals_tab(self) -> None:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(12)

            header = QFrame(page)
            header.setObjectName("manualsHeader")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(14, 12, 14, 12)
            header_layout.setSpacing(10)
            title_box = QVBoxLayout()
            title_box.setSpacing(3)
            title = QLabel("Мануалы внутри лаунчера")
            title.setObjectName("cardTitle")
            text = QLabel("Быстрые инструкции, логика работы и ручная диагностика открываются здесь — без поиска файлов в папках.")
            text.setObjectName("cardText")
            text.setWordWrap(True)
            title_box.addWidget(title)
            title_box.addWidget(text)
            header_layout.addLayout(title_box, 1)
            folder_btn = QPushButton("Открыть папку manuals")
            folder_btn.clicked.connect(self.open_manuals_folder)
            header_layout.addWidget(folder_btn, 0)
            layout.addWidget(header)

            manual_tabs = QTabWidget(page)
            manual_tabs.setObjectName("manualTabs")
            layout.addWidget(manual_tabs, 1)

            pages = [
                ("Быстрый старт", ["README_RU.md", "QUICKSTART_RU.md"], "С чего начать: запуск, документы, первый вопрос."),
                ("Настройка", ["INSTALL_LAUNCHER_RU.md", "INSTALL_WINDOWS_RU.md", "LAUNCHER_GUIDE_RU.md", "LAUNCHER_AUTOMATION_RU.md", "SEARCH_SETTINGS_RU.md", "THEMES_RU.md"], "Установка, авто-подготовка, темы, профили и ручные настройки."),
                ("Модели", ["MODELS.md", "LLM_AUTO_MODE_RU.md", "LLM_FORCE_MODE_RU.md", "LLM_TROUBLESHOOTING_RU.md", "PERFORMANCE_AND_LLM_RU.md"], "Как ChatKodi выбирает модели, как вернуть Авто и что делать, если LLM не поднялась."),
                ("Логика работы", ["PROJECT_LOGIC_RU.md", "UNIVERSAL_FACT_MODEL_RU.md", "STATISTICS_LOGIC_RU.md", "ACTIVITY_REPORT_RU.md"], "Как работает поиск, факты, статистика и подробные отчёты."),
                ("Источники", ["SOURCE_BUTTONS_RU.md", "CONTEXT_ASK_RU.md", "EVIDENCE_COORDINATES_RU.md"], "Кнопки источников, уточнения по данным и координаты доказательств."),
                ("Отладка", ["DEBUGGING_RU.md", "PERFORMANCE_AND_LLM_RU.md", "LLM_FORCE_MODE_RU.md", "LLM_TROUBLESHOOTING_RU.md", "FORMAT_SUPPORT_RU.md", "KNOWN_ISSUES_1_6_6_RC1.md", "AUTOMATION_ROADMAP_1_9_2_0_RU.md"], "Что делать, если что-то не запускается, не индексируется или работает странно."),
            ]
            for tab_title, files, intro in pages:
                browser = QTextBrowser(manual_tabs)
                browser.setObjectName("manualBrowser")
                browser.setOpenExternalLinks(True)
                browser.setProperty("manualSpec", (tab_title, files, intro))
                browser.setHtml(_manual_page_html(tab_title, files, intro, self.theme))
                manual_tabs.addTab(browser, tab_title)

            self.manuals_page = page
            self.tabs.addTab(page, "Мануалы")

        def _build_logs_tab(self) -> None:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(14, 14, 14, 14)
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            layout.addWidget(self.log, 1)
            self.tabs.addTab(page, "Лог")

        def _apply_style(self) -> None:
            p = _launcher_palette(self.theme)
            self.setStyleSheet(f"""
                QMainWindow, QWidget#launcherRoot {{ background: {p['bg']}; color: {p['fg']}; font-family: Segoe UI, Arial; }}
                QTabWidget::pane {{ border: 1px solid {p['border']}; border-radius: 16px; background: {p['sunken']}; }}
                QTabBar::tab {{ background: {p['tab']}; color: {p['muted']}; padding: 10px 16px; border-top-left-radius: 10px; border-top-right-radius: 10px; }}
                QTabBar::tab:selected {{ color: {p['title']}; background: {p['tab_selected']}; }}
                QLabel {{ color: {p['fg']}; }}
                QLabel#title {{ color: {p['title']}; }}
                QLabel#subtitle, QLabel#cardText {{ color: {p['muted']}; }}
                QLabel#readyBadge {{ background: {p['accent_bg']}; color: {p['accent_soft']}; border: 1px solid {p['accent_border']}; border-radius: 16px; padding: 12px; font-weight: 800; }}
                QFrame#heroCard, QFrame#card, QFrame#primaryCard {{ background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 18px; }}
                QFrame#primaryCard {{ border-color: {p['accent_border']}; }}
                QLabel#cardTitle {{ color: {p['accent_soft']}; font-weight: 750; }}
                QLabel#largeCardTitle {{ color: {p['title']}; font-size: 19px; font-weight: 850; }}
                QPushButton {{ background: {p['button']}; color: {p['fg']}; border: 1px solid {p['border']}; border-radius: 12px; padding: 9px 13px; }}
                QPushButton:hover {{ background: {p['button_hover']}; border-color: {p['accent_border']}; }}
                QPushButton:pressed {{ background: {p['sunken']}; }}
                QPushButton#primaryButton {{ background: {p['accent']}; color: #042414; border-color: {p['accent']}; font-weight: 850; padding: 10px 16px; }}
                QPushButton#primaryButton:hover {{ background: {p['accent_soft']}; }}
                QPlainTextEdit, QComboBox {{ background: {p['sunken']}; color: {p['fg']}; border: 1px solid {p['border']}; border-radius: 13px; padding: 9px 11px; selection-background-color: {p['accent_bg']}; }}
                QCheckBox#forceLlmCheck {{ color: {p['fg']}; spacing: 8px; padding: 6px 2px; }}
                QCheckBox#forceLlmCheck::indicator {{ width: 18px; height: 18px; border-radius: 5px; border: 1px solid {p['accent_border']}; background: {p['sunken']}; }}
                QCheckBox#forceLlmCheck::indicator:checked {{ background: {p['accent']}; border-color: {p['accent']}; }}
                QComboBox#wideCombo {{ min-height: 24px; }}
                QComboBox QAbstractItemView {{ background: {p['panel']}; color: {p['fg']}; selection-background-color: {p['accent_bg']}; selection-color: {p['title']}; border: 1px solid {p['border']}; outline: none; padding: 6px; }}
                QTextBrowser#manualBrowser {{ background: {p['sunken']}; color: {p['fg']}; border: 1px solid {p['border']}; border-radius: 14px; padding: 0; }}
                QFrame#manualsHeader {{ background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 16px; }}
                QTabWidget#manualTabs::pane {{ border: 1px solid {p['border']}; border-radius: 14px; background: {p['sunken']}; }}
                QScrollBar:vertical {{ background: transparent; width: 12px; margin: 4px; }}
                QScrollBar::handle:vertical {{ background: {p['border']}; border-radius: 6px; min-height: 22px; }}
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            """)

        def _refresh_manual_theme(self) -> None:
            try:
                tabs = self.manuals_page.findChild(QTabWidget, "manualTabs") if hasattr(self, "manuals_page") else None
                if not tabs:
                    return
                for i in range(tabs.count()):
                    browser = tabs.widget(i)
                    spec = browser.property("manualSpec") if browser else None
                    if spec:
                        tab_title, files, intro = spec
                        browser.setHtml(_manual_page_html(tab_title, files, intro, self.theme))
            except Exception:
                pass

        def _on_theme_changed(self) -> None:
            if hasattr(self, "theme_choice"):
                self.theme = normalize_theme(self.theme_choice.currentData())
            self._apply_style()
            self._refresh_manual_theme()
            self.save_launch_profile(show_dialog=False)

        def reset_to_auto(self) -> None:
            if hasattr(self, "launch_profile"):
                self._set_combo_data(self.launch_profile, "auto")
            if hasattr(self, "model_choice"):
                self._set_combo_data(self.model_choice, "auto")
                if hasattr(self, "force_llm_checkbox"):
                    self.force_llm_checkbox.setChecked(False)
            self.save_launch_profile(show_dialog=False)
            QMessageBox.information(self, "Авто включён", "Профиль и модель вернулись в режим Авто. Теперь лаунчер сам подберёт запуск под устройство.")

        def _set_combo_data(self, combo: QComboBox, value: str) -> None:
            for idx in range(combo.count()):
                if str(combo.itemData(idx)) == str(value):
                    combo.setCurrentIndex(idx)
                    return

        def append_log(self, text: str, *, jump: bool = True) -> None:
            self.log.appendPlainText(text.rstrip() + "\n")
            if jump:
                self.tabs.setCurrentWidget(self.log.parentWidget())

        def run_command(self, command: list[str], *, on_success: Callable[[], None] | None = None) -> None:
            if self.worker and self.worker.isRunning():
                QMessageBox.information(self, "Команда выполняется", "Дождитесь завершения текущей операции.")
                return
            self.append_log("> " + " ".join(command))
            self.worker = CommandWorker(lambda: _run_process(command), self)
            self.worker.finished_with_output.connect(lambda code, output: self._on_command_finished(code, output, on_success))
            self.worker.start()

        def run_command_sequence(self, commands: list[list[str]], *, title: str, on_success: Callable[[], None] | None = None) -> None:
            if self.worker and self.worker.isRunning():
                QMessageBox.information(self, "Команда выполняется", "Дождитесь завершения текущей операции.")
                return
            self.append_log(f"=== {title} ===")
            self.worker = CommandWorker(lambda: _run_sequence(commands), self)
            self.worker.finished_with_output.connect(lambda code, output: self._on_command_finished(code, output, on_success))
            self.worker.start()

        def _on_command_finished(self, code: int, output: str, on_success: Callable[[], None] | None = None) -> None:
            self.append_log(output or "(нет вывода)")
            self.append_log(f"[итоговый код] {code}")
            self.refresh_doctor(silent=True)
            if code == 0 and on_success:
                on_success()

        def _auto_setup_commands(self) -> list[list[str]]:
            if _is_production():
                return [_doctor_command()]
            report = self.last_report or {}
            models = report.get("models", {})
            req_file = "requirements-app.txt" if (ROOT / "requirements-app.txt").exists() else "requirements.txt"

            commands: list[list[str]] = [
                [sys.executable, "-m", "tools.bootstrap", "--create-venv", "--install-launcher"],
            ]

            commands.append([_python_executable(), "-m", "tools.bootstrap", "--install-requirements", "--requirements", req_file])

            # The current installer reuses valid GGUF files and also prepares
            # mmproj and llama-server, not merely model directories.
            commands.append([_python_executable(), "download_models.py"])
            commands.append([_python_executable(), "-m", "tools.doctor"])
            return commands

        def auto_prepare(self) -> None:
            if _is_production():
                self.run_command(_doctor_command(), on_success=self.refresh_model_list)
                return
            self.run_command_sequence(self._auto_setup_commands(), title="Автоподготовка ChatKodi", on_success=self.refresh_model_list)

        def auto_prepare_and_start(self) -> None:
            if _is_production():
                self.run_command(_doctor_command(), on_success=lambda: (self.refresh_model_list(), self.start_chatkodi()))
                return
            self.run_command_sequence(self._auto_setup_commands(), title="Автоподготовка и запуск ChatKodi", on_success=lambda: (self.refresh_model_list(), self.start_chatkodi()))

        def refresh_model_list(self) -> None:
            current = self.settings.get("model_path") or "auto"
            if hasattr(self, "model_choice"):
                self.model_choice.blockSignals(True)
                self.model_choice.clear()
                self.model_choice.addItem("Авто — выбрать скачанную/рекомендованную модель", "auto")
                for item in _discover_installed_llms():
                    self.model_choice.addItem(f"{item['name']} • {item['size']}", item["path"])
                self.model_choice.addItem("Только поиск — не загружать LLM", "search_only")
                self._set_combo_data(self.model_choice, current)
                self.model_choice.blockSignals(False)
                try:
                    self.model_choice.currentIndexChanged.disconnect(self.save_launch_profile)
                except Exception:
                    pass
                self.model_choice.currentIndexChanged.connect(lambda *_: self.save_launch_profile(show_dialog=False))

        def refresh_doctor(self, silent: bool = False) -> None:
            try:
                report = build_doctor_report()
                self.last_report = report
                py = report.get("python", {})
                models = report.get("models", {})
                runtime = report.get("runtime_profile", {})
                docs = report.get("documents", {})
                doctor = report.get("doctor", {})
                ready = not bool(doctor.get("blockers"))
                self.ready_badge.setText("Готов к запуску" if ready else "Нужна подготовка")
                self.python_card.body_label.setText(f"{py.get('python')} • venv={'да' if py.get('is_venv') else 'нет'}\n{py.get('executable')}")
                self.models_card.body_label.setText(f"Embeddings: {'есть' if models.get('embeddings') else 'нет'} • LLM: {'есть' if models.get('llm') else 'нет'}\n{models.get('llm_name') or models.get('error') or 'Авто выберет после скачивания'}")
                self.runtime_card.body_label.setText(runtime.get("summary", "—"))
                self.docs_card.body_label.setText(f"{docs.get('count', 0)} файлов\n{docs.get('path')}")
                if hasattr(self, "model_text"):
                    self.model_text.setPlainText(json.dumps(models, ensure_ascii=False, indent=2))
                if not silent:
                    self.append_log(json.dumps(doctor, ensure_ascii=False, indent=2))
            except Exception as exc:
                if not silent:
                    self.append_log(f"Doctor failed: {exc}")

        def _launch_env(self) -> dict[str, str]:
            profile = self.launch_profile.currentData() if hasattr(self, "launch_profile") else "auto"
            model_choice = self.model_choice.currentData() if hasattr(self, "model_choice") else "auto"
            env: dict[str, str] = {}
            # Auto в rc5 — безопасный и быстрый: на CPU/low-RAM он не будет
            # сам выбирать 4B/7B HF-модель, если есть риск многоминутной загрузки.
            env["CHATKODI_MODEL_AUTO_POLICY"] = "fast_safe"
            force_llm = bool(getattr(self, "force_llm_checkbox", None) and self.force_llm_checkbox.isChecked())
            if force_llm:
                env["CHATKODI_FORCE_LLM"] = "1"
                env["CHATKODI_ALLOW_HEAVY_CPU_LLM"] = "1"
            if profile and profile != "auto":
                if profile == "search_only":
                    env["CHATKODI_DISABLE_LLM"] = "1"
                    env["CHATKODI_PROFILE"] = "academic_safe"
                else:
                    env["CHATKODI_PROFILE"] = str(profile)
                env["CHATKODI_LAUNCH_PROFILE"] = str(profile)
            if model_choice and model_choice not in {"auto", "search_only"}:
                env["CHATKODI_MODEL_PATH"] = str(model_choice)
                # Ручной выбор модели — это осознанное действие пользователя.
                # Даже если модель тяжёлая для CPU, не игнорируем её молча.
                env["CHATKODI_ALLOW_HEAVY_CPU_LLM"] = "1"
                env["CHATKODI_FORCE_LLM"] = "1"
            elif model_choice == "search_only":
                env["CHATKODI_DISABLE_LLM"] = "1"
            env["CHATKODI_THEME"] = self.theme
            if _is_production():
                env["CHATKODI_PRODUCTION"] = "1"
                env["CHATKODI_APP_ROOT"] = str(ROOT)
                if (ROOT / "portable.flag").exists():
                    env["CHATKODI_PORTABLE"] = "1"
                    env["CHATKODI_RUNTIME_DIR"] = str(ROOT / "runtime")
            return env

        def start_chatkodi(self) -> None:
            if self._app_process is not None and self._app_process.poll() is None:
                QMessageBox.information(
                    self,
                    "ChatKodi уже запускается",
                    "Процесс приложения уже работает. Дождитесь окна загрузки; повторный запуск может исчерпать память.",
                )
                return
            env = self._launch_env()
            if env.get("CHATKODI_DISABLE_LLM") == "1":
                answer = QMessageBox.question(
                    self,
                    "Запуск без LLM?",
                    "Сейчас выбран режим без локальной LLM. Уточняющие вопросы по найденным данным и обычная генерация будут недоступны.\n\nЗапустить только поиск?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if answer != QMessageBox.Yes:
                    self.reset_to_auto()
                    env = self._launch_env()
            self.save_launch_profile(show_dialog=False)
            profile_text = self.launch_profile.currentText() if hasattr(self, "launch_profile") else "Авто"
            model_text = self.model_choice.currentText() if hasattr(self, "model_choice") else "Авто"
            self.append_log(f"Запуск ChatKodi: профиль={profile_text}; модель={model_text}", jump=False)
            try:
                # Важный момент для режима «Авто»: не наследуем старые ручные
                # переменные из консоли/прошлого запуска. Иначе можно случайно
                # получить «LLM не загружена» из-за CHATKODI_DISABLE_LLM=1, хотя
                # в лаунчере выбран обычный автоматический режим.
                launch_env = os.environ.copy()
                for key in (
                    "CHATKODI_DISABLE_LLM",
                    "CHATKODI_MODEL_PATH",
                    "CHATKODI_PROFILE",
                    "CHATKODI_LAUNCH_PROFILE",
                    "CHATKODI_THEME",
                    "CHATKODI_MODEL_AUTO_POLICY",
                    "CHATKODI_ALLOW_HEAVY_CPU_LLM",
                    "CHATKODI_FORCE_LLM",
                    "CHATKODI_MAX_CPU_LLM_SIZE_GB",
                    "CHATKODI_LLM_STARTUP_MODE",
                    "CHATKODI_PRODUCTION",
                    "CHATKODI_PORTABLE",
                    "CHATKODI_APP_ROOT",
                    "CHATKODI_RUNTIME_DIR",
                    "CHATKODI_TRUST_PATH_OVERRIDES",
                ):
                    launch_env.pop(key, None)
                launch_env.update(env)
                app_exe = _chatkodi_app_executable()
                creationflags = 0
                if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                    creationflags = subprocess.CREATE_NO_WINDOW
                if app_exe:
                    self._app_process = subprocess.Popen(
                        [str(app_exe)],
                        cwd=str(ROOT),
                        env=launch_env,
                        creationflags=creationflags,
                    )
                else:
                    self._app_process = subprocess.Popen(
                        [_python_executable(), "main.py"],
                        cwd=str(ROOT),
                        env=launch_env,
                        creationflags=creationflags,
                    )
                self.start_btn.setEnabled(False)
                self.start_btn.setText("ChatKodi запускается…")
                QTimer.singleShot(2500, self._check_started_process)
            except Exception as exc:
                QMessageBox.warning(self, "Не удалось запустить", str(exc))

        def _check_started_process(self) -> None:
            process = self._app_process
            if process is None:
                return
            code = process.poll()
            if code is None:
                self.start_btn.setText("ChatKodi работает")
                QTimer.singleShot(2000, self._poll_app_process)
                return
            self._app_process = None
            self._restore_start_button()
            log_path = runtime_root() / "logs" / "chatkodi.log"
            QMessageBox.warning(
                self,
                "ChatKodi завершился при запуске",
                f"Приложение завершилось с кодом {code}.\n\nДиагностический лог: {log_path}",
            )

        def _poll_app_process(self) -> None:
            process = self._app_process
            if process is None:
                self._restore_start_button()
                return
            if process.poll() is None:
                QTimer.singleShot(2000, self._poll_app_process)
                return
            self._app_process = None
            self._restore_start_button()

        def _restore_start_button(self) -> None:
            self.start_btn.setEnabled(True)
            self.start_btn.setText("Запустить ChatKodi")

        def open_manuals(self) -> None:
            """Открывает встроенный просмотр manuals в лаунчере."""
            if hasattr(self, "manuals_page"):
                self.tabs.setCurrentWidget(self.manuals_page)
            else:
                self.open_manuals_folder()

        def open_manuals_folder(self) -> None:
            path = ROOT / "manuals"
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(path)])

        def create_support_bundle(self) -> None:
            if _is_production():
                self.run_command(_doctor_command())
                return
            self.run_command([_python_executable(), "support_bundle.py"])

        def save_launch_profile(self, *args, show_dialog: bool = True) -> None:
            try:
                payload = {
                    "launch_profile": self.launch_profile.currentData() if hasattr(self, "launch_profile") else "auto",
                    "model_path": self.model_choice.currentData() if hasattr(self, "model_choice") else "auto",
                    "model_label": self.model_choice.currentText() if hasattr(self, "model_choice") else "Авто",
                    "force_llm": bool(getattr(self, "force_llm_checkbox", None) and self.force_llm_checkbox.isChecked()),
                    "theme": self.theme,
                }
                _save_launcher_settings(payload)
                self.settings = payload
                if show_dialog:
                    QMessageBox.information(self, "Сохранено", f"Настройки лаунчера сохранены:\n{_settings_path()}")
            except Exception as exc:
                QMessageBox.warning(self, "Ошибка сохранения", str(exc))

    app = QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(run_launcher())
