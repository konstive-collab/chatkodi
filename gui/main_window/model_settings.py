"""«Модель и ассистент» panel on the Settings page."""

from __future__ import annotations

import sys

from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from core.agent.settings import AssistantSettings, apply_to_environment, load_settings, save_settings

_RESTART_FIELDS = ("context_tokens", "device", "vision")


def _creativity_word(value: float) -> str:
    if value <= 0.15:
        return "строго"
    if value <= 0.45:
        return "сдержанно"
    if value <= 0.75:
        return "живо"
    return "свободно"


class ChatKodiModelSettingsMixin:
    def _init_model_settings(self) -> None:
        self.assistant_settings = load_settings()
        self._settings_at_start = AssistantSettings(**vars(self.assistant_settings))
        self.suggestions_enabled = self.assistant_settings.suggestions
        self.animate_answers = self.assistant_settings.animate
        self.allow_console_commands = self.assistant_settings.commands
        self.voice_name = self.assistant_settings.voice

    def _build_model_settings(self, parent) -> QFrame:
        if not hasattr(self, "assistant_settings"):
            self._init_model_settings()
        s = self.assistant_settings
        frame = QFrame(parent)
        frame.setObjectName("surfaceFrame")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)
        title = QLabel("Модель и ассистент", frame)
        title.setObjectName("surfaceTitle")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        outer.addWidget(title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        outer.addLayout(grid)
        row = 0

        def label(text: str, hint: str = "") -> QLabel:
            widget = QLabel(text, frame)
            if hint:
                widget.setToolTip(hint)
            return widget

        # creativity
        self.creativity_slider = QSlider(Qt.Horizontal, frame)
        self.creativity_slider.setRange(0, 10)
        self.creativity_slider.setValue(int(round(s.creativity * 10)))
        self.creativity_value = QLabel(frame)
        self.creativity_value.setMinimumWidth(110)
        box = QHBoxLayout()
        box.addWidget(self.creativity_slider, 1)
        box.addWidget(self.creativity_value)
        grid.addWidget(label("Творчество в письмах и разговоре", "Ответы по документам всегда строгие (температура 0) — ради точности"), row, 0)
        grid.addLayout(box, row, 1)
        row += 1

        def combo(items, current):
            widget = QComboBox(frame)
            widget.setObjectName("compactCombo")
            for value, text in items:
                widget.addItem(text, value)
            index = widget.findData(current)
            widget.setCurrentIndex(max(0, index))
            return widget

        self.length_combo = combo([("short", "Короче"), ("normal", "Обычно"), ("long", "Подробнее")], s.answer_length)
        grid.addWidget(label("Длина ответов"), row, 0)
        grid.addWidget(self.length_combo, row, 1)
        row += 1
        self.context_combo = combo([(8192, "8 тыс. токенов — экономно"), (16384, "16 тыс. токенов — рекомендуется"), (32768, "32 тыс. токенов — длинные документы")], s.context_tokens)
        grid.addWidget(label("Объём контекста *", "Сколько текста модель видит за раз; больше — больше видеопамяти"), row, 0)
        grid.addWidget(self.context_combo, row, 1)
        row += 1
        self.device_combo = combo([("auto", "Авто — видеокарта, если есть"), ("gpu", "Видеокарта"), ("cpu", "Процессор (медленно)")], s.device)
        grid.addWidget(label("Где работает модель *"), row, 0)
        grid.addWidget(self.device_combo, row, 1)
        row += 1
        self.voice_combo = combo([("irina", "Ирина"), ("dmitri", "Дмитрий")], s.voice)
        grid.addWidget(label("Голос ответов"), row, 0)
        grid.addWidget(self.voice_combo, row, 1)
        row += 1
        self.autonomy_combo = combo([("ask", "Только с разрешения"), ("auto", "Автомат — спрашивает только о рискованном"), ("full", "Полная самостоятельность")], s.autonomy)
        grid.addWidget(label(
            "Самостоятельность",
            "Чтение и поиск всегда без вопросов. «Автомат» сам выполняет обычные действия (открыть, скопировать, "
            "переименовать с отменой) и спрашивает о рискованных. Необратимое (удаление, форматирование) "
            "подтверждается всегда, при любой настройке.",
        ), row, 0)
        grid.addWidget(self.autonomy_combo, row, 1)
        row += 1

        checks = QGridLayout()
        checks.setHorizontalSpacing(18)
        self.vision_check = QCheckBox("Зрение: фото и сканы *", frame)
        self.vision_check.setChecked(s.vision)
        self.speech_check = QCheckBox("Голосовой ввод (микрофон)", frame)
        self.speech_check.setChecked(s.speech)
        self.commands_check = QCheckBox("Команды PowerShell на компьютере", frame)
        self.commands_check.setChecked(s.commands)
        self.suggest_check = QCheckBox("Подсказки следующего шага", frame)
        self.suggest_check.setChecked(s.suggestions)
        self.animate_check = QCheckBox("Ответ печатается по словам", frame)
        self.animate_check.setChecked(s.animate)
        self.memory_check = QCheckBox("Память между чатами", frame)
        self.memory_check.setChecked(s.memory)
        self.memory_check.setToolTip("Ассистент запоминает устойчивые факты о вас и вашей работе и учитывает их во всех чатах")
        for index, widget in enumerate((self.vision_check, self.speech_check, self.commands_check, self.suggest_check, self.animate_check, self.memory_check)):
            checks.addWidget(widget, index // 2, index % 2)
        outer.addLayout(checks)

        bottom = QHBoxLayout()
        self.restart_hint = QLabel("* применяется после перезапуска модели", frame)
        self.restart_hint.setObjectName("surfaceHint")
        instructions = QPushButton("Личные инструкции…", frame)
        instructions.setObjectName("secondaryButton")
        instructions.clicked.connect(self._edit_personal_instructions)
        memory = QPushButton("Память…", frame)
        memory.setObjectName("secondaryButton")
        memory.clicked.connect(self._edit_assistant_memory)
        folders = QPushButton("Папки для поиска…", frame)
        folders.setObjectName("secondaryButton")
        folders.clicked.connect(self._edit_search_folders)
        self.restart_btn = QPushButton("Перезапустить ChatKodi", frame)
        self.restart_btn.setObjectName("secondaryButton")
        self.restart_btn.clicked.connect(self._restart_application)
        bottom.addWidget(self.restart_hint, 1)
        bottom.addWidget(instructions)
        bottom.addWidget(memory)
        bottom.addWidget(folders)
        bottom.addWidget(self.restart_btn)
        outer.addLayout(bottom)

        self.creativity_slider.valueChanged.connect(self._on_model_settings_changed)
        for widget in (self.length_combo, self.context_combo, self.device_combo, self.voice_combo, self.autonomy_combo):
            widget.currentIndexChanged.connect(self._on_model_settings_changed)
        for widget in (self.vision_check, self.speech_check, self.commands_check, self.suggest_check, self.animate_check, self.memory_check):
            widget.toggled.connect(self._on_model_settings_changed)
        self._update_settings_labels()
        return frame

    def _update_settings_labels(self) -> None:
        value = self.creativity_slider.value() / 10
        self.creativity_value.setText(f"{value:.1f} — {_creativity_word(value)}".replace(".", ","))
        changed = any(getattr(self.assistant_settings, f) != getattr(self._settings_at_start, f) for f in _RESTART_FIELDS)
        self.restart_hint.setText("Нужен перезапуск, чтобы применить отмеченные *" if changed else "* применяется после перезапуска модели")
        self.restart_btn.setProperty("accent", changed)
        self.restart_btn.setObjectName("commandRunButton" if changed else "secondaryButton")
        self.restart_btn.style().unpolish(self.restart_btn)
        self.restart_btn.style().polish(self.restart_btn)

    def _on_model_settings_changed(self, *_args) -> None:
        s = self.assistant_settings
        s.creativity = self.creativity_slider.value() / 10
        s.answer_length = self.length_combo.currentData()
        s.context_tokens = int(self.context_combo.currentData())
        s.device = self.device_combo.currentData()
        s.voice = self.voice_combo.currentData()
        s.vision = self.vision_check.isChecked()
        s.speech = self.speech_check.isChecked()
        s.commands = self.commands_check.isChecked()
        s.suggestions = self.suggest_check.isChecked()
        s.animate = self.animate_check.isChecked()
        s.autonomy = self.autonomy_combo.currentData()
        s.memory = self.memory_check.isChecked()
        save_settings(s)
        # applied right away
        self.suggestions_enabled = s.suggestions
        self.animate_answers = s.animate
        self.allow_console_commands = s.commands
        if getattr(self, "voice_name", "") != s.voice:
            self.voice_name = s.voice
            self._tts = None
        for agent in getattr(self, "_agents", {}).values():
            agent.settings = s
            agent.allow_commands = s.commands
        if hasattr(self, "voice_btn"):
            self.voice_btn.setVisible(s.speech)
        self._update_settings_labels()
        self.status.showMessage("Настройки сохранены")

    def _edit_assistant_memory(self) -> None:
        """What the assistant remembers between chats: read, delete, add, clear."""
        from core.agent.memory import get_memory

        memory = get_memory()
        dialog = QDialog(self)
        dialog.setWindowTitle("Память ассистента")
        dialog.resize(620, 440)
        layout = QVBoxLayout(dialog)
        hint = QLabel(
            "Факты, которые ассистент запомнил из разговоров и учитывает во всех чатах. "
            "Лишнее можно удалить, нужное — добавить вручную.", dialog,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        facts = QListWidget(dialog)
        facts.setWordWrap(True)
        layout.addWidget(facts, 1)

        def refresh() -> None:
            facts.clear()
            for fact in memory.facts:
                item = QListWidgetItem(fact.text)
                item.setData(Qt.UserRole, fact.id)
                facts.addItem(item)
            if not memory.facts:
                empty = QListWidgetItem("Пока ничего не запомнено")
                empty.setFlags(Qt.NoItemFlags)
                facts.addItem(empty)

        entry = QLineEdit(dialog)
        entry.setPlaceholderText("Например: Пользователь — инженер ПТО ООО «СтройЛаб»")
        add = QPushButton("Добавить", dialog)
        remove = QPushButton("Удалить выбранное", dialog)
        clear = QPushButton("Очистить всё", dialog)
        close = QPushButton("Закрыть", dialog)
        for btn in (add, remove, clear, close):
            btn.setObjectName("secondaryButton")
        add_row = QHBoxLayout()
        add_row.addWidget(entry, 1)
        add_row.addWidget(add)
        layout.addLayout(add_row)
        buttons = QHBoxLayout()
        buttons.addWidget(remove)
        buttons.addWidget(clear)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        def add_fact() -> None:
            if memory.add(entry.text()):
                entry.clear()
                refresh()

        def remove_fact() -> None:
            for item in facts.selectedItems():
                if item.data(Qt.UserRole):
                    memory.remove(item.data(Qt.UserRole))
            refresh()

        def clear_all() -> None:
            if QMessageBox.question(dialog, "Очистить память", "Забыть всё, что ассистент запомнил?") == QMessageBox.Yes:
                memory.clear()
                refresh()

        add.clicked.connect(add_fact)
        entry.returnPressed.connect(add_fact)
        remove.clicked.connect(remove_fact)
        clear.clicked.connect(clear_all)
        close.clicked.connect(dialog.accept)
        refresh()
        dialog.exec_()

    def _edit_search_folders(self) -> None:
        """Folders the assistant searches besides the library, Desktop, Documents and Downloads."""
        s = self.assistant_settings
        dialog = QDialog(self)
        dialog.setWindowTitle("Папки для поиска файлов")
        dialog.resize(560, 360)
        layout = QVBoxLayout(dialog)
        hint = QLabel(
            "Ассистент ищет файлы сначала в библиотеке, затем на Рабочем столе, в Документах и Загрузках. "
            "Здесь можно добавить другие папки, например сетевую папку отдела.", dialog,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        folders = QListWidget(dialog)
        folders.addItems(list(s.search_folders))
        layout.addWidget(folders, 1)
        add = QPushButton("Добавить папку…", dialog)
        remove = QPushButton("Убрать", dialog)
        close = QPushButton("Закрыть", dialog)
        for btn in (add, remove, close):
            btn.setObjectName("secondaryButton")
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch(1)
        row.addWidget(close)
        layout.addLayout(row)

        def store() -> None:
            s.search_folders = [folders.item(i).text() for i in range(folders.count())]
            save_settings(s)

        def add_folder() -> None:
            folder = QFileDialog.getExistingDirectory(dialog, "Папка для поиска")
            if folder and folder not in s.search_folders:
                folders.addItem(folder)
                store()

        def remove_folder() -> None:
            for item in folders.selectedItems():
                folders.takeItem(folders.row(item))
            store()

        add.clicked.connect(add_folder)
        remove.clicked.connect(remove_folder)
        close.clicked.connect(dialog.accept)
        dialog.exec_()

    def _restart_application(self) -> None:
        apply_to_environment(self.assistant_settings)
        if getattr(sys, "frozen", False):
            program, args = sys.executable, sys.argv[1:]
        else:
            program, args = sys.executable, sys.argv
        if QProcess.startDetached(program, args):
            QApplication.quit()
        else:
            self.status.showMessage("Не удалось перезапустить — закройте и откройте ChatKodi вручную")
