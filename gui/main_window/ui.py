"""Сборка современного интерфейса главного окна ChatKodi."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, QRect, Qt, QTimer
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QButtonGroup,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.chat_mode import ChatMode
from core.date_range_parser import parse_date_constraints
from core.entity_resolution import extract_structured_entities
from core.config import get_config
from core.logging_utils import get_logger
from core.paths import resolve_project_path, runtime_root
from core.system_profile import detect_system_profile
from core.search_preferences import QUERY_MODE_LABELS, STRICTNESS_LABELS, SOURCE_SCOPE_LABELS, normalize_query_mode, normalize_strictness, normalize_source_scope
from gui.styles import load_styles
from core.theme_manager import THEME_ORDER, next_theme, theme_icon, theme_label as get_theme_label, theme_short_label

logger = get_logger(__name__)


class ChatKodiUiMixin:
    def _init_ui(self) -> None:
        self.setMinimumSize(960, 640)
        self.setAcceptDrops(True)
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                min(1600, max(960, int(available.width() * 0.96))),
                min(980, max(640, int(available.height() * 0.94))),
            )
        else:
            self.resize(1440, 860)
        self.setWindowTitle("ChatKodi 2.0.0")
        if os.path.exists("assets/app_icon.png"):
            self.setWindowIcon(QIcon("assets/app_icon.png"))

        central = QWidget(self)
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._create_menu_bar()
        self._create_sidebar()
        root.addWidget(self.sidebar_frame)

        self.main_splitter = QSplitter(Qt.Horizontal, central)
        self.main_splitter.setChildrenCollapsible(False)
        root.addWidget(self.main_splitter, 1)

        self.page_stack = QStackedWidget(self.main_splitter)
        self.page_stack.setObjectName("workspaceStack")
        self.main_splitter.addWidget(self.page_stack)

        self._create_chat_page()
        self._create_documents_page()
        self._create_index_page()
        self._create_diagnostics_page()
        self._create_settings_page()

        from gui.components.source_pane import SourcePane

        self.source_pane = SourcePane(self.main_splitter, on_open_external=self._open_source_path)
        self.main_splitter.addWidget(self.source_pane)
        self.source_pane.closed.connect(lambda: QTimer.singleShot(30, self._fit_chat_bar))
        self._create_inspector_panel()
        self.main_splitter.addWidget(self.inspector_panel)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([1200, 0, 0])
        self.main_splitter.splitterMoved.connect(lambda *_: self._fit_chat_bar())

        self._create_status_bar()
        if hasattr(self, "_install_shortcuts"):
            self._install_shortcuts()
        self._apply_theme()
        self._switch_workspace("chat")
        self._refresh_workspace_views()
        if not self._restore_chat_history():
            self._show_welcome_message()
        self._apply_responsive_layout()

    def _create_menu_bar(self) -> None:
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        file_menu = QMenu("Файл", self)
        menubar.addMenu(file_menu)

        select_folder_action = QAction("Выбрать папку с документами...", self)
        select_folder_action.triggered.connect(self._select_docs_folder)
        file_menu.addAction(select_folder_action)

        add_docs_action = QAction("Добавить документы…", self)
        add_docs_action.triggered.connect(self._add_documents)
        file_menu.addAction(add_docs_action)

        reindex_action = QAction("Обновить индекс", self)
        reindex_action.triggered.connect(self._update_index)
        file_menu.addAction(reindex_action)

        export_chat_action = QAction("Экспорт диалога…", self)
        export_chat_action.triggered.connect(self._export_chat_history)
        file_menu.addAction(export_chat_action)

        clear_chat_action = QAction("Очистить историю диалога…", self)
        clear_chat_action.triggered.connect(self._clear_chat_history)
        file_menu.addAction(clear_chat_action)

        attach_action = QAction("Приложить файлы к чату…", self)
        attach_action.setShortcut("Ctrl+O")
        attach_action.triggered.connect(self._choose_attachments)
        file_menu.addAction(attach_action)

        instructions_action = QAction("Личные инструкции…", self)
        instructions_action.triggered.connect(self._edit_personal_instructions)
        file_menu.addAction(instructions_action)

        outputs_action = QAction("Открыть папку созданных файлов", self)
        outputs_action.triggered.connect(self._open_outputs_folder)
        file_menu.addAction(outputs_action)

        self.voice_dialog_action = QAction("Голосовой диалог (слушать после ответа)", self)
        self.voice_dialog_action.setCheckable(True)
        self.voice_dialog_action.toggled.connect(self._toggle_voice_dialog)
        file_menu.addAction(self.voice_dialog_action)

        file_menu.addSeparator()

        self.streaming_enabled_action = QAction("Streaming ответов", self)
        self.streaming_enabled_action.setCheckable(True)
        self.streaming_enabled_action.setChecked(True)
        file_menu.addAction(self.streaming_enabled_action)

        docs_menu = QMenu("Документы", self)
        menubar.addMenu(docs_menu)

        self.full_docs_action = QAction("Целые документы", self)
        self.full_docs_action.setCheckable(True)
        self.full_docs_action.setChecked(bool(getattr(self, "use_full_documents", False)))
        self.full_docs_action.triggered.connect(self._set_full_documents_mode)
        docs_menu.addAction(self.full_docs_action)

        self.chunks_action = QAction("Фрагменты", self)
        self.chunks_action.setCheckable(True)
        self.chunks_action.setChecked(not bool(getattr(self, "use_full_documents", False)))
        self.chunks_action.triggered.connect(self._set_chunks_mode)
        docs_menu.addAction(self.chunks_action)

        docs_group = QActionGroup(self)
        docs_group.addAction(self.full_docs_action)
        docs_group.addAction(self.chunks_action)
        docs_group.setExclusive(True)

        mode_menu = QMenu("Режим", self)
        menubar.addMenu(mode_menu)
        mode_menu.menuAction().setVisible(False)  # 2.0: the assistant chooses the way to answer itself

        self.agent_action = QAction("Ассистент (файлы, задачи, документы)", self)
        self.agent_action.setCheckable(True)
        self.agent_action.setChecked(True)
        self.agent_action.triggered.connect(lambda: self._set_mode(ChatMode.AGENT))
        mode_menu.addAction(self.agent_action)

        self.rag_action = QAction("RAG (поиск + LLM)", self)
        self.rag_action.setCheckable(True)
        self.rag_action.triggered.connect(lambda: self._set_mode(ChatMode.RAG))
        mode_menu.addAction(self.rag_action)

        self.chat_action = QAction("Чат (только LLM)", self)
        self.chat_action.setCheckable(True)
        self.chat_action.triggered.connect(lambda: self._set_mode(ChatMode.CHAT))
        mode_menu.addAction(self.chat_action)

        self.search_action = QAction("Поиск (только документы)", self)
        self.search_action.setCheckable(True)
        self.search_action.triggered.connect(lambda: self._set_mode(ChatMode.EXTRACT))
        mode_menu.addAction(self.search_action)

        view_menu = QMenu("Вид", self)
        menubar.addMenu(view_menu)
        self.toggle_inspector_action = QAction("Показать правую панель", self)
        self.toggle_inspector_action.setShortcut("F4")
        self.toggle_inspector_action.setCheckable(True)
        self.toggle_inspector_action.triggered.connect(lambda checked: self._toggle_inspector(force=checked))
        view_menu.addAction(self.toggle_inspector_action)

        self.toggle_overview_action = QAction("Показать сводку на главной", self)
        self.toggle_overview_action.setCheckable(True)
        self.toggle_overview_action.triggered.connect(lambda checked: self._toggle_chat_overview(force=checked))
        view_menu.addAction(self.toggle_overview_action)

        tools_menu = QMenu("Инструменты", self)
        menubar.addMenu(tools_menu)
        health_action = QAction("Проверка системы", self)
        health_action.triggered.connect(self._run_healthcheck_dialog)
        tools_menu.addAction(health_action)
        bundle_action = QAction("Собрать support bundle", self)
        bundle_action.triggered.connect(self._create_support_bundle_dialog)
        tools_menu.addAction(bundle_action)

        help_menu = QMenu("Справка", self)
        menubar.addMenu(help_menu)
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        stats_action = QAction("Статистика документов", self)
        stats_action.triggered.connect(self._show_documents_stats)
        help_menu.addAction(stats_action)

    def _create_sidebar(self) -> None:
        from core.theme_manager import theme_palette
        from gui.components.fonts import BRAND_FAMILY, family_or_fallback, ui_family
        from gui.components.resources import ResourceMonitor

        self.sidebar_frame = QFrame(self)
        self.sidebar_frame.setObjectName("sidebarFrame")
        self.sidebar_frame.setFixedWidth(258)

        layout = QVBoxLayout(self.sidebar_frame)
        layout.setContentsMargins(14, 16, 14, 12)
        layout.setSpacing(8)

        palette = theme_palette(getattr(self, "current_theme", "classic"))
        self.sidebar_title_label = QLabel(self.sidebar_frame)
        self.sidebar_title_label.setObjectName("sidebarWordmark")
        self.sidebar_title_label.setTextFormat(Qt.RichText)
        self.sidebar_title_label.setFont(QFont(family_or_fallback(BRAND_FAMILY), 15))
        self._set_wordmark(palette)
        layout.addWidget(self.sidebar_title_label)
        layout.addSpacing(4)

        # chats: new chat, search, list
        self._build_chat_sidebar(layout)

        # sections: compact list under the chats
        self.sidebar_button_group = QButtonGroup(self.sidebar_frame)
        self.sidebar_button_group.setExclusive(True)
        self.workspace_buttons = {}
        sections = QFrame(self.sidebar_frame)
        sections.setObjectName("sidebarSections")
        sections_layout = QVBoxLayout(sections)
        sections_layout.setContentsMargins(0, 6, 0, 2)
        sections_layout.setSpacing(1)
        for page_name, text in [
            ("chat", "Чат"),
            ("documents", "Документы"),
            ("index", "База и индекс"),
            ("diagnostics", "Диагностика"),
            ("settings", "Настройки"),
        ]:
            btn = QPushButton(text, sections)
            btn.setObjectName("navButton")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty("navKey", page_name)
            btn.clicked.connect(lambda checked=False, name=page_name: self._switch_workspace(name))
            self.sidebar_button_group.addButton(btn)
            self.workspace_buttons[page_name] = btn
            sections_layout.addWidget(btn)
            if page_name == "chat":
                btn.hide()  # chats in the list above open the chat
        layout.addWidget(sections)

        # PC resources: one quiet line; a click unfolds the rings and the chart
        self.resource_toggle = QPushButton("Ресурсы ПК", self.sidebar_frame)
        self.resource_toggle.setObjectName("resourceToggle")
        self.resource_toggle.setCheckable(True)
        self.resource_toggle.setCursor(Qt.PointingHandCursor)
        self.resource_toggle.setToolTip("Показать загрузку процессора, памяти и видеокарты")
        layout.addWidget(self.resource_toggle)
        self.resource_monitor = ResourceMonitor(self.sidebar_frame, palette, family=ui_family())
        self.resource_monitor.setMinimumHeight(0)
        self.resource_monitor.setMaximumHeight(0)
        layout.addWidget(self.resource_monitor)
        self._resource_anim = QPropertyAnimation(self.resource_monitor, b"maximumHeight", self)
        self._resource_anim.setDuration(220)
        self._resource_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._resource_anim.valueChanged.connect(lambda v: self.resource_monitor.setMinimumHeight(int(v)))
        self._resource_summary = ("", "ok")

        def show_summary(text: str, level: str) -> None:
            self._resource_summary = (text, level)
            arrow = "▾" if self.resource_toggle.isChecked() else "▸"
            self.resource_toggle.setText(f"Ресурсы ПК   {text}   {arrow}")
            if self.resource_toggle.property("level") != level:
                self.resource_toggle.setProperty("level", level)
                self.resource_toggle.style().unpolish(self.resource_toggle)
                self.resource_toggle.style().polish(self.resource_toggle)

        def toggle(checked: bool) -> None:
            self._resource_anim.stop()
            self._resource_anim.setStartValue(self.resource_monitor.maximumHeight())
            self._resource_anim.setEndValue(150 if checked else 0)
            self._resource_anim.start()
            show_summary(*self._resource_summary)

        self.resource_monitor.sampled.connect(show_summary)
        self.resource_toggle.toggled.connect(toggle)

        # values still refreshed by the status code; shown in tooltips only
        self.sidebar_runtime_card = QFrame(self.sidebar_frame)
        self.sidebar_runtime_card.hide()
        self.sidebar_profile_value = QLabel("", self.sidebar_runtime_card)
        self.sidebar_docs_value = QLabel("", self.sidebar_runtime_card)
        self.sidebar_llm_value = QLabel("", self.sidebar_runtime_card)
        self.sidebar_subtitle_label = QLabel("", self.sidebar_runtime_card)
        self._refresh_nav_icons(palette)
        self._refresh_chat_list()

    def _set_wordmark(self, palette) -> None:
        accent = palette.get("accent", "#22c55e")
        title = palette.get("title", "#ffffff")
        self.sidebar_title_label.setText(f'<span style="color:{title};">chatkodi</span><span style="color:{accent};">.</span>')

    def _refresh_nav_icons(self, palette) -> None:
        from PyQt5.QtCore import QSize

        from gui.components import icons

        color = palette.get("muted", "#94a3b8")
        draw = {"documents": icons.document, "index": icons.database, "diagnostics": icons.pulse, "settings": icons.gear}
        for key, button in getattr(self, "workspace_buttons", {}).items():
            if key in draw:
                button.setIcon(draw[key](color))
                button.setIconSize(QSize(17, 17))
        if hasattr(self, "resource_monitor"):
            self.resource_monitor.set_palette(palette)
        if hasattr(self, "sidebar_title_label"):
            self._set_wordmark(palette)
    def _create_chat_page(self) -> None:
        page = QWidget(self.page_stack)
        page.setObjectName("chatPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(14)

        header = QFrame(page)
        header.setObjectName("pageHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 16, 18, 16)
        header_layout.setSpacing(16)

        title_box = QVBoxLayout()
        title_box.setSpacing(6)
        self.logo_label = QLabel("Чат и поиск", header)
        self.logo_label.setObjectName("pageTitle")
        self.logo_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.subtitle_label = QLabel("Задавайте вопросы по документам и локальной модели. Основные действия вынесены в верхнее меню.", header)
        self.subtitle_label.setObjectName("pageSubtitle")
        self.subtitle_label.setWordWrap(True)
        title_box.addWidget(self.logo_label)
        title_box.addWidget(self.subtitle_label)

        meta_box = QVBoxLayout()
        meta_box.setSpacing(4)
        self.info_label = QLabel("Документы не загружены", header)
        self.info_label.setObjectName("headerInfo")
        self.info_label.setWordWrap(False)
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.folder_label = QLabel("", header)
        self.folder_label.setObjectName("headerHint")
        self.folder_label.setWordWrap(False)
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        meta_box.addWidget(self.info_label)
        meta_box.addWidget(self.folder_label)
        header_layout.addLayout(title_box, 5)
        header_layout.addLayout(meta_box, 5)
        layout.addWidget(header)
        header.hide()  # 2.0: the chat bar below replaces the big header

        cards = QGridLayout()
        cards.setContentsMargins(0, 0, 0, 0)
        cards.setHorizontalSpacing(10)
        cards.setVerticalSpacing(10)

        self.home_profile_card = self._create_stat_card("Профиль системы", cards, 0, 0)
        self.home_docs_card = self._create_stat_card("Документы в базе", cards, 0, 1)
        self.home_mode_card = self._create_stat_card("Режим ответа", cards, 0, 2)
        self.home_safe_card = self._create_stat_card("Режим защиты", cards, 0, 3)
        self.chat_overview_frame = QFrame(page)
        self.chat_overview_frame.setObjectName("transparentFrame")
        self.chat_overview_frame.setLayout(cards)
        self.chat_overview_frame.hide()
        self.chat_overview_visible = False
        layout.addWidget(self.chat_overview_frame)

        mode_frame = QFrame(page)
        mode_frame.setObjectName("modeFrame")
        mode_layout = QVBoxLayout(mode_frame)
        mode_layout.setContentsMargins(14, 10, 14, 10)
        mode_layout.setSpacing(8)

        primary_row = QHBoxLayout()
        primary_row.setSpacing(10)

        auto_label = QLabel("●  Авто", mode_frame)
        auto_label.setObjectName("autoModeBadge")
        auto_label.setToolTip("По умолчанию ChatKodi сам выбирает стратегию поиска и ответа")

        self.mode_agent = QPushButton("Ассистент", mode_frame)
        self.mode_agent.setObjectName("modeButton")
        self.mode_agent.setCheckable(True)
        self.mode_agent.setChecked(True)
        self.mode_agent.setToolTip("Файлы, фото и сканы в чат, проверка по нормативам, документы по образцу, перевод, команды")
        self.mode_agent.clicked.connect(lambda: self._set_mode(ChatMode.AGENT))
        self.mode_rag = QPushButton("Документы + ответ", mode_frame)
        self.mode_rag.setObjectName("modeButton")
        self.mode_rag.setCheckable(True)
        self.mode_rag.clicked.connect(lambda: self._set_mode(ChatMode.RAG))
        self.mode_chat = QPushButton("Только чат", mode_frame)
        self.mode_chat.setObjectName("modeButton")
        self.mode_chat.setCheckable(True)
        self.mode_chat.clicked.connect(lambda: self._set_mode(ChatMode.CHAT))
        self.mode_search = QPushButton("Только поиск", mode_frame)
        self.mode_search.setObjectName("modeButton")
        self.mode_search.setCheckable(True)
        self.mode_search.clicked.connect(lambda: self._set_mode(ChatMode.EXTRACT))
        self.mode_group = QButtonGroup(mode_frame)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.mode_agent)
        self.mode_group.addButton(self.mode_rag)
        self.mode_group.addButton(self.mode_chat)
        self.mode_group.addButton(self.mode_search)

        self.mode_status = QLabel("Авто подбирает поиск и строгость", mode_frame)
        self.mode_status.setObjectName("modeStatus")
        self.mode_status.setWordWrap(False)

        self.advanced_search_toggle = QPushButton("▸ Доп. настройки", mode_frame)
        self.advanced_search_toggle.setObjectName("advancedToggleButton")
        self.advanced_search_toggle.setCheckable(True)
        self.advanced_search_toggle.setCursor(Qt.PointingHandCursor)
        self.advanced_search_toggle.setToolTip("Показать ручные настройки поиска. В обычной работе лучше оставить Авто.")
        self.advanced_search_toggle.clicked.connect(self._toggle_advanced_search_panel)

        # one thin bar: chat title on the left, mode switch and options on the right
        self._hidden_holder = QFrame(page)
        self._hidden_holder.hide()
        auto_label.setParent(self._hidden_holder)
        self.mode_status.setParent(self._hidden_holder)  # kept for status code, not shown
        self.chat_title_label = QLabel("Новый чат", mode_frame)
        self.chat_title_label.setObjectName("chatTitle")
        self.chat_title_label.setMinimumWidth(120)
        mode_switch = QFrame(mode_frame)
        mode_switch.setObjectName("modeSwitch")
        switch_layout = QHBoxLayout(mode_switch)
        switch_layout.setContentsMargins(3, 3, 3, 3)
        switch_layout.setSpacing(2)
        for button in (self.mode_agent, self.mode_rag, self.mode_chat, self.mode_search):
            button.setCursor(Qt.PointingHandCursor)
            switch_layout.addWidget(button)
        self.advanced_search_toggle.setText("Поиск ▾")
        primary_row.addWidget(self.chat_title_label, 1)
        primary_row.addWidget(mode_switch, 0)
        primary_row.addWidget(self.advanced_search_toggle, 0)
        # no mode choice: the assistant decides how to answer (library, files, chat)
        mode_switch.hide()
        self.advanced_search_toggle.hide()
        self.theme_chips_frame = self._create_theme_chips(mode_frame)
        primary_row.addWidget(self.theme_chips_frame, 0, Qt.AlignVCenter)
        mode_layout.addLayout(primary_row)
        self.chat_bar_frame = mode_frame

        self.advanced_search_panel = QFrame(mode_frame)
        self.advanced_search_panel.setObjectName("advancedSearchPanel")
        self.advanced_search_panel.setVisible(False)
        self.advanced_search_panel.setMaximumHeight(0)
        advanced_layout = QHBoxLayout(self.advanced_search_panel)
        advanced_layout.setContentsMargins(10, 8, 10, 8)
        advanced_layout.setSpacing(10)

        advanced_hint = QLabel("Ручные параметры — для тонкой настройки. Основа работы остаётся автоматической.", self.advanced_search_panel)
        advanced_hint.setObjectName("advancedHint")
        advanced_hint.setWordWrap(True)

        self.query_mode_combo = QComboBox(self.advanced_search_panel)
        self.query_mode_combo.setObjectName("compactCombo")
        for value, label in QUERY_MODE_LABELS.items():
            self.query_mode_combo.addItem(label, value)
        self._set_combo_data(self.query_mode_combo, getattr(self, "query_mode_preference", "auto"))
        self.query_mode_combo.currentIndexChanged.connect(self._on_query_mode_preference_changed)

        self.search_strictness_combo = QComboBox(self.advanced_search_panel)
        self.search_strictness_combo.setObjectName("compactCombo")
        for value, label in STRICTNESS_LABELS.items():
            self.search_strictness_combo.addItem(label, value)
        self._set_combo_data(self.search_strictness_combo, getattr(self, "search_strictness_preference", "balanced"))
        self.search_strictness_combo.currentIndexChanged.connect(self._on_search_strictness_preference_changed)

        self.source_scope_combo = QComboBox(self.advanced_search_panel)
        self.source_scope_combo.setObjectName("compactCombo")
        for value, label in SOURCE_SCOPE_LABELS.items():
            self.source_scope_combo.addItem(label, value)
        self._set_combo_data(self.source_scope_combo, getattr(self, "source_scope_preference", "all"))
        self.source_scope_combo.currentIndexChanged.connect(self._on_source_scope_preference_changed)

        advanced_layout.addWidget(advanced_hint, 1)
        advanced_layout.addWidget(QLabel("Тип:", self.advanced_search_panel), 0)
        advanced_layout.addWidget(self.query_mode_combo, 0)
        advanced_layout.addWidget(QLabel("Строгость:", self.advanced_search_panel), 0)
        advanced_layout.addWidget(self.search_strictness_combo, 0)
        advanced_layout.addWidget(QLabel("Где искать:", self.advanced_search_panel), 0)
        advanced_layout.addWidget(self.source_scope_combo, 0)
        mode_layout.addWidget(self.advanced_search_panel)
        layout.addWidget(mode_frame)

        self.chat_scroll = QScrollArea(page)
        self.chat_scroll.setObjectName("chatScrollArea")
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_container = QWidget(self.chat_scroll)
        self.chat_container.setObjectName("chatContainer")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(18, 18, 18, 18)
        self.chat_layout.setSpacing(12)
        self.chat_layout.addStretch()
        self.chat_scroll.setWidget(self.chat_container)
        layout.addWidget(self.chat_scroll, 1)

        self.input_frame = QFrame(page)
        self.input_frame.setObjectName("inputFrame")
        input_layout = QVBoxLayout(self.input_frame)
        input_layout.setContentsMargins(16, 14, 16, 14)
        input_layout.setSpacing(8)

        entry_row = QHBoxLayout()
        entry_row.setSpacing(10)
        self.input_field = QLineEdit(self.input_frame)
        self.input_field.setPlaceholderText("Введите вопрос к документам или модели...")
        self.input_field.returnPressed.connect(self._on_send)
        self.send_btn = QPushButton("Отправить", self.input_frame)
        self.send_btn.setObjectName("sendButton")
        self.send_btn.clicked.connect(self._on_send)
        self.stop_btn = QPushButton("Остановить", self.input_frame)
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setToolTip("Остановить текущую генерацию ответа")
        self.stop_btn.clicked.connect(self._cancel_streaming)
        self.stop_btn.setVisible(False)
        self.send_anim = QPropertyAnimation(self.send_btn, b"geometry")
        self.send_anim.setDuration(140)
        self.send_anim.setEasingCurve(QEasingCurve.OutQuad)
        entry_row.addWidget(self.input_field, 1)
        entry_row.addWidget(self.send_btn)
        entry_row.addWidget(self.stop_btn)

        input_layout.addLayout(entry_row)
        self._build_attachment_widgets(input_layout, entry_row)
        if hasattr(self, "_build_voice_widgets"):
            self._build_voice_widgets(entry_row)
        layout.addWidget(self.input_frame)

        self.page_stack.addWidget(page)
        self.workspace_pages = {"chat": 0}

    def _create_documents_page(self) -> None:
        page = QWidget(self.page_stack)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = self._build_page_header(
            "Документы",
            "Просмотр файлов корпуса, быстрый переход по содержимому и метаданным.",
            [("Добавить", self._add_documents), ("Папка", self._select_docs_folder), ("Сведения", lambda: self._open_inspector("document"))],
        )
        layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal, page)
        splitter.setChildrenCollapsible(False)

        left = QFrame(splitter)
        left.setObjectName("surfaceFrame")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)
        left_layout.addWidget(self._small_caption("Файлы корпуса"))
        self.documents_list = QListWidget(left)
        self.documents_list.setObjectName("documentsList")
        self.documents_list.itemSelectionChanged.connect(self._on_document_item_selected)
        left_layout.addWidget(self.documents_list, 1)

        right = QFrame(splitter)
        right.setObjectName("surfaceFrame")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(10)
        self.doc_preview_title = QLabel("Документ не выбран", right)
        self.doc_preview_title.setObjectName("surfaceTitle")
        self.doc_preview_meta = QLabel("", right)
        self.doc_preview_meta.setObjectName("surfaceHint")
        self.doc_preview_meta.setWordWrap(True)
        self.doc_preview_text = QTextEdit(right)
        self.doc_preview_text.setObjectName("documentPreview")
        self.doc_preview_text.setReadOnly(True)
        right_layout.addWidget(self.doc_preview_title)
        right_layout.addWidget(self.doc_preview_meta)
        right_layout.addWidget(self.doc_preview_text, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 900])
        layout.addWidget(splitter, 1)

        self.page_stack.addWidget(page)
        self.workspace_pages["documents"] = self.page_stack.count() - 1

    def _create_index_page(self) -> None:
        page = QWidget(self.page_stack)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = self._build_page_header(
            "Индекс и база",
            "Состояние корпуса, чанков, векторного поиска и профиля запуска.",
            [("Обновить индекс", self._update_index), ("Проверка системы", self._run_healthcheck_dialog), ("Система", lambda: self._open_inspector("system"))],
        )
        layout.addWidget(header)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.index_docs_card = self._create_stat_card("Документы", grid, 0, 0)
        self.index_chunks_card = self._create_stat_card("Чанки", grid, 0, 1)
        self.index_folder_card = self._create_stat_card("Папка корпуса", grid, 0, 2)
        self.index_search_card = self._create_stat_card("Поиск", grid, 0, 3)
        wrap = QFrame(page)
        wrap.setObjectName("transparentFrame")
        wrap.setLayout(grid)
        layout.addWidget(wrap)

        self.index_summary_text = QPlainTextEdit(page)
        self.index_summary_text.setObjectName("diagnosticsView")
        self.index_summary_text.setReadOnly(True)
        layout.addWidget(self.index_summary_text, 1)

        self.page_stack.addWidget(page)
        self.workspace_pages["index"] = self.page_stack.count() - 1

    def _create_diagnostics_page(self) -> None:
        page = QWidget(self.page_stack)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = self._build_page_header(
            "Диагностика",
            "Логи, статус модулей, самопроверка и сбор support bundle.",
            [("Проверка системы", self._run_healthcheck_dialog), ("Support bundle", self._create_support_bundle_dialog), ("Панель", lambda: self._open_inspector("system"))],
        )
        layout.addWidget(header)

        self.diagnostics_summary = QLabel("", page)
        self.diagnostics_summary.setObjectName("surfaceHint")
        self.diagnostics_summary.setWordWrap(True)
        layout.addWidget(self.diagnostics_summary)

        self.diagnostics_view = QPlainTextEdit(page)
        self.diagnostics_view.setObjectName("diagnosticsView")
        self.diagnostics_view.setReadOnly(True)
        layout.addWidget(self.diagnostics_view, 1)

        self.page_stack.addWidget(page)
        self.workspace_pages["diagnostics"] = self.page_stack.count() - 1

    def _create_settings_page(self) -> None:
        page = QWidget(self.page_stack)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = self._build_page_header(
            "Настройки",
            "Текущие параметры запуска, тема, профиль железа и рабочая папка программы.",
            [("Сменить тему", self._toggle_theme), ("Папка документов", self._select_docs_folder), ("Папка образцов", self._select_generation_catalog_folder), ("Система", lambda: self._open_inspector("system"))],
        )
        layout.addWidget(header)
        if hasattr(self, "_build_model_settings"):
            layout.addWidget(self._build_model_settings(page))

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.settings_theme_card = self._create_stat_card("Тема", grid, 0, 0)
        self.settings_profile_card = self._create_stat_card("Профиль", grid, 0, 1)
        self.settings_docs_card = self._create_stat_card("Папка документов", grid, 1, 0)
        self.settings_version_card = self._create_stat_card("Версия", grid, 1, 1)
        self.settings_query_card = self._create_stat_card("Режим запроса", grid, 2, 0)
        self.settings_strictness_card = self._create_stat_card("Строгость поиска", grid, 2, 1)
        self.settings_generation_card = self._create_stat_card("Автогенерация", grid, 3, 0)
        self.settings_catalog_card = self._create_stat_card("Каталог образцов", grid, 3, 1)
        wrap = QFrame(page)
        wrap.setObjectName("transparentFrame")
        wrap.setLayout(grid)
        layout.addWidget(wrap)

        generation_controls = QFrame(page)
        generation_controls.setObjectName("surfaceFrame")
        generation_layout = QHBoxLayout(generation_controls)
        generation_layout.setContentsMargins(14, 10, 14, 10)
        self.generation_enabled_checkbox = QCheckBox("Автоматически распознавать запросы на создание", generation_controls)
        self.generation_enabled_checkbox.setChecked(bool(getattr(self, "generation_enabled", True)))
        self.generation_enabled_checkbox.toggled.connect(self._on_generation_settings_changed)
        generation_layout.addWidget(self.generation_enabled_checkbox, 1)
        self.generation_creative_checkbox = QCheckBox("Разрешать явно отмеченные новые упражнения", generation_controls)
        self.generation_creative_checkbox.setChecked(bool(getattr(self, "generation_creative_enabled", True)))
        self.generation_creative_checkbox.toggled.connect(self._on_generation_settings_changed)
        generation_layout.addWidget(self.generation_creative_checkbox, 1)
        rebuild_catalog_btn = QPushButton("Обновить каталог", generation_controls)
        rebuild_catalog_btn.setObjectName("secondaryButton")
        rebuild_catalog_btn.clicked.connect(self._rebuild_generation_catalog)
        generation_layout.addWidget(rebuild_catalog_btn, 0)
        select_catalog_btn = QPushButton("Выбрать папку", generation_controls)
        select_catalog_btn.setObjectName("secondaryButton")
        select_catalog_btn.clicked.connect(self._select_generation_catalog_folder)
        generation_layout.addWidget(select_catalog_btn, 0)
        layout.addWidget(generation_controls)

        self.settings_text = QPlainTextEdit(page)
        self.settings_text.setObjectName("diagnosticsView")
        self.settings_text.setReadOnly(True)
        self.settings_text.setMinimumHeight(220)
        self.settings_text.setMaximumHeight(320)
        layout.addWidget(self.settings_text)
        layout.addStretch(1)

        # the page grew (model settings): it scrolls instead of squeezing rows
        scroll = QScrollArea(self.page_stack)
        scroll.setObjectName("settingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page.setObjectName("settingsPage")
        scroll.setWidget(page)
        self.page_stack.addWidget(scroll)
        self.workspace_pages["settings"] = self.page_stack.count() - 1

    def _create_inspector_panel(self) -> None:
        self.inspector_panel = QFrame(self.main_splitter)
        self.inspector_panel.setObjectName("inspectorPanel")
        self.inspector_panel.setMinimumWidth(0)
        self.inspector_panel.setMaximumWidth(0)
        self.inspector_panel.hide()
        self.inspector_visible = False
        self.inspector_animation = QPropertyAnimation(self.inspector_panel, b"maximumWidth", self)
        self.inspector_animation.setDuration(180)
        self.inspector_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.inspector_animation.finished.connect(self._on_inspector_animation_finished)

        layout = QVBoxLayout(self.inspector_panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        top = QHBoxLayout()
        self.inspector_title = QLabel("Контекст", self.inspector_panel)
        self.inspector_title.setObjectName("surfaceTitle")
        self.inspector_close_btn = QPushButton("✕", self.inspector_panel)
        self.inspector_close_btn.setObjectName("iconButton")
        self.inspector_close_btn.clicked.connect(lambda: self._toggle_inspector(force=False))
        top.addWidget(self.inspector_title)
        top.addStretch(1)
        top.addWidget(self.inspector_close_btn)
        layout.addLayout(top)

        self.inspector_stack = QStackedWidget(self.inspector_panel)
        layout.addWidget(self.inspector_stack, 1)

        self.chat_inspector_page = QWidget()
        chat_layout = QVBoxLayout(self.chat_inspector_page)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(8)
        self.chat_inspector_summary = QLabel("Пока нет ответа. После запроса здесь появятся дополнительные сведения по источнику, лучшему фрагменту и альтернативным кандидатам.")
        self.chat_inspector_summary.setWordWrap(True)
        self.chat_inspector_summary.setObjectName("surfaceHint")
        self.chat_inspector_source = QTextEdit()
        self.chat_inspector_source.setObjectName("inspectorText")
        self.chat_inspector_source.setReadOnly(True)
        self.chat_inspector_context = QTextEdit()
        self.chat_inspector_context.setObjectName("inspectorText")
        self.chat_inspector_context.setReadOnly(True)
        self.chat_inspector_extra = QTextEdit()
        self.chat_inspector_extra.setObjectName("inspectorText")
        self.chat_inspector_extra.setReadOnly(True)
        chat_layout.addWidget(self._small_caption("Разбор запроса"))
        chat_layout.addWidget(self.chat_inspector_summary)
        chat_layout.addWidget(self._small_caption("Лучший источник и фрагмент"))
        chat_layout.addWidget(self.chat_inspector_source, 1)
        chat_layout.addWidget(self._small_caption("Контекст вокруг совпадения"))
        chat_layout.addWidget(self.chat_inspector_context, 1)
        chat_layout.addWidget(self._small_caption("Альтернативные кандидаты и сигналы"))
        chat_layout.addWidget(self.chat_inspector_extra, 1)
        self.inspector_stack.addWidget(self.chat_inspector_page)

        self.doc_inspector_page = QWidget()
        doc_layout = QVBoxLayout(self.doc_inspector_page)
        doc_layout.setContentsMargins(0, 0, 0, 0)
        doc_layout.setSpacing(8)
        self.doc_inspector_title = QLabel("Документ не выбран")
        self.doc_inspector_title.setObjectName("surfaceTitle")
        self.doc_inspector_meta = QLabel("")
        self.doc_inspector_meta.setWordWrap(True)
        self.doc_inspector_meta.setObjectName("surfaceHint")
        self.doc_inspector_text = QTextEdit()
        self.doc_inspector_text.setObjectName("inspectorText")
        self.doc_inspector_text.setReadOnly(True)
        doc_layout.addWidget(self.doc_inspector_title)
        doc_layout.addWidget(self.doc_inspector_meta)
        doc_layout.addWidget(self.doc_inspector_text, 1)
        self.inspector_stack.addWidget(self.doc_inspector_page)

        self.system_inspector_page = QWidget()
        sys_layout = QVBoxLayout(self.system_inspector_page)
        sys_layout.setContentsMargins(0, 0, 0, 0)
        sys_layout.setSpacing(8)
        self.system_inspector_summary = QLabel("Сведения о профиле системы и режиме запуска.")
        self.system_inspector_summary.setWordWrap(True)
        self.system_inspector_summary.setObjectName("surfaceHint")
        self.system_inspector_text = QTextEdit()
        self.system_inspector_text.setObjectName("inspectorText")
        self.system_inspector_text.setReadOnly(True)
        sys_layout.addWidget(self.system_inspector_summary)
        sys_layout.addWidget(self.system_inspector_text, 1)
        self.inspector_stack.addWidget(self.system_inspector_page)

        self.inspector_pages = {"chat": 0, "document": 1, "system": 2}

    def _create_status_bar(self) -> None:
        self.status = QStatusBar(self)
        self.status.setObjectName("statusBar")
        self.setStatusBar(self.status)
        self.status.showMessage("Готов к работе")

        self.streaming_indicator = QLabel("", self.status)
        self.status.addPermanentWidget(self.streaming_indicator)
        self.status_docs_mode_label = QLabel("📄 ЦЕЛЫЕ" if self.use_full_documents else "🧩 ФРАГМЕНТЫ", self.status)
        self.status.addPermanentWidget(self.status_docs_mode_label)
        self.status_profile_label = QLabel("", self.status)
        self.status.addPermanentWidget(self.status_profile_label)

    def _create_theme_chips(self, parent: QWidget) -> QFrame:
        """Компактный выбор темы без перегруза интерфейса.

        Три кружка в правом верхнем углу держат тему под рукой, но не превращают
        рабочую панель в настройщик. Ручные подробности остаются в разделе
        «Настройки».
        """
        frame = QFrame(parent)
        frame.setObjectName("themeChipsFrame")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QLabel("Тема", frame)
        label.setObjectName("themeChipsLabel")
        layout.addWidget(label)

        self.theme_chip_group = QButtonGroup(frame)
        self.theme_chip_group.setExclusive(True)
        for key in THEME_ORDER:
            btn = QPushButton("", frame)
            btn.setObjectName("themeChipButton")
            btn.setProperty("themeKey", key)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(theme_short_label(key))
            btn.clicked.connect(lambda _checked=False, k=key: self._set_theme(k))
            self.theme_chip_group.addButton(btn)
            layout.addWidget(btn)
        self._sync_theme_chip_state()
        return frame

    def _build_page_header(self, title: str, subtitle: str, buttons: list[tuple[str, callable]]) -> QFrame:
        frame = QFrame(self.page_stack)
        frame.setObjectName("pageHeader")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(6)
        title_label = QLabel(title, frame)
        title_label.setObjectName("pageTitle")
        subtitle_label = QLabel(subtitle, frame)
        subtitle_label.setObjectName("pageSubtitle")
        subtitle_label.setWordWrap(True)
        left.addWidget(title_label)
        left.addWidget(subtitle_label)

        right = QHBoxLayout()
        right.setSpacing(8)
        for text, callback in buttons:
            btn = QPushButton(text, frame)
            btn.setObjectName("secondaryButton")
            btn.clicked.connect(callback)
            right.addWidget(btn)

        layout.addLayout(left, 1)
        layout.addStretch(1)
        layout.addLayout(right)
        return frame

    def _small_caption(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionCaption")
        return label

    def _create_stat_card(self, title: str, grid: QGridLayout, row: int, col: int) -> dict[str, QLabel]:
        frame = QFrame(self.page_stack)
        frame.setObjectName("statCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        title_label = QLabel(title, frame)
        title_label.setObjectName("statTitle")
        value_label = QLabel("—", frame)
        value_label.setObjectName("statValue")
        value_label.setWordWrap(True)  # long folder paths must not widen the page
        value_label.setMinimumWidth(0)
        from PyQt5.QtWidgets import QSizePolicy

        value_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        hint_label = QLabel("", frame)
        hint_label.setObjectName("statHint")
        hint_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(hint_label)
        grid.addWidget(frame, row, col)
        return {"frame": frame, "title": title_label, "value": value_label, "hint": hint_label}

    def _apply_theme(self) -> None:
        self.setStyleSheet(load_styles(self.current_theme))
        if hasattr(self, "theme_btn"):
            self.theme_btn.setText(theme_icon(self.current_theme))
        self._sync_theme_chip_state()
        if hasattr(self, "_rerender_messages"):
            self._rerender_messages()
        if hasattr(self, "_refresh_assistant_icons"):
            self._refresh_assistant_icons()
        if hasattr(self, "_refresh_nav_icons"):
            from core.theme_manager import theme_palette

            self._refresh_nav_icons(theme_palette(self.current_theme))

    def _toggle_theme(self) -> None:
        self._set_theme(next_theme(self.current_theme))

    def _set_theme(self, theme: str) -> None:
        self.current_theme = theme
        self._apply_theme()
        self._save_settings()
        self._refresh_settings_page()
        self.status.showMessage(f"Тема: {get_theme_label(self.current_theme)}")

    def _sync_theme_chip_state(self) -> None:
        group = getattr(self, "theme_chip_group", None)
        if group is None:
            return
        for button in group.buttons():
            button.blockSignals(True)
            button.setChecked(str(button.property("themeKey")) == str(self.current_theme))
            button.blockSignals(False)
            try:
                button.style().unpolish(button)
                button.style().polish(button)
            except Exception:
                pass

    def _fade_in_widget(self, widget: QWidget, duration_ms: int = 150) -> None:
        """Мягко проявляет виджет после смены содержимого.

        Анимация короткая и безопасная: если графический backend/драйвер ведёт
        себя странно, приложение просто покажет виджет без эффекта.
        """
        if widget is None:
            return
        try:
            profile = getattr(self, "system_profile", None)
            low_ram = bool(profile and getattr(profile, "total_ram_gb", 99) and getattr(profile, "total_ram_gb", 99) <= 8.5)
            if low_ram or str(os.environ.get("CHATKODI_DISABLE_UI_ANIMATIONS", "")).lower() in {"1", "true", "yes", "on"}:
                return
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
            anim = QPropertyAnimation(effect, b"opacity", self)
            anim.setDuration(int(duration_ms))
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            if not hasattr(self, "_ui_fade_animations"):
                self._ui_fade_animations = []
            self._ui_fade_animations.append(anim)

            def cleanup() -> None:
                try:
                    widget.setGraphicsEffect(None)
                except Exception:
                    pass
                try:
                    self._ui_fade_animations.remove(anim)
                except Exception:
                    pass

            anim.finished.connect(cleanup)
            anim.start()
        except Exception:
            try:
                widget.setGraphicsEffect(None)
            except Exception:
                pass

    def _switch_workspace(self, page_name: str) -> None:
        if page_name not in self.workspace_pages:
            return
        target_index = self.workspace_pages[page_name]
        already_visible = self.page_stack.currentIndex() == target_index
        self.page_stack.setCurrentIndex(target_index)
        if not already_visible:
            self._fade_in_widget(self.page_stack.currentWidget(), duration_ms=220)
        for name, btn in self.workspace_buttons.items():
            btn.setChecked(name == page_name)
        if page_name == "chat" and self.inspector_visible:
            self._set_inspector_context("chat")
        elif page_name == "documents" and self.inspector_visible:
            self._set_inspector_context("document")
        elif page_name in {"index", "diagnostics", "settings"} and self.inspector_visible:
            self._set_inspector_context("system")

    def _set_combo_data(self, combo: QComboBox, value: str) -> None:
        """Выбирает пункт QComboBox по userData, не завися от текста локализации."""
        for idx in range(combo.count()):
            if str(combo.itemData(idx)) == str(value):
                combo.setCurrentIndex(idx)
                return

    def _sync_search_preferences_to_manager(self) -> None:
        """Передаёт мягкие настройки поиска в ChatManager."""
        if self.chat_manager and hasattr(self.chat_manager, "set_runtime_preferences"):
            self.chat_manager.set_runtime_preferences(
                query_mode=getattr(self, "query_mode_preference", "auto"),
                strictness=getattr(self, "search_strictness_preference", "balanced"),
                source_scope=getattr(self, "source_scope_preference", "all"),
            )

    def _on_query_mode_preference_changed(self) -> None:
        value = self.query_mode_combo.currentData() if hasattr(self, "query_mode_combo") else "auto"
        self.query_mode_preference = normalize_query_mode(value)
        self._sync_search_preferences_to_manager()
        self._save_settings()
        if hasattr(self, "_refresh_mode_status"):
            self._refresh_mode_status()
        self.status.showMessage(f"Режим запроса: {QUERY_MODE_LABELS.get(self.query_mode_preference, 'Авто')}")
        self._refresh_workspace_views()

    def _on_search_strictness_preference_changed(self) -> None:
        value = self.search_strictness_combo.currentData() if hasattr(self, "search_strictness_combo") else "balanced"
        self.search_strictness_preference = normalize_strictness(value)
        self._sync_search_preferences_to_manager()
        self._save_settings()
        if hasattr(self, "_refresh_mode_status"):
            self._refresh_mode_status()
        self.status.showMessage(f"Строгость поиска: {STRICTNESS_LABELS.get(self.search_strictness_preference, 'Сбалансированно')}")
        self._refresh_workspace_views()

    def _on_source_scope_preference_changed(self) -> None:
        value = self.source_scope_combo.currentData() if hasattr(self, "source_scope_combo") else "all"
        self.source_scope_preference = normalize_source_scope(value)
        self._sync_search_preferences_to_manager()
        self._save_settings()
        self.status.showMessage(f"Область поиска: {SOURCE_SCOPE_LABELS.get(self.source_scope_preference, 'Весь корпус')}")
        self._refresh_workspace_views()

    def _toggle_advanced_search_panel(self, checked: bool) -> None:
        """Скрывает ручные настройки, чтобы основной интерфейс оставался спокойным."""
        checked = bool(checked)
        if hasattr(self, "advanced_search_toggle"):
            self.advanced_search_toggle.setText(("▾" if checked else "▸") + " Доп. настройки")
        panel = getattr(self, "advanced_search_panel", None)
        if panel is None:
            return
        if not hasattr(self, "_advanced_search_animations"):
            self._advanced_search_animations = []
        panel.setVisible(True)
        target_height = max(1, panel.sizeHint().height()) if checked else 0
        start_height = panel.maximumHeight()
        if start_height <= 0 and checked:
            start_height = 0
        elif start_height <= 0:
            start_height = max(1, panel.sizeHint().height())
        anim = QPropertyAnimation(panel, b"maximumHeight", self)
        anim.setDuration(170)
        anim.setEasingCurve(QEasingCurve.OutCubic if checked else QEasingCurve.InCubic)
        anim.setStartValue(start_height)
        anim.setEndValue(target_height)

        def finish() -> None:
            if checked:
                panel.setMaximumHeight(16777215)
            else:
                panel.setVisible(False)
            try:
                self._advanced_search_animations.remove(anim)
            except Exception:
                pass

        anim.finished.connect(finish)
        self._advanced_search_animations.append(anim)
        anim.start()

    def _show_search_reason_panel(self) -> None:
        """Быстрый вход в правую панель «Почему такой ответ?»."""
        self._open_inspector("chat")

    def _toggle_inspector(self, force: bool | None = None) -> None:
        target = (not self.inspector_visible) if force is None else bool(force)
        self.toggle_inspector_action.blockSignals(True)
        self.toggle_inspector_action.setChecked(target)
        self.toggle_inspector_action.blockSignals(False)

        self.inspector_animation.stop()
        if target:
            self.inspector_visible = True
            self.inspector_panel.show()
            self.inspector_panel.setMinimumWidth(340)
            self.inspector_animation.setStartValue(max(0, self.inspector_panel.maximumWidth()))
            self.inspector_animation.setEndValue(360)
        else:
            self.inspector_visible = False
            self.inspector_animation.setStartValue(max(0, self.inspector_panel.maximumWidth() or 360))
            self.inspector_animation.setEndValue(0)
        self.inspector_animation.start()

    def _on_inspector_animation_finished(self) -> None:
        if not self.inspector_visible:
            self.inspector_panel.hide()
            self.inspector_panel.setMinimumWidth(0)
        self._fit_header_texts()
        if hasattr(self, "_refresh_message_widths"):
            self._refresh_message_widths()

    def _open_inspector(self, context: str) -> None:
        self._set_inspector_context(context)
        if not self.inspector_visible:
            self._toggle_inspector(force=True)
        if context == "chat":
            self._update_chat_inspector(answer=getattr(self, "last_answer_text", ""), error=getattr(self, "last_error_text", "") or None)
        elif context == "document":
            self._update_document_inspector(getattr(self, "current_document", None))
        else:
            self._refresh_system_inspector()

    def _set_inspector_context(self, context: str) -> None:
        idx = self.inspector_pages.get(context, 0)
        self.inspector_stack.setCurrentIndex(idx)
        titles = {"chat": "Источник, фрагмент и разбор", "document": "Сведения о документе", "system": "Состояние системы"}
        self.inspector_title.setText(titles.get(context, "Контекст"))

    def _toggle_chat_overview(self, force: bool | None = None) -> None:
        self.chat_overview_visible = (not getattr(self, "chat_overview_visible", False)) if force is None else bool(force)
        self.chat_overview_frame.setVisible(self.chat_overview_visible)
        if hasattr(self, "toggle_overview_action"):
            self.toggle_overview_action.blockSignals(True)
            self.toggle_overview_action.setChecked(self.chat_overview_visible)
            self.toggle_overview_action.blockSignals(False)

    def _preview_text(self, text: str, limit: int = 320) -> str:
        clean = self._clean_plain_text(text or "")
        if len(clean) <= limit:
            return clean
        return clean[:limit].rstrip() + "…"

    def _collect_chat_inspector_context(self, question: str) -> dict:
        cached_question = getattr(self, "_chat_inspector_cache_question", None)
        if cached_question == question:
            return getattr(self, "_chat_inspector_cache_payload", {})

        payload = {
            "question": question,
            "entities": [],
            "dates": [],
            "docs": [],
            "chunks": [],
            "error": None,
            "identifiers": [],
            "object_terms": [],
        }

        try:
            payload["entities"] = [entity.raw for entity in extract_structured_entities(question)]
        except Exception:
            payload["entities"] = []

        try:
            parsed = parse_date_constraints(question) or {}
            raw_dates = parsed.get("dates") or []
            payload["dates"] = [str(d) for d in raw_dates if d]
            if parsed.get("start") and parsed.get("end") and not payload["dates"]:
                payload["dates"] = [f"{parsed.get('start')} → {parsed.get('end')}"]
        except Exception:
            payload["dates"] = []

        if self.chat_manager:
            try:
                intent = self.chat_manager.analyze_question(question)
                payload["identifiers"] = list(getattr(intent, "identifiers", []) or [])
                payload["object_terms"] = list(getattr(intent, "object_terms", []) or [])
            except Exception:
                payload["identifiers"] = []
                payload["object_terms"] = []

        if self.chat_manager:
            try:
                search_payload = self.chat_manager.get_last_search_result(question)
                for score, doc in (search_payload.get("relevant_docs_all") or [])[:3]:
                    payload["docs"].append({
                        "score": float(score),
                        "doc_id": doc.get("doc_id"),
                        "name": doc.get("filename") or doc.get("name") or doc.get("doc_id"),
                        "date_match": bool(doc.get("date_match")),
                        "id_match": bool(doc.get("id_match")),
                    })
                for score, meta in (search_payload.get("results_all") or [])[:3]:
                    payload["chunks"].append({
                        "score": float(score),
                        "doc_id": meta.get("doc_id"),
                        "chunk_id": meta.get("chunk_id"),
                        "date_match": bool(meta.get("date_match")),
                        "anchor_match": bool(meta.get("anchor_match")),
                        "entity_match_count": int(meta.get("entity_match_count", 0) or 0),
                        "object_term_hits": int(meta.get("object_term_hits", 0) or 0),
                        "preview": self._preview_text(meta.get("text", ""), 520),
                    })
            except Exception as exc:
                payload["error"] = str(exc)

        self._chat_inspector_cache_question = question
        self._chat_inspector_cache_payload = payload
        return payload

    def _set_full_documents_mode(self):
        self.use_full_documents = True
        if self.chat_manager:
            self.chat_manager.use_full_documents = True
        self._update_docs_mode_ui()
        logger.info("Document mode switched to full documents")

    def _set_chunks_mode(self):
        self.use_full_documents = False
        if self.chat_manager:
            self.chat_manager.use_full_documents = False
        self._update_docs_mode_ui()
        logger.info("Document mode switched to chunks")

    def _toggle_docs_mode(self):
        self.use_full_documents = not self.use_full_documents
        if self.chat_manager:
            self.chat_manager.use_full_documents = self.use_full_documents
        self._update_docs_mode_ui()
        mode_text = "ЦЕЛЫЕ ДОКУМЕНТЫ" if self.use_full_documents else "ФРАГМЕНТЫ"
        logger.info("Document mode toggled: %s", mode_text)

    def _update_docs_mode_ui(self) -> None:
        self.full_docs_action.setChecked(self.use_full_documents)
        self.chunks_action.setChecked(not self.use_full_documents)
        if hasattr(self, "docs_mode_btn"):
            self.docs_mode_btn.setText("Целые документы" if self.use_full_documents else "Фрагменты")
        self.status_docs_mode_label.setText("📄 ЦЕЛЫЕ" if self.use_full_documents else "🧩 ФРАГМЕНТЫ")
        mode_text = "ЦЕЛЫЕ ДОКУМЕНТЫ" if self.use_full_documents else "ФРАГМЕНТЫ"
        self.subtitle_label.setText(f"Задавайте вопросы по документам и локальной модели • {mode_text}")
        if hasattr(self, "_refresh_mode_status"):
            self._refresh_mode_status()
        self._refresh_workspace_views()

    def _set_mode(self, mode: ChatMode):
        self.current_mode = mode
        if mode in (ChatMode.RAG, ChatMode.CHAT, ChatMode.AGENT) and hasattr(self, "_ensure_llm_ready"):
            try:
                self._ensure_llm_ready(eager=False, reprobe=True)
            except Exception:
                pass
        if self.chat_manager:
            self.chat_manager.set_mode(mode)
        self._update_mode_display()
        self.agent_action.setChecked(mode == ChatMode.AGENT)
        self.rag_action.setChecked(mode == ChatMode.RAG)
        self.chat_action.setChecked(mode == ChatMode.CHAT)
        self.search_action.setChecked(mode == ChatMode.EXTRACT)
        self.mode_agent.setChecked(mode == ChatMode.AGENT)
        self.mode_rag.setChecked(mode == ChatMode.RAG)
        self.mode_chat.setChecked(mode == ChatMode.CHAT)
        self.mode_search.setChecked(mode == ChatMode.EXTRACT)
        self._refresh_workspace_views()

    def _refresh_mode_status(self) -> None:
        """Показывает короткую человеческую сводку режима без перегруза панели."""
        mode_name = self.chat_manager.get_mode_name() if self.chat_manager else "RAG"
        query_mode = getattr(self, "query_mode_preference", "auto")
        strictness = getattr(self, "search_strictness_preference", "balanced")
        if query_mode == "auto" and strictness == "balanced":
            text = "Авто подбирает поиск и строгость"
        else:
            pref = QUERY_MODE_LABELS.get(query_mode, "Авто")
            strict = STRICTNESS_LABELS.get(strictness, "Сбалансированно")
            text = f"{mode_name} • {pref} • {strict}"
        if hasattr(self, "mode_status"):
            self.mode_status.setText(text)

    def _update_mode_display(self) -> None:
        if not self.chat_manager:
            return
        self._refresh_mode_status()
        if self.current_mode == ChatMode.AGENT:
            self.input_field.setPlaceholderText("Спросите, поручите задачу или приложите файл или фото (скрепка, перетаскивание, Ctrl+V)…")
        elif self.current_mode == ChatMode.RAG:
            docs_mode = "целым документам" if self.use_full_documents else "фрагментам документов"
            self.input_field.setPlaceholderText(f"Спросите по {docs_mode}: факт, отчёт, список или статистику…")
        elif self.current_mode == ChatMode.CHAT:
            self.input_field.setPlaceholderText("Свободный чат с локальной моделью…")
        elif self.current_mode == ChatMode.EXTRACT:
            self.input_field.setPlaceholderText("Найти документы или фрагменты без генерации…")

    def _clean_plain_text(self, text: str) -> str:
        if not text:
            return ""
        import re
        clean_text = re.sub(r"<[^>]+>", "", text)
        clean_text = clean_text.replace("**", "").replace("*", "")
        clean_text = re.sub(r"\n\s*\n", "\n", clean_text)
        return clean_text.strip()

    def _show_welcome_message(self) -> None:
        """Welcome card with clickable example questions instead of a wall of text."""
        from gui.components.motion import fade_in

        row = QWidget(self.chat_container)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        card = QFrame(row)
        card.setObjectName("welcomeCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(22, 20, 22, 18)
        card_layout.setSpacing(10)

        title = QLabel("Чем помочь?", card)
        title.setObjectName("welcomeTitle")
        count = 0
        processor = getattr(self, "doc_processor", None)
        try:
            count = len(getattr(processor, "documents", None) or [])
        except TypeError:
            count = 0
        docs_phrase = f"{count} документам" if count else "вашим документам"
        text = QLabel(
            f"Спрашивайте по {docs_phrase} — отвечу с цитатой из источника. "
            "Приложите файл, фото или скан — распознаю, проверю по нормативам, оформлю по образцу, переведу. "
            "Всё работает локально, без интернета.",
            card,
        )
        text.setObjectName("welcomeText")
        text.setWordWrap(True)
        card_layout.addWidget(title)
        card_layout.addWidget(text)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        for index, (label, question, send) in enumerate(self._welcome_examples()):
            chip = QPushButton(label, card)
            chip.setObjectName("exampleChip")
            chip.setCursor(Qt.PointingHandCursor)
            chip.setToolTip("Отправить вопрос" if send else "Вставить начало вопроса в поле ввода")
            chip.clicked.connect(lambda _checked=False, q=question, s=send: self._ask_example(q, s))
            grid.addWidget(chip, index // 2, index % 2)
        card_layout.addLayout(grid)
        hint = QLabel("Скрепка слева от поля — приложить файл; можно перетащить файлы в окно или вставить картинку Ctrl+V. Микрофон — спросить голосом.", card)
        hint.setObjectName("welcomeHint")
        hint.setWordWrap(True)
        card_layout.addWidget(hint)

        row_layout.addWidget(card, 0, Qt.AlignLeft)
        row_layout.addStretch(1)
        row.setProperty("messageBubble", card)
        row.setProperty("messageSender", "ai")
        row.setProperty("messageWelcome", True)
        row.setProperty("messageText", "")
        min_width, max_width = self._message_width_bounds("ai", "", True)
        card.setMinimumWidth(min(min_width, 720))
        card.setMaximumWidth(min(max_width, 860))
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, row)
        fade_in(row, 320)

    def _welcome_examples(self) -> list[tuple[str, str, bool]]:
        examples: list[tuple[str, str, bool]] = [
            ("Что ты умеешь?", "Что ты умеешь?", True),
            ("Приложить файл или фото…", "__attach__", True),
        ]
        latest = None
        processor = getattr(self, "doc_processor", None)
        try:
            from core.fact_lookup import corpus_view

            view = corpus_view(processor) if processor is not None else None
            if view is not None:
                days = [doc.day for doc in view.docs if doc.day]
                latest = max(days) if days else None
        except Exception:
            latest = None
        if latest is not None:
            day_text = latest.strftime("%d.%m.%Y")
            examples.append((f"Перескажи документ за {day_text}", f"Кратко перескажи документ за {day_text}", True))
        examples.append(("Найти в документах …", "Найди в документах ", False))
        if latest is None:
            examples.append(("Кратко перескажи …", "Кратко перескажи документ ", False))
        return examples[:4]

    def _ask_example(self, question: str, send: bool) -> None:
        if question == "__attach__":
            self._choose_attachments()
            return
        self.input_field.setText(question)
        self.input_field.setFocus()
        self.input_field.setCursorPosition(len(question))
        if send:
            self._on_send()

    def _fit_chat_bar(self) -> None:
        """Narrow chat (a source open beside it, a small window): shorter mode names, no theme dots."""
        frame = getattr(self, "chat_bar_frame", None)
        if frame is None:
            return
        narrow = frame.width() < 1020
        tight = frame.width() < 820
        labels = {
            "mode_agent": ("Ассистент", "Ассистент"),
            "mode_rag": ("Документы + ответ", "Документы"),
            "mode_chat": ("Только чат", "Чат"),
            "mode_search": ("Только поиск", "Поиск"),
        }
        for name, (full, short) in labels.items():
            button = getattr(self, name, None)
            if button is not None:
                button.setText(short if narrow else full)
        if hasattr(self, "theme_chips_frame"):
            self.theme_chips_frame.setVisible(not narrow)

    def _apply_responsive_layout(self) -> None:
        """Hide secondary texts and shorten labels on narrow windows."""
        width = self.width()
        self._fit_chat_bar()
        if hasattr(self, "mode_status"):
            self.mode_status.setVisible(width >= 1280)
        if hasattr(self, "subtitle_label"):
            self.subtitle_label.setVisible(width >= 1180)
        if hasattr(self, "mode_rag"):
            compact = width < 1200
            self.mode_rag.setText("Документы" if compact else "Документы + ответ")
            self.mode_chat.setText("Чат" if compact else "Только чат")
            self.mode_search.setText("Поиск" if compact else "Только поиск")

    def _elide_middle(self, text: str, limit: int) -> str:
        if not text or len(text) <= limit:
            return text
        head = max(8, int(limit * 0.45))
        tail = max(8, limit - head - 1)
        return text[:head].rstrip() + "…" + text[-tail:].lstrip()

    def _fit_header_texts(self) -> None:
        width = self.width() if self.width() > 0 else 1440
        scale = max(0.55, min(1.0, width / 1440))
        max_info = int((74 if self.inspector_visible else 96) * scale)
        max_folder = int((86 if self.inspector_visible else 110) * scale)
        info = getattr(self, "_header_info_full_text", self.info_label.text())
        folder = getattr(self, "_header_folder_full_text", self.folder_label.text())
        self.info_label.setText(self._elide_text(info, max_info))
        # The end of a path (the folder name) matters most: elide the middle.
        self.folder_label.setText(self._elide_middle(folder, max_folder))
        self.info_label.setToolTip(info)
        self.folder_label.setToolTip(folder)

    def _elide_text(self, text: str, limit: int) -> str:
        if not text or len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def _show_about(self) -> None:
        about = (
            "ChatKodi 2.0.0\n\n"
            "Локальная поисково-диалоговая система.\n\n"
            "Основные функции:\n"
            "• RAG-поиск по документам с локальной LLM\n"
            "• Чат-режим без документов\n"
            "• Поиск по документам без генерации\n"
            "• Диагностика, профили железа и support bundle\n"
            "• Лаунчер, manuals, настройки строгости поиска и объяснимая панель ответа\n"
            "• Универсальное сопоставление сущностей вида «буквы + номер»\n"
        )
        QMessageBox.about(self, "О программе", about)

    def _refresh_workspace_views(self) -> None:
        self._refresh_sidebar_summary()
        self._fit_header_texts()
        self._refresh_overview_cards()
        self._refresh_documents_views()
        self._refresh_index_page()
        self._refresh_diagnostics_page()
        self._refresh_settings_page()
        self._refresh_system_inspector()

    def _refresh_sidebar_summary(self) -> None:
        docs_count = len(self.doc_processor.documents) if self.doc_processor else 0
        profile_name = getattr(self.system_profile, "profile_name", "unknown")
        llm_state = self._get_llm_state_label()
        self.sidebar_profile_value.setText(f"Профиль: {profile_name}")
        self.sidebar_docs_value.setText(f"Документы: {docs_count}")
        self.sidebar_llm_value.setText(f"LLM: {llm_state}")

    def _refresh_overview_cards(self) -> None:
        docs_count = len(self.doc_processor.documents) if self.doc_processor else 0
        profile_name = getattr(self.system_profile, "profile_name", "unknown")
        mode_name = self.chat_manager.get_mode_name() if self.chat_manager else "RAG"
        llm_state = self._get_llm_state_label()
        safe_state = "Активен" if getattr(self.recovery, "safe_mode_active", False) else "Стабильный"
        self.home_profile_card["value"].setText(profile_name)
        self.home_profile_card["hint"].setText(self.system_profile.summary_line)
        self.home_docs_card["value"].setText(str(docs_count))
        self.home_docs_card["hint"].setText(str(resolve_project_path(self.docs_folder)))
        self.home_mode_card["value"].setText(mode_name)
        self.home_mode_card["hint"].setText(f"LLM: {llm_state}")
        self.home_safe_card["value"].setText(safe_state)
        self.home_safe_card["hint"].setText("Safe-mode включится автоматически при проблемах" if safe_state == "Стабильный" else "Система уже работает с ограничениями")
        self.status_profile_label.setText(profile_name)

        info_text = self.doc_processor.get_document_info() if self.doc_processor else "Документы не загружены"
        folder = getattr(self.doc_processor, "docs_folder", resolve_project_path(self.docs_folder))
        self._header_info_full_text = info_text
        self._header_folder_full_text = f"Папка: {folder}"
        self._fit_header_texts()

    def _refresh_documents_views(self) -> None:
        if not hasattr(self, "documents_list"):
            return
        current_id = getattr(self.current_document, "id", None)
        self.documents_list.blockSignals(True)
        self.documents_list.clear()
        docs = []
        if self.doc_processor and self.doc_processor.documents:
            docs = sorted(self.doc_processor.documents, key=lambda d: (d.modified_time, d.name), reverse=True)
        for doc in docs:
            item = QListWidgetItem(doc.filename or doc.name)
            item.setData(Qt.UserRole, doc.id)
            item.setToolTip(f"{doc.name}\n{doc.size_kb:.1f} KB • {doc.size_chars:,} символов")
            self.documents_list.addItem(item)
            if current_id and doc.id == current_id:
                item.setSelected(True)
        self.documents_list.blockSignals(False)
        if docs and (current_id is None or self.documents_list.currentRow() < 0):
            self.documents_list.setCurrentRow(0)
            self.current_document = docs[0]
            self._render_document_preview(docs[0])
        elif not docs:
            self.current_document = None
            self.doc_preview_title.setText("Документы не загружены")
            self.doc_preview_meta.setText("Добавьте файлы в корпус, чтобы открыть их здесь.")
            self.doc_preview_text.setPlainText("")
            self._update_document_inspector(None)

    def _on_document_item_selected(self) -> None:
        if not self.doc_processor or not self.doc_processor.documents:
            return
        item = self.documents_list.currentItem()
        if not item:
            return
        doc_id = item.data(Qt.UserRole)
        doc = next((d for d in self.doc_processor.documents if d.id == doc_id), None)
        if not doc:
            return
        self.current_document = doc
        self._render_document_preview(doc)
        self._update_document_inspector(doc)

    def _render_document_preview(self, doc) -> None:
        self.doc_preview_title.setText(doc.filename or doc.name)
        modified = datetime.fromtimestamp(doc.modified_time).strftime("%d.%m.%Y %H:%M") if getattr(doc, "modified_time", 0) else "—"
        meta = getattr(doc, "metadata", {}) or {}
        family = meta.get("format_family") or "text"
        line_count = meta.get("line_count") or (doc.content.count('\n') + 1 if getattr(doc, "content", None) else 0)
        self.doc_preview_meta.setText(
            f"Путь: {doc.name}\n"
            f"Тип: {family} • Строк: {line_count} • Размер: {doc.size_kb:.1f} KB • "
            f"{doc.size_chars:,} символов • Обновлён: {modified}"
        )
        self.doc_preview_text.setPlainText(doc.content or "")
        self._fade_in_widget(self.doc_preview_text, duration_ms=120)

    def _refresh_index_page(self) -> None:
        docs_count = len(self.doc_processor.documents) if self.doc_processor else 0
        chunk_count = len(getattr(self.doc_processor, "text_chunks", []) or []) if self.doc_processor else 0
        folder = str(getattr(self.doc_processor, "docs_folder", resolve_project_path(self.docs_folder)))
        search_state = "Готов" if self.hybrid_search else "Ограничен"
        self.index_docs_card["value"].setText(str(docs_count))
        self.index_docs_card["hint"].setText("Количество загруженных документов")
        self.index_chunks_card["value"].setText(str(chunk_count))
        self.index_chunks_card["hint"].setText("Чанки, из которых строится поиск")
        self.index_folder_card["value"].setText(folder)
        self.index_folder_card["hint"].setText("Текущий путь корпуса")
        self.index_search_card["value"].setText(search_state)
        self.index_search_card["hint"].setText("Поиск доступен только при наличии эмбеддингов и документов")

        stats = getattr(self.doc_processor, "stats", {}) or {}
        summary = {
            "документы": docs_count,
            "чанки": chunk_count,
            "общий размер MB": round(stats.get("total_size_mb", 0.0), 2),
            "всего символов": stats.get("total_chars", 0),
            "загрузка сек": round(stats.get("load_time_seconds", 0.0), 3),
            "папка": folder,
            "профиль": getattr(self.system_profile, "summary_line", "—"),
        }
        self.index_summary_text.setPlainText(json.dumps(summary, ensure_ascii=False, indent=2))

    def _refresh_diagnostics_page(self) -> None:
        profile = self.system_profile.summary_line
        incidents = len(getattr(self.recovery, "incidents", []))
        diag = [
            f"Профиль: {profile}",
            f"Safe-mode: {'да' if getattr(self.recovery, 'safe_mode_active', False) else 'нет'}",
            f"Зафиксированных инцидентов: {incidents}",
            f"Runtime dir: {runtime_root()}",
        ]
        self.diagnostics_summary.setText(" • ".join(diag))
        tail = self._read_logs_tail()
        self.diagnostics_view.setPlainText(tail or "Логи пока не найдены. После работы программы здесь появится хвост журналов.")

    def _refresh_settings_page(self) -> None:
        cfg = get_config()
        version = cfg.data.get("version", "1.6")
        theme_label = get_theme_label(self.current_theme)
        self.settings_theme_card["value"].setText(theme_label)
        self.settings_theme_card["hint"].setText("Тема интерфейса")
        self.settings_profile_card["value"].setText(getattr(self.system_profile, "profile_name", "unknown"))
        self.settings_profile_card["hint"].setText(self.system_profile.summary_line)
        self.settings_docs_card["value"].setText(str(resolve_project_path(self.docs_folder)))
        self.settings_docs_card["hint"].setText("Корпус документов")
        self.settings_version_card["value"].setText(version)
        self.settings_version_card["hint"].setText("Текущая версия конфигурации")
        self.settings_query_card["value"].setText(QUERY_MODE_LABELS.get(getattr(self, "query_mode_preference", "auto"), "Авто"))
        self.settings_query_card["hint"].setText("Мягкая подсказка маршрутизации запроса")
        self.settings_strictness_card["value"].setText(STRICTNESS_LABELS.get(getattr(self, "search_strictness_preference", "balanced"), "Сбалансированно"))
        self.settings_strictness_card["hint"].setText("Насколько широко собирать кандидаты")
        generation = getattr(getattr(self, "chat_manager", None), "generation_service", None)
        self.settings_generation_card["value"].setText("Включена" if getattr(self, "generation_enabled", True) else "Выключена")
        self.settings_generation_card["hint"].setText("Спорт и юридические документы — автоматически")
        catalog_summary = generation.catalog.summary() if generation and generation.catalog.records else {}
        self.settings_catalog_card["value"].setText(str(catalog_summary.get("sources", "не построен")))
        self.settings_catalog_card["hint"].setText(str(getattr(self, "generation_catalog_root", "—")))

        text = {
            "тема": theme_label,
            "документы": str(resolve_project_path(self.docs_folder)),
            "рекурсивно": bool(self.docs_recursive),
            "режим": self.current_mode.value,
            "режим_запроса": getattr(self, "query_mode_preference", "auto"),
            "строгость_поиска": getattr(self, "search_strictness_preference", "balanced"),
            "область_источников": getattr(self, "source_scope_preference", "all"),
            "генерация": {
                "включена": bool(getattr(self, "generation_enabled", True)),
                "творческие_дополнения": bool(getattr(self, "generation_creative_enabled", True)),
                "каталог": str(getattr(self, "generation_catalog_root", "")),
                "сводка": catalog_summary,
            },
            "профиль": self.system_profile.as_dict() if hasattr(self.system_profile, "as_dict") else self.system_profile.summary_line,
            "resilience": {
                "safe_mode": getattr(self.recovery, "safe_mode_active", False),
                "политика": getattr(getattr(self.recovery, "policy", None), "__dict__", {}),
            },
        }
        self.settings_text.setPlainText(json.dumps(text, ensure_ascii=False, indent=2))

    def _on_generation_settings_changed(self) -> None:
        self.generation_enabled = bool(self.generation_enabled_checkbox.isChecked())
        self.generation_creative_enabled = bool(self.generation_creative_checkbox.isChecked())
        generation = getattr(getattr(self, "chat_manager", None), "generation_service", None)
        if generation is not None:
            generation.enabled = self.generation_enabled
            generation.config["allow_creative_exercises"] = self.generation_creative_enabled
        self._save_settings()
        self._refresh_settings_page()

    def _rebuild_generation_catalog(self) -> None:
        generation = getattr(getattr(self, "chat_manager", None), "generation_service", None)
        if generation is None:
            QMessageBox.warning(self, "Каталог недоступен", "Сервис генерации ещё не запущен.")
            return
        self.status.showMessage("Проверяю внешний корпус…")
        QApplication.processEvents()
        try:
            generation.catalog.scan()
            summary = generation.catalog.summary()
            self._refresh_settings_page()
            self.status.showMessage(
                f"Каталог готов: источников {summary.get('sources', 0)}, шаблонов {summary.get('templates', 0)}"
            )
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка каталога", str(exc))

    def _select_generation_catalog_folder(self) -> None:
        current = Path(str(getattr(self, "generation_catalog_root", "")))
        if not current.is_absolute():
            current = resolve_project_path(current)
        selected = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с методичками и образцами",
            str(current if current.exists() else Path.home()),
            QFileDialog.ShowDirsOnly,
        )
        if not selected:
            return
        self.generation_catalog_root = str(Path(selected).resolve())
        generation = getattr(getattr(self, "chat_manager", None), "generation_service", None)
        if generation is not None:
            generation.set_catalog_root(self.generation_catalog_root)
        self._save_settings()
        self._refresh_settings_page()
        self.status.showMessage("Папка образцов выбрана. Нажмите «Обновить каталог».")

    def _refresh_system_inspector(self) -> None:
        llm_state = self._get_llm_state_label()
        self.system_inspector_summary.setText(
            f"Профиль: {self.system_profile.profile_name} • LLM: {llm_state} • Safe-mode: {'вкл' if getattr(self.recovery, 'safe_mode_active', False) else 'выкл'}"
        )
        payload = {
            "profile": self.system_profile.as_dict() if hasattr(self.system_profile, "as_dict") else self.system_profile.summary_line,
            "runtime_root": str(runtime_root()),
            "docs_folder": str(resolve_project_path(self.docs_folder)),
            "llm_state": llm_state,
            "documents": len(self.doc_processor.documents) if self.doc_processor else 0,
            "chunks": len(getattr(self.doc_processor, "text_chunks", []) or []) if self.doc_processor else 0,
            "safe_mode": getattr(self.recovery, "safe_mode_active", False),
            "incidents": getattr(self.recovery, "incidents", []),
        }
        self.system_inspector_text.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))

    def _update_chat_inspector(self, answer: str | None = None, error: str | None = None) -> None:
        question = self.last_user_message or "—"
        details_ready = self.inspector_visible and self.inspector_stack.currentIndex() == self.inspector_pages.get("chat", 0)
        payload = self._collect_chat_inspector_context(question) if (details_ready and question != "—") else {}

        summary_lines = [f"Запрос: {question}"]
        entities = payload.get("entities") or []
        identifiers = payload.get("identifiers") or []
        object_terms = payload.get("object_terms") or []
        dates = payload.get("dates") or []
        if entities:
            summary_lines.append("Сущности: " + ", ".join(dict.fromkeys(entities[:4])))
        elif object_terms:
            summary_lines.append("Ориентиры: " + ", ".join(dict.fromkeys(object_terms[:4])))
        if identifiers:
            summary_lines.append("Идентификаторы: " + ", ".join(dict.fromkeys(identifiers[:4])))
        if dates:
            summary_lines.append("Даты: " + ", ".join(dates[:3]))
        if not details_ready:
            summary_lines.append("Откройте правую панель, чтобы увидеть источник, контекст и альтернативные кандидаты.")
        self.chat_inspector_summary.setText("\n".join(summary_lines))

        if not details_ready:
            self.chat_inspector_source.setPlainText("Панель скрыта. Откройте её через Вид → Показать правую панель или клавишей F4.")
            self.chat_inspector_context.setPlainText("Здесь будет расширенный контекст вокруг найденного места документа: не дубликат ответа, а дополнительные сведения для проверки.")
            self.chat_inspector_extra.setPlainText("Здесь появятся альтернативные документы и фрагменты, чтобы было видно, из чего система выбирала.")
            return

        if payload.get("error"):
            self.chat_inspector_source.setPlainText(f"Не удалось собрать контекст источника: {payload['error']}")
            self.chat_inspector_context.setPlainText("Контекст не собран из-за ошибки служебного запроса.")
            self.chat_inspector_extra.setPlainText("Откройте раздел «Диагностика» или соберите support bundle.")
            return

        docs = payload.get("docs") or []
        chunks = payload.get("chunks") or []
        best_chunk = chunks[0] if chunks else None
        related_doc = None
        if best_chunk:
            related_doc = next((doc for doc in docs if str(doc.get("doc_id")) == str(best_chunk.get("doc_id"))), None)

        if best_chunk:
            source_lines = []
            if related_doc:
                source_lines.append(f"Документ: {related_doc.get('name')}")
                source_lines.append(f"Итоговый score: {related_doc.get('score', 0.0):.3f}")
                source_lines.append(f"Совпадение по дате: {'да' if related_doc.get('date_match') else 'нет'}")
                source_lines.append(f"Совпадение по якорю: {'да' if related_doc.get('id_match') else 'нет'}")
                source_lines.append("")
            source_lines.append("Лучший найденный фрагмент:")
            source_lines.append(best_chunk.get("preview") or "—")
            self.chat_inspector_source.setPlainText("\n".join(source_lines))

            context_lines = [
                f"Фрагмент относится к документу: {related_doc.get('name') if related_doc else best_chunk.get('doc_id')}",
                f"Сигналы: дата={'да' if best_chunk.get('date_match') else 'нет'}, якорь={'да' if best_chunk.get('anchor_match') else 'нет'}, entity_hits={best_chunk.get('entity_match_count', 0)}, object_hits={best_chunk.get('object_term_hits', 0)}",
                "",
                "Что это даёт:",
                "• видно, на каком сигнале держится выбор",
                "• можно сравнить с альтернативными кандидатами",
                "• легче понять, почему ответ взят именно отсюда",
            ]
            self.chat_inspector_context.setPlainText("\n".join(context_lines))
        else:
            fallback = "Система не смогла собрать лучший фрагмент для правой панели."
            if self.current_document is not None:
                fallback += f"\n\nТекущий документ: {self.current_document.name}"
            self.chat_inspector_source.setPlainText(fallback)
            self.chat_inspector_context.setPlainText("Проверьте, есть ли релевантные документы и не пуст ли индекс.")

        extra_lines = []
        if docs:
            extra_lines.append("Альтернативные документы:")
            for idx, doc in enumerate(docs[:3], start=1):
                extra_lines.append(
                    f"{idx}. {doc.get('name')}  |  score={doc.get('score', 0.0):.3f}  |  дата={'да' if doc.get('date_match') else 'нет'}  |  якорь={'да' if doc.get('id_match') else 'нет'}"
                )
            extra_lines.append("")
        if len(chunks) > 1:
            extra_lines.append("Альтернативные фрагменты:")
            for idx, chunk in enumerate(chunks[1:4], start=2):
                extra_lines.append(f"{idx}. doc={chunk.get('doc_id')}  |  score={chunk.get('score', 0.0):.3f}")
                extra_lines.append(self._preview_text(chunk.get('preview', ''), 220))
                extra_lines.append("")
        answer_text = self._clean_plain_text(answer if answer is not None else (self.last_answer_text or ""))
        if answer_text:
            extra_lines.append("Кратко по ответу:")
            extra_lines.append(self._preview_text(answer_text, 260))
            extra_lines.append("")
        if error:
            extra_lines.append(f"Ошибка: {error}")
            extra_lines.append("")
        extra_lines.append(f"Режим: {self.chat_manager.get_mode_name() if self.chat_manager else self.current_mode.value}")
        extra_lines.append(f"Документный режим: {'целые документы' if self.use_full_documents else 'фрагменты'}")
        extra_lines.append(f"Режим запроса: {QUERY_MODE_LABELS.get(getattr(self, 'query_mode_preference', 'auto'), 'Авто')}")
        extra_lines.append(f"Строгость: {STRICTNESS_LABELS.get(getattr(self, 'search_strictness_preference', 'balanced'), 'Сбалансированно')}")
        if self.chat_manager and hasattr(self.chat_manager, "get_last_search_trace"):
            trace = self.chat_manager.get_last_search_trace() or {}
            if trace:
                extra_lines.append("")
                extra_lines.append("Как система искала:")
                extra_lines.append(f"• тип запроса: {trace.get('question_type', '—')} / даты: {trace.get('date_mode', '—')}")
                extra_lines.append(f"• бюджет: {trace.get('budget_reason', '—')}")
                extra_lines.append(f"• контекст: {trace.get('context_mode', '—')}, символов: {trace.get('context_chars', 0)}")
                extra_lines.append(f"• документы: {trace.get('docs_selected', 0)}, фрагменты: {trace.get('chunks_selected', 0)}, rescue: {trace.get('rescue_chunks', 0)}")
                notes = trace.get('budget_notes') or []
                for note in notes[:3]:
                    extra_lines.append(f"  - {note}")
        extra_lines.append(f"LLM: {self._get_llm_state_label()}")
        extra_lines.append(f"Документов в базе: {len(self.doc_processor.documents) if self.doc_processor else 0}")
        self.chat_inspector_extra.setPlainText("\n".join(extra_lines))

    def _update_document_inspector(self, doc) -> None:
        if doc is None:
            self.doc_inspector_title.setText("Документ не выбран")
            self.doc_inspector_meta.setText("Выберите документ в разделе «Документы»." )
            self.doc_inspector_text.setPlainText("")
            return
        self.doc_inspector_title.setText(doc.filename or doc.name)
        modified = datetime.fromtimestamp(doc.modified_time).strftime("%d.%m.%Y %H:%M") if getattr(doc, "modified_time", 0) else "—"
        self.doc_inspector_meta.setText(
            f"Путь: {doc.name}\nРазмер: {doc.size_kb:.1f} KB • {doc.size_chars:,} символов • Изменён: {modified}"
        )
        excerpt = doc.get_excerpt(2500) if hasattr(doc, "get_excerpt") else (doc.content[:2500] if getattr(doc, "content", None) else "")
        self.doc_inspector_text.setPlainText(excerpt)

    def _get_llm_state_label(self) -> str:
        model = getattr(self, "llm_model", None)
        if model is None:
            if getattr(self, "models_available", {}).get("llm"):
                return "доступна по запросу"
            return "отключена"
        if hasattr(model, "is_loaded"):
            try:
                return "доступна по запросу" if not model.is_loaded() else "активна"
            except Exception:
                pass
        if hasattr(model, "is_available"):
            try:
                return "активна" if model.is_available() else "недоступна"
            except Exception:
                return "неизвестно"
        return "активна"

    def _run_healthcheck_dialog(self) -> None:
        try:
            from tools.healthcheck import run_healthcheck

            report = run_healthcheck()
            text = json.dumps(report, ensure_ascii=False, indent=2)
            self.diagnostics_view.setPlainText(text)
            self.system_inspector_text.setPlainText(text)
            self._open_inspector("system")
            self.status.showMessage("Самопроверка завершена")
        except Exception as exc:
            QMessageBox.critical(self, "Проверка системы", f"Не удалось выполнить проверку: {exc}")

    def _create_support_bundle_dialog(self) -> None:
        try:
            from tools.support_bundle import build_support_bundle

            bundle = build_support_bundle()
            self.status.showMessage(f"Support bundle создан: {bundle.name}")
            QMessageBox.information(self, "Support bundle", f"Диагностический архив создан:\n{bundle}")
        except Exception as exc:
            QMessageBox.critical(self, "Support bundle", f"Не удалось собрать архив: {exc}")

    def _read_logs_tail(self) -> str:
        candidates = [
            resolve_project_path("logs/chatkodi.log"),
            resolve_project_path("logs/telemetry/events.jsonl"),
        ]
        chunks = []
        for path in candidates:
            if not Path(path).exists():
                continue
            try:
                data = Path(path).read_text(encoding="utf-8", errors="ignore")
                chunks.append(f"===== {Path(path).name} =====\n{data[-5000:]}")
            except Exception:
                continue
        return "\n\n".join(chunks)
