"""Сохранение и загрузка пользовательских настроек главного окна."""

import json
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import (
    QEasingCurve,
    QFileSystemWatcher,
    QPropertyAnimation,
    QRect,
    QThread,
    QTimer,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPalette, QPen, QTextCursor
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.styles import load_styles
from core.atomic_io import atomic_write_json, read_json_with_backup
from core.chat_mode import ChatManager, ChatMode
from core.logging_utils import get_logger
from core.paths import resolve_project_path
from core.portable_settings import (
    SETTINGS_APP_ROOT_KEY,
    current_settings_app_root,
    normalize_docs_folder_setting,
)
from core.search_preferences import normalize_query_mode, normalize_strictness, normalize_source_scope
from core.theme_manager import normalize_theme

logger = get_logger(__name__)

from gui.components.loading import LoadingScreen
from gui.components.console import ConsoleOutput
from gui.workers.streaming import StreamingWorker
from gui.workers.reindex import ReindexWorker


class ChatKodiSettingsMixin:
        def _load_settings(self):
            """Загружает пользовательские runtime-настройки, если файл уже существует."""
            try:
                if not self.settings_path.exists():
                    return
                data = read_json_with_backup(self.settings_path)
                if not isinstance(data, dict):
                    return
                raw_docs_folder = data.get("docs_folder", self.docs_folder)
                normalized_docs_folder = normalize_docs_folder_setting(
                    raw_docs_folder,
                    stored_app_root=data.get(SETTINGS_APP_ROOT_KEY),
                    default=self.docs_folder,
                )
                if normalized_docs_folder != raw_docs_folder:
                    logger.warning(
                        "Ignoring stale portable docs folder setting: saved=%r current=%r",
                        raw_docs_folder,
                        normalized_docs_folder,
                    )
                self.docs_folder = normalized_docs_folder
                self.docs_recursive = bool(data.get("docs_recursive", self.docs_recursive))
                # Тема из лаунчера передаётся через CHATKODI_THEME и должна иметь
                # приоритет над старым файлом настроек главного окна. Иначе
                # пользователь меняет тему в лаунчере, а рабочее окно остаётся в
                # предыдущей теме — визуально это выглядит как рассинхрон трёх окон.
                if "CHATKODI_THEME" not in os.environ:
                    self.current_theme = normalize_theme(data.get("theme", self.current_theme))
                self.query_mode_preference = normalize_query_mode(data.get("query_mode", getattr(self, "query_mode_preference", "auto")))
                self.search_strictness_preference = normalize_strictness(data.get("search_strictness", getattr(self, "search_strictness_preference", "balanced")))
                self.source_scope_preference = normalize_source_scope(data.get("source_scope", getattr(self, "source_scope_preference", "all")))
                self.generation_enabled = bool(data.get("generation_enabled", getattr(self, "generation_enabled", True)))
                self.generation_creative_enabled = bool(data.get("generation_creative_enabled", getattr(self, "generation_creative_enabled", True)))
                self.generation_catalog_root = str(
                    data.get(
                        "generation_catalog_root",
                        getattr(self, "generation_catalog_root", "documents"),
                    )
                )
                if normalized_docs_folder != raw_docs_folder:
                    self._save_settings()
            except Exception as exc:
                logger.warning("Failed to load user settings from %s: %s", self.settings_path, exc)

        def _save_settings(self):
            """Сохраняет текущие runtime-настройки для следующего запуска."""
            try:
                data = {
                    "docs_folder": self.docs_folder,
                    "docs_recursive": self.docs_recursive,
                    "theme": self.current_theme,
                    "query_mode": getattr(self, "query_mode_preference", "auto"),
                    "search_strictness": getattr(self, "search_strictness_preference", "balanced"),
                    "source_scope": getattr(self, "source_scope_preference", "all"),
                    "generation_enabled": bool(getattr(self, "generation_enabled", True)),
                    "generation_creative_enabled": bool(getattr(self, "generation_creative_enabled", True)),
                    "generation_catalog_root": str(
                        getattr(self, "generation_catalog_root", "documents")
                    ),
                    SETTINGS_APP_ROOT_KEY: current_settings_app_root(),
                }
                atomic_write_json(self.settings_path, data)
            except Exception as exc:
                logger.warning("Failed to save user settings to %s: %s", self.settings_path, exc)
