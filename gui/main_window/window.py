"""Главное окно ChatKodi, собранное из небольших модулей."""

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
from core.chat_mode import ChatManager, ChatMode
from core.logging_utils import get_logger
from core.paths import resolve_project_path

logger = get_logger(__name__)

from gui.components.loading import LoadingScreen
from gui.components.console import ConsoleOutput
from gui.workers.streaming import StreamingWorker
from gui.workers.reindex import ReindexWorker

from .settings import ChatKodiSettingsMixin
from .bootstrap import ChatKodiBootstrapMixin
from .ui import ChatKodiUiMixin
from .messages import ChatKodiMessagesMixin
from .documents import ChatKodiDocumentsMixin
from .assistant import ChatKodiAssistantMixin
from .voice import ChatKodiVoiceMixin
from .chats import ChatKodiChatsMixin
from .palette import ChatKodiPaletteMixin
from .model_settings import ChatKodiModelSettingsMixin


class ChatKodiApp(
    QMainWindow,
    ChatKodiSettingsMixin,
    ChatKodiBootstrapMixin,
    ChatKodiUiMixin,
    ChatKodiMessagesMixin,
    ChatKodiDocumentsMixin,
    ChatKodiAssistantMixin,
    ChatKodiVoiceMixin,
    ChatKodiChatsMixin,
    ChatKodiPaletteMixin,
    ChatKodiModelSettingsMixin,
):
    response_ready = pyqtSignal(str, str)
    response_error = pyqtSignal(str)
    streaming_token = pyqtSignal(str)
    _suggestions_ready = pyqtSignal(object, object)  # bubble, [next steps] from a worker thread

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self._suggestions_ready.connect(self._show_suggestions)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._on_resize_settled)
        ChatKodiBootstrapMixin.__init__(self, parent)

    # Qt virtual overrides live here: QMainWindow precedes the mixins in the
    # MRO, so an override defined only in a mixin would never be called.
    def eventFilter(self, obj, event):  # noqa: N802 - Qt override
        if self._assistant_event_filter(obj, event):
            return True
        return QMainWindow.eventFilter(self, obj, event)

    def dragEnterEvent(self, event):  # noqa: N802 - Qt override
        ChatKodiDocumentsMixin.dragEnterEvent(self, event)

    def dropEvent(self, event):  # noqa: N802 - Qt override
        ChatKodiDocumentsMixin.dropEvent(self, event)

    def dragLeaveEvent(self, event):  # noqa: N802 - Qt override
        self._hide_drop_overlay()
        QMainWindow.dragLeaveEvent(self, event)

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        # Bubble widths, header elision and compact labels depend on the
        # window width; recompute once the user stops dragging.
        self._resize_timer.start(60)

    def _on_resize_settled(self) -> None:
        if not hasattr(self, "chat_layout"):
            return
        try:
            self._apply_responsive_layout()
            self._fit_header_texts()
            self._refresh_message_widths()
            if hasattr(self, "_refresh_chat_list"):
                self._refresh_chat_list()  # titles are elided to the list width
        except Exception as exc:  # layout helpers must never break the window
            logger.debug("Responsive relayout failed: %s", exc)
