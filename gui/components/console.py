"""Консольный виджет приложения."""

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

class ConsoleOutput(QTextEdit):
    """Консоль для вывода сообщений."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 9))
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #0f172a;
                color: #e5e7eb;
                border: 1px solid #1f2937;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Consolas', monospace;
            }
            QScrollBar:vertical {
                background: #1f2937;
                width: 12px;
                border-radius: 6px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #4b5563;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #6b7280;
            }
            QScrollBar::add-line:vertical, 
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.max_lines = 100  # Максимальное количество строк
        self.lines_count = 0
    
    def log(self, message: str, level: str = "info"):
        """Добавляет сообщение в консоль с цветом в зависимости от уровня."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if level == "info":
            color = "#22c55e"
            prefix = "INFO"
        elif level == "warning":
            color = "#f59e0b"
            prefix = "WARN"
        elif level == "error":
            color = "#ef4444"
            prefix = "ERROR"
        else:
            color = "#6b7280"
            prefix = "DEBUG"
            
        html = f'<span style="color: #9ca3af">[{timestamp}]</span> '
        html += f'<span style="color: {color}">{prefix}:</span> '
        html += f'<span style="color: #e5e7eb">{message}</span><br>'
        
        self.append(html)
        self.lines_count += 1
        
        # Ограничиваем количество строк
        if self.lines_count > self.max_lines:
            self._trim_lines()
        
        # Автопрокрутка вниз
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)
    
    def _trim_lines(self):
        """Удаляет старые строки, если превышен лимит."""
        html = self.toHtml()
        # Разделяем на строки и оставляем только последние max_lines
        lines = html.split('<br>')
        if len(lines) > self.max_lines:
            lines = lines[-self.max_lines:]
            self.setHtml('<br>'.join(lines))
            self.lines_count = len(lines)

