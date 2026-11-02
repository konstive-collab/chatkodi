"""Виджеты загрузки для главного окна ChatKodi."""

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
from core.theme_manager import normalize_theme, theme_label, theme_palette
from core.chat_mode import ChatManager, ChatMode
from core.logging_utils import get_logger
from core.paths import resolve_project_path

logger = get_logger(__name__)

from .console import ConsoleOutput

class LoadingSpinner(QWidget):
    """Анимированный спиннер загрузки."""
    
    def __init__(self, parent=None, theme: str = "classic"):
        super().__init__(parent)
        self.theme = normalize_theme(theme)
        self.angle = 0.0
        self._phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_rotation)
        self.timer.start(16)  # ~60 FPS: smooth instead of the old 20 FPS jumps
        self.setFixedSize(88, 88)

    def update_rotation(self):
        self.angle = (self.angle + 5.0) % 360
        self._phase = (self._phase + 0.035) % 2.0
        self.update()

    def paintEvent(self, event):
        import math

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(10, 10, -10, -10)

        # Soft track ring.
        track = QPen(QColor(self._ring_color()), 6)
        track.setCapStyle(Qt.RoundCap)
        painter.setPen(track)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(rect)

        # "Breathing" arc: its length grows and shrinks while it rotates.
        breath = (math.sin(self._phase * math.pi) + 1.0) / 2.0
        span = 40 + 220 * breath
        arc = QPen(QColor(self._accent_color()), 6)
        arc.setCapStyle(Qt.RoundCap)
        painter.setPen(arc)
        painter.drawArc(rect, int(-self.angle * 16), int(-span * 16))

    def _accent_color(self) -> str:
        return "#16a34a" if self.theme == "light" else "#22c55e"

    def _ring_color(self) -> str:
        return "#cbd5e1" if self.theme == "light" else "#374151"

    def _inner_color(self) -> str:
        if self.theme == "light":
            return "#ffffff"
        if self.theme == "dark":
            return "#101a18"
        return "#1f2937"

class LoadingScreen(QMainWindow):
    """Экран загрузки приложения."""
    
    loading_complete = pyqtSignal()
    loading_error = pyqtSignal(str)
    
    def __init__(self, theme: str = "classic"):
        super().__init__()
        self.theme = normalize_theme(theme)
        self.setWindowTitle("ChatKodi - Загрузка")
        self.setMinimumSize(720, 520)
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(1280, max(720, int(available.width() * 0.90)))
            height = min(820, max(520, int(available.height() * 0.90)))
            self.resize(width, height)
        else:
            self.resize(1100, 720)
        
        # Устанавливаем стиль из единой палитры ChatKodi.
        self._apply_theme()
        
        self._init_ui()
    

    def _palette(self) -> dict[str, str]:
        return theme_palette(self.theme)

    def _fg(self) -> str: return self._palette()["fg"]
    def _muted(self) -> str: return self._palette()["muted"]
    def _panel(self) -> str: return self._palette()["panel"]
    def _border(self) -> str: return self._palette()["border"]
    def _button_bg(self) -> str: return self._palette()["button"]
    def _button_hover(self) -> str: return self._palette()["button_hover"]

    def _apply_theme(self) -> None:
        p = self._palette()
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {p['bg']}; }}
            QLabel {{ color: {p['fg']}; }}
            QProgressBar {{
                border: 1px solid {p['border']};
                border-radius: 4px;
                text-align: center;
                background-color: {p['panel']};
            }}
            QProgressBar::chunk {{
                background-color: {p['accent']};
                border-radius: 3px;
            }}
        """)

    def _init_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(15)
        
        # Заголовок
        title_layout = QVBoxLayout()
        title_layout.setSpacing(8)
        
        self.logo_label = QLabel("ChatKodi", self)
        self.logo_label.setFont(QFont("Segoe UI", 32, QFont.Bold))
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setStyleSheet(f"color: {self._fg()};")
        
        self.subtitle_label = QLabel("Локальный ассистент по документам · работает без интернета", self)
        self.subtitle_label.setFont(QFont("Segoe UI", 12))
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setStyleSheet(f"color: {self._muted()}")
        
        title_layout.addWidget(self.logo_label)
        title_layout.addWidget(self.subtitle_label)
        
        # Спиннер
        spinner_layout = QHBoxLayout()
        spinner_layout.addStretch()
        
        self.spinner = LoadingSpinner(self, theme=self.theme)
        spinner_layout.addWidget(self.spinner)
        spinner_layout.addStretch()
        
        # Статус загрузки
        status_layout = QVBoxLayout()
        status_layout.setSpacing(8)
        
        self.status_label = QLabel("Инициализация приложения...", self)
        self.status_label.setFont(QFont("Segoe UI", 10))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"color: {self._fg()};")
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        
        self._progress_animation = QPropertyAnimation(self.progress_bar, b"value", self)
        self._progress_animation.setDuration(450)
        self._progress_animation.setEasingCurve(QEasingCurve.OutCubic)

        status_layout.addWidget(self.status_label)
        self.progress_bar.setFixedWidth(560)
        status_layout.addWidget(self.progress_bar, 0, Qt.AlignHCenter)

        # Stage checklist: what is done, what is loading now.
        self._stage_labels: list[QLabel] = []
        stages_row = QHBoxLayout()
        stages_row.setSpacing(18)
        stages_row.addStretch()
        for title, _threshold in self.STAGES:
            label = QLabel(title, self)
            label.setFont(QFont("Segoe UI", 9))
            self._stage_labels.append(label)
            stages_row.addWidget(label)
        stages_row.addStretch()
        status_layout.addSpacing(6)
        status_layout.addLayout(stages_row)
        self._render_stages(0)

        # Журнал загрузки: нужен поддержке, а не пользователю — скрыт по умолчанию.
        self.details_btn = QPushButton("Подробнее ▾", self)
        self.details_btn.setCursor(Qt.PointingHandCursor)
        self.details_btn.setFlat(True)
        self.details_btn.setStyleSheet(
            f"QPushButton {{ color: {self._muted()}; background: transparent; border: none; padding: 4px 8px; }}"
            f"QPushButton:hover {{ color: {self._fg()}; }}"
        )
        self.details_btn.clicked.connect(self._toggle_details)
        details_row = QHBoxLayout()
        details_row.addStretch()
        details_row.addWidget(self.details_btn)
        details_row.addStretch()
        status_layout.addLayout(details_row)

        # Консоль вывода
        console_frame = QFrame(self)
        self.console_frame = console_frame
        console_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {self._panel()};
                border: 1px solid {self._border()};
                border-radius: 12px;
            }}
        """)
        
        console_layout = QVBoxLayout(console_frame)
        console_layout.setContentsMargins(12, 12, 12, 12)
        console_layout.setSpacing(6)
        
        console_label = QLabel("Консоль загрузки:", console_frame)
        console_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        console_label.setStyleSheet(f"color: {self._muted()};")
        
        self.console = ConsoleOutput(console_frame)
        self.console.setMinimumHeight(180)
        self.console.setStyleSheet(f"""
            QTextEdit {{
                background-color: {self._palette()['sunken']};
                color: {self._fg()};
                border: 1px solid {self._border()};
                border-radius: 8px;
                padding: 8px;
                font-family: 'Consolas', monospace;
            }}
            QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
            QScrollBar::handle:vertical {{ background: {self._border()}; border-radius: 6px; min-height: 20px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)
        
        console_layout.addWidget(console_label)
        console_layout.addWidget(self.console)
        
        # Кнопка отмены
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("Отмена", self)
        self.cancel_btn.setFixedSize(100, 36)
        self.cancel_btn.setFont(QFont("Segoe UI", 9))
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._button_bg()};
                color: {self._fg()};
                border: 1px solid {self._border()};
                border-radius: 10px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {self._button_hover()};
            }}
        """)
        
        button_layout.addWidget(self.cancel_btn)
        button_layout.addStretch()
        
        # Собираем все вместе
        main_layout.addStretch(1)
        main_layout.addLayout(title_layout)
        main_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))
        main_layout.addLayout(spinner_layout)
        main_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))
        main_layout.addLayout(status_layout)
        main_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))
        main_layout.addWidget(console_frame, 1)
        main_layout.addStretch(1)
        main_layout.addLayout(button_layout)
        console_frame.setVisible(False)

    # (title, progress value at which the stage is complete)
    STAGES = (
        ("Проверка моделей", 15),
        ("Модель поиска", 30),
        ("Языковая модель", 50),
        ("Документы", 65),
        ("Индекс", 80),
    )

    def _render_stages(self, progress: int) -> None:
        palette = self._palette()
        active_found = False
        for label, (title, threshold) in zip(self._stage_labels, self.STAGES):
            if progress >= threshold:
                label.setText(f"✓  {title}")
                label.setStyleSheet(f"color: {palette['accent']};")
            elif not active_found:
                active_found = True
                label.setText(f"●  {title}")
                label.setStyleSheet(f"color: {palette['title']}; font-weight: 600;")
            else:
                label.setText(f"○  {title}")
                label.setStyleSheet(f"color: {palette['muted']};")

    def _toggle_details(self) -> None:
        visible = not self.console_frame.isVisible()
        self.console_frame.setVisible(visible)
        self.details_btn.setText("Скрыть журнал ▴" if visible else "Подробнее ▾")

    def _on_cancel(self):
        """Обработчик отмены загрузки."""
        self.loading_error.emit("Загрузка отменена пользователем")
        self.close()
    
    def update_status(self, message: str, progress: int = None):
        """Обновляет статус загрузки."""
        self.status_label.setText(message)
        self.console.log(message, "info")

        if progress is not None:
            self._progress_animation.stop()
            self._progress_animation.setStartValue(self.progress_bar.value())
            self._progress_animation.setEndValue(int(progress))
            self._progress_animation.start()
            self._render_stages(int(progress))

    def show_error(self, error_message: str):
        """Показывает ошибку."""
        self.status_label.setText("Ошибка загрузки!")
        self.status_label.setStyleSheet("color: #ef4444")
        self.console.log(error_message, "error")
        # The log is exactly what the user needs to see now.
        self.console_frame.setVisible(True)
        self.details_btn.setText("Скрыть журнал ▴")
        
        self.cancel_btn.setText("Закрыть")
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self.close)
    
    def complete_loading(self):
        """Завершает загрузку."""
        self.progress_bar.setValue(100)
        self.status_label.setText("Загрузка завершена!")
        self.console.log("✅ Все системы запущены", "info")
        QTimer.singleShot(1000, self.loading_complete.emit)
