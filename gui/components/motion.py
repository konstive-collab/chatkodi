"""Small animation helpers for the chat: fade-in, smooth scrolling, typing dots.

Qt style sheets have no transitions, so motion is done with property
animations. Every helper degrades to an instant change if anything fails:
an animation must never break the chat.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, Optional

from PyQt5.QtCore import QEasingCurve, QObject, QPropertyAnimation, QTimer, Qt
from PyQt5.QtWidgets import QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QScrollArea, QWidget


def animations_enabled() -> bool:
    """Same switch as the page fade: CHATKODI_DISABLE_UI_ANIMATIONS=1 turns motion off."""
    return str(os.environ.get("CHATKODI_DISABLE_UI_ANIMATIONS", "")).strip().lower() not in {"1", "true", "yes", "on"}


def fade_in(widget: QWidget, duration_ms: int = 220, on_finished: Optional[Callable[[], None]] = None) -> None:
    """Fade a freshly shown widget in, then drop the effect (keeps text crisp)."""
    if not animations_enabled():
        if on_finished:
            on_finished()
        return
    try:
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.0)
        widget.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", widget)
        animation.setDuration(duration_ms)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)

        def finish() -> None:
            try:
                widget.setGraphicsEffect(None)
            except RuntimeError:
                pass
            if on_finished:
                on_finished()

        animation.finished.connect(finish)
        animation.start(QPropertyAnimation.DeleteWhenStopped)
    except Exception:
        widget.setGraphicsEffect(None)
        if on_finished:
            on_finished()


def pulse(widget: QWidget, low: float = 0.55, duration_ms: int = 320) -> None:
    """Short dim-and-back feedback for a pressed button (no geometry jumps)."""
    if not animations_enabled():
        return
    try:
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", widget)
        animation.setDuration(duration_ms)
        animation.setKeyValueAt(0.0, 1.0)
        animation.setKeyValueAt(0.35, low)
        animation.setKeyValueAt(1.0, 1.0)
        animation.setEasingCurve(QEasingCurve.InOutSine)
        animation.finished.connect(lambda: widget.setGraphicsEffect(None))
        animation.start(QPropertyAnimation.DeleteWhenStopped)
    except Exception:
        widget.setGraphicsEffect(None)


class SmoothScroller(QObject):
    """Keeps a chat scrolled to the newest message with an eased animation.

    The old code jumped to ``maximum()`` 50 ms after adding a bubble, before the
    bubble had its final height, so the last answer was cut off. Here every
    range change re-targets the animation while the view "sticks" to the
    bottom; scrolling up by hand releases it.
    """

    def __init__(self, area: QScrollArea, duration_ms: int = 280) -> None:
        super().__init__(area)
        self._bar = area.verticalScrollBar()
        self._animation = QPropertyAnimation(self._bar, b"value", self)
        self._animation.setDuration(duration_ms)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self.stick = True
        self._bar.rangeChanged.connect(self._on_range_changed)
        self._bar.valueChanged.connect(self._on_value_changed)

    def scroll_to_bottom(self, animated: bool = True) -> None:
        self.stick = True
        target = self._bar.maximum()
        if not animated or not animations_enabled() or abs(target - self._bar.value()) < 4:
            self._animation.stop()
            self._bar.setValue(target)
            return
        self._animation.stop()
        self._animation.setStartValue(self._bar.value())
        self._animation.setEndValue(target)
        self._animation.start()

    def _on_range_changed(self, _minimum: int, _maximum: int) -> None:
        if self.stick:
            # Defer one event-loop turn: the layout may still be growing.
            QTimer.singleShot(0, lambda: self.scroll_to_bottom(animated=True))

    def _on_value_changed(self, value: int) -> None:
        if self._animation.state() == QPropertyAnimation.Running:
            return
        self.stick = value >= self._bar.maximum() - 48


class TypingIndicator(QFrame):
    """AI bubble with animated dots and a phase text while the answer is built."""

    PHASES = (
        (0.0, "Ищу в документах"),
        (2.5, "Сверяю цифры и даты"),
        (7.0, "Формулирую ответ"),
        (20.0, "Локальная модель ещё думает"),
    )

    def __init__(self, parent: QWidget, palette: Dict[str, str]) -> None:
        super().__init__(parent)
        self.setProperty("aiBubble", "true")
        self.setObjectName("typingBubble")
        self._palette = palette
        self._step = 0
        self._started = time.monotonic()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 18, 12)
        layout.setSpacing(10)
        self._dots = QLabel(self)
        self._dots.setTextFormat(Qt.RichText)
        self._text = QLabel(self)
        self._text.setObjectName("typingText")
        layout.addWidget(self._dots)
        layout.addWidget(self._text)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(320)
        self._tick()

    def _tick(self) -> None:
        accent = self._palette.get("accent", "#22c55e")
        dim = self._palette.get("border", "#374151")
        self._step = (self._step + 1) % 3
        dots = "".join(
            f'<span style="color:{accent if index == self._step else dim}; font-size:18px;">●</span>&nbsp;'
            for index in range(3)
        )
        self._dots.setText(dots)
        elapsed = time.monotonic() - self._started
        phase = self.PHASES[0][1]
        for threshold, label in self.PHASES:
            if elapsed >= threshold:
                phase = label
        seconds = f" · {int(elapsed)} с" if elapsed >= 3 else ""
        self._text.setText(f"{phase}…{seconds}")

    def stop(self) -> None:
        self._timer.stop()


class AnimatedHeight(QObject):
    """Animate ``maximumHeight`` of a collapsible panel (expand/collapse)."""

    def __init__(self, widget: QWidget, duration_ms: int = 220) -> None:
        super().__init__(widget)
        self._widget = widget
        self._animation = QPropertyAnimation(widget, b"maximumHeight", self)
        self._animation.setDuration(duration_ms)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)

    def expand(self) -> None:
        self._widget.setVisible(True)
        target = max(self._widget.sizeHint().height(), 1)
        self._animation.stop()
        self._animation.setStartValue(0)
        self._animation.setEndValue(target)
        try:
            self._animation.finished.disconnect()
        except TypeError:
            pass
        self._animation.finished.connect(lambda: self._widget.setMaximumHeight(16777215))
        self._animation.start()

    def collapse(self) -> None:
        self._animation.stop()
        self._animation.setStartValue(self._widget.height())
        self._animation.setEndValue(0)
        try:
            self._animation.finished.disconnect()
        except TypeError:
            pass
        self._animation.finished.connect(lambda: self._widget.setVisible(False))
        self._animation.start()
