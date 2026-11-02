"""Small live diagram of PC resources for the sidebar: four rings (CPU, RAM,
GPU load, VRAM) and a sparkline of the main load. Values glide smoothly;
nvidia-smi is polled in a background thread so the UI never waits."""

from __future__ import annotations

import subprocess
import threading
from collections import deque
from typing import Dict, Optional

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QWidget

_CREATE_NO_WINDOW = 0x08000000


def _gpu_sample() -> Optional[Dict[str, float]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, timeout=3, creationflags=_CREATE_NO_WINDOW,
        ).stdout.decode("ascii", "replace").strip().splitlines()
        load, used, total = (float(x) for x in out[0].split(","))
        return {"load": load, "used_gb": used / 1024, "total_gb": total / 1024}
    except Exception:
        return None


class ResourceMonitor(QWidget):
    sampled = pyqtSignal(str, str)  # one-line summary, its level: ok | warn | hot
    RINGS = (("cpu", "ЦП"), ("ram", "ОЗУ"), ("gpu", "ГП"), ("vram", "Видео"))

    def __init__(self, parent=None, palette: Optional[Dict[str, str]] = None, family: str = "Segoe UI"):
        super().__init__(parent)
        self.setObjectName("resourceMonitor")
        self.family = family
        self.colors = {}
        self.set_palette(palette or {})
        self.target = {key: 0.0 for key, _ in self.RINGS}
        self.shown = dict(self.target)
        self.details: Dict[str, str] = {}
        self.history = deque([0.0] * 60, maxlen=60)
        self.has_gpu = True
        self._gpu: Optional[Dict[str, float]] = None
        self._gpu_busy = False
        self.setMinimumHeight(118)
        self._sample_timer = QTimer(self)
        self._sample_timer.timeout.connect(self._sample)
        self._sample_timer.start(1500)
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate)
        self._anim_timer.start(40)
        self._sample()

    def set_palette(self, p: Dict[str, str]) -> None:
        self.colors = {
            "track": QColor(p.get("border_soft", "#1f2937")),
            "accent": QColor(p.get("accent", "#22c55e")),
            "warn": QColor("#f59e0b"),
            "hot": QColor("#ef4444"),
            "text": QColor(p.get("title", "#f9fafb")),
            "muted": QColor(p.get("muted", "#94a3b8")),
        }
        self.update()

    # ------------------------------------------------------------ data

    def _poll_gpu(self) -> None:
        self._gpu = _gpu_sample()
        self._gpu_busy = False

    def _sample(self) -> None:
        try:
            import psutil

            self.target["cpu"] = float(psutil.cpu_percent(interval=None))
            mem = psutil.virtual_memory()
            self.target["ram"] = float(mem.percent)
            self.details["cpu"] = f"Процессор: {self.target['cpu']:.0f}%"
            self.details["ram"] = f"Оперативная память: {mem.used / 1024 ** 3:.1f} из {mem.total / 1024 ** 3:.1f} ГБ"
        except Exception:
            pass
        if not self._gpu_busy:
            self._gpu_busy = True
            threading.Thread(target=self._poll_gpu, daemon=True).start()
        gpu = self._gpu
        self.has_gpu = gpu is not None or self.has_gpu and self._gpu_busy
        if gpu:
            self.target["gpu"] = gpu["load"]
            self.target["vram"] = 100.0 * gpu["used_gb"] / max(0.1, gpu["total_gb"])
            self.details["gpu"] = f"Загрузка видеокарты: {gpu['load']:.0f}%"
            self.details["vram"] = f"Видеопамять: {gpu['used_gb']:.1f} из {gpu['total_gb']:.1f} ГБ"
        self.history.append(self.target["gpu"] if gpu else self.target["cpu"])
        main = self.target["gpu"] if gpu else self.target["cpu"]
        summary = f"ЦП {self.target['cpu']:.0f}% · ГП {self.target['gpu']:.0f}%" if gpu else f"ЦП {self.target['cpu']:.0f}% · ОЗУ {self.target['ram']:.0f}%"
        # video memory is normally full of loaded models: only a real overflow is a warning
        peak = max(self.target["cpu"], main, self.target["vram"] if gpu and self.target["vram"] >= 97 else 0)
        self.sampled.emit(summary, "hot" if peak >= 90 else ("warn" if peak >= 75 else "ok"))
        self.setToolTip("\n".join(self.details.get(k, "") for k, _ in self.RINGS if self.details.get(k)))

    def _animate(self) -> None:
        moved = False
        for key, value in self.target.items():
            delta = value - self.shown[key]
            if abs(delta) > 0.2:
                self.shown[key] += delta * 0.18
                moved = True
        if moved:
            self.update()

    # ------------------------------------------------------------ paint

    def _color_for(self, value: float) -> QColor:
        if value >= 90:
            return self.colors["hot"]
        if value >= 75:
            return self.colors["warn"]
        return self.colors["accent"]

    def paintEvent(self, _event):  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        count = len(self.RINGS)
        cell = width / count
        size = min(46.0, cell - 10)
        top = 4.0
        for index, (key, label) in enumerate(self.RINGS):
            cx = cell * index + cell / 2
            rect = QRectF(cx - size / 2, top, size, size)
            available = key not in {"gpu", "vram"} or self.has_gpu
            value = max(0.0, min(100.0, self.shown[key])) if available else 0.0
            pen = QPen(self.colors["track"], 4.2)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawArc(rect, 0, 360 * 16)
            if available and value > 0.5:
                pen.setColor(self._color_for(value))
                painter.setPen(pen)
                painter.drawArc(rect, 90 * 16, int(-value * 3.6 * 16))
            painter.setPen(self.colors["text"])
            painter.setFont(QFont(self.family, 8, QFont.Bold))
            painter.drawText(rect, Qt.AlignCenter, f"{value:.0f}" if available else "—")
            painter.setPen(self.colors["muted"])
            painter.setFont(QFont(self.family, 8))
            painter.drawText(QRectF(cx - cell / 2, top + size + 3, cell, 15), Qt.AlignHCenter | Qt.AlignTop, label)

        # sparkline of the main load (GPU, or CPU without a GPU)
        caption_top = top + size + 22
        painter.setPen(self.colors["muted"])
        painter.setFont(QFont(self.family, 8))
        caption = "нагрузка ГП за 90 с" if self.has_gpu else "нагрузка ЦП за 90 с"
        painter.drawText(QRectF(4, caption_top, width - 8, 14), Qt.AlignLeft | Qt.AlignVCenter, caption)
        chart = QRectF(4, caption_top + 16, width - 8, self.height() - (caption_top + 18))
        if chart.height() < 12:
            return
        values = list(self.history)
        step = chart.width() / max(1, len(values) - 1)
        line = QPainterPath()
        for i, v in enumerate(values):
            point = QPointF(chart.left() + i * step, chart.bottom() - chart.height() * max(0.0, min(100.0, v)) / 100.0)
            line.moveTo(point) if i == 0 else line.lineTo(point)
        area = QPainterPath(line)
        area.lineTo(chart.right(), chart.bottom())
        area.lineTo(chart.left(), chart.bottom())
        area.closeSubpath()
        fill = QColor(self.colors["accent"])
        fill.setAlpha(38)
        painter.fillPath(area, fill)
        pen = QPen(self.colors["accent"], 1.6)
        painter.setPen(pen)
        painter.drawPath(line)
