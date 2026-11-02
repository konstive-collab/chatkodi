"""Small QThread wrapper for expensive application startup stages."""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable

from PyQt5.QtCore import QThread


class StartupTask(QThread):
    """Run one callable and retain its result for the GUI-thread callback."""

    def __init__(self, task: Callable[[], Any], parent=None) -> None:
        super().__init__(parent)
        self.task = task
        self.result: Any = None
        self.error = ""
        self.elapsed_seconds = 0.0

    def run(self) -> None:
        started = time.perf_counter()
        try:
            self.result = self.task()
        except Exception:
            self.error = traceback.format_exc()
        finally:
            self.elapsed_seconds = time.perf_counter() - started
