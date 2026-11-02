"""Background threads for the assistant: a chat turn and confirmed commands."""

from __future__ import annotations

import threading
from typing import List

from PyQt5.QtCore import QThread, pyqtSignal

from core.logging_utils import get_logger

logger = get_logger(__name__)


class AgentWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, agent, message: str, files: List[str]):
        super().__init__()
        self.agent = agent
        self.message = message
        self.files = list(files)
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()
        model = getattr(self.agent.manager, "llm_model", None)
        cancel = getattr(model, "cancel_generation", None)
        if cancel:
            cancel()

    def run(self) -> None:
        model = getattr(self.agent.manager, "llm_model", None)
        attach = getattr(model, "attach_generation_cancel_event", None)
        detach = getattr(model, "detach_generation_cancel_event", None)
        if attach:
            attach(self.cancel_event)
        try:
            result = self.agent.handle(self.message, self.files, progress=self.progress.emit, cancel_event=self.cancel_event)
            self.finished_ok.emit(result)
        except Exception as exc:
            logger.exception("Assistant turn failed")
            self.failed.emit(str(exc))
        finally:
            if detach:
                detach(self.cancel_event)


class CommandWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, agent, proposal):
        super().__init__()
        self.agent = agent
        self.proposal = proposal

    def run(self) -> None:
        try:
            self.finished_ok.emit(self.agent.run_commands(self.proposal, progress=self.progress.emit))
        except Exception as exc:
            logger.exception("Command run failed")
            self.failed.emit(str(exc))
