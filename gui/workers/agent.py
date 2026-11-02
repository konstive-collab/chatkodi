"""Background threads for the assistant: a chat turn and confirmed commands."""

from __future__ import annotations

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

    def run(self) -> None:
        try:
            result = self.agent.handle(self.message, self.files, progress=self.progress.emit)
            self.finished_ok.emit(result)
        except Exception as exc:
            logger.exception("Assistant turn failed")
            self.failed.emit(str(exc))


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
