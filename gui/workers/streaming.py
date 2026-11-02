"""Worker thread for chat generation and verified document answers."""

import threading

from PyQt5.QtCore import QThread, pyqtSignal

from core.chat_mode import ChatMode
from core.logging_utils import get_logger

logger = get_logger(__name__)

class StreamingWorker(QThread):
    """
    Рабочий поток для streaming-генерации ответов.
    """
    token_received = pyqtSignal(str)
    generation_finished = pyqtSignal(str)
    generation_cancelled = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, chat_manager, question: str, mode: ChatMode, use_full_documents: bool = True):
        super().__init__()
        self.chat_manager = chat_manager
        self.question = question
        self.mode = mode
        self.use_full_documents = use_full_documents
        self._cancel_requested = threading.Event()

    def cancel(self) -> None:
        """Request cooperative cancellation without terminating the QThread."""
        self._cancel_requested.set()
        manager = self.chat_manager
        llm = getattr(manager, "llm_model", None) if manager else None
        cancel = getattr(llm, "cancel_generation", None)
        if callable(cancel):
            try:
                cancel()
            except Exception as exc:
                logger.warning("Failed to request LLM cancellation: %s", exc)
    
    def run(self):
        """Запуск генерации в отдельном потоке."""
        llm = getattr(self.chat_manager, "llm_model", None) if self.chat_manager else None
        attach_cancel = getattr(llm, "attach_generation_cancel_event", None)
        detach_cancel = getattr(llm, "detach_generation_cancel_event", None)
        if callable(attach_cancel):
            attach_cancel(self._cancel_requested)
        try:
            if self._cancel_requested.is_set():
                self.generation_cancelled.emit("")
                return
            if self.chat_manager:
                self.chat_manager.set_mode(self.mode)
            # Получаем streaming callback для RAG или CHAT режима
            if self.mode == ChatMode.RAG:
                # Document answers must be validated as a complete set of
                # claims and exact quotes before any text reaches the UI. The
                # manager owns that single pipeline for both GUI and CLI. Do
                # not replay an already completed answer word by word and call
                # it streaming: publish the validated answer atomically.
                answer = self.chat_manager.process_message(self.question, forced_mode=ChatMode.RAG)
                if self._cancel_requested.is_set():
                    self.generation_cancelled.emit("")
                    return
                self.generation_finished.emit(str(answer))
                return
                
            elif self.mode == ChatMode.CHAT:
                streaming_callback = self.chat_manager.llm_model.generate_clean_answer(
                    question=self.question,
                    context="",
                    rag_mode=False,
                    streaming_callback=True
                )
                if self._cancel_requested.is_set():
                    self.cancel()
                    self.generation_cancelled.emit("")
                    return
            else:
                # Для режима поиска streaming не нужен
                answer = self.chat_manager.process_message(self.question)
                self.generation_finished.emit(answer)
                return
            
            # Если получили streaming callback
            if hasattr(streaming_callback, 'get_stream'):
                full_response = ""
                
                for chunk in streaming_callback.get_stream():
                    if self._cancel_requested.is_set():
                        self.generation_cancelled.emit(full_response)
                        return
                    if chunk:
                        self.token_received.emit(chunk)
                        full_response += chunk

                if self._cancel_requested.is_set():
                    self.generation_cancelled.emit(full_response)
                    return
                self.generation_finished.emit(full_response)
            else:
                # Обычный ответ
                answer = str(streaming_callback)
                if self._cancel_requested.is_set():
                    self.generation_cancelled.emit("")
                else:
                    self.generation_finished.emit(answer)
                
        except Exception as e:
            if self._cancel_requested.is_set():
                self.generation_cancelled.emit("")
            else:
                self.error_occurred.emit(str(e))
        finally:
            if callable(detach_cancel):
                detach_cancel(self._cancel_requested)
    
    def _get_rag_context(self) -> str:
        """Получает RAG-контекст с учётом выбранного режима документов."""
        if not self.chat_manager:
            return ""

        try:
            if hasattr(self.chat_manager, "build_rag_context"):
                _, context = self.chat_manager.build_rag_context(
                    self.question,
                    use_full_documents=self.use_full_documents,
                )
                logger.info(
                    "Using %s context in streaming worker: %s chars",
                    "full-documents" if self.use_full_documents else "chunk",
                    len(context or ""),
                )
                return context or ""
        except Exception as exc:
            logger.warning("Failed to build RAG context in streaming worker: %s", exc)

        return ""
