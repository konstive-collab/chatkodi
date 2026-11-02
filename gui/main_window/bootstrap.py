"""Запуск приложения и инициализация моделей, документов и поиска."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import QFileSystemWatcher, QTimer
from PyQt5.QtWidgets import QApplication, QWidget

from core.chat_mode import ChatMode
from core.chat_history import ChatHistoryStore
from core.config import get_config
from core.logging_utils import get_logger
from core.paths import resolve_project_path
from core.resilience import RecoveryManager
from core.system_profile import detect_system_profile
from core.theme_manager import env_theme, normalize_theme
from gui.components.loading import LoadingScreen
from gui.workers.startup import StartupTask

logger = get_logger(__name__)


def _llm_failure_text(default: str) -> str:
    """Prefer the concrete load failure reason (pagefile, crash, damaged files)."""
    try:
        from core.llm.runtime import describe_llm_status

        detail = describe_llm_status()
    except Exception:
        detail = ""
    return detail or default



class ChatKodiBootstrapMixin:
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self.model_loader = None
        self.doc_processor = None
        self.chat_manager = None
        self.llm_model = None
        self.hybrid_search = None
        self.current_theme = env_theme("classic")
        self.models_available = {"embeddings": False, "llm": False}
        self._llm_loader = None
        self.last_user_message = ""
        self.last_answer_text = ""
        self.last_error_text = ""
        self.current_document = None

        self.config = get_config()
        history_cfg = dict(self.config.data.get("history") or {})
        self.chat_history_enabled = bool(history_cfg.get("enabled", True))
        self.chat_history_store = ChatHistoryStore(
            history_cfg.get("path") or "chat_history.jsonl",
            max_entries=int(history_cfg.get("max_entries", 200)),
        )
        if hasattr(self, "_init_chat_sessions"):
            # several chats: the single 2.1 history becomes the first chat
            self._init_chat_sessions()
        if hasattr(self, "_init_model_settings"):
            self._init_model_settings()
        generation_cfg = dict(self.config.data.get("generation") or {})
        self.generation_enabled = bool(generation_cfg.get("enabled", True))
        self.generation_creative_enabled = bool(generation_cfg.get("allow_creative_exercises", True))
        self.generation_catalog_root = str(generation_cfg.get("catalog_root") or "documents")
        self.system_profile = detect_system_profile(self.config.data)
        self.recovery = RecoveryManager.from_config_data(self.config.data, profile=self.system_profile)
        self.recovery.startup_event()

        self.settings_path = resolve_project_path("chatkodi_settings.json")
        self.docs_folder = "docs"
        self.docs_recursive = True
        self.query_mode_preference = "auto"
        self.search_strictness_preference = "balanced"
        self.source_scope_preference = "all"
        self._load_settings()

        self.fs_watcher = QFileSystemWatcher(self)
        self.fs_watcher.directoryChanged.connect(self._on_docs_folder_changed)
        self.fs_watcher.fileChanged.connect(self._on_docs_folder_changed)
        self.reindex_debounce = QTimer(self)
        self.reindex_debounce.setSingleShot(True)
        self.reindex_debounce.timeout.connect(lambda: self._update_index(silent=True))

        self.current_mode = ChatMode.AGENT
        self.use_full_documents = False  # фрагменты по умолчанию

        self.is_streaming = False
        self.current_streaming_worker = None
        self.current_streaming_bubble = None
        self.current_streaming_body = None
        self.streaming_buffer = ""
        self.streaming_timer = QTimer(self)
        self.streaming_timer.timeout.connect(self._flush_streaming_buffer)

        self.response_ready.connect(self._on_response_ready)
        self.response_error.connect(self._on_response_error)
        self.streaming_token.connect(self._on_streaming_token)

        self.loading_screen = LoadingScreen(theme=self.current_theme)
        self.loading_screen.loading_complete.connect(self._on_loading_complete)
        self.loading_screen.loading_error.connect(self._on_loading_error)
        self.loading_screen.show()

        self._startup_worker: StartupTask | None = None
        self._startup_cancel_requested = False
        self._startup_stage_label = ""
        self._startup_stage_started = 0.0
        self._startup_progress = 0
        self._startup_elapsed_timer = QTimer(self)
        self._startup_elapsed_timer.setInterval(1000)
        self._startup_elapsed_timer.timeout.connect(self._update_startup_elapsed)

        QTimer.singleShot(100, self._start_loading)

    def _start_loading(self) -> None:
        try:
            self.loading_screen.console.log(f"Профиль системы: {self.system_profile.summary_line}", "debug")

            self.loading_screen.update_status("Проверяю доступность моделей...", 10)
            self.recovery.run_stage("check_models", self._check_models, required=False)
            self._start_embedding_stage()
        except Exception as exc:
            self.loading_screen.show_error(str(exc))

    def _run_startup_task(
        self,
        *,
        stage_name: str,
        status: str,
        progress: int,
        task: Callable[[], object],
        on_success: Callable[[object, float], None],
        on_failure: Callable[[str, float], None],
    ) -> None:
        if self._startup_cancel_requested:
            QApplication.quit()
            return
        if self._startup_worker is not None and self._startup_worker.isRunning():
            raise RuntimeError("Предыдущий этап запуска еще выполняется")

        self._startup_stage_label = status
        self._startup_stage_started = time.perf_counter()
        self._startup_progress = progress
        self.loading_screen.update_status(status, progress)
        self._startup_elapsed_timer.start()

        worker = StartupTask(task, self)
        self._startup_worker = worker
        worker.finished.connect(
            lambda current=worker: self._finish_startup_task(
                current,
                stage_name=stage_name,
                on_success=on_success,
                on_failure=on_failure,
            )
        )
        worker.start()

    def _finish_startup_task(
        self,
        worker: StartupTask,
        *,
        stage_name: str,
        on_success: Callable[[object, float], None],
        on_failure: Callable[[str, float], None],
    ) -> None:
        self._startup_elapsed_timer.stop()
        if self._startup_worker is worker:
            self._startup_worker = None
        elapsed = float(worker.elapsed_seconds or 0.0)
        error = str(worker.error or "").strip()
        telemetry = getattr(self.recovery, "telemetry", None)
        if telemetry:
            try:
                telemetry.event(
                    "startup_stage_failed" if error else "startup_stage_ok",
                    {
                        "stage": stage_name,
                        "elapsed_ms": round(elapsed * 1000.0, 2),
                        **({"error": error[-2000:]} if error else {}),
                    },
                )
            except Exception:
                pass

        worker.deleteLater()
        if self._startup_cancel_requested:
            QApplication.quit()
            return
        try:
            if error:
                on_failure(error, elapsed)
            else:
                on_success(worker.result, elapsed)
        except Exception as exc:
            logger.exception("Startup continuation failed after %s", stage_name)
            self.loading_screen.show_error(str(exc))

    def _update_startup_elapsed(self) -> None:
        if not self._startup_stage_label or not self._startup_stage_started:
            return
        elapsed = int(time.perf_counter() - self._startup_stage_started)
        self.loading_screen.status_label.setText(
            f"{self._startup_stage_label} Прошло: {elapsed} сек."
        )
        self.loading_screen.progress_bar.setValue(self._startup_progress)

    def _start_embedding_stage(self) -> None:
        if not self.models_available.get("embeddings"):
            self.model_loader = None
            self.loading_screen.console.log(
                "Embedding-модель недоступна: режим поиска будет отключен",
                "warning",
            )
            self._start_llm_stage()
            return

        def build_embedding_loader():
            from core.models import AIModelLoader

            loader = AIModelLoader()
            if not loader.load_models():
                raise RuntimeError("Не удалось загрузить embedding-модель")
            return loader

        self._run_startup_task(
            stage_name="load_embedding_model",
            status="Загружаю модель эмбеддингов...",
            progress=25,
            task=build_embedding_loader,
            on_success=self._on_embedding_loaded,
            on_failure=self._on_embedding_failed,
        )

    def _on_embedding_loaded(self, model_loader, elapsed: float) -> None:
        self.model_loader = model_loader
        self.loading_screen.update_status(
            f"Embedding-модель загружена за {elapsed:.1f} сек.",
            32,
        )
        self._start_llm_stage()

    def _on_embedding_failed(self, error: str, elapsed: float) -> None:
        self.model_loader = None
        logger.warning("Embedding startup failed after %.1fs: %s", elapsed, error)
        self.recovery.activate_safe_mode(
            "embedding_load_failed",
            elapsed_seconds=round(elapsed, 2),
            error=error[-1000:],
        )
        self.loading_screen.console.log(
            "Embedding-модель не загрузилась. Приложение продолжит запуск без поиска.",
            "warning",
        )
        self._start_llm_stage()

    def _start_llm_stage(self) -> None:
        if not self.models_available.get("llm") and not self._probe_llm_direct():
            self.llm_model = None
            self.loading_screen.update_status(
                "LLM не найдена. Документы останутся доступны без генерации.",
                50,
            )
            self._start_documents_stage()
            return

        startup_mode = self.recovery.choose_llm_startup_mode()
        if startup_mode != "eager":
            self.loading_screen.update_status(
                "Регистрирую локальную LLM без тяжелой загрузки...",
                40,
            )
            self.recovery.run_stage("load_llm_model", self._load_llm_model, required=False)
            self._start_documents_stage()
            return

        loader = self._make_llm_loader()
        self._run_startup_task(
            stage_name="load_llm_model",
            status="Загружаю локальную LLM. Это может занять несколько минут...",
            progress=42,
            task=loader,
            on_success=self._on_llm_loaded,
            on_failure=self._on_llm_failed,
        )

    def _on_llm_loaded(self, llm_model, elapsed: float) -> None:
        self.llm_model = llm_model
        available = bool(llm_model)
        if available and hasattr(llm_model, "is_available"):
            available = bool(llm_model.is_available())
        if not available:
            self._on_llm_failed("LLM loader returned an unavailable model", elapsed)
            return

        info = llm_model.get_model_info() if hasattr(llm_model, "get_model_info") else {}
        model_name = str(info.get("name") or "локальная модель")
        self._sync_chat_manager_llm()
        self.loading_screen.update_status(
            f"LLM загружена за {elapsed:.1f} сек.: {model_name}",
            52,
        )
        try:
            from core.llm.runtime import get_placement_warning

            placement = get_placement_warning()
            if placement:
                self.loading_screen.console.log(placement, "warning")
        except Exception:
            pass
        self._start_documents_stage()

    def _on_llm_failed(self, error: str, elapsed: float) -> None:
        self.llm_model = None
        self._sync_chat_manager_llm()
        logger.warning("LLM startup failed after %.1fs: %s", elapsed, error)
        self.recovery.activate_safe_mode(
            "llm_load_failed",
            elapsed_seconds=round(elapsed, 2),
            error=error[-1000:],
        )
        self.loading_screen.console.log(
            _llm_failure_text("LLM не загрузилась. Приложение продолжит запуск с доказательным поиском."),
            "warning",
        )
        self._start_documents_stage()

    def _start_documents_stage(self) -> None:
        def build_document_processor():
            from core.document_processor import DocumentProcessor

            processor = DocumentProcessor(
                docs_folder=self.docs_folder,
                recursive=self.docs_recursive,
                max_file_size_mb=100,
            )
            processor.load_documents()
            return processor

        self._run_startup_task(
            stage_name="load_documents",
            status="Читаю и структурирую документы...",
            progress=60,
            task=build_document_processor,
            on_success=self._on_documents_loaded,
            on_failure=self._on_documents_failed,
        )

    def _on_documents_loaded(self, processor, elapsed: float) -> None:
        self.doc_processor = processor
        total_docs = len(processor.documents)
        total_chunks = len(processor.text_chunks)
        if total_docs:
            self.loading_screen.update_status(
                f"Загружено документов: {total_docs}, фрагментов: {total_chunks} ({elapsed:.1f} сек.)",
                68,
            )
        else:
            self.loading_screen.console.log(
                f"В папке {processor.docs_folder} нет читаемых документов",
                "warning",
            )
        self._start_search_stage()

    def _on_documents_failed(self, error: str, elapsed: float) -> None:
        self.doc_processor = None
        logger.warning("Document startup failed after %.1fs: %s", elapsed, error)
        self.recovery.activate_safe_mode(
            "document_load_failed",
            elapsed_seconds=round(elapsed, 2),
            error=error[-1000:],
        )
        self.loading_screen.console.log(
            "Документы не загрузились. Папку можно будет выбрать после запуска.",
            "warning",
        )
        self._start_search_stage()

    def _start_search_stage(self) -> None:
        embedding_model = (
            getattr(self.model_loader, "embedding_model", None)
            if self.model_loader
            else None
        )
        if not embedding_model or not self.doc_processor or not self.doc_processor.documents:
            self.hybrid_search = None
            self.loading_screen.console.log(
                "Поиск пока не создан: нет эмбеддингов или документов",
                "warning",
            )
            self._finish_startup_pipeline()
            return

        def build_search():
            from core.search import HybridSearch

            return HybridSearch(
                embedding_model=embedding_model,
                doc_processor=self.doc_processor,
                use_document_level=True,
            )

        self._run_startup_task(
            stage_name="init_search",
            status="Строю поисковый индекс...",
            progress=75,
            task=build_search,
            on_success=self._on_search_ready,
            on_failure=self._on_search_failed,
        )

    def _on_search_ready(self, search, elapsed: float) -> None:
        self.hybrid_search = search
        self.loading_screen.update_status(
            f"Поисковый индекс готов за {elapsed:.1f} сек.",
            84,
        )
        self._finish_startup_pipeline()

    def _on_search_failed(self, error: str, elapsed: float) -> None:
        self.hybrid_search = None
        logger.warning("Search startup failed after %.1fs: %s", elapsed, error)
        self.recovery.activate_safe_mode(
            "search_index_failed",
            elapsed_seconds=round(elapsed, 2),
            error=error[-1000:],
        )
        self.loading_screen.console.log(
            "Поисковый индекс не создался. Его можно перестроить после запуска.",
            "warning",
        )
        self._finish_startup_pipeline()

    def _finish_startup_pipeline(self) -> None:
        self.loading_screen.update_status("Создаю менеджер чата...", 90)
        self.recovery.run_stage("init_chat_manager", self._init_chat_manager, required=False)

        finish_message = "Загрузка завершена!"
        if self.recovery.safe_mode_active:
            finish_message = "Загрузка завершена в safe-mode"
        self.loading_screen.update_status(finish_message, 100)
        QTimer.singleShot(800, self.loading_screen.complete_loading)

    def _check_models(self) -> None:
        try:
            from download_models import check_models_availability

            self.models_available = check_models_availability() or {"embeddings": False, "llm": False}
            if not self.models_available.get("llm"):
                self._probe_llm_direct()
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"

            if self.models_available.get("embeddings") and self.models_available.get("llm"):
                self.loading_screen.update_status("Локальные модели найдены", 15)
            elif self.models_available.get("embeddings"):
                self.loading_screen.update_status("Embedding-модель найдена. LLM будет ограничена.", 15)
            elif self.models_available.get("llm"):
                self.loading_screen.update_status("Найдена только LLM. Поиск по документам будет ограничен.", 15)
                self.recovery.activate_safe_mode("embedding_model_missing")
            else:
                self.loading_screen.update_status("Локальные модели не найдены. Стартуем с ограничениями.", 15)
                self.recovery.activate_safe_mode("all_models_missing")
        except Exception as exc:
            self.models_available = {"embeddings": False, "llm": False}
            self.recovery.activate_safe_mode("model_probe_failed", error=str(exc))
            self.loading_screen.console.log(f"Проверка моделей не удалась: {exc}", "warning")

    def _load_embedding_model(self) -> None:
        if not self.models_available.get("embeddings"):
            self.model_loader = None
            self.loading_screen.console.log("Embedding-модель недоступна: режим поиска будет отключён", "warning")
            return

        from core.models import AIModelLoader

        self.model_loader = AIModelLoader()
        success = self.model_loader.load_models()
        if not success:
            raise RuntimeError("Не удалось загрузить embedding-модель")
        self.loading_screen.update_status("Embedding-модель загружена", 30)

    def _make_llm_loader(self):
        from core.local_llm import load_llm_model

        lightweight = self.recovery.should_use_lightweight_llm()

        def _loader():
            return load_llm_model(lightweight=lightweight)

        self._llm_loader = _loader
        return _loader

    def _sync_chat_manager_llm(self) -> None:
        if self.chat_manager is not None:
            self.chat_manager.llm_model = self.llm_model
            generation = getattr(self.chat_manager, "generation_service", None)
            if generation is not None:
                generation.llm_model = self.llm_model

    def _probe_llm_direct(self) -> bool:
        """Пытается найти локальную LLM напрямую, без зависимости от downloader-метаданных."""
        try:
            from core.llm.runtime import LocalLLM

            probe = LocalLLM(device="auto")
            llm_path = getattr(probe, "model_path", None)
            if llm_path:
                info = probe.get_model_info() if hasattr(probe, "get_model_info") else {}
                self.models_available.update({
                    "llm": True,
                    "llm_path": llm_path,
                    "llm_name": info.get("name") or Path(llm_path).name,
                    "llm_size": info.get("size", "Unknown"),
                    "llm_type": "llm",
                })
                logger.info("Direct LLM probe found local model: %s", llm_path)
                return True
        except Exception as exc:
            logger.info("Direct LLM probe did not find a usable model: %s", exc)
        return False

    def _reprobe_llm_availability(self) -> bool:
        try:
            from download_models import check_models_availability

            latest = check_models_availability() or {}
            if latest.get("llm"):
                self.models_available.update(latest)
                return True
        except Exception as exc:
            logger.warning("LLM reprobe failed: %s", exc)

        if self._probe_llm_direct():
            return True
        return bool(self.models_available.get("llm"))

    def _ensure_llm_ready(self, *, eager: bool = False, reprobe: bool = True) -> bool:
        model = getattr(self, "llm_model", None)
        if model is not None:
            if eager and hasattr(model, "load_now"):
                loaded = model.load_now()
                ok = bool(loaded and getattr(loaded, "is_available", lambda: True)())
                self._sync_chat_manager_llm()
                return ok
            try:
                ok = bool(model.is_available())
            except Exception:
                ok = True
            self._sync_chat_manager_llm()
            return ok

        if reprobe and not self.models_available.get("llm"):
            self._reprobe_llm_availability()

        if not self.models_available.get("llm") and eager:
            self._probe_llm_direct()

        if not self.models_available.get("llm"):
            # Для явного запроса пользователя пытаемся один раз загрузить LLM напрямую,
            # даже если downloader-метаданные ничего не нашли. Это спасает старые
            # установки, где модель есть локально, но не зарегистрирована.
            if eager:
                try:
                    loader = self._llm_loader or self._make_llm_loader()
                    self.llm_model = loader()
                    self._sync_chat_manager_llm()
                    if self.llm_model:
                        try:
                            return bool(self.llm_model.is_available())
                        except Exception:
                            return True
                except Exception as exc:
                    logger.warning("Direct eager LLM load failed: %s", exc)
            self._sync_chat_manager_llm()
            return False

        startup_mode = self.recovery.choose_llm_startup_mode()
        loader = self._llm_loader or self._make_llm_loader()

        if startup_mode == "disabled":
            self._sync_chat_manager_llm()
            return False

        if startup_mode == "lazy" and not eager:
            self.llm_model = self.recovery.build_lazy_llm(loader, model_label="local_llm")
        else:
            self.llm_model = loader()

        self._sync_chat_manager_llm()
        if not self.llm_model:
            return False
        try:
            return bool(self.llm_model.is_available())
        except Exception:
            return True

    def _load_llm_model(self) -> None:
        if not self.models_available.get("llm") and not self._probe_llm_direct():
            self.loading_screen.update_status("LLM не найдена. Поиск продолжит работать без генерации.", 50)
            self.llm_model = None
            self._sync_chat_manager_llm()
            return

        startup_mode = self.recovery.choose_llm_startup_mode()
        loader = self._make_llm_loader()

        if startup_mode == "disabled":
            self.llm_model = None
            self.recovery.activate_safe_mode("llm_disabled_by_policy", startup_mode=startup_mode)
            self.loading_screen.update_status("LLM отключена политикой safe-mode.", 50)
            self._sync_chat_manager_llm()
            return

        if startup_mode == "lazy":
            self.llm_model = self.recovery.build_lazy_llm(loader, model_label="local_llm")
            self.loading_screen.update_status("LLM в lazy-режиме: загрузится по первому запросу.", 50)
            self._sync_chat_manager_llm()
            return

        self.loading_screen.update_status("Загружаю локальную LLM. Первый запуск может занять несколько минут...", 42)
        QApplication.processEvents()
        self.llm_model = loader()
        if self.llm_model and hasattr(self.llm_model, "is_available") and self.llm_model.is_available():
            info = self.llm_model.get_model_info()
            model_name = info.get("name", "Unknown")
            self.loading_screen.update_status(f"LLM загружена: {model_name}", 50)
        else:
            logger.warning("LLM model was detected but failed to initialize")
            self.loading_screen.update_status(
                _llm_failure_text("LLM не загрузилась. Продолжаем без генерации."),
                50,
            )
            self.llm_model = None
        self._sync_chat_manager_llm()

    def _load_documents(self) -> None:
        from core.document_processor import DocumentProcessor

        self.doc_processor = DocumentProcessor(
            docs_folder=self.docs_folder,
            recursive=self.docs_recursive,
            max_file_size_mb=100,
        )
        success = self.doc_processor.load_documents()
        if not success:
            self.loading_screen.update_status("⚠️ Нет документов для обработки", 65)
        else:
            total_docs = len(self.doc_processor.documents)
            total_chars = sum(doc.size_chars for doc in self.doc_processor.documents)
            self.loading_screen.update_status(f"✅ Загружено документов: {total_docs}, символов: {total_chars:,}", 65)

    def _init_search(self) -> None:
        from core.search import HybridSearch

        if self.doc_processor and self.doc_processor.documents and self.model_loader and getattr(self.model_loader, "embedding_model", None):
            self.hybrid_search = HybridSearch(
                embedding_model=self.model_loader.embedding_model,
                doc_processor=self.doc_processor,
                use_document_level=True,
            )
            self.loading_screen.update_status(f"✅ Поисковая система готова ({len(self.doc_processor.documents)} документов)", 80)
        else:
            self.loading_screen.update_status("⚠️ Поиск запущен без индекса (нет документов или эмбеддингов)", 80)
            self.hybrid_search = None

    def _init_chat_manager(self) -> None:
        from core.chat_mode import ChatManager

        self.chat_manager = ChatManager(llm_model=self.llm_model, hybrid_search=self.hybrid_search)
        generation = getattr(self.chat_manager, "generation_service", None)
        if generation is not None:
            generation.enabled = bool(getattr(self, "generation_enabled", True))
            generation.config["allow_creative_exercises"] = bool(getattr(self, "generation_creative_enabled", True))
            catalog_root = Path(str(getattr(self, "generation_catalog_root", "documents")))
            if not catalog_root.is_absolute():
                catalog_root = resolve_project_path(catalog_root)
            generation.set_catalog_root(catalog_root)
        if hasattr(self.chat_manager, "set_runtime_preferences"):
            self.chat_manager.set_runtime_preferences(
                query_mode=getattr(self, "query_mode_preference", "auto"),
                strictness=getattr(self, "search_strictness_preference", "balanced"),
                source_scope=getattr(self, "source_scope_preference", "all"),
            )
        self.loading_screen.update_status("✅ Менеджер чата создан", 90)

    def _on_loading_complete(self) -> None:
        self.loading_screen.close()
        self._init_ui()
        self._update_mode_display()
        if hasattr(self, "_refresh_workspace_views"):
            self._refresh_workspace_views()
        self.show()

    def _on_loading_error(self, error_message: str) -> None:
        logger.error("Startup failed: %s", error_message)
        worker = getattr(self, "_startup_worker", None)
        if worker is not None and worker.isRunning():
            self._startup_cancel_requested = True
            logger.warning("Startup cancellation deferred until the active stage exits")
            return
        QTimer.singleShot(2000, QApplication.quit)
