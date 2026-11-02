"""Document-folder selection, copy and reindex operations."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from core.logging_utils import get_logger
from core.documents.processor import SUPPORTED_EXTENSIONS
from gui.workers.reindex import ReindexWorker

logger = get_logger(__name__)


class ChatKodiDocumentsMixin:
    @staticmethod
    def _is_supported_drop_path(path: str | Path) -> bool:
        candidate = Path(path)
        return candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS

    @staticmethod
    def _available_destination(docs_dir: Path, source: Path) -> Path:
        """Choose a non-destructive destination for a dropped/imported file."""
        candidate = docs_dir / source.name
        try:
            if candidate.exists() and candidate.resolve() == source.resolve():
                return candidate
        except OSError:
            pass
        if not candidate.exists():
            return candidate
        counter = 2
        while True:
            candidate = docs_dir / f"{source.stem} ({counter}){source.suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def _import_document_paths(self, paths) -> tuple[int, int, list[str]]:
        docs_dir = Path(self.doc_processor.docs_folder if self.doc_processor else self.docs_folder).resolve()
        docs_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        skipped = 0
        errors: list[str] = []
        for raw_path in paths:
            source = Path(raw_path)
            if not self._is_supported_drop_path(source):
                skipped += 1
                continue
            destination = self._available_destination(docs_dir, source)
            try:
                if destination.exists() and destination.resolve() == source.resolve():
                    skipped += 1
                    continue
                shutil.copy2(source, destination)
                copied += 1
            except OSError as exc:
                errors.append(f"{source.name}: {exc}")
        if copied:
            self._update_index()
        return copied, skipped, errors

    def _add_documents(self) -> None:
        docs_dir = os.path.abspath(str(self.doc_processor.docs_folder if self.doc_processor else self.docs_folder))
        os.makedirs(docs_dir, exist_ok=True)

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите документы",
            docs_dir,
            (
                "Документы ChatKodi (*.txt *.md *.markdown *.log *.pdf *.doc *.docx *.rtf "
                "*.html *.htm *.xhtml *.csv *.tsv *.json *.jsonl *.xml *.yaml *.yml "
                "*.xlsx *.xlsm *.xltx *.xltm *.xls *.pptx *.ppt *.odt *.ods *.odp *.epub "
                "*.eml *.rst *.tex *.ipynb *.ini *.cfg *.conf *.toml);;"
                "Текст и web (*.txt *.md *.log *.html *.htm *.xhtml *.csv *.json *.jsonl *.xml *.rst);;"
                "Office/PDF (*.pdf *.doc *.docx *.rtf *.xlsx *.xls *.pptx *.ppt *.odt *.ods *.odp);;"
                "Все файлы (*.*)"
            ),
        )
        if not files:
            return

        copied, skipped, errors = self._import_document_paths(files)
        details = f"Добавлено документов: {copied}."
        if skipped:
            details += f" Пропущено: {skipped}."
        if copied:
            details += " Индекс обновляется в фоне."
        if errors:
            QMessageBox.warning(self, "Добавление завершено с ошибками", details + "\n\n" + "\n".join(errors[:8]))
        else:
            QMessageBox.information(self, "Готово", details)

    def _drop_attaches_to_chat(self) -> bool:
        """Files dropped on the chat go into the conversation; on other pages
        they are added to the document library."""
        pages = getattr(self, "workspace_pages", {})
        return "chat" in pages and self.page_stack.currentIndex() == pages["chat"]

    def dragEnterEvent(self, event) -> None:
        from core.agent.attachments import is_supported

        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        accept = is_supported if self._drop_attaches_to_chat() else self._is_supported_drop_path
        accepted = [Path(url.toLocalFile()) for url in urls if url.isLocalFile() and accept(Path(url.toLocalFile()))]
        if accepted:
            event.acceptProposedAction()
            if self._drop_attaches_to_chat() and hasattr(self, "_show_drop_overlay"):
                self._show_drop_overlay(accepted)
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if hasattr(self, "_hide_drop_overlay"):
            self._hide_drop_overlay()
        if self._drop_attaches_to_chat():
            self._attach_files(paths)
            event.acceptProposedAction()
            return
        copied, skipped, errors = self._import_document_paths(paths)
        if copied:
            self.status.showMessage(f"Добавлено документов: {copied}. Индекс обновляется…")
            event.acceptProposedAction()
        else:
            event.ignore()
        if errors:
            QMessageBox.warning(self, "Не все документы добавлены", "\n".join(errors[:8]))
        elif skipped and not copied:
            QMessageBox.information(self, "Документы", "Поддерживаемых новых документов не найдено.")

    def _select_docs_folder(self) -> None:
        current_dir = os.path.abspath(str(self.doc_processor.docs_folder if self.doc_processor else self.docs_folder))
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с документами",
            current_dir,
            QFileDialog.ShowDirsOnly,
        )
        if not folder:
            return

        reply = QMessageBox.question(
            self,
            "Рекурсивный поиск",
            "Искать документы во всех подпапках?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Cancel:
            return

        recursive = reply == QMessageBox.Yes
        self._pending_docs_folder = folder
        self._pending_docs_recursive = recursive
        if hasattr(self, "folder_label"):
            self.folder_label.setText(f"Папка: {folder}")
        self.status.showMessage("Читаю новую папку документов...")
        self._update_index(silent=False, new_docs_folder=folder, recursive=recursive)

    def _reset_docs_watcher(self) -> None:
        try:
            for path in self.fs_watcher.directories():
                self.fs_watcher.removePath(path)
            for path in self.fs_watcher.files():
                self.fs_watcher.removePath(path)

            if not self.doc_processor:
                return
            root = str(self.doc_processor.docs_folder)
            if not os.path.isdir(root):
                return

            self.fs_watcher.addPath(root)
            if not getattr(self.doc_processor, "recursive", False):
                return

            added = 0
            max_watched_dirs = 512
            for dirpath, dirnames, _ in os.walk(root):
                for dirname in list(dirnames):
                    full = Path(dirpath) / dirname
                    try:
                        if hasattr(self.doc_processor, "_is_ignored_path") and self.doc_processor._is_ignored_path(full):
                            continue
                    except Exception:
                        pass
                    try:
                        self.fs_watcher.addPath(str(full))
                        added += 1
                        if added >= max_watched_dirs:
                            logger.info("Document watcher limit reached: %s folders", max_watched_dirs)
                            return
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("Failed to reset document watcher: %s", exc)

    def _on_docs_folder_changed(self, _path: str) -> None:
        self.reindex_debounce.start(900)

    def _update_index(self, silent: bool = False, new_docs_folder: str | None = None, recursive: bool | None = None) -> None:
        self.status.showMessage("Обновление индекса...")
        try:
            if not self.doc_processor:
                if not silent:
                    QMessageBox.warning(self, "Индекс", "Документы еще не инициализированы.")
                return

            worker = getattr(self, "_reindex_worker", None)
            if worker and worker.isRunning():
                if not silent:
                    QMessageBox.information(self, "Индекс", "Индекс уже обновляется.")
                return

            active_search = None
            if self.chat_manager and getattr(self.chat_manager, "hybrid_search", None):
                active_search = self.chat_manager.hybrid_search
            elif getattr(self, "hybrid_search", None):
                active_search = self.hybrid_search

            embedding_model = getattr(self.model_loader, "embedding_model", None) if getattr(self, "model_loader", None) else None
            self._reindex_worker = ReindexWorker(
                self.doc_processor,
                active_search,
                self,
                embedding_model=embedding_model,
                new_docs_folder=new_docs_folder,
                recursive=recursive,
            )
            self._reindex_worker.done.connect(
                lambda ok, msg, processor, search: self._on_reindex_done(
                    ok,
                    msg,
                    processor,
                    search,
                    silent,
                )
            )
            self._reindex_worker.start()
        except Exception as exc:
            self.status.showMessage("Ошибка обновления")
            if not silent:
                QMessageBox.critical(self, "Ошибка", f"Не удалось обновить индекс: {exc}")

    def _on_reindex_done(self, ok: bool, msg: str, replacement_processor, active_search, silent: bool) -> None:
        if ok:
            pending_folder = getattr(self, "_pending_docs_folder", None)
            pending_recursive = getattr(self, "_pending_docs_recursive", None)
            if pending_folder:
                self.docs_folder = pending_folder
                self.docs_recursive = bool(pending_recursive)
                self._pending_docs_folder = None
                self._pending_docs_recursive = None
                self._save_settings()

            if replacement_processor is not None:
                self.doc_processor = replacement_processor
                self.hybrid_search = active_search
                if self.chat_manager:
                    self.chat_manager.hybrid_search = active_search

            self._reset_docs_watcher()
            if self.doc_processor and hasattr(self, "info_label"):
                try:
                    self.info_label.setText(self.doc_processor.get_document_info())
                except Exception:
                    pass
            if hasattr(self, "_refresh_workspace_views"):
                self._refresh_workspace_views()
            self.status.showMessage("Индекс обновлен")
            if not silent:
                QMessageBox.information(self, "Готово", msg or "Индекс документов обновлен.")
            return

        self._pending_docs_folder = None
        self._pending_docs_recursive = None
        self.status.showMessage("Ошибка обновления")
        if not silent:
            QMessageBox.critical(self, "Ошибка", f"Не удалось обновить индекс: {msg}")

    def _show_documents_stats(self) -> None:
        if not self.doc_processor:
            QMessageBox.information(self, "Статистика", "Документы не загружены")
            return

        stats = self.doc_processor.stats
        message = (
            "СТАТИСТИКА ДОКУМЕНТОВ\n\n"
            f"Всего документов: {stats.get('total_documents', 0)}\n"
            f"Общий размер: {stats.get('total_size_mb', 0):.1f} MB\n"
            f"Всего символов: {stats.get('total_chars', 0):,}\n"
            f"Время загрузки: {stats.get('load_time_seconds', 0):.2f} сек\n"
        )
        if stats.get("by_extension"):
            message += "\nПо типам файлов:\n"
            for ext, count in stats["by_extension"].items():
                message += f"  - {ext}: {count}\n"
        if stats.get("failed_files"):
            message += f"\nНе прочитано файлов: {len(stats.get('failed_files') or [])}"
            details = list(stats.get("failed_files_details") or [])[:5]
            for item in details:
                message += f"\n  - {item.get('file')}: {item.get('reason')}"
        if stats.get("scan_warnings"):
            message += f"\nОшибок доступа при сканировании: {len(stats.get('scan_warnings') or [])}"
        if stats.get("skipped_large_files"):
            message += f"\nПропущено слишком больших файлов: {len(stats.get('skipped_large_files') or [])}"
        QMessageBox.information(self, "Статистика документов", message)
