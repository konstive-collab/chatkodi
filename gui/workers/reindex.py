"""Background document reindex worker."""

from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal

from core.logging_utils import get_logger

logger = get_logger(__name__)


class ReindexWorker(QThread):
    """Reload documents and rebuild or create the search index outside the UI thread."""

    done = pyqtSignal(bool, str, object, object)

    def __init__(
        self,
        doc_processor,
        hybrid_search,
        parent=None,
        *,
        embedding_model=None,
        new_docs_folder=None,
        recursive=None,
    ):
        super().__init__(parent)
        self.doc_processor = doc_processor
        self.hybrid_search = hybrid_search
        self.embedding_model = embedding_model
        self.new_docs_folder = new_docs_folder
        self.recursive = recursive

    def run(self):
        try:
            from core.document_processor import DocumentProcessor

            target_folder = self.new_docs_folder or str(self.doc_processor.docs_folder)
            target_recursive = (
                bool(self.recursive)
                if self.recursive is not None
                else bool(getattr(self.doc_processor, "recursive", True))
            )
            replacement_processor = DocumentProcessor(
                docs_folder=str(target_folder),
                recursive=target_recursive,
                max_file_size_mb=int(getattr(self.doc_processor, "max_file_size_mb", 100) or 100),
            )
            if not replacement_processor.load_documents():
                stats = dict(getattr(replacement_processor, "stats", {}) or {})
                failures = list(stats.get("failed_files_details") or [])
                warnings = list(stats.get("scan_warnings") or [])
                details = failures[:3] or warnings[:3]
                reason = "; ".join(
                    (
                        f"{item.get('file') or item.get('path')}: "
                        f"{item.get('message') or item.get('reason')}"
                    ).strip(": ")
                    for item in details
                )
                suffix = f" Причина: {reason}" if reason else ""
                self.done.emit(
                    False,
                    "В выбранной папке нет читаемых поддерживаемых документов. "
                    f"Предыдущий индекс сохранен.{suffix}",
                    None,
                    None,
                )
                return

            replacement_search = None
            if self.embedding_model is not None:
                try:
                    from core.search import HybridSearch

                    replacement_search = HybridSearch(
                        embedding_model=self.embedding_model,
                        doc_processor=replacement_processor,
                        use_document_level=True,
                    )
                except Exception as exc:
                    logger.exception("Search index creation failed")
                    self.done.emit(
                        False,
                        f"Документы прочитаны, но новый поисковый индекс не создался: {exc}. "
                        "Предыдущий индекс сохранен.",
                        None,
                        None,
                    )
                    return

            document_count = len(replacement_processor.documents)
            chunk_count = len(replacement_processor.text_chunks)
            self.done.emit(
                True,
                f"Индекс обновлен: документов {document_count}, фрагментов {chunk_count}.",
                replacement_processor,
                replacement_search,
            )
        except Exception as exc:
            logger.exception("Background reindex failed")
            self.done.emit(False, str(exc) or type(exc).__name__, None, None)
