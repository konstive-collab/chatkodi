"""Load document readers with actionable errors and check their imports."""
from __future__ import annotations

import importlib
import sys


READERS = {
    'pdfplumber': ('PDF', 'pdfplumber==0.11.7'),
    'openpyxl': ('Excel', 'openpyxl==3.1.5'),
    'docx': ('Word', 'python-docx==1.2.0'),
    'pptx': ('PowerPoint', 'python-pptx==1.0.2'),
}


class DocumentDependencyError(RuntimeError):
    """A reader or one of its transitive dependencies cannot be imported."""


def load_reader(module):
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as exc:
        label, package = READERS[module]
        missing = getattr(exc, 'name', None)
        detail = f'отсутствует модуль {missing}' if missing else str(exc)
        raise DocumentDependencyError(
            f'Компонент чтения {label} «{module}» недоступен: {detail}. '
            'В лаунчере откройте «Ручная установка» → «Восстановить компоненты документов», '
            'затем повторите запрос. '
            f'Пакет: {package}. Python: {sys.executable}'
        ) from exc


def check_document_readers():
    importlib.invalidate_caches()  # pip may have just installed a missing reader
    errors = {}
    for module in READERS:
        try:
            load_reader(module)
        except DocumentDependencyError as exc:
            errors[module] = str(exc)
    return {'ok': not errors, 'python': sys.executable, 'errors': errors}
