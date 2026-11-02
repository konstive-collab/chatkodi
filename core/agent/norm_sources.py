"""Resolve a named standards document and read original library PDFs in full."""
from __future__ import annotations

import re
from pathlib import Path


def select_named(sources, message):
    """An explicitly named file wins over automatic selection.

    Do not interpret text *inside* a document as a file-selection instruction.
    An unresolved quoted document name must not fall back to unrelated norms.
    """
    key = message.casefold()
    hints = re.findall(r'(?:из|по)\s+документ\w*\s+[«"]([^»"]+)[»"]', message, re.I)
    if not hints:
        hints = re.findall(r'(?:из|по)\s+документ\w*\s+([^\s,.;!?«"]+)', message, re.I)
        hints = [h for h in hints if h.casefold() not in {"с", "со", "который", "которого", "этого"}]
    if hints:
        named = [(name, text) for name, text in sources if any(h.casefold() in name.casefold() for h in hints)]
        if named:
            return list(dict.fromkeys(named)), ""
        return [], "Не найден указанный документ: " + ", ".join(hints)
    named = [(name, text) for name, text in sources
             if len(Path(name).stem) >= 3 and (name.casefold() in key or Path(name).stem.casefold() in key)]
    return (list(dict.fromkeys(named)) if named else list(sources)), ""


def read_library(docs, llm, progress, cache):
    from core.agent.pdf_tables import read_pdf
    from core.agent.attachments import _tables_as_markdown

    sources, issues = [], []
    for doc in docs:
        path = Path(str(doc.get("path") or ""))
        label = str(doc.get("filename") or doc.get("name") or path.name or "документ")
        if path.suffix.lower() == ".pdf" and path.is_file():
            stat = path.stat()
            key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, bool(getattr(llm, "supports_vision", False)))
            try:
                if key not in cache:
                    cache[key] = read_pdf(path, llm, progress)
                reading = cache[key]
                sources.append((label, reading.text))
                issues.extend(f"{label}: {issue}" for issue in reading.issues)
            except Exception as exc:
                issues.append(f"{label}: PDF не прочитан ({type(exc).__name__})")
        elif path.is_file():
            text = _tables_as_markdown(path) or str(doc.get("content") or "")
            sources.append((label, text))
        else:
            sources.append((label, str(doc.get("content") or "")))
            if path.suffix.lower() == ".pdf":
                issues.append(f"{label}: исходный PDF недоступен, проверен только найденный фрагмент")
    return sources, issues
