import json
from email.message import EmailMessage

import numpy as np
from docx import Document as WordDocument

from core.document_processor import DocumentProcessor
from core.embedding_cache import EmbeddingCache


def test_external_folders_with_app_like_names_are_indexed(tmp_path):
    for folder_name in ("manuals", "logs", "core", "scripts"):
        folder = tmp_path / folder_name
        folder.mkdir()
        (folder / f"{folder_name}.txt").write_text(f"useful content from {folder_name}", encoding="utf-8")

    processor = DocumentProcessor(str(tmp_path), recursive=True)
    assert processor.load_documents()
    assert sorted(processor.get_document_names()) == [
        "core\\core.txt",
        "logs\\logs.txt",
        "manuals\\manuals.txt",
        "scripts\\scripts.txt",
    ]


def test_extensionless_text_file_is_discovered_from_nested_folder(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "NOTICE").write_text(
        "Plain text without a filename extension.",
        encoding="utf-8",
    )

    processor = DocumentProcessor(str(tmp_path), recursive=True)
    assert processor.load_documents()
    assert processor.get_document_names() == ["nested\\NOTICE"]


def test_corpus_ignore_control_file_is_never_indexed(tmp_path):
    (tmp_path / ".chatkodiignore").write_text("*.md\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("searchable", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("hidden", encoding="utf-8")

    processor = DocumentProcessor(str(tmp_path), recursive=True)
    assert processor.load_documents()
    assert processor.get_document_names() == ["visible.txt"]


def test_docx_preserves_heading_table_and_block_order(tmp_path):
    path = tmp_path / "structured.docx"
    doc = WordDocument()
    doc.add_heading("Quarterly Review", level=1)
    doc.add_paragraph("Before the table")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Latency"
    table.cell(1, 1).text = "42 ms"
    doc.add_paragraph("After the table")
    doc.save(path)

    processor = DocumentProcessor(str(tmp_path))
    assert processor.load_documents()
    content = processor.documents[0].content
    assert "# Quarterly Review" in content
    # Table rows carry their header, so a row stays meaningful after chunking.
    assert content.index("Before the table") < content.index("Metric: Latency | Value: 42 ms") < content.index("After the table")


def test_csv_json_and_eml_are_read_as_structured_text(tmp_path):
    (tmp_path / "items.csv").write_text('name,comment\nalpha,"contains, comma"\n', encoding="utf-8")
    (tmp_path / "record.json").write_text(json.dumps({"service": {"state": "ready"}}), encoding="utf-8")

    message = EmailMessage()
    message["Subject"] = "Deployment status"
    message["From"] = "ops@example.test"
    message["To"] = "team@example.test"
    message.set_content("Release candidate is ready for validation.")
    (tmp_path / "status.eml").write_bytes(message.as_bytes())

    processor = DocumentProcessor(str(tmp_path))
    assert processor.load_documents()
    by_name = {document.name: document.content for document in processor.documents}
    assert "name: alpha | comment: contains, comma" in by_name["items.csv"]
    assert "service.state: ready" in by_name["record.json"]
    assert "Subject: Deployment status" in by_name["status.eml"]
    assert "ready for validation" in by_name["status.eml"]


def test_chunker_keeps_words_and_short_final_fact(tmp_path):
    processor = DocumentProcessor(str(tmp_path))
    text = "# Scope\n" + "Sentence one contains complete words. " * 70 + "\nFinal fact: omega."
    chunks = processor._split_text_into_chunks(text, max_chars=240, overlap=60, min_chars=80)
    assert chunks
    assert any("Final fact: omega." in chunk_text for _start, _end, chunk_text in chunks)
    for _start, _end, chunk_text in chunks:
        assert not chunk_text.startswith("entence ")
        assert not chunk_text.endswith(" Sent")


def test_embedding_cache_is_namespaced_hashed_and_atomic(tmp_path):
    cache_path = tmp_path / "cache.npz"
    meta_path = tmp_path / "cache.json"
    secret_text = "confidential document sentence"
    cache = EmbeddingCache(str(cache_path), str(meta_path), namespace="model-a")
    cache.set(secret_text, np.asarray([1.0, 2.0], dtype=np.float32))
    assert cache.save(force=True)

    raw_meta = meta_path.read_text(encoding="utf-8")
    payload = json.loads(raw_meta)
    assert payload["schema_version"] == 2
    assert payload["namespace"] == "model-a"
    assert secret_text not in raw_meta

    reloaded = EmbeddingCache(str(cache_path), str(meta_path), namespace="model-a")
    assert np.allclose(reloaded.get(secret_text), [1.0, 2.0])
    changed = EmbeddingCache(str(cache_path), str(meta_path), namespace="model-b")
    assert changed.get(secret_text) is None


def test_initials_are_not_split_by_sentence_boundary_repair(tmp_path):
    path = tmp_path / "shift.docx"
    doc = WordDocument()
    doc.add_paragraph("Состав смены: Иванов И.П. (кладовщик), Смирнова Е.О. (менеджер).Далее текст")
    doc.save(path)

    processor = DocumentProcessor(str(tmp_path))
    assert processor.load_documents()
    content = processor.documents[0].content
    assert "Иванов И.П. (кладовщик), Смирнова Е.О. (менеджер)." in content
    assert "\nДалее текст" in content
