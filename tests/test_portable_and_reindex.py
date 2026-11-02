from __future__ import annotations

import json
from pathlib import Path

from core.chat.manager import ChatManager
from core.document_processor import DocumentProcessor
from core.model_integrity import inspect_hf_model_payload
from gui.workers.reindex import ReindexWorker
from tools.portable_verify import create_manifest, verify_portable


def _write_model(root: Path, name: str) -> Path:
    model = root / name
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"w" * 2048)
    return model


def _write_gguf(path: Path) -> Path:
    """Minimal file that passes the GGUF header check (magic, v3, 1 tensor, 1 kv)."""
    import struct

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + b"w" * (1024 * 1024))
    return path


LLM_GGUF = "models_cache/llms/generic-llm-Q4_K_M.gguf"
EMBEDDING_GGUF = "models_cache/embeddings/generic-embedding-Q8_0.gguf"


def _write_portable_payload(root: Path) -> Path:
    root.mkdir()
    (root / "ChatKodiApp.exe").write_bytes(b"app")
    (root / "ChatKodiLauncher.exe").write_bytes(b"launcher")
    (root / "portable.flag").write_text("portable", encoding="ascii")
    (root / "VERSION.txt").write_text("test", encoding="utf-8")
    (root / "chatkodi_config.json").write_text("{}", encoding="utf-8")
    internal = root / "_internal"
    internal.mkdir()
    (internal / "python311.dll").write_bytes(b"runtime")
    _write_gguf(root / LLM_GGUF)
    _write_gguf(root / EMBEDDING_GGUF)
    server = root / "llama.cpp" / "b1" / "llama-server.exe"
    server.parent.mkdir(parents=True)
    server.write_bytes(b"server")
    (server.parent / "ggml-cuda.dll").write_bytes(b"cuda")
    (root / "docs").mkdir()
    (root / "docs" / "user.txt").write_text("not part of the build", encoding="utf-8")
    (root / "runtime" / "logs").mkdir(parents=True)
    (root / "runtime" / "logs" / "app.log").write_text("mutable", encoding="utf-8")
    return root


def test_portable_manifest_quick_and_full_verification(tmp_path: Path) -> None:
    root = _write_portable_payload(tmp_path / "portable")
    manifest = create_manifest(root)
    assert manifest["file_count"] > 0
    assert all(not item["path"].startswith(("docs/", "runtime/")) for item in manifest["files"])
    assert verify_portable(root, full=False)["ok"] is True
    assert verify_portable(root, full=True)["ok"] is True

    weight = root / LLM_GGUF
    data = bytearray(weight.read_bytes())
    data[-16:] = b"x" * 16  # same size, header intact, different content
    weight.write_bytes(bytes(data))
    assert verify_portable(root, full=False)["ok"] is True
    full_report = verify_portable(root, full=True)
    assert full_report["ok"] is False
    assert any("SHA-256 mismatch" in item for item in full_report["errors"])

    weight.write_bytes(b"short")
    quick_report = verify_portable(root, full=False)
    assert quick_report["ok"] is False
    assert any("size mismatch" in item or "truncated" in item for item in quick_report["errors"])


def test_model_integrity_detects_missing_index_shard(tmp_path: Path) -> None:
    model = _write_model(tmp_path, "sharded")
    (model / "model.safetensors").unlink()
    (model / "model-00001-of-00002.safetensors").write_bytes(b"a" * 2048)
    index = {
        "weight_map": {
            "layer.0": "model-00001-of-00002.safetensors",
            "layer.1": "model-00002-of-00002.safetensors",
        }
    }
    (model / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    report = inspect_hf_model_payload(model)
    assert report["ok"] is False
    assert any("missing referenced weight shards" in item for item in report["errors"])


def test_reindex_builds_replacement_without_mutating_live_processor(tmp_path: Path) -> None:
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "old.txt").write_text("old searchable content", encoding="utf-8")
    (new_dir / "new.txt").write_text("new searchable content", encoding="utf-8")
    current = DocumentProcessor(str(old_dir))
    assert current.load_documents() is True

    emitted = []
    worker = ReindexWorker(
        current,
        None,
        embedding_model=None,
        new_docs_folder=str(new_dir),
        recursive=True,
    )
    worker.done.connect(lambda *args: emitted.append(args))
    worker.run()

    assert len(emitted) == 1
    ok, _message, replacement, replacement_search = emitted[0]
    assert ok is True
    assert replacement_search is None
    assert replacement is not current
    assert replacement.get_document_names() == ["new.txt"]
    assert current.get_document_names() == ["old.txt"]
    assert current.docs_folder == old_dir.resolve()


def test_fuzzy_coverage_handles_typos_and_close_inflections() -> None:
    coverage = ChatManager._fuzzy_term_coverage(
        {"противнк", "сосредоточить", "направление", "сила"},
        {"противник", "сосредотачивать", "направление", "усилие"},
    )
    assert coverage >= 0.75
    assert ChatManager._fuzzy_term_coverage(
        {"contract", "payment"},
        {"weather", "temperature"},
    ) == 0.0


def test_missing_documentation_does_not_block_portable_start(tmp_path: Path) -> None:
    from tools.portable_verify import startup_preflight

    root = _write_portable_payload(tmp_path / "portable")
    manuals = root / "manuals"
    manuals.mkdir()
    (manuals / "GUIDE_RU.md").write_text("doc", encoding="utf-8")
    nested = root / "_internal" / "sklearn" / ".libs"
    nested.mkdir(parents=True)
    (nested / "msvcp140.dll").write_bytes(b"dup")
    (nested / "vcruntime140.dll").write_bytes(b"dup")
    manifest = create_manifest(root)
    critical = {item["path"] for item in manifest["files"] if item["critical"]}
    assert "_internal/python311.dll" in critical
    assert "_internal/sklearn/.libs/vcruntime140.dll" not in critical

    (manuals / "GUIDE_RU.md").unlink()
    (nested / "msvcp140.dll").unlink()
    (nested / "vcruntime140.dll").unlink()

    preflight = startup_preflight(root)
    assert preflight["ok"] is True
    full = verify_portable(root, full=True)
    assert full["ok"] is True
    assert any("manuals/GUIDE_RU.md" in item for item in full["warnings"])


def test_damaged_llm_degrades_but_missing_embedding_blocks(tmp_path: Path) -> None:
    from tools.portable_verify import startup_preflight

    root = _write_portable_payload(tmp_path / "portable")
    create_manifest(root)
    (root / LLM_GGUF).write_bytes(b"short")
    preflight = startup_preflight(root)
    assert preflight["ok"] is True
    assert preflight["llm_degraded"] is True

    (root / EMBEDDING_GGUF).unlink()
    preflight = startup_preflight(root)
    assert preflight["ok"] is False
    assert preflight["blocking_errors"]


def _write_hub_style_embedding(root: Path) -> Path:
    """Embedding folder as an old snapshot_download leaves it."""
    model = _write_model(root, "hub-embedding")
    (model / "modules.json").write_text("[]", encoding="utf-8")
    (model / "1_Pooling").mkdir()
    (model / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")
    (model / "pytorch_model.bin").write_bytes(b"p" * 2048)
    (model / "openvino").mkdir()
    (model / "openvino" / "openvino_model.bin").write_bytes(b"o" * 2048)
    (model / "openvino" / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")
    download = model / ".cache" / "huggingface" / "download" / "openvino"
    download.mkdir(parents=True)
    (model / ".cache" / "huggingface" / ".gitignore").write_text("*", encoding="utf-8")
    (download / "openvino_model_qint8_quantized.xml.metadata").write_text("meta", encoding="utf-8")
    (download.parent / "model.safetensors.metadata").write_text("meta", encoding="utf-8")
    return model


def test_prune_model_payload_keeps_only_what_the_app_loads(tmp_path: Path) -> None:
    from tools.portable_verify import model_payload_junk, prune_model_payload

    model = _write_hub_style_embedding(tmp_path)
    report = prune_model_payload(model)

    remaining = sorted(path.relative_to(model).as_posix() for path in model.rglob("*") if path.is_file())
    assert remaining == [
        "1_Pooling/config.json",
        "config.json",
        "model.safetensors",
        "modules.json",
        "tokenizer.json",
    ]
    assert not (model / ".cache").exists()
    assert not (model / "openvino").exists()
    assert report["removed_files"] == 6
    assert "pytorch_model.bin" in report["removed"]
    assert inspect_hf_model_payload(model)["ok"] is True
    assert model_payload_junk(model) == []


def test_prune_keeps_pytorch_weights_when_they_are_the_only_ones(tmp_path: Path) -> None:
    from tools.portable_verify import prune_model_payload

    model = _write_model(tmp_path, "legacy")
    (model / "model.safetensors").unlink()
    (model / "pytorch_model.bin").write_bytes(b"p" * 2048)
    assert prune_model_payload(model)["removed_files"] == 0
    assert (model / "pytorch_model.bin").is_file()
    assert inspect_hf_model_payload(model)["ok"] is True


def test_llama_server_files_are_critical_and_block_start(tmp_path: Path) -> None:
    from tools.portable_verify import startup_preflight

    root = _write_portable_payload(tmp_path / "portable")
    manifest = create_manifest(root)
    flags = {item["path"]: item["critical"] for item in manifest["files"]}
    assert flags["llama.cpp/b1/llama-server.exe"] is True
    assert flags["llama.cpp/b1/ggml-cuda.dll"] is True
    assert flags[LLM_GGUF] is True and flags[EMBEDDING_GGUF] is True

    (root / "llama.cpp" / "b1" / "ggml-cuda.dll").unlink()
    preflight = startup_preflight(root)
    assert preflight["ok"] is False
    assert any("ggml-cuda.dll" in item for item in preflight["blocking_errors"])


def test_gguf_header_check_rejects_non_gguf(tmp_path: Path) -> None:
    from core.model_integrity import inspect_gguf_model

    good = _write_gguf(tmp_path / "ok.gguf")
    assert inspect_gguf_model(good)["ok"] is True
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"\0" * (2 * 1024 * 1024))
    assert inspect_gguf_model(bad)["ok"] is False
    assert inspect_gguf_model(tmp_path / "missing.gguf")["ok"] is False


def test_verify_portable_command_logs_missing_manuals(tmp_path: Path) -> None:
    from tools.portable_verify import PREFLIGHT_LOG, startup_preflight, verify_and_log

    root = _write_portable_payload(tmp_path / "portable")
    (root / "manuals").mkdir()
    (root / "manuals" / "GUIDE_RU.md").write_text("doc", encoding="utf-8")
    create_manifest(root)
    log_path = root / PREFLIGHT_LOG

    clean = verify_and_log(root)
    assert clean["ok"] is True
    assert clean["warning_count"] == 0
    assert clean["log_path"] == str(log_path)
    assert "verify-portable (quick): ok" in log_path.read_text(encoding="utf-8")

    (root / "manuals" / "GUIDE_RU.md").unlink()
    # The start-time preflight only looks at critical files.
    assert startup_preflight(root)["warnings"] == []
    for full in (False, True):
        report = verify_and_log(root, full=full)
        assert report["ok"] is True
        assert report["warnings"] == ["missing file: manuals/GUIDE_RU.md"]
    log = log_path.read_text(encoding="utf-8")
    assert log.count("WARNING missing file: manuals/GUIDE_RU.md") == 2
    assert "verify-portable (full): ok" in log
    assert "warnings=1" in log


def test_main_verify_portable_writes_log(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    root = _write_portable_payload(tmp_path / "portable")
    (root / "manuals").mkdir()
    (root / "manuals" / "GUIDE_RU.md").write_text("doc", encoding="utf-8")
    create_manifest(root)
    (root / "manuals" / "GUIDE_RU.md").unlink()

    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(
        {
            "CHATKODI_APP_ROOT": str(root),
            "CHATKODI_RUNTIME_DIR": str(root / "runtime"),
            "PYTHONIOENCODING": "utf-8",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(repo / "main.py"), "--verify-portable"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    report = json.loads(completed.stdout[completed.stdout.index("{"):])
    assert report["warnings"] == ["missing file: manuals/GUIDE_RU.md"]
    log = (root / "runtime" / "logs" / "portable_preflight.log").read_text(encoding="utf-8")
    assert "WARNING missing file: manuals/GUIDE_RU.md" in log


def test_portable_zip_matches_manifest(tmp_path: Path) -> None:
    import zipfile

    from tools.portable_verify import create_portable_zip, verify_portable_zip

    root = _write_portable_payload(tmp_path / "ChatKodi")
    (root / "manuals").mkdir()
    (root / "manuals" / "ИНСТРУКЦИЯ.md").write_text("кириллица", encoding="utf-8")
    report = create_portable_zip(root, tmp_path / "out.zip")
    assert report["ok"] is True, report
    with zipfile.ZipFile(tmp_path / "out.zip") as bundle:
        names = bundle.namelist()
    assert "ChatKodi/manuals/ИНСТРУКЦИЯ.md" in names
    assert not any(name.startswith("ChatKodi/docs/") and not name.endswith("/") for name in names)
    assert verify_portable_zip(tmp_path / "out.zip")["ok"] is True
