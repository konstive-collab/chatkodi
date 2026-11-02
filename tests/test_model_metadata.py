import json
from pathlib import Path

from tools.model_manager.downloader import SmartModelDownloader


def _write_embedding_model(path: Path) -> None:
    path.mkdir(parents=True)
    metadata = json.dumps({"padding": "x" * 80})
    (path / "config.json").write_text(metadata, encoding="utf-8")
    (path / "tokenizer.json").write_text(metadata, encoding="utf-8")
    (path / "sentence_bert_config.json").write_text(metadata, encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"w" * 2048)


def test_read_only_model_probe_preserves_portable_metadata(tmp_path: Path) -> None:
    cache = tmp_path / "models_cache"
    model = cache / "embeddings" / "sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    _write_embedding_model(model)
    metadata = {
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
            "path": "models_cache/embeddings/sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2",
            "model_type": "embedding",
        }
    }
    metadata_path = cache / "models_meta.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    before = metadata_path.read_bytes()

    downloader = SmartModelDownloader(cache_dir=str(cache), offline_mode=True, read_only=True)
    available, resolved = downloader.is_model_available(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    assert available is True
    assert Path(resolved).resolve() == model.resolve()
    assert metadata_path.read_bytes() == before


def test_saved_model_choice_uses_cache_relative_path(tmp_path: Path) -> None:
    cache = tmp_path / "models_cache"
    model = cache / "llms" / "generic-model"
    model.mkdir(parents=True)
    downloader = SmartModelDownloader(cache_dir=str(cache))

    assert downloader.save_model_choice("generic/model", str(model), {}) is True
    saved = json.loads((cache / "model_choice.json").read_text(encoding="utf-8"))
    assert saved["model_path"] == str(Path("llms") / "generic-model")
