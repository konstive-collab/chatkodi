"""Fast structural checks for local Hugging Face model directories."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


MODEL_HASH_MANIFEST = ".chatkodi-sha256.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_model_hash_manifest(model_path: str | Path) -> dict[str, Any]:
    """Create a local cryptographic baseline after a complete download."""
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files: dict[str, dict[str, Any]] = {}
    excluded = {MODEL_HASH_MANIFEST, "offline_config.json"}
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        if item.name in excluded or item.suffix in {".tmp", ".part", ".incomplete"}:
            continue
        files[relative] = {
            "sha256": _sha256_file(item),
            "size": item.stat().st_size,
        }
    payload = {
        "algorithm": "sha256",
        "files": files,
    }
    manifest_path = path / MODEL_HASH_MANIFEST
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return payload


def verify_model_hash_manifest(model_path: str | Path) -> dict[str, Any]:
    """Verify files against ChatKodi's post-download SHA-256 manifest."""
    path = Path(model_path)
    manifest_path = path / MODEL_HASH_MANIFEST
    if not manifest_path.is_file():
        return {
            "ok": False,
            "manifest_present": False,
            "errors": ["missing SHA-256 manifest"],
            "checked_files": 0,
        }
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ok": False,
            "manifest_present": True,
            "errors": [f"cannot read SHA-256 manifest: {exc}"],
            "checked_files": 0,
        }
    files = payload.get("files")
    if payload.get("algorithm") != "sha256" or not isinstance(files, dict) or not files:
        return {
            "ok": False,
            "manifest_present": True,
            "errors": ["invalid SHA-256 manifest"],
            "checked_files": 0,
        }
    errors: list[str] = []
    checked = 0
    for relative, expected in files.items():
        try:
            candidate = (path / str(relative)).resolve()
            candidate.relative_to(path.resolve())
        except (OSError, ValueError):
            errors.append(f"unsafe hashed path: {relative}")
            continue
        if not candidate.is_file():
            errors.append(f"missing hashed file: {relative}")
            continue
        try:
            if not isinstance(expected, dict):
                raise ValueError("file record is not an object")
            expected_size = int(expected.get("size", -1))
            expected_hash = str(expected.get("sha256", "")).lower()
            if len(expected_hash) != 64 or any(ch not in "0123456789abcdef" for ch in expected_hash):
                raise ValueError("invalid SHA-256 value")
            if candidate.stat().st_size != expected_size:
                errors.append(f"size mismatch: {relative}")
                continue
            if _sha256_file(candidate) != expected_hash:
                errors.append(f"SHA-256 mismatch: {relative}")
                continue
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"invalid hash record for {relative}: {exc}")
            continue
        checked += 1
    return {
        "ok": not errors,
        "manifest_present": True,
        "errors": errors,
        "checked_files": checked,
    }


def inspect_hf_model_payload(
    model_path: str | Path,
    *,
    require_tokenizer: bool = True,
) -> dict[str, Any]:
    """Inspect model metadata and shard references without hashing large files."""
    path = Path(model_path)
    errors: list[str] = []
    warnings: list[str] = []
    referenced_weights: set[Path] = set()

    if not path.is_dir():
        return {
            "ok": False,
            "path": str(path),
            "errors": ["model directory does not exist"],
            "warnings": [],
            "weight_files": 0,
            "total_weight_bytes": 0,
        }

    has_config = (path / "config.json").is_file() or (path / "modules.json").is_file()
    if not has_config:
        errors.append("missing config.json/modules.json")

    tokenizer_names = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "spiece.model",
        "sentencepiece.bpe.model",
    )
    has_tokenizer = any(any(path.rglob(name)) for name in tokenizer_names)
    if require_tokenizer and not has_tokenizer:
        errors.append("missing tokenizer files")

    index_files = list(path.rglob("*.safetensors.index.json"))
    index_files.extend(path.rglob("pytorch_model*.bin.index.json"))
    for index_path in index_files:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                errors.append(f"invalid or empty weight_map in {index_path.name}")
                continue
            for relative_name in weight_map.values():
                candidate = index_path.parent / str(relative_name)
                referenced_weights.add(candidate)
        except Exception as exc:
            errors.append(f"cannot read {index_path.name}: {exc}")

    if referenced_weights:
        weight_files = sorted(referenced_weights)
    else:
        weight_files = sorted(
            {
                *path.rglob("*.safetensors"),
                *path.rglob("pytorch_model*.bin"),
            }
        )

    if not weight_files:
        errors.append("model weight files are missing")

    total_weight_bytes = 0
    missing_weights: list[str] = []
    empty_weights: list[str] = []
    for weight_path in weight_files:
        try:
            if not weight_path.is_file():
                missing_weights.append(str(weight_path.relative_to(path)))
                continue
            size = weight_path.stat().st_size
            if size < 1024:
                empty_weights.append(str(weight_path.relative_to(path)))
            total_weight_bytes += size
        except Exception as exc:
            missing_weights.append(f"{weight_path.name} ({exc})")

    if missing_weights:
        preview = ", ".join(missing_weights[:8])
        errors.append(f"missing referenced weight shards: {preview}")
    if empty_weights:
        preview = ", ".join(empty_weights[:8])
        errors.append(f"empty or truncated weight files: {preview}")

    partial_files = [
        item
        for pattern in ("*.incomplete", "*.part", "*.tmp")
        for item in path.rglob(pattern)
        if item.is_file()
    ]
    if partial_files:
        warnings.append(
            "partial download artifacts present: "
            + ", ".join(str(item.relative_to(path)) for item in partial_files[:8])
        )

    critical_metadata = [path / "config.json", path / "tokenizer.json"]
    for metadata_path in critical_metadata:
        if metadata_path.exists():
            try:
                if metadata_path.stat().st_size < 2:
                    errors.append(f"empty metadata file: {metadata_path.name}")
            except OSError as exc:
                errors.append(f"cannot read metadata {metadata_path.name}: {exc}")

    return {
        "ok": not errors,
        "path": str(path),
        "errors": errors,
        "warnings": warnings,
        "index_files": [item.name for item in index_files],
        "weight_files": len(weight_files),
        "total_weight_bytes": total_weight_bytes,
    }


def inspect_gguf_model(model_path: str | Path) -> dict[str, Any]:
    """Header check of a GGUF file: magic, version, tensor and metadata counts.

    Catches missing, empty, truncated-to-header and non-GGUF files without
    reading the weights (the full check is the SHA-256 in the portable manifest).
    """
    import struct

    path = Path(model_path)
    errors: list[str] = []
    info: dict[str, Any] = {"ok": False, "path": str(path), "errors": errors}
    if not path.is_file():
        errors.append("missing GGUF file")
        return info
    size = path.stat().st_size
    info["size_bytes"] = size
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError as exc:
        errors.append(f"cannot read: {exc}")
        return info
    if len(header) < 24 or header[:4] != b"GGUF":
        errors.append("not a GGUF file (bad magic)")
        return info
    version, tensors, kv_count = struct.unpack("<IQQ", header[4:24])
    info.update({"version": version, "tensor_count": tensors, "metadata_count": kv_count})
    if version < 2 or version > 10:
        errors.append(f"unsupported GGUF version {version}")
    if tensors == 0 or kv_count == 0:
        errors.append("GGUF header has no tensors or metadata")
    if size < 1024 * 1024:
        errors.append("GGUF file is too small to hold weights")
    info["ok"] = not errors
    return info


__all__ = [
    "MODEL_HASH_MANIFEST",
    "create_model_hash_manifest",
    "inspect_gguf_model",
    "inspect_hf_model_payload",
    "verify_model_hash_manifest",
]
