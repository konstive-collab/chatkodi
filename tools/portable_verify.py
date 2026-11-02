"""Create and verify the integrity manifest of a portable ChatKodi folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from core.model_integrity import inspect_gguf_model

MANIFEST_NAME = "portable_manifest.json"
MANIFEST_SCHEMA = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_mutable_or_user_file(relative_path: Path) -> bool:
    parts = tuple(part.lower() for part in relative_path.parts)
    return bool(parts and parts[0] in {"runtime", "docs"})


_CRITICAL_ROOT_FILES = {
    "chatkodiapp.exe",
    "chatkodilauncher.exe",
    "chatkodi_config.json",
    "version.txt",
    "portable.flag",
}
_CRITICAL_RUNTIME_DLLS = {
    "python311.dll",
    "python3.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "base_library.zip",
}

# Parts of a Hugging Face model folder that ChatKodi never reads (it loads
# safetensors weights through the torch backend): hub download bookkeeping,
# OpenVINO exports, weights for other frameworks and the PyTorch pickle copy of
# weights that also exist as safetensors. The downloader already skips most of
# them; older model caches still carry them.
_MODEL_JUNK_DIRS = {".cache", "openvino"}
_MODEL_JUNK_SUFFIXES = (".h5", ".msgpack", ".ot", ".tflite")


def model_payload_junk(model_dir: str | Path) -> list[Path]:
    """Return files of ``model_dir`` that are not needed to load the model."""
    root = Path(model_dir)
    has_safetensors = any(root.glob("*.safetensors"))
    junk: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = [part.lower() for part in path.relative_to(root).parts]
        name = parts[-1]
        if any(part in _MODEL_JUNK_DIRS for part in parts[:-1]):
            junk.append(path)
        elif name.endswith(_MODEL_JUNK_SUFFIXES):
            junk.append(path)
        elif has_safetensors and name.startswith("pytorch_model") and name.endswith((".bin", ".bin.index.json")):
            junk.append(path)
    return junk


def prune_model_payload(model_dir: str | Path) -> dict[str, Any]:
    """Delete :func:`model_payload_junk` files and the folders they leave empty."""
    root = Path(model_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {root}")
    removed: list[str] = []
    removed_bytes = 0
    for path in model_payload_junk(root):
        removed_bytes += path.stat().st_size
        path.unlink()
        removed.append(path.relative_to(root).as_posix())
    for directory in sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    return {
        "ok": True,
        "root": str(root),
        "removed_files": len(removed),
        "removed_bytes": removed_bytes,
        "removed": removed[:100],
    }


def _is_critical(relative_path: Path) -> bool:
    """Critical = without this file the app cannot start or cannot search.

    Name-based rules apply only to the root and the top of ``_internal``:
    many wheels ship private copies of the MSVC runtime in nested ``.libs``
    folders, and a missing duplicate must never block the start.
    """
    normalized = relative_path.as_posix().lower()
    parts = normalized.split("/")
    name = parts[-1]
    if normalized.startswith("models_cache/speech/"):
        return False  # voice is optional: a damaged file disables voice, not the app
    if normalized.startswith("models_cache/"):
        # Hub download bookkeeping is never read at runtime; its long nested
        # paths are the first to be skipped by an unzip tool hitting MAX_PATH.
        return not any(part in _MODEL_JUNK_DIRS for part in parts[:-1])
    if parts[0] == "llama.cpp":
        # llama-server and its CUDA/ggml DLLs run both models.
        return name.endswith((".exe", ".dll"))
    if len(parts) == 1:
        return normalized in _CRITICAL_ROOT_FILES
    if len(parts) == 2 and parts[0] == "_internal":
        return name in _CRITICAL_RUNTIME_DLLS
    return normalized.endswith("/platforms/qwindows.dll")


def _iter_payload_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.name == MANIFEST_NAME or _is_mutable_or_user_file(relative):
            continue
        yield path


def _canonical_records(records: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        records,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def create_manifest(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    for required in ("ChatKodiApp.exe", "ChatKodiLauncher.exe", "portable.flag"):
        if not (root_path / required).is_file():
            raise FileNotFoundError(f"portable payload is missing {required}: {root_path}")

    records: list[dict[str, Any]] = []
    for path in sorted(_iter_payload_files(root_path), key=lambda item: item.as_posix().lower()):
        relative = path.relative_to(root_path)
        critical = _is_critical(relative)
        record: dict[str, Any] = {
            "path": relative.as_posix(),
            "size": path.stat().st_size,
            "critical": critical,
        }
        if critical:
            record["sha256"] = _sha256(path)
        records.append(record)

    version_path = root_path / "VERSION.txt"
    version = version_path.read_text(encoding="utf-8").strip() if version_path.is_file() else "unknown"
    build_id = hashlib.sha256(_canonical_records(records)).hexdigest()
    payload = {
        "schema": MANIFEST_SCHEMA,
        "product": "ChatKodi",
        "version": version,
        "build_id": build_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_count": len(records),
        "critical_file_count": sum(1 for item in records if item.get("critical")),
        "total_bytes": sum(int(item["size"]) for item in records),
        "files": records,
    }
    manifest_path = root_path / MANIFEST_NAME
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, manifest_path)
    return payload


def _safe_manifest_path(root: Path, relative_name: str) -> Path | None:
    try:
        candidate = (root / Path(relative_name)).resolve()
        candidate.relative_to(root)
        return candidate
    except (OSError, ValueError):
        return None


def _verify_model_directories(root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    model_root = root / "models_cache"
    if not model_root.is_dir():
        return [{"ok": False, "path": str(model_root), "errors": ["models_cache is missing"]}]

    candidates: list[Path] = []
    for family in ("llms", "embeddings"):
        family_root = model_root / family
        if family_root.is_dir():
            candidates += sorted(family_root.rglob("*.gguf"))
    if not candidates:
        return [{"ok": False, "path": str(model_root), "errors": ["no GGUF models found"]}]
    for candidate in candidates:
        checks.append(inspect_gguf_model(candidate))
    return checks


def _component_for(relative_name: str) -> str:
    normalized = relative_name.replace("\\", "/").lower()
    if "models_cache/llms/" in normalized or normalized.endswith("models_cache/llms"):
        return "llm"
    if "models_cache/embeddings/" in normalized or normalized.endswith("models_cache/embeddings"):
        return "embedding"
    if "models_cache/speech/" in normalized:
        return "speech"
    if normalized.startswith("models_cache") or "/models_cache" in normalized:
        return "models"
    if normalized.startswith("llama.cpp"):
        return "embedding"  # without llama-server even search cannot start
    return "app"


def verify_portable(root: str | Path, *, full: bool = False, all_files: bool | None = None) -> dict[str, Any]:
    """Verify a portable folder against its manifest.

    ``errors`` lists failures of critical files (the app cannot work as
    shipped). ``warnings`` lists everything that is wrong but harmless for the
    start: missing documentation, a missing duplicate DLL in a wheel's private
    folder, an edited manifest. ``issues`` carries the same findings with a
    component tag so callers can decide what blocks the start.

    Quick mode checks presence and size of critical files only; ``all_files``
    (default: same as ``full``) extends that to every manifest file, and
    ``full`` also hashes the critical files.
    """
    if all_files is None:
        all_files = full
    root_path = Path(root).resolve()
    manifest_path = root_path / MANIFEST_NAME
    issues: list[dict[str, str]] = []
    checked_files = 0
    hashed_files = 0

    def add(severity: str, component: str, message: str) -> None:
        issues.append({"severity": severity, "component": component, "message": message})

    manifest: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        add("error", "manifest", f"cannot read {MANIFEST_NAME}: {exc}")
    else:
        raw_records = manifest.get("files")
        if manifest.get("schema") != MANIFEST_SCHEMA or not isinstance(raw_records, list):
            add("error", "manifest", "unsupported or invalid portable manifest schema")
        else:
            records = raw_records
            expected_build_id = hashlib.sha256(_canonical_records(records)).hexdigest()
            if expected_build_id != manifest.get("build_id"):
                add("warning", "manifest", "portable manifest build_id does not match its file table")

    selected = records if all_files else [item for item in records if item.get("critical")]
    for item in selected:
        relative_name = str(item.get("path") or "")
        critical = bool(item.get("critical"))
        severity = "error" if critical else "warning"
        component = _component_for(relative_name)
        path = _safe_manifest_path(root_path, relative_name)
        if path is None:
            add("error", component, f"unsafe manifest path: {relative_name}")
            continue
        checked_files += 1
        if not path.is_file():
            add(severity, component, f"missing file: {relative_name}")
            continue
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            add(severity, component, f"cannot stat {relative_name}: {exc}")
            continue
        expected_size = int(item.get("size") or 0)
        if actual_size != expected_size:
            add(
                severity,
                component,
                f"size mismatch: {relative_name} (expected {expected_size}, got {actual_size})",
            )
            continue
        expected_hash = str(item.get("sha256") or "")
        if full and expected_hash:
            hashed_files += 1
            actual_hash = _sha256(path)
            if actual_hash.lower() != expected_hash.lower():
                add(severity, component, f"SHA-256 mismatch: {relative_name}")

    model_checks = _verify_model_directories(root_path)
    for model_check in model_checks:
        if not model_check.get("ok"):
            details = "; ".join(str(item) for item in model_check.get("errors") or [])
            model_path = str(model_check.get("path") or "")
            add("error", _component_for(model_path), f"invalid model payload {model_path}: {details}")

    errors = [item["message"] for item in issues if item["severity"] == "error"]
    warnings = [item["message"] for item in issues if item["severity"] != "error"]
    return {
        "ok": not errors,
        "root": str(root_path),
        "mode": "full" if full else "quick",
        "all_files": all_files,
        "version": manifest.get("version"),
        "build_id": manifest.get("build_id"),
        "manifest_file_count": len(records),
        "checked_files": checked_files,
        "hashed_files": hashed_files,
        "model_checks": model_checks,
        "errors": errors[:100],
        "warnings": warnings[:100],
        "warning_count": len(warnings),
        "issues": issues[:200],
    }


# Components without which ChatKodi cannot do its job at all. Everything else
# (LLM payload, docs, duplicate DLLs, manifest bookkeeping) degrades gracefully:
# without the LLM the app still searches and answers from structured evidence.
_BLOCKING_COMPONENTS = {"app", "embedding", "models"}


def startup_preflight(root: str | Path) -> dict[str, Any]:
    """Quick start-time check that blocks only on truly fatal problems."""
    report = verify_portable(root, full=False)
    blocking: list[str] = []
    degraded: list[str] = []
    for issue in report.get("issues") or []:
        message = str(issue.get("message") or "")
        if issue.get("severity") == "error" and issue.get("component") in _BLOCKING_COMPONENTS:
            blocking.append(message)
        else:
            degraded.append(message)
    llm_degraded = any(
        issue.get("component") == "llm" and issue.get("severity") == "error"
        for issue in report.get("issues") or []
    )
    return {
        "ok": not blocking,
        "blocking_errors": blocking,
        "warnings": degraded,
        "llm_degraded": llm_degraded,
        "build_id": report.get("build_id"),
        "report": report,
    }


PREFLIGHT_LOG = Path("runtime") / "logs" / "portable_preflight.log"


def _append_preflight_log(root: str | Path, header: str, errors: list[str], warnings: list[str]) -> Path | None:
    try:
        path = Path(root) / PREFLIGHT_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"[{stamp}] {header}"]
        lines.extend(f"  ERROR   {item}" for item in errors[:50])
        lines.extend(f"  WARNING {item}" for item in warnings[:200])
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return path
    except OSError:
        return None


def write_preflight_log(root: str | Path, preflight: dict[str, Any]) -> Path | None:
    """Persist non-blocking findings so support can see them without a crash."""
    warnings = list(preflight.get("warnings") or [])
    blocking = list(preflight.get("blocking_errors") or [])
    if not warnings and not blocking:
        return None
    return _append_preflight_log(
        root,
        f"portable preflight: blocking={len(blocking)} warnings={len(warnings)}",
        blocking,
        warnings,
    )


def verify_and_log(root: str | Path, *, full: bool = False) -> dict[str, Any]:
    """Explicit check behind ``--verify-portable`` and VERIFY_CHATKODI.cmd.

    Unlike the start-time preflight it looks at every manifest file (missing
    manuals included) and always appends its result to the preflight log, so a
    clean re-check is visible after an earlier warning.
    """
    report = verify_portable(root, full=full, all_files=True)
    errors = list(report.get("errors") or [])
    status = "ok" if report.get("ok") else "FAILED"
    header = (
        f"verify-portable ({report['mode']}): {status} "
        f"checked={report['checked_files']} errors={len(errors)} warnings={report['warning_count']}"
    )
    log_path = _append_preflight_log(root, header, errors, list(report.get("warnings") or []))
    report["log_path"] = str(log_path) if log_path else None
    return report


def format_blocking_message(preflight: dict[str, Any]) -> str:
    errors = "\n".join(f"- {item}" for item in (preflight.get("blocking_errors") or [])[:8])
    return (
        "Portable-папка ChatKodi скопирована не полностью или повреждена.\n"
        f"{errors}\n\n"
        "Скопируйте папку заново целиком (лучше архивом) и запустите VERIFY_CHATKODI.cmd для полной проверки."
    )


def create_portable_zip(root: str | Path, zip_path: str | Path, *, folder_name: str | None = None) -> dict[str, Any]:
    """Pack the portable folder and prove the archive matches the manifest.

    The manifest is regenerated from the folder right before packing, so the
    archive, the folder and the manifest describe exactly the same payload.
    Names are stored as UTF-8 with forward slashes, which every unzip tool on
    Windows reads correctly (Compress-Archive in Windows PowerShell 5.1 does
    not guarantee either).
    """
    import zipfile

    root_path = Path(root).resolve()
    archive = Path(zip_path).resolve()
    prefix = (folder_name or root_path.name).strip("/\\")
    manifest = create_manifest(root_path)
    expected = {str(item["path"]): int(item["size"]) for item in manifest["files"]}

    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as bundle:
        for path in sorted(root_path.rglob("*"), key=lambda item: item.as_posix().lower()):
            relative = path.relative_to(root_path)
            if _is_mutable_or_user_file(relative) and path.is_file():
                continue
            arcname = f"{prefix}/{relative.as_posix()}"
            if path.is_dir():
                continue
            # Model weights are already compressed; storing them is much faster.
            compress = zipfile.ZIP_STORED if path.suffix.lower() in {".safetensors", ".bin", ".gguf", ".zip"} else zipfile.ZIP_DEFLATED
            bundle.write(path, arcname, compress_type=compress)
        for empty_dir in ("docs/", "runtime/", "runtime/logs/"):
            bundle.writestr(f"{prefix}/{empty_dir}", b"")
    os.replace(temporary, archive)

    report = verify_portable_zip(archive, expected=expected, folder_name=prefix)
    report["manifest_build_id"] = manifest.get("build_id")
    return report


def verify_portable_zip(
    zip_path: str | Path,
    *,
    expected: dict[str, int] | None = None,
    folder_name: str | None = None,
) -> dict[str, Any]:
    """Check that every manifest file is inside the archive with its size."""
    import zipfile

    archive = Path(zip_path)
    errors: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        infos = [info for info in bundle.infolist() if not info.is_dir()]
        names: dict[str, int] = {}
        detected_prefix = folder_name
        for info in infos:
            name = info.filename.replace("\\", "/")
            if detected_prefix is None and "/" in name:
                detected_prefix = name.split("/", 1)[0]
            names[name] = info.file_size
        prefix = f"{detected_prefix}/" if detected_prefix else ""
        if expected is None:
            manifest_name = f"{prefix}{MANIFEST_NAME}"
            if manifest_name not in names:
                return {"ok": False, "zip": str(archive), "errors": [f"{MANIFEST_NAME} is not in the archive"]}
            manifest = json.loads(bundle.read(manifest_name).decode("utf-8"))
            expected = {str(item["path"]): int(item["size"]) for item in manifest.get("files") or []}
        lowered: dict[str, str] = {}
        for name in names:
            key = name.lower()
            if key in lowered:
                errors.append(f"case-insensitive duplicate in archive: {name} / {lowered[key]}")
            lowered[key] = name
        for relative, size in expected.items():
            arcname = f"{prefix}{relative}"
            if arcname not in names:
                errors.append(f"missing in archive: {relative}")
            elif names[arcname] != size:
                errors.append(f"size mismatch in archive: {relative} (expected {size}, got {names[arcname]})")
    return {
        "ok": not errors,
        "zip": str(archive),
        "zip_bytes": archive.stat().st_size if archive.exists() else 0,
        "expected_files": len(expected or {}),
        "errors": errors[:100],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or verify a ChatKodi portable manifest")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-manifest", metavar="ROOT")
    action.add_argument("--verify", metavar="ROOT")
    action.add_argument("--make-zip", nargs=2, metavar=("ROOT", "ZIP"))
    action.add_argument("--verify-zip", metavar="ZIP")
    action.add_argument("--prune-model-payload", metavar="MODEL_DIR", help="delete model files ChatKodi never loads")
    parser.add_argument("--full", action="store_true", help="verify every file size and all stored hashes")
    args = parser.parse_args()

    try:
        if args.prune_model_payload:
            output = prune_model_payload(args.prune_model_payload)
        elif args.write_manifest:
            result = create_manifest(args.write_manifest)
            output = {
                "ok": True,
                "root": str(Path(args.write_manifest).resolve()),
                "manifest": str(Path(args.write_manifest).resolve() / MANIFEST_NAME),
                "version": result.get("version"),
                "build_id": result.get("build_id"),
                "file_count": result.get("file_count"),
                "critical_file_count": result.get("critical_file_count"),
                "total_bytes": result.get("total_bytes"),
            }
        elif args.make_zip:
            output = create_portable_zip(args.make_zip[0], args.make_zip[1])
        elif args.verify_zip:
            output = verify_portable_zip(args.verify_zip)
        else:
            output = verify_portable(args.verify, full=bool(args.full))
    except Exception as exc:
        output = {"ok": False, "error": str(exc)}

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
