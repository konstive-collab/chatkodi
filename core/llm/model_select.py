"""Find GGUF models in the model caches and pick the LLM for this machine."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from core.gpu_info import primary_gpu
from core.logging_utils import get_logger
from core.paths import model_cache_search_dirs

logger = get_logger(__name__)

GGUF_MAGIC = b"GGUF"
# Measured on RTX 4060 8 GB: KV cache + compute buffers at 16k context ~0.9 GB,
# Windows desktop ~1.3 GB. Without this room Windows silently spills VRAM to
# system RAM and generation drops ~6x (9B: 68 -> 12 tok/s).
VRAM_HEADROOM_GB = 2.2
# Reserved for the embedding server that runs next to the LLM (measured ~1.5 GB).
EMBEDDING_RESERVE_GB = 1.5
# Auto mode prefers these when they fit: on the ChatKodi eval Qwen3.5-4B was
# more accurate than 9B (22/22 vs 16/22; 9B answered "not found" on negatives).
PREFERRED_LLM_PREFIXES = ("qwen3.5-4b",)


def file_size_gb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 3)


def is_gguf(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == GGUF_MAGIC
    except OSError:
        return False


def _is_split_tail(name: str) -> bool:
    match = re.search(r"-(\d{5})-of-\d{5}\.gguf$", name)
    return bool(match and match.group(1) != "00001")


def list_gguf(subdir: str) -> List[Path]:
    """All ``*.gguf`` files under ``<cache>/<subdir>`` (no projectors, no split tails)."""
    found: List[Path] = []
    seen = set()
    for base in model_cache_search_dirs():
        root = base / subdir
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.gguf")):
            low = path.name.lower()
            if low.startswith("mmproj") or _is_split_tail(low):
                continue
            key = str(path.resolve()).lower()
            if key in seen or not is_gguf(path):
                continue
            seen.add(key)
            found.append(path)
    return found


def model_size_label(path: Path) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", path.stem)
    return f"{match.group(1)}B" if match else "Unknown"


def quant_label(path: Path) -> str:
    match = re.search(r"((?:UD-)?I?Q\d[\w]*|BF16|F16|F32)$", path.stem, flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def describe_model(path: Path) -> Dict[str, object]:
    size_bytes = path.stat().st_size if path.exists() else 0
    quant = quant_label(path)
    name = path.stem
    if quant and name.upper().endswith(quant):
        name = name[: -len(quant)].rstrip("-_.")
    return {
        "name": name,
        "size": model_size_label(path),
        "parameters": model_size_label(path),
        "format": "gguf",
        "quantization": quant,
        "total_size": size_bytes,
        "total_size_gb": round(size_bytes / (1024 ** 3), 2),
    }


def choose_llm(candidates: List[Path], *, embedding_on_gpu: bool = True) -> Optional[Path]:
    """A preferred (measured most accurate) model that fits into VRAM, else the
    largest one that fits; on CPU the smallest one."""
    if not candidates:
        return None
    by_size = sorted(candidates, key=lambda p: (file_size_gb(p), str(p).lower()))
    gpu = primary_gpu()
    if gpu is None:
        return by_size[0]
    budget = gpu.total_gb - VRAM_HEADROOM_GB - (EMBEDDING_RESERVE_GB if embedding_on_gpu else 0.0)
    fitting = [p for p in by_size if file_size_gb(p) <= budget]
    preferred = [p for p in fitting if p.name.lower().startswith(PREFERRED_LLM_PREFIXES)]
    choice = (preferred or fitting or by_size)[-1 if (preferred or fitting) else 0]
    logger.info(
        "LLM choice: %s (%.1f GB) for %s %.1f GB VRAM, budget %.1f GB, candidates=%s",
        choice.name, file_size_gb(choice), gpu.name, gpu.total_gb, budget,
        [p.name for p in by_size],
    )
    return choice


def find_llm(manual_path: Optional[str] = None) -> Optional[Path]:
    """CHATKODI_MODEL_PATH (a .gguf file or a folder with one) wins over auto choice."""
    manual = manual_path or os.environ.get("CHATKODI_MODEL_PATH") or ""
    if manual:
        path = Path(manual)
        if path.is_file() and is_gguf(path):
            return path
        if path.is_dir():
            inner = [p for p in sorted(path.glob("*.gguf")) if is_gguf(p) and not p.name.lower().startswith("mmproj")]
            if inner:
                return inner[0]
        logger.warning("Manual LLM path is not a GGUF model: %s", manual)
    return choose_llm(list_gguf("llms"), embedding_on_gpu=bool(list_gguf("embeddings")))


def find_mmproj(llm_path: Path) -> Optional[Path]:
    """Vision projector for ``llm_path``: CHATKODI_MMPROJ_PATH, else
    ``mmproj-<model base>*.gguf`` next to the model (e.g. mmproj-Qwen3.5-4B-F16.gguf).
    ``CHATKODI_DISABLE_VISION=1`` turns images off."""
    if str(os.environ.get("CHATKODI_DISABLE_VISION", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return None
    override = os.environ.get("CHATKODI_MMPROJ_PATH", "")
    if override and Path(override).is_file() and is_gguf(Path(override)):
        return Path(override)
    base = describe_model(llm_path)["name"]
    base = re.sub(r"[-_.](?:UD-)?$", "", str(base)).lower()
    for path in sorted(llm_path.parent.glob("mmproj*.gguf")):
        if base and base in path.name.lower() and is_gguf(path):
            return path
    return None


def find_embedding_model() -> Optional[Path]:
    override = os.environ.get("CHATKODI_EMBEDDING_MODEL_PATH", "")
    if override and Path(override).is_file() and is_gguf(Path(override)):
        return Path(override)
    models = list_gguf("embeddings")
    return models[0] if models else None
