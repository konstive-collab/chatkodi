"""Pre-flight memory checks and crash memory for local LLM loading.

Two failure modes motivated this module:

* On Windows ``safetensors`` maps every shard copy-on-write. Such a view is
  charged against the *commit limit* (RAM + pagefile) in full, so a 4 GB shard
  needs 4 GB of free commit even on a machine with plenty of physical RAM.
  When the pagefile is small, Windows answers with ``os error 1455``; inside
  the GUI this often surfaces as a hard segfault instead of an exception.
* A segfault kills the process before ``resilience`` can record the incident,
  so every next start repeats the same crash.

The guard therefore (1) estimates the commit the load needs and refuses to
start a load that cannot succeed, and (2) leaves an "in-flight" marker on disk
for the duration of the load. A marker that survives into the next start means
the previous load died hard; the next start skips the automatic load until the
memory situation actually changes.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core.logging_utils import get_logger

logger = get_logger(__name__)

_GB = 1024 ** 3
_INFLIGHT_NAME = "llm_load_inflight.json"
_CRASH_NAME = "llm_load_crash.json"
# Extra commit needed on top of the weights: tokenizer, CUDA context, Python
# heap growth and the KV cache of the first generation.
_COMMIT_OVERHEAD_GB = 1.2
# How much free commit must grow after a hard crash before an automatic retry.
_RETRY_COMMIT_GAIN_GB = 2.0


@dataclass
class CommitCheck:
    ok: bool
    needed_gb: float
    free_gb: float
    largest_shard_gb: float
    weights_gb: float
    message: str = ""


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on", "да"}


def free_commit_gb() -> float:
    """Free commit charge (RAM + pagefile) in GB; 0.0 when unknown."""
    try:
        from core.system_profile import _get_available_memory_gb

        _ram, commit = _get_available_memory_gb()
        return float(commit or 0.0)
    except Exception:
        return 0.0


def _weight_files(model_path: str | Path) -> list[Path]:
    root = Path(model_path)
    files: list[Path] = []
    for pattern in ("*.safetensors", "*.bin", "*.gguf"):
        files.extend(item for item in root.glob(pattern) if item.is_file())
    return files


def estimate_load_commit_gb(
    model_path: str | Path,
    *,
    device: str,
    gpu_budget_gb: float = 0.0,
) -> tuple[float, float, float]:
    """Return ``(needed_gb, largest_shard_gb, weights_gb)`` for loading a model.

    CUDA: one mapped shard at a time plus the layers that do not fit on the GPU
    and stay resident in RAM. CPU: the whole model lives in RAM.
    """
    sizes = [item.stat().st_size / _GB for item in _weight_files(model_path)]
    weights = float(sum(sizes))
    largest = float(max(sizes) if sizes else 0.0)
    if device == "cuda":
        cpu_resident = max(0.0, weights - max(0.0, float(gpu_budget_gb)))
        needed = largest + cpu_resident + _COMMIT_OVERHEAD_GB
    else:
        needed = weights + _COMMIT_OVERHEAD_GB
    return round(needed, 2), round(largest, 2), round(weights, 2)


def pagefile_advice(needed_gb: float, free_gb: float, model_name: str) -> str:
    return (
        f"Не хватает виртуальной памяти Windows для загрузки модели {model_name}: "
        f"нужно около {needed_gb:.1f} ГБ свободной памяти (ОЗУ + файл подкачки), "
        f"сейчас свободно {free_gb:.1f} ГБ.\n"
        "Как исправить: Панель управления → Система → Дополнительные параметры системы → "
        "Быстродействие → Параметры → Дополнительно → Виртуальная память → "
        "включите «Размер по выбору системы» (или задайте не меньше 16 ГБ) и перезагрузите ПК. "
        "Либо выберите в лаунчере модель поменьше.\n"
        "Поиск по документам продолжает работать и без LLM."
    )


def check_commit_for_load(
    model_path: str | Path,
    *,
    device: str,
    gpu_budget_gb: float = 0.0,
    model_name: str = "",
) -> CommitCheck:
    needed, largest, weights = estimate_load_commit_gb(model_path, device=device, gpu_budget_gb=gpu_budget_gb)
    free = free_commit_gb()
    if free <= 0.0 or weights <= 0.0:
        # Unknown platform/metric or no weights: do not block on a guess.
        return CommitCheck(True, needed, free, largest, weights)
    ok = free >= needed
    message = "" if ok else pagefile_advice(needed, free, model_name or Path(model_path).name)
    return CommitCheck(ok, needed, free, largest, weights, message)


# --------------------------------------------------------------------------- #
# Crash memory
# --------------------------------------------------------------------------- #


def _state_dir() -> Path:
    try:
        from core.paths import runtime_root

        base = runtime_root() / "logs"
    except Exception:
        base = Path(".chatkodi") / "logs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(pid))
    except Exception:
        return False


class LoadSentinel:
    """Marks a model load as in-flight; removed only when the load returns."""

    def __init__(self, model_path: str | Path, *, free_commit: float, needed_commit: float) -> None:
        self.path = _state_dir() / _INFLIGHT_NAME
        self.payload = {
            "model_path": str(model_path),
            "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "free_commit_gb": round(float(free_commit), 2),
            "needed_commit_gb": round(float(needed_commit), 2),
        }

    def __enter__(self) -> "LoadSentinel":
        try:
            self.path.write_text(json.dumps(self.payload, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            logger.debug("Cannot write LLM load sentinel: %s", exc)
        return self

    def __exit__(self, *_exc: Any) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


def collect_previous_crash() -> Optional[dict[str, Any]]:
    """Turn a stale in-flight marker into a recorded crash and return it."""
    state = _state_dir()
    inflight = state / _INFLIGHT_NAME
    crash_path = state / _CRASH_NAME
    marker = _read_json(inflight) if inflight.exists() else None
    if marker and not _pid_alive(int(marker.get("pid") or 0)):
        previous = _read_json(crash_path) or {}
        same_model = previous.get("model_path") == marker.get("model_path")
        marker["crash_count"] = int(previous.get("crash_count") or 0) + 1 if same_model else 1
        marker["detected"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            crash_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
            inflight.unlink()
        except OSError:
            pass
        logger.error(
            "Previous LLM load did not finish (process crashed): model=%s free_commit=%.1fGB needed=%.1fGB",
            marker.get("model_path"),
            float(marker.get("free_commit_gb") or 0.0),
            float(marker.get("needed_commit_gb") or 0.0),
        )
    return _read_json(crash_path) if crash_path.exists() else None


def crash_blocks_autoload(model_path: str | Path, free_commit_now: float) -> Optional[str]:
    """Return a user message when the previous hard crash should block a retry."""
    if _env_flag("CHATKODI_FORCE_LLM"):
        return None
    crash = collect_previous_crash()
    if not crash or crash.get("model_path") != str(model_path):
        return None
    before = float(crash.get("free_commit_gb") or 0.0)
    needed = float(crash.get("needed_commit_gb") or 0.0)
    if free_commit_now > 0 and (free_commit_now >= before + _RETRY_COMMIT_GAIN_GB or (needed and free_commit_now >= needed + 1.0)):
        logger.info("Free commit grew from %.1fGB to %.1fGB since the crash; retrying LLM load", before, free_commit_now)
        return None
    return (
        "Прошлая попытка загрузить LLM аварийно завершила программу "
        f"(свободно было {before:.1f} ГБ виртуальной памяти, сейчас {free_commit_now:.1f} ГБ). "
        "Автозагрузка модели отключена, чтобы программа не падала снова. "
        "Увеличьте файл подкачки Windows или выберите модель поменьше — "
        "после этого загрузка включится автоматически.\n"
        "Поиск по документам работает и без LLM."
    )


def clear_crash_record() -> None:
    try:
        (_state_dir() / _CRASH_NAME).unlink()
    except OSError:
        pass
