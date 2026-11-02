"""NVIDIA GPU detection without torch (nvidia-smi)."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional


@dataclass(frozen=True)
class GpuInfo:
    name: str
    total_mib: int
    free_mib: int
    driver: str

    @property
    def total_gb(self) -> float:
        return self.total_mib / 1024.0

    @property
    def free_gb(self) -> float:
        return self.free_mib / 1024.0


def _nvidia_smi() -> Optional[str]:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    candidate = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
    return candidate if os.path.isfile(candidate) else None


def query_gpus() -> List[GpuInfo]:
    """Current NVIDIA GPUs (fresh free-memory numbers on every call).

    ``CHATKODI_FORCE_CPU=1`` hides the GPU (models then run on the CPU).
    """
    if os.environ.get("CHATKODI_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}:
        return []
    exe = _nvidia_smi()
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=0x08000000 if os.name == "nt" else 0,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(GpuInfo(parts[0], int(float(parts[1])), int(float(parts[2])), parts[3]))
        except ValueError:
            continue
    return gpus


@lru_cache(maxsize=1)
def primary_gpu() -> Optional[GpuInfo]:
    """First GPU, cached (name/total memory do not change during a run)."""
    gpus = query_gpus()
    return gpus[0] if gpus else None


def cuda_available() -> bool:
    return primary_gpu() is not None


def free_vram_gb() -> float:
    gpus = query_gpus()
    return gpus[0].free_gb if gpus else 0.0
