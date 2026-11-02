"""Manage a local ``llama-server`` (llama.cpp) process.

ChatKodi runs two servers: one for the chat LLM and one for embeddings. Both
listen on 127.0.0.1 on a free port and are bound to a Windows Job Object, so
they die together with ChatKodi even after a crash.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from core.logging_utils import get_logger
from core.paths import model_cache_search_dirs, project_root, resolve_project_path

logger = get_logger(__name__)

LLAMA_SERVER_ENV = "CHATKODI_LLAMA_SERVER"
_EXE_NAME = "llama-server.exe" if os.name == "nt" else "llama-server"


def find_llama_server() -> Optional[Path]:
    """Locate llama-server: env override, app folder, then model caches."""
    override = os.environ.get(LLAMA_SERVER_ENV, "").strip()
    if override and Path(override).is_file():
        return Path(override)
    roots = [project_root() / "llama.cpp"]
    if getattr(sys, "frozen", False):
        roots.insert(0, Path(sys.executable).resolve().parent / "llama.cpp")
    roots += [base / "llama.cpp" for base in model_cache_search_dirs()]
    for root in roots:
        if not root.is_dir():
            continue
        found = sorted(root.rglob(_EXE_NAME), key=lambda p: str(p).lower(), reverse=True)
        if found:
            return found[0]
    return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


_JOB_HANDLE = None
_JOB_LOCK = threading.Lock()


def _assign_to_kill_on_close_job(process: subprocess.Popen) -> None:
    """Kill the child when ChatKodi exits (Windows only, best effort)."""
    if os.name != "nt":
        return
    global _JOB_HANDLE
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        with _JOB_LOCK:
            if _JOB_HANDLE is None:
                job = kernel32.CreateJobObjectW(None, None)
                if not job:
                    return

                class _BasicLimits(ctypes.Structure):
                    _fields_ = [
                        ("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD),
                    ]

                class _IoCounters(ctypes.Structure):
                    _fields_ = [(name, ctypes.c_uint64) for name in (
                        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
                    )]

                class _ExtendedLimits(ctypes.Structure):
                    _fields_ = [
                        ("BasicLimitInformation", _BasicLimits),
                        ("IoInfo", _IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t),
                    ]

                info = _ExtendedLimits()
                info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
                    kernel32.CloseHandle(job)
                    return
                _JOB_HANDLE = job
            kernel32.AssignProcessToJobObject(_JOB_HANDLE, wintypes.HANDLE(int(process._handle)))  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("Could not bind llama-server to a job object: %s", exc)


class LlamaServerError(RuntimeError):
    pass


class LlamaServer:
    """One llama-server process plus a tiny JSON/SSE client for it."""

    def __init__(self, name: str, model_path: Path, args: List[str], binary: Optional[Path] = None) -> None:
        self.name = name
        self.model_path = Path(model_path)
        self.args = list(args)
        self.binary = Path(binary) if binary else find_llama_server()
        self.port = 0
        self.process: Optional[subprocess.Popen] = None
        self.log_path = resolve_project_path(f"logs/llama_server_{name}.log")
        self.load_seconds = 0.0

    # ---------------- lifecycle ---------------- #

    def start(self, timeout: float = 180.0) -> None:
        if self.is_running():
            return
        if self.binary is None or not self.binary.is_file():
            raise FileNotFoundError("llama-server не найден: положите llama.cpp в папку llama.cpp рядом с программой")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        self.port = _free_port()
        cmd = [
            str(self.binary), "-m", str(self.model_path),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--no-webui", *self.args,
        ]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(self.log_path, "w", encoding="utf-8", errors="replace")
        creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        logger.info("Starting llama-server[%s]: %s", self.name, " ".join(cmd))
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=str(self.binary.parent),
                creationflags=creationflags,
            )
        finally:
            log_file.close()
        _assign_to_kill_on_close_job(self.process)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise LlamaServerError(
                    f"llama-server[{self.name}] завершился с кодом {self.process.returncode}: {self.log_tail()}"
                )
            if self._health_ok():
                self.load_seconds = time.perf_counter() - started
                logger.info("llama-server[%s] ready on port %s in %.1f s", self.name, self.port, self.load_seconds)
                return
            time.sleep(0.2)
        self.stop()
        raise LlamaServerError(f"llama-server[{self.name}] не ответил за {timeout:.0f} с: {self.log_tail()}")

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def log_tail(self, lines: int = 12) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return " | ".join(line.strip() for line in text[-lines:] if line.strip())

    # ---------------- HTTP ---------------- #

    def _connection(self, timeout: float) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)

    def _health_ok(self) -> bool:
        try:
            conn = self._connection(2.0)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return resp.status == 200
        except OSError:
            return False

    def post_json(self, path: str, payload: Dict[str, Any], timeout: float = 600.0) -> Dict[str, Any]:
        conn = self._connection(timeout)
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status != 200:
                raise LlamaServerError(f"{path}: HTTP {resp.status}: {data[:300].decode('utf-8', 'replace')}")
            return json.loads(data.decode("utf-8"))
        finally:
            conn.close()

    def stream(self, path: str, payload: Dict[str, Any], timeout: float = 600.0) -> "SseStream":
        return SseStream(self._connection(timeout), path, payload)


class SseStream:
    """Server-sent events from llama-server; ``close()`` aborts generation."""

    def __init__(self, conn: http.client.HTTPConnection, path: str, payload: Dict[str, Any]) -> None:
        self.conn = conn
        body = json.dumps({**payload, "stream": True}, ensure_ascii=False).encode("utf-8")
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        self.resp = conn.getresponse()
        if self.resp.status != 200:
            data = self.resp.read()
            conn.close()
            raise LlamaServerError(f"{path}: HTTP {self.resp.status}: {data[:300].decode('utf-8', 'replace')}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        while True:
            line = self.resp.readline()
            if not line:
                return
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                return
            try:
                yield json.loads(data.decode("utf-8"))
            except ValueError:
                continue

    def close(self) -> None:
        # Dropping the connection makes llama-server stop generating.
        try:
            self.conn.close()
        except Exception:
            pass
