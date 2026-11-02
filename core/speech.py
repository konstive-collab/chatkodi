"""Offline speech: recognition with whisper.cpp (whisper-server on the GPU)
and synthesis with Piper. Nothing leaves the computer.

Layout in a model cache (``models/`` next to the program or the dev cache):
    speech/whisper/**/whisper-server.exe   (CUDA DLLs are shared with llama.cpp)
    speech/ggml-*.bin                      (whisper model, large-v3-turbo preferred)
    speech/piper/**/piper.exe
    speech/voices/<voice>.onnx (+ .onnx.json)
"""

from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

from core.llm.server import _assign_to_kill_on_close_job, _free_port, find_llama_server
from core.logging_utils import get_logger
from core.paths import model_cache_search_dirs, project_root, resolve_project_path

logger = get_logger(__name__)

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _speech_roots() -> List[Path]:
    roots = [project_root() / "speech"]
    if getattr(sys, "frozen", False):
        roots.insert(0, Path(sys.executable).resolve().parent / "speech")
    roots += [base / "speech" for base in model_cache_search_dirs()]
    return [r for r in roots if r.is_dir()]


def _find(pattern: str) -> Optional[Path]:
    for root in _speech_roots():
        found = sorted(root.rglob(pattern), key=lambda p: str(p).lower())
        if found:
            return found[0]
    return None


def find_whisper_server() -> Optional[Path]:
    override = os.environ.get("CHATKODI_WHISPER_SERVER", "")
    if override and Path(override).is_file():
        return Path(override)
    return _find("whisper-server.exe" if os.name == "nt" else "whisper-server")


def find_whisper_model() -> Optional[Path]:
    override = os.environ.get("CHATKODI_WHISPER_MODEL", "")
    if override and Path(override).is_file():
        return Path(override)
    models = []
    for root in _speech_roots():
        models += sorted(root.glob("ggml-*.bin"))
    # large-v3-turbo: best Russian quality per second on a mid GPU
    models.sort(key=lambda p: (0 if "turbo" in p.name else 1 if "large" in p.name else 2, p.name))
    return models[0] if models else None


def find_piper() -> Optional[Path]:
    return _find("piper.exe" if os.name == "nt" else "piper")


def list_voices() -> List[Path]:
    voices: List[Path] = []
    for root in _speech_roots():
        voices += [p for p in sorted((root / "voices").glob("*.onnx")) if p.with_suffix(".onnx.json").is_file()]
    return voices


_CODE = re.compile(r"(?<![\w-])[А-ЯЁ]{1,5}-?\d{1,5}[А-ЯЁ]?(?![\w-])")
_LATIN_CODE = re.compile(r"(?<![\w-])[A-Za-z]{1,5}[- ]?\d{1,5}[A-Za-z]?(?![\w-])")
_TRANSLIT = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "E", "Ж": "ZH", "З": "Z", "И": "I", "Й": "I",
    "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U", "Ф": "F",
    "Х": "H", "Ц": "C", "Ч": "CH", "Ш": "SH", "Щ": "SH", "Ы": "Y", "Э": "E", "Ю": "YU", "Я": "YA",
}


def _code_key(token: str) -> str:
    return "".join(_TRANSLIT.get(ch, ch) for ch in token.upper() if ch.isalnum())


def code_vocabulary(texts, limit: int = 60) -> List[str]:
    """Letter-number codes used in the user's documents («ДД-10», «ЕП-25»), most frequent first."""
    counts: dict = {}
    for text in texts:
        for token in _CODE.findall(text or ""):
            counts[token] = counts.get(token, 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]


def restore_codes(text: str, vocabulary: List[str]) -> str:
    """«DD-10» heard for a code the documents write as «ДД-10» -> the document spelling."""
    if not vocabulary:
        return text
    by_key = {}
    for token in vocabulary:
        by_key.setdefault(_code_key(token), token)
    return _LATIN_CODE.sub(lambda m: by_key.get(_code_key(m.group(0)), m.group(0)), text)


class SpeechRecognizer:
    """whisper-server kept running after the first use (model load ~2 s)."""

    def __init__(self, language: str = "ru"):
        self.vocabulary: List[str] = []
        self.binary = find_whisper_server()
        self.model = find_whisper_model()
        self.language = language
        self.port = 0
        self.process: Optional[subprocess.Popen] = None
        self.log_path = resolve_project_path("logs/whisper_server.log")
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.binary and self.model)

    def unavailable_reason(self) -> str:
        if not self.binary:
            return "Не найден whisper.cpp (папка speech/whisper)."
        if not self.model:
            return "Не найдена модель распознавания речи (speech/ggml-*.bin)."
        return ""

    def start(self, timeout: float = 120.0) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        if not self.available:
            raise FileNotFoundError(self.unavailable_reason())
        self.port = _free_port()
        env = dict(os.environ)
        llama = find_llama_server()
        if llama is not None:
            # cuBLAS / cudart are shared with llama.cpp instead of a second copy
            env["PATH"] = str(llama.parent) + os.pathsep + env.get("PATH", "")
        threads = str(max(2, min(8, (os.cpu_count() or 4) // 2)))
        cmd = [str(self.binary), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
               "-l", self.language, "-t", threads]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8", errors="replace") as log_file:
            self.process = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                cwd=str(self.binary.parent), env=env, creationflags=_CREATE_NO_WINDOW,
            )
        _assign_to_kill_on_close_job(self.process)
        started = time.perf_counter()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"whisper-server завершился с кодом {self.process.returncode}")
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                conn.request("GET", "/")
                conn.getresponse().read()
                conn.close()
                logger.info("whisper-server ready on port %s in %.1f s", self.port, time.perf_counter() - started)
                return
            except OSError:
                time.sleep(0.2)
        self.stop()
        raise RuntimeError("whisper-server не запустился вовремя")

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def transcribe_file(self, path: Path) -> str:
        with self._lock:
            self.start()
            boundary = uuid.uuid4().hex
            data = Path(path).read_bytes()
            parts = []
            fields = [("response_format", "json"), ("temperature", "0.0"), ("language", self.language)]
            if self.language == "ru":
                # keeps Russian letter codes in Cyrillic («ДД-10», not «DD10»), as in the documents
                codes = ", ".join(self.vocabulary[:12]) or "АБ-10, ВК-2"
                fields.append(("prompt", f"Запрос на русском языке. Обозначения пишутся кириллицей: {codes}."))
            for name, value in fields:
                parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{Path(path).name}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n"
            )
            parts.append(f"--{boundary}--\r\n".encode())
            body = b"".join(parts)
            started = time.perf_counter()
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
            try:
                conn.request("POST", "/inference", body=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
                resp = conn.getresponse()
                raw = resp.read()
            finally:
                conn.close()
            if resp.status != 200:
                raise RuntimeError(f"Распознавание речи: HTTP {resp.status}: {raw[:200]!r}")
            text = json.loads(raw.decode("utf-8")).get("text", "")
            text = restore_codes(re.sub(r"\s+", " ", text).strip(), self.vocabulary)
            logger.info("Speech recognized: %.1f s, %s chars", time.perf_counter() - started, len(text))
            return text


class SpeechSynthesizer:
    """Piper text-to-speech into a WAV file."""

    def __init__(self, voice: str = ""):
        self.binary = find_piper()
        voices = list_voices()
        chosen = [v for v in voices if voice and voice in v.name]
        self.voice = (chosen or voices or [None])[0]

    @property
    def available(self) -> bool:
        return bool(self.binary and self.voice)

    def synthesize(self, text: str, out_path: Path) -> Path:
        if not self.available:
            raise FileNotFoundError("Синтез речи недоступен: нет Piper или голоса (папка speech).")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [str(self.binary), "-m", str(self.voice), "-f", str(out_path)],
            input=text.encode("utf-8"), capture_output=True, timeout=300,
            cwd=str(self.binary.parent), creationflags=_CREATE_NO_WINDOW,
        )
        if proc.returncode != 0 or not out_path.is_file():
            raise RuntimeError(f"Piper: {proc.stderr[-300:].decode('utf-8', 'replace')}")
        return out_path


def speakable(text: str, limit: int = 900) -> str:
    """Answer text for reading aloud: no Markdown, tables, code or file lists."""
    lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("|", "```", "📎", "📚", "Основание", "Расчёт", "⚠", "✓")):
            continue
        stripped = re.sub(r"[*_#`>]+", "", stripped)
        stripped = re.sub(r"\s+", " ", stripped).strip(" -•")
        if stripped:
            lines.append(stripped)
    spoken = " ".join(lines)
    if len(spoken) > limit:
        cut = spoken[:limit]
        spoken = cut[: cut.rfind(".") + 1] or cut
    return spoken
