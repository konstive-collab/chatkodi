"""Local LLM on llama.cpp (``llama-server``) with GGUF models.

The public interface (``LocalLLM``, ``load_llm_model``, status helpers) is the
same as in the torch/transformers era, so the chat, search and GUI code do not
depend on the backend.
"""

import os
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from core.gpu_info import primary_gpu
from core.logging_utils import get_logger
from core.llm import prompts
from core.llm.model_select import describe_model, find_llm, find_mmproj
from core.llm.server import LlamaServer, LlamaServerError

logger = get_logger(__name__)

_LAST_LLM_LOAD_STATUS: Dict[str, Any] = {
    "available": False,
    "reason": "not_attempted",
    "detail": "",
}

DEFAULT_CTX_TOKENS = 16384


def classify_llm_unavailability(error: BaseException) -> Tuple[str, str]:
    message = str(error or "")
    lowered = message.lower()
    if (
        getattr(error, "winerror", None) == 1455
        or "winerror 1455" in lowered
        or "os error 1455" in lowered
        or "paging file is too small" in lowered
        or "файл подкачки слишком мал" in lowered
        or "pagefile" in lowered
    ):
        return (
            "insufficient_commit_memory",
            "Не хватает виртуальной памяти Windows (WinError 1455): файл подкачки (pagefile) слишком мал "
            "для загрузки модели. Включите «Размер по выбору системы» для файла подкачки или выберите модель поменьше. "
            "Поиск по документам работает и без LLM.",
        )
    if "out of memory" in lowered and ("cuda" in lowered or "vram" in lowered or "device" in lowered):
        return "insufficient_vram", "GPU memory is insufficient; select a smaller model or use CPU mode."
    if isinstance(error, FileNotFoundError):
        if "llama-server" in lowered:
            return "dependency_missing", message
        return "model_not_found", message
    if "failed to load model" in lowered or "invalid magic" in lowered or "corrupted" in lowered:
        return "model_incomplete", message
    if "out of memory" in lowered or "cannot allocate memory" in lowered or "failed to allocate" in lowered:
        return "insufficient_ram", message
    return "load_failed", message or type(error).__name__


_PLACEMENT_WARNING: Dict[str, Any] = {}


def get_placement_warning() -> str:
    """User-facing note when the model does not fit into VRAM (slow answers)."""
    if not _PLACEMENT_WARNING:
        return ""
    data = _PLACEMENT_WARNING
    return (
        f"Модель {data['model_gb']} ГБ не помещается в видеопамять (доступно ~{data['vram_budget_gb']} ГБ), "
        "часть слоёв работает на CPU. Ответы LLM будут медленнее. "
        "Точные ответы по документам (даты, суммы, госномера, подсчёты) от этого не зависят."
    )


def _llm_force_flag() -> bool:
    return str(os.environ.get("CHATKODI_FORCE_LLM", "")).strip().lower() in {"1", "true", "yes", "on", "да"}


_USER_REASON_TEXT = {
    "user_disabled": "LLM отключена в настройках лаунчера.",
    "model_not_found": "Локальная LLM не найдена. Скачайте модель в лаунчере.",
    "model_incomplete": "Файл LLM повреждён или скачан не полностью. Перекачайте модель в лаунчере.",
    "dependency_missing": "Не найден llama.cpp (llama-server). Переустановите программу.",
    "insufficient_vram": "Видеопамяти недостаточно для выбранной модели. Выберите модель поменьше или режим CPU.",
    "insufficient_ram": "Оперативной памяти недостаточно для выбранной модели. Выберите модель поменьше.",
    "payload_damaged": "Файлы LLM в portable-папке повреждены. Скопируйте папку заново. Поиск работает без LLM.",
}


def describe_llm_status(status: Optional[Dict[str, Any]] = None) -> str:
    """Human-readable (Russian) reason why the LLM is not available."""
    status = status or get_last_llm_load_status()
    if status.get("available"):
        return ""
    reason = str(status.get("reason") or "")
    detail = str(status.get("detail") or "")
    if reason == "insufficient_commit_memory" and detail:
        return detail
    if reason in _USER_REASON_TEXT:
        return _USER_REASON_TEXT[reason]
    if reason == "not_attempted":
        return ""
    return f"LLM не загрузилась: {detail or reason}"


def get_last_llm_load_status() -> Dict[str, Any]:
    return dict(_LAST_LLM_LOAD_STATUS)


class StreamingCallback:
    """Класс для обработки streaming-ответов в реальном времени."""

    def __init__(self):
        self.queue = Queue()
        self.finished = False
        self.full_text = ""

    def __call__(self, token_text: str, is_final: bool = False):
        if token_text:
            self.full_text += token_text
            self.queue.put(("token", token_text))
        if is_final:
            self.finished = True
            self.queue.put(("finish", self.full_text))

    def error(self, error_msg: str):
        self.finished = True
        self.queue.put(("error", error_msg))

    def get_stream(self) -> Generator[str, None, None]:
        while not self.finished or not self.queue.empty():
            try:
                msg_type, data = self.queue.get(timeout=0.1)
                if msg_type == "token":
                    yield data
                elif msg_type == "finish":
                    break
                elif msg_type == "error":
                    raise Exception(data)
            except Empty:
                if self.finished:
                    break
                continue


def _cache_prompt_enabled() -> bool:
    return str(os.environ.get("CHATKODI_LLM_CACHE_PROMPT", "1")).strip().lower() not in {"0", "false", "no", "off"}


def _server_args(device: str, mmproj: Optional[Path] = None) -> List[str]:
    ctx = int(os.environ.get("CHATKODI_LLM_CTX", DEFAULT_CTX_TOKENS) or DEFAULT_CTX_TOKENS)
    # -lv 4 logs "offloaded N/M layers to GPU" (read by _check_placement).
    args = ["-c", str(ctx), "-np", "1", "--jinja", "--reasoning", "off", "-lv", "4"]
    if mmproj is not None:
        # The vision projector lets the same model read photos and scans.
        args += ["--mmproj", str(mmproj)]
        if device != "cuda":
            args += ["--no-mmproj-offload"]
    if device == "cuda":
        # --fit: llama.cpp offloads as many layers as fit into the free VRAM.
        args += ["-fa", "on", "-fit", "on", "-fitc", "8192"]
    else:
        args += ["-ngl", "0"]
    return args


class LocalLLM:
    """Локальная LLM: GGUF-модель в фоновом llama-server."""

    supports_json_schema = True

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = "auto",
                 cache_dir: Optional[str] = None,
                 max_memory: Dict = None):
        self.cache_dir = cache_dir
        self.max_memory = max_memory
        self.prompt_cache: Dict[str, str] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.stats = {
            "total_queries": 0,
            "rag_queries": 0,
            "chat_queries": 0,
            "successful": 0,
            "failed": 0,
            "avg_response_time": 0,
            "full_documents_queries": 0,
        }
        gpu = primary_gpu()
        self.gpu_name = gpu.name if gpu else ""
        self.device = ("cuda" if gpu else "cpu") if device == "auto" else device

        path = find_llm(model_path)
        if path is None:
            raise FileNotFoundError(
                "LLM модель не найдена: положите *.gguf в models_cache/llms/ "
                "или укажите CHATKODI_MODEL_PATH"
            )
        self.model_path = str(path)
        self.mmproj_path = find_mmproj(path)
        self.model_info = describe_model(path)
        self.quantization = str(self.model_info.get("quantization") or "")
        self.server: Optional[LlamaServer] = None
        self.model_loaded = False
        self.availability_reason = "not_loaded"
        self.last_load_error = ""
        self._generation_cancel_event = threading.Event()
        self._external_generation_cancel_event: Optional[threading.Event] = None
        self._generation_lock = threading.RLock()
        logger.info(
            "Initializing local LLM: model=%s size=%s device=%s path=%s",
            self.model_info["name"], self.model_info["size"], self.device, self.model_path,
        )

    # ---------------- load / unload ---------------- #

    def load_model(self) -> bool:
        started = time.perf_counter()
        try:
            self.server = LlamaServer("llm", Path(self.model_path), _server_args(self.device, self.mmproj_path))
            self.server.start()
            self._check_placement()
            self.model_loaded = True
            self.availability_reason = "ready"
            self.last_load_error = ""
            logger.info(
                "Local LLM loaded: model=%s quant=%s device=%s load_seconds=%.2f",
                self.model_info["name"], self.quantization, self.device, time.perf_counter() - started,
            )
            return True
        except Exception as exc:
            logger.exception("Failed to load local LLM from %s: %s", self.model_path, exc)
            if self.server is not None:
                self.server.stop()
            self.server = None
            self.model_loaded = False
            self.availability_reason, self.last_load_error = classify_llm_unavailability(exc)
            return False

    def _check_placement(self) -> None:
        """Record a warning when --fit left some layers on the CPU."""
        _PLACEMENT_WARNING.clear()
        if self.device != "cuda" or self.server is None:
            return
        log = ""
        try:
            log = self.server.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        import re

        match = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
        if match and int(match.group(1)) < int(match.group(2)):
            gpu = primary_gpu()
            _PLACEMENT_WARNING.update({
                "model_gb": self.model_info.get("total_size_gb", 0),
                "vram_budget_gb": round(gpu.total_gb, 1) if gpu else 0,
                "cpu_layers": int(match.group(2)) - int(match.group(1)),
                "total_layers": int(match.group(2)),
            })
            logger.warning("LLM partially on CPU: %s of %s layers on GPU", match.group(1), match.group(2))

    def unload(self) -> None:
        if self.server is not None:
            self.server.stop()
        self.server = None
        self.model_loaded = False

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

    # ---------------- generation ---------------- #

    def generate_with_callback(self,
                               messages: List[Dict[str, str]],
                               callback: Callable[[str, bool], None],
                               max_tokens: int = 800,
                               temperature: float = 0.3,
                               top_p: float = 0.9,
                               context_limit: int = 32000,
                               deadline_seconds: Optional[float] = None,
                               json_schema: Optional[Dict[str, Any]] = None,
                               token_logprobs: Optional[List[Dict[str, Any]]] = None) -> str:
        """Streams tokens into ``callback``; ``deadline_seconds`` caps the wall time.

        ``json_schema`` constrains decoding (llama.cpp grammar): the output is
        always valid JSON matching the schema. ``token_logprobs`` (a list) is
        filled with every generated token, its log-probability and the top
        alternatives — how sure the model was of each character it read.
        """
        if not self.is_available():
            callback("Модель не загружена", True)
            return ""
        self.stats["total_queries"] += 1
        self._generation_cancel_event.clear()
        cancel_events = [self._generation_cancel_event]
        if self._external_generation_cancel_event is not None:
            cancel_events.append(self._external_generation_cancel_event)
        if any(e.is_set() for e in cancel_events):
            callback("", True)
            return ""
        deadline = time.monotonic() + float(deadline_seconds) if deadline_seconds else None
        payload = {
            "messages": _fit_messages(messages, context_limit),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repeat_penalty": 1.05,
            "seed": 42,
            "cache_prompt": _cache_prompt_enabled(),
        }
        if json_schema:
            payload["response_format"] = {"type": "json_object", "schema": json_schema}
        if token_logprobs is not None:
            payload["logprobs"] = True
            payload["top_logprobs"] = 4
        started = time.time()
        full_response = ""
        try:
            with self._generation_lock:
                stream = self.server.stream("/v1/chat/completions", payload)
                try:
                    for event in stream:
                        if any(e.is_set() for e in cancel_events):
                            break
                        if deadline is not None and time.monotonic() >= deadline:
                            break
                        choices = event.get("choices") or []
                        delta = (choices[0].get("delta") or {}).get("content") if choices else None
                        if token_logprobs is not None and choices:
                            token_logprobs.extend((choices[0].get("logprobs") or {}).get("content") or [])
                        if delta:
                            full_response += delta
                            callback(delta, False)
                finally:
                    stream.close()
            elapsed = time.time() - started
            done = self.stats["successful"]
            self.stats["avg_response_time"] = (self.stats["avg_response_time"] * done + elapsed) / (done + 1)
            self.stats["successful"] += 1
            callback("", True)
            return full_response
        except Exception as exc:
            logger.exception("Local LLM generation failed")
            self.stats["failed"] += 1
            error_msg = f"Ошибка генерации: {str(exc)[:100]}"
            callback(error_msg, True)
            return error_msg

    @property
    def supports_vision(self) -> bool:
        return self.mmproj_path is not None and self.model_loaded

    def cancel_generation(self) -> None:
        self._generation_cancel_event.set()

    def attach_generation_cancel_event(self, event: threading.Event) -> None:
        self._external_generation_cancel_event = event

    def detach_generation_cancel_event(self, event: threading.Event) -> None:
        if self._external_generation_cancel_event is event:
            self._external_generation_cancel_event = None

    def generate_streaming(self,
                           messages: List[Dict[str, str]],
                           max_tokens: int = 800,
                           temperature: float = 0.3,
                           top_p: float = 0.9,
                           context_limit: int = 32000) -> StreamingCallback:
        callback = StreamingCallback()

        def worker():
            try:
                self.generate_with_callback(
                    messages=messages, callback=callback, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, context_limit=context_limit,
                )
            except Exception as exc:
                callback.error(str(exc))

        threading.Thread(target=worker, daemon=True).start()
        return callback

    # ---------------- answers ---------------- #

    def _get_runtime_profile(self, rag_mode: bool = False, use_full_documents: bool = False) -> Dict[str, Any]:
        """Generation settings. Answers from documents are greedy (temperature 0):
        accuracy over variety. Context limits are in characters (~3 per token)."""
        if rag_mode and use_full_documents:
            return {"max_tokens": 700, "temperature": 0.0, "top_p": 1.0, "context_limit": 32000}
        if rag_mode:
            return {"max_tokens": 400, "temperature": 0.0, "top_p": 1.0, "context_limit": 24000}
        return {"max_tokens": 400, "temperature": 0.5, "top_p": 0.92, "context_limit": 8000}

    def _build_prompt(self, question: str, context: str, rag_mode: bool, use_full_documents: bool) -> str:
        if rag_mode and context:
            if use_full_documents:
                return prompts.rag_prompt_full_documents(question, context)
            return prompts.rag_prompt_fact_only(question, context)
        return prompts.chat_prompt(question)

    def generate_clean_answer(self,
                              question: str,
                              context: str = "",
                              rag_mode: bool = False,
                              streaming_callback: Union[bool, Callable] = None,
                              use_cache: bool = True,
                              use_full_documents: bool = False) -> Union[str, StreamingCallback]:
        if not self.is_available():
            return "Модель не загружена"
        if rag_mode:
            self.stats["rag_queries"] += 1
            if use_full_documents:
                self.stats["full_documents_queries"] += 1
        else:
            self.stats["chat_queries"] += 1

        profile = self._get_runtime_profile(rag_mode=rag_mode, use_full_documents=use_full_documents)
        trimmed = prompts.trim_context(context, profile["context_limit"])
        prompt = self._build_prompt(question, trimmed, rag_mode, use_full_documents)
        settings = {k: profile[k] for k in ("max_tokens", "temperature", "top_p", "context_limit")}

        if streaming_callback is True:
            return self.generate_streaming(messages=[{"role": "user", "content": prompt}], **settings)

        cache_key = f"{question}_{hash(trimmed)}_{rag_mode}_{use_full_documents}_{profile['max_tokens']}"
        if use_cache and cache_key in self.prompt_cache:
            self.cache_hits += 1
            cached = self.prompt_cache[cache_key]
            if callable(streaming_callback):
                for i in range(0, len(cached), 8):
                    streaming_callback(cached[i:i + 8], False)
                streaming_callback("", True)
            return cached
        self.cache_misses += 1

        def run(prompt_text: str, temperature: float, sink: Optional[Callable] = None) -> str:
            parts: List[str] = []

            def collect(token: str, is_final: bool) -> None:
                if token:
                    parts.append(token)
                if sink is not None:
                    sink(token, is_final)

            self.generate_with_callback(
                messages=[{"role": "user", "content": prompt_text}], callback=collect,
                max_tokens=profile["max_tokens"], temperature=temperature,
                top_p=profile["top_p"], context_limit=profile["context_limit"],
            )
            raw = "".join(parts)
            if use_full_documents:
                return prompts.clean_rag_response(raw, question, trimmed)
            return prompts.clean_response(raw, question)

        sink = streaming_callback if callable(streaming_callback) else None
        cleaned = run(prompt, profile["temperature"], sink)
        if rag_mode and trimmed and prompts.looks_like_rag_refusal(cleaned):
            logger.warning("Detected spurious RAG refusal; retrying with stricter fact-only prompt")
            retry = run(prompts.rag_prompt_fact_only(question, trimmed), 0.0)
            if retry and not prompts.looks_like_rag_refusal(retry):
                cleaned = retry

        if cleaned and "ошибка" not in cleaned.lower():
            self.prompt_cache[cache_key] = cleaned
            if len(self.prompt_cache) > 50:
                del self.prompt_cache[next(iter(self.prompt_cache))]
        return cleaned

    def generate_from_full_documents(self,
                                     question: str,
                                     documents_context: str,
                                     max_tokens: int = 1500,
                                     temperature: float = 0.2) -> str:
        if not self.is_available():
            return "Модель не загружена"
        profile = self._get_runtime_profile(rag_mode=True, use_full_documents=True)
        context = prompts.trim_context(documents_context, profile["context_limit"])
        logger.info("Using full-documents context: %s chars", len(context))
        try:
            response = self.generate_with_callback(
                messages=[{"role": "user", "content": prompts.rag_prompt_full_documents(question, context)}],
                callback=lambda token, final: None,
                max_tokens=min(int(max_tokens), int(profile["max_tokens"])),
                temperature=min(float(temperature), float(profile["temperature"])),
                top_p=profile["top_p"],
                context_limit=profile["context_limit"],
            )
            return prompts.clean_rag_response(response, question, context)
        except Exception as exc:
            logger.warning("Full-documents generation failed: %s", exc)
            return f"Ошибка анализа документов: {exc}"

    def quick_test(self, question: str = "Сколько будет 2+2?") -> str:
        return self.generate_clean_answer(question, rag_mode=False, use_cache=False)

    def rag_test(self, context: str, question: str) -> str:
        return self.generate_clean_answer(question, context=context, rag_mode=True, use_cache=False)

    # ---------------- info ---------------- #

    def is_available(self) -> bool:
        if not (self.model_loaded and self.server is not None):
            return False
        if self.server.is_running():
            return True
        return self._restart_crashed_server()

    def _restart_crashed_server(self) -> bool:
        """Bring a crashed llama-server back (at most 3 times, once a minute)."""
        now = time.monotonic()
        attempts = getattr(self, "_restart_times", [])
        if len(attempts) >= 3 or (attempts and now - attempts[-1] < 60):
            return False
        attempts.append(now)
        self._restart_times = attempts
        logger.warning("llama-server[llm] is not running (exit code %s); restarting", getattr(self.server.process, "returncode", None))
        try:
            self.server.process = None
            self.server.start()
            logger.info("llama-server[llm] restarted in %.1f s", self.server.load_seconds)
            return True
        except Exception as exc:
            logger.error("llama-server[llm] restart failed: %s", exc)
            self.availability_reason, self.last_load_error = classify_llm_unavailability(exc)
            return False

    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "name": self.model_info.get("name", "Unknown"),
            "size": self.model_info.get("size", "Unknown"),
            "parameters": self.model_info.get("parameters", "Unknown"),
            "path": self.model_path,
            "device": self.device,
            "backend": "llama.cpp",
            "loaded": self.model_loaded,
            "availability_reason": self.availability_reason,
            "last_load_error": self.last_load_error,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.prompt_cache),
            "total_size": self.model_info.get("total_size", 0),
            "total_size_gb": self.model_info.get("total_size_gb", 0),
            "quantization": self.quantization,
            "load_seconds": round(self.server.load_seconds, 2) if self.server else 0.0,
            "stats": self.stats,
        }
        if self.gpu_name:
            info["gpu"] = self.gpu_name
        return info

    def clear_cache(self):
        self.prompt_cache.clear()
        logger.info("Prompt cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()


def _fit_messages(messages: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    """Trim the longest message (the documents) so the total fits ``limit`` chars.

    Unlike cutting the formatted prompt, the question at the end survives.
    """
    msgs: List[Dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        # Multimodal content (text + images) is passed through as is.
        msgs.append({"role": m.get("role", "user"), "content": content if isinstance(content, list) else str(content or "")})
    texts = [m for m in msgs if isinstance(m["content"], str)]
    total = sum(len(m["content"]) for m in texts)
    if not limit or total <= limit or not texts:
        return msgs
    longest = max(texts, key=lambda m: len(m["content"]))
    keep = max(1000, len(longest["content"]) - (total - limit))
    text = longest["content"]
    head = int(keep * 0.75)
    longest["content"] = text[:head] + "\n…\n" + text[len(text) - (keep - head):]
    return msgs


def load_llm_model(lightweight: bool = False, max_memory: Dict = None) -> Optional[LocalLLM]:
    """Find the GGUF model and start llama-server. Returns None if unavailable."""
    started = time.perf_counter()
    global _LAST_LLM_LOAD_STATUS
    try:
        if str(os.environ.get("CHATKODI_DISABLE_LLM", "")).strip().lower() in {"1", "true", "yes", "on"}:
            logger.info("Local LLM loading disabled by CHATKODI_DISABLE_LLM")
            _LAST_LLM_LOAD_STATUS = {"available": False, "reason": "user_disabled", "detail": "CHATKODI_DISABLE_LLM"}
            return None
        if str(os.environ.get("CHATKODI_LLM_PAYLOAD_DAMAGED", "")).strip() == "1" and not _llm_force_flag():
            logger.warning("Local LLM loading skipped: portable preflight found a damaged LLM payload")
            _LAST_LLM_LOAD_STATUS = {"available": False, "reason": "payload_damaged", "detail": "portable preflight"}
            return None

        llm = LocalLLM(device="auto", max_memory=max_memory)
        if llm.load_model():
            info = llm.get_model_info()
            logger.info(
                "Local LLM is ready: model=%s quant=%s device=%s path=%s total_seconds=%.2f",
                info["name"], info["quantization"], info["device"], info["path"], time.perf_counter() - started,
            )
            _LAST_LLM_LOAD_STATUS = {"available": True, "reason": "ready", "detail": info["path"]}
            return llm
        logger.warning("Local LLM could not be loaded")
        _LAST_LLM_LOAD_STATUS = {"available": False, "reason": llm.availability_reason, "detail": llm.last_load_error}
        return None
    except Exception as exc:
        logger.exception("Critical error while preparing local LLM: %s", exc)
        reason, detail = classify_llm_unavailability(exc)
        _LAST_LLM_LOAD_STATUS = {"available": False, "reason": reason, "detail": detail}
        return None
