"""Самовосстановление, деградация и ленивые прокси для тяжёлых компонентов."""

from __future__ import annotations

import json
import threading
import time
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Mapping

from core.logging_utils import get_logger
from core.paths import ensure_parent
from core.telemetry import TelemetryLogger
from core.system_profile import SystemProfile


logger = get_logger(__name__)


@dataclass(frozen=True)
class ResiliencePolicy:
    enabled: bool = True
    safe_mode_enabled: bool = True
    continue_on_optional_stage_failure: bool = True
    llm_startup_mode: str = "auto"
    record_incidents: bool = True
    incident_dir: str = ".chatkodi/logs/recovery"
    auto_disable_llm_after_failure: bool = True
    auto_apply_runtime_hints: bool = True
    max_llm_failures: int = 2

    @classmethod
    def from_config_data(cls, data: Mapping[str, Any] | None) -> "ResiliencePolicy":
        data = data or {}
        raw = data.get("resilience", {}) if isinstance(data, Mapping) else {}
        return cls(
            enabled=bool(raw.get("enabled", True)),
            safe_mode_enabled=bool(raw.get("safe_mode_enabled", True)),
            continue_on_optional_stage_failure=bool(raw.get("continue_on_optional_stage_failure", True)),
            llm_startup_mode=str(raw.get("llm_startup_mode", "auto") or "auto").lower(),
            record_incidents=bool(raw.get("record_incidents", True)),
            incident_dir=str(raw.get("incident_dir", ".chatkodi/logs/recovery")),
            auto_disable_llm_after_failure=bool(raw.get("auto_disable_llm_after_failure", True)),
            auto_apply_runtime_hints=bool(raw.get("auto_apply_runtime_hints", True)),
            max_llm_failures=max(1, int(raw.get("max_llm_failures", 2) or 2)),
        )


class RecoveryManager:
    def __init__(self, policy: ResiliencePolicy, profile: SystemProfile, telemetry: TelemetryLogger | None = None) -> None:
        self.policy = policy
        self.profile = profile
        self.telemetry = telemetry
        self.incidents: list[dict[str, Any]] = []
        self.safe_mode_active = False

    @classmethod
    def from_config_data(cls, data: Mapping[str, Any] | None, profile: SystemProfile) -> "RecoveryManager":
        telemetry = TelemetryLogger.from_config_data(data)
        return cls(ResiliencePolicy.from_config_data(data), profile, telemetry=telemetry)

    def _incident_path(self) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return ensure_parent(Path(self.policy.incident_dir) / f"incident_{stamp}.jsonl")

    def _record_incident(self, payload: Mapping[str, Any]) -> None:
        body = dict(payload)
        body.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.incidents.append(body)
        if self.policy.record_incidents:
            try:
                path = self._incident_path()
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(body, ensure_ascii=False) + "\n")
            except Exception:
                pass
        if self.telemetry:
            try:
                self.telemetry.event("recovery_incident", body)
            except Exception:
                pass

    def startup_event(self) -> None:
        payload = {
            "profile": self.profile.as_dict(),
            "policy": asdict(self.policy),
        }
        logger.info("Runtime profile active: %s", self.profile.summary_line)
        if self.telemetry:
            try:
                self.telemetry.event("runtime_profile", payload)
            except Exception:
                pass

    def note(self, kind: str, message: str, **extra: Any) -> None:
        payload = {"kind": kind, "message": message}
        if extra:
            payload.update(extra)
        logger.info("%s: %s", kind, message)
        if self.telemetry:
            try:
                self.telemetry.event(kind, payload)
            except Exception:
                pass

    def activate_safe_mode(self, reason: str, **extra: Any) -> None:
        self.safe_mode_active = True
        payload = {"kind": "safe_mode", "reason": reason}
        if extra:
            payload.update(extra)
        logger.warning("Safe mode activated: %s", reason)
        self._record_incident(payload)

    def run_stage(self, stage_name: str, func: Callable[[], Any], *, required: bool = False) -> Any:
        started = time.perf_counter()
        try:
            result = func()
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            if self.telemetry:
                try:
                    self.telemetry.event("startup_stage_ok", {"stage": stage_name, "elapsed_ms": elapsed_ms})
                except Exception:
                    pass
            return result
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            payload = {
                "kind": "startup_stage_failed",
                "stage": stage_name,
                "required": bool(required),
                "elapsed_ms": elapsed_ms,
                "error": str(exc),
            }
            logger.warning("Startup stage failed: %s | %s", stage_name, exc)
            self._record_incident(payload)
            if required and not self.policy.continue_on_optional_stage_failure:
                raise
            if self.policy.safe_mode_enabled:
                self.safe_mode_active = True
            return None

    def choose_llm_startup_mode(self) -> str:
        disabled_llm = str(os.environ.get("CHATKODI_DISABLE_LLM", "")).strip().lower() in {"1", "true", "yes", "on"}
        if disabled_llm:
            return "disabled"
        explicit_mode = str(os.environ.get("CHATKODI_LLM_STARTUP_MODE", "")).strip().lower()
        if explicit_mode in {"eager", "lazy", "disabled"}:
            return explicit_mode

        # Явное разрешение тяжёлой модели не должно блокировать открытие окна
        # на CPU-машине. На CUDA загрузка остаётся ранней, на CPU — ленивой.
        forced_llm = str(os.environ.get("CHATKODI_FORCE_LLM", "")).strip().lower() in {"1", "true", "yes", "on", "да"}
        if forced_llm:
            return "eager" if self.profile.has_cuda else "lazy"
        mode = self.policy.llm_startup_mode
        if mode in {"eager", "lazy", "disabled"}:
            return mode
        return self.profile.llm_startup_mode

    def should_use_lightweight_llm(self) -> bool:
        return bool(self.profile.prefer_lightweight_llm)

    def build_lazy_llm(self, loader: Callable[[], Any], model_label: str = "local_llm") -> "LazyLLMProxy":
        return LazyLLMProxy(loader, recovery=self, model_label=model_label)


class LazyLLMProxy:
    """Ленивая оболочка вокруг LLM, сохраняющая функциональность без тяжёлого старта."""

    def __init__(self, loader: Callable[[], Any], recovery: RecoveryManager | None = None, model_label: str = "local_llm") -> None:
        self._loader = loader
        self._recovery = recovery
        self._model_label = model_label
        self._lock = threading.Lock()
        self._model: Any = None
        self._disabled = False
        self._last_error: str | None = None
        self._failures = 0

    def __bool__(self) -> bool:
        return not self._disabled

    def is_loaded(self) -> bool:
        return self._model is not None

    def is_available(self) -> bool:
        if self._disabled:
            return False
        if self._model is not None:
            try:
                return bool(self._model.is_available())
            except Exception:
                return True
        return True

    def disable(self, reason: str) -> None:
        self._disabled = True
        self._last_error = reason
        if self._recovery:
            self._recovery.activate_safe_mode(reason=reason, component=self._model_label)

    def _ensure_loaded(self) -> Any:
        if self._disabled:
            return None
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if self._disabled:
                return None
            try:
                if self._recovery:
                    self._recovery.note("llm_lazy_load", f"Loading {self._model_label} on demand")
                model = self._loader()
                if model is None:
                    raise RuntimeError("loader returned None")
                self._model = model
                return self._model
            except Exception as exc:
                self._last_error = str(exc)
                self._failures += 1
                if self._recovery:
                    self._recovery._record_incident({
                        "kind": "llm_lazy_load_failed",
                        "component": self._model_label,
                        "error": str(exc),
                        "failures": self._failures,
                    })
                    if self._recovery.policy.auto_disable_llm_after_failure and self._failures >= self._recovery.policy.max_llm_failures:
                        self.disable(f"LLM disabled after repeated failures: {exc}")
                return None

    def load_now(self) -> Any:
        return self._ensure_loaded()

    def get_model_info(self) -> dict[str, Any]:
        if self._model is not None:
            try:
                info = dict(self._model.get_model_info())
            except Exception:
                info = {}
            info.setdefault("loaded", True)
            info.setdefault("mode", "lazy_loaded")
            return info
        return {
            "name": self._model_label,
            "loaded": False,
            "mode": "lazy",
            "disabled": self._disabled,
            "last_error": self._last_error,
            "failures": self._failures,
        }

    def clear_cache(self) -> None:
        if self._model is not None and hasattr(self._model, "clear_cache"):
            try:
                self._model.clear_cache()
            except Exception:
                pass

    def get_stats(self) -> dict[str, Any]:
        if self._model is not None and hasattr(self._model, "get_stats"):
            try:
                return dict(self._model.get_stats())
            except Exception:
                pass
        return {"lazy": True, "loaded": self._model is not None, "failures": self._failures}

    def _call_or_fallback(self, method_name: str, fallback: str, *args: Any, **kwargs: Any) -> Any:
        model = self._ensure_loaded()
        if model is None:
            return fallback
        try:
            method = getattr(model, method_name)
            return method(*args, **kwargs)
        except Exception as exc:
            self._last_error = str(exc)
            self._failures += 1
            if self._recovery:
                self._recovery._record_incident({
                    "kind": "llm_runtime_failed",
                    "component": self._model_label,
                    "method": method_name,
                    "error": str(exc),
                    "failures": self._failures,
                })
                if self._recovery.policy.auto_disable_llm_after_failure and self._failures >= self._recovery.policy.max_llm_failures:
                    self.disable(f"LLM disabled after runtime failures: {exc}")
            return fallback

    def generate_clean_answer(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_or_fallback(
            "generate_clean_answer",
            "LLM временно недоступна. Программа продолжает работать в безопасном режиме.",
            *args,
            **kwargs,
        )

    def generate_from_full_documents(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_or_fallback(
            "generate_from_full_documents",
            "LLM временно недоступна. Документы найдены, но генерация ответа отключена.",
            *args,
            **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        model = self._ensure_loaded()
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)
