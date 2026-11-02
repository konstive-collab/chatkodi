import sys
from pathlib import Path

from core.paths import project_root, runtime_root
from core.resilience import RecoveryManager, ResiliencePolicy
from core.system_profile import SystemProfile


def _profile(*, cuda: bool) -> SystemProfile:
    return SystemProfile(
        profile_name="gpu_fast" if cuda else "desktop_cpu",
        total_ram_gb=32.0,
        cpu_cores=12,
        machine="AMD64",
        platform_name="Windows",
        has_cuda=cuda,
        gpu_name="GPU" if cuda else "",
        gpu_vram_gb=8.0 if cuda else 0.0,
        llm_startup_mode="eager" if cuda else "lazy",
        context_char_limit=16000 if cuda else 12000,
        prefer_lightweight_llm=True,
        notes=(),
    )


def test_production_startup_respects_hardware_profile(monkeypatch):
    for name in ("CHATKODI_DISABLE_LLM", "CHATKODI_FORCE_LLM", "CHATKODI_LLM_STARTUP_MODE"):
        monkeypatch.delenv(name, raising=False)
    assert RecoveryManager(ResiliencePolicy(llm_startup_mode="auto"), _profile(cuda=False)).choose_llm_startup_mode() == "lazy"
    assert RecoveryManager(ResiliencePolicy(llm_startup_mode="auto"), _profile(cuda=True)).choose_llm_startup_mode() == "eager"


def test_frozen_portable_ignores_stale_machine_paths(tmp_path, monkeypatch):
    app_root = tmp_path / "moved" / "ChatKodi"
    app_root.mkdir(parents=True)
    executable = app_root / "ChatKodiApp.exe"
    executable.write_bytes(b"")
    (app_root / "portable.flag").write_text("portable", encoding="ascii")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setenv("CHATKODI_APP_ROOT", str(tmp_path / "old-drive" / "ChatKodi"))
    monkeypatch.setenv("CHATKODI_RUNTIME_DIR", str(tmp_path / "old-drive" / "runtime"))
    monkeypatch.setenv("CHATKODI_PORTABLE", "1")
    monkeypatch.delenv("CHATKODI_TRUST_PATH_OVERRIDES", raising=False)

    assert project_root() == app_root.resolve()
    assert runtime_root() == (app_root / "runtime").resolve()
