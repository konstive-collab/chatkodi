from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.llm import load_guard


class _FakeWeight:
    def __init__(self, size: int) -> None:
        self._size = size

    def stat(self):
        return type("Stat", (), {"st_size": self._size})()


def _model(tmp_path: Path, shards: list[int], monkeypatch) -> Path:
    # Real multi-GB files would make the test slow; only sizes matter here.
    monkeypatch.setattr(load_guard, "_weight_files", lambda _path: [_FakeWeight(size) for size in shards])
    return tmp_path / "model"


def test_cuda_estimate_counts_largest_shard_and_cpu_offload(tmp_path: Path, monkeypatch) -> None:
    gb = 1024 ** 3
    model = _model(tmp_path, [4 * gb, 3 * gb], monkeypatch)
    needed, largest, weights = load_guard.estimate_load_commit_gb(model, device="cuda", gpu_budget_gb=5.0)
    assert largest == pytest.approx(4.0)
    assert weights == pytest.approx(7.0)
    # 4 GB mapped shard + 2 GB that stays on CPU + overhead
    assert needed == pytest.approx(4.0 + 2.0 + load_guard._COMMIT_OVERHEAD_GB)


def test_commit_check_blocks_with_pagefile_advice(tmp_path: Path, monkeypatch) -> None:
    gb = 1024 ** 3
    model = _model(tmp_path, [4 * gb], monkeypatch)
    monkeypatch.setattr(load_guard, "free_commit_gb", lambda: 3.0)
    check = load_guard.check_commit_for_load(model, device="cuda", gpu_budget_gb=8.0, model_name="Qwen3-4B")
    assert check.ok is False
    assert "файл подкачки" in check.message
    assert "Qwen3-4B" in check.message


def test_stale_inflight_marker_blocks_until_memory_grows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(load_guard, "_state_dir", lambda: tmp_path)
    monkeypatch.delenv("CHATKODI_FORCE_LLM", raising=False)
    marker = {"model_path": "M", "pid": 999_999_999, "free_commit_gb": 5.0, "needed_commit_gb": 6.5}
    (tmp_path / load_guard._INFLIGHT_NAME).write_text(json.dumps(marker), encoding="utf-8")

    message = load_guard.crash_blocks_autoload("M", free_commit_now=5.2)
    assert message and "аварийно" in message
    assert not (tmp_path / load_guard._INFLIGHT_NAME).exists()
    assert load_guard.crash_blocks_autoload("other-model", free_commit_now=5.2) is None
    # the user enlarged the pagefile: automatic retry is allowed again
    assert load_guard.crash_blocks_autoload("M", free_commit_now=9.0) is None


def test_sentinel_is_removed_after_normal_return(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(load_guard, "_state_dir", lambda: tmp_path)
    with load_guard.LoadSentinel("M", free_commit=10, needed_commit=6):
        assert (tmp_path / load_guard._INFLIGHT_NAME).exists()
    assert not (tmp_path / load_guard._INFLIGHT_NAME).exists()
    assert load_guard.collect_previous_crash() is None
