from pathlib import Path

from core.gpu_info import GpuInfo
from core.llm import model_select

SIZES_GB = {
    "Qwen3.5-4B-Q4_K_M.gguf": 2.55,
    "Qwen3.5-9B-Q4_K_M.gguf": 5.29,
    "Other-1.5B-Q4_K_M.gguf": 1.1,
    "Other-7B-Q4_K_M.gguf": 4.4,
}


def _choose(monkeypatch, names, vram_gb):
    gpu = GpuInfo("Test GPU", int(vram_gb * 1024), int(vram_gb * 1024), "1") if vram_gb else None
    monkeypatch.setattr(model_select, "primary_gpu", lambda: gpu)
    monkeypatch.setattr(model_select, "file_size_gb", lambda p: SIZES_GB[p.name])
    choice = model_select.choose_llm([Path(name) for name in names], embedding_on_gpu=True)
    return choice.name if choice else None


def test_8gb_card_picks_4b_not_9b(monkeypatch) -> None:
    # 9B + embeddings + desktop does not fit 8 GB: VRAM spills to RAM, 6x slower.
    assert _choose(monkeypatch, SIZES_GB.keys(), 8.0) == "Qwen3.5-4B-Q4_K_M.gguf"


def test_preferred_model_wins_on_a_big_card(monkeypatch) -> None:
    assert _choose(monkeypatch, SIZES_GB.keys(), 24.0) == "Qwen3.5-4B-Q4_K_M.gguf"


def test_without_preferred_model_the_largest_fitting_is_used(monkeypatch) -> None:
    names = ["Other-1.5B-Q4_K_M.gguf", "Other-7B-Q4_K_M.gguf"]
    assert _choose(monkeypatch, names, 8.0) == "Other-1.5B-Q4_K_M.gguf"
    assert _choose(monkeypatch, names, 12.0) == "Other-7B-Q4_K_M.gguf"


def test_cpu_only_takes_the_smallest(monkeypatch) -> None:
    assert _choose(monkeypatch, SIZES_GB.keys(), 0) == "Other-1.5B-Q4_K_M.gguf"


def test_describe_model_parses_name_size_and_quant(tmp_path) -> None:
    path = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    path.write_bytes(b"GGUF" + b"\0" * 16)
    info = model_select.describe_model(path)
    assert info["name"] == "Qwen3.5-4B"
    assert info["size"] == "4B"
    assert info["quantization"] == "Q4_K_M"
    assert model_select.is_gguf(path)
