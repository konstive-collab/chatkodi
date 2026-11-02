from pathlib import Path

from core.atomic_io import atomic_write_json, read_json_with_backup


def test_atomic_json_uses_backup_when_primary_is_corrupted(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    atomic_write_json(path, {"value": 1})
    atomic_write_json(path, {"value": 2})
    path.write_text("{truncated", encoding="utf-8")

    assert read_json_with_backup(path) == {"value": 1}
