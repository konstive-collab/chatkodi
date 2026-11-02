"""Files on the computer beyond the chat: find, attach, open, copy to the
library, rename and move (with an undo log).

Search order: the library first, then the user's Desktop, Documents and
Downloads and the folders added in the settings. Names and folder names are
matched (content search inside the library is the RAG pipeline's job).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from core.paths import runtime_root

_SKIP_DIRS = {
    "appdata", "$recycle.bin", "windows", "program files", "program files (x86)", "programdata",
    "node_modules", ".git", "__pycache__", ".venv", "venv", "site-packages", "system volume information",
}
_TYPE_WORDS = {
    "image": ("фото", "снимок", "снимк", "картин", "изображен", "скриншот", "скан"),
    ".pdf": ("pdf", "пдф"),
    ".docx": ("word", "ворд", "docx"),
    ".xlsx": ("excel", "эксел", "xlsx", "таблиц"),
}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
_STOP = {"найди", "найти", "файл", "файлы", "файла", "документ", "документы", "папке", "папку", "папка", "где",
         "лежит", "мой", "мои", "наш", "наши", "компьютере", "открой", "приложи", "этот", "тот", "который", "последний",
         "чат", "чату", "чата", "разговору", "меня", "переименуй", "перенеси", "перемести", "скопируй", "библиотеку"}


@dataclass
class FoundFile:
    path: Path
    score: float
    where: str  # «библиотека» | «Рабочий стол» | …

    @property
    def modified(self) -> float:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return 0.0


def _norm(text: str) -> str:
    return re.sub(r"[_\-.]+", " ", (text or "").lower().replace("ё", "е"))


def _stems(query: str) -> List[str]:
    words = re.findall(r"[a-zа-я0-9]{3,}", _norm(query))
    return [w[:5] if len(w) > 5 and not w.isdigit() else w for w in words if w not in _STOP]


def search_roots(library: Optional[Path], extra: Sequence[str] = ()) -> List[Tuple[Path, str]]:
    home = Path.home()
    roots: List[Tuple[Path, str]] = []
    if library is not None:
        roots.append((Path(library), "библиотека"))
    for folder in extra:
        roots.append((Path(folder), Path(folder).name))
    for name, label in (("Desktop", "Рабочий стол"), ("Documents", "Документы"), ("Downloads", "Загрузки")):
        for base in (home, home / "OneDrive"):
            if (base / name).is_dir():
                roots.append((base / name, label))
    seen, unique = set(), []
    for path, label in roots:
        key = str(path.resolve()).lower() if path.exists() else ""
        if key and key not in seen:
            seen.add(key)
            unique.append((path, label))
    return unique


def find_files(query: str, roots: Sequence[Tuple[Path, str]], *, limit: int = 12,
               max_files: int = 60000, seconds: float = 6.0) -> List[FoundFile]:
    """Files whose name or folders match the query words (best first; the library wins ties)."""
    from core.agent.attachments import is_supported

    stems = _stems(query)
    low = _norm(query)
    wanted_ext = {ext for ext, words in _TYPE_WORDS.items() if any(w in low for w in words)}
    if not stems and not wanted_ext:
        return []
    deadline = time.monotonic() + seconds
    found: List[FoundFile] = []
    scanned = 0
    for order, (root, label) in enumerate(roots):
        for folder, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d.lower() not in _SKIP_DIRS and not d.startswith(".")]
            folder_norm = _norm(str(Path(folder).relative_to(root))) if Path(folder) != root else ""
            for name in files:
                scanned += 1
                if scanned > max_files or time.monotonic() > deadline:
                    break
                path = Path(folder) / name
                ext = path.suffix.lower()
                if not is_supported(path):
                    continue
                if wanted_ext and not (ext in wanted_ext or ("image" in wanted_ext and ext in _IMAGE_EXT)):
                    continue
                name_norm = _norm(path.stem)
                hits = sum(2.0 for s in stems if s in name_norm) + sum(1.0 for s in stems if s in folder_norm and s not in name_norm)
                if stems and hits == 0:
                    continue
                score = hits / max(1, 2 * len(stems)) + (0.05 if order == 0 else 0.0)
                found.append(FoundFile(path, score, label))
    found.sort(key=lambda f: (-f.score, -f.modified))
    return found[:limit]


# ---------------------------------------------------------------- changes with undo

def undo_dir() -> Path:
    folder = runtime_root() / "undo"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def apply_moves(moves: Iterable[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[str], Optional[Path]]:
    """Move/rename files; never overwrites. Returns (done, errors, undo log)."""
    done, errors = [], []
    for src, dst in moves:
        source, target = Path(src), Path(dst)
        try:
            if not source.exists():
                errors.append(f"{source.name}: файл не найден")
                continue
            if target.exists():
                errors.append(f"{target.name}: такой файл уже есть — не перезаписываю")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            done.append((str(source), str(target)))
        except OSError as exc:
            errors.append(f"{source.name}: {exc}")
    log = None
    if done:
        log = undo_dir() / f"moves_{time.strftime('%Y%m%d_%H%M%S')}.json"
        log.write_text(json.dumps({"moves": done}, ensure_ascii=False, indent=1), encoding="utf-8")
    return done, errors, log


def undo_moves(log: Path) -> Tuple[int, List[str]]:
    data = json.loads(Path(log).read_text(encoding="utf-8"))
    back = [(dst, src) for src, dst in reversed(data.get("moves", []))]
    done, errors, _ = apply_moves(back)
    Path(log).unlink(missing_ok=True)
    return len(done), errors


def copy_into(path: Path, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / path.name
    index = 2
    while target.exists():
        target = folder / f"{path.stem} ({index}){path.suffix}"
        index += 1
    shutil.copy2(path, target)
    return target
