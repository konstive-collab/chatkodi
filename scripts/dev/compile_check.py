#!/usr/bin/env python3
"""Быстрая проверка синтаксиса Python-файлов проекта."""
from __future__ import annotations
import compileall
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
paths = [root / "core", root / "gui", root / "tools", root / "main.py", root / "launcher.py"]
ok = True
for path in paths:
    if path.is_dir():
        ok = compileall.compile_dir(str(path), quiet=1)
    elif path.exists():
        ok = compileall.compile_file(str(path), quiet=1) and ok
print("compile_check:", "OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
