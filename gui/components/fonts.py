"""Bundled fonts (SIL OFL): Onest for the interface, Unbounded for the wordmark.

They are registered from ``assets/fonts`` (next to the sources or inside the
frozen bundle), so the look does not depend on fonts installed in Windows.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from core.logging_utils import get_logger

logger = get_logger(__name__)

UI_FAMILY = "Onest"
UI_MEDIUM = "Onest Medium"
UI_SEMIBOLD = "Onest SemiBold"
BRAND_FAMILY = "Unbounded SemiBold"
FALLBACK = "Segoe UI"
_LOADED: List[str] = []


def _font_dirs() -> List[Path]:
    from core.paths import project_root

    dirs = [project_root() / "assets" / "fonts"]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(Path(meipass) / "assets" / "fonts")
    return [d for d in dirs if d.is_dir()]


def load_app_fonts() -> List[str]:
    """Register the bundled fonts once; returns the family names Qt sees."""
    if _LOADED:
        return list(_LOADED)
    from PyQt5.QtGui import QFontDatabase

    for folder in _font_dirs():
        for path in sorted(folder.glob("*.ttf")):
            font_id = QFontDatabase.addApplicationFont(str(path))
            if font_id < 0:
                logger.warning("Font could not be loaded: %s", path.name)
                continue
            _LOADED.extend(QFontDatabase.applicationFontFamilies(font_id))
        if _LOADED:
            break
    logger.info("Bundled fonts: %s", sorted(set(_LOADED)))
    return list(_LOADED)


def ui_family() -> str:
    return UI_FAMILY if UI_FAMILY in load_app_fonts() else FALLBACK


def family_or_fallback(family: str) -> str:
    return family if family in load_app_fonts() else FALLBACK


def apply_app_font(app) -> None:
    from PyQt5.QtGui import QFont

    load_app_fonts()
    font = QFont(ui_family(), 10)
    font.setHintingPreference(QFont.PreferNoHinting)
    font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(font)
