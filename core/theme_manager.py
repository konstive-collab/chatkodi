"""Единая система тем ChatKodi.

Один источник правды для трёх окон: лаунчера, экрана загрузки и основной
программы. До rc6 часть цветов жила отдельно в разных модулях, из-за чего
лаунчер, загрузчик и рабочее окно выглядели как разные приложения. Теперь все
берут палитру отсюда, а конкретные QSS-стили только раскладывают эти цвета по
виджетам.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

THEME_ORDER = ("classic", "dark", "light")
THEME_LABELS = {
    "classic": "Классическая",
    "dark": "Тёмная",
    "light": "Светлая",
}
THEME_SHORT_LABELS = {
    "classic": "Классика",
    "dark": "Тёмная",
    "light": "Светлая",
}
THEME_ICONS = {
    "classic": "●",
    "dark": "●",
    "light": "●",
}
THEME_CHIP_COLORS = {
    "classic": "#22c55e",
    "dark": "#2dd46f",
    "light": "#f8fafc",
}
_THEME_ALIASES = {
    "default": "classic",
    "auto": "classic",
    "classic": "classic",
    "классическая": "classic",
    "классика": "classic",
    "previous": "classic",
    "old": "classic",
    "dark": "dark",
    "тёмная": "dark",
    "темная": "dark",
    "night": "dark",
    "green": "dark",
    "зелёная": "dark",
    "зеленая": "dark",
    "light": "light",
    "светлая": "light",
    "day": "light",
    "white": "light",
}

# Классическая тема — старая сине-графитовая база + зелёный акцент. Она стала
# темой по умолчанию, потому что лучше совпадает со старым экраном загрузки.
THEME_PALETTES: dict[str, dict[str, str]] = {
    "classic": {
        "bg": "#020617",
        "panel": "#111827",
        "panel_2": "#1f2937",
        "sunken": "#0b1220",
        "surface": "#111827",
        "surface_soft": "#0f172a",
        "fg": "#e5e7eb",
        "title": "#f9fafb",
        "muted": "#9ca3af",
        "muted_2": "#94a3b8",
        "border": "#374151",
        "border_soft": "#273449",
        "button": "#1f2937",
        "button_hover": "#273449",
        "button_pressed": "#0f172a",
        "accent": "#22c55e",
        "accent_hover": "#4ade80",
        "accent_text": "#042414",
        "accent_soft": "#bbf7d0",
        "accent_bg": "rgba(34,197,94,0.10)",
        "accent_bg_strong": "rgba(34,197,94,0.16)",
        "accent_border": "rgba(34,197,94,0.30)",
        "input": "#0b1220",
        "code_bg": "rgba(255,255,255,0.07)",
        "user_bubble": "#123528",
        "ai_bubble": "#111827",
        "welcome_bubble": "#0f172a",
    },
    "dark": {
        "bg": "#07110f",
        "panel": "#101a18",
        "panel_2": "#14211d",
        "sunken": "#0b1412",
        "surface": "#101a18",
        "surface_soft": "#0d1715",
        "fg": "#e5eee8",
        "title": "#f3fbf5",
        "muted": "#91a69a",
        "muted_2": "#84988d",
        "border": "#263a34",
        "border_soft": "#20352e",
        "button": "#14211d",
        "button_hover": "#192923",
        "button_pressed": "#0f1b17",
        "accent": "#2dd46f",
        "accent_hover": "#58e58f",
        "accent_text": "#042414",
        "accent_soft": "#c9f7d8",
        "accent_bg": "rgba(45,212,111,0.10)",
        "accent_bg_strong": "rgba(45,212,111,0.15)",
        "accent_border": "rgba(45,212,111,0.32)",
        "input": "#0b1412",
        "code_bg": "rgba(255,255,255,0.07)",
        "user_bubble": "#123427",
        "ai_bubble": "#101a18",
        "welcome_bubble": "#0f1d19",
    },
    "light": {
        "bg": "#f5f7f4",
        "panel": "#ffffff",
        "panel_2": "#fbfdfb",
        "sunken": "#f8faf8",
        "surface": "#ffffff",
        "surface_soft": "#fbfdfb",
        "fg": "#17211c",
        "title": "#132019",
        "muted": "#68796f",
        "muted_2": "#718076",
        "border": "#d9e3dc",
        "border_soft": "#ccd9d0",
        "button": "#ffffff",
        "button_hover": "#f0fdf4",
        "button_pressed": "#e7fbed",
        "accent": "#16a34a",
        "accent_hover": "#22c55e",
        "accent_text": "#052e16",
        "accent_soft": "#166534",
        "accent_bg": "#e7fbed",
        "accent_bg_strong": "#d8fbe6",
        "accent_border": "#86efac",
        "input": "#ffffff",
        "code_bg": "rgba(22,101,52,0.08)",
        "user_bubble": "#d8fbe6",
        "ai_bubble": "#ffffff",
        "welcome_bubble": "#f6fbf8",
    },
}


def normalize_theme(value: str | None, default: str = "classic") -> str:
    """Возвращает один из поддерживаемых ключей темы."""
    key = str(value or "").strip().lower()
    return _THEME_ALIASES.get(key) or _THEME_ALIASES.get(str(default).lower(), "classic")


def theme_palette(value: str | None = None) -> dict[str, str]:
    """Возвращает копию палитры, чтобы callers случайно не мутировали глобал."""
    return deepcopy(THEME_PALETTES[normalize_theme(value)])


def next_theme(value: str | None) -> str:
    """Циклически переключает тему: классическая → тёмная → светлая."""
    current = normalize_theme(value)
    idx = THEME_ORDER.index(current)
    return THEME_ORDER[(idx + 1) % len(THEME_ORDER)]


def theme_label(value: str | None) -> str:
    return THEME_LABELS.get(normalize_theme(value), THEME_LABELS["classic"])


def theme_short_label(value: str | None) -> str:
    return THEME_SHORT_LABELS.get(normalize_theme(value), THEME_SHORT_LABELS["classic"])


def theme_icon(value: str | None) -> str:
    return THEME_ICONS.get(normalize_theme(value), THEME_ICONS["classic"])


def theme_chip_color(value: str | None) -> str:
    return THEME_CHIP_COLORS.get(normalize_theme(value), THEME_CHIP_COLORS["classic"])


def env_theme(default: str = "classic") -> str:
    return normalize_theme(os.environ.get("CHATKODI_THEME"), default=default)


def with_theme_env(theme: str, env: dict[str, Any] | None = None) -> dict[str, str]:
    """Возвращает окружение с нормализованной темой для дочернего процесса."""
    merged = {str(k): str(v) for k, v in (env or os.environ).items()}
    merged["CHATKODI_THEME"] = normalize_theme(theme)
    return merged
