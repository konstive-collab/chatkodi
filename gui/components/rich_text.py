"""Render ChatKodi answers (a small Markdown subset) as Qt rich text.

Answers are built by the fact layer, structured builders and the LLM with
``**bold**``, ``> quote`` lines, ``- lists``, ``1. lists``, ```code``` spans and
``_italic_`` notes. The chat used to strip all of it and show raw ``>`` marks;
this renderer keeps the structure and styles it from the active theme.

Qt's HTML engine supports only a subset of CSS and does not blend ``rgba()``
colours reliably, so translucent palette colours are pre-blended into solid
ones against the bubble background.
"""

from __future__ import annotations

import html
import re
from typing import Dict, List

_RGBA = re.compile(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)")


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def solid_color(color: str, background: str) -> str:
    """Blend an ``rgba(...)`` colour over ``background`` into ``#rrggbb``."""
    match = _RGBA.fullmatch(str(color or "").strip())
    if not match:
        return color
    r, g, b = (int(match.group(i)) for i in (1, 2, 3))
    alpha = float(match.group(4)) if match.group(4) is not None else 1.0
    br, bg, bb = _hex_to_rgb(background) if str(background).startswith("#") else (0, 0, 0)
    mix = lambda top, bottom: int(round(top * alpha + bottom * (1 - alpha)))  # noqa: E731
    return f"#{mix(r, br):02x}{mix(g, bg):02x}{mix(b, bb):02x}"


_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")
_ITALIC = re.compile(r"(?<![\w*])_(?!\s)(.+?)(?<!\s)_(?![\w])")
_BULLET = re.compile(r"^\s*(?:[-*•]|—)\s+(.*)$")
_NUMBERED = re.compile(r"^\s*(\d{1,3})[.)]\s+(.*)$")
_HEADING = re.compile(r"^\s*#{1,4}\s+(.*)$")
_MUTED_PREFIXES = ("📚", "Метод:", "Источник:", "Источники:", "🧩", "Отобранные источники", "Проверенные кандидаты")


def _inline(text: str, colors: Dict[str, str]) -> str:
    escaped = html.escape(text, quote=False)
    escaped = _CODE.sub(
        lambda m: f'<span style="background-color:{colors["code_bg"]}; font-family:Consolas,monospace;">&nbsp;{m.group(1)}&nbsp;</span>',
        escaped,
    )
    escaped = _BOLD.sub(lambda m: f'<b style="color:{colors["title"]};">{m.group(1)}</b>', escaped)
    escaped = _ITALIC.sub(lambda m: f"<i>{m.group(1)}</i>", escaped)
    return escaped


def markdown_to_html(text: str, palette: Dict[str, str]) -> str:
    """Convert an answer to Qt rich text using theme ``palette`` colours."""
    background = palette.get("ai_bubble", "#111827")
    colors = {
        "title": palette.get("title", "#f9fafb"),
        "fg": palette.get("fg", "#e5e7eb"),
        "muted": palette.get("muted_2", palette.get("muted", "#94a3b8")),
        "accent": solid_color(palette.get("accent", "#22c55e"), background),
        "accent_soft": palette.get("accent_soft", "#bbf7d0"),
        "quote_bg": solid_color(palette.get("accent_bg", "rgba(34,197,94,0.10)"), background),
        "code_bg": solid_color(palette.get("code_bg", "rgba(255,255,255,0.07)"), background),
    }
    lines = str(text or "").replace("\r\n", "\n").split("\n")
    out: List[str] = []
    list_kind = ""
    first_content = True

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            out.append(f"</{list_kind}>")
            list_kind = ""

    table_rows: List[List[str]] = []

    def flush_table() -> None:
        if not table_rows:
            return
        border = colors["muted"]
        cells_html = []
        for index, cells in enumerate(table_rows):
            tag = "th" if index == 0 else "td"
            style = f"border:1px solid {border}; padding:3px 6px;" + (f" background-color:{colors['quote_bg']};" if index == 0 else "")
            cells_html.append("<tr>" + "".join(f'<{tag} style="{style}">{_inline(c, colors)}</{tag}>' for c in cells) + "</tr>")
        out.append(
            f'<table cellspacing="0" cellpadding="4" style="margin-top:4px; margin-bottom:4px; border-collapse:collapse; '
            f'border:1px solid {border};">' + "".join(cells_html) + "</table>"
        )
        table_rows.clear()

    code_lines: List[str] = []
    in_code = False

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                body = "<br>".join(html.escape(item, quote=False).replace(" ", "&nbsp;") for item in code_lines)
                out.append(
                    f'<table width="100%" cellspacing="0" cellpadding="6" style="margin-top:3px; margin-bottom:3px;">'
                    f'<tr><td style="background-color:{colors["code_bg"]}; font-family:Consolas,monospace;">{body}</td></tr></table>'
                )
                code_lines.clear()
            else:
                flush_table()
                close_list()
            in_code = not in_code
            first_content = False
            continue
        if in_code:
            code_lines.append(raw)
            continue
        if stripped.startswith("|") and stripped.count("|") >= 2:
            close_list()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c or "--") for c in cells):  # |---|---| separator
                table_rows.append(cells)
            first_content = False
            continue
        flush_table()
        if not line.strip():
            close_list()
            out.append('<div style="margin-top:4px;"></div>')
            continue

        if stripped.startswith(">"):
            close_list()
            body = _inline(stripped.lstrip("> ").strip(), colors)
            out.append(
                f'<table width="100%" cellspacing="0" cellpadding="6" style="margin-top:2px; margin-bottom:2px;">'
                f'<tr><td width="3" style="background-color:{colors["accent"]};"></td>'
                f'<td style="background-color:{colors["quote_bg"]}; color:{colors["fg"]};">{body}</td></tr></table>'
            )
            first_content = False
            continue

        heading = _HEADING.match(stripped)
        if heading:
            close_list()
            out.append(f'<p style="margin-top:6px; margin-bottom:2px;"><b style="font-size:14px; color:{colors["title"]};">{_inline(heading.group(1), colors)}</b></p>')
            first_content = False
            continue

        bullet = _BULLET.match(line)
        numbered = None if bullet else _NUMBERED.match(line)
        if bullet or numbered:
            kind = "ul" if bullet else "ol"
            if list_kind != kind:
                close_list()
                out.append(f'<{kind} style="margin-top:2px; margin-bottom:2px; margin-left:-18px;">')
                list_kind = kind
            body = bullet.group(1) if bullet else numbered.group(2)
            out.append(f'<li style="margin-bottom:2px;">{_inline(body, colors)}</li>')
            first_content = False
            continue

        close_list()
        if stripped.startswith(_MUTED_PREFIXES):
            out.append(f'<p style="margin-top:2px; margin-bottom:0px; color:{colors["muted"]}; font-size:11px;">{_inline(stripped, colors)}</p>')
            first_content = False
            continue
        if stripped.startswith("_") and stripped.endswith("_") and len(stripped) > 2:
            out.append(f'<p style="margin:0px; color:{colors["muted"]};"><i>{html.escape(stripped.strip("_"), quote=False)}</i></p>')
            continue
        if first_content and stripped.startswith("**"):
            # The direct answer (value, total, name) leads and is larger.
            out.append(
                f'<p style="margin-top:0px; margin-bottom:4px; font-size:16px; color:{colors["accent_soft"]};">'
                f"{_inline(stripped, colors).replace('color:' + colors['title'], 'color:' + colors['accent_soft'])}</p>"
            )
            first_content = False
            continue
        out.append(f'<p style="margin-top:0px; margin-bottom:2px;">{_inline(stripped, colors)}</p>')
        first_content = False
    flush_table()
    close_list()
    # Trim leading/trailing spacers.
    while out and out[0].startswith('<div style="margin-top:4px;">'):
        out.pop(0)
    while out and out[-1].startswith('<div style="margin-top:4px;">'):
        out.pop()
    return f'<div style="color:{colors["fg"]};">' + "".join(out) + "</div>"


def plain_to_html(text: str) -> str:
    return html.escape(str(text or ""), quote=False).replace("\n", "<br>")
