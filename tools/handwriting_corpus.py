"""Handwritten tables with an exact key, for measuring how well photos of
handwriting are read (cell by cell).

Rendered fonts are read almost perfectly; real handwriting is not. To come
closer to a pen on paper every character is turned, shifted and scaled a little,
table lines are drawn by hand, some numbers are written over crossed-out ones,
and the sheet is "photographed": shadow, perspective, blur, JPEG, a low
resolution, some sheets sideways. The key lists every cell as written.

    python -m tools.handwriting_corpus --out <dir>      # sheets + key.json
    python tools/eval_handwriting.py --corpus <dir>      # cell accuracy
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = Path("C:/Windows/Fonts")
HANDS = ["segoepr.ttf", "segoesc.ttf", "Inkfree.ttf", "comici.ttf", "Gabriola.ttf", "Mistral.ttf"]
SURNAMES = ["Федоров", "Захаров", "Петров", "Авдеев", "Егоров", "Погодин", "Лебедев", "Морозова", "Кузнецова",
            "Соколов", "Волкова", "Никитин", "Орлова", "Белов", "Тихонов", "Григорьева", "Савельев", "Ершова",
            "Медведев", "Карпова", "Зуев", "Филиппова", "Носков", "Ларина"]
LETTERS = "АБВГДЕИКЛМНОПРСТ"
INK = [(28, 40, 120), (20, 25, 60), (35, 60, 150)]


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


def _hand_text(img: Image.Image, xy, text: str, font, ink, rnd: random.Random, jitter: float) -> None:
    """Character by character with a small turn, shift and scale each."""
    x, y = xy
    slant = rnd.uniform(-4, 4) * jitter
    for ch in text:
        if ch == " ":
            x += font.size * 0.35
            continue
        size = max(10, int(font.size * rnd.uniform(1 - 0.12 * jitter, 1 + 0.12 * jitter)))
        f = font.font_variant(size=size)
        w = int(f.getlength(ch)) + 8
        tile = Image.new("L", (w + 12, size + 24), 0)
        ImageDraw.Draw(tile).text((6, 4), ch, font=f, fill=255)
        tile = tile.rotate(slant + rnd.uniform(-7, 7) * jitter, resample=Image.BICUBIC, expand=False)
        dy = rnd.uniform(-3, 3) * jitter
        img.paste(Image.new("RGB", tile.size, ink), (int(x), int(y + dy)), tile)
        x += f.getlength(ch) * rnd.uniform(0.92, 1.05)


def _hand_line(draw: ImageDraw.ImageDraw, a, b, rnd: random.Random, ink=(40, 40, 90)) -> None:
    """A ruled line drawn by hand: slightly wavy, not quite straight."""
    (x0, y0), (x1, y1) = a, b
    steps = 12
    points = []
    for i in range(steps + 1):
        t = i / steps
        wobble = math.sin(t * math.pi * rnd.uniform(1, 3)) * rnd.uniform(0, 2.5)
        if abs(x1 - x0) > abs(y1 - y0):
            points.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t + wobble))
        else:
            points.append((x0 + (x1 - x0) * t + wobble, y0 + (y1 - y0) * t))
    draw.line(points, fill=ink, width=2)


def _number(rnd: random.Random, kind: str) -> str:
    if kind == "int":
        return str(rnd.randint(3, 25))
    if kind == "dec1":
        return f"{rnd.randint(8, 30)},{rnd.randint(0, 9)}"
    if kind == "dec2":
        return f"{rnd.randint(10, 69)},{rnd.randint(0, 99):02d}"
    return str(rnd.randint(120, 290))


def make_sheet(index: int, seed: int) -> tuple[Image.Image, dict]:
    rnd = random.Random(seed)
    hand = HANDS[index % len(HANDS)]
    jitter = 0.6 + 0.25 * (index // len(HANDS))  # later sheets are messier
    font = _font(hand, rnd.choice([34, 36, 38]))
    ink = rnd.choice(INK)
    kinds = rnd.choice([["int", "dec1", "dec2", "total"], ["int", "int", "dec1", "total"], ["dec1", "dec2", "total"]])
    header = ["№", "ФИО"] + [f"упр.{i}" for i in range(1, len(kinds))] + ["Итого"]
    names = rnd.sample(SURNAMES, rnd.randint(7, 10))
    rows = []
    for n, surname in enumerate(names, 1):
        cells = [_number(rnd, k) for k in kinds]
        if rnd.random() < 0.15:
            cells[rnd.randrange(len(cells) - 1)] = ""  # an empty cell
        rows.append([str(n), f"{surname} {rnd.choice(LETTERS)}.{rnd.choice(LETTERS)}."] + cells)
    widths = [70, 360] + [170] * len(kinds)
    width = sum(widths) + 160
    row_h = int(font.size * 1.75)
    height = 260 + row_h * (len(rows) + 1) + 140
    page = Image.new("RGB", (width, height), (247, 245, 236))
    draw = ImageDraw.Draw(page)
    _hand_text(page, (80, 60), "Ведомость сдачи нормативов", font, ink, rnd, jitter)
    top, left = 170, 80
    xs = [left]
    for w in widths:
        xs.append(xs[-1] + w)
    for r in range(len(rows) + 2):
        _hand_line(draw, (left, top + r * row_h), (xs[-1], top + r * row_h), rnd)
    for x in xs:
        _hand_line(draw, (x, top), (x, top + (len(rows) + 1) * row_h), rnd)
    small = font.font_variant(size=int(font.size * 0.8))
    for r, cells in enumerate([header] + rows):
        for c, cell in enumerate(cells):
            if not cell:
                continue
            y = top + r * row_h + rnd.uniform(2, row_h * 0.25)
            x = xs[c] + rnd.uniform(6, 22)
            if r > 0 and c >= 2 and rnd.random() < 0.06 * jitter:
                # a number written over a crossed-out one
                _hand_text(page, (x, y), _number(rnd, kinds[c - 2]), small, (120, 120, 150), rnd, jitter)
                draw.line([(x, y + row_h * 0.45), (x + 90, y + row_h * 0.2)], fill=ink, width=3)
            _hand_text(page, (x, y), cell, font if r else small, ink, rnd, jitter)
    key = {"hand": hand, "jitter": round(jitter, 2), "header": header, "rows": rows}
    return page, key


def phone_shot(img: Image.Image, rnd: random.Random, *, sideways: int = 0, long_side: int = 1600) -> Image.Image:
    w, h = img.size
    shade = Image.linear_gradient("L").rotate(rnd.choice([0, 90, 180, 270])).resize((w, h))
    shade = shade.point(lambda v: 255 - int(v * rnd.uniform(0.15, 0.35)))
    img = Image.composite(img, Image.new("RGB", (w, h), (0, 0, 0)), shade)
    pad = 70
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (95, 88, 80))
    canvas.paste(img, (pad, pad))
    canvas = canvas.rotate(rnd.uniform(-3, 3), resample=Image.BICUBIC, fillcolor=(95, 88, 80))
    cw, ch = canvas.size
    d = lambda: rnd.uniform(0, 0.035)
    quad = (cw * d(), ch * d(), cw * d(), ch * (1 - d()), cw * (1 - d()), ch * (1 - d()), cw * (1 - d()), ch * d())
    canvas = canvas.transform((cw, ch), Image.QUAD, quad, resample=Image.BICUBIC)
    import numpy as np

    noise = np.random.default_rng(rnd.randrange(1 << 30)).normal(0, 9, (ch, cw, 1))
    canvas = Image.fromarray(np.clip(np.asarray(canvas, dtype=np.float32) + noise, 0, 255).astype("uint8"))
    canvas = canvas.filter(ImageFilter.GaussianBlur(rnd.uniform(0.8, 1.6)))
    if sideways:
        canvas = canvas.rotate(sideways, expand=True)
    k = long_side / max(canvas.size)
    return canvas.resize((int(canvas.width * k), int(canvas.height * k)), Image.LANCZOS)


def build(out: Path, count: int = 12) -> list:
    out.mkdir(parents=True, exist_ok=True)
    keys = []
    for index in range(count):
        rnd = random.Random(500 + index)
        sheet, key = make_sheet(index, seed=900 + index)
        sideways = 270 if index % 4 == 3 else 0  # a quarter of the sheets are shot sideways
        shot = phone_shot(sheet, rnd, sideways=sideways, long_side=rnd.choice([1400, 1600, 2000]))
        name = f"hand_{index:02d}.jpg"
        shot.save(out / name, "JPEG", quality=rnd.choice([60, 72, 85]))
        keys.append(dict(key, photo=name, sideways=sideways))
    (out / "key.json").write_text(json.dumps(keys, ensure_ascii=False, indent=1), encoding="utf-8")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    for key in build(Path(args.out), args.count):
        print(key["photo"], key["hand"], key["jitter"], len(key["rows"]), "sideways" if key["sideways"] else "")


if __name__ == "__main__":
    main()
