"""A sheet photographed sideways is read in the side positions and the readable one is kept.

The fake model «reads» only when a red mark of the test image is in its top-left
corner, like the real model that turns sideways handwriting into «38006 АА».
"""

from __future__ import annotations

import base64
import io

from PIL import Image

from core.agent import vision

GOOD = "| № | ФИО | упр.1 |\n|---|---|---|\n| 1 | Федоров Иван | 14 |\n| 2 | Захаров Андрей | 8 |\n| 3 | Петров Дмитрий | 20 |"
BAD = "| № | ОУО | yrp.1 |\n|---|---|---|\n| 1 | 38006 АА | 19 |\n| 2 | 38006 АА | 8 |\n| 3 | 38006 АА | 20 |"


def _photo(mark: str) -> bytes:
    img = Image.new("RGB", (40, 80), "white")
    x = 0 if mark == "left" else img.width - 12
    img.paste((255, 0, 0), (x, 0, x + 12, 12))  # big enough to survive JPEG
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def _fake_reader(calls):
    def ask(llm, data, prompt, max_tokens=2500, probs=None):
        url = vision.image_payload(data)
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as img:
            red, green, blue = img.convert("RGB").getpixel((0, 0))
        calls.append(img.size)
        return GOOD if red > 200 and green < 120 else BAD
    return ask


def test_upright_photo_is_read_once(monkeypatch):
    calls = []
    monkeypatch.setattr(vision, "ask_image", _fake_reader(calls))
    assert vision.transcribe_image(None, _photo("left")) == GOOD and len(calls) == 1


def test_sideways_photo_is_turned_to_the_readable_side(monkeypatch):
    calls, said = [], []
    monkeypatch.setattr(vision, "ask_image", _fake_reader(calls))
    # the mark at the top-right: turning 90° counter-clockwise brings it to the top-left
    assert vision.transcribe_image(None, _photo("right"), said.append) == GOOD
    assert len(calls) == 3 and said and "боком" in said[0]


def test_unreadable_in_every_position_stays_as_it_was(monkeypatch):
    monkeypatch.setattr(vision, "ask_image", lambda llm, data, prompt, max_tokens=2500, probs=None: BAD)
    assert vision.transcribe_image(None, _photo("right")) == BAD


def test_readability_separates_words_from_garbage():
    assert vision.readability(GOOD) >= vision.READABLE > vision.readability(BAD)
