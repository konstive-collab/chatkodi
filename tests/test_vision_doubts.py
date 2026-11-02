"""Numbers the model read unsure of are shown, with its second choice, and coloured in Excel."""

from __future__ import annotations

import math

from core.agent import vision
from core.agent.attachments import Attachment


def _tok(token: str, p: float = 0.99, alts=()):
    top = [{"token": token, "logprob": math.log(p)}] + [{"token": t, "logprob": math.log(q)} for t, q in alts]
    return {"token": token, "logprob": math.log(p), "top_logprobs": top}


def _reading(last: str = "3"):
    """«| 6 | Егоров А.А | 20 | 25<last> |» with the last digit unsure (5 almost as likely)."""
    tokens = ["|", " №", " |", " ФИО", " |", " упр", ".1", " |", " Итог", " |", "\n", "|---|---|---|---|", "\n",
              "|", " ", "6", " |", " Егоров", " А", ".А", " |", " ", "2", "0", " |", " ", "2", "5"]
    probs = [_tok(t) for t in tokens]
    probs.append(_tok(last, 0.46, [("5" if last != "5" else "3", 0.45), ("8", 0.06)]))
    probs += [_tok(" |"), _tok("\n")]
    return probs


def _second_look(monkeypatch, *again: str):
    calls = iter(again)

    def ask(llm, data, prompt, max_tokens=2500, probs=None, max_side=0, upscale=False):
        reading = _reading(next(calls))
        probs.extend(reading)
        return "".join(p["token"] for p in reading)

    monkeypatch.setattr(vision, "ask_image", ask)
    monkeypatch.setattr(vision, "_contrast", lambda data: data)
    text, cells = vision.cell_confidence(_reading())
    return vision.second_look(None, b"", text, vision.doubtful(cells))


def test_two_of_three_readings_decide_and_the_doubt_stays_visible(monkeypatch):
    text, doubts = _second_look(monkeypatch, "5", "5")
    assert "| 255 |" in text and "| 253 |" not in text
    assert doubts[0].value == "255" and doubts[0].readings == ["253", "255", "255"]
    assert doubts[0].describe() == "«6 Егоров А.А», «Итог»: 255 (прочитано 3 раза: 253 / 255 / 255)"


def test_three_alike_clear_the_doubt_and_no_agreement_keeps_the_first(monkeypatch):
    text, doubts = _second_look(monkeypatch, "3", "3")
    assert "| 253 |" in text and doubts == []
    text, doubts = _second_look(monkeypatch, "5", "8")
    assert "| 253 |" in text and doubts[0].value == "253" and doubts[0].readings == ["253", "255", "258"]


def test_unsure_digit_gives_the_cell_and_the_alternative():
    text, cells = vision.cell_confidence(_reading())
    assert "| 6 | Егоров А.А | 20 | 253 |" in text
    doubt = [c for c in cells if c.confidence < vision.DOUBT_BELOW]
    assert len(doubt) == 1 and doubt[0].value == "253" and doubt[0].alternative == "255"
    assert doubt[0].row_label == "6 Егоров А.А" and doubt[0].column_name == "Итог" and doubt[0].column == 3
    assert doubt[0].describe() == "«6 Егоров А.А», «Итог»: 253 — возможно 255"
    assert [c.value for c in cells if c.confidence >= vision.DOUBT_BELOW] == ["20"]


def test_doubts_are_said_and_coloured_in_excel(tmp_path):
    from openpyxl import load_workbook

    from core.agent.agent import _doubts_note, attachment_outcome
    from core.agent.outputs import markdown_to_xlsx

    text, cells = vision.cell_confidence(_reading())
    item = Attachment(id="1", name="лист.jpg", path=tmp_path / "x.jpg", kind="image", text=text, recognized=True,
                      read=True, doubts=vision.doubtful(cells))
    assert "253 — возможно 255" in _doubts_note(item)
    assert attachment_outcome(item) == ("распознано · ⚠ проверьте цифры", True)
    ws = load_workbook(markdown_to_xlsx(text, tmp_path / "t.xlsx", item.doubts)).active
    assert ws["D2"].value == 253 and ws["D2"].fill.fgColor.rgb.endswith("FCD5B4") and "255" in ws["D2"].comment.text
    assert ws["C2"].comment is None
