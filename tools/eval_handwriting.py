"""Cell accuracy of reading handwritten tables from photos.

Sheets and key: ``python -m tools.handwriting_corpus --out <dir>``; the user's
real sideways photo (tools/fixtures/vedomost_bokom.*) is added with its
ambiguous cells left out. Every number cell is compared exactly (after «.» ->
«,»), an empty cell must stay empty, a name counts when its surname is right.

    python tools/eval_handwriting.py --corpus <dir> [--variant base] [--report out.json]
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import re
import time
from typing import Callable, Dict, List

FIXTURE = _ROOT / "tools" / "fixtures" / "vedomost_bokom.json"


def _cell(text: str) -> str:
    text = re.sub(r"\*+", "", text or "").strip().replace(".", ",")
    return re.sub(r"\s+", "", text)


def _surname(text: str) -> str:
    words = re.findall(r"[А-Яа-яЁё]{3,}", text or "")
    return words[0].lower().replace("ё", "е") if words else ""


def score(key: Dict, text: str, cells_seen=()) -> Dict:
    from core.agent.table_check import parse_markdown_tables

    sure = {(c.row_label.split()[0] if c.row_label.split() else "", c.column): c.confidence for c in cells_seen}
    confidence = []

    tables = parse_markdown_tables(text)
    table = max(tables, key=lambda t: len(t.rows)) if tables else None
    got_rows = (table.rows + table.total_rows) if table else []
    by_number = {r[0].strip(): r for r in got_rows if r and r[0].strip().isdigit()}
    skip = {tuple(x) for x in key.get("uncertain_cells", [])}
    cells = cells_ok = names = names_ok = 0
    misses: List[str] = []
    for i, want in enumerate(key["rows"]):
        got = by_number.get(want[0]) or (got_rows[i] if i < len(got_rows) else [])
        if (i, 1) not in skip:
            names += 1
            if _surname(got[1] if len(got) > 1 else "") == _surname(want[1]):
                names_ok += 1
        for c in range(2, len(want)):
            if (i, c) in skip:
                continue
            cells += 1
            value = _cell(got[c]) if c < len(got) else "∅"
            if value == _cell(want[c]):
                cells_ok += 1
            else:
                misses.append(f"r{want[0]}c{c}: {want[c] or '·'} → {value or '·'}")
            if (want[0], c) in sure and want[c]:
                confidence.append({"ok": value == _cell(want[c]), "confidence": round(sure[(want[0], c)], 3), "cell": f"r{want[0]}c{c}"})
    return {"cells": cells, "cells_ok": cells_ok, "names": names, "names_ok": names_ok,
            "rows_want": len(key["rows"]), "rows_got": len(got_rows), "misses": misses, "confidence": confidence}


def variants() -> Dict[str, Callable]:
    from core.agent import vision

    return {"base": lambda llm, data: vision.read_image(llm, data)}


def confidence_table(rows: List[Dict]) -> List[str]:
    """For thresholds of the least sure digit: how many wrong cells get flagged and
    how many right ones are flagged needlessly."""
    out = []
    pairs = [(c["ok"], c["confidence"]) for r in rows for c in r.get("confidence", [])]
    wrong = sum(1 for ok, _ in pairs if not ok)
    for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        caught = sum(1 for ok, p in pairs if not ok and p < t)
        noise = sum(1 for ok, p in pairs if ok and p < t)
        out.append(f"below {t:.1f}: flags {caught + noise:>3} — wrong caught {caught}/{wrong}, right flagged {noise}/{len(pairs) - wrong}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--variant", default="base")
    parser.add_argument("--report", default="")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from tools.eval_synthetic import build_manager

    corpus = Path(args.corpus).resolve()
    keys = json.loads((corpus / "key.json").read_text(encoding="utf-8"))
    real = json.loads(FIXTURE.read_text(encoding="utf-8"))
    keys.append(dict(real, photo=str(FIXTURE.with_suffix(".jpg"))))
    if args.only:
        keys = [k for k in keys if args.only in k["photo"]]
    docs = corpus / "_docs"
    docs.mkdir(exist_ok=True)
    manager, _ = build_manager(docs, True)
    llm = manager.llm_model
    read = variants()[args.variant]
    rows, total = [], {"cells": 0, "cells_ok": 0, "names": 0, "names_ok": 0}
    for key in keys:
        photo = Path(key["photo"]) if Path(key["photo"]).is_absolute() else corpus / key["photo"]
        t0 = time.perf_counter()
        text, seen = read(llm, photo.read_bytes())
        result = score(key, text, seen)
        result.update(photo=photo.name, seconds=round(time.perf_counter() - t0, 1), text=text)
        rows.append(result)
        for k in total:
            total[k] += result[k]
        print(f"{photo.name:<22} cells {result['cells_ok']:>3}/{result['cells']:<3} names {result['names_ok']:>2}/{result['names']:<2} "
              f"rows {result['rows_got']}/{result['rows_want']}  {result['seconds']}s  {'; '.join(result['misses'][:4])}", flush=True)
    summary = {"variant": args.variant, "cells": f"{total['cells_ok']}/{total['cells']}",
               "cell_accuracy": round(total["cells_ok"] / max(1, total["cells"]), 3),
               "names": f"{total['names_ok']}/{total['names']}",
               "seconds": round(sum(r["seconds"] for r in rows), 1)}
    print(json.dumps(summary, ensure_ascii=False))
    print("\n".join(confidence_table(rows)))
    if args.report:
        Path(args.report).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
