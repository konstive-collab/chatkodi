"""LLM-focused eval set from the synthetic corpus: day retellings and paraphrased
questions whose answers need reading, not lookup (22 cases).

    python tools/llm_eval_corpus.py <synthetic corpus folder> <output folder>
    python tools/eval_synthetic.py --corpus <output folder> --with-llm
"""
import json
import random
import re
import shutil
import sys
from pathlib import Path

from docx import Document

SRC = Path(sys.argv[1])          # synthetic corpus folder (with docs/)
OUT = Path(sys.argv[2])          # output folder
KEYS = {
    "противопожарного": "противопожар", "стеллажей": "стеллаж", "упаковочных": "упаков",
    "закупок": "поставок", "бытовой химии": "инвентаризац", "инструктаж": "инструктаж",
    "кассовых терминалов": "касс", "претензия": "претензи", "уборка": "уборк",
    "видеонаблюдения": "видеонаблюд", "сверку расчётов": "сверк", "совещание": "совещани",
    "освещения": "освещени", "ценников": "ценник", "Холодильное": "холодильн",
}
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
          "сентября", "октября", "ноября", "декабря"]


def doc_text(path: Path) -> str:
    if path.suffix == ".txt":
        return path.read_text(encoding="utf-8")
    return "\n".join(p.text for p in Document(str(path)).paragraphs)


days = {}
for path in sorted((SRC / "docs").iterdir()):
    text = doc_text(path)
    m = re.search(r"за (\d{1,2}) ([а-я]+) (\d{4}) года", text)
    if not m:
        continue
    found = [stem for marker, stem in KEYS.items() if marker in text]
    days[(int(m.group(1)), m.group(2))] = found

rng = random.Random(11)
cases = []
for (day, month), stems in rng.sample(sorted(days.items()), 12):
    picks = rng.sample(stems, 3)
    q = rng.choice([
        f"Что происходило в филиале {day} {month}?",
        f"Перескажи отчёт за {day} {month}",
        f"Какие хозяйственные работы были {day} {month}?",
    ])
    # A good summary mentions at least 2 of 3 sampled events; graded with expect_any below.
    cases.append({"category": "summary_events", "question": q, "expect_any": picks, "min_hits": 2})

semantic = [
    ("Жаловались ли покупатели на товар?", ["претензи"]),
    ("Чинили ли свет в каком-нибудь помещении?", ["светильник", "освещени"]),
    ("Обновляли ли софт на кассах?", ["программн", "касс"]),
    ("Проверяли ли остатки бытовой химии на складе, и нашли ли расхождения?", ["расхождений нет", "инвентаризац"]),
    ("Обучали ли сотрудников технике безопасности?", ["инструктаж"]),
    ("Были ли проблемы с холодильниками?", ["штатн", "в норме"]),
    ("Что делал маркетинг к выходным?", ["ценник"]),
    ("Нашли ли нарушения при проверке пожарного оборудования?", ["не выявлено", "замечаний не"]),
    ("Как отработала охрана с камерами?", ["штатн"]),
    ("Что сделала бухгалтерия с поставщиками?", ["сверк"]),
]
for q, expect in semantic:
    cases.append({"category": "semantic_paraphrase", "question": q, "expect_any": expect, "min_hits": 1})

OUT.mkdir(parents=True, exist_ok=True)
if not (OUT / "docs").exists():
    shutil.copytree(SRC / "docs", OUT / "docs")
(OUT / "cases.json").write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
print(len(cases), "cases")

