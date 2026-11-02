"""A large bank of tester-style scenarios with ground truth from the corpus.

``tools.synthetic_corpus`` writes the documents and 25 template questions.
Testers ask far more varied things: keyword queries, typos, short dates,
totals and maxima over a month, rankings, comparisons, follow-ups («а 15-го?»),
questions about things that are not in the documents, small talk. This module
builds ~300 such scenarios; every expected value is computed from the corpus
generator itself, never typed by hand.

Usage::

    python -m tools.synthetic_corpus --out runtime/synthetic --days 62
    python -m tools.scenario_bank --out runtime/synthetic/scenarios.json
    python -m tools.gui_e2e --docs runtime/synthetic/docs --questions runtime/synthetic/scenarios.json --out runtime/gui_e2e
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.synthetic_corpus import MONTHS_GEN, MONTHS_PREP, STAFF, build_corpus

_LATIN_TO_CYRILLIC = str.maketrans("ABEKMHOPCTYX", "АВЕКМНОРСТУХ")
_MONTHS_NOM = [
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
]
NOT_FOUND_MARKERS = ["не найден", "не найдено", "нет данных", "нет информации", "не удалось", "отсутству", "не содержат"]


def _date_ru(day) -> str:
    return f"{day.day} {MONTHS_GEN[day.month - 1]} {day.year} года"


def _date_short_ru(day) -> str:
    return f"{day.day} {MONTHS_GEN[day.month - 1]}"


def _typo_month(day, rng: random.Random) -> str:
    month = MONTHS_GEN[day.month - 1]
    if len(month) > 5:
        position = rng.randint(2, len(month) - 2)
        month = month[:position] + month[position + 1:]
    return f"{day.day} {month} {day.year}"


def _short(person) -> str:
    return f"{person[0]} {person[1][0]}.{person[2][0]}."


def _full(person) -> str:
    return f"{person[0]} {person[1]} {person[2]}"


def _plate_display(plate: str) -> str:
    compact = plate.replace(" ", "").upper().translate(_LATIN_TO_CYRILLIC)
    return f"{compact[:6]} {compact[6:]}"


def build_scenarios(seed: int = 7, days: int = 62, scenario_seed: int = 11) -> list[dict[str, Any]]:
    corpus = build_corpus(seed=seed, days=days)
    rng = random.Random(scenario_seed)
    cases: list[dict[str, Any]] = []

    def add(category: str, question: str, **expect: Any) -> None:
        cases.append({"id": f"{category}_{len(cases) + 1:03d}", "category": category, "question": question, **expect})

    by_day = {record.day: record for record in corpus.days}
    months = sorted({(record.day.year, record.day.month) for record in corpus.days})

    # 1. Revenue by date in the ways testers actually type it.
    for record in corpus.days:
        add("revenue_formal", f"Какая выручка {_date_ru(record.day)}?", expect_numbers=[record.revenue], forbid_numbers=[record.plan])
        add("revenue_keyword", f"выручка {record.day:%d.%m.%Y}", expect_numbers=[record.revenue])
    for record in rng.sample(corpus.days, k=min(20, len(corpus.days))):
        add("revenue_colloquial", f"Сколько заработал филиал {_date_short_ru(record.day)} {record.day.year}?", expect_numbers=[record.revenue])
        add("revenue_typo", f"скока выручки было {_typo_month(record.day, rng)}", expect_numbers=[record.revenue])
        add("revenue_short_date", f"Покажи выручку за {record.day:%d.%m}", expect_numbers=[record.revenue])

    # 2. Plan vs actual.
    for record in rng.sample(corpus.days, k=min(12, len(corpus.days))):
        verdict = "не выполнен" if record.revenue < record.plan else "план выполнен"
        add("plan_vs_fact", f"Выполнен ли план продаж {_date_ru(record.day)}?", expect_numbers=[record.plan, record.revenue], expect_text=[verdict])

    # 3. Days worked per person and month (short and full name).
    for year, month in months:
        for index, person in enumerate(STAFF):
            worked = [r.day for r in corpus.days if (r.day.year, r.day.month) == (year, month) and index in r.staff]
            question = f"Сколько дней работал {_short(person)} в {MONTHS_PREP[month - 1]} {year} года?"
            if worked:
                add("person_days", question, expect_numbers=[len(worked)])
            else:
                add("person_days", question, expect_text_any=NOT_FOUND_MARKERS)
    for person_index in rng.sample(range(len(STAFF)), k=4):
        person = STAFF[person_index]
        year, month = rng.choice(months)
        worked = [r.day for r in corpus.days if (r.day.year, r.day.month) == (year, month) and person_index in r.staff]
        if worked:
            add("person_days_fullname", f"Сколько смен отработал {_full(person)} в {MONTHS_PREP[month - 1]} {year}?", expect_numbers=[len(worked)])

    # 4. Driver of a vehicle on a date.
    trips = [(record, trip) for record in corpus.days for trip in record.trips]
    for record, trip in rng.sample(trips, k=min(25, len(trips))):
        add("plate_driver", f"Кто был водителем автомобиля {trip['spelling']} {_date_ru(record.day)}?", expect_text=[STAFF[trip["driver"]][0]])

    # 5. Days a vehicle went out in a month.
    for plate in rng.sample(corpus.plates, k=min(6, len(corpus.plates))):
        year, month = months[-1]
        dates = sorted({r.day for r in corpus.days if (r.day.year, r.day.month) == (year, month) and any(t["plate"] == plate for t in r.trips)})
        if dates:
            add("plate_dates", f"В какие дни в {MONTHS_PREP[month - 1]} {year} года выезжал автомобиль {plate}?", expect_dates=[d.isoformat() for d in dates])

    # 6. Totals, averages, maxima, minima per month.
    for year, month in months:
        rows = [r for r in corpus.days if (r.day.year, r.day.month) == (year, month)]
        label = f"{MONTHS_PREP[month - 1]} {year}"
        total = sum(r.revenue for r in rows)
        best = max(rows, key=lambda r: r.revenue)
        worst = min(rows, key=lambda r: r.revenue)
        add("aggregate_sum", f"Какая общая выручка в {label}?", expect_numbers=[total])
        add("aggregate_sum", f"Сколько всего выручки за {_MONTHS_NOM[month - 1]} {year}?", expect_numbers=[total])
        add("aggregate_avg", f"Средняя дневная выручка в {label}?", expect_numbers=[round(total / len(rows))])
        add("aggregate_max", f"В какой день {MONTHS_GEN[month - 1]} {year} была самая большая выручка?", expect_numbers=[best.revenue], expect_text=[f"{best.day:%d.%m.%Y}"])
        add("aggregate_min", f"Какая минимальная выручка в {label}?", expect_numbers=[worst.revenue], expect_text=[f"{worst.day:%d.%m.%Y}"])

    # 7. Rankings: drivers and vehicles with the most trips.
    for year, month in months:
        label = f"{MONTHS_PREP[month - 1]} {year}"
        drivers = Counter(STAFF[t["driver"]][0] for r in corpus.days if (r.day.year, r.day.month) == (year, month) for t in r.trips)
        top = drivers.most_common(2)
        if top and (len(top) == 1 or top[0][1] != top[1][1]):
            add("ranking_driver", f"Какой водитель чаще всех выезжал в {label}?", expect_text=[top[0][0]], expect_numbers=[top[0][1]])
        plates = Counter(t["plate"] for r in corpus.days if (r.day.year, r.day.month) == (year, month) for t in r.trips)
        top_plates = plates.most_common(2)
        if top_plates and (len(top_plates) == 1 or top_plates[0][1] != top_plates[1][1]):
            add("ranking_vehicle", f"Какая машина чаще всех выезжала в {label}?", expect_text=[_plate_display(top_plates[0][0])], expect_numbers=[top_plates[0][1]])

    # 8. All routes of a vehicle.
    for plate in rng.sample(corpus.plates, k=min(6, len(corpus.plates))):
        routes = Counter(t["route"] for r in corpus.days for t in r.trips if t["plate"] == plate)
        if routes:
            add("entity_routes", f"Какие маршруты были у машины {plate}?", expect_text=[routes.most_common(1)[0][0]], expect_numbers=[len(routes)])

    # 9. Comparisons of two days.
    for _ in range(10):
        first, second = sorted(rng.sample(corpus.days, k=2), key=lambda r: r.day)
        add(
            "compare_days",
            f"Сравни выручку {_date_short_ru(first.day)} и {_date_ru(second.day)}",
            expect_numbers=[first.revenue, second.revenue, abs(second.revenue - first.revenue)],
        )

    # 10. Follow-up chains (the second question depends on the first).
    by_month: dict[tuple[int, int], list] = defaultdict(list)
    for record in corpus.days:
        by_month[(record.day.year, record.day.month)].append(record)
    for _ in range(10):
        month_rows = by_month[rng.choice(months)]
        first, second = rng.sample(month_rows, k=2)
        add("followup_base", f"Какая выручка {_date_ru(first.day)}?", expect_numbers=[first.revenue])
        add("followup", f"а {second.day.day}-го?", expect_numbers=[second.revenue], depends_on_previous=True)

    # 11. Things that are not in the documents.
    last_day = max(by_day)
    for offset in (5, 9, 14):
        missing = last_day.replace(day=1)
        missing = missing.replace(month=missing.month % 12 + 1, year=missing.year + (missing.month == 12), day=offset)
        add("negative_date", f"Какая выручка была {_date_ru(missing)}?", expect_text_any=NOT_FOUND_MARKERS)
    for name in ("Петров С.С.", "Сидорова А.В."):
        add("negative_person", f"Сколько дней работал {name} в {MONTHS_PREP[months[0][1] - 1]} {months[0][0]} года?", expect_text_any=NOT_FOUND_MARKERS)
    for question in ("сколько стоит слон на луне", "Какая погода будет завтра в Москве?", "Кто выиграл чемпионат мира по футболу?"):
        add("negative_offtopic", question, expect_text_any=NOT_FOUND_MARKERS, forbid_text=["подтверждённых фактов", "подтвержденных фактов"])

    # 12. Small talk.
    add("small_talk", "Привет!", expect_text_any=["ChatKodi"])
    add("small_talk", "Что ты умеешь?", expect_text_any=["умею"])
    add("small_talk", "Спасибо!", expect_text_any=["Пожалуйста"])

    # 13. Retelling a day report: must mention the revenue of that day.
    for record in rng.sample(corpus.days, k=4):
        add("summary_day", f"Кратко перескажи отчёт за {_date_ru(record.day)}", expect_numbers=[record.revenue])

    # 14. Shift composition.
    for record in rng.sample(corpus.days, k=6):
        add("shift_on_date", f"Кто работал в смене {_date_ru(record.day)}?", expect_text=[STAFF[i][0] for i in record.staff])

    # 15. Regulation and authorship (undated documents / document footers).
    add("policy", "Какой пробег за смену нужно согласовывать с руководителем?", expect_numbers=[300])
    add("policy", "Что водитель должен отмечать в путевом листе?", expect_text=["госномер"])
    add("author", "Кто составляет ежедневные отчеты филиала?", expect_text=["Белов"])
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the ChatKodi tester-style scenario bank")
    parser.add_argument("--out", default="runtime/synthetic/scenarios.json")
    parser.add_argument("--seed", type=int, default=7, help="corpus seed (must match tools.synthetic_corpus)")
    parser.add_argument("--days", type=int, default=62)
    args = parser.parse_args()
    cases = build_scenarios(seed=args.seed, days=args.days)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    categories = Counter(case["category"] for case in cases)
    print(json.dumps({"ok": True, "cases": len(cases), "out": str(out), "categories": categories}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
