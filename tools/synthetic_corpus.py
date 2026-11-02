"""Deterministic synthetic corpus + gold questions for search-quality metrics.

The corpus imitates what testers feed ChatKodi: daily branch reports with a
revenue figure (next to a misleading "plan" figure), a shift roster and a
vehicle log with licence plates written in several spellings. Facts are
buried after filler text, so "document found, value not found" regressions
are visible.

Usage::

    python -m tools.synthetic_corpus --out runtime/synthetic --days 62
    python -m tools.eval_synthetic --corpus runtime/synthetic
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

MONTHS_GEN = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]
MONTHS_PREP = [
    "январе", "феврале", "марте", "апреле", "мае", "июне",
    "июле", "августе", "сентябре", "октябре", "ноябре", "декабре",
]

STAFF = [
    ("Иванов", "Иван", "Петрович", "кладовщик"),
    ("Кузнецова", "Мария", "Андреевна", "кассир"),
    ("Орлов", "Дмитрий", "Викторович", "инженер"),
    ("Смирнова", "Елена", "Олеговна", "менеджер"),
    ("Попов", "Артём", "Сергеевич", "водитель"),
    ("Лебедев", "Павел", "Ильич", "водитель"),
    ("Новикова", "Ольга", "Дмитриевна", "бухгалтер"),
    ("Морозов", "Кирилл", "Алексеевич", "электрик"),
    ("Волкова", "Анна", "Игоревна", "кассир"),
    ("Соколов", "Никита", "Романович", "охранник"),
]

PLATE_LETTERS = "АВЕКМНОРСТУХ"
ROUTES = [
    "склад — магазин №3", "склад — ТЦ «Галактика»", "Москва — Тула", "склад — рынок",
    "офис — банк", "склад — база «Восток»", "Москва — Серпухов", "склад — филиал «Южный»",
]

FILLER = [
    "Проведена плановая проверка противопожарного оборудования, замечаний не выявлено.",
    "В торговом зале выполнена перестановка стеллажей согласно новому планограммному решению.",
    "Поступила партия упаковочных материалов, приёмка проведена в полном объёме.",
    "Отдел закупок согласовал график поставок на следующую неделю.",
    "На складе проведена частичная инвентаризация бытовой химии, расхождений нет.",
    "Сотрудники прошли инструктаж по охране труда и работе с погрузчиком.",
    "Обновлено программное обеспечение кассовых терминалов до последней версии.",
    "Получена претензия покупателя по качеству товара, претензия передана в отдел качества.",
    "Произведена уборка прилегающей территории, вывезен крупногабаритный мусор.",
    "Охрана сообщила о штатной работе системы видеонаблюдения в течение суток.",
    "Бухгалтерия завершила сверку расчётов с поставщиками за прошлую неделю.",
    "Проведено совещание руководителей отделов по итогам работы за неделю.",
    "Выполнен ремонт освещения в зоне погрузки, заменены четыре светильника.",
    "Маркетинговый отдел подготовил макеты акционных ценников на выходные.",
    "Холодильное оборудование работало в штатном режиме, температура в норме.",
]


def _rub(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _short(person: tuple[str, str, str, str]) -> str:
    return f"{person[0]} {person[1][0]}.{person[2][0]}."


def _full(person: tuple[str, str, str, str]) -> str:
    return f"{person[0]} {person[1]} {person[2]}"


def _plate(rng: random.Random) -> tuple[str, str]:
    letters = [rng.choice(PLATE_LETTERS) for _ in range(3)]
    digits = rng.randint(100, 999)
    region = rng.choice(["77", "50", "177", "750", "71"])
    canonical = f"{letters[0]}{digits}{letters[1]}{letters[2]} {region}"
    return canonical, region


def _plate_variants(canonical: str) -> list[str]:
    body, region = canonical.split(" ")
    spaced = f"{body[0]} {body[1:4]} {body[4:]} {region}"
    return [canonical, f"{body}{region}", spaced, canonical.lower()]


@dataclass
class DayRecord:
    day: date
    revenue: int
    plan: int
    checks: int
    staff: list[int]
    trips: list[dict[str, Any]]
    file_name: str
    layout: str = "docx"
    facts_at_end: bool = False


@dataclass
class Corpus:
    days: list[DayRecord] = field(default_factory=list)
    plates: list[str] = field(default_factory=list)


def build_corpus(seed: int = 7, days: int = 62, start: date = date(2025, 7, 1)) -> Corpus:
    rng = random.Random(seed)
    plates = []
    while len(plates) < 12:
        candidate, _region = _plate(rng)
        if candidate not in plates:
            plates.append(candidate)
    corpus = Corpus(plates=plates)
    drivers = [index for index, person in enumerate(STAFF) if person[3] == "водитель"]
    for offset in range(days):
        current = start + timedelta(days=offset)
        if current.weekday() == 6:  # Sunday: branch closed, no report
            continue
        staff = sorted(rng.sample(range(len(STAFF)), rng.randint(3, 5)))
        for driver in drivers:
            if driver not in staff and rng.random() < 0.6:
                staff.append(driver)
        if not any(index in drivers for index in staff):
            staff.append(drivers[0])  # every trip needs a driver who is on shift
        staff = sorted(set(staff))
        on_duty_drivers = [index for index in staff if index in drivers]
        trips = []
        for plate in rng.sample(plates, rng.randint(2, 4)):
            out_hour = rng.randint(6, 14)
            trips.append({
                "plate": plate,
                "spelling": rng.choice(_plate_variants(plate)),
                "driver": rng.choice(on_duty_drivers),
                "route": rng.choice(ROUTES),
                "out": f"{out_hour:02d}:{rng.choice(['00', '15', '30', '45'])}",
                "back": f"{out_hour + rng.randint(2, 6):02d}:{rng.choice(['05', '20', '40', '50'])}",
                "km": rng.randint(18, 260),
            })
        revenue = rng.randint(620, 1950) * 1000 + rng.choice([0, 250, 500, 750]) * 1
        if rng.random() < 0.5:
            revenue += rng.randint(1, 9) * 100
        plan = rng.randint(9, 16) * 100_000
        naming = rng.random()
        if naming < 0.4:
            file_name = f"Ежедневный отчет {current:%d.%m.%Y}.docx"
        elif naming < 0.8:
            file_name = f"report_{current:%m%d}.docx" if days <= 365 else f"report_{current:%Y%m%d}.docx"
        else:
            file_name = f"Отчет филиала №{offset + 1}.docx"
        layout = "txt" if rng.random() < 0.15 else "docx"
        if layout == "txt":
            file_name = file_name.rsplit(".", 1)[0] + ".txt"
        corpus.days.append(
            DayRecord(
                day=current,
                revenue=revenue,
                plan=plan,
                checks=rng.randint(180, 640),
                staff=staff,
                trips=trips,
                file_name=file_name,
                layout=layout,
                facts_at_end=rng.random() < 0.5,
            )
        )
    return corpus


def _day_paragraphs(record: DayRecord, rng: random.Random) -> tuple[list[str], list[list[str]]]:
    d = record.day
    header = [
        "ЕЖЕДНЕВНЫЙ ОТЧЕТ ФИЛИАЛА «СЕВЕРНЫЙ»",
        f"за {d.day} {MONTHS_GEN[d.month - 1]} {d.year} года",
    ]
    staff_line = "Состав смены: " + ", ".join(
        f"{_short(STAFF[i])} ({STAFF[i][3]})" for i in record.staff
    ) + "."
    finance = [
        f"План по выручке на день: {_rub(record.plan)} руб.",
        f"Фактическая выручка за день составила {_rub(record.revenue)} руб.",
        f"Количество чеков: {record.checks}, средний чек {_rub(record.revenue // max(1, record.checks))} руб.",
    ]
    filler = rng.sample(FILLER, k=min(len(FILLER), rng.randint(6, 10)))
    named = rng.choice(record.staff)
    filler.insert(
        rng.randint(0, len(filler)),
        f"Специалист {_full(STAFF[named])} выполнил(а) поручение руководителя филиала.",
    )
    trips_rows = [["Госномер", "Водитель", "Маршрут", "Выезд", "Возврат", "Пробег, км"]]
    for trip in record.trips:
        trips_rows.append([
            trip["spelling"], _short(STAFF[trip["driver"]]), trip["route"], trip["out"], trip["back"], str(trip["km"]),
        ])
    body: list[str] = list(header)
    body.append("1. Общие сведения")
    body.append(staff_line)
    if record.facts_at_end:
        body.append("2. Хозяйственная деятельность")
        body.extend(filler)
        body.append("3. Финансовые показатели")
        body.extend(finance)
    else:
        body.append("2. Финансовые показатели")
        body.extend(finance)
        body.append("3. Хозяйственная деятельность")
        body.extend(filler)
    body.append("4. Работа автотранспорта")
    return body, trips_rows


def write_corpus(corpus: Corpus, out_dir: Path, seed: int = 7) -> Path:
    import docx  # python-docx

    docs_dir = out_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for stale in docs_dir.glob("*"):
        if stale.is_file():
            stale.unlink()
    rng = random.Random(seed + 1)
    for record in corpus.days:
        paragraphs, rows = _day_paragraphs(record, rng)
        path = docs_dir / record.file_name
        if record.layout == "txt":
            lines = list(paragraphs)
            for row in rows[1:]:
                lines.append(
                    f"Автомобиль {row[0]}, водитель {row[1]}, маршрут {row[2]}, выезд {row[3]}, возврат {row[4]}, пробег {row[5]} км."
                )
            path.write_text("\n".join(lines), encoding="utf-8")
            continue
        document = docx.Document()
        for paragraph in paragraphs:
            document.add_paragraph(paragraph)
        table = document.add_table(rows=len(rows), cols=len(rows[0]))
        for r_index, row in enumerate(rows):
            for c_index, value in enumerate(row):
                table.cell(r_index, c_index).text = value
        document.add_paragraph("Отчет составил: руководитель филиала Белов С.Н.")
        document.save(str(path))

    # distractors: a monthly plan and a policy, both full of numbers and dates
    plan = docx.Document()
    plan.add_paragraph("ПЛАН ПРОДАЖ ФИЛИАЛА «СЕВЕРНЫЙ» НА АВГУСТ 2025 ГОДА")
    for record in corpus.days:
        if record.day.month == 8:
            plan.add_paragraph(f"{record.day:%d.%m.%Y}: плановая выручка {_rub(record.plan)} руб.")
    plan.save(str(docs_dir / "План продаж август 2025.docx"))
    policy = docx.Document()
    policy.add_paragraph("РЕГЛАМЕНТ ИСПОЛЬЗОВАНИЯ СЛУЖЕБНОГО АВТОТРАНСПОРТА")
    policy.add_paragraph("Водитель обязан отметить в путевом листе госномер автомобиля, время выезда и возврата.")
    policy.add_paragraph("Пробег свыше 300 км за смену согласуется с руководителем филиала.")
    policy.save(str(docs_dir / "Регламент автотранспорта.docx"))

    cases = build_cases(corpus, seed=seed)
    (out_dir / "cases.json").write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    return docs_dir


def _date_ru(d: date) -> str:
    return f"{d.day} {MONTHS_GEN[d.month - 1]} {d.year} года"


def build_cases(corpus: Corpus, seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed + 2)
    cases: list[dict[str, Any]] = []
    sample = rng.sample(corpus.days, k=min(12, len(corpus.days)))
    for record in sample[:8]:
        phrasing = rng.choice([
            "Какая выручка {d}?",
            "Какая выручка была {d}?",
            "Сколько составила выручка за {d}?",
            "выручка {dd}",
        ])
        question = phrasing.format(d=_date_ru(record.day), dd=f"{record.day:%d.%m.%Y}")
        cases.append({
            "category": "revenue_by_date",
            "question": question,
            "expect_numbers": [record.revenue],
            "forbid_numbers": [record.plan],
            "expected_source": record.file_name,
        })
    for record in sample[8:12]:
        trip = rng.choice(record.trips)
        cases.append({
            "category": "plate_on_date",
            "question": f"Кто был водителем автомобиля {trip['plate']} {_date_ru(record.day)}?",
            "expect_text": [STAFF[trip["driver"]][0]],
            "expected_source": record.file_name,
        })
    for plate in rng.sample(corpus.plates, k=4):
        dates = [r.day for r in corpus.days if (r.day.year, r.day.month) == (2025, 8) and any(t["plate"] == plate for t in r.trips)]
        cases.append({
            "category": "plate_all_dates",
            "question": f"В какие дни в августе 2025 года выезжал автомобиль {plate}?",
            "expect_dates": [d.isoformat() for d in dates],
        })
    for person_index in rng.sample(range(len(STAFF)), k=5):
        person = STAFF[person_index]
        month = rng.choice([7, 8])
        worked = [r.day for r in corpus.days if (r.day.year, r.day.month) == (2025, month) and person_index in r.staff]
        phrasing = rng.choice([
            "Сколько дней работал {name} в {month} 2025 года?",
            "Сколько смен отработал специалист {name} в {month} 2025?",
        ])
        cases.append({
            "category": "person_days_count",
            "question": phrasing.format(name=_short(person), month=MONTHS_PREP[month - 1]),
            "expect_numbers": [len(worked)],
            "expect_dates_optional": [d.isoformat() for d in worked],
        })
    for record in rng.sample(corpus.days, k=4):
        cases.append({
            "category": "shift_on_date",
            "question": f"Кто работал в смене {_date_ru(record.day)}?",
            "expect_text": [STAFF[i][0] for i in record.staff],
            "expected_source": record.file_name,
        })
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the ChatKodi synthetic evaluation corpus")
    parser.add_argument("--out", default="runtime/synthetic")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--days", type=int, default=62)
    args = parser.parse_args()
    corpus = build_corpus(seed=args.seed, days=args.days)
    out = Path(args.out)
    docs = write_corpus(corpus, out, seed=args.seed)
    print(json.dumps({"ok": True, "docs": str(docs), "documents": len(corpus.days) + 2, "cases": str(out / "cases.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
