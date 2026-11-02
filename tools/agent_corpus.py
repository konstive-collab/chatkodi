"""Evaluation corpus for the assistant: a library (norms, document templates,
plus the diverse documents as unrelated material) and an inbox of files the
user drops into the chat (phone photos, a scanned PDF, a DOCX memo).

    python -m tools.agent_corpus --out <folder>
    python tools/eval_agent.py --corpus <folder>

Scenarios and gold checks: ``tools/agent_cases.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from tools import diverse_corpus, vision_corpus

CASES_FILE = Path(__file__).with_name("agent_cases.json")
FIZRA_SHEET = [
    ["№", "Фамилия", "Бег 60 м, с", "Подтяг., раз", "Прыжок, см", "Оценка"],
    ["1", "Иванов", "8,4", "11", "225", "5"], ["2", "Петрова", "8,9", "8", "205", ""],
    ["3", "Сидоров", "9,3", "5", "190", "4"], ["4", "Кузнецова", "8,6", "10", "215", "4"],
    ["5", "Морозов", "9,8", "3", "170", ""], ["6", "Лебедева", "8,5", "7", "200", "5"],
    ["7", "Новиков", "9,0", "6", "", "3"], ["8", "Орлова", "8,3", "9", "230", ""],
]
FIXTURES = Path(__file__).with_name("fixtures")
# a user's test: fitness norms of a 9th grade, the same scales as on the photo vedomost_fizra.jpg
FIZRA_NORMS = """Нормативы по физической культуре, 9 класс. Оценка сдачи нормативов.

Бег 60 м: до 8,5 с включительно — оценка 5; 8,6–9,0 с — оценка 4; 9,1–9,6 с — оценка 3; больше 9,6 с — оценка 2.
Подтягивания: 10 раз и более — оценка 5; 7–9 раз — оценка 4; 4–6 раз — оценка 3; меньше 4 раз — оценка 2.
Прыжок в длину с места: 220 см и более — оценка 5; 200–219 см — оценка 4; 180–199 см — оценка 3; меньше 180 см — оценка 2.

Итоговая оценка — среднее арифметическое трёх оценок за упражнения, округляется до целого числа (0,5 округляется вверх).
Если результат хотя бы одного упражнения не указан, итоговую оценку рассчитать нельзя.
"""


def _docx(path: Path, blocks: list) -> None:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(13)
    for block in blocks:
        if isinstance(block, list):
            grid = doc.add_table(rows=len(block), cols=len(block[0]))
            grid.style = doc.styles["Table Grid"]
            for r, row in enumerate(block):
                for c, value in enumerate(row):
                    grid.cell(r, c).text = value
        elif block.startswith("# "):
            doc.add_heading(block[2:], level=1)
        elif block.startswith("## "):
            doc.add_heading(block[3:], level=2)
        else:
            doc.add_paragraph(block)
    doc.save(str(path))


def _pptx(path: Path, background: str, slides: list) -> None:
    """An example presentation: 16:9, its own background colour, a title slide and content slides."""
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    fill = prs.slide_master.background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor.from_string(background)
    title, subtitle = slides[0]
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle
    for heading, body in slides[1:]:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = heading
        frame = slide.placeholders[1].text_frame
        frame.text = body[0]
        for line in body[1:]:
            frame.add_paragraph().text = line
    prs.save(str(path))


SLIDES_EXAMPLE_BACKGROUND = "1F3A5F"
SLIDES_ATTACHED_BACKGROUND = "5B2333"


def write_library(docs: Path) -> None:
    _pptx(docs / "Итоги месяца отдела логистики (образец презентации).pptx", SLIDES_EXAMPLE_BACKGROUND, [
        ("Итоги работы отдела логистики за сентябрь 2026", "ООО «Вектор», 01.10.2026"),
        ("Ключевые показатели", ["Отгружено заказов: 980 (план 1 000)", "Доля возвратов: 2,3 %", "Средний срок доставки: 2,4 дня"]),
        ("Проблемы", ["Задержки поставок упаковки", "Нехватка водителей в выходные"]),
        ("Решения и планы", ["Второй поставщик упаковки с 15.10", "Набор двух водителей"]),
    ])
    _docx(docs / "Положение о контроле прочности бетона.docx", [
        "# Положение о контроле прочности бетона на объектах ООО «СтройЛаб»",
        "1. Общие положения",
        "1.1. Контроль прочности бетона ведётся по результатам испытаний серий образцов-кубов в возрасте 28 суток.",
        "2. Требования к прочности",
        "2.1. Для бетона класса B25 средняя прочность серии должна быть не менее 32,7 МПа.",
        "2.2. Для бетона класса B30 средняя прочность серии должна быть не менее 39,2 МПа.",
        "2.3. Прочность каждого образца в серии должна быть не менее 28,0 МПа для класса B25 и не менее 33,6 МПа для класса B30.",
        "2.4. Серия должна состоять не менее чем из 6 образцов.",
        "3. Оформление результатов",
        "3.1. По результатам испытаний лаборатория составляет заключение о прочности бетона по установленной форме "
        "в течение 3 рабочих дней после испытания.",
        "3.2. Заключение подписывают инженер, проводивший испытание, и заведующий лабораторией.",
    ])
    _docx(docs / "Заключение о прочности бетона (образец).docx", [
        "# ЗАКЛЮЧЕНИЕ № 12/26 о прочности бетона",
        "Дата: 02.09.2026",
        "Объект: склад, ул. Промышленная, 7",
        "Конструкция: фундаментная плита",
        "Основание: протокол испытаний № 98/26 от 01.09.2026",
        "Класс бетона по проекту: B30",
        "## Результаты испытаний",
        [["Показатель", "Фактическое значение", "Требование", "Вывод"],
         ["Средняя прочность серии, МПа", "41,5", "не менее 39,2", "соответствует"],
         ["Минимальная прочность образца, МПа", "37,8", "не менее 33,6", "соответствует"],
         ["Количество образцов", "6", "не менее 6", "соответствует"]],
        "## Вывод",
        "Бетон конструкции соответствует классу B30. Конструкция может быть принята.",
        "Инженер лаборатории ____________ Иванова Т. С.",
        "Заведующий лабораторией ____________ Шестаков П. И.",
    ])
    _docx(docs / "Инструкция по охране труда в цехе 3.docx", [
        "# Инструкция по охране труда для работников цеха № 3",
        "1. Микроклимат",
        "1.1. Температура воздуха в рабочей зоне цеха должна быть от 17 до 23 °C.",
        "1.2. При температуре выше 23 °C мастер смены обязан включить приточную вентиляцию.",
        "1.3. Замеры температуры проводятся каждые 2 часа с 08:00 до 16:00 и записываются в журнал.",
        "2. Средства защиты",
        "2.1. Работа на станках допускается только в защитных очках и перчатках.",
        "2.2. Ответственный за выдачу средств защиты — кладовщик цеха.",
    ])
    _docx(docs / "Положение о закупке канцелярских товаров.docx", [
        "# Положение о закупке канцелярских товаров и расходных материалов",
        "1. Счёт на сумму не более 30 000 руб. оплачивается без согласования с финансовым директором.",
        "2. Счета на большую сумму оплачиваются только после согласования с финансовым директором.",
        "3. Картриджи закупаются не более 2 шт. в месяц на один отдел.",
        "4. Счёт оплачивается не позднее 5 рабочих дней с даты получения.",
        "5. Поставщиком канцелярских товаров по договору является ООО «Альфа-Офис».",
    ])
    _docx(docs / "Ответ на претензию (образец письма).docx", [
        "ООО «Вектор»",
        "Исх. № 1180 от 03.07.2026",
        "Генеральному директору ООО «Север» Петрову А. А.",
        "Уважаемый Андрей Алексеевич!",
        "Ваше письмо от 28.06.2026 получено и рассмотрено.",
        "Сообщаем, что повторная приёмка товара будет проведена 07.07.2026 в присутствии вашего представителя.",
        "Просим подтвердить участие до 05.07.2026.",
        "С уважением,",
        "директор по логистике ____________ Кравцова Е. Н.",
    ])
    (docs / "Нормативы_физкультура.txt").write_text(FIZRA_NORMS, encoding="utf-8")
    # an example of the kind «методические рекомендации» on another topic: its sections are the form
    _docx(docs / "Методические рекомендации по проведению учебной эвакуации (образец).docx", [
        "# Методические рекомендации по проведению учебной эвакуации персонала",
        "## 1. Общие положения",
        "Рекомендации определяют порядок подготовки и проведения учебной эвакуации работников офиса.",
        "## 2. Организация и подготовка",
        "Руководитель тренировки назначается приказом. За неделю до тренировки проверяются эвакуационные выходы.",
        "## 3. Порядок проведения",
        "По сигналу оповещения работники покидают помещения по планам эвакуации и собираются у места сбора.",
        "## 4. Контроль и оценка",
        "Время эвакуации фиксируется; результаты разбираются с руководителями подразделений.",
        "## 5. Заключительные положения",
        "Учебная эвакуация проводится не реже двух раз в год.",
    ])


def write_inbox(inbox: Path) -> None:
    from docx import Document
    from PIL import Image

    vision_corpus.build(inbox)
    # a user's photo of a handwritten fitness sheet with the scales above the table, and the
    # same sheet without the scales (they are only in the library then)
    shutil.copy2(FIXTURES / "vedomost_fizra.jpg", inbox / "vedomost_fizra.jpg")
    # a user's real phone photo of a handwritten sheet shot sideways (key: fixtures/vedomost_bokom.json)
    shutil.copy2(FIXTURES / "vedomost_bokom.jpg", inbox / "vedomost_bokom.jpg")
    # the same sheet typed into Excel: with the scales in the rows above the table, and without them
    from openpyxl import Workbook

    for name, with_scales in (("vedomost_fizra.xlsx", True), ("vedomost_fizra_bez_shkal.xlsx", False)):
        wb = Workbook()
        ws = wb.active
        ws.title = "9Б"
        ws.append(["Ведомость сдачи нормативов — 9 «Б» класс"])
        if with_scales:
            for line in FIZRA_NORMS.strip().splitlines()[2:5] + [FIZRA_NORMS.strip().splitlines()[6]]:
                ws.append([line])
        ws.append([])
        for row in FIZRA_SHEET:
            ws.append([float(v.replace(",", ".")) if re.fullmatch(r"\d+,\d", v) else int(v) if v.isdigit() else (v or None) for v in row])
        wb.save(inbox / name)
    with Image.open(inbox / "vedomost_fizra.jpg") as img:
        img.crop((0, 400, img.width, img.height)).save(inbox / "vedomost_fizra_bez_shkal.jpg", "JPEG", quality=90)
    # a scanned PDF: the invoice photo as an image-only page
    with Image.open(inbox / "schet_318.jpg") as img:
        img.convert("RGB").save(inbox / "skan_scheta.pdf", "PDF", resolution=150)
    doc = Document()
    for text in (
        "Служебная записка",
        "Директору ООО «Вектор» Назарову И. В.",
        "В связи с приёмом двух новых сотрудников в отдел продаж с 01.11.2026 прошу выделить 2 ноутбука и "
        "2 рабочих места, а также оформить им пропуска до 30.10.2026.",
        "Ориентировочная стоимость ноутбуков — 140 000 руб.",
        "Начальник отдела продаж Сорокина М. Ю., 22.10.2026",
    ):
        doc.add_paragraph(text)
    doc.save(str(inbox / "sluzhebnaya_zapiska.docx"))

    # a month report to turn into slides, and an example deck of another look sent along with it
    _docx(inbox / "otchet_sklad_oktyabr.docx", [
        "# Отчёт о работе склада № 2 за октябрь 2026",
        "Подготовил: начальник склада Гусев О. В., 28.10.2026",
        "## Показатели",
        [["Показатель", "План", "Факт"],
         ["Отгружено заказов", "1 200", "1 240"],
         ["Доля возвратов, %", "2,0", "1,8"],
         ["Средний срок сборки заказа, ч", "4,0", "3,6"]],
        "## Проблемы",
        "В октябре дважды не хватило погрузчиков на вечерней смене; 14 заказов отгружены с опозданием на сутки.",
        "Поставщик стрейч-плёнки сорвал поставку 17.10.2026.",
        "## Предложения",
        "Арендовать один погрузчик на вечернюю смену с 01.11.2026 (85 000 руб. в месяц).",
        "Заключить договор с резервным поставщиком плёнки.",
    ])
    _pptx(inbox / "obrazec_prezentacii.pptx", SLIDES_ATTACHED_BACKGROUND, [
        ("Отчёт подразделения", "Отдел, дата"),
        ("Цифры месяца", ["Показатель 1", "Показатель 2"]),
        ("Что мешало", ["Проблема 1"]),
        ("Что предлагаем", ["Предложение 1"]),
    ])

    base = [
        "# ДОГОВОР ОКАЗАНИЯ УСЛУГ № 77/26",
        "г. Москва, 01.10.2026",
        "ООО «Вектор» (Заказчик) и ООО «КлинСервис» (Исполнитель) заключили настоящий договор.",
        "1. Исполнитель оказывает услуги по уборке офисных помещений площадью 850 кв. м.",
        "2. Стоимость услуг составляет 1 200 000 руб. в год, оплата ежемесячно.",
        "3. Срок начала оказания услуг — не позднее 30 дней с даты подписания.",
        "4. За нарушение сроков Исполнитель уплачивает неустойку 0,1 % от стоимости за каждый день.",
        "5. Договор действует до 31.12.2027.",
        "Подписи сторон: ____________ / ____________",
    ]
    _docx(inbox / "dogovor_redakciya_1.docx", base)
    changed = list(base)
    changed[4] = "2. Стоимость услуг составляет 1 350 000 руб. в год, оплата ежеквартально."
    changed[5] = "3. Срок начала оказания услуг — не позднее 45 дней с даты подписания."
    changed[6] = "4. За нарушение сроков Исполнитель уплачивает неустойку 0,05 % от стоимости за каждый день."
    changed.insert(8, "6. Исполнитель обязан застраховать ответственность перед третьими лицами.")
    _docx(inbox / "dogovor_redakciya_2.docx", changed)


def write_corpus(out: Path) -> Path:
    diverse_corpus.write_corpus(out)  # out/docs with the diverse documents
    docs = out / "docs"
    write_library(docs)
    inbox = out / "inbox"
    if inbox.exists():
        shutil.rmtree(inbox)
    write_inbox(inbox)
    (out / "agent_cases.json").write_text(CASES_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    out = Path(parser.parse_args().out).resolve()
    docs = write_corpus(out)
    print(json.dumps({"docs": str(docs), "documents": len(list(docs.iterdir())), "inbox": len(list((out / "inbox").iterdir()))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
