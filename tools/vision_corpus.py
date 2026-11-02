"""Synthetic "phone photos" of documents for the vision/attachment eval.

Every image is rendered from known content and then degraded like a phone
shot (tilt, perspective, uneven light, noise, blur, JPEG), so recognition can
be scored against exact gold values. Run:

    python -m tools.vision_corpus --out <dir>

The gold facts and questions live in ``tools/vision_cases.json``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = Path("C:/Windows/Fonts")


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


class Page:
    def __init__(self, width: int = 1400, height: int = 1900, paper=(250, 248, 240)):
        self.img = Image.new("RGB", (width, height), paper)
        self.draw = ImageDraw.Draw(self.img)
        self.y = 70
        self.margin = 80

    def text(self, line: str, font: ImageFont.FreeTypeFont, *, gap: int = 12, center: bool = False, x: int | None = None, ink=(25, 25, 30)):
        w = self.draw.textlength(line, font=font)
        left = (self.img.width - w) / 2 if center else (self.margin if x is None else x)
        self.draw.text((left, self.y), line, font=font, fill=ink)
        self.y += int(font.size * 1.25) + gap

    def table(self, header: list[str], rows: list[list[str]], widths: list[int], font, head_font):
        x0 = self.margin
        row_h = int(font.size * 1.9)
        lines = [header] + rows
        top = self.y
        for r, cells in enumerate(lines):
            x = x0
            for c, cell in enumerate(cells):
                self.draw.rectangle([x, self.y, x + widths[c], self.y + row_h], outline=(40, 40, 40), width=2)
                f = head_font if r == 0 else font
                self.draw.text((x + 10, self.y + (row_h - f.size) / 2 - 2), cell, font=f, fill=(20, 20, 25))
                x += widths[c]
            self.y += row_h
        self.y += 20
        return top

    def skip(self, px: int):
        self.y += px


def _phone_shot(img: Image.Image, seed: int, *, tilt: float = 2.0) -> Image.Image:
    rnd = random.Random(seed)
    w, h = img.size
    # uneven light: darker towards one corner
    shade = Image.linear_gradient("L").rotate(rnd.choice([0, 90, 180, 270])).resize((w, h))
    shade = shade.point(lambda v: 255 - int(v * 0.22))
    img = Image.composite(img, Image.new("RGB", (w, h), (0, 0, 0)), shade)
    # canvas around the sheet (table surface)
    pad = 90
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (92, 84, 74))
    canvas.paste(img, (pad, pad))
    canvas = canvas.rotate(rnd.uniform(-tilt, tilt), resample=Image.BICUBIC, fillcolor=(92, 84, 74))
    # mild perspective
    cw, ch = canvas.size
    d = lambda: rnd.uniform(0, 0.025)
    quad = (cw * d(), ch * d(), cw * d(), ch * (1 - d()), cw * (1 - d()), ch * (1 - d()), cw * (1 - d()), ch * d())
    canvas = canvas.transform((cw, ch), Image.QUAD, quad, resample=Image.BICUBIC)
    # sensor noise + slight defocus
    noise = Image.effect_noise((cw, ch), 18).convert("RGB")
    canvas = Image.blend(canvas, noise, 0.06)
    canvas = canvas.filter(ImageFilter.GaussianBlur(0.7))
    return canvas.resize((int(cw * 0.8), int(ch * 0.8)), Image.LANCZOS)


def concrete_protocol() -> Image.Image:
    p = Page()
    b, r, s = _font("arialbd.ttf", 34), _font("arial.ttf", 28), _font("arial.ttf", 26)
    p.text("ООО «СтройЛаб» — испытательная лаборатория", _font("arial.ttf", 26), center=True)
    p.skip(20)
    p.text("ПРОТОКОЛ ИСПЫТАНИЙ № 145/26", b, center=True)
    p.text("от 14.10.2026", r, center=True)
    p.skip(20)
    for line in (
        "Объект: жилой дом, ул. Садовая, 12",
        "Конструкция: плита перекрытия 3 этажа",
        "Класс бетона по проекту: B25",
        "Метод: ГОСТ 10180, образцы-кубы 100×100×100 мм",
        "Возраст бетона: 28 суток; температура в помещении 20 °C",
    ):
        p.text(line, r)
    p.skip(16)
    p.table(
        ["№", "Маркировка", "Масса, г", "Нагрузка, кН", "Прочность, МПа"],
        [["1", "3П-1", "2398", "352,0", "35,2"], ["2", "3П-2", "2405", "338,5", "33,9"],
         ["3", "3П-3", "2387", "361,2", "36,1"], ["4", "3П-4", "2410", "344,8", "34,5"],
         ["5", "3П-5", "2392", "329,9", "33,0"], ["6", "3П-6", "2401", "357,3", "35,7"]],
        [70, 260, 200, 260, 300], s, _font("arialbd.ttf", 26),
    )
    p.text("Средняя прочность серии: 34,7 МПа", _font("arialbd.ttf", 28))
    p.skip(30)
    p.text("Испытание провела: инженер Логинова Е. А.", r)
    p.text("Заведующий лабораторией: Шестаков П. И.", r)
    return p.img


def blood_test() -> Image.Image:
    p = Page(paper=(255, 255, 255))
    b, r, s = _font("calibrib.ttf", 38), _font("calibri.ttf", 30), _font("calibri.ttf", 28)
    p.text("Медицинский центр «Здоровье+»", _font("calibrib.ttf", 30))
    p.text("КЛИНИЧЕСКИЙ АНАЛИЗ КРОВИ", b, center=True)
    p.skip(10)
    p.text("Пациент: Морозов Андрей Викторович, 1984 г. р.", r)
    p.text("Дата взятия материала: 21.10.2026     Заказ № 77412", r)
    p.skip(16)
    p.table(
        ["Показатель", "Результат", "Ед. изм.", "Референсные значения"],
        [["Гемоглобин", "128", "г/л", "130–160"], ["Эритроциты", "4,6", "×10¹²/л", "4,0–5,0"],
         ["Лейкоциты", "11,2", "×10⁹/л", "4,0–9,0"], ["Тромбоциты", "245", "×10⁹/л", "180–320"],
         ["СОЭ", "18", "мм/ч", "2–15"], ["Гематокрит", "41", "%", "40–48"]],
        [330, 220, 220, 420], s, _font("calibrib.ttf", 28),
    )
    p.text("Врач клинической лабораторной диагностики: Белова Н. С.", r)
    p.text("Результаты не являются диагнозом и требуют консультации врача.", _font("calibrii.ttf", 26))
    return p.img


def handwritten_log() -> Image.Image:
    p = Page(1300, 1500, paper=(246, 244, 232))
    # notebook ruling
    for y in range(120, 1500, 62):
        p.draw.line([(0, y), (1300, y)], fill=(185, 205, 230), width=2)
    hand = _font("segoepr.ttf", 40)
    p.y = 66
    ink = (30, 45, 120)
    p.text("Замеры температуры, цех № 3", hand, gap=8, ink=ink)
    p.text("20.10.2026", hand, gap=8, ink=ink)
    for t, v in (("08:00", "18,5°"), ("10:00", "21,0°"), ("12:00", "23,4°"), ("14:00", "24,1°"), ("16:00", "22,7°")):
        p.text(f"{t} — {v}", hand, gap=8, x=140, ink=ink)
    p.text("Вентиляция вкл. в 12:30", hand, gap=8, ink=ink)
    p.text("Мастер смены: Ковалёв", hand, gap=8, ink=ink)
    return p.img


def invoice() -> Image.Image:
    p = Page(paper=(252, 252, 250))
    b, r, s = _font("timesbd.ttf", 38), _font("times.ttf", 30), _font("times.ttf", 28)
    p.text("ООО «Альфа-Офис»   ИНН 7714123456", r)
    p.text("г. Москва, ул. Ленинская, д. 5", r)
    p.skip(20)
    p.text("СЧЁТ НА ОПЛАТУ № 318 от 16.10.2026", b, center=True)
    p.skip(10)
    p.text("Покупатель: ООО «Вектор», ИНН 5029111222", r)
    p.skip(10)
    p.table(
        ["№", "Наименование", "Кол-во", "Цена, руб.", "Сумма, руб."],
        [["1", "Бумага А4, 500 л.", "20 пачек", "345,00", "6 900,00"],
         ["2", "Картридж HP 59A", "3 шт.", "7 450,00", "22 350,00"],
         ["3", "Папка-регистратор", "15 шт.", "210,00", "3 150,00"],
         ["4", "Ручка шариковая", "50 шт.", "32,00", "1 600,00"]],
        [60, 420, 190, 250, 300], s, _font("timesbd.ttf", 28),
    )
    p.text("Итого: 34 000,00 руб.", _font("timesbd.ttf", 32), x=760)
    p.text("В том числе НДС 20%: 5 666,67 руб.", r, x=600)
    p.skip(20)
    p.text("Оплатить до 23.10.2026.", r)
    p.text("Руководитель ______________ Назаров И. В.", r)
    return p.img


def business_letter() -> Image.Image:
    p = Page(paper=(255, 255, 253))
    r = _font("georgia.ttf", 30)
    p.text("АО «Северный терминал»", _font("georgiab.ttf", 32))
    p.text("Исх. № 2041 от 19.10.2026", r)
    p.skip(30)
    p.text("Уважаемый господин Шмидт!", _font("georgiab.ttf", 30))
    p.skip(10)
    for line in (
        "Подтверждаем получение партии из 12 контейнеров 18 октября.",
        "Два контейнера (№ TGHU 482915 и № MSKU 330172) прибыли",
        "с повреждёнными пломбами, акт осмотра прилагаем.",
        "Просим согласовать повторную проверку груза до 25 октября.",
        "Хранение на терминале бесплатно в течение 7 дней.",
    ):
        p.text(line, r, gap=10)
    p.skip(30)
    p.text("С уважением,", r)
    p.text("коммерческий директор О. В. Гусева", r)
    return p.img


def nameplate() -> Image.Image:
    img = Image.new("RGB", (1200, 700), (178, 182, 186))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([30, 30, 1170, 670], 30, outline=(60, 60, 60), width=6)
    f, fb = _font("arial.ttf", 40), _font("arialbd.ttf", 48)
    y = 70
    for line, font in (
        ("НАСОС ЦЕНТРОБЕЖНЫЙ", fb), ("К 80-65-160", fb), ("Подача Q = 50 м³/ч", f), ("Напор H = 32 м", f),
        ("Мощность N = 7,5 кВт", f), ("n = 2900 об/мин", f), ("Зав. № 11873     2019 г.", f),
    ):
        d.text((80, y), line, font=font, fill=(25, 25, 25))
        y += int(font.size * 1.55)
    return img


# A teacher's handwritten grade sheet: the scale and the rounding rule are on the
# sheet itself; one average and two grades are wrong, two grades are missing,
# one student lacks a score. Gold values: GRADE_SHEET_KEY.
GRADE_SHEET_ROWS = [
    # name, К1, К2, К3, average written, grade written
    ("Алексеев П.", "92", "88", "95", "92", "5"),
    ("Борисова А.", "78", "81", "74", "78", "4"),
    ("Васильев Д.", "65", "70", "58", "67", "3"),
    ("Григорьева Е.", "88", "84", "90", "87", "4"),
    ("Дмитриев К.", "45", "52", "40", "46", "3"),
    ("Егорова М.", "71", "", "77", "", ""),
    ("Жуков С.", "84", "85", "85", "", ""),
    ("Зайцева Л.", "69", "72", "70", "70", "4"),
]
GRADE_SHEET_SCALE = "Оценка по среднему баллу: 85–100 — «5», 70–84 — «4», 50–69 — «3», менее 50 — «2»."
GRADE_SHEET_RULE = "Средний балл округляется до целого."
# student -> (average, grade, what the program must do); None = not enough data
GRADE_SHEET_KEY = {
    "Алексеев П.": (92, "5", "ok"),
    "Борисова А.": (78, "4", "ok"),
    "Васильев Д.": (64, "3", "average fixed"),
    "Григорьева Е.": (87, "5", "grade fixed"),
    "Дмитриев К.": (46, "2", "grade fixed"),
    "Егорова М.": (None, None, "missing data"),
    "Жуков С.": (85, "5", "filled"),
    "Зайцева Л.": (70, "4", "ok"),
}


def grade_sheet() -> Image.Image:
    p = Page(1500, 1500, paper=(247, 245, 236))
    hand, small = _font("segoepr.ttf", 38), _font("segoepr.ttf", 32)
    ink = (28, 40, 115)
    p.y = 60
    p.text("Ведомость, группа ТМ-21, «Материаловедение»", hand, gap=14, ink=ink)
    header = ["№", "Студент", "К1", "К2", "К3", "Ср. балл", "Оценка"]
    rows = [[str(i), *row] for i, row in enumerate(GRADE_SHEET_ROWS, 1)]
    widths = [60, 330, 130, 130, 130, 200, 180]
    row_h = 66
    x0 = 60
    for r, cells in enumerate([header] + rows):
        x = x0
        for c, cell in enumerate(cells):
            p.draw.rectangle([x, p.y, x + widths[c], p.y + row_h], outline=(60, 60, 90), width=2)
            if cell:
                p.draw.text((x + 12, p.y + 8), cell, font=small, fill=ink)
            x += widths[c]
        p.y += row_h
    p.y += 30
    p.text("Оценка по среднему баллу:", hand, gap=6, ink=ink)
    p.text("85–100 — «5», 70–84 — «4»,", hand, gap=6, x=120, ink=ink)
    p.text("50–69 — «3», менее 50 — «2».", hand, gap=6, x=120, ink=ink)
    p.text(GRADE_SHEET_RULE, hand, gap=6, ink=ink)
    p.text("Преподаватель: Смирнова О. В.", hand, gap=6, ink=ink)
    return p.img


def _hand_table(p: Page, header: list[str], rows: list[list[str]], widths: list[int], font, ink, row_h: int = 64):
    for cells in [header] + rows:
        x = 60
        for c, cell in enumerate(cells):
            p.draw.rectangle([x, p.y, x + widths[c], p.y + row_h], outline=(60, 60, 90), width=2)
            if cell:
                p.draw.text((x + 12, p.y + 8), cell, font=font, fill=ink)
            x += widths[c]
        p.y += row_h
    p.y += 30


# an exam sheet with points and grades in words; the scale is NOT on the sheet (the user gives it)
EXAM_ROWS = [("Белов А.", "93", "отл."), ("Волкова Н.", "81", "хор."), ("Гусев П.", "58", "удовл."),
             ("Демина О.", "74", "хор."), ("Ершов В.", "", "неуд."), ("Жданова К.", "90", "")]


def exam_sheet() -> Image.Image:
    p = Page(1400, 1150, paper=(248, 246, 238))
    hand, small = _font("segoepr.ttf", 38), _font("segoepr.ttf", 32)
    ink = (30, 35, 110)
    p.y = 60
    p.text("Экзаменационная ведомость, «Химия», гр. 2-А", hand, gap=20, ink=ink)
    _hand_table(p, ["№", "Студент", "Баллы", "Оценка"], [[str(i), *r] for i, r in enumerate(EXAM_ROWS, 1)],
                [70, 420, 200, 300], small, ink)
    p.text("Экзаменатор: Орлов Д. С.", hand, gap=6, ink=ink)
    return p.img


# a 1000 m run in minutes:seconds, scale on the sheet; one absence
RUN_ROWS = [("Антонов", "3:28", "5"), ("Быкова", "3:50", "5"), ("Власов", "4:35", ""), ("Голубева", "н/я", "3"),
            ("Дорохов", "3:51", "")]


def run_sheet() -> Image.Image:
    p = Page(1400, 1150, paper=(247, 245, 236))
    hand, small = _font("segoepr.ttf", 36), _font("segoepr.ttf", 32)
    ink = (28, 40, 115)
    p.y = 60
    p.text("Бег 1000 м, 10 «А», 14.10", hand, gap=10, ink=ink)
    p.text("до 3:30 — «5»; 3:31–3:50 — «4»; 3:51–4:20 — «3»; больше 4:20 — «2»", small, gap=24, ink=ink)
    _hand_table(p, ["№", "Фамилия", "Время", "Оценка"], [[str(i), *r] for i, r in enumerate(RUN_ROWS, 1)],
                [70, 380, 220, 220], small, ink)
    return p.img


IMAGES = {
    "protokol_ispytaniy_betona.jpg": (concrete_protocol, 2.5),
    "analiz_krovi.jpg": (blood_test, 2.0),
    "zamery_temperatury.jpg": (handwritten_log, 3.0),
    "schet_318.jpg": (invoice, 1.5),
    "pismo_terminal.jpg": (business_letter, 2.0),
    "shildik_nasosa.jpg": (nameplate, 4.0),
    "vedomost_ocenok.jpg": (grade_sheet, 2.0),
    "ekzamen_himiya.jpg": (exam_sheet, 1.5),
    "beg_1000.jpg": (run_sheet, 1.5),
}


def build(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (name, (render, tilt)) in enumerate(IMAGES.items()):
        shot = _phone_shot(render(), seed=100 + index, tilt=tilt)
        path = out / name
        shot.save(path, "JPEG", quality=72)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for path in build(Path(args.out)):
        print(path)


if __name__ == "__main__":
    main()
