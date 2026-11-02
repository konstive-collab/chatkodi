from __future__ import annotations

from datetime import datetime, timedelta
import re
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as dateutil_parse
from typing import Optional, Tuple, List, Dict, Any

MONTHS_RU = {
    'январь': 1, 'января': 1,
    'февраль': 2, 'февраля': 2,
    'март': 3, 'марта': 3,
    'апрель': 4, 'апреля': 4,
    'май': 5, 'мая': 5,
    'июнь': 6, 'июня': 6,
    'июль': 7, 'июля': 7,
    'август': 8, 'августа': 8,
    'сентябрь': 9, 'сентября': 9,
    'октябрь': 10, 'октября': 10,
    'ноябрь': 11, 'ноября': 11,
    'декабрь': 12, 'декабря': 12,
}

MONTHS_EN = {
    'january': 1, 'jan': 1,
    'february': 2, 'feb': 2,
    'march': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'may': 5,
    'june': 6, 'jun': 6,
    'july': 7, 'jul': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10,
    'november': 11, 'nov': 11,
    'december': 12, 'dec': 12,
}

MONTHS = {**MONTHS_RU, **MONTHS_EN}

# Предложный падеж: «в августе», «в мае» — так спрашивают про месяц целиком.
MONTHS_RU_PREPOSITIONAL = {
    'январе': 1, 'феврале': 2, 'марте': 3, 'апреле': 4, 'мае': 5, 'июне': 6,
    'июле': 7, 'августе': 8, 'сентябре': 9, 'октябре': 10, 'ноябре': 11, 'декабре': 12,
}
_MONTHS_PERIOD = {**MONTHS, **MONTHS_RU_PREPOSITIONAL}


def _month_number(value: str) -> Optional[int]:
    return MONTHS.get((value or "").strip().lower().rstrip("."))

_REL_UNITS = {
    "день": "days", "дня": "days", "дней": "days", "сутки": "days", "суток": "days",
    "неделя": "weeks", "недели": "weeks", "недель": "weeks",
    "месяц": "months", "месяца": "months", "месяцев": "months",
    "год": "years", "года": "years", "лет": "years",
}


def _safe_dt(y: int, m: int, d: int) -> Optional[datetime]:
    try:
        return datetime(y, m, d)
    except Exception:
        return None


def _normalize_dt(dt: Optional[datetime]) -> Optional[datetime]:
    if not dt:
        return None
    return datetime(dt.year, dt.month, dt.day)


def _has_explicit_year(value: str) -> bool:
    return bool(re.search(r"\b(?:19|20)\d{2}\b", value or ""))


def _replace_year(dt: datetime, year: int) -> Optional[datetime]:
    return _safe_dt(year, dt.month, dt.day)


def _align_range_years(
    left: datetime,
    right: datetime,
    left_raw: str,
    right_raw: str,
) -> Tuple[datetime, datetime]:
    """Propagate a year written on only one endpoint of a natural-language range."""
    left_has_year = _has_explicit_year(left_raw)
    right_has_year = _has_explicit_year(right_raw)

    if not left_has_year and right_has_year:
        aligned = _replace_year(left, right.year)
        if aligned:
            left = aligned
        if left > right:
            previous = _replace_year(left, right.year - 1)
            if previous:
                left = previous
    elif left_has_year and not right_has_year:
        aligned = _replace_year(right, left.year)
        if aligned:
            right = aligned
        if right < left:
            following = _replace_year(right, left.year + 1)
            if following:
                right = following
    elif not left_has_year and not right_has_year and right < left:
        following = _replace_year(right, left.year + 1)
        if following:
            right = following

    return left, right


def _parse_one_date(s: str, today: Optional[datetime] = None) -> Optional[datetime]:
    s = (s or "").strip().lower()
    today = today or datetime.now()

    m = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", s)
    if m:
        d, mo, y = map(int, m.groups())
        if y < 100:
            y += 2000
        return _safe_dt(y, mo, d)

    m = re.fullmatch(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        return _safe_dt(y, mo, d)

    m = re.fullmatch(
        r"(\d{1,2})\s+([a-zа-яё.]+)(?:\s+(\d{4}))?(?:\s*(?:г(?:ода?)?|year))?",
        s,
    )
    if m:
        d = int(m.group(1))
        mo = _month_number(m.group(2))
        y = int(m.group(3)) if m.group(3) else today.year
        if mo:
            return _safe_dt(y, mo, d)

    try:
        return dateutil_parse(s, dayfirst=True)
    except Exception:
        return None


def _parse_discrete_dates(q: str, today: datetime) -> Optional[Dict[str, Any]]:
    patterns = [
        r"(?:на|в|за|события\s+за)?\s*(?P<days>\d{1,2}(?:\s*(?:,|и|and)\s*\d{1,2}){1,15})\s+(?P<month>[a-zа-яё.]+)(?:\s+(?P<year>\d{4}))?(?:\s*(?:г(?:ода?)?|year))?",
        r"(?:на|в|за|события\s+за)?\s*(?P<days>\d{1,2}(?:\s*(?:,|и)\s*\d{1,2}){1,15})\.(?P<month_num>\d{1,2})\.(?P<year>\d{2,4})",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if not m:
            continue
        raw_days = re.split(r"\s*(?:,|и|and)\s*", (m.group('days') or '').strip())
        days = []
        for x in raw_days:
            try:
                days.append(int(x))
            except Exception:
                pass
        if len(days) < 2:
            continue
        mo = int(m.group('month_num')) if m.groupdict().get('month_num') else _month_number(m.group('month') or '')
        year_raw = m.groupdict().get('year')
        y = int(year_raw) if year_raw else today.year
        if y < 100:
            y += 2000
        if not mo:
            continue
        dates: List[datetime] = []
        for d in days:
            dt = _safe_dt(y, mo, d)
            if dt:
                dates.append(dt)
        if len(dates) >= 2:
            dates = sorted({_normalize_dt(dt) for dt in dates if dt}, key=lambda x: x.toordinal())
            return {'mode': 'discrete_dates', 'start': dates[0], 'end': dates[-1], 'dates': dates, 'granularity': 'day'}
    return None


def parse_date_constraints(query: str, reference_date: Optional[datetime] = None) -> Dict[str, Any]:
    empty = {'mode': 'none', 'start': None, 'end': None, 'dates': [], 'granularity': None}
    if not query:
        return empty

    q = query.lower().strip()
    now = reference_date or datetime.now()
    today = datetime(now.year, now.month, now.day)

    discrete = _parse_discrete_dates(q, today)
    if discrete:
        return discrete

    # относительные даты
    if re.search(r"\bпозавчера\b", q):
        dt = today - timedelta(days=2)
        return {'mode': 'single', 'start': dt, 'end': dt, 'dates': [dt], 'granularity': 'day'}
    if re.search(r"\bвчера\b", q):
        dt = today - timedelta(days=1)
        return {'mode': 'single', 'start': dt, 'end': dt, 'dates': [dt], 'granularity': 'day'}
    if re.search(r"\bсегодня\b", q):
        return {'mode': 'single', 'start': today, 'end': today, 'dates': [today], 'granularity': 'day'}

    if re.search(r"\b(?:эта|этот|текущая|текущий)\s+недел", q):
        weekday = today.weekday()
        start = today - timedelta(days=weekday)
        end = start + timedelta(days=6)
        return {'mode': 'range', 'start': start, 'end': end, 'dates': [], 'granularity': 'day'}
    if re.search(r"\b(?:этот|текущий)\s+месяц\b", q):
        start = datetime(today.year, today.month, 1)
        end = (start + relativedelta(months=1)) - relativedelta(days=1)
        return {'mode': 'range', 'start': start, 'end': _normalize_dt(end), 'dates': [], 'granularity': 'day'}

    m = re.search(r"(?:за\s+)?последн(?:ие|ий|яя)\s+(\d+)\s+([а-яё]+)", q)
    if m:
        n = int(m.group(1))
        unit_word = m.group(2)
        unit = _REL_UNITS.get(unit_word)
        if unit:
            end = today
            start = end - (timedelta(**{unit: n}) if unit in ('days', 'weeks') else relativedelta(**{unit: n}))
            return {'mode': 'range', 'start': _normalize_dt(start), 'end': end, 'dates': [], 'granularity': 'day'}

    if re.search(r"\bпрошл(?:ая|ый|ое)\s+недел", q):
        weekday = today.weekday()
        this_monday = today - timedelta(days=weekday)
        last_monday = this_monday - timedelta(days=7)
        last_sunday = this_monday - timedelta(days=1)
        return {'mode': 'range', 'start': last_monday, 'end': last_sunday, 'dates': [], 'granularity': 'day'}
    if re.search(r"\bпрошл(?:ый|ая|ое)\s+месяц\b", q):
        first_this_month = datetime(today.year, today.month, 1)
        first_last_month = first_this_month - relativedelta(months=1)
        last_last_month = first_this_month - relativedelta(days=1)
        return {'mode': 'range', 'start': first_last_month, 'end': _normalize_dt(last_last_month), 'dates': [], 'granularity': 'day'}
    if re.search(r"\bпрошл(?:ый|ая|ое)\s+год\b", q):
        start = datetime(today.year - 1, 1, 1)
        end = datetime(today.year - 1, 12, 31)
        return {'mode': 'range', 'start': start, 'end': end, 'dates': [], 'granularity': 'day'}

    range_patterns = [
        r"(?:с\s+)?(?P<d1>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\s*(?:по|–|-|—|до)\s*(?P<d2>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
        r"(?:между)\s+(?P<d1>\d{4}[./-]\d{1,2}[./-]\d{1,2})\s*(?:и|–|-|—|до)\s*(?P<d2>\d{4}[./-]\d{1,2}[./-]\d{1,2})",
        r"(?P<d1>\d{4}[./-]\d{1,2}[./-]\d{1,2})\s*(?:по|–|-|—|до)\s*(?P<d2>\d{4}[./-]\d{1,2}[./-]\d{1,2})",
        r"(?P<d1>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\s*(?:–|-|—)\s*(?P<d2>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
        r"(?:(?:с|from)\s+)?(?P<d1>\d{1,2}\s+[a-zа-яё.]+(?:\s+\d{4})?)\s*(?:по|–|-|—|до|to)\s*(?P<d2>\d{1,2}\s+[a-zа-яё.]+(?:\s+\d{4})?)",
    ]
    for pat in range_patterns:
        m = re.search(pat, q)
        if m:
            raw_d1 = m.group('d1')
            raw_d2 = m.group('d2')
            d1 = _normalize_dt(_parse_one_date(raw_d1, today=today))
            d2 = _normalize_dt(_parse_one_date(raw_d2, today=today))
            if d1 and d2:
                d1, d2 = _align_range_years(d1, d2, raw_d1, raw_d2)
                if d1 > d2:
                    d1, d2 = d2, d1
                return {'mode': 'range', 'start': d1, 'end': d2, 'dates': [], 'granularity': 'day'}

    m = re.search(r"(?:(?:с|from)\s+)?(\d{1,2})\s*(?:по|–|-|—|to)\s*(\d{1,2})\s+([a-zа-яё.]+)(?:\s+(\d{4}))?", q)
    if m:
        d1, d2 = int(m.group(1)), int(m.group(2))
        mo = _month_number(m.group(3))
        y = int(m.group(4)) if m.group(4) else today.year
        if mo:
            start = _safe_dt(y, mo, d1)
            end = _safe_dt(y, mo, d2)
            if start and end:
                if start > end:
                    start, end = end, start
                return {'mode': 'range', 'start': start, 'end': end, 'dates': [], 'granularity': 'day'}

    any_date = re.search(r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})|(\d{4}[./-]\d{1,2}[./-]\d{1,2})", q)
    if any_date:
        dt = _normalize_dt(_parse_one_date(any_date.group(0), today=today))
        if dt:
            return {'mode': 'single', 'start': dt, 'end': dt, 'dates': [dt], 'granularity': 'day'}

    m = re.search(r"(?:на|в|за|on|at)\s*(\d{1,2})\s+([a-zа-яё.]+)(?:\s+(\d{4}))?(?:\s*(?:г(?:ода?)?|year))?", q)
    if m:
        d = int(m.group(1))
        mo = _month_number(m.group(2))
        y = int(m.group(3)) if m.group(3) else today.year
        if mo:
            dt = _safe_dt(y, mo, d)
            if dt:
                return {'mode': 'single', 'start': dt, 'end': dt, 'dates': [dt], 'granularity': 'day'}

    m = re.search(r"(?<!\d)(\d{1,2})\s+([a-zа-яё.]+)(?:\s+(\d{4}))?(?:\s*(?:г(?:ода?)?|year))?", q)
    if m:
        d = int(m.group(1))
        mo = _month_number(m.group(2))
        y = int(m.group(3)) if m.group(3) else today.year
        if mo:
            dt = _safe_dt(y, mo, d)
            if dt:
                return {'mode': 'single', 'start': dt, 'end': dt, 'dates': [dt], 'granularity': 'day'}

    # Месяц целиком: «в августе 2025», «за июль», «in March 2025». Перебираем
    # все предлоги: первое «в» в вопросе часто относится не к дате («в какие дни»).
    for m in re.finditer(r"(?<![a-zа-яё])(?:в|во|за|in|during)\s+([a-zа-яё.]+)(?:\s+(\d{4}))?\b", q):
        mo = _MONTHS_PERIOD.get(m.group(1).strip(".").lower())
        if not mo:
            continue
        y = int(m.group(2)) if m.group(2) else today.year
        start = datetime(y, mo, 1)
        end = (start + relativedelta(months=1)) - relativedelta(days=1)
        return {'mode': 'range', 'start': start, 'end': _normalize_dt(end), 'dates': [], 'granularity': 'month'}

    m = re.search(r"(?<![a-zа-яё])(?:в|за|in|during)\s+((?:19|20)\d{2})\s*(?:год[ауе]?|г\.?)(?![a-zа-яё])", q)
    if m:
        y = int(m.group(1))
        return {'mode': 'range', 'start': datetime(y, 1, 1), 'end': datetime(y, 12, 31), 'dates': [], 'granularity': 'year'}

    try:
        dt = dateutil_parse(q, dayfirst=True)
        if dt:
            dt = _normalize_dt(dt)
            return {'mode': 'single', 'start': dt, 'end': dt, 'dates': [dt], 'granularity': 'day'}
    except Exception:
        pass

    return empty


def parse_date_range(query: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    parsed = parse_date_constraints(query)
    return parsed.get('start'), parsed.get('end')


def date_to_str(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d") if dt else "None"
