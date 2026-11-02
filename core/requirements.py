"""Program check of a value against a written requirement: «не менее 32,7 МПа»,
«до 9,0», «4,0–5,0», «на сумму не более 30 000 руб.». The model only copies the
value and the requirement; whether it holds is decided here."""

from __future__ import annotations

import re
from typing import Optional

from core.reasoned_answer import numbers_in

_NUMBER_REQ = r"(-?\d+(?:[.,]\d+)?)"
_THOUSANDS = re.compile(r"(?<=\d)[\s  ](?=\d{3}(?!\d))")  # «30 000» is one number
_TIME_OR_DATE = re.compile(r"\d{1,2}:\d{2}|\d{1,2}\.\d{1,2}\.\d{2,4}")
_UNIT_AFTER = re.compile(r"\d(?:[\d\s.,]*\d)?\s*(°\s*[cс]?|%|[a-zа-яё]{1,6})", re.IGNORECASE)
_NOT_UNITS = {"до", "от", "и", "по", "с", "в", "на", "или"}
# a requirement says how a number must compare: words or signs before it
_REQUIREMENT = re.compile(r"(?:не\s+)?(?:менее|более|ниже|выше|меньше|больше|свыше|превыша\w*)|минимум|максимум|[≥≤<>]|"
                          r"\bот\s*-?\d|\bдо\s*-?\d|\d\s*(?:–|—|-|…)\s*\d", re.IGNORECASE)


def is_requirement(text: str) -> bool:
    return bool(_REQUIREMENT.search(text or "")) and bool(numbers_in(text or "", dates=False))


def _unit(text: str) -> str:
    """Unit after the last number: «от 17 до 23 °C» -> «°», «34 000,00 руб.» -> «руб»."""
    units = [m.group(1).lower().replace(" ", "") for m in _UNIT_AFTER.finditer(text or "")]
    units = [u for u in units if u not in _NOT_UNITS]
    if not units:
        return ""
    unit = units[-1]
    return "°" if unit.startswith("°") else unit[:3]


def requirement_holds(value_text: str, requirement: str) -> Optional[bool]:
    """Program check of «не менее 32,7», «до 9,0», «4,0–5,0», «≥ 20» against a value
    (a range of values «33,0–36,1» must hold at both ends). None when it cannot
    be decided safely: a time or a date, or different units on the two sides."""
    if _TIME_OR_DATE.search(value_text or "") or _TIME_OR_DATE.search(requirement or ""):
        return None
    unit_v, unit_r = _unit(value_text), _unit(requirement)
    if unit_v and unit_r and unit_v != unit_r:
        return None
    values = numbers_in(value_text or "", dates=False)
    if not values:
        return None
    verdicts = [_holds(v, requirement) for v in values[:2]]
    if any(v is None for v in verdicts):
        return None
    return all(verdicts)


def _holds(value: float, requirement: str) -> Optional[bool]:
    req = _THOUSANDS.sub("", (requirement or "").lower().replace("ё", "е")).replace(",", ".")
    num = lambda s: float(s.replace(",", "."))
    rng = re.search(rf"(?:от\s*)?{_NUMBER_REQ}\s*(?:–|—|-|…|\.\.\.?|до)\s*{_NUMBER_REQ}", req)
    if rng and (rng.group(0).startswith("от") or not re.search(r"не\s+(?:менее|более)|[≥≤<>]", req)):
        low, high = num(rng.group(1)), num(rng.group(2))
        if low <= high:
            return low <= value <= high
    lo = re.search(rf"(?:не\s+менее|не\s+ниже|не\s+меньше|минимум|≥|>=|от)\s*{_NUMBER_REQ}", req)
    hi = re.search(rf"(?:не\s+более|не\s+выше|не\s+больше|не\s+превыша\w*|максимум|≤|<=|до)\s*{_NUMBER_REQ}", req)
    if lo or hi:
        ok = True
        if lo:
            ok = ok and value >= num(lo.group(1))
        if hi:
            ok = ok and value <= num(hi.group(1))
        return ok
    strict_lo = re.search(rf"(?:более|больше|выше|свыше|превыша\w*|>)\s*{_NUMBER_REQ}", req)
    strict_hi = re.search(rf"(?:менее|меньше|ниже|<)\s*{_NUMBER_REQ}", req)
    if strict_lo or strict_hi:
        ok = True
        if strict_lo:
            ok = ok and value > num(strict_lo.group(1))
        if strict_hi:
            ok = ok and value < num(strict_hi.group(1))
        return ok
    return None
