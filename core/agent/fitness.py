"""Exact exercise-number/result/points sheets. No LLM arithmetic or interpolation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from core.agent.table_check import Table, parse_markdown_tables, to_number
from core.agent.grading import fmt, value_of


@dataclass
class ExerciseGroup:
    exercise: int
    result: int
    points: int


def groups(table: Table) -> list[ExerciseGroup]:
    out = []
    for c, header in enumerate(table.header):
        if re.search(r"№\s*упражнен", header, re.I):
            if c + 2 < len(table.header) and re.search("результат", table.header[c + 1], re.I) \
                    and re.search("балл", table.header[c + 2], re.I):
                out.append(ExerciseGroup(c, c + 1, c + 2))
    return out


def _id(text: str) -> str:
    text = re.sub(r"^(?:упражнение\s*)?№?\s*", "", text.strip(), flags=re.I)
    number = to_number(text)
    return str(int(number)) if number is not None and number.is_integer() else text


def standards(sources: Sequence[tuple[str, str]]) -> tuple[dict, list[str], list[str]]:
    """Read the printed pairs, even when a lookup is split into side-by-side columns.

    Exercise identity must appear explicitly in the preceding title. Conflicting
    mappings are rejected as a whole, including potentially different categories.
    """
    class Mappings(dict):
        pass

    maps = Mappings()
    maps.origins = {}
    origins, conflicts, per_document = {}, set(), {}
    for name, text in sources:
        exercise = ""
        location = name
        lines = text.splitlines()
        index = 0
        while index < len(lines):
            line = lines[index]
            page = re.match(r"--- Страница (\d+) ---", line)
            if page:
                location = f"{name}, стр. {page.group(1)}"
            title = re.search(r"упражнени[еяю]\s*№\s*(\d+)", line, re.I)
            if title:
                exercise = title.group(1)
            if exercise and line.lstrip().startswith("|"):
                end = index
                while end < len(lines) and lines[end].lstrip().startswith("|"):
                    end += 1
                for table in parse_markdown_tables("\n".join(lines[index:end])):
                    pairs = [c for c in range(len(table.header) - 1)
                             if re.search(r"количество|результат", table.header[c], re.I)
                             and re.search(r"балл", table.header[c + 1], re.I)]
                    for row in table.rows:
                        for c in pairs:
                            if c + 1 >= len(row):
                                continue
                            value, points = value_of(row[c]), to_number(row[c + 1])
                            if value is None or points is None:
                                continue
                            mapping = maps.setdefault(exercise, {})
                            if value in mapping and mapping[value] != points:
                                conflicts.add(exercise)
                            mapping[value] = points
                            per_document.setdefault(exercise, {}).setdefault(name, {})[value] = points
                            origins.setdefault(exercise, set()).add(location)
                index = end
            else:
                index += 1
    for exercise, documents in per_document.items():
        variants = list(documents.values())
        if any(mapping != variants[0] for mapping in variants[1:]):
            conflicts.add(exercise)
    for exercise in conflicts:
        maps.pop(exercise, None)
    maps.origins = {ex: sorted(origins[ex]) for ex in maps}
    used = sorted({name for ex in maps for name in origins[ex]})
    return maps, used, sorted(conflicts)


def _cell(row, column):
    return row[column].strip() if column < len(row) else ""


def calculate(table: Table, sources: Sequence[tuple[str, str]], path: Path, doubts=()):
    """Generate an auditable report and a workbook with editable live lookups.

    Only exact listed results receive points. No rules for total-to-rating or
    category selection are invented; those fields remain as supplied.
    """
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import PatternFill
    from openpyxl.utils import get_column_letter

    slots = groups(table)
    maps, used, conflicts = standards(sources)
    wb = Workbook()
    ws = wb.active
    ws.title = "Ведомость"
    lookup = wb.create_sheet("Нормативы")
    lookup.append(["№ упражнения", "Результат", "Баллы", "Источник"])
    ranges = {}
    for exercise, mapping in sorted(maps.items()):
        first = lookup.max_row + 1
        for value, points in sorted(mapping.items()):
            lookup.append([exercise, value, points, ", ".join(maps.origins.get(exercise, used))])
        ranges[exercise] = (first, lookup.max_row)
    header = list(table.header)
    total = next((c for c, h in enumerate(header) if re.search(r"сумма\s*балл", h, re.I)), None)
    if total is None:
        total = len(header)
        header.append("Сумма баллов")
    check = len(header)
    header.append("Проверка расчёта")
    ws.append(header)
    report = ["**Расчёт баллов по таблицам нормативов**", "", "| Строка | Упражнения: результат → баллы | Сумма | Проверка |", "|---|---|---|---|"]
    label_col = next((c for c, h in enumerate(header) if re.search(r"фио|фамили|нагрудный", h, re.I)), 0)
    for index, row in enumerate(table.rows, 2):
        if not any(row):
            continue
        for c, text in enumerate(row, 1):
            target = ws.cell(index, c)
            # Preserve text as text, including strings beginning with '='.
            number = to_number(text)
            if number is not None and re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?", text.strip()):
                target.value = number
            else:
                target.value = text or None
                if text:
                    target.data_type = "s"
        errors, details, values, refs, active = [], [], [], [], 0
        for slot in slots:
            raw_ex = _cell(row, slot.exercise)
            raw_result = _cell(row, slot.result)
            ex = _id(raw_ex)
            ex_ref = f"{get_column_letter(slot.exercise + 1)}{index}"
            result_ref = f"{get_column_letter(slot.result + 1)}{index}"
            points_cell = ws.cell(index, slot.points + 1)
            refs.append(points_cell.coordinate)
            # Select by the live exercise number so edits never reuse a stale scale.
            formula = '"нет норматива"'
            for key, (first, last) in reversed(list(ranges.items())):
                source_range = f"'Нормативы'!$B${first}:$C${last}"
                formula = f'IF({ex_ref}&""="{key}",IFERROR(VLOOKUP({result_ref},{source_range},2,FALSE),"нет результата в нормативе"),{formula})'
            points_cell.value = f'=IF(AND({ex_ref}="",{result_ref}=""),"",IF(OR({ex_ref}="",{result_ref}=""),"нет данных",{formula}))'
            if not raw_ex and not raw_result:
                continue
            active += 1
            value = value_of(raw_result)
            # Store colon times in the same seconds used by the lookup table.
            if value is not None and (":" in raw_result or not isinstance(ws.cell(index, slot.result + 1).value, (int, float))):
                ws.cell(index, slot.result + 1, value)
                note = f"Исходное время: {raw_result}; значение в секундах" if ":" in raw_result else f"Исходный результат: {raw_result}"
                ws.cell(index, slot.result + 1).comment = Comment(note, "ChatKodi")
            mapping = maps.get(ex)
            points = mapping.get(value) if mapping is not None else None
            row_label = " ".join(c for c in row[:2] if c)
            uncertain = [d for d in doubts if d.row_label == row_label
                         and d.column in (slot.exercise, slot.result, label_col)]
            if uncertain:
                points_cell.value = '="проверьте распознавание"'
                points_cell.comment = Comment("; ".join(d.describe() for d in uncertain), "ChatKodi")
                reason = "неуверенное распознавание: " + "; ".join(d.describe() for d in uncertain)
            elif not raw_ex or value is None:
                reason = "нет номера упражнения или результата"
            elif mapping is None:
                reason = f"нет однозначного норматива для упражнения № {ex}"
            elif points is None:
                reason = f"результат {raw_result} отсутствует в нормативе № {ex}"
            else:
                reason = ""
            if reason:
                errors.append(reason)
                points_cell.fill = PatternFill("solid", fgColor="FCE4D6")
            else:
                values.append(points)
                details.append(f"№ {ex}: {raw_result} → {fmt(points)}")
                old = _cell(row, slot.points)
                if old and to_number(old) != points:
                    points_cell.comment = Comment(f"Было: {old}. По нормативу: {fmt(points)}", "ChatKodi")
        if not active:
            errors.append("нет результатов упражнений")
        ref_list = ",".join(refs)
        # Require numeric lookups for every active group before summing.
        active_formula = "+".join(f'IF(OR({get_column_letter(s.exercise + 1)}{index}<>"",{get_column_letter(s.result + 1)}{index}<>""),1,0)' for s in slots)
        ws.cell(index, total + 1, f'=IF(AND(({active_formula})>0,COUNT({ref_list})=({active_formula})),SUM({ref_list}),"не рассчитано")')
        ws.cell(index, check + 1, "; ".join(errors) or "Баллы проверены по нормативам")
        label = _cell(row, label_col) or str(index - 1)
        report.append(f"| {label} | {'; '.join(details) or '—'} | {fmt(sum(values)) if not errors else 'не рассчитано'} | {'; '.join(errors) or '✓'} |")
    report += ["", "Источник нормативов: " + (", ".join(used) or "подходящие таблицы не найдены") + ".",
               "Оценка физической подготовленности и квалификационный уровень не пересчитывались: нужны правила перевода суммы баллов с учётом категории, пола и возраста.",
               "Результаты, отсутствующие в таблицах, не интерполируются и не округляются."]
    if conflicts:
        report.append("Противоречащие нормативы: упражнения № " + ", ".join(conflicts) + "; баллы по ним не вычислены.")
    ws.freeze_panes = "A2"
    wb.calculation.fullCalcOnLoad = True
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return "\n".join(report), used, ("не найдены однозначные таблицы нормативов" if not maps else "")
