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


def active_results(table):
    slots = groups(table)
    return sum(any(g.exercise < len(row) and g.result < len(row) and row[g.exercise].strip()
                   and row[g.result].strip() for g in slots) for row in table.rows)


def point_pairs(header):
    value = lambda h: bool(re.search(r"количество|кол[. -]*во|повтор|результат|repetitions|count|\bраз\b", h, re.I))
    points = lambda h: bool(re.search(r"балл|очки|points|score", h, re.I))
    out, c = [], 0
    while c + 1 < len(header):
        if value(header[c]) and points(header[c + 1]):
            out.append((c, c + 1)); c += 2
        elif points(header[c]) and value(header[c + 1]):
            out.append((c + 1, c)); c += 2
        else:
            c += 1
    return out


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
            if re.match(r"Приложение\s*№\s*\d+", line.strip(), re.I):
                exercise = ""
            title = re.search(r"упражнени[еяюи]\s*(?:№|N[оo]?\.?|номер)?\s*(\d+)\b", line, re.I)
            if title:
                exercise = title.group(1)
            if exercise and line.lstrip().startswith("|"):
                end = index
                while end < len(lines) and lines[end].lstrip().startswith("|"):
                    end += 1
                for table in parse_markdown_tables("\n".join(lines[index:end])):
                    pairs = point_pairs(table.header)
                    for row in table.rows:
                        for value_col, points_col in pairs:
                            if max(value_col, points_col) >= len(row):
                                continue
                            value, points = value_of(row[value_col]), to_number(row[points_col])
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


def calculate(table: Table, sources: Sequence[tuple[str, str]], path: Path, doubts=(), template_path=None, cancel_event=None, required_exercises=None):
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
    requested = required_exercises if required_exercises is not None else {
        _id(_cell(row, g.exercise)) for row in table.rows for g in slots if _cell(row, g.exercise)}
    for exercise in list(maps):
        if exercise not in requested:
            maps.pop(exercise)
    used = sorted({name for exercise in maps for name in maps.origins.get(exercise, [])})
    from core.agent import fitness_assessment
    profiles = fitness_assessment.read_profiles(sources)
    taxonomy = fitness_assessment.exercise_taxonomy(sources)
    grade_col = next((i for i,h in enumerate(table.header) if re.search(r"оценка.*физическ", h, re.I)), None)
    level_col = next((i for i,h in enumerate(table.header) if re.search(r"квалификационн.*уровень", h, re.I)), None)
    assessment_counts = {"evaluated": 0, "pending": 0, "failed": 0}
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
    check = next((i for i, h in enumerate(header) if h == "Проверка расчёта"), None)
    if check is None:
        check = len(header)
        header.append("Проверка расчёта")
    ws.append(header)
    report = ["**Расчёт баллов по таблицам нормативов**", "", "| Строка | Упражнения: результат → баллы | Сумма | Проверка |", "|---|---|---|---|"]
    label_col = next((c for c, h in enumerate(header) if re.search(r"фио|фамили|нагрудный", h, re.I)), 0)
    native_values = {}
    for index, row in enumerate(table.rows, 2):
        if cancel_event is not None and cancel_event.is_set():
            from core.agent.cancellation import TaskCancelled
            raise TaskCancelled()
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
        performed = []
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
                native_values[points_cell.coordinate] = ""
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
            native_values[points_cell.coordinate] = (points if not reason else
                "проверьте распознавание" if uncertain else "нет данных" if not raw_ex or value is None else
                "нет норматива" if mapping is None else "нет результата в нормативе")
            if reason:
                errors.append(reason)
                points_cell.fill = PatternFill("solid", fgColor="FCE4D6")
            else:
                values.append(points)
                performed.append(ex)
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
        native_values[f"{get_column_letter(total + 1)}{index}"] = sum(values) if not errors else "не рассчитано"
        assessment_note = ""
        selected_profile = fitness_assessment.select(profiles, table.header, row) if grade_col is not None else None
        if selected_profile:
            profile, sex_col, course_col = selected_profile
            grade, level, assessment_note = fitness_assessment.assess(profile, values, performed, taxonomy)
            assessment_counts["evaluated"] += 1
            assessment_counts["pending"] += int(grade == "Н/Д")
            assessment_counts["failed"] += int(grade == 2)
            grade_ref = f"{get_column_letter(grade_col + 1)}{index}"
            sex_ref, course_ref = f"{get_column_letter(sex_col + 1)}{index}", f"{get_column_letter(course_col + 1)}{index}"
            sex_literal, course_literal = row[sex_col].replace('"', '""'), row[course_col].replace('"', '""')
            guard = f'{sex_ref}="{sex_literal}",{course_ref}="{course_literal}"'
            current_ids = []
            for slot in slots:
                raw_id = _cell(row, slot.exercise)
                ref = f'{get_column_letter(slot.exercise + 1)}{index}'
                escaped = raw_id.replace('"', '""')
                guard += f',{ref}&""="{escaped}"'
                if raw_id:
                    current_ids.append(_id(raw_id))
            composition_ok, _ = fitness_assessment.composition(profile, current_ids, taxonomy)
            grade_formula, level_formula = fitness_assessment.formulas(profile, refs, grade_ref, guard, composition_ok=composition_ok)
            ws.cell(index, grade_col + 1, grade_formula)
            native_values[grade_ref] = grade
            if level_col is not None:
                ws.cell(index, level_col + 1, level_formula)
                native_values[f"{get_column_letter(level_col + 1)}{index}"] = level
        ws.cell(index, check + 1, "; ".join(errors + ([assessment_note] if assessment_note else [])) or "Баллы проверены по нормативам")
        label = _cell(row, label_col) or str(index - 1)
        report.append(f"| {label} | {'; '.join(details) or '—'} | {fmt(sum(values)) if not errors else 'не рассчитано'} | {'; '.join(errors + ([assessment_note] if assessment_note else [])) or '✓'} |")
    report += ["", "Источник нормативов: " + (", ".join(used) or "подходящие таблицы не найдены") + ".",
               "Оценка физической подготовленности и квалификационный уровень не пересчитывались: нужны правила перевода суммы баллов с учётом категории, пола и возраста.",
               "Результаты, отсутствующие в таблицах, не интерполируются и не округляются."]
    if assessment_counts["evaluated"]:
        report = [line for line in report if not line.startswith("Оценка физической подготовленности и квалификационный уровень не пересчитывались")]
        report.append(f'Профиль нормативов сопоставлен для {assessment_counts["evaluated"]} строк; не выполнен минимум в {assessment_counts["failed"]}; итоговая проверка не завершена в {assessment_counts["pending"]}.')
        report.append('Н/Д означает недостаточность данных, а не неудовлетворительную оценку. Суммы при неполной проверке промежуточные.')
        report.append('Сводные показатели пересчитываются по текущим результатам. Отсутствие результатов не означает освобождение по здоровью; неподтверждённые медицинские сведения обозначаются Н/Д.')
        for profile in profiles:
            if any(fitness_assessment.select([profile], table.header, row) for row in table.rows):
                if profile.source not in used:
                    used.append(profile.source)
                report.append(f'Основание итоговой оценки: {profile.source}; минимум {profile.minimum:g}, упражнений {profile.count}; «{profile.quote}».')
                lookup.append([f'Итог: {profile.course} курс ({profile.sex})', profile.minimum, profile.count, profile.source + ' | ' + profile.quote])
    if conflicts:
        report.append("Противоречащие нормативы: упражнения № " + ", ".join(conflicts) + "; баллы по ним не вычислены.")
    ws.freeze_panes = "A2"
    wb.calculation.fullCalcOnLoad = True
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    if template_path is not None:
        from core.agent.workbook_template import preserve_calculation

        preserve_calculation(template_path, path, table, path, fitness_mode=True, cached=native_values, footer_rules=fitness_assessment.unit_rules(sources))
    incomplete = "не найдены однозначные таблицы нормативов" if not maps else ""
    if assessment_counts["pending"]:
        incomplete = "не все итоговые оценки определены: не хватает результатов упражнений"
    return "\n".join(report), used, incomplete
