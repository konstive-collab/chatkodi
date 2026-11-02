"""Locate original Excel rows and apply computed cells without resaving its layout."""
from __future__ import annotations

import re
from pathlib import Path
from core.agent.office_documents import OfficeFile


def locate_table(path, table):
    from openpyxl import load_workbook
    from core.agent.attachments import _cell, _fitness_header
    book = load_workbook(path, read_only=True, data_only=True)
    matches = []
    try:
        for sheet in book:
            blocks, block = [], []
            for number, cells in enumerate(sheet.iter_rows(), 1):
                row = [_cell(c.value) for c in cells]
                if any(row):
                    block.append((number, row))
                elif block:
                    blocks.append(block)
                    block = []
            if block:
                blocks.append(block)
            for block in blocks:
                width = max(max((i + 1 for i, v in enumerate(row) if v), default=0) for _, row in block)
                values = [row[:width] for _, row in block]
                flat = _fitness_header(values)
                if flat[0] != table.header or flat[1:] != table.rows:
                    continue
                leaf = next((i for i, row in enumerate(values[:12])
                             if sum(bool(re.fullmatch(r"Результат", c, re.I)) for c in row) >= 2
                             and any(re.search(r"№\s*упражнен", c, re.I) for c in row)), 0)
                start = leaf + 1
                if leaf and start < len(values) and all(not v or re.fullmatch(r"\d+", v) for v in values[start]):
                    start += 1
                data_rows = [r for r, _ in block[start:]]
                if len(data_rows) == len(table.rows):
                    matches.append((sheet.title, block[leaf][0], data_rows))
        if len(matches) != 1:
            raise ValueError('Не удалось однозначно сопоставить таблицу с исходным листом. Укажите нужный лист.')
        return matches[0]
    finally:
        book.close()


def preserve_calculation(template, generated, table, target, *, fitness_mode=False, cached=None, editable_columns=(), footer_rules=None):
    """Retain every original part except calculated worksheet cells and audit data."""
    from openpyxl import load_workbook
    from openpyxl.formula.translate import Translator
    from openpyxl.utils import get_column_letter
    from core.agent import fitness

    sheet_name, header_row, physical_rows = locate_table(template, table)
    original = OfficeFile(template)
    part = next(p for name, p in original.sheets if name == sheet_name)
    calculated = load_workbook(generated, data_only=False)
    data = calculated['Ведомость']
    audit = OfficeFile(generated)
    norms_part = next((p for name, p in audit.sheets if name in {'Нормативы', 'Шкала'}), None)
    norms_name = ''
    if norms_part:
        from core.agent.office_documents import S, encoded, xml
        audit_xml = xml(audit.parts[norms_part])
        for cell in audit_xml.iter(f'{{{S}}}c'):
            cell.attrib.pop('s', None)  # generated style IDs do not belong to the template
        norms_name = original.add_sheet('Нормативы ChatKodi', encoded(audit_xml))
    score_columns = {g.points for g in fitness.groups(table)} if fitness_mode else set()
    if fitness_mode:
        score_columns |= {i for i, h in enumerate(table.header) if re.search(r'сумма\s*балл|Проверка расч', h, re.I)}
    else:
        score_columns |= {i for i, h in enumerate(table.header) if re.search(r'оценк|ср\.?\s*балл|итог|сумма', h, re.I)}
    score_columns |= {c for c in editable_columns if c is not None}
    if cached:
        from openpyxl.utils.cell import column_index_from_string
        score_columns |= {column_index_from_string(re.match(r"[A-Z]+", ref)[0]) - 1 for ref in cached}
    # New result/check columns are appended, not inserted through the template.
    score_columns |= set(range(len(table.header), data.max_column))
    for ci in sorted(score_columns):
        col = get_column_letter(ci + 1)
        if ci >= len(table.header):
            original.set_cell(part, f'{col}{header_row}', str(data.cell(1, ci + 1).value or ''))
        for logical, real_row in enumerate(physical_rows, 2):
            source = data.cell(logical, ci + 1)
            value = source.value
            ref = f'{col}{real_row}'
            if value is None or (fitness_mode and cached is not None and cached.get(source.coordinate) == ""):
                continue
            if isinstance(value, str) and value.startswith('='):
                value = Translator(value, origin=source.coordinate).translate_formula(ref)
                for name in ('Нормативы', 'Шкала'):
                    if norms_name:
                        value = value.replace(f"'{name}'!", f"'{norms_name}'!")
                original.set_cell(part, ref, value, 'formula', cached=(cached or {}).get(source.coordinate))
            else:
                original.set_cell(part, ref, str(value), 'number' if isinstance(value, (int, float)) else 'text')
    # Keep numeric-looking text and Excel times in the original input cells.
    # Adjust lookup references, not the input's value, number format or style.
    if fitness_mode:
        from core.agent.office_documents import S, encoded
        root = original.tree(part)
        base = load_workbook(template, read_only=True, data_only=False)
        try:
            sheet = base[sheet_name]
            for logical, real_row in enumerate(physical_rows, 2):
                for group in fitness.groups(table):
                    point = f'{get_column_letter(group.points + 1)}{real_row}'
                    result = f'{get_column_letter(group.result + 1)}{real_row}'
                    cell = root.find(f'.//{{{S}}}c[@r="{point}"]/{{{S}}}f')
                    if cell is None:
                        continue
                    existing = sheet[result].value
                    raw = table.rows[logical - 2][group.result] if group.result < len(table.rows[logical - 2]) else ''
                    if hasattr(existing, 'hour'):
                        replacement = f'ROUND({result}*86400,3)'
                    elif ':' in raw:
                        replacement = f'(VALUE(LEFT({result},FIND(":",{result})-1))*60+VALUE(MID({result},FIND(":",{result})+1,99)))'
                    else:
                        unit = re.sub(r'^[-+]?\d+(?:[.,]\d+)?', '', raw).strip()
                        numeric = f'SUBSTITUTE({result},"{unit}","")' if unit else result
                        replacement = f'IFERROR(VALUE({numeric}),{result})'
                    # Replace only the VLOOKUP argument, not blank/missing checks.
                    cell.text = cell.text.replace(f'VLOOKUP({result},', f'VLOOKUP({replacement},')
            original.changed[part] = encoded(root)
        finally:
            base.close()
    calculated.close()
    if fitness_mode and cached:
        refresh_footer(original, part, table, physical_rows, cached, footer_rules)
    original.save(target)
    return Path(target)


def fitness_tables(path):
    """Read each physical sheet separately; summary sheets without inputs stay intact."""
    from openpyxl import load_workbook
    from core.agent.attachments import _cell, _fitness_header
    from core.agent.table_check import Table
    from core.agent.fitness import groups, active_results
    from core.agent.grading import _person_header
    book = load_workbook(path, read_only=True, data_only=True)
    result = []
    try:
        for sheet in book:
            block = []
            for cells in sheet.iter_rows(values_only=True):
                row = [_cell(v) for v in cells]
                if any(row):
                    block.append(row)
                elif block:
                    result += _fitness_block(sheet.title, block)
                    block = []
            if block:
                result += _fitness_block(sheet.title, block)
    finally:
        book.close()
    return result


def _fitness_block(sheet, rows):
    from core.agent.attachments import _fitness_header
    from core.agent.table_check import Table
    from core.agent.fitness import groups, active_results
    from core.agent.grading import _person_header
    width = max(max((i + 1 for i, v in enumerate(r) if v), default=0) for r in rows)
    flat = _fitness_header([r[:width] for r in rows])
    if len(flat) < 2:
        return []
    table = Table(flat[0], flat[1:], [])
    slots = groups(table)
    if slots and (_person_header(table.header) or len(slots) > 1) and active_results(table):
        return [(sheet, table)]
    return []


def calculate_book(template, sources, target, *, doubts=(), cancel_event=None, progress=None):
    import shutil
    import tempfile
    from core.agent.fitness import calculate, groups, _id
    tables = fitness_tables(template)
    if not tables:
        raise ValueError('В книге нет листов с исходными результатами упражнений. Проверьте листы групп и сохранённые значения формул.')
    required = {_id(row[g.exercise]) for _, table in tables for row in table.rows for g in groups(table) if g.exercise < len(row) and row[g.exercise].strip()}
    bodies, used, issues = [], [], []
    with tempfile.TemporaryDirectory(prefix='chatkodi-book-') as temp:
        base = Path(template)
        for i, (sheet, table) in enumerate(tables):
            if progress:
                progress(f'Заполняю лист «{sheet}»: {i + 1} из {len(tables)}…')
            step = Path(temp) / f'step-{i}.xlsx'
            body, names, incomplete = calculate(table, sources, step, doubts=doubts, template_path=base, cancel_event=cancel_event, required_exercises=required)
            bodies.append(f'**Лист «{sheet}»**\n\n{body}')
            used += names
            if incomplete:
                issues.append(f'{sheet}: {incomplete}')
            base = step
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        pending = Path(target).with_suffix(Path(target).suffix + '.pending')
        shutil.copyfile(base, pending)
        pending.replace(target)
    return '\n\n'.join(bodies), list(dict.fromkeys(used)), '; '.join(issues)


def corrected_doubts(source, name, doubts, changes):
    """Clear OCR doubts only for fields explicitly corrected in the edit plan."""
    if not doubts:
        return []
    from core.agent.attachments import _tables_as_markdown
    from core.agent import grading
    from openpyxl.utils import get_column_letter
    table, _, _ = grading.pick_table([(name, _tables_as_markdown(source))])
    sheet_name, _, rows = locate_table(source, table)
    native = OfficeFile(source)
    sheet_index = next(i for i, (title, _) in enumerate(native.sheets) if title == sheet_name)
    changed_ids = {c['id'] for c in changes}
    unresolved = []
    for doubt in doubts:
        index = next((i for i, row in enumerate(table.rows)
                      if ' '.join(c for c in row[:2] if c) == doubt['row_label']), None)
        key = f"xlsx:{sheet_index}:{get_column_letter(doubt['column'] + 1)}{rows[index]}" if index is not None else ''
        if key not in changed_ids:
            unresolved.append(doubt)
    return unresolved


def recalculate_edit(path, sources, *, doubts=(), cancel_event=None, progress=None):
    """Stage fitness recalculation before publishing the user's edited revision."""
    if not fitness_tables(path):
        return None
    if not sources:
        raise ValueError('Для пересчёта не сохранены нормативы; приложите документ с нормативами повторно')
    path = Path(path)
    pending = path.with_name('.recalculated-' + path.name)
    try:
        result = calculate_book(path, sources, pending, doubts=doubts, cancel_event=cancel_event, progress=progress)
        pending.replace(path)
        return result
    finally:
        pending.unlink(missing_ok=True)


def refresh_footer(native, part, table, physical_rows, cached, rules=None):
    """Repair recognized summary fields from calculated rows, retaining cell styles.

    Missing results are never counted as medical exemptions. Unknown final ranks
    are not replaced with stale example numbers from the source template.
    """
    from core.agent.office_documents import S, colnum
    from openpyxl.utils import get_column_letter
    from collections import Counter
    grade_col = next((i for i,h in enumerate(table.header) if re.search(r'оценка.*физическ', h, re.I)), None)
    level_col = next((i for i,h in enumerate(table.header) if re.search(r'квалификационн.*уровень', h, re.I)), None)
    total_col = next((i for i,h in enumerate(table.header) if re.search(r'сумма\s*балл',h,re.I)), None)
    if grade_col is None or not any(f'{get_column_letter(grade_col+1)}{i+2}' in cached for i in range(len(physical_rows))):
        return
    grades = [cached.get(f'{get_column_letter(grade_col+1)}{i+2}', 'Н/Д') for i in range(len(physical_rows))]
    levels = [cached.get(f'{get_column_letter(level_col+1)}{i+2}', 'Н/Д') for i in range(len(physical_rows))] if level_col is not None else []
    grade_counts, level_counts = Counter(grades), Counter(levels)
    pending = any(g == 'Н/Д' for g in grades)
    listed = len(physical_rows)
    checked = sum(isinstance(cached.get(f'{get_column_letter(total_col+1)}{i+2}'), (float,int)) for i in range(listed)) if total_col is not None else 0
    positive = sum(grade_counts[g] for g in [3,4,5])
    grange = f'{get_column_letter(grade_col+1)}{physical_rows[0]}:{get_column_letter(grade_col+1)}{physical_rows[-1]}'
    qrange = f'{get_column_letter(level_col+1)}{physical_rows[0]}:{get_column_letter(level_col+1)}{physical_rows[-1]}' if level_col is not None else ''
    source_root = native.tree(part)
    labels = []
    for row in source_root.findall(f'{{{S}}}sheetData/{{{S}}}row'):
        if int(row.get('r')) <= max(physical_rows):
            continue
        cells = row.findall(f'{{{S}}}c')
        for index, cell in enumerate(cells):
            value = native.cell_value(cell)
            if not value or value.startswith('='):
                continue
            text = ' '.join(value.lower().strip('"«» ').split())
            fields = []
            for candidate in cells[index+1:]:
                v = native.cell_value(candidate)
                if not v:
                    continue
                if v.startswith('=') or re.fullmatch(r'\d+(?:\.\d+)?',v):
                    fields.append((candidate.get('r'),v))
                elif v.strip().lower() not in {'чел.','чел','%'}:
                    break
            labels.append((text, cell.get('r'), fields, cells[index+1:]))
    listed_ref = next((f[0][0] for text,_,f,_ in labels if text == 'по списку' and f), None)
    checked_ref = next((f[0][0] for text,_,f,_ in labels if text == 'всего проверено' and f), None)
    denominator = checked_ref or str(checked)
    for text, ref, fields, right in labels:
        grade = {'отлично':5,'хорошо':4,'удовлетворительно':3,'неудовлетворительно':2}.get(text)
        level = next((code for word,code in [('высший','ВК'),('первый','I'),('второй','II'),('третий','III')]
                      if text.startswith(word) and 'квалификац' in text), None)
        if text.startswith('оценка подразделени'):
            target = next((c.get('r') for c in right if native.cell_value(c).strip()), None)
            if target:
                value = 'Н/Д'
                if not pending and checked and rules:
                    p5, g5, p4, g4, p3 = rules
                    value = 5 if positive*100/listed >= p5 and grade_counts[5]*100/checked >= g5 else (
                        4 if positive*100/listed >= p4 and (grade_counts[5]+grade_counts[4])*100/checked >= g4 else
                        3 if positive*100/listed >= p3 else 2)
                native.set_cell(part,target,str(value),'number' if isinstance(value,int) else 'text')
            continue
        if not fields:
            continue
        count_ref = fields[0][0]
        if grade:
            count = grade_counts[grade]
            native.set_cell(part,count_ref,f'=COUNTIF({grange},"{grade}")','formula',cached=count)
            for pct,_ in fields[1:]:
                native.set_cell(part,pct,f'=IFERROR({count_ref}*100/{denominator},"Н/Д")','formula',cached=count*100/checked if checked else 'Н/Д')
        elif level and qrange:
            count = 'Н/Д' if pending else level_counts[level]
            native.set_cell(part,count_ref,f'=IF(COUNTIF({grange},"Н/Д")>0,"Н/Д",COUNTIF({qrange},"{level}"))','formula',cached=count)
            for pct,_ in fields[1:]:
                native.set_cell(part,pct,f'=IFERROR({count_ref}*100/{listed_ref or listed},"Н/Д")','formula',cached='Н/Д' if pending else count*100/listed)
        elif text == 'по списку':
            native.set_cell(part,count_ref,str(listed),'number')
        elif text == 'всего проверено':
            native.set_cell(part,count_ref,str(checked),'number')
            for pct,_ in fields[1:]:
                native.set_cell(part,pct,f'=IFERROR({count_ref}*100/{listed_ref or listed},"Н/Д")','formula',cached=checked*100/listed)
        elif text.startswith('получили положительные оценки'):
            native.set_cell(part,count_ref,str(positive),'number')
            for pct,_ in fields[1:]:
                native.set_cell(part,pct,f'=IFERROR({count_ref}*100/{denominator},"Н/Д")','formula',cached=positive*100/checked if checked else 'Н/Д')
        elif text.startswith('освобождено по состоянию здоровья') and fields[0][1].startswith('='):
            native.set_cell(part,count_ref,'Н/Д')
            for pct,_ in fields[1:]:
                native.set_cell(part,pct,'Н/Д')
