"""Verified assessment rows from a supplied norms table, never invented cutoffs.

Currently handles cadet course rows (male/female) with explicit 2–5-exercise
threshold columns. Other populations are reported as requiring a matched profile.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from core.agent.table_check import parse_markdown_tables, to_number


def compact(s):
    return re.sub(r'[^а-яёa-z0-9]+', '', s.casefold()).replace('ё', 'е')


@dataclass
class Assessment:
    course: int
    onwards: bool
    sex: str
    count: int
    minimum: float
    grades: list
    levels: list
    source: str
    quote: str
    known_minimum_failure: bool


def read_profiles(sources):
    profiles = []
    for name, text in sources:
        chapter, page, sex = False, '', ''
        count_columns, grade_columns, minimum_column, levels = {}, {}, None, {}
        fail = 'ненабранминимумбалловводномфизическомупражнении' in compact(text)
        for block in re.split(r'(?=--- Страница \d+ ---)', text):
            m = re.match(r'--- Страница (\d+) ---', block)
            if m:
                page = m.group(1)
            if re.search(r'Приложение\s*№\s*6\b', block):
                chapter = True
            if re.search(r'Приложение\s*№\s*7\b', block):
                chapter = False
            for table in parse_markdown_tables(block):
                if not chapter and not any('минимумбаллов' in compact(h) for h in table.header):
                    continue
                rows = [table.header] + table.rows
                for row in rows[:4]:
                    for ci, h in enumerate(row):
                        key = compact(h)
                        if 'минимумбаллов' in key:
                            minimum_column = ci
                        for word, count in [('двух',2),('трех',3),('четырех',4),('пяти',5)]:
                            if f'в{word}упражнениях' in key:
                                count_columns[count] = ci
                        for label, code in [('высший','ВК'),('первый','I'),('второй','II'),('третий','III')]:
                            if label in key and 'квалификационн' in key:
                                levels[ci] = code
                if minimum_column is None:
                    continue
                for row in table.rows:
                    if len(row) < minimum_column + 1:
                        continue
                    description = compact(row[1]) if len(row) > 1 else ''
                    if description.startswith('курсанты'):
                        sex = 'ж' if 'женского' in description else 'м'
                        continue
                    if 'военнослужащие' in description:
                        sex = ''
                    if not sex or not re.search(r'\b\d+.*курс', row[1], re.I):
                        continue
                    course = re.search(r'\d+', row[1])
                    minimum = to_number(row[minimum_column])
                    if minimum is None:
                        continue
                    available = []
                    for count, start in count_columns.items():
                        numbers = [to_number(row[c]) if c < len(row) else None for c in range(start, start+3)]
                        if all(n is not None for n in numbers):
                            available.append((count, numbers))
                    if len(available) != 1:
                        continue
                    count, numbers = available[0]
                    ranks = [(code, to_number(row[ci])) for ci, code in sorted(levels.items()) if ci < len(row)]
                    if any(value is None for _, value in ranks):
                        ranks = []
                    profiles.append(Assessment(int(course[0]), 'последующ' in row[1], sex, count, minimum,
                        list(zip(['5','4','3'], numbers)), ranks,
                        name + (f', стр. {page}' if page else ''), ' | '.join(row), fail))
    return profiles


def select(profiles, header, row):
    sex_col = next((i for i,h in enumerate(header) if compact(h) in {'пол','полвоеннослужащего'}), None)
    course_col = next((i for i,h in enumerate(header) if 'курсобучения' in compact(h) or compact(h) == 'курс'), None)
    if sex_col is None or course_col is None or max(sex_col, course_col) >= len(row):
        return None
    course = re.search(r'(\d+)\s*курс', row[course_col], re.I)
    if not course:
        return None
    sex = row[sex_col].strip().casefold()[:1]
    course = int(course[1])
    found = [p for p in profiles if p.sex == sex and (course >= p.course if p.onwards else course == p.course)]
    if len(found) != 1:
        return None
    return found[0], sex_col, course_col


def assess(profile, points, exercises=None, taxonomy=None):
    count = len(points)
    if points and min(points) < profile.minimum and profile.known_minimum_failure:
        return 2, 'нет', f'не выполнен минимум {profile.minimum:g} баллов в упражнении'
    if count != profile.count:
        return 'Н/Д', 'Н/Д', f'результатов {count} из {profile.count}; итоговая проверка не завершена'
    total = sum(points)
    if min(points) < profile.minimum:
        return 2, 'нет', f'не выполнен минимум {profile.minimum:g} баллов'
    if exercises is not None:
        valid, reason = composition(profile, exercises, taxonomy or {})
        if not valid:
            return 'Н/Д', 'Н/Д', reason
    grade = next((int(g) for g, minimum in profile.grades if total >= minimum), 2)
    level = next((code for code, minimum in profile.levels if total >= minimum), 'нет') if grade != 2 else 'нет'
    return grade, level, 'итог по найденному профилю нормативов'


def formulas(profile, refs, grade_ref, guard, composition_ok=True):
    joined = ','.join(refs)
    total, count, minimum = f'SUM({joined})', f'COUNT({joined})', f'MIN({joined})'
    grade = '2'
    for g, threshold in reversed(profile.grades):
        grade = f'IF({total}>={threshold:g},{g},{grade})'
    grade = f'IF({count}<>{profile.count},"Н/Д",{grade})' if composition_ok else '"Н/Д"'
    if profile.known_minimum_failure:
        grade = f'IF({minimum}<{profile.minimum:g},2,{grade})'
    grade = f'=IF(AND({guard}),IF({count}=0,"Н/Д",{grade}),"Н/Д")'
    level = '"нет"'
    for label, threshold in reversed(profile.levels):
        level = f'IF({total}>={threshold:g},"{label}",{level})'
    level = f'=IF(NOT(ISNUMBER({grade_ref})),"Н/Д",IF({grade_ref}=2,"нет",{level}))'
    return grade, level


def unit_rules(sources):
    for name, text in sources:
        key = compact(text)
        a = re.search(r'отличноеслиполученонеменее(\d+)положительныхоценокотсписочногосоставаподразделенияприэтом(\d+)иболе', key)
        b = re.search(r'хорошоеслиполученонеменее(\d+)положительныхоценокотсписочногосоставаподразделенияприэтом(\d+)иболе', key)
        c = re.search(r'удовлетворительноеслиполученонеменее(\d+)положительныхоценокотсписочногосоставаподразделения', key)
        if a and b and c:
            return tuple(map(int, (a[1], a[2], b[1], b[2], c[1])))
    return None


def exercise_taxonomy(sources):
    result = {}
    pattern = re.compile(r'(?m)^Приложение\s*№\s*(\d+)|^Общие физические упражнения|^Специальные физические упражнения|качества\s*[«"](Сила|Быстрота|Ловкость|Выносливость)[»"]|^Упражнение\s*№\s*(\d+)', re.I)
    for _, text in sources:
        kind, quality = '', ''
        for match in pattern.finditer(text):
            value = match[0].lower()
            if match[1]:
                kind, quality = '', ''
            elif value.startswith('общие'):
                kind, quality = 'ОФУ', ''
            elif value.startswith('специальные'):
                kind, quality = 'СФУ', ''
            elif match[2]:
                quality = match[2].lower()
            elif match[3] and kind:
                result[match[3]] = (kind, quality)
    return result


def composition(profile, exercises, taxonomy):
    if len(exercises) != profile.count or len(set(exercises)) != profile.count:
        return False, 'не подтверждён полный набор разных упражнений КПК'
    classifications = [taxonomy.get(ex) for ex in exercises]
    if any(c is None for c in classifications):
        return False, 'не подтверждён состав КПК по видам упражнений'
    general = [quality for kind, quality in classifications if kind == 'ОФУ']
    special = sum(kind == 'СФУ' for kind, _ in classifications)
    allowed = ([{'ловкость','сила','выносливость'}, {'быстрота','сила','ловкость'}]
               if profile.count == 5 else [{'быстрота','сила','выносливость'}, {'ловкость','сила','выносливость'}])
    if len(general) != 3 or special != profile.count - 3 or set(general) not in allowed:
        return False, 'состав КПК не соответствует требуемым физическим качествам и ОФУ/СФУ'
    return True, ''
