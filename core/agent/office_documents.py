"""Surgical OOXML edits: preserve untouched package parts byte-for-byte.

No conversion to Markdown and back. This edits cell contents and text runs,
not page layout, charts, SmartArt, embedded objects or VBA.
"""
from __future__ import annotations

import difflib
import io
import json
import posixpath
import re
import zipfile
from pathlib import Path
from lxml import etree as E

EXTENSIONS = {'.xlsx', '.xlsm', '.docx', '.pptx'}
S = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
P = 'http://schemas.openxmlformats.org/package/2006/relationships'
C = 'http://schemas.openxmlformats.org/package/2006/content-types'
XML = '{http://www.w3.org/XML/1998/namespace}'


def xml(data):
    return E.fromstring(data, E.XMLParser(resolve_entities=False, no_network=True))


def encoded(root):
    return E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)


def colnum(ref):
    n = 0
    for c in re.match(r'[A-Z]+', ref)[0]:
        n = n * 26 + ord(c) - 64
    return n


class OfficeFile:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.suffix.lower() not in EXTENSIONS:
            raise ValueError('Нужен файл XLSX/XLSM, DOCX или PPTX')
        with zipfile.ZipFile(self.path) as z:
            if sum(i.file_size for i in z.infolist()) > 250_000_000:
                raise ValueError('Файл слишком большой для полного редактирования в этом режиме')
            self.entries = z.infolist()
            self.parts = {i.filename: z.read(i) for i in self.entries}
        if any(n.startswith('_xmlsignatures/') for n in self.parts):
            raise ValueError('Файл содержит цифровую подпись: для правок нужна неподписанная рабочая копия')
        self.changed = {}
        self.targets = {}
        self.sheets = []
        if self.path.suffix.lower() in {'.xlsx', '.xlsm'}:
            book = xml(self.parts['xl/workbook.xml'])
            rels = xml(self.parts['xl/_rels/workbook.xml.rels'])
            targets = {r.get('Id'): r.get('Target') for r in rels}
            self.shared = []
            if 'xl/sharedStrings.xml' in self.parts:
                self.shared = [''.join(si.itertext()) for si in xml(self.parts['xl/sharedStrings.xml'])]
            for sheet in book.findall(f'{{{S}}}sheets/{{{S}}}sheet'):
                target = targets[sheet.get(f'{{{R}}}id')]
                part = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
                if part.startswith('xl/worksheets/'):
                    self.sheets.append((sheet.get('name'), part))

    def tree(self, part):
        return xml(self.changed.get(part, self.parts[part]))

    def cell_value(self, cell):
        if cell is None:
            return ''
        formula = cell.find(f'{{{S}}}f')
        if formula is not None:
            return '=' + (formula.text or '')
        if cell.get('t') == 'inlineStr':
            return ''.join(cell.find(f'{{{S}}}is').itertext())
        value = cell.findtext(f'{{{S}}}v') or ''
        if cell.get('t') == 's':
            return self.shared[int(value)]
        return value

    def inventory(self, message='', limit=22000):
        """Addressed native contents, with explicit selection when a file is large."""
        groups = []
        self.targets = {}
        for si, (name, part) in enumerate(self.sheets):
            root = self.tree(part)
            rows = root.findall(f'{{{S}}}sheetData/{{{S}}}row')
            width = min(80, max([colnum(c.get('r')) for row in rows for c in row] or [1]))
            from openpyxl.utils import get_column_letter
            for ri, row in enumerate(rows):
                cells = {c.get('r'): c for c in row.findall(f'{{{S}}}c')}
                values = {}
                for ci in range(1, width + 1):
                    ref = f'{get_column_letter(ci)}{row.get("r")}'
                    values[f'xlsx:{si}:{ref}'] = self.cell_value(cells.get(ref))
                groups.append((f'Лист {si}: {name}; строка {row.get("r")}', values, (part, 'excel'), ri < 8))
        if not self.sheets:
            kind = 'word' if self.path.suffix.lower() == '.docx' else 'ppt'
            ns = W if kind == 'word' else A
            names = ([n for n in self.parts if re.fullmatch(r'word/(?:document|header\d+|footer\d+)\.xml', n)]
                     if kind == 'word' else [n for n in self.parts if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)])
            for part in sorted(names):
                for index, paragraph in enumerate(self.tree(part).iter(f'{{{ns}}}p')):
                    key = f'{part}:p{index}'
                    text = ''.join(t.text or '' for t in paragraph.iter(f'{{{ns}}}t'))
                    groups.append((part, {key: text}, (part, kind), index < 5))
        full_size = sum(len(json.dumps(v, ensure_ascii=False)) + len(label) for label, v, _, _ in groups)
        if full_size > limit and re.search(r"переведи|перевести|сократи|весь\s+(?:документ|текст)|целиком", message, re.I):
            raise ValueError('Полный текст слишком большой для одной проверяемой правки. Укажите раздел или диапазон; необработанные части нельзя выдавать за готовые.')
        tokens = re.findall(r'[А-Яа-яЁёA-Za-z]{4,}', message.casefold())
        explicit_rows = {int(n) for n in re.findall(r'(?:строк\w*\s*|[A-Z]{1,3})(\d+)', message, re.I)}
        selected = []
        for label, values, meta, header in groups:
            text = ' '.join(values.values()).casefold()
            relevant = any(t[:max(4, len(t)-2)] in text for t in tokens if len(t) >= 4)
            relevant |= bool(explicit_rows and any(k.startswith('xlsx:') and int(re.search(r'\d+$',k)[0]) in explicit_rows for k in values))
            if full_size <= limit or header or relevant:
                selected.append((label, values, meta))
        lines, used = [], 0
        for label, values, meta in selected:
            line = label + '\n' + json.dumps(values, ensure_ascii=False)
            if used + len(line) > limit:
                raise ValueError('В файле слишком много подходящих строк. Укажите лист, ФИО, номер строки или нужный фрагмент точнее.')
            lines.append(line)
            used += len(line)
            for key, value in values.items():
                self.targets[key] = (meta[0], meta[1], value)
        if not lines:
            raise ValueError('В файле не найдены доступные ячейки или текстовые фрагменты')
        if full_size > limit:
            lines.insert(0, '[Большой документ: показаны заголовки и строки по запросу; остальные части сохраняются без изменений]')
        return '\n'.join(lines)

    def get_value(self, key):
        if key.startswith('xlsx:'):
            _, si, ref = key.split(':')
            part = self.sheets[int(si)][1]
            return self.cell_value(self.tree(part).find(f'.//{{{S}}}c[@r="{ref}"]'))
        part, index = key.rsplit(':p', 1)
        ns = W if part.startswith('word/') else A
        paragraph = list(self.tree(part).iter(f'{{{ns}}}p'))[int(index)]
        return ''.join(t.text or '' for t in paragraph.iter(f'{{{ns}}}t'))

    def set_cell(self, part, ref, value, kind='text', cached=None):
        if not re.fullmatch(r'[A-Z]{1,3}[1-9]\d{0,6}', ref) or colnum(ref) > 16384 or int(re.search(r'\d+', ref)[0]) > 1048576:
            raise ValueError('Некорректный адрес ячейки')
        root = self.tree(part)
        from openpyxl.utils.cell import range_boundaries, coordinate_to_tuple
        r, c = coordinate_to_tuple(ref)
        for merge in root.findall(f'{{{S}}}mergeCells/{{{S}}}mergeCell'):
            c1, r1, c2, r2 = range_boundaries(merge.get('ref'))
            if r1 <= r <= r2 and c1 <= c <= c2 and (r, c) != (r1, c1):
                raise ValueError(f'{ref}: нельзя записать в середину объединённой области {merge.get("ref")}')
        data = root.find(f'{{{S}}}sheetData')
        if data is None:
            raise ValueError('На листе нет области данных')
        row = data.find(f'{{{S}}}row[@r="{r}"]')
        if row is None:
            row = E.Element(f'{{{S}}}row', r=str(r))
            at = next((i for i, child in enumerate(data) if int(child.get('r')) > r), len(data))
            data.insert(at, row)
        cell = row.find(f'{{{S}}}c[@r="{ref}"]')
        if cell is None:
            cell = E.Element(f'{{{S}}}c', r=ref)
            at = next((i for i, child in enumerate(row) if child.tag == f'{{{S}}}c' and colnum(child.get('r')) > c), len(row))
            row.insert(at, cell)
        for child in list(cell):
            if child.tag in {f'{{{S}}}f', f'{{{S}}}v', f'{{{S}}}is'}:
                cell.remove(child)
        cell.attrib.pop('t', None)
        if kind == 'formula':
            if not str(value).startswith('='):
                raise ValueError('Формула должна начинаться с =')
            E.SubElement(cell, f'{{{S}}}f').text = str(value)[1:]
            if cached is not None:
                if isinstance(cached, (float, int)):
                    E.SubElement(cell, f'{{{S}}}v').text = str(cached)
                else:
                    cell.set('t', 'str')
                    E.SubElement(cell, f'{{{S}}}v').text = str(cached)
        elif kind == 'number':
            from decimal import Decimal, InvalidOperation
            try:
                number = Decimal(str(value).replace(',', '.'))
                if not number.is_finite():
                    raise InvalidOperation()
            except InvalidOperation as exc:
                raise ValueError('Некорректное число') from exc
            E.SubElement(cell, f'{{{S}}}v').text = str(number)
        elif kind == 'text':
            cell.set('t', 'inlineStr')
            text = E.SubElement(E.SubElement(cell, f'{{{S}}}is'), f'{{{S}}}t')
            text.set(XML + 'space', 'preserve')
            text.text = str(value)
        elif kind != 'clear':
            raise ValueError('Неподдерживаемый тип значения')
        self.changed[part] = encoded(root)

    def set_paragraph(self, part, index, value, kind):
        ns = W if kind == 'word' else A
        root = self.tree(part)
        paragraph = list(root.iter(f'{{{ns}}}p'))[index]
        nodes = list(paragraph.iter(f'{{{ns}}}t'))
        old = ''.join(n.text or '' for n in nodes)
        if not nodes:
            run = E.Element(f'{{{ns}}}r')
            node = E.SubElement(run, f'{{{ns}}}t')
            node.text = value
            end = paragraph.find(f'{{{ns}}}endParaRPr')
            paragraph.insert(list(paragraph).index(end) if end is not None else len(paragraph), run)
        else:
            # Apply only differing spans, from the end. Untouched runs keep fonts,
            # hyperlinks, bold/italic, bookmarks, fields and embedded objects.
            for tag, a, b, x, y in reversed(difflib.SequenceMatcher(a=old, b=value, autojunk=False).get_opcodes()):
                if tag == 'equal':
                    continue
                offset, start_node, start_at = 0, None, 0
                for node in nodes:
                    length = len(node.text or '')
                    if start_node is None and offset <= a <= offset + length:
                        start_node, start_at = node, a - offset
                    lo, hi = max(a - offset, 0), min(b - offset, length)
                    if lo < hi:
                        node.text = (node.text or '')[:lo] + (node.text or '')[hi:]
                    offset += length
                if start_node is None:
                    start_node, start_at = nodes[-1], len(nodes[-1].text or '')
                text = start_node.text or ''
                start_node.text = text[:start_at] + value[x:y] + text[start_at:]
            for node in nodes:
                node.set(XML + 'space', 'preserve')
        self.changed[part] = encoded(root)

    def patch(self, changes, sources=''):
        if not isinstance(changes, list) or not changes or len(changes) > 200:
            raise ValueError('Нет конкретных правок или их слишком много для одного шага')
        seen = set()
        for change in changes:
            key = change.get('id')
            if key not in self.targets or key in seen:
                raise ValueError('Модель указала неизвестный или повторный адрес фрагмента')
            seen.add(key)
            part, kind, before = self.targets[key]
            if change.get('before') != before:
                raise ValueError(f'{key}: исходное значение не совпало; правка не применяется вслепую')
            after = change.get('after')
            if not isinstance(after, str) or len(after) > 20000:
                raise ValueError('Некорректный текст правки')
            evidence = str(change.get('evidence') or '')
            if evidence and ' '.join(evidence.split()).casefold() not in ' '.join(sources.split()).casefold():
                raise ValueError('Основание правки не найдено в исходных данных')
            if kind == 'excel':
                ref = key.rsplit(':', 1)[1]
                self.set_cell(part, ref, after, change.get('kind', 'text'))
            else:
                if change.get('kind') not in {'text', 'clear'}:
                    raise ValueError('В текстовом фрагменте разрешена только замена текста')
                self.set_paragraph(part, int(key.rsplit(':p', 1)[1]), after, kind)
        return len(seen)

    def recalculate(self):
        if self.sheets:
            root = self.tree('xl/workbook.xml')
            props = root.find(f'{{{S}}}calcPr')
            if props is None:
                props = E.SubElement(root, f'{{{S}}}calcPr')
            props.set('fullCalcOnLoad', '1')
            props.set('forceFullCalc', '1')
            props.set('calcMode', 'auto')
            self.changed['xl/workbook.xml'] = encoded(root)

    def save(self, path):
        self.recalculate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.pending')
        try:
            with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
                for info in self.entries:
                    z.writestr(info, self.changed.get(info.filename, self.parts[info.filename]))
                for name, content in self.changed.items():
                    if name not in self.parts:
                        z.writestr(name, content)
            with zipfile.ZipFile(temp) as z:
                if z.testzip():
                    raise ValueError('Проверка целостности документа не прошла')
                for name, content in self.parts.items():
                    if name not in self.changed and z.read(name) != content:
                        raise ValueError('Изменилась нетронутая часть оригинального файла')
                for name in self.changed:
                    if name.endswith('.xml'):
                        xml(z.read(name))
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        return path

    def add_sheet(self, base_name, sheet_xml):
        # Reuse only a sheet marked as generated by us; never replace a user's sheet.
        marker = b'ChatKodi generated audit sheet'
        for name, part in self.sheets:
            if marker in self.parts.get(part, b''):
                self.changed[part] = sheet_xml.replace(b'<sheetData>', b'<!--ChatKodi generated audit sheet--><sheetData>', 1)
                return name
        book, rels = self.tree('xl/workbook.xml'), self.tree('xl/_rels/workbook.xml.rels')
        content = self.tree('[Content_Types].xml')
        existing = {name for name, _ in self.sheets}
        name, i = base_name[:31], 1
        while name in existing:
            i += 1
            name = base_name[:26] + f' ({i})'
        sheets = book.find(f'{{{S}}}sheets')
        sid = max(int(s.get('sheetId')) for s in sheets) + 1
        n = sid
        while f'xl/worksheets/sheet{n}.xml' in self.parts:
            n += 1
        part = f'xl/worksheets/sheet{n}.xml'
        used_rids = {r.get('Id') for r in rels}
        rid = 'rIdChatKodi'
        while rid in used_rids:
            rid += 'X'
        E.SubElement(sheets, f'{{{S}}}sheet', name=name, sheetId=str(sid), attrib={f'{{{R}}}id': rid})
        E.SubElement(rels, f'{{{P}}}Relationship', Id=rid, Type=R + '/worksheet', Target=f'worksheets/sheet{n}.xml')
        E.SubElement(content, f'{{{C}}}Override', PartName='/' + part,
                     ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml')
        self.changed.update({'xl/workbook.xml': encoded(book), 'xl/_rels/workbook.xml.rels': encoded(rels),
                             '[Content_Types].xml': encoded(content), part: sheet_xml.replace(b'<sheetData>', b'<!--ChatKodi generated audit sheet--><sheetData>', 1)})
        return name
