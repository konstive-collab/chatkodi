"""Persistent active documents, immutable revisions, sources and an audit trail."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from core.atomic_io import atomic_write_json
from core.agent.office_documents import EXTENSIONS, OfficeFile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class DocumentWorkspace:
    def __init__(self, root):
        self.root = Path(root)
        self.state = {'active': '', 'documents': {}}
        self.error = ''
        self._recovered = False
        try:
            if (self.root / 'current.json').exists():
                self.state = json.loads((self.root / 'current.json').read_text(encoding='utf-8'))
                if not isinstance(self.state.get('documents'), dict):
                    raise ValueError()
        except (ValueError, OSError):
            backup = self.root / 'current.json.bak'
            try:
                state = json.loads(backup.read_text(encoding='utf-8'))
                if not isinstance(state.get('documents'), dict):
                    raise ValueError()
                self.state, self._recovered = state, True
                self.error = 'Рабочие документы восстановлены из резервной копии состояния; проверьте номер версии.'
            except (ValueError, OSError):
                self.error = 'Не удалось восстановить рабочие документы чата; приложите последнюю версию повторно.'

    def _write(self, state):
        atomic_write_json(self.root / 'current.json', state, backup=not self._recovered)
        self._recovered = False

    def save(self):
        self._write(self.state)

    @property
    def active(self):
        return self.state['documents'].get(self.state.get('active'))

    def path(self, record=None):
        record = record or self.active
        if not record:
            raise ValueError('Нет активного рабочего документа')
        path = (self.root / record['path']).resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file():
            raise ValueError('Последняя версия документа недоступна; приложите её повторно')
        if digest(path) != record['digest']:
            raise ValueError('Рабочий файл изменён вне чата. Приложите изменённый файл, чтобы не потерять внешние правки.')
        return path

    def register(self, name, source, key=None, action='edit_document'):
        source = Path(source)
        if source.suffix.lower() not in EXTENSIONS:
            raise ValueError('Редактирование с сохранением оформления доступно для XLSX/XLSM, DOCX и PPTX')
        for record in self.state['documents'].values():
            if (key and record.get('attachment') == key) or (record['name'] == name and record['digest'] == digest(source)):
                self.state['active'] = record['id']
                self.save()
                return record
        OfficeFile(source)  # reject malformed packages before registering
        same_name = [r for r in self.state['documents'].values() if r['name'].casefold() == Path(name).name.casefold()]
        if len(same_name) == 1:
            self.state['active'] = same_name[0]['id']
            target, revision = self.next_path()
            target.parent.mkdir(parents=True)
            shutil.copy2(source, target)
            self.commit(target, revision, 'Загружена обновлённая копия документа', action=same_name[0]['action'])
            self.active['attachment'] = key
            self.save()
            return self.active
        doc_id = uuid.uuid4().hex[:12]
        clean = Path(name).name
        target = self.root / doc_id / 'v0000' / clean
        target.parent.mkdir(parents=True)
        shutil.copy2(source, target)
        record = dict(id=doc_id, name=clean, attachment=key, path=str(target.relative_to(self.root)), digest=digest(target),
                      revision=0, action=action, history=[], norms=[])
        record['history'].append(dict(path=record['path'], digest=record['digest'], revision=0, summary='Исходный документ'))
        self.state['documents'][doc_id] = record
        self.state['active'] = doc_id
        self.save()
        return record

    def next_path(self, record=None):
        record = record or self.active
        if not record:
            raise ValueError('Нет активного документа')
        revision = record['revision'] + 1
        path = self.root / record['id'] / f'v{revision:04d}' / record['name']
        while path.parent.exists():
            revision += 1
            path = self.root / record['id'] / f'v{revision:04d}' / record['name']
        return path, revision

    def commit(self, path, revision, summary, *, action=None, expected=None, changes=None):
        record = self.active
        if expected and digest(self.path(record)) != expected:
            raise ValueError('Активный документ изменился во время обработки; результат не стал активным')
        OfficeFile(path)  # result must still be a readable native package
        item = dict(path=str(Path(path).relative_to(self.root)), revision=revision, digest=digest(path),
                    summary=summary, time=time.time(), norms=record.get('norms', []), changes=changes or [])
        updated = dict(record, path=item['path'], revision=revision, digest=item['digest'],
                       history=(record['history'] + [item])[-100:], action=action or record['action'])
        state = dict(self.state, documents=dict(self.state['documents']))
        state['documents'][record['id']] = updated
        with (self.root / 'journal.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(document=record['id'], **item), ensure_ascii=False) + '\n')
        self._write(state)
        self.state = state
        return updated

    def remember_norms(self, sources):
        refs = []
        for name, text in sources:
            key = hashlib.sha256(text.encode('utf-8')).hexdigest()
            folder = self.root / 'sources'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'{key}.txt'
            if not path.exists():
                path.write_text(text, encoding='utf-8')
            refs.append(dict(name=name, path=str(path.relative_to(self.root))))
        self.active['norms'] = refs
        self.active['history'][-1]['norms'] = refs
        self.save()

    def norms(self):
        result = []
        for item in (self.active or {}).get('norms', []):
            path = (self.root / item['path']).resolve()
            if not path.is_relative_to(self.root.resolve()) or not path.is_file():
                raise ValueError('Сохранённый норматив недоступен; приложите его повторно')
            result.append((item['name'], path.read_text(encoding='utf-8')))
        return result

    def rollback(self, revision=None):
        record = self.active
        history = record['history']
        candidates = [h for h in history if h['revision'] == revision] if revision is not None else history[-2:-1]
        if len(candidates) != 1:
            raise ValueError('Нужная предыдущая версия не найдена')
        original = self.root / candidates[0]['path']
        if digest(original) != candidates[0]['digest']:
            raise ValueError('Предыдущая версия изменена вне чата; восстановление остановлено')
        target, number = self.next_path()
        target.parent.mkdir(parents=True)
        shutil.copy2(original, target)
        self.commit(target, number, f'Восстановлена версия {candidates[0]["revision"]}')
        self.active['norms'] = candidates[0].get('norms', [])
        self.save()
        return target
