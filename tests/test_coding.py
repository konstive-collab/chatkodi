"""Coding is independent of library evidence; edits use the latest per-chat files."""
import json
from pathlib import Path

import pytest

from core.agent.agent import Agent
from core.agent.attachments import AttachmentStore, read_attachment
from core.agent import coding


class Model:
    def is_available(self):
        return True


class Manager:
    llm_model = Model()
    hybrid_search = None

    def process_message(self, *args, **kwargs):
        pytest.fail('Coding request entered the document library')

    def _document_count(self):
        return 0


def agent(tmp_path, monkeypatch):
    monkeypatch.setenv('CHATKODI_RUNTIME_DIR', str(tmp_path / 'runtime'))
    result = Agent(Manager(), AttachmentStore(tmp_path / 'chat'))
    result._library_docs = lambda *a, **kw: pytest.fail('Code generation searched the library')
    return result


@pytest.mark.parametrize('message', [
    'Сделай калькулятор', 'Можешь написать калькулятор на Python?',
    'Напиши код для калькулятора', 'Создай сайт с калькулятором',
    'Создай программу учета доходов', 'Write a calculator in Python',
])
def test_creation_bypasses_document_router(tmp_path, monkeypatch, message):
    a = agent(tmp_path, monkeypatch)
    a._json = lambda *args, **kwargs: pytest.fail('Coding intent should not require the document router')
    route = a.route(message, [])
    assert route['action'] == 'code' and route['source'] == 'none'


def test_create_edit_and_restart_use_current_files_not_previous_chat_text(tmp_path, monkeypatch):
    a = agent(tmp_path, monkeypatch)
    source = 'def add(a, b):\n    return a + b\n'
    updated = source + '\ndef subtract(a, b):\n    return a - b\n'
    third = updated + '\ndef multiply(a, b):\n    return a * b\n'
    payloads = iter([
        dict(summary='Создан калькулятор', files=[dict(name='calculator.py', content=source), dict(name='settings.json', content='{"theme": "dark"}')]),
        dict(summary='Добавлено вычитание', files=[dict(name='calculator.py', content=updated)]),
    ])
    prompts = []
    def generate(prompt, schema, max_tokens=0):
        assert schema is coding.SCHEMA
        prompts.append(prompt)
        return next(payloads)
    a._json = generate
    one = a.handle('Напиши калькулятор на Python')
    assert one.action == 'code' and not one.incomplete and len(one.files) == 2
    original_path = next(f.path for f in one.files if f.fmt == 'py')
    two = a.handle('Добавь вычитание')
    assert two.action == 'code' and not two.incomplete
    assert json.dumps({'calculator.py': source, 'settings.json': '{"theme": "dark"}'}, ensure_ascii=False) in prompts[1]
    assert original_path.read_text() == source  # previous revision remains available
    assert a._code_project.revision == 2
    # A fresh Agent for the same chat recovers the current manifest from disk.
    restored = agent(tmp_path, monkeypatch)
    assert restored._code_project.snapshot()['calculator.py'] == updated
    def again(prompt, schema, max_tokens=0):
        assert 'subtract' in prompt and 'Добавь вычитание' in prompt
        return dict(summary='Добавлено умножение', files=[dict(name='calculator.py', content=third)])
    restored._json = again
    three = restored.handle('Теперь добавь умножение')
    assert not three.incomplete and restored._code_project.revision == 3
    assert restored._code_project.snapshot() == {'calculator.py': third, 'settings.json': '{"theme": "dark"}'}


def test_generated_code_is_saved_but_never_executed(tmp_path, monkeypatch):
    a = agent(tmp_path, monkeypatch)
    marker = tmp_path / 'must_not_exist'
    source = f'from pathlib import Path\nPath({str(marker)!r}).touch()\n'
    a._json = lambda *args, **kw: dict(summary='Пример', files=[dict(name='example.py', content=source)])
    result = a.handle('Напиши код на Python')
    assert not result.incomplete and result.files[0].path.is_file()
    assert not marker.exists()


def test_invalid_revision_cannot_replace_last_working_revision(tmp_path, monkeypatch):
    a = agent(tmp_path, monkeypatch)
    a._code_project.save({'calculator.py': 'answer = 40 + 55\n'}, 'Начало')
    a._json = lambda *args, **kw: dict(summary='Изменено', files=[dict(name='calculator.py', content='def broken(:\n')])
    result = a.handle('Исправь код калькулятора')
    assert result.incomplete and 'Синтаксическая ошибка Python' in result.text and not result.files
    assert a._code_project.revision == 1 and a._code_project.snapshot()['calculator.py'] == 'answer = 40 + 55\n'


@pytest.mark.parametrize('name', ['../outside.py', 'C:\\bad.py', '/tmp/bad.py', 'bad.exe', 'CON.py'])
def test_model_cannot_write_outside_project(tmp_path, monkeypatch, name):
    a = agent(tmp_path, monkeypatch)
    a._json = lambda *args, **kw: dict(summary='Код', files=[dict(name=name, content='print(1)')])
    result = a.handle('Напиши код на Python')
    assert result.incomplete and not result.files and not a._code_project.active


def test_html_attachment_retains_script_and_can_be_edited(tmp_path, monkeypatch):
    a = agent(tmp_path, monkeypatch)
    source = '<!doctype html><html><body><script>const add = (a,b) => a+b;</script></body></html>'
    path = tmp_path / 'calculator.html'
    path.write_text(source)
    def generate(prompt, schema, max_tokens=0):
        assert schema is coding.SCHEMA
        assert 'const add = (a,b) => a+b;' in prompt
        return dict(summary='Добавлена кнопка', files=[dict(name='calculator.html', content=source.replace('<body>', '<body><button>Сброс</button>'))])
    a._json = generate
    result = a.handle('Добавь кнопку сброса', [path])
    assert result.action == 'code' and not result.incomplete
    assert '<button>Сброс</button>' in a._code_project.snapshot()['calculator.html']
    assert path.read_text() == source  # original user attachment is untouched


def test_no_model_explains_missing_model_not_missing_library(tmp_path, monkeypatch):
    a = agent(tmp_path, monkeypatch)
    a.manager = Manager()
    a.manager.llm_model = None
    result = a.handle('Создай калькулятор')
    assert result.action == 'code' and result.incomplete == 'локальная модель не загружена'
    assert 'Библиотека документов для этой задачи не требуется' in result.text


def test_chats_are_isolated_and_other_tasks_still_use_the_router(tmp_path, monkeypatch):
    a = agent(tmp_path, monkeypatch)
    a._code_project.save({'calculator.py': 'x = 1\n'}, 'Первый проект')
    other = Agent(Manager(), AttachmentStore(tmp_path / 'another-chat'))
    assert not other._code_project.active
    for message in ['Составь учебную программу', 'Напиши письмо', 'Создай программу тренировок', 'Сделай ведомость', 'Открой калькулятор', 'Напиши письмо по Трудовому кодексу', 'Создай приложение к рапорту']:
        calls = []
        a._json = lambda *args, **kw: (calls.append(1) or dict(action='chat', source='none'))
        assert a.route(message, [])['action'] != 'code'
        assert calls


def test_external_editor_changes_are_used_on_next_revision(tmp_path, monkeypatch):
    a = agent(tmp_path, monkeypatch)
    paths = a._code_project.save({'calculator.py': 'answer = 1\n'}, 'Проект')
    paths['calculator.py'].write_text('answer = 2\n')
    def generate(prompt, schema, max_tokens=0):
        assert 'answer = 2' in prompt
        return dict(summary='Изменено', files=[dict(name='calculator.py', content='answer = 3\n')])
    a._json = generate
    assert not a.handle('Добавь комментарий в код').incomplete
    assert a._code_project.snapshot()['calculator.py'] == 'answer = 3\n'
