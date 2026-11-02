"""An old venv must report the missing PDF reader, not blame the uploaded photo."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import document_dependencies as dependencies
from core.agent.attachments import AttachmentStore, read_attachment
from core.agent.agent import Agent
from core.agent.cancellation import TaskCancelled
from core.agent.norm_sources import read_library
from core.agent.pdf_tables import read_pdf


def missing_pdf(monkeypatch, name='pdfplumber'):
    original = dependencies.importlib.import_module

    def load(module):
        if module == 'pdfplumber':
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return original(module)

    monkeypatch.setattr(dependencies.importlib, 'import_module', load)


@pytest.mark.parametrize('missing', ['pdfplumber', 'pdfminer'])
def test_pdf_error_names_reader_and_missing_transitive_dependency(tmp_path, monkeypatch, missing):
    missing_pdf(monkeypatch, missing)
    with pytest.raises(dependencies.DocumentDependencyError) as error:
        read_pdf(tmp_path / 'norms.pdf')
    assert missing in str(error.value)
    assert 'Восстановить компоненты документов' in str(error.value)
    assert 'pdfplumber==0.11.7' in str(error.value)


def test_healthcheck_imports_readers_and_doctor_reports_missing_component(monkeypatch):
    missing_pdf(monkeypatch)
    readers = dependencies.check_document_readers()
    assert not readers['ok'] and set(readers['errors']) == {'pdfplumber'}
    from tools import doctor
    monkeypatch.setattr(doctor, 'run_healthcheck', lambda: {
        'python': {'ok': True}, 'project': {'ok': True},
        'models': {'embeddings': True, 'llm': True}, 'document_readers': readers,
    })
    report = doctor.build_doctor_report()
    assert report['doctor']['status'] == 'issues_found'
    assert any('pdfplumber' in b for b in report['doctor']['blockers'])


def test_attachment_failure_remains_visible_in_grading_and_is_retried(tmp_path, monkeypatch):
    monkeypatch.setenv('CHATKODI_RUNTIME_DIR', str(tmp_path / 'runtime'))
    from openpyxl import Workbook
    book = Workbook()
    book.active.append(['ФИО', '№ упражнения', 'Результат', 'Балл'])
    book.active.append(['Участник', 1, 20, None])
    source = tmp_path / 'sheet.xlsx'; book.save(source)
    pdf = tmp_path / 'norms.pdf'; pdf.write_bytes(b'%PDF-test')
    store = AttachmentStore(tmp_path / 'attachments')
    sheet = read_attachment(store.add(source), None)
    item = store.add(pdf)
    with monkeypatch.context() as patch:
        missing_pdf(patch)
        read_attachment(item, None)
        assert not item.ok and 'pdfplumber' in item.error
        agent = Agent(SimpleNamespace(llm_model=None, hybrid_search=None), store=store)
        result = agent._do_grade('Заполни баллы', dict(source='attachments', task='баллы', search_query='', output='xlsx'), [sheet, item], lambda _: None)
        assert not result.files and result.incomplete
        assert 'pdfplumber' in result.text and 'Восстановить компоненты документов' in result.text
        assert 'на каждой фотографии' not in result.text
    from core.agent import pdf_tables
    monkeypatch.setattr(pdf_tables, 'read_pdf', lambda *a, **kw: pdf_tables.PDFReading(
        text='Таблица перевода результата выполнения упражнения № 1 в баллы\n'
             '| Количество раз | Баллы |\n|---|---|\n|20|40|', pages=1))
    agent = Agent(SimpleNamespace(llm_model=None, hybrid_search=None), store=store)
    result = agent.handle('Заполни баллы')
    assert item.ok and not item.error and result.files and not result.incomplete


def test_library_pdf_failure_preserves_actionable_error_and_cancel_propagates(tmp_path, monkeypatch):
    pdf = tmp_path / 'norms.pdf'; pdf.write_bytes(b'%PDF-test')
    docs = [dict(path=str(pdf), filename=pdf.name)]
    with monkeypatch.context() as patch:
        missing_pdf(patch)
        sources, issues = read_library(docs, None, None, {})
        assert not sources and 'pdfplumber' in issues[0] and 'Восстановить' in issues[0]
    from core.agent import pdf_tables

    def cancelled(*args):
        raise TaskCancelled()

    monkeypatch.setattr(pdf_tables, 'read_pdf', cancelled)
    with pytest.raises(TaskCancelled):
        read_library(docs, None, None, {})


def test_repair_package_versions_match_the_application_requirements():
    requirements = (Path(__file__).parents[1] / 'requirements-app.txt').read_text()
    assert all(package in requirements for _, package in dependencies.READERS.values())
