"""Native office formatting, durable revisions and grade/edit/recalculate sessions."""
import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from docx import Document
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.comments import Comment
from pptx import Presentation
from pptx.util import Inches

from core.agent.agent import Agent
from core.agent.attachments import AttachmentStore, read_attachment
from core.agent import document_editing
from core.agent.document_workspace import DocumentWorkspace
from core.agent.office_documents import OfficeFile


NORMS = 'Таблица перевода результата выполнения упражнения № 1 в баллы\n| Количество раз | Баллы |\n|---|---|\n| 20 | 40 |\n| 25 | 50 |\n'


class Model:
    def is_available(self):
        return True


class Manager:
    llm_model = Model()
    hybrid_search = None

    def _document_count(self):
        return 0

    def process_message(self, *a, **kw):
        pytest.fail('Native document edits must not enter generic RAG')


def make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv('CHATKODI_RUNTIME_DIR', str(tmp_path / 'rt'))
    return Agent(Manager(), AttachmentStore(tmp_path / 'att'))


def package(path):
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


def excel(path):
    wb = Workbook()
    ws = wb.active
    ws.title = 'Оригинал'
    ws.merge_cells('A1:D1')
    ws['A1'] = 'Утвержденная ведомость'
    ws.append(['ФИО', '№ упражнения', 'Результат', 'Балл', 'Сумма баллов'])
    ws.append(['Иванов', 1, 20, None, None])
    ws['A3'].font = Font(name='Arial', bold=True, color='FF0000')
    ws['D3'].fill = PatternFill('solid', fgColor='FFFF00')
    ws['C3'].number_format = '0.00'
    ws['A3'].comment = Comment('Не удалять', 'Пользователь')
    ws.column_dimensions['A'].width = 32
    ws.row_dimensions[3].height = 27
    ws.freeze_panes = 'B3'
    ws.print_area = 'A1:E3'
    ws.page_setup.orientation = 'landscape'
    wb.create_sheet('Другой лист')['A1'] = 'Нетронутые данные'
    wb.save(path)
    return path


def plain_excel(path):
    wb = Workbook()
    ws = wb.active
    ws.title = 'Оригинал'
    ws.append(['ФИО', '№ упражнения', 'Результат', 'Балл', 'Сумма баллов'])
    ws.append(['Иванов', 1, 20, None, None])
    ws['D2'].fill = PatternFill('solid', fgColor='FFFF00')
    ws.column_dimensions['A'].width = 32
    ws.print_area = 'A1:E2'
    wb.create_sheet('Другой лист')['A1'] = 'Нетронутые данные'
    wb.save(path)
    return path


def test_word_text_edit_preserves_other_parts_runs_and_page_setup(tmp_path):
    source = tmp_path / 'source.docx'
    doc = Document()
    p = doc.add_paragraph()
    p.add_run('Ответственный: ').bold = True
    p.add_run('Иванов').italic = True
    p.add_run('.')
    doc.sections[0].header.paragraphs[0].text = 'Фирменный бланк'
    doc.sections[0].footer.paragraphs[0].text = 'Подпись'
    doc.add_table(rows=2, cols=2).cell(0, 0).text = 'Таблица'
    doc.save(source)
    native = OfficeFile(source)
    native.inventory()
    native.patch([dict(id='word/document.xml:p0', before='Ответственный: Иванов.', after='Ответственный: Петров.', kind='text', evidence='')])
    result = tmp_path / 'result.docx'
    native.save(result)
    before, after = package(source), package(result)
    for name, content in before.items():
        if name != 'word/document.xml':
            assert after[name] == content, name
    updated = Document(result)
    assert updated.paragraphs[0].text == 'Ответственный: Петров.'
    assert updated.paragraphs[0].runs[0].bold and updated.paragraphs[0].runs[1].italic
    assert updated.tables[0].cell(0, 0).text == 'Таблица'
    assert updated.sections[0].page_width == doc.sections[0].page_width


def test_powerpoint_text_edit_preserves_masters_theme_size_and_run_styles(tmp_path):
    source = tmp_path / 'source.pptx'
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(5), Inches(1))
    run = box.text_frame.paragraphs[0].add_run()
    run.text, run.font.bold = 'Первый заголовок', True
    deck.save(source)
    native = OfficeFile(source)
    native.inventory()
    native.patch([dict(id='ppt/slides/slide1.xml:p0', before='Первый заголовок', after='Новый заголовок', kind='text', evidence='')])
    target = tmp_path / 'result.pptx'
    native.save(target)
    before, after = package(source), package(target)
    for name in before:
        if name != 'ppt/slides/slide1.xml':
            assert before[name] == after[name], name
    result = Presentation(target)
    shape = result.slides[0].shapes[0]
    assert shape.text == 'Новый заголовок' and shape.text_frame.paragraphs[0].runs[0].font.bold
    assert (shape.left, shape.top, shape.width, shape.height) == (box.left, box.top, box.width, box.height)


def test_excel_native_patch_preserves_styles_print_settings_comments_other_sheet(tmp_path):
    source = excel(tmp_path / 'source.xlsx')
    native = OfficeFile(source)
    native.inventory()
    native.patch([dict(id='xlsx:0:C3', before='20', after='25', kind='number', evidence='25')], 'Исправь результат на 25')
    target = tmp_path / 'result.xlsx'
    native.save(target)
    before, after = package(source), package(target)
    for name in before:
        if name not in {'xl/worksheets/sheet1.xml', 'xl/workbook.xml'}:
            assert before[name] == after[name], name
    wb = load_workbook(target)
    ws = wb['Оригинал']
    assert ws['C3'].value == 25 and ws['C3'].number_format == '0.00'
    assert ws['A3'].font.bold and ws['D3'].fill.fgColor.rgb.endswith('FFFF00')
    assert ws['A3'].comment.text == 'Не удалять'
    assert str(ws.merged_cells) == 'A1:D1'
    assert ws.column_dimensions['A'].width == 32 and ws.row_dimensions[3].height == 27
    assert ws.freeze_panes == 'B3' and ws.page_setup.orientation == 'landscape'
    wb.close()


def test_sequential_word_edits_restore_after_restart_and_rollback(tmp_path, monkeypatch):
    source = tmp_path / 'letter.docx'
    doc = Document()
    doc.add_paragraph('Иванов')
    doc.save(source)
    original = source.read_bytes()
    agent = make_agent(tmp_path, monkeypatch)
    agent._json = lambda *a, **kw: dict(summary='Исправлено', changes=[dict(id='word/document.xml:p0', before='Иванов', after='Петров', kind='text', evidence='Петров')])
    one = agent.handle('Замени Иванов на Петров', [source])
    assert not one.incomplete and Document(one.files[0].path).paragraphs[0].text == 'Петров'
    restarted = make_agent(tmp_path, monkeypatch)
    assert restarted.store.items[0].read and restarted.store.items[0].text
    restarted._json = lambda *a, **kw: dict(summary='Исправлено', changes=[dict(id='word/document.xml:p0', before='Петров', after='Сидоров', kind='text', evidence='Сидоров')])
    two = restarted.handle('Замени Петров на Сидоров')
    assert not two.incomplete and Document(two.files[0].path).paragraphs[0].text == 'Сидоров'
    undo = restarted.handle('Верни предыдущую версию')
    assert not undo.incomplete and Document(undo.files[0].path).paragraphs[0].text == 'Петров'
    assert source.read_bytes() == original


def test_fitness_edit_recalculates_current_version_and_retains_template(tmp_path, monkeypatch):
    source = plain_excel(tmp_path / 'ведомость.xlsx')
    norms = tmp_path / 'нормативы.txt'
    norms.write_text(NORMS)
    agent = make_agent(tmp_path, monkeypatch)
    agent._json = lambda *a, **kw: pytest.fail('Exact grading should not call the model')
    first = agent.handle('Заполни баллы', [source, norms])
    assert not first.incomplete
    original_parts = package(source)
    filled = load_workbook(first.files[0].path, data_only=True)
    assert filled['Оригинал']['D2'].value == 40 and filled['Оригинал']['E2'].value == 40
    filled.close()
    restarted = make_agent(tmp_path, monkeypatch)
    def correct(prompt, schema, max_tokens=0):
        assert schema is document_editing.SCHEMA
        assert 'xlsx:0:C2' in prompt
        return dict(summary='Результат исправлен', changes=[dict(id='xlsx:0:C2', before='20', after='25', kind='number', evidence='25')])
    restarted._json = correct
    second = restarted.handle('Исправь: у Иванова результат упражнения 1 — 25')
    assert not second.incomplete, second.text
    filled = load_workbook(second.files[0].path, data_only=True)
    assert filled['Оригинал']['C2'].value == 25
    assert filled['Оригинал']['D2'].value == 50 and filled['Оригинал']['E2'].value == 50
    assert filled['Оригинал']['D2'].fill.fgColor.rgb.endswith('FFFF00')
    assert filled['Оригинал'].column_dimensions['A'].width == 32
    assert len(filled.sheetnames) == 3  # audit sheet reused instead of multiplying
    filled.close()
    assert original_parts['xl/styles.xml'] == package(second.files[0].path)['xl/styles.xml']
    assert original_parts['xl/worksheets/sheet2.xml'] == package(second.files[0].path)['xl/worksheets/sheet2.xml']


def test_invalid_model_changes_stop_after_three_attempts_without_publishing(tmp_path, monkeypatch):
    source = tmp_path / 'letter.docx'
    doc = Document()
    doc.add_paragraph('Иванов')
    doc.save(source)
    agent = make_agent(tmp_path, monkeypatch)
    calls = []
    def wrong(*a, **kw):
        calls.append(1)
        return dict(summary='Исправлено', changes=[dict(id='word/document.xml:p0', before='Другое значение', after='Петров', kind='text', evidence='')])
    agent._json = wrong
    result = agent.handle('Исправь документ', [source])
    assert result.incomplete and len(calls) == 3
    assert agent._documents.active['revision'] == 0
    assert Document(agent._documents.path()).paragraphs[0].text == 'Иванов'


def test_external_changes_are_not_silently_overwritten(tmp_path):
    path = plain_excel(tmp_path / 'source.xlsx')
    workspace = DocumentWorkspace(tmp_path / 'documents')
    workspace.register(path.name, path)
    current = workspace.path()
    with current.open('ab') as f:
        f.write(b'external edit')
    with pytest.raises(ValueError, match='вне чата'):
        workspace.path()


def test_four_groups_and_empty_summary_are_filled_separately(tmp_path, monkeypatch):
    path = tmp_path / '4 группы и сводная.xlsx'
    wb = Workbook()
    summary = wb.active
    summary.title = 'Сводная'
    header = ['ФИО', '№ упражнения', 'Результат', 'Балл', 'Сумма баллов']
    summary.append(header)
    for i in range(20):
        summary.append([f'Сводный {i}', None, None, None, None])
    for index in range(1, 5):
        ws = wb.create_sheet(f'Группа {index}')
        ws.append(header)
        ws.append([f'Студент {index}', 1, 20 if index % 2 else 25, None, None])
        ws['D2'].fill = PatternFill('solid', fgColor='FFFF00')
    wb.save(path)
    old = package(path)['xl/worksheets/sheet1.xml']
    norms = tmp_path / 'norms.txt'
    norms.write_text(NORMS)
    agent = make_agent(tmp_path, monkeypatch)
    agent._json = lambda *a, **kw: pytest.fail('No model arithmetic')
    result = agent.handle('Заполни баллы', [path, norms])
    assert not result.incomplete, result.text
    output = load_workbook(result.files[0].path, data_only=True)
    for index in range(1, 5):
        assert output[f'Группа {index}']['D2'].value == (40 if index % 2 else 50)
        assert output[f'Группа {index}']['D2'].fill.fgColor.rgb.endswith('FFFF00')
        assert f'Лист «Группа {index}»' in result.text
    assert output['Сводная']['C2'].value is None
    assert old == package(result.files[0].path)['xl/worksheets/sheet1.xml']
    assert len(output.sheetnames) == 6
    output.close()


def test_review_until_ready_uses_updated_file_and_stops_when_no_more_changes(tmp_path, monkeypatch):
    path = tmp_path / 'draft.docx'
    doc = Document()
    doc.add_paragraph('Первый текст')
    doc.save(path)
    agent = make_agent(tmp_path, monkeypatch)
    calls = []
    def review(prompt, schema, max_tokens=0):
        calls.append(prompt)
        if len(calls) == 1:
            return dict(summary='Исправлено', changes=[dict(id='word/document.xml:p0', before='Первый текст', after='Доработанный текст', kind='text', evidence='')])
        assert 'Доработанный текст' in prompt
        return dict(summary='Подтверждённых дефектов больше не найдено', changes=[])
    agent._json = review
    result = agent.handle('Доработай документ до идеала', [path])
    assert not result.incomplete and len(calls) == 2
    assert Document(result.files[0].path).paragraphs[0].text == 'Доработанный текст'


def test_stop_before_publish_retains_original(tmp_path, monkeypatch):
    import threading
    path = tmp_path / 'draft.docx'
    doc = Document()
    doc.add_paragraph('Исходный текст')
    doc.save(path)
    event = threading.Event()
    agent = make_agent(tmp_path, monkeypatch)
    def cancel_after_generation(*a, **kw):
        event.set()
        return dict(summary='Изменено', changes=[dict(id='word/document.xml:p0', before='Исходный текст', after='Новый текст', kind='text', evidence='')])
    agent._json = cancel_after_generation
    result = agent.handle('Исправь документ', [path], cancel_event=event)
    assert result.action == 'cancelled' and agent._documents.active['revision'] == 0
    assert Document(agent._documents.path()).paragraphs[0].text == 'Исходный текст'


def test_user_can_target_earlier_uploaded_file_instead_of_active_one(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch)
    for name, text in [('first.docx', 'Иванов'), ('second.docx', 'Петров')]:
        path = tmp_path / name
        doc = Document(); doc.add_paragraph(text); doc.save(path)
        item = read_attachment(agent.store.add(path), agent.llm)
    agent._documents.register('second.docx', agent.store.items[1].path, key=agent.store.items[1].id)
    agent._json = lambda *a, **kw: dict(summary='Исправлено', changes=[dict(id='word/document.xml:p0', before='Иванов', after='Сидоров', kind='text', evidence='')])
    result = agent.handle('Исправь first.docx: замени Иванов на Сидоров')
    assert not result.incomplete and result.files[0].name == 'first.docx'
    assert Document(result.files[0].path).paragraphs[0].text == 'Сидоров'


def test_all_explicitly_requested_documents_get_separate_revisions(tmp_path, monkeypatch):
    paths = []
    for name in ['first.docx', 'second.docx']:
        path = tmp_path / name
        doc = Document(); doc.add_paragraph('Иванов'); doc.save(path)
        paths.append(path)
    agent = make_agent(tmp_path, monkeypatch)
    agent._json = lambda *a, **kw: dict(summary='Исправлено', changes=[dict(id='word/document.xml:p0', before='Иванов', after='Петров', kind='text', evidence='')])
    result = agent.handle('Доработай все файлы: замени Иванов на Петров', paths)
    assert not result.incomplete and len(result.files) == 2
    assert all(Document(f.path).paragraphs[0].text == 'Петров' for f in result.files)


def test_empty_summary_is_not_returned_as_successfully_filled(tmp_path, monkeypatch):
    path = tmp_path / 'empty.xlsx'
    wb = Workbook()
    wb.active.append(['ФИО', '№ упражнения', 'Результат', 'Балл'])
    wb.active.append(['Иванов', None, None, None])
    wb.save(path)
    norms = tmp_path / 'norms.txt'; norms.write_text(NORMS)
    agent = make_agent(tmp_path, monkeypatch)
    agent._json = lambda *a, **kw: pytest.fail('Should diagnose empty inputs directly')
    result = agent.handle('Заполни баллы', [path, norms])
    assert result.incomplete == 'не найдены исходные результаты' and not result.files


def test_active_revision_can_be_reissued_without_model_or_conversion(tmp_path, monkeypatch):
    path=tmp_path/'letter.docx';doc=Document();doc.add_paragraph('Иванов');doc.save(path)
    agent=make_agent(tmp_path,monkeypatch)
    agent._documents.register(path.name,path)
    agent._json=lambda *a,**kw: pytest.fail('Reissuing a file must not ask a model')
    result=agent.handle('Выдай последнюю версию')
    assert not result.incomplete and result.files[0].path.read_bytes()==path.read_bytes()


def test_workspace_can_recover_its_last_backup_after_manifest_corruption(tmp_path):
    path=plain_excel(tmp_path/'source.xlsx')
    root=tmp_path/'documents'
    workspace=DocumentWorkspace(root);workspace.register(path.name,path);workspace.save()
    (root/'current.json').write_text('{broken')
    restored=DocumentWorkspace(root)
    assert restored.active and restored.path().is_file()
    assert 'резервной' in restored.error


def test_attachment_cache_survives_restart_without_rereading_the_original(tmp_path):
    source=tmp_path/'norms.txt';source.write_text(NORMS)
    store=AttachmentStore(tmp_path/'att')
    item=read_attachment(store.add(source),None)
    second=AttachmentStore(tmp_path/'att')
    assert second.items[0].text==item.text and second.items[0].read


def test_legacy_chat_recovers_owned_copies_when_no_attachment_index_exists(tmp_path):
    folder=tmp_path/'att';folder.mkdir()
    (folder/'ab12cd34_norms.txt').write_text(NORMS)
    (folder/'unrelated.txt').write_text('Do not auto-import')
    store=AttachmentStore(folder)
    assert len(store.items)==1 and store.items[0].name=='norms.txt'
    assert not store.items[0].read
    assert read_attachment(store.items[0],None).ok
    assert AttachmentStore(folder).items[0].read


def test_exact_excel_grading_does_not_require_a_language_model(tmp_path, monkeypatch):
    source=plain_excel(tmp_path/'source.xlsx');norms=tmp_path/'norms.txt';norms.write_text(NORMS)
    agent=make_agent(tmp_path,monkeypatch)
    agent.manager.llm_model=None
    result=agent.handle('Заполни баллы',[source,norms])
    assert not result.incomplete and result.files
    book=load_workbook(result.files[0].path,data_only=True)
    assert book['Оригинал']['D2'].value==40
    book.close()


def test_updated_upload_with_same_name_continues_one_document_history(tmp_path):
    path=tmp_path/'letter.docx'
    doc=Document();doc.add_paragraph('Первый текст');doc.save(path)
    workspace=DocumentWorkspace(tmp_path/'documents')
    original=workspace.register(path.name,path,key='first')
    doc=Document();doc.add_paragraph('Обновленный текст');doc.save(path)
    updated=workspace.register(path.name,path,key='second')
    assert updated['id']==original['id'] and updated['revision']==1
    assert Document(workspace.path()).paragraphs[0].text=='Обновленный текст'
    restored=workspace.rollback(0)
    assert Document(restored).paragraphs[0].text=='Первый текст'
