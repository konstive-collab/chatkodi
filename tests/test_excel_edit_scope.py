"""Addressed corrections in filled workbooks must not pull every score formula."""
from types import SimpleNamespace

import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font

from core.agent.agent import Agent
from core.agent.attachments import AttachmentStore
from core.agent.office_documents import OfficeFile, encoded


REQUEST = ('В последней версии ведомости на листе «231» измени M8 с 20 на 25. '
           'Пересчитай баллы и сумму, сохрани оформление и верни исправленный Excel.')
NORMS = ('Таблица перевода результата выполнения упражнения № 1 в баллы\n'
         '| Количество раз | Баллы |\n|---|---|\n|20|40|\n|25|50|\n\n'
         'Таблица перевода результата выполнения упражнения № 2 в баллы\n'
         '| Количество раз | Баллы |\n|---|---|\n|45|55|\n')


def results_book(path):
    book = Workbook(); sheet = book.active; sheet.title = '231'
    sheet['A1'] = 'Учебная ведомость'
    sheet['B4'] = 'Фамилия'; sheet['I4'] = 'Курс обучения'; sheet['J4'] = 'Пол'
    sheet['AA4'] = 'Сумма баллов'
    sheet['L5'] = 'Общие физические упражнения'
    for first in (12, 15, 18, 21, 24):
        for offset, title in enumerate(('№ упражнения', 'Результат', 'Балл')):
            sheet.cell(6, first + offset, title)
    for col in range(1, 28):
        sheet.cell(7, col, col)
    for row in range(8, 28):
        for col, value in {2: f'Участник {row}', 9: '3 курс', 10: 'м', 12: 1, 13: 20, 15: 2, 16: 45}.items():
            sheet.cell(row, col, value)
    sheet['M8'].font = Font(name='Arial', bold=True)
    sheet.column_dimensions['B'].width = 32
    sheet.print_area = 'A1:AA27'
    book.save(path); book.close()
    return path


def large_filled_book(path):
    results_book(path)
    book = load_workbook(path)
    for row in range(8, 28):
        for col in (14, 17, 27):
            book.active.cell(row, col, '=IFERROR(' + '+'.join(['1'] * 500) + ',0)')
    book.save(path); book.close()
    native = OfficeFile(path)
    audit = native.tree(native.sheets[0][1])
    native.add_sheet('Нормативы ChatKodi', encoded(audit))
    native.save(path)
    return path


@pytest.mark.parametrize('message', [REQUEST, "На листе '231' измени $m$8 с 20 на 25", "Измени '231'!$M$8 с 20 на 25"])
def test_explicit_cell_overrides_common_words_and_excludes_audit_sheet(tmp_path, message):
    source = large_filled_book(tmp_path / 'filled.xlsx')
    native = OfficeFile(source)
    inventory = native.inventory(message, limit=11000)
    assert len(inventory) <= 11000
    assert set(native.targets) == {'xlsx:0:M8'}
    assert native.targets['xlsx:0:M8'][2] == '20'
    assert 'xlsx:1:' not in inventory and 'xlsx:0:M9' not in inventory
    with pytest.raises(ValueError, match='неизвестный'):
        native.patch([dict(id='xlsx:0:M9', before='20', after='25', kind='number')])


def test_small_workbook_still_restricts_editable_targets_to_requested_range(tmp_path):
    native = OfficeFile(results_book(tmp_path / 'book.xlsx'))
    native.inventory('На листе «231» заполни M8:N9', limit=22000)
    assert set(native.targets) == {'xlsx:0:M8', 'xlsx:0:N8', 'xlsx:0:M9', 'xlsx:0:N9'}
    assert native.targets['xlsx:0:N8'][2] == ''


def test_unknown_or_ambiguous_sheet_is_never_guessed(tmp_path):
    source = results_book(tmp_path / 'book.xlsx')
    book = load_workbook(source); book.copy_worksheet(book.active).title = '232'; book.save(source); book.close()
    native = OfficeFile(source)
    with pytest.raises(ValueError, match='Не найден лист'):
        native.inventory('На листе «999» измени M8 на 25')
    with pytest.raises(ValueError, match='Укажите лист'):
        native.inventory('Измени M8 с 20 на 25')
    native.inventory('На листе «232» измени M8 с 20 на 25')
    assert set(native.targets) == {'xlsx:1:M8'}


def test_surname_selection_is_not_expanded_by_score_and_result_words(tmp_path):
    source = large_filled_book(tmp_path / 'filled.xlsx')
    native = OfficeFile(source)
    native.set_cell(native.sheets[0][1], 'B8', 'Иванов')
    native.save(source)
    native = OfficeFile(source)
    inventory = native.inventory('Исправь у Иванова результат упражнения 1 на 25, пересчитай баллы и сумму', limit=11000)
    assert 'xlsx:0:M8' in inventory and 'xlsx:0:M9' not in inventory
    assert 'xlsx:1:' not in inventory


def test_filled_fitness_workbook_edit_recalculates_after_restart_without_reupload(tmp_path, monkeypatch):
    monkeypatch.setenv('CHATKODI_RUNTIME_DIR', str(tmp_path / 'runtime'))
    source = results_book(tmp_path / 'source.xlsx')
    norms = tmp_path / 'norms.txt'; norms.write_text(NORMS)

    def create_agent():
        model = SimpleNamespace(is_available=lambda: True)
        return Agent(SimpleNamespace(llm_model=model, hybrid_search=None), AttachmentStore(tmp_path / 'attachments'))

    agent = create_agent()
    agent._json = lambda *a, **kw: pytest.fail('Initial exact grading needs no model')
    first = agent.handle('Заполни баллы', [source, norms])
    assert first.files, first.text
    initial_revision = agent._documents.active['revision']
    original_digest = agent._documents.active['digest']
    restarted = create_agent()

    def plan(prompt, *args, **kwargs):
        assert 'xlsx:0:M8' in prompt and 'xlsx:0:M9' not in prompt
        return dict(summary='Результат исправлен', changes=[dict(
            id='xlsx:0:M8', before='20', after='25', kind='number', evidence='25')])

    restarted._json = plan
    result = restarted.handle(REQUEST)
    assert result.files and result.action == 'grade', result.text
    book = load_workbook(result.files[0].path, data_only=True)
    sheet = book['231']
    assert (sheet['M8'].value, sheet['N8'].value, sheet['Q8'].value, sheet['AA8'].value) == (25, 50, 55, 105)
    assert sheet['M9'].value == 20 and sheet['AA9'].value == 95
    assert sheet['M8'].font.bold and sheet.column_dimensions['B'].width == 32
    assert len(book.sheetnames) == 2
    book.close()
    assert restarted._documents.active['digest'] != original_digest
    assert restarted._documents.active['revision'] == initial_revision + 1
    restored = create_agent()
    current = restored.handle('Выдай последнюю версию')
    current_book = load_workbook(current.files[0].path, data_only=True)
    assert current_book['231']['AA8'].value == 105
    current_book.close()
    undone = restored.handle('Верни предыдущую версию')
    old_book = load_workbook(undone.files[0].path, data_only=True)
    assert old_book['231']['M8'].value == 20 and old_book['231']['AA8'].value == 95
    old_book.close()
    original = load_workbook(source)
    assert original['231']['M8'].value == 20
    original.close()


def test_recalculation_failure_never_publishes_an_intermediate_revision(tmp_path, monkeypatch):
    from core.agent import workbook_template
    monkeypatch.setenv('CHATKODI_RUNTIME_DIR', str(tmp_path / 'runtime'))
    source = results_book(tmp_path / 'source.xlsx')
    norms = tmp_path / 'norms.txt'; norms.write_text(NORMS)
    manager = SimpleNamespace(llm_model=SimpleNamespace(is_available=lambda: True), hybrid_search=None)
    agent = Agent(manager, AttachmentStore(tmp_path / 'attachments'))
    first = agent.handle('Заполни баллы', [source, norms])
    assert first.files
    digest = agent._documents.active['digest']
    revision = agent._documents.active['revision']
    agent._json = lambda *a, **kw: dict(summary='Исправлено', changes=[dict(
        id='xlsx:0:M8', before='20', after='25', kind='number', evidence='25')])

    def broken(*args, **kwargs):
        raise ValueError('Не удалось пересчитать нормативы')

    monkeypatch.setattr(workbook_template, 'recalculate_edit', broken)
    result = agent.handle(REQUEST)
    assert not result.files and result.incomplete
    assert agent._documents.active['digest'] == digest and agent._documents.active['revision'] == revision
    book = load_workbook(agent._documents.path(), data_only=True)
    assert book['231']['M8'].value == 20 and book['231']['AA8'].value == 95
    book.close()
