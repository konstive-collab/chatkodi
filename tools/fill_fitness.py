"""Deterministic local PDF → original Excel workflow, without loading an LLM."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from core.agent.pdf_tables import read_pdf
from core.agent.workbook_template import calculate_book, fitness_tables
from core.paths import runtime_root
from core.atomic_io import atomic_write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description='Заполнить исходную Excel-ведомость по нормативному PDF')
    parser.add_argument('--excel', required=True, type=Path)
    parser.add_argument('--norms', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args=parser.parse_args(argv)
    if args.excel.resolve()==args.output.resolve():
        parser.error('Для результата укажите новый файл: исходная ведомость сохраняется')
    if args.output.exists():
        parser.error('Файл результата уже существует; укажите новое имя, чтобы сохранить предыдущую версию')
    if not fitness_tables(args.excel):
        parser.error('В книге не найдены листы с исходными результатами упражнений')
    signature=hashlib.sha256(args.norms.read_bytes()).hexdigest()
    cache=runtime_root()/'norm_pdf_cache'/f'{signature}.txt'
    if cache.is_file():
        text=cache.read_text(encoding='utf-8')
    else:
        reading=read_pdf(args.norms, progress=print)
        if reading.issues:
            print('PDF прочитан не полностью: '+'; '.join(reading.issues))
            return 1
        text=reading.text
        cache.parent.mkdir(parents=True,exist_ok=True)
        cache.write_text(text,encoding='utf-8')
    body,used,incomplete=calculate_book(args.excel,[(args.norms.name,text)],args.output,progress=print)
    report=args.output.with_suffix('.report.txt')
    report.write_text(body,encoding='utf-8')
    print('Результат: '+str(args.output))
    print('Проверка и источники: '+str(report))
    if incomplete:
        print('Частично завершено: '+incomplete)
        return 2  # file is valid; some results or normative criteria remain missing
    print('Расчёт завершён.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
