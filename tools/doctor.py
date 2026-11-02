#!/usr/bin/env python3
"""Расширенная самопроверка проекта и окружения ChatKodi."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tools.healthcheck import run_healthcheck


def build_doctor_report() -> Dict[str, Any]:
    report = run_healthcheck()
    recommendations: List[str] = []
    blockers: List[str] = []

    py = report.get('python', {})
    if not py.get('ok', False):
        blockers.append('Требуется Python 3.10+ и корректное виртуальное окружение.')

    project = report.get('project', {})
    if not project.get('ok', False):
        missing = ', '.join(project.get('missing', [])[:10])
        blockers.append(f'Не хватает файлов проекта: {missing}')

    readers = report.get('document_readers', {})
    if not readers.get('ok', True):
        blockers.extend(readers.get('errors', {}).values())

    docs = report.get('documents', {})
    if docs.get('count', 0) == 0:
        recommendations.append('В папке документов нет поддерживаемых файлов. Положите пользовательские документы в docs/ или выберите другую папку корпуса.')

    models = report.get('models', {})
    if not models.get('embeddings', False):
        blockers.append('Embedding-модель не найдена. Без неё поиск и индексация не запустятся.')
    if not models.get('llm', False):
        recommendations.append('LLM-модель не найдена. Structured-режимы будут работать, но чат и полный RAG будут ограничены.')
    if models.get('error'):
        recommendations.append(f'Проверка моделей вернула ошибку: {models.get("error")}')
    for model_key, label in (("embeddings_integrity", "Embedding-модель"), ("llm_integrity", "LLM-модель")):
        integrity = models.get(model_key) or {}
        if integrity and not integrity.get("ok", False):
            details = '; '.join(str(item) for item in integrity.get("errors") or [])
            blockers.append(f'{label} скопирована не полностью: {details}')

    runtime = report.get('runtime_profile', {})
    if runtime.get('has_cuda') and float(runtime.get('gpu_vram_gb') or 0.0) <= 8.1:
        recommendations.append('GPU с 8 ГБ VRAM: для больших корпусов используйте chunk-first режим и structured-ответы.')
    if float(runtime.get('total_ram_gb') or 0.0) <= 16.5:
        recommendations.append('На системе мало RAM. Рекомендуется ограничить контекст и держать корпус на диске, а не в памяти.')

    if not blockers and not recommendations:
        recommendations.append('Критичных проблем не найдено. Можно запускать базовые сценарии и benchmark-прогоны.')

    report['doctor'] = {
        'status': 'ok' if not blockers else 'issues_found',
        'blockers': blockers,
        'recommendations': recommendations,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description='Расширенная проверка окружения ChatKodi')
    parser.add_argument('--json', action='store_true', help='вывести полный отчёт в JSON')
    parser.add_argument('--save', default='', help='сохранить отчёт в указанный файл JSON')
    args = parser.parse_args()

    report = build_doctor_report()
    if args.save:
        out = Path(args.save).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"ChatKodi doctor | version={report['project_version']}")
        print(f"Runtime: {report['runtime_profile']['summary']}")
        doctor = report.get('doctor', {})
        print(f"Status: {doctor.get('status', 'unknown')}")
        if doctor.get('blockers'):
            print('Блокирующие проблемы:')
            for item in doctor['blockers']:
                print(f"- {item}")
        if doctor.get('recommendations'):
            print('Рекомендации:')
            for item in doctor['recommendations']:
                print(f"- {item}")
    return 0 if report.get('doctor', {}).get('status') == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
