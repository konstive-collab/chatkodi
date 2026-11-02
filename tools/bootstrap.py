#!/usr/bin/env python3
"""Bootstrap окружения ChatKodi 1.8 с пошаговыми действиями и venv."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tools.doctor import build_doctor_report


def _run(cmd: list[str]) -> int:
    print('>', ' '.join(cmd))
    return subprocess.call(cmd)




def _venv_python(root: Path) -> Path:
    if sys.platform == "win32":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _ensure_venv(root: Path) -> Path:
    py = _venv_python(root)
    if py.exists():
        return py
    print(f"Creating virtual environment: {root / '.venv'}")
    code = _run([sys.executable, '-m', 'venv', str(root / '.venv')])
    if code != 0 or not py.exists():
        raise RuntimeError('Не удалось создать .venv. Проверьте Python и модуль venv.')
    return py


def _delegate_to_venv(root: Path, args: list[str]) -> int:
    py = _ensure_venv(root)
    cmd = [str(py), str(root / 'tools' / 'bootstrap.py'), *args]
    return _run(cmd)

def _pip_install(requirements_path: Path) -> int:
    return _run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_path), '--disable-pip-version-check'])


def main() -> int:
    parser = argparse.ArgumentParser(description='Bootstrap окружения ChatKodi')
    parser.add_argument('--create-venv', action='store_true', help='создать .venv и выполнить установку внутри него')
    parser.add_argument('--download-models', action='store_true', help='после проверок вызвать download_models.py')
    parser.add_argument('--install-requirements', action='store_true', help='установить зависимости приложения')
    parser.add_argument('--requirements', default='requirements.txt', help='какой requirements-файл использовать')
    parser.add_argument('--install-launcher', action='store_true', help='установить только зависимости лаунчера')
    parser.add_argument('--doctor-json', action='store_true', help='печатать doctor-отчёт в JSON')
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent

    if args.create_venv and sys.prefix == getattr(sys, 'base_prefix', sys.prefix):
        delegated_args = []
        if args.install_requirements:
            delegated_args.extend(['--install-requirements', '--requirements', args.requirements])
        if args.install_launcher:
            delegated_args.append('--install-launcher')
        if args.download_models:
            delegated_args.append('--download-models')
        if args.doctor_json:
            delegated_args.append('--doctor-json')
        if not delegated_args:
            delegated_args = ['--install-requirements']
        return _delegate_to_venv(root, delegated_args)

    # ZIP distributions intentionally exclude the user's personal config.
    # Materialize defaults only when no user configuration exists.
    config = root / 'chatkodi_config.json'
    if not config.exists():
        from core.config import DEFAULT_CONFIG
        from core.atomic_io import atomic_write_json
        atomic_write_json(config, DEFAULT_CONFIG)

    report = build_doctor_report()
    doctor = report.get('doctor', {})

    if args.doctor_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print('ChatKodi bootstrap')
        print(f"- project_version: {report.get('project_version')}")
        print(f"- runtime_profile: {report['runtime_profile']['summary']}")
        print(f"- status: {doctor.get('status')}")
        if doctor.get('blockers'):
            print('- blockers:')
            for item in doctor['blockers']:
                print(f"  * {item}")
        if doctor.get('recommendations'):
            print('- recommendations:')
            for item in doctor['recommendations'][:6]:
                print(f"  * {item}")

    if args.install_launcher:
        req = root / 'requirements-launcher.txt'
        rc = _pip_install(req if req.exists() else root / 'requirements.txt')
        if rc != 0:
            return rc

    if args.install_requirements:
        req = root / args.requirements
        if not req.exists():
            req = root / 'requirements.txt'
        rc = _pip_install(req)
        if rc != 0:
            return rc
        from core.document_dependencies import check_document_readers
        readers = check_document_readers()
        if not readers['ok']:
            for error in readers['errors'].values():
                print(error)
            return 1

    if args.download_models:
        return _run([sys.executable, str(root / 'download_models.py')])

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
