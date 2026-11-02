#!/usr/bin/env python3
"""Собирает расширенный диагностический архив для передачи разработчику/поддержке."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import os
import platform
import shutil
import sys
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

from core.config import get_config
from core.paths import project_root, resolve_project_path, runtime_root
from tools.healthcheck import run_healthcheck
from tools.doctor import build_doctor_report


def _iter_existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]


def _copy_tree_if_exists(src: Path, dst_root: Path, root_for_rel: Path, *, exclude_roots: Iterable[Path] = ()) -> None:
    """Копирует дерево, не заходя в собственную временную папку bundle.

    Старый вариант создавал temp внутри .chatkodi/logs/support, а потом rglob() по
    logs копировал эту же temp-папку внутрь себя. Так появлялись рекурсивные
    ``_bundle_tmp/.../_bundle_tmp/...``. Здесь temp живёт вне runtime, плюс есть
    явный exclude на случай пользовательского output-dir внутри logs.
    """
    if not src.exists():
        return
    exclude_resolved = []
    for item in exclude_roots or []:
        try:
            exclude_resolved.append(item.resolve())
        except Exception:
            pass
    for path in src.rglob('*'):
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if any(resolved == ex or resolved.is_relative_to(ex) for ex in exclude_resolved):
            continue
        if any(part.startswith('_bundle_tmp_') for part in path.parts):
            continue
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root_for_rel) if path.is_relative_to(root_for_rel) else Path(path.name)
        except Exception:
            rel = Path(path.name)
        dest = dst_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def _copy_latest_files(src: Path, dst_root: Path, pattern: str, limit: int = 5) -> None:
    if not src.exists():
        return
    files = sorted((p for p in src.rglob(pattern) if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    for path in files:
        dest = dst_root / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def build_support_bundle(output_dir: Path | None = None) -> Path:
    root = project_root()
    runtime = runtime_root()
    cfg = get_config()
    version = cfg.data.get('version', 'unknown')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = output_dir or resolve_project_path('logs/support')
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / f'chatkodi_support_bundle_{version}_{ts}.zip'

    # Temp намеренно создаётся вне .chatkodi/logs/support. Это устраняет
    # самую неприятную рекурсию: support-bundle больше не копирует сам себя.
    temp_dir = Path(tempfile.mkdtemp(prefix=f'chatkodi_bundle_{ts}_'))

    try:
        metadata = {
            'project_version': version,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'platform': platform.platform(),
            'python': sys.version,
            'cwd': str(root),
            'runtime_root': str(runtime),
            'config_path': os.environ.get('CHATKODI_CONFIG', ''),
            'runtime_dir_env': os.environ.get('CHATKODI_RUNTIME_DIR', ''),
            'query_planner_enabled': bool((cfg.data.get('query_planner') or {}).get('enabled', False)),
            'query_planner_use_llm_router': bool((cfg.data.get('query_planner') or {}).get('use_llm_router', False)),
        }
        (temp_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        (temp_dir / 'healthcheck.json').write_text(json.dumps(run_healthcheck(), ensure_ascii=False, indent=2), encoding='utf-8')
        (temp_dir / 'doctor_report.json').write_text(json.dumps(build_doctor_report(), ensure_ascii=False, indent=2), encoding='utf-8')
        event_meta_path = runtime / 'events_meta.json'
        if event_meta_path.exists():
            try:
                event_meta = json.loads(event_meta_path.read_text(encoding='utf-8'))
            except Exception:
                event_meta = {}
            (temp_dir / 'event_store_report.json').write_text(json.dumps(event_meta, ensure_ascii=False, indent=2), encoding='utf-8')

        files_to_copy = _iter_existing([
            root / 'chatkodi_config.json',
            resolve_project_path('chatkodi_settings.json'),
            root / 'VERSION.txt',
            runtime / 'device_config.json',
            runtime / 'embedding_cache_meta.json',
            runtime / 'faiss_meta.pkl',
            runtime / 'events.db',
            runtime / 'events_meta.json',
            runtime / 'query_history.jsonl',
        ])

        for file_path in files_to_copy:
            try:
                rel = file_path.relative_to(root)
            except Exception:
                rel = Path('runtime') / file_path.relative_to(runtime)
            dest = temp_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)

        for log_dir in [resolve_project_path('logs'), runtime / 'logs']:
            if log_dir.exists():
                _copy_tree_if_exists(log_dir, temp_dir, root if log_dir.is_relative_to(root) else runtime, exclude_roots=[temp_dir, out_dir])

        for rel_name in [Path('logs/diagnostics'), Path('logs/telemetry'), Path('logs/recovery'), Path('logs/benchmarks')]:
            src = runtime / rel_name
            if src.exists():
                _copy_tree_if_exists(src, temp_dir / 'runtime', runtime, exclude_roots=[temp_dir, out_dir])

        sample_cases = root / 'benchmark_cases.sample.json'
        if sample_cases.exists():
            dest = temp_dir / 'benchmark_cases.sample.json'
            shutil.copy2(sample_cases, dest)

        # Самые свежие артефакты в отдельной папке summary/
        summary_root = temp_dir / 'summary'
        _copy_latest_files(runtime / 'logs' / 'diagnostics', summary_root / 'diagnostics', '*.json', limit=5)
        _copy_latest_files(runtime / 'logs' / 'telemetry', summary_root / 'telemetry', '*.jsonl', limit=3)
        _copy_latest_files(runtime / 'logs' / 'benchmarks', summary_root / 'benchmarks', '*.json', limit=3)
        _copy_latest_files(runtime / 'logs' / 'recovery', summary_root / 'recovery', '*.json', limit=3)

        with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for path in temp_dir.rglob('*'):
                if path.is_file():
                    zf.write(path, arcname=path.relative_to(temp_dir))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description='Собирает support bundle для ChatKodi')
    parser.add_argument('--output-dir', default='', help='каталог для готового архива')
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    bundle = build_support_bundle(output_dir)
    print(f'Support bundle created: {bundle}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
