"""Install the GGUF models and llama.cpp runtime actually used by ChatKodi.

Use repository-pinned Hugging Face metadata and GitHub release SHA-256 digests;
never report success merely because a legacy Transformers directory exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.request
import uuid
import zipfile
from pathlib import Path

from core.atomic_io import atomic_write_json
from core.paths import runtime_model_cache_dir

MODEL_SPECS = [
    ('embeddings', ['Qwen/Qwen3-Embedding-0.6B-GGUF'], r'(?i)Qwen3-Embedding-0\.6B.*Q8_0\.gguf$', None),
    ('llms', ['Qwen/Qwen3.5-4B-GGUF', 'unsloth/Qwen3.5-4B-GGUF'], r'(?i)Qwen3\.5-4B.*Q4_K_M\.gguf$', None),
    ('llms', ['Qwen/Qwen3.5-4B-GGUF', 'unsloth/Qwen3.5-4B-GGUF'], r'(?i)mmproj.*(?:F16|BF16)\.gguf$', 'mmproj-Qwen3.5-4B-F16.gguf'),
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_models(cache, *, force=False):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError
    from core.llm.model_select import find_embedding_model, find_llm, find_mmproj
    from core.model_integrity import inspect_gguf_model

    llm = find_llm()
    available = [find_embedding_model(), llm, find_mmproj(llm) if llm else None]
    api, info_cache = HfApi(), {}
    for (folder, repos, pattern, renamed), existing in zip(MODEL_SPECS, available):
        if not force and existing is not None and inspect_gguf_model(existing).get('ok'):
            print(f'Используется существующий GGUF: {existing.name}')
            continue
        found = None
        for repo in repos:
            try:
                if repo not in info_cache:
                    info_cache[repo] = api.model_info(repo, files_metadata=True)
                info = info_cache[repo]
            except RepositoryNotFoundError:
                continue
            matches = [f for f in info.siblings if re.search(pattern, f.rfilename) and '-of-' not in f.rfilename]
            if matches:
                found = repo, info, sorted(matches, key=lambda f: f.rfilename)[0]
                break
        if found is None:
            raise RuntimeError('В источниках не найден совместимый GGUF: ' + ', '.join(repos))
        repo, info, file = found
        expected = getattr(file.lfs, 'sha256', None) if file.lfs else None
        if not expected or not re.fullmatch(r'[a-fA-F0-9]{64}', expected):
            raise RuntimeError(f'Нет SHA-256 в метаданных {repo}/{file.rfilename}; загрузка остановлена')
        print(f'Источник модели: {repo}, commit {info.sha}; файл {file.rfilename}')
        downloaded = Path(hf_hub_download(repo_id=repo, filename=file.rfilename, revision=info.sha, force_download=force))
        if sha256(downloaded).lower() != expected.lower():
            raise RuntimeError(f'Контрольная сумма не совпала: {file.rfilename}')
        if not inspect_gguf_model(downloaded).get('ok'):
            raise RuntimeError(f'Проверка структуры GGUF не прошла: {file.rfilename}')
        target = cache / folder / (renamed or Path(file.rfilename).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_suffix('.gguf.pending')
        shutil.copyfile(downloaded, pending)
        pending.replace(target)
        atomic_write_json(target.with_suffix('.source.json'), dict(repo=repo, revision=info.sha,
                          filename=file.rfilename, sha256=expected, size=target.stat().st_size))
        print(f'GGUF проверен и сохранён: {target.name}')


def get_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'ChatKodi-setup', 'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def cuda_version():
    try:
        p = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        m = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', p.stdout)
        return tuple(map(int, m.groups())) if m else None
    except (OSError, subprocess.SubprocessError):
        return None


def select_assets(release, cuda):
    assets = release.get('assets') or []
    if cuda:
        bins = []
        for asset in assets:
            match = re.fullmatch(r'llama-.*-bin-win-cuda-(\d+)\.(\d+)-x64\.zip', asset['name'])
            if match and tuple(map(int, match.groups())) <= cuda:
                bins.append((tuple(map(int, match.groups())), asset))
        if not bins:
            raise RuntimeError('Для драйвера NVIDIA не найден совместимый CUDA runtime llama.cpp. Обновите драйвер либо установите llama.cpp вручную.')
        version, main = max(bins, key=lambda x: x[0])
        extra = [a for a in assets if re.fullmatch(r'cudart-.*win-cuda-' + re.escape(f'{version[0]}.{version[1]}') + r'-x64\.zip', a['name'])]
        return [main] + extra
    candidates = [a for a in assets if re.fullmatch(r'llama-.*-bin-win-(?:cpu|avx2)-x64\.zip', a['name'])]
    if len(candidates) != 1:
        raise RuntimeError('Не найден однозначный Windows x64 CPU runtime llama.cpp')
    return candidates


def version_ok(exe):
    try:
        result = subprocess.run([str(exe), '--version'], capture_output=True, timeout=30,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def install_runtime(cache, *, force=False):
    from core.llm.server import find_llama_server
    current = find_llama_server()
    if current and not force and version_ok(current):
        print('llama-server доступен: ' + str(current))
        return
    if platform.system() != 'Windows' or platform.machine().lower() not in {'amd64', 'x86_64'}:
        raise RuntimeError('Автоустановка llama.cpp поддерживает Windows x64; на этой системе установите llama-server отдельно')
    release = get_json('https://api.github.com/repos/ggml-org/llama.cpp/releases/latest')
    assets = select_assets(release, cuda_version())
    managed = cache / 'llama.cpp' / 'managed'
    managed.mkdir(parents=True, exist_ok=True)
    stage = managed / ('install-' + uuid.uuid4().hex[:8])
    stage.mkdir()
    try:
        for asset in assets:
            digest = asset.get('digest') or ''
            if not re.fullmatch(r'sha256:[a-fA-F0-9]{64}', digest):
                raise RuntimeError('GitHub не предоставил SHA-256 для ' + asset['name'])
            archive = stage / 'download.zip'
            request = urllib.request.Request(asset['browser_download_url'], headers={'User-Agent': 'ChatKodi-setup'})
            print('Загрузка llama.cpp: ' + asset['name'])
            with urllib.request.urlopen(request, timeout=120) as response, archive.open('wb') as f:
                shutil.copyfileobj(response, f)
            if sha256(archive).lower() != digest.split(':', 1)[1].lower():
                raise RuntimeError('Не совпала контрольная сумма ' + asset['name'])
            with zipfile.ZipFile(archive) as z:
                if z.testzip():
                    raise RuntimeError('Повреждён ZIP runtime')
                for entry in z.infolist():
                    dest = (stage / entry.filename).resolve()
                    if not dest.is_relative_to(stage.resolve()):
                        raise RuntimeError('Недопустимый путь в ZIP runtime')
                z.extractall(stage)
            archive.unlink()
        exes = list(stage.rglob('llama-server.exe'))
        if len(exes) != 1 or not version_ok(exes[0]):
            raise RuntimeError('llama-server не запускается. Проверьте драйвер NVIDIA и зависимости runtime по журналу.')
        atomic_write_json(managed / 'current.json', dict(executable=str(exes[0].relative_to(managed)), release=release['tag_name']))
        print('llama-server установлен, проверка --version пройдена')
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def check_ready():
    from core.llm.model_select import find_embedding_model, find_llm, find_mmproj
    from core.llm.server import find_llama_server
    from core.model_integrity import inspect_gguf_model
    llm, embedding, server = find_llm(), find_embedding_model(), find_llama_server()
    projector = find_mmproj(llm) if llm else None
    items = {'LLM GGUF': llm, 'Embedding GGUF': embedding, 'Vision mmproj': projector, 'llama-server': server}
    ready = True
    for name, path in items.items():
        good = bool(path) and (version_ok(path) if name == 'llama-server' else inspect_gguf_model(path).get('ok'))
        print(f'{name}: {"OK" if good else "НЕ ГОТОВО"}' + (f' ({path.name})' if path else ''))
        ready &= good
    return bool(ready)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Совместимые GGUF-модели и llama.cpp для ChatKodi')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--repair', action='store_true')
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--manual', action='store_true')
    args = parser.parse_args(argv)
    if args.cleanup or args.manual:
        print('Удаление старых моделей не выполняется. Для ручного выбора используйте вкладку «Модели» лаунчера.')
        return 1
    if args.check or args.list or args.offline:
        return 0 if check_ready() else 1
    try:
        cache = runtime_model_cache_dir()
        ensure_models(cache, force=args.force)
        install_runtime(cache, force=args.force)
        if not check_ready():
            raise RuntimeError('После установки не все компоненты готовы; ложный успех не выставляется')
        print('GGUF-модели, проектор зрения и llama-server готовы.')
        return 0
    except Exception as exc:
        print(f'Подготовка не завершена: {exc}')
        return 1
