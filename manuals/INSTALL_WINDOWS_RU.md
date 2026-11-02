# Установка ChatKodi 2.0 на Windows

Для обычной работы установка не нужна: распакуйте portable-архив и запустите `START_CHATKODI.cmd`.
Модели и llama.cpp уже лежат в архиве. Ниже — установка из исходного кода (для разработки).

## Что нужно

- Python 3.11 (64-bit);
- для работы на видеокарте — драйвер NVIDIA с поддержкой CUDA 12 (версия 527 или новее);
- llama.cpp для Windows с CUDA 12.4 (`llama-b<сборка>-bin-win-cuda-12.4-x64.zip`
  и `cudart-llama-bin-win-cuda-12.4-x64.zip` со страницы релизов github.com/ggml-org/llama.cpp);
- модели GGUF: `Qwen3.5-4B-Q4_K_M.gguf` и `Qwen3-Embedding-0.6B-Q8_0.gguf`.

## Окружение

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt -r requirements-build.txt
```

PyTorch не нужен: обе модели работают в `llama-server`.

## Модели и llama.cpp

Положите файлы так (папка `models` рядом с папкой кода или любая другая):

```text
models\
├─ llms\Qwen3.5-4B-Q4_K_M.gguf
├─ embeddings\Qwen3-Embedding-0.6B-Q8_0.gguf
└─ llama.cpp\<сборка>\llama-server.exe  (оба zip распаковать в одну папку)
```

и укажите путь перед запуском:

```powershell
$env:CHATKODI_EXTRA_MODEL_CACHE_DIRS = "D:\path\to\models"
.\.venv\Scripts\python.exe main.py
```

## Проверка и сборка

```powershell
.\.venv\Scripts\python.exe -m tools.doctor
.\.venv\Scripts\python.exe -m pytest -q
powershell -ExecutionPolicy Bypass -File scripts\build\build_release.ps1 -SkipInstaller
```

Скрипт сборки берёт модели из `..\models` (или из `CHATKODI_MODEL_SOURCE_ROOT`).
