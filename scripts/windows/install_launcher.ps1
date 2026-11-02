<#
ChatKodi 1.8 — минимальная установка лаунчера.
Сценарий для обычного пользователя:
  1) запустить INSTALL_LAUNCHER.bat или START_HERE.bat;
  2) дождаться установки лаунчера;
  3) в лаунчере нажать «Подготовить и запустить» или «Подготовить автоматически».

Этот скрипт ставит только то, что нужно для самого лаунчера. Полное окружение
ChatKodi, модели и диагностика выполняются уже из лаунчера.
#>
param(
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

function Write-Step($Text) {
    Write-Host "`n[ChatKodi] $Text" -ForegroundColor Green
}

function Find-Python {
    $candidates = @(
        @{Exe="py"; Args=@("-3.11")},
        @{Exe="py"; Args=@("-3.10")},
        @{Exe="py"; Args=@("-3")},
        @{Exe="python"; Args=@()}
    )
    foreach ($item in $candidates) {
        try {
            & $item.Exe @($item.Args) -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) { return $item }
        } catch { }
    }
    return $null
}

Write-Step "Корневая папка: $Root"
$py = Find-Python
if ($null -eq $py) {
    Write-Host "Python 3.10+ не найден. Пробую установить Python 3.11 через winget..." -ForegroundColor Yellow
    try {
        winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
    } catch {
        throw "Не удалось запустить winget. Установите Python 3.10+ вручную и повторите запуск. Ошибка: $_"
    }
    $py = Find-Python
    if ($null -eq $py) { throw "Python всё ещё не найден после winget. Перезапустите терминал или установите Python вручную." }
}

$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Write-Step "Создаю виртуальное окружение .venv"
    & $py.Exe @($py.Args) -m venv .venv
}

Write-Step "Обновляю pip"
& $VenvPy -m pip install --upgrade pip setuptools wheel --disable-pip-version-check

Write-Step "Ставлю минимальные зависимости лаунчера"
if (Test-Path (Join-Path $Root "requirements-launcher.txt")) {
    & $VenvPy -m pip install -r requirements-launcher.txt --disable-pip-version-check
} else {
    & $VenvPy -m pip install PyQt5 psutil huggingface-hub python-dateutil --disable-pip-version-check
}

Write-Step "Проверяю лаунчер"
& $VenvPy -c "import PyQt5, psutil, huggingface_hub; print('launcher deps: OK')"

if (-not $NoStart) {
    Write-Step "Запускаю ChatKodi Launcher"
    & $VenvPy launcher.py
} else {
    Write-Host "`nГотово. Запуск: START_LAUNCHER.bat" -ForegroundColor Green
}
