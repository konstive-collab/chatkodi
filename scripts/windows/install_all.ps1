<#
ChatKodi 1.9 — полная базовая установка Windows.
Запуск из корня проекта:
  powershell -ExecutionPolicy Bypass -File scripts\windows\install_all.ps1
#>
param(
    [switch]$SkipModelCheck
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

Write-Host "[ChatKodi] Root: $Root" -ForegroundColor Green

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

$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $VenvPy) {
    # a .venv copied from another computer points to a Python that is not here ("No Python at ...")
    $works = $true
    try { & $VenvPy -c "import sys" 2>$null; $works = ($LASTEXITCODE -eq 0) } catch { $works = $false }
    if (-not $works) {
        Write-Host "[ChatKodi] .venv does not start on this computer - recreating it" -ForegroundColor Yellow
        Remove-Item -Recurse -Force (Join-Path $Root ".venv")
    }
}
if (-not (Test-Path $VenvPy)) {
    $py = Find-Python
    if ($null -eq $py) {
        Write-Host "[ChatKodi] Python не найден. Попробую поставить через winget..." -ForegroundColor Yellow
        winget install -e --id Python.Python.3.11
        $py = Find-Python
        if ($null -eq $py) { throw "Python не найден даже после winget. Установите Python 3.10+ вручную." }
    }
    Write-Host "[ChatKodi] Создаю .venv..." -ForegroundColor Green
    & $py.Exe @($py.Args) -m venv .venv
}

& $VenvPy -m pip install --upgrade pip setuptools wheel --disable-pip-version-check

# Models run in llama.cpp (llama-server, GGUF): no PyTorch to install.
$ReqApp = Join-Path $Root "requirements-app.txt"
if (Test-Path $ReqApp) {
    & $VenvPy -m pip install -r $ReqApp --disable-pip-version-check
} else {
    & $VenvPy -m pip install -r requirements.txt --disable-pip-version-check
}

Write-Host "[ChatKodi] Проверяю модели и окружение..." -ForegroundColor Green
if (-not $SkipModelCheck) {
    & $VenvPy download_models.py --check
}
& $VenvPy -m tools.doctor

Write-Host "[ChatKodi] Полная базовая установка завершена." -ForegroundColor Green
Write-Host "[ChatKodi] Запуск лаунчера: START_LAUNCHER.bat"
