# Python environment for running ChatKodi from the sources (tests, evals).
#
#   powershell -ExecutionPolicy Bypass -File tools\dev_env.ps1 [-Test]
#
# A .venv records the absolute path of the Python it was made from, so a .venv
# copied from another computer (or a full backup of the project) does not
# start there ("No Python at ..."). This script checks .venv in the project
# root, recreates it from the Python of this computer when it is missing or
# broken, installs requirements-dev.txt and prints the interpreter to use.
param([switch]$Test)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"

function Test-Python([string]$exe) {
    if (-not (Test-Path $exe)) { return $false }
    try { & $exe -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null } catch { return $false }
    return ($LASTEXITCODE -eq 0)
}

if (-not (Test-Python $py)) {
    if (Test-Path $venv) {
        Write-Host "[ChatKodi] .venv does not start here (made on another computer?) - recreating it" -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venv
    }
    $base = $null
    foreach ($candidate in @(@("py", "-3.11"), @("py", "-3"), @("python"))) {
        try {
            & $candidate[0] @($candidate | Select-Object -Skip 1) -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) { $base = $candidate; break }
        } catch { }
    }
    if ($null -eq $base) { throw "Python 3.10+ not found. Install Python 3.11 (python.org or: winget install -e --id Python.Python.3.11)." }
    Write-Host "[ChatKodi] Creating .venv with: $($base -join ' ')" -ForegroundColor Green
    & $base[0] @($base | Select-Object -Skip 1) -m venv $venv
}

& $py -m pip install --disable-pip-version-check -q -r (Join-Path $root "requirements-dev.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
Write-Host "[ChatKodi] Ready: $py" -ForegroundColor Green
if ($Test) {
    Push-Location $root
    try { & $py -m pytest -q -p no:cacheprovider } finally { Pop-Location }
    exit $LASTEXITCODE
}
