param(
    [switch]$SkipInstaller,
    [switch]$SkipSmoke,
    [switch]$SkipZip,
    [switch]$MakeZip,
    [switch]$SkipTests,
    [switch]$SkipBenchmark,
    [switch]$SkipFullVerify,
    [switch]$IncrementalBuild,
    [string]$PortableFolderName = "ChatKodi"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

$Version = (Get-Content (Join-Path $Root "VERSION.txt") -Raw).Trim()
if (-not $Version) { $Version = "1.9.0-rc1" }
if ($PortableFolderName -notmatch '^[A-Za-z0-9._-]+$') {
    throw "PortableFolderName contains unsupported characters: $PortableFolderName"
}

$VenvPy = $env:CHATKODI_BUILD_PYTHON
if (-not $VenvPy) { $VenvPy = Join-Path $Root ".venv\Scripts\python.exe" }
if (-not (Test-Path $VenvPy)) {
    $SiblingVenv = Join-Path (Split-Path -Parent $Root) "chatkodi_v1.6\.venv\Scripts\python.exe"
    if (Test-Path $SiblingVenv) { $VenvPy = $SiblingVenv }
}
if (-not (Test-Path $VenvPy)) {
    throw "Build Python not found. Set CHATKODI_BUILD_PYTHON or create .venv."
}

function Invoke-Step {
    param(
        [string]$Title,
        [scriptblock]$Body
    )
    Write-Host ""
    Write-Host "== $Title ==" -ForegroundColor Cyan
    & $Body
}

function Assert-SafeRecursiveTarget {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$AllowedRoot
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $fullRoot = [System.IO.Path]::GetFullPath($AllowedRoot)
    $trimmedPath = $fullPath.TrimEnd('\', '/')
    $trimmedRoot = $fullRoot.TrimEnd('\', '/')
    $rootPrefix = $trimmedRoot + [System.IO.Path]::DirectorySeparatorChar

    if ($trimmedPath.Equals($trimmedRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
        -not $fullPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing recursive delete outside expected root: $fullPath"
    }

    return $fullPath
}

function Require-InnoSetup {
    $cmd = Get-Command "iscc.exe" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $candidates = @(
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    throw "Inno Setup compiler is not installed or not in PATH. Install Inno Setup 6 and rerun this script."
}

function Copy-DirectoryClean {
    param(
        [Parameter(Mandatory=$true)][string]$Source,
        [Parameter(Mandatory=$true)][string]$Destination,
        [switch]$PreferHardLinks
    )
    if (-not (Test-Path $Source)) {
        throw "Required payload source is missing: $Source"
    }
    $safeDestination = Assert-SafeRecursiveTarget -Path $Destination -AllowedRoot $Root
    if (Test-Path $Destination) {
        Remove-Item -LiteralPath $safeDestination -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $safeDestination) | Out-Null

    $sourceRoot = [System.IO.Path]::GetFullPath((Resolve-Path $Source).Path).TrimEnd('\', '/')
    $destinationRoot = [System.IO.Path]::GetFullPath($safeDestination).TrimEnd('\', '/')
    $sameVolume = [System.IO.Path]::GetPathRoot($sourceRoot).Equals(
        [System.IO.Path]::GetPathRoot($destinationRoot),
        [System.StringComparison]::OrdinalIgnoreCase
    )
    if (-not $PreferHardLinks -or -not $sameVolume) {
        Copy-Item -LiteralPath $Source -Destination $safeDestination -Recurse -Force
        return
    }

    New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null
    foreach ($directory in Get-ChildItem -LiteralPath $sourceRoot -Recurse -Directory) {
        $relative = $directory.FullName.Substring($sourceRoot.Length).TrimStart('\', '/')
        New-Item -ItemType Directory -Force -Path (Join-Path $destinationRoot $relative) | Out-Null
    }

    $linked = 0
    $copied = 0
    foreach ($file in Get-ChildItem -LiteralPath $sourceRoot -Recurse -File) {
        $relative = $file.FullName.Substring($sourceRoot.Length).TrimStart('\', '/')
        $target = Join-Path $destinationRoot $relative
        try {
            New-Item -ItemType HardLink -Path $target -Target $file.FullName -ErrorAction Stop | Out-Null
            $linked += 1
        } catch {
            Copy-Item -LiteralPath $file.FullName -Destination $target -Force
            $copied += 1
        }
    }
    Write-Host "Hardlink staging: linked=$linked copied_fallback=$copied source=$sourceRoot"
}

function Reset-PortableRuntimePayload {
    param(
        [Parameter(Mandatory=$true)][string]$AppDir
    )

    $runtime = Join-Path $AppDir "runtime"
    if (Test-Path $runtime) {
        $safeRuntime = Assert-SafeRecursiveTarget -Path $runtime -AllowedRoot $Root
        Remove-Item -LiteralPath $safeRuntime -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $runtime "logs") | Out-Null

    $rootRuntimeDirs = @("logs", ".chatkodi")
    foreach ($name in $rootRuntimeDirs) {
        $path = Join-Path $AppDir $name
        if (Test-Path $path) {
            $safePath = Assert-SafeRecursiveTarget -Path $path -AllowedRoot $Root
            Remove-Item -LiteralPath $safePath -Recurse -Force
        }
    }

    $rootRuntimeFiles = @(
        "chatkodi_settings.json",
        "launcher_settings.json",
        "documents_index.json",
        "embedding_cache.npz",
        "embedding_cache_meta.json",
        "faiss_index.bin",
        "faiss_meta.pkl",
        "llm_analysis_cache.json",
        "synonyms_cache.pkl",
        "events.db",
        "events_meta.json",
        "query_history.jsonl"
    )
    foreach ($name in $rootRuntimeFiles) {
        $path = Join-Path $AppDir $name
        if (Test-Path $path) {
            Remove-Item -LiteralPath $path -Force
        }
    }
}

function Write-JsonNoBom {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)]$Value
    )
    $json = $Value | ConvertTo-Json -Depth 5
    $encoding = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $json, $encoding)
}

function Sign-IfConfigured {
    param([string[]]$Paths)

    $signtool = $env:CHATKODI_SIGNTOOL
    if (-not $signtool) {
        $cmd = Get-Command "signtool.exe" -ErrorAction SilentlyContinue
        if ($cmd) { $signtool = $cmd.Source }
    }

    $cert = $env:CHATKODI_SIGN_CERT
    $password = $env:CHATKODI_SIGN_PASSWORD
    $timestamp = $env:CHATKODI_SIGN_TIMESTAMP
    if (-not $timestamp) { $timestamp = "http://timestamp.digicert.com" }

    if (-not $signtool -or -not $cert -or -not (Test-Path $cert)) {
        Write-Host "signing skipped: CHATKODI_SIGNTOOL/CHATKODI_SIGN_CERT not configured" -ForegroundColor Yellow
        return
    }

    foreach ($path in $Paths) {
        if (-not (Test-Path $path)) { continue }
        $args = @("sign", "/fd", "SHA256", "/tr", $timestamp, "/td", "SHA256", "/f", $cert)
        if ($password) { $args += @("/p", $password) }
        $args += $path
        & $signtool @args
        if ($LASTEXITCODE -ne 0) { throw "signtool failed for $path" }
    }
}

$DistRoot = Join-Path $Root "dist"
$DistDir = Join-Path $DistRoot "ChatKodi"
$ReleaseDir = Join-Path $Root "release"
$ReleasePortableDir = Join-Path $ReleaseDir $PortableFolderName
$BuildDir = Join-Path $Root "build"
$IsccPath = $null

if (-not $SkipInstaller) {
    Invoke-Step "Check installer toolchain" {
        $script:IsccPath = Require-InnoSetup
        Write-Host "Inno Setup compiler: $script:IsccPath"
    }
}

Invoke-Step "Clean previous build output" {
    $cleanTargets = @($DistDir, $ReleasePortableDir)
    if (-not $IncrementalBuild) { $cleanTargets += $BuildDir }
    foreach ($path in $cleanTargets) {
        $safePath = Assert-SafeRecursiveTarget -Path $path -AllowedRoot $Root
        if (Test-Path $safePath) { Remove-Item -LiteralPath $safePath -Recurse -Force }
    }
    New-Item -ItemType Directory -Force -Path $DistRoot, $ReleaseDir | Out-Null
}

Invoke-Step "Install build/runtime requirements" {
    & $VenvPy -m pip install -r requirements-app.txt -r requirements-build.txt --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
    & $VenvPy -c "import numpy, pymorphy3, PyQt5.QtCore as q; print('toolchain:', numpy.__version__, pymorphy3.__version__, q.QT_VERSION_STR)"
    if ($LASTEXITCODE -ne 0) { throw "Runtime dependency import check failed" }
}

# GGUF models and llama.cpp live outside the repository (default: ..\models).
$ModelSourceRoot = $env:CHATKODI_MODEL_SOURCE_ROOT
if (-not $ModelSourceRoot) { $ModelSourceRoot = Join-Path (Split-Path -Parent $Root) "models" }
$LlmFile = if ($env:CHATKODI_BUILD_LLM) { $env:CHATKODI_BUILD_LLM } else { "Qwen3.5-4B-Q4_K_M.gguf" }
$EmbeddingFile = if ($env:CHATKODI_BUILD_EMBEDDING) { $env:CHATKODI_BUILD_EMBEDDING } else { "Qwen3-Embedding-0.6B-Q8_0.gguf" }
$LlmSource = Join-Path $ModelSourceRoot "llms\$LlmFile"
$EmbeddingSource = Join-Path $ModelSourceRoot "embeddings\$EmbeddingFile"
$LlamaSource = Get-ChildItem -LiteralPath (Join-Path $ModelSourceRoot "llama.cpp") -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName "llama-server.exe") } |
    Sort-Object Name -Descending | Select-Object -First 1
foreach ($required in @($LlmSource, $EmbeddingSource)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Model not found: $required (set CHATKODI_MODEL_SOURCE_ROOT)" }
}
if (-not $LlamaSource) { throw "llama.cpp build with llama-server.exe not found under $ModelSourceRoot\llama.cpp" }
Write-Host "Models: $LlmSource ; $EmbeddingSource ; llama.cpp: $($LlamaSource.FullName)"

Invoke-Step "Source verification" {
    $previousExtraCaches = $env:CHATKODI_EXTRA_MODEL_CACHE_DIRS
    $previousModelPath = $env:CHATKODI_MODEL_PATH
    $env:CHATKODI_EXTRA_MODEL_CACHE_DIRS = $ModelSourceRoot
    $env:CHATKODI_MODEL_PATH = $LlmSource
    try {
        & $VenvPy scripts\dev\compile_check.py
        if ($LASTEXITCODE -ne 0) { throw "compile_check failed" }
        & $VenvPy -m tools.doctor --json
        if ($LASTEXITCODE -ne 0) { throw "doctor failed" }
        if (-not $SkipTests) {
            & $VenvPy -m pytest -q
            if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
        }
        if (-not $SkipBenchmark) {
            & $VenvPy main.py --search-benchmark search_accuracy_cases.json --docs-folder (Join-Path $Root "docs")
            if ($LASTEXITCODE -ne 0) { throw "source search benchmark failed" }
        }
    } finally {
        $env:CHATKODI_EXTRA_MODEL_CACHE_DIRS = $previousExtraCaches
        $env:CHATKODI_MODEL_PATH = $previousModelPath
    }
}

Invoke-Step "PyInstaller build" {
    $pyinstallerArgs = @("-m", "PyInstaller", "--noconfirm")
    if (-not $IncrementalBuild) { $pyinstallerArgs += "--clean" }
    $pyinstallerArgs += "packaging\chatkodi_production.spec"
    & $VenvPy @pyinstallerArgs
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
    if (-not (Test-Path (Join-Path $DistDir "ChatKodiLauncher.exe"))) { throw "ChatKodiLauncher.exe was not produced" }
    if (-not (Test-Path (Join-Path $DistDir "ChatKodiApp.exe"))) { throw "ChatKodiApp.exe was not produced" }
}

Invoke-Step "Prune build-only runtime artifacts" {
    $InternalRoot = Join-Path $DistDir "_internal"
    if (-not (Test-Path $InternalRoot)) { $InternalRoot = $DistDir }
    $before = (Get-ChildItem -LiteralPath $DistDir -Recurse -File | Measure-Object Length -Sum).Sum

    foreach ($relative in @(
        "PyQt5\Qt5\qml",
        "PyQt5\bindings"
    )) {
        $path = Join-Path $InternalRoot $relative
        if (Test-Path $path) {
            $safePath = Assert-SafeRecursiveTarget -Path $path -AllowedRoot $DistDir
            Remove-Item -LiteralPath $safePath -Recurse -Force
        }
    }
    foreach ($forbidden in @("torch", "transformers", "sentence_transformers")) {
        if (Test-Path (Join-Path $InternalRoot $forbidden)) { throw "$forbidden leaked into the bundle" }
    }
    $after = (Get-ChildItem -LiteralPath $DistDir -Recurse -File | Measure-Object Length -Sum).Sum
    Write-Host ("Pruned {0:N1} MB of build-only files" -f (($before - $after) / 1MB))
}

Invoke-Step "Copy production app resources" {
    Copy-Item -LiteralPath (Join-Path $Root "chatkodi_config.json") -Destination (Join-Path $DistDir "chatkodi_config.json") -Force
    Copy-Item -LiteralPath (Join-Path $Root "VERSION.txt") -Destination (Join-Path $DistDir "VERSION.txt") -Force
    Copy-Item -LiteralPath (Join-Path $Root "search_accuracy_cases.json") -Destination (Join-Path $DistDir "search_accuracy_cases.json") -Force
    Copy-DirectoryClean -Source (Join-Path $Root "manuals") -Destination (Join-Path $DistDir "manuals")
}

Invoke-Step "Add GGUF models and llama.cpp" {
    $PayloadCache = Join-Path $DistDir "models_cache"
    New-Item -ItemType Directory -Force -Path (Join-Path $PayloadCache "llms"), (Join-Path $PayloadCache "embeddings") | Out-Null
    foreach ($pair in @(@($LlmSource, "llms"), @($EmbeddingSource, "embeddings"))) {
        $target = Join-Path (Join-Path $PayloadCache $pair[1]) (Split-Path -Leaf $pair[0])
        # Hard link when possible: a multi-GB copy only on another volume.
        try {
            New-Item -ItemType HardLink -Path $target -Target $pair[0] -ErrorAction Stop | Out-Null
        } catch {
            Copy-Item -LiteralPath $pair[0] -Destination $target -Force
        }
        Write-Host ("Model: {0} ({1:N0} MB)" -f $target, ((Get-Item -LiteralPath $target).Length / 1MB))
    }
    # Vision projector of the LLM (photos and scans): mmproj-<model>*.gguf next to it.
    $LlmBase = ($LlmFile -replace '-(?:UD-)?(?:I?Q\d\w*|BF16|F16|F32)\.gguf$', '')
    $Mmproj = Get-ChildItem -LiteralPath (Split-Path -Parent $LlmSource) -Filter "mmproj*.gguf" -File |
        Where-Object { $_.Name -like "*$LlmBase*" } | Select-Object -First 1
    if (-not $Mmproj) { throw "Vision projector mmproj-$LlmBase*.gguf not found next to $LlmSource" }
    $target = Join-Path (Join-Path $PayloadCache "llms") $Mmproj.Name
    try { New-Item -ItemType HardLink -Path $target -Target $Mmproj.FullName -ErrorAction Stop | Out-Null }
    catch { Copy-Item -LiteralPath $Mmproj.FullName -Destination $target -Force }
    Write-Host ("Vision: {0} ({1:N0} MB)" -f $target, ($Mmproj.Length / 1MB))
    # Speech: whisper.cpp server + model, Piper + voices (CUDA DLLs come from llama.cpp).
    $SpeechSource = Join-Path $ModelSourceRoot "speech"
    if (-not (Test-Path (Join-Path $SpeechSource "whisper\whisper-server.exe"))) { throw "whisper-server.exe not found under $SpeechSource\whisper" }
    Copy-DirectoryClean -Source $SpeechSource -Destination (Join-Path $PayloadCache "speech") -PreferHardLinks
    $speechMb = (Get-ChildItem -LiteralPath (Join-Path $PayloadCache "speech") -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
    Write-Host ("Speech: {0:N0} MB" -f $speechMb)
    # llama-server, every shared DLL (ggml, CUDA, llama, llama-common, mtmd) and
    # licenses; other llama.cpp tools and their *-impl.dll stay out.
    $LlamaTarget = Join-Path $DistDir ("llama.cpp\" + $LlamaSource.Name)
    New-Item -ItemType Directory -Force -Path $LlamaTarget | Out-Null
    Get-ChildItem -LiteralPath $LlamaSource.FullName -File |
        Where-Object {
            $_.Name -eq "llama-server.exe" -or
            $_.Name -like "LICENSE*" -or
            ($_.Extension -eq ".dll" -and $_.Name -notmatch '^llama-(?!server-).*-impl\.dll$')
        } |
        ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $LlamaTarget -Force }
    $llamaMb = (Get-ChildItem -LiteralPath $LlamaTarget -File | Measure-Object Length -Sum).Sum / 1MB
    Write-Host ("llama.cpp: {0} ({1:N0} MB)" -f $LlamaTarget, $llamaMb)
    & (Join-Path $LlamaTarget "llama-server.exe") --version
    if ($LASTEXITCODE -ne 0) { throw "bundled llama-server does not start" }
}

Invoke-Step "Prepare portable folder layout" {
    New-Item -ItemType Directory -Force -Path `
        (Join-Path $DistDir "docs"), `
        (Join-Path $DistDir "runtime"), `
        (Join-Path $DistDir "runtime\logs") | Out-Null
    Set-Content -Encoding ASCII -Path (Join-Path $DistDir "portable.flag") -Value "ChatKodi portable distribution"
    @'
@echo off
set CHATKODI_PORTABLE=1
set "CHATKODI_APP_ROOT=%~dp0"
set "CHATKODI_RUNTIME_DIR=%~dp0runtime"
start "" "%~dp0ChatKodiLauncher.exe"
'@ | Set-Content -Encoding ASCII (Join-Path $DistDir "START_CHATKODI.cmd")
    @'
@echo off
set CHATKODI_PORTABLE=1
set "CHATKODI_APP_ROOT=%~dp0"
set "CHATKODI_RUNTIME_DIR=%~dp0runtime"
"%~dp0ChatKodiApp.exe" --verify-portable --full
echo.
if errorlevel 1 (
  echo ChatKodi portable verification FAILED.
) else (
  echo ChatKodi portable verification passed.
)
echo Result and warnings are appended to runtime\logs\portable_preflight.log
pause
'@ | Set-Content -Encoding ASCII (Join-Path $DistDir "VERIFY_CHATKODI.cmd")
    @'
ChatKodi portable build

1. Start ChatKodi with START_CHATKODI.cmd or ChatKodiLauncher.exe.
2. Put user documents into the docs folder or choose another folder in the app.
3. Runtime settings, logs and indexes are stored in the runtime folder.
4. Models are bundled in models_cache and work offline.
5. After copying the folder to another disk or PC, run VERIFY_CHATKODI.cmd once.
'@ | Set-Content -Encoding UTF8 (Join-Path $DistDir "README_PORTABLE.txt")
}

Invoke-Step "Packaged diagnostics" {
    $env:CHATKODI_PRODUCTION = "1"
    $env:CHATKODI_PORTABLE = "1"
    $env:CHATKODI_APP_ROOT = $DistDir
    $env:CHATKODI_RUNTIME_DIR = (Join-Path $DistDir "runtime")
    & (Join-Path $DistDir "ChatKodiApp.exe") --doctor-json
    if ($LASTEXITCODE -ne 0) { throw "packaged doctor failed" }
    & (Join-Path $DistDir "ChatKodiApp.exe") --docs-diagnose --docs-folder (Join-Path $Root "docs")
    if ($LASTEXITCODE -ne 0) { throw "packaged document ingestion failed" }
    if (-not $SkipBenchmark) {
        & (Join-Path $DistDir "ChatKodiApp.exe") --search-benchmark (Join-Path $Root "search_accuracy_cases.json") --docs-folder (Join-Path $Root "docs")
        if ($LASTEXITCODE -ne 0) { throw "packaged search benchmark failed" }
    }
    if (-not $SkipSmoke) {
        & (Join-Path $DistDir "ChatKodiApp.exe") --llm-smoke
        if ($LASTEXITCODE -ne 0) { throw "packaged LLM smoke failed" }
    }
}

Invoke-Step "Clean portable runtime state" {
    Reset-PortableRuntimePayload -AppDir $DistDir
    Write-Host "Runtime folder reset: $(Join-Path $DistDir "runtime")"
}

Invoke-Step "Optional code signing for exe files" {
    Sign-IfConfigured -Paths @(
        (Join-Path $DistDir "ChatKodiLauncher.exe"),
        (Join-Path $DistDir "ChatKodiApp.exe")
    )
}

Invoke-Step "Create and verify portable manifest" {
    & $VenvPy -m tools.portable_verify --write-manifest $DistDir
    if ($LASTEXITCODE -ne 0) { throw "portable manifest creation failed" }
    & $VenvPy -m tools.portable_verify --verify $DistDir
    if ($LASTEXITCODE -ne 0) { throw "portable quick verification failed" }
    & (Join-Path $DistDir "ChatKodiLauncher.exe") --self-test-json
    if ($LASTEXITCODE -ne 0) { throw "packaged launcher self-test failed" }
    if (-not $SkipFullVerify) {
        & $VenvPy -m tools.portable_verify --verify $DistDir --full
        if ($LASTEXITCODE -ne 0) { throw "portable full verification failed" }
    }
}

Invoke-Step "Stage portable release folder" {
    Copy-DirectoryClean -Source $DistDir -Destination $ReleasePortableDir -PreferHardLinks
    # The manifest must describe exactly what the user receives: regenerate it
    # from the final staged folder (last write step of the build).
    & $VenvPy -m tools.portable_verify --write-manifest $ReleasePortableDir
    if ($LASTEXITCODE -ne 0) { throw "staged portable manifest creation failed" }
    & $VenvPy -m tools.portable_verify --verify $ReleasePortableDir
    if ($LASTEXITCODE -ne 0) { throw "staged portable quick verification failed" }
    & (Join-Path $ReleasePortableDir "ChatKodiLauncher.exe") --self-test-json
    if ($LASTEXITCODE -ne 0) { throw "staged launcher self-test failed" }
    if (-not $SkipFullVerify) {
        & $VenvPy -m tools.portable_verify --verify $ReleasePortableDir --full
        if ($LASTEXITCODE -ne 0) { throw "staged portable full verification failed" }
    }
    $releaseDocs = Join-Path $ReleasePortableDir "docs"
    $releaseDocFiles = @(Get-ChildItem -LiteralPath $releaseDocs -Recurse -File -ErrorAction SilentlyContinue)
    if ($releaseDocFiles.Count -ne 0) { throw "Release docs folder is not empty" }
    Write-Host "Portable folder: $ReleasePortableDir"
}

if (-not $SkipZip) {
    Invoke-Step "Portable zip package" {
        $ZipPath = Join-Path $ReleaseDir "ChatKodi-$Version-portable.zip"
        if (Test-Path $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }
        # Python zipfile: UTF-8 names, forward slashes, and a post-pack check
        # that every manifest entry is inside the archive with the right size.
        & $VenvPy -m tools.portable_verify --make-zip $ReleasePortableDir $ZipPath
        if ($LASTEXITCODE -ne 0) { throw "Portable zip does not match the manifest" }
        if (-not (Test-Path $ZipPath)) { throw "Portable zip was not produced" }
        Write-Host "Portable zip: $ZipPath"
    }
}

if (-not $SkipInstaller) {
    Invoke-Step "Inno Setup installer" {
        $Iscc = $script:IsccPath
        if (-not $Iscc) { $Iscc = Require-InnoSetup }
        $env:CHATKODI_BUILD_VERSION = $Version
        $env:CHATKODI_DIST_DIR = $DistDir
        & $Iscc (Join-Path $Root "installer\ChatKodi.iss")
        if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
    }

    Invoke-Step "Optional code signing for installer" {
        $Setup = Join-Path $ReleaseDir "ChatKodi-$Version-setup.exe"
        Sign-IfConfigured -Paths @($Setup)
    }
}

Write-Host ""
Write-Host "Release build finished." -ForegroundColor Green
Write-Host "Dist: $DistDir"
Write-Host "Portable folder: $ReleasePortableDir"
if ($SkipInstaller) {
    Write-Host "Installer: skipped"
} else {
    Write-Host "Installer: $(Join-Path $ReleaseDir "ChatKodi-$Version-setup.exe")"
}
if (-not $SkipZip) {
    Write-Host "Portable zip: $(Join-Path $ReleaseDir "ChatKodi-$Version-portable.zip")"
} else {
    Write-Host "Portable zip: skipped"
}
