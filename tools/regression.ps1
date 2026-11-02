# Full quality regression of ChatKodi: unit tests, the 13-case benchmark, the
# LLM / diverse / report-series sets, the assistant scenarios and (optionally)
# a compact window check (2 scenarios of each kind). Corpora are (re)built under -Work.
#
#   powershell -ExecutionPolicy Bypass -File tools\regression.ps1 -Work D:\eval -Tag run1 [-SkipGui]
#
# Needs CHATKODI_EXTRA_MODEL_CACHE_DIRS (or models next to the app) and the venv python in -Python.
param(
    [string]$Work = (Join-Path $PSScriptRoot "..\runtime\eval"),
    [string]$Tag = "run",
    [string]$Python = "",
    [switch]$SkipGui
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root
New-Item -ItemType Directory -Force $Work | Out-Null
$Work = (Resolve-Path $Work).Path
$env:PYTHONUTF8 = "1"
if (-not $Python) {
    # the project's own .venv (tools\dev_env.ps1 makes it), else the shared one next to the project
    foreach ($candidate in @("$root\.venv\Scripts\python.exe", "$root\..\chatkodi_v1.6\.venv\Scripts\python.exe")) {
        if (Test-Path $candidate) {
            try { & $candidate -c "import sys" 2>$null } catch { continue }
            if ($LASTEXITCODE -eq 0) { $Python = $candidate; break }
        }
    }
    if (-not $Python) { throw "No working Python environment: run tools\dev_env.ps1 first." }
}
$py = $Python

function Build-Corpora {
    if (-not (Test-Path "$Work\synthetic\scenarios.json")) {
        & $py -m tools.synthetic_corpus --out "$Work\synthetic" | Out-Null
        & $py -m tools.scenario_bank --out "$Work\synthetic\scenarios.json" | Out-Null
    }
    if (-not (Test-Path "$Work\llm_eval\cases.json")) { & $py tools\llm_eval_corpus.py "$Work\synthetic" "$Work\llm_eval" | Out-Null }
    if (-not (Test-Path "$Work\diverse\cases.json")) { & $py -m tools.diverse_corpus --out "$Work\diverse" | Out-Null }
    if (-not (Test-Path "$Work\series\cases.json")) {
        Copy-Item docs "$Work\series\docs" -Recurse
        Copy-Item tools\report_series_cases.json "$Work\series\cases.json"
    }
    & $py -m tools.agent_corpus --out "$Work\agent" | Out-Null
}
Build-Corpora

$env:CHATKODI_RUNTIME_DIR = "$Work\rt_${Tag}_pytest"
$pt = cmd /c "`"$py`" -m pytest -q -p no:cacheprovider 2>&1" | Select-String -Pattern 'passed|failed' | Select-Object -Last 1
"pytest: $pt"
$env:CHATKODI_RUNTIME_DIR = "$Work\rt_${Tag}_bench"
$bench = cmd /c "`"$py`" main.py --search-benchmark search_accuracy_cases.json --docs-folder docs 2>&1" | Select-String '"cases_(passed|failed)"' | ForEach-Object { $_.Line.Trim() }
"benchmark 13: $($bench -join ' ')"

foreach ($set in @(@('llm_eval', 'LLM 22'), @('diverse', 'DIVERSE 60'), @('series', 'SERIES 44'))) {
    $env:CHATKODI_RUNTIME_DIR = "$Work\rt_${Tag}_$($set[0])"
    cmd /c "`"$py`" tools\eval_synthetic.py --corpus `"$Work\$($set[0])`" --with-llm --report `"$Work\$($set[0])_$Tag.json`" > `"$Work\$($set[0])_$Tag.txt`" 2>&1"
    $j = Get-Content "$Work\$($set[0])_$Tag.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    $ok = @($j.rows | Where-Object ok).Count
    "{0}: {1}/{2}" -f $set[1], $ok, $j.rows.Count
    $j.rows | Where-Object { -not $_.ok } | ForEach-Object { "   BAD [$($_.category)] $($_.question)" }
}

$env:CHATKODI_RUNTIME_DIR = "$Work\rt_${Tag}_agent"
cmd /c "`"$py`" tools\eval_agent.py --corpus `"$Work\agent`" --report `"$Work\agent_$Tag.json`" > `"$Work\agent_$Tag.txt`" 2>&1"
$j = Get-Content "$Work\agent_$Tag.json" -Raw -Encoding UTF8 | ConvertFrom-Json
"ASSISTANT: {0}/{1}, median {2}s, max {3}s" -f $j.passed, $j.total, $j.median_s, $j.max_s
$j.rows | Where-Object { -not $_.ok } | ForEach-Object { "   BAD [$($_.action)] $($_.message) $($_.attach -join ',')" }

if (-not $SkipGui) {
    Remove-Item Env:CHATKODI_RUNTIME_DIR -ErrorAction SilentlyContinue
    cmd /c "`"$py`" -m tools.gui_e2e --docs `"$Work\synthetic\docs`" --questions `"$Work\synthetic\scenarios.json`" --per-category 2 --out `"$Work\gui_$Tag`" > `"$Work\gui_$Tag.log`" 2>&1"
    Get-Content "$Work\gui_$Tag.log" -Encoding UTF8 | Where-Object { $_ -match '"summary"' } | ForEach-Object {
        $s = ($_ | ConvertFrom-Json).summary
        "GUI (2 вопроса каждого вида): {0}/{1}, median {2}s, failed: {3}" -f $s.passed, $s.questions, $s.median_seconds, ($s.failed_ids -join ',')
    }
}
