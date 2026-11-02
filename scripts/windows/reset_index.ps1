$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
$Runtime = Join-Path $Root ".chatkodi"
$targets = @("events.db", "events_meta.json", "documents_index.json", "embedding_cache.npz", "embedding_cache_meta.json", "faiss_index.bin", "faiss_meta.pkl")
foreach ($name in $targets) {
    $path = Join-Path $Runtime $name
    if (Test-Path $path) {
        Remove-Item $path -Force
        Write-Host "removed $path"
    }
}
Write-Host "[ChatKodi] Индексы сброшены. models_cache не трогал."
