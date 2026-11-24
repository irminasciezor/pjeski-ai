# nocny run: 18/50/mobile -> pets -> jedzenie, potem spanie
# powershell -ExecutionPolicy Bypass -File .\run_nocny.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Brak venv: $py"
    exit 1
}

$skrypty = @(
    "run.py",
    "douczanie_pets.py",
    "douczanie_jedzenie.py"
)

Write-Host "=== START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "Kolejnosc: 18/50/mobile (aug, do 2h) -> pets 2h -> jedzenie 2h"
foreach ($s in $skrypty) {
    Write-Host ""
    Write-Host "=== reset GPU przed $s ==="
    & $py -c "from gpu_utils import przygotuj_gpu; przygotuj_gpu()"
    Write-Host "=== LECI: $s  $(Get-Date -Format 'HH:mm:ss') ==="
    & $py $s
    if ($LASTEXITCODE -ne 0) {
        Write-Host "BLAD w $s (kod $LASTEXITCODE) - retry za 10s..."
        Start-Sleep -Seconds 10
        & $py -c "from gpu_utils import przygotuj_gpu; przygotuj_gpu()"
        & $py $s
        if ($LASTEXITCODE -ne 0) {
            Write-Host "BLAD w $s po retry (kod $LASTEXITCODE) - przerywam kolejke"
            exit $LASTEXITCODE
        }
    }
    Write-Host "=== OK: $s ==="
}

Write-Host ""
Write-Host "=== KONIEC WSZYSTKIEGO $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "Uspiam komputer za 30s..."
Start-Sleep -Seconds 30
rundll32.exe powrprof.dll,SetSuspendState 0,1,0
