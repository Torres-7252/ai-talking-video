$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Write-Host ""
Write-Host "  Local AI Talking Video" -ForegroundColor Cyan
Write-Host "  Ditto | GPT-SoVITS | 1920x1080 | 25 fps" -ForegroundColor DarkCyan
Write-Host ""

$CandidatePythons = @(
    (Join-Path $ProjectRoot ".venv-liveportrait\Scripts\python.exe"),
    (Join-Path $ProjectRoot ".venv-web\Scripts\python.exe")
)

$CandidatePythons += @(
    (Join-Path $ProjectRoot ".venv-mimicmotion\Scripts\python.exe")
)

$PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
if ($PyLauncher) {
    foreach ($PythonLine in (& $PyLauncher.Source -0p 2>$null)) {
        if ($PythonLine -match '([A-Za-z]:\\.*python.exe)\s*$') {
            $CandidatePythons += $Matches[1]
        }
    }
}
$SystemPython = Get-Command python.exe -ErrorAction SilentlyContinue
if ($SystemPython) {
    $CandidatePythons += $SystemPython.Source
}

$WebPython = $null
foreach ($Candidate in $CandidatePythons | Select-Object -Unique) {
    if (-not (Test-Path -LiteralPath $Candidate)) {
        continue
    }
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & $Candidate -c "import fastapi, uvicorn, multipart, PIL, yaml, cv2, numpy, torch, soundfile, pytorch_lightning, transformers, funasr" *> $null
    $CandidateExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference
    if ($CandidateExitCode -eq 0) {
        $WebPython = $Candidate
        break
    }
}

if (-not $WebPython) {
    Write-Host "[ERROR] No Python runtime has the complete app dependencies." -ForegroundColor Red
    Write-Host "Run: python -m pip install -r requirements.txt" -ForegroundColor Gray
    exit 1
}

Write-Host "  Runtime: $WebPython" -ForegroundColor DarkGray

Write-Host "[1/2] Checking the local runtime and models..." -ForegroundColor Yellow
& $WebPython scripts\check_environment.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Some generation components are not ready." -ForegroundColor Yellow
    Write-Host "The console will still start so assets can be managed." -ForegroundColor Yellow
    Write-Host "Ditto: powershell -ExecutionPolicy Bypass -File scripts\install_ditto_runtime.ps1" -ForegroundColor Gray
    Write-Host "MuseTalk: powershell -ExecutionPolicy Bypass -File scripts\download_musetalk_models.ps1" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[2/2] Starting http://127.0.0.1:8080" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Gray
& $WebPython scripts\web_server.py
