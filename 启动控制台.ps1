$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Write-Host ""
Write-Host "  Local AI Talking Video" -ForegroundColor Cyan
Write-Host "  Ditto | GPT-SoVITS | 1920x1080 | 25 fps" -ForegroundColor DarkCyan
Write-Host ""

if (-not (Get-Command python.exe -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python was not found." -ForegroundColor Red
    exit 1
}

Write-Host "[1/2] Checking the local runtime and models..." -ForegroundColor Yellow
& python scripts\check_environment.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "The environment is not ready. Run the installer named by the failed check." -ForegroundColor Red
    Write-Host "Ditto: powershell -ExecutionPolicy Bypass -File scripts\install_ditto_runtime.ps1" -ForegroundColor Gray
    Write-Host "MuseTalk: powershell -ExecutionPolicy Bypass -File scripts\download_musetalk_models.ps1" -ForegroundColor Gray
    exit 1
}

Write-Host ""
Write-Host "[2/2] Starting http://127.0.0.1:8080" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Gray
& python scripts\web_server.py
