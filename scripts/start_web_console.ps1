$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-liveportrait\Scripts\python.exe"
$Url = "http://127.0.0.1:8080"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python runtime was not found: $Python"
}

$listener = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $listener) {
    Start-Process -FilePath $Python -ArgumentList "scripts\web_server.py" `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden
}

for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 $Url
        if ($response.StatusCode -eq 200) {
            Start-Process $Url
            exit 0
        }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

throw "AI Talking Video Factory did not start within 30 seconds."
