param(
    [ValidateRange(1, 16)]
    [int]$MaxWorkers = 2,
    [ValidateRange(1, 16)]
    [int]$XetConcurrency = 8
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$MuseTalkRoot = Join-Path $ProjectRoot "app\backend\providers\lipsync\MuseTalk"
$ModelsRoot = Join-Path $MuseTalkRoot "models"
$Hf = (Get-Command hf.exe -ErrorAction Stop).Source
$Curl = (Get-Command curl.exe -ErrorAction Stop).Source

# Eight Xet streams fit comfortably in this machine's 16 GB RAM while avoiding
# the very slow single-stream LFS bridge.
$env:HF_XET_FIXED_DOWNLOAD_CONCURRENCY = "$XetConcurrency"
$env:HF_HUB_ETAG_TIMEOUT = "30"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "300"

function Invoke-HfDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string[]]$Files
    )

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Write-Host "Downloading $Repository -> $Destination"
    & $Hf download $Repository @Files --local-dir $Destination --max-workers $MaxWorkers
    if ($LASTEXITCODE -ne 0) {
        throw "Hugging Face download failed for $Repository (exit code $LASTEXITCODE)"
    }
}

if (-not (Test-Path $MuseTalkRoot -PathType Container)) {
    throw "MuseTalk source directory was not found: $MuseTalkRoot"
}

Invoke-HfDownload `
    -Repository "TMElyralab/MuseTalk" `
    -Destination $ModelsRoot `
    -Files @("musetalkV15/musetalk.json", "musetalkV15/unet.pth")

Invoke-HfDownload `
    -Repository "stabilityai/sd-vae-ft-mse" `
    -Destination (Join-Path $ModelsRoot "sd-vae") `
    -Files @("config.json", "diffusion_pytorch_model.bin")

Invoke-HfDownload `
    -Repository "openai/whisper-tiny" `
    -Destination (Join-Path $ModelsRoot "whisper") `
    -Files @("config.json", "preprocessor_config.json", "pytorch_model.bin")

Invoke-HfDownload `
    -Repository "yzd-v/DWPose" `
    -Destination (Join-Path $ModelsRoot "dwpose") `
    -Files @("dw-ll_ucoco_384.pth")

Invoke-HfDownload `
    -Repository "ManyOtherFunctions/face-parse-bisent" `
    -Destination (Join-Path $ModelsRoot "face-parse-bisent") `
    -Files @("79999_iter.pth", "resnet18-5c106cde.pth")

$S3fdPath = Join-Path $MuseTalkRoot "musetalk\utils\face_detection\detection\sfd\s3fd.pth"
if (-not (Test-Path $S3fdPath -PathType Leaf)) {
    Write-Host "Downloading S3FD face detector -> $S3fdPath"
    & $Curl `
        --fail `
        --location `
        --retry 5 `
        --retry-all-errors `
        --output $S3fdPath `
        "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
    if ($LASTEXITCODE -ne 0) {
        throw "S3FD download failed (exit code $LASTEXITCODE)"
    }
}

Push-Location $ProjectRoot
try {
    & python -c "from app.backend.providers.lipsync import missing_musetalk_files; missing = missing_musetalk_files(); print('MuseTalk model check: OK' if not missing else '\n'.join(map(str, missing))); raise SystemExit(bool(missing))"
    if ($LASTEXITCODE -ne 0) {
        throw "MuseTalk model verification failed"
    }

    $Incomplete = Get-ChildItem -Path $ModelsRoot -Recurse -File | Where-Object {
        $_.Name.EndsWith(".incomplete") -or $_.Name.EndsWith(".part")
    }
    if ($Incomplete) {
        $Incomplete | ForEach-Object { Write-Host "Incomplete file: $($_.FullName)" }
        throw "Incomplete model downloads remain under $ModelsRoot"
    }
}
finally {
    Pop-Location
}
