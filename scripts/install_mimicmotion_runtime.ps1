[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Runtime = Join-Path $ProjectRoot 'app\backend\providers\motion\MimicMotion'
$Venv = Join-Path $ProjectRoot '.venv-mimicmotion'
$Python = Join-Path $Venv 'Scripts\python.exe'
$Commit = '6907bdcc259a6a048d41a365e840d22274f9256c'
$Repository = 'https://github.com/Tencent/MimicMotion.git'

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
    }
}

Write-Host 'MimicMotion isolated runtime installer'
Write-Host "Project: $ProjectRoot"
Write-Host "Pinned commit: $Commit"

if (-not $WhatIfPreference) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw 'Git is required but was not found in PATH.'
    }
    try {
        Invoke-External -FilePath py -ArgumentList @('-3.10', '-c', 'import sys; print(sys.executable)')
    }
    catch {
        throw 'Python 3.10 is required. Install it with: winget install Python.Python.3.10'
    }
}

if (-not (Test-Path $Python) -and $PSCmdlet.ShouldProcess($Venv, 'Create Python 3.10 virtual environment')) {
    Invoke-External -FilePath py -ArgumentList @('-3.10', '-m', 'venv', $Venv)
}

if (-not (Test-Path (Join-Path $Runtime '.git')) -and $PSCmdlet.ShouldProcess($Runtime, 'Clone MimicMotion')) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Runtime) | Out-Null
    Invoke-External -FilePath git -ArgumentList @('clone', $Repository, $Runtime)
}

if (Test-Path (Join-Path $Runtime '.git')) {
    $Dirty = (& git -C $Runtime status --porcelain)
    if ($Dirty) {
        throw "MimicMotion checkout has local changes: $Runtime"
    }
    $CurrentCommit = (& git -C $Runtime rev-parse HEAD).Trim()
    if ($CurrentCommit -ne $Commit -and $PSCmdlet.ShouldProcess($Runtime, "Checkout pinned commit $Commit")) {
        Invoke-External -FilePath git -ArgumentList @('-C', $Runtime, 'fetch', 'origin', $Commit)
        Invoke-External -FilePath git -ArgumentList @('-C', $Runtime, 'checkout', '--detach', $Commit)
    }
}

$DependenciesReady = $false
if ((Test-Path $Python) -and -not $WhatIfPreference) {
    & $Python -c "import torch, diffusers, decord; assert torch.__version__.startswith('2.3.1+cu121'); assert torch.cuda.is_available()" 2>$null
    $DependenciesReady = $LASTEXITCODE -eq 0
}
if (-not $DependenciesReady -and $PSCmdlet.ShouldProcess($Venv, 'Install pinned MimicMotion dependencies')) {
    Invoke-External -FilePath $Python -ArgumentList @('-m', 'pip', 'install', '--upgrade', 'pip')
    Invoke-External -FilePath $Python -ArgumentList @('-m', 'pip', 'install', 'torch==2.3.1', 'torchvision==0.18.1', 'torchaudio==2.3.1', '--index-url', 'https://download.pytorch.org/whl/cu121')
    Invoke-External -FilePath $Python -ArgumentList @('-m', 'pip', 'install', 'diffusers==0.27.0', 'transformers==4.32.1', 'huggingface_hub==0.24.7', 'decord==0.6.0', 'einops==0.8.1', 'omegaconf==2.3.0', 'onnxruntime-gpu==1.18.1', 'opencv-python==4.10.0.84', 'matplotlib==3.9.2', 'tqdm==4.66.5', 'av==12.3.0', 'accelerate==0.33.0')
}

$Models = Join-Path $Runtime 'models'
$Checkpoint = Join-Path $Models 'MimicMotion_1-1.pth'
$Detector = Join-Path $Models 'DWPose\yolox_l.onnx'
$Pose = Join-Path $Models 'DWPose\dw-ll_ucoco_384.onnx'
$BaseModel = Join-Path $Models 'stable-video-diffusion-img2vid-xt-1-1\model_index.json'
if ((-not (Test-Path $Checkpoint) -or -not (Test-Path $Detector) -or -not (Test-Path $Pose) -or -not (Test-Path $BaseModel)) -and $PSCmdlet.ShouldProcess($Models, 'Download MimicMotion, DWPose, and Stable Video Diffusion weights')) {
    New-Item -ItemType Directory -Force -Path $Models | Out-Null
    $HfCli = Join-Path $Venv 'Scripts\huggingface-cli.exe'
    if (-not (Test-Path $HfCli)) {
        throw "Hugging Face CLI was not installed in $Venv"
    }
    Invoke-External -FilePath $HfCli -ArgumentList @('download', 'tencent/MimicMotion', 'MimicMotion_1-1.pth', '--local-dir', $Models)
    Invoke-External -FilePath $HfCli -ArgumentList @('download', 'yzd-v/DWPose', 'yolox_l.onnx', 'dw-ll_ucoco_384.onnx', '--local-dir', (Join-Path $Models 'DWPose'))
    Invoke-External -FilePath $HfCli -ArgumentList @('download', 'stabilityai/stable-video-diffusion-img2vid-xt-1-1', '--local-dir', (Join-Path $Models 'stable-video-diffusion-img2vid-xt-1-1'))
}

if (-not $WhatIfPreference) {
    Invoke-External -FilePath $Python -ArgumentList @('-c', 'import torch, diffusers, transformers, decord, onnxruntime; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))')
    Write-Host 'MimicMotion runtime installation completed.' -ForegroundColor Green
}
