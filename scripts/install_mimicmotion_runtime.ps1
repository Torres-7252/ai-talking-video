[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Runtime = Join-Path $ProjectRoot 'app\backend\providers\motion\MimicMotion'
$Venv = Join-Path $ProjectRoot '.venv-mimicmotion'
$Python = Join-Path $Venv 'Scripts\python.exe'
$DownloadVenv = Join-Path $ProjectRoot '.venv-hf-download'
$DownloadPython = Join-Path $DownloadVenv 'Scripts\python.exe'
$HfCli = Join-Path $DownloadVenv 'Scripts\hf.exe'
$Commit = '6907bdcc259a6a048d41a365e840d22274f9256c'
$Repository = 'https://github.com/Tencent/MimicMotion.git'
$ArchiveUrl = "https://codeload.github.com/Tencent/MimicMotion/zip/$Commit"
$CommitMarker = Join-Path $Runtime '.mimicmotion-commit'

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

function Test-ModelArtifact {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][long]$Size,
        [string]$Sha256 = ''
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    if ((Get-Item -LiteralPath $Path).Length -ne $Size) {
        return $false
    }
    if ($Sha256) {
        return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Sha256
    }
    return $true
}

function Ensure-HfArtifact {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$Filename,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][long]$Size,
        [string]$Sha256 = ''
    )
    $Artifact = Join-Path $Destination ($Filename -replace '/', '\')
    if (Test-ModelArtifact -Path $Artifact -Size $Size -Sha256 $Sha256) {
        Write-Host "Verified: $Artifact"
        return
    }
    Invoke-External -FilePath $HfCli -ArgumentList @(
        'download', $Repository, $Filename, '--local-dir', $Destination, '--force-download'
    )
    if (-not (Test-ModelArtifact -Path $Artifact -Size $Size -Sha256 $Sha256)) {
        throw "Downloaded model failed integrity validation: $Artifact"
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

if (-not (Test-Path (Join-Path $Runtime '.git')) -and -not (Test-Path $CommitMarker) -and $PSCmdlet.ShouldProcess($Runtime, 'Download pinned MimicMotion source archive')) {
    $Archive = Join-Path $env:TEMP "MimicMotion-$Commit.zip"
    $ExtractRoot = Join-Path $env:TEMP "MimicMotion-$Commit"
    Invoke-WebRequest -Uri $ArchiveUrl -OutFile $Archive
    if (Test-Path $ExtractRoot) {
        Remove-Item -LiteralPath $ExtractRoot -Recurse -Force
    }
    Expand-Archive -LiteralPath $Archive -DestinationPath $ExtractRoot -Force
    $Extracted = Get-ChildItem -LiteralPath $ExtractRoot -Directory | Select-Object -First 1
    if (-not $Extracted) {
        throw "MimicMotion source archive was empty: $Archive"
    }
    New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
    Get-ChildItem -LiteralPath $Extracted.FullName -Force | Copy-Item -Destination $Runtime -Recurse -Force
    Set-Content -LiteralPath $CommitMarker -Value $Commit -NoNewline -Encoding ascii
    Remove-Item -LiteralPath $Archive, $ExtractRoot -Recurse -Force
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
elseif (Test-Path $CommitMarker) {
    $ArchivedCommit = (Get-Content -LiteralPath $CommitMarker -Raw).Trim()
    if ($ArchivedCommit -ne $Commit) {
        throw "MimicMotion archive is pinned to $ArchivedCommit, expected $Commit"
    }
}

$TorchReady = $false
if ((Test-Path $Python) -and -not $WhatIfPreference) {
    try {
        & $Python -c "import torch; assert torch.__version__.startswith('2.3.'); assert torch.cuda.is_available()" 2>$null
        $TorchReady = $LASTEXITCODE -eq 0
    }
    catch {
        $TorchReady = $false
    }
}
if (-not $TorchReady -and $PSCmdlet.ShouldProcess($Venv, 'Install CUDA PyTorch')) {
    Invoke-External -FilePath $Python -ArgumentList @('-m', 'pip', 'install', '--upgrade', 'pip')
    Invoke-External -FilePath $Python -ArgumentList @('-m', 'pip', 'install', 'torch==2.3.1', 'torchvision==0.18.1', 'torchaudio==2.3.1', '--index-url', 'https://download.pytorch.org/whl/cu121')
}

$DependenciesReady = $false
if ((Test-Path $Python) -and -not $WhatIfPreference) {
    try {
        & $Python -c "import diffusers, transformers, decord, omegaconf, onnxruntime" 2>$null
        $DependenciesReady = $LASTEXITCODE -eq 0
    }
    catch {
        $DependenciesReady = $false
    }
}
if (-not $DependenciesReady -and $PSCmdlet.ShouldProcess($Venv, 'Install pinned MimicMotion dependencies')) {
    Invoke-External -FilePath $Python -ArgumentList @('-m', 'pip', 'install', '--index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple', '--timeout', '60', 'numpy==1.26.4', 'diffusers==0.27.0', 'transformers==4.32.1', 'huggingface_hub==0.24.7', 'decord==0.6.0', 'einops==0.8.1', 'omegaconf==2.3.0', 'onnxruntime-gpu==1.18.1', 'opencv-python==4.10.0.84', 'matplotlib==3.9.2', 'tqdm==4.66.5')
}

$Models = Join-Path $Runtime 'models'
$Checkpoint = Join-Path $Models 'MimicMotion_1-1.pth'
$Detector = Join-Path $Models 'DWPose\yolox_l.onnx'
$Pose = Join-Path $Models 'DWPose\dw-ll_ucoco_384.onnx'
$BaseModel = Join-Path $Models 'stable-video-diffusion-img2vid-xt-1-1\model_index.json'
$ModelIntegrityReady = (
    (Test-ModelArtifact -Path $Checkpoint -Size 3049867447 -Sha256 'b812659ea273b2758c918facf759af5d7cad9564dc35156c59ec17e93f9749a4') -and
    (Test-ModelArtifact -Path $Detector -Size 216746733 -Sha256 '7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411') -and
    (Test-ModelArtifact -Path $Pose -Size 134399116 -Sha256 '724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843') -and
    (Test-Path -LiteralPath $BaseModel -PathType Leaf)
)
if (-not $ModelIntegrityReady -and $PSCmdlet.ShouldProcess($Models, 'Download and verify MimicMotion runtime weights')) {
    New-Item -ItemType Directory -Force -Path $Models | Out-Null
    if (-not (Test-Path $DownloadPython)) {
        Invoke-External -FilePath py -ArgumentList @('-3.10', '-m', 'venv', $DownloadVenv)
    }
    if (-not (Test-Path $HfCli)) {
        Invoke-External -FilePath $DownloadPython -ArgumentList @(
            '-m', 'pip', 'install', '--upgrade', 'huggingface_hub[hf_xet]==0.36.2'
        )
    }
    $env:HF_XET_FIXED_DOWNLOAD_CONCURRENCY = '8'
    Ensure-HfArtifact -Repository 'tencent/MimicMotion' -Filename 'MimicMotion_1-1.pth' -Destination $Models -Size 3049867447 -Sha256 'b812659ea273b2758c918facf759af5d7cad9564dc35156c59ec17e93f9749a4'

    $DWPose = Join-Path $Models 'DWPose'
    Ensure-HfArtifact -Repository 'yzd-v/DWPose' -Filename 'yolox_l.onnx' -Destination $DWPose -Size 216746733 -Sha256 '7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411'
    Ensure-HfArtifact -Repository 'yzd-v/DWPose' -Filename 'dw-ll_ucoco_384.onnx' -Destination $DWPose -Size 134399116 -Sha256 '724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843'

    $Svd = Join-Path $Models 'stable-video-diffusion-img2vid-xt-1-1'
    $SvdRepository = 'weights/stable-video-diffusion-img2vid-xt-1-1'
    Ensure-HfArtifact -Repository $SvdRepository -Filename 'model_index.json' -Destination $Svd -Size 496
    Ensure-HfArtifact -Repository $SvdRepository -Filename 'feature_extractor/preprocessor_config.json' -Destination $Svd -Size 518
    Ensure-HfArtifact -Repository $SvdRepository -Filename 'image_encoder/config.json' -Destination $Svd -Size 685
    Ensure-HfArtifact -Repository $SvdRepository -Filename 'image_encoder/model.fp16.safetensors' -Destination $Svd -Size 1264217240 -Sha256 'ae616c24393dd1854372b0639e5541666f7521cbe219669255e865cb7f89466a'
    Ensure-HfArtifact -Repository $SvdRepository -Filename 'scheduler/scheduler_config.json' -Destination $Svd -Size 533
    Ensure-HfArtifact -Repository $SvdRepository -Filename 'unet/config.json' -Destination $Svd -Size 984
    Ensure-HfArtifact -Repository $SvdRepository -Filename 'vae/config.json' -Destination $Svd -Size 607
    Ensure-HfArtifact -Repository $SvdRepository -Filename 'vae/diffusion_pytorch_model.fp16.safetensors' -Destination $Svd -Size 195531910 -Sha256 'af602cd0eb4ad6086ec94fbf1438dfb1be5ec9ac03fd0215640854e90d6463a3'
}

if (-not $WhatIfPreference) {
    Invoke-External -FilePath $Python -ArgumentList @('-c', 'import torch, diffusers, transformers, decord, omegaconf, onnxruntime; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))')
    Write-Host 'MimicMotion runtime installation completed.' -ForegroundColor Green
}
