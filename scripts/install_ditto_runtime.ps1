[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Runtime = Join-Path $ProjectRoot 'app\backend\providers\avatar\Ditto'
$Python = Join-Path $ProjectRoot '.venv-liveportrait\Scripts\python.exe'
$Patch = Join-Path $PSScriptRoot 'ditto-windows-putback.patch'
$Commit = 'c3e47eee2e626500017a0556b470d6d4182f85e8'
$Repository = 'https://github.com/antgroup/ditto-talkinghead.git'

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

Write-Host 'Ditto PyTorch runtime installer'
Write-Host "Project: $ProjectRoot"
Write-Host "Pinned commit: $Commit"

if (-not (Test-Path $Python)) {
    throw 'LivePortrait Python 3.10 runtime is required. Run scripts\install_liveportrait_runtime.ps1 first.'
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git is required but was not found in PATH.'
}

if (-not (Test-Path (Join-Path $Runtime '.git'))) {
    if ($PSCmdlet.ShouldProcess($Runtime, 'Clone Ditto')) {
        New-Item -ItemType Directory -Force -Path (Split-Path $Runtime) | Out-Null
        Invoke-External -FilePath git -ArgumentList @('clone', $Repository, $Runtime)
    }
}

if (Test-Path (Join-Path $Runtime '.git')) {
    $CurrentCommit = (& git -C $Runtime rev-parse HEAD).Trim()
    if ($CurrentCommit -ne $Commit) {
        $Dirty = (& git -C $Runtime status --porcelain)
        if ($Dirty) {
            throw "Ditto checkout has local changes and cannot switch commits: $Runtime"
        }
        if ($PSCmdlet.ShouldProcess($Runtime, "Checkout pinned commit $Commit")) {
            Invoke-External -FilePath git -ArgumentList @('-C', $Runtime, 'fetch', 'origin', $Commit)
            Invoke-External -FilePath git -ArgumentList @('-C', $Runtime, 'checkout', '--detach', $Commit)
        }
    }

    $PutBack = Join-Path $Runtime 'core\atomic_components\putback.py'
    $Patched = Select-String -LiteralPath $PutBack -SimpleMatch 'Windows fallback for the optional Cython image blender.' -Quiet
    if (-not $Patched -and $PSCmdlet.ShouldProcess($PutBack, 'Apply Windows NumPy image blender fallback')) {
        Invoke-External -FilePath git -ArgumentList @('-C', $Runtime, 'apply', $Patch)
    }
}

if ($PSCmdlet.ShouldProcess($Python, 'Install Ditto Python dependencies')) {
    Invoke-External -FilePath $Python -ArgumentList @(
        '-m', 'pip', 'install',
        'mediapipe==0.10.21',
        'librosa==0.10.2.post1',
        'filetype==1.2.0',
        'einops==0.8.1',
        'huggingface_hub==0.34.4'
    )
}

$Checkpoints = Join-Path $Runtime 'checkpoints'
$DownloadCode = @'
from huggingface_hub import snapshot_download
import sys

snapshot_download(
    repo_id='digital-avatar/ditto-talkinghead',
    local_dir=sys.argv[1],
    allow_patterns=[
        'ditto_cfg/v0.4_hubert_cfg_pytorch.pkl',
        'ditto_pytorch/**',
    ],
    max_workers=4,
)
'@
if ($PSCmdlet.ShouldProcess($Checkpoints, 'Download official Ditto PyTorch weights')) {
    Invoke-External -FilePath $Python -ArgumentList @('-c', $DownloadCode, $Checkpoints)
}

if (-not $WhatIfPreference) {
    $PreviousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = $ProjectRoot
    try {
        Invoke-External -FilePath $Python -ArgumentList @(
            '-c',
            "from app.backend.providers.avatar import missing_model_files; missing=missing_model_files(); print('Ditto model check: OK' if not missing else missing); raise SystemExit(bool(missing))"
        )
        Push-Location $Runtime
        try {
            Invoke-External -FilePath $Python -ArgumentList @('-c', "from stream_pipeline_offline import StreamSDK; print('Ditto import check: OK')")
        }
        finally {
            Pop-Location
        }
    }
    finally {
        $env:PYTHONPATH = $PreviousPythonPath
    }
    Write-Host 'Ditto runtime installation completed.' -ForegroundColor Green
}
