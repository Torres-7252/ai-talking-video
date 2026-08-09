[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$SkipTemplate
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Runtime = Join-Path $ProjectRoot 'app\backend\providers\motion\LivePortrait'
$Venv = Join-Path $ProjectRoot '.venv-liveportrait'
$Python = Join-Path $Venv 'Scripts\python.exe'
$Templates = Join-Path $ProjectRoot 'motion\templates'
$Drivers = Join-Path $ProjectRoot 'motion\drivers'
$Commit = '9b294b3d0536135442ea73cb01e6cb3ca7029dd3'
$Repository = 'https://github.com/KlingAIResearch/LivePortrait.git'

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

Write-Host 'LivePortrait isolated runtime installer'
Write-Host "Project: $ProjectRoot"
Write-Host "Pinned commit: $Commit"

if ($WhatIfPreference) {
    Write-Host 'WhatIf: Python 3.10, Git, network, and model downloads will be checked during a real run.'
}
else {
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

if (-not (Test-Path $Python)) {
    if ($PSCmdlet.ShouldProcess($Venv, 'Create Python 3.10 virtual environment')) {
        Invoke-External -FilePath py -ArgumentList @('-3.10', '-m', 'venv', $Venv)
    }
}

if (-not (Test-Path (Join-Path $Runtime '.git'))) {
    if ($PSCmdlet.ShouldProcess($Runtime, 'Clone LivePortrait')) {
        New-Item -ItemType Directory -Force -Path (Split-Path $Runtime) | Out-Null
        Invoke-External -FilePath git -ArgumentList @('clone', $Repository, $Runtime)
    }
}

if (Test-Path (Join-Path $Runtime '.git')) {
    $Dirty = (& git -C $Runtime status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect LivePortrait checkout: $Runtime"
    }
    if ($Dirty) {
        throw "LivePortrait checkout has local changes. Clean it before reinstalling: $Runtime"
    }
    $CurrentCommit = (& git -C $Runtime rev-parse HEAD).Trim()
    if ($CurrentCommit -ne $Commit) {
        if ($PSCmdlet.ShouldProcess($Runtime, "Checkout pinned commit $Commit")) {
            Invoke-External -FilePath git -ArgumentList @('-C', $Runtime, 'fetch', 'origin', $Commit)
            Invoke-External -FilePath git -ArgumentList @('-C', $Runtime, 'checkout', '--detach', $Commit)
        }
    }
}

$DependenciesReady = $false
if ((Test-Path $Python) -and -not $WhatIfPreference) {
    & $Python -c "import albucore, torch; assert albucore.__version__ == '0.0.12'; assert torch.__version__.startswith('2.3.0+cu121'); assert torch.cuda.is_available()" 2>$null
    $DependenciesReady = $LASTEXITCODE -eq 0
}

if (-not $DependenciesReady -and $PSCmdlet.ShouldProcess($Venv, 'Install CUDA PyTorch and LivePortrait dependencies')) {
    $Constraints = Join-Path $ProjectRoot 'scripts\liveportrait-constraints.txt'
    Invoke-External -FilePath $Python -ArgumentList @('-m', 'pip', 'install', '--upgrade', 'pip')
    Invoke-External -FilePath $Python -ArgumentList @('-m', 'pip', 'install', '-r', (Join-Path $Runtime 'requirements.txt'), '-c', $Constraints)
    Invoke-External -FilePath $Python -ArgumentList @('-m', 'pip', 'install', 'torch==2.3.0', 'torchvision==0.18.0', 'torchaudio==2.3.0', '--index-url', 'https://download.pytorch.org/whl/cu121')
    Invoke-External -FilePath $Python -ArgumentList @('-m', 'pip', 'install', 'huggingface_hub[cli]')
}

$Weights = Join-Path $Runtime 'pretrained_weights'
$RequiredWeight = Join-Path $Weights 'liveportrait\base_models\appearance_feature_extractor.pth'
if (-not (Test-Path $RequiredWeight)) {
    if ($PSCmdlet.ShouldProcess($Weights, 'Download official KlingTeam/LivePortrait weights')) {
        $HfCli = Join-Path $Venv 'Scripts\hf.exe'
        if (-not (Test-Path $HfCli)) {
            $HfCli = Join-Path $Venv 'Scripts\huggingface-cli.exe'
        }
        if (-not (Test-Path $HfCli)) {
            throw "Hugging Face CLI was not installed in $Venv"
        }
        Invoke-External -FilePath $HfCli -ArgumentList @('download', 'KlingTeam/LivePortrait', '--local-dir', $Weights, '--include', 'liveportrait/*', 'insightface/*')
    }
}

if (-not $SkipTemplate) {
    $SourceDriver = Join-Path $Runtime 'assets\examples\driving\d0.mp4'
    $SourcePortrait = Join-Path $Runtime 'assets\examples\source\s0.jpg'
    $Driver = Join-Path $Drivers 'steady.mp4'
    $Template = Join-Path $Templates 'steady.pkl'
    $GeneratedTemplate = [System.IO.Path]::ChangeExtension($Driver, '.pkl')
    $SetupOutput = Join-Path $Runtime 'animations\setup'

    if (-not (Test-Path $Template)) {
        if ($PSCmdlet.ShouldProcess($Template, 'Generate reusable steady motion template')) {
            New-Item -ItemType Directory -Force -Path $Drivers, $Templates, $SetupOutput | Out-Null
            $LoopFilter = '[0:v]trim=end_frame=25,setpts=PTS-STARTPTS,split=2[forward][reverse];[reverse]reverse[backward];[forward][backward]concat=n=2:v=1:a=0,fps=25[loop]'
            Invoke-External -FilePath ffmpeg -ArgumentList @('-hide_banner', '-loglevel', 'error', '-i', $SourceDriver, '-filter_complex', $LoopFilter, '-map', '[loop]', '-an', '-y', $Driver)
            $PreviousPythonUtf8 = $env:PYTHONUTF8
            $PreviousPythonIoEncoding = $env:PYTHONIOENCODING
            $env:PYTHONUTF8 = '1'
            $env:PYTHONIOENCODING = 'utf-8'
            Push-Location $Runtime
            try {
                Invoke-External -FilePath $Python -ArgumentList @((Join-Path $Runtime 'inference.py'), '--source', $SourcePortrait, '--driving', $Driver, '--output-dir', $SetupOutput, '--driving-multiplier', '0.35', '--source-max-dim', '1280')
            }
            finally {
                Pop-Location
                $env:PYTHONUTF8 = $PreviousPythonUtf8
                $env:PYTHONIOENCODING = $PreviousPythonIoEncoding
            }
            if (-not (Test-Path $GeneratedTemplate)) {
                throw "LivePortrait did not create the expected template: $GeneratedTemplate"
            }
            Copy-Item -LiteralPath $GeneratedTemplate -Destination $Template -Force
        }
    }
}

if (-not $WhatIfPreference) {
    Invoke-External -FilePath $Python -ArgumentList @('-c', 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))')
    Write-Host 'LivePortrait runtime installation completed.' -ForegroundColor Green
}
