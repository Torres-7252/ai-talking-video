$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runtime = Join-Path $ProjectRoot ".venv-cosyvoice"
$SeedRuntime = Join-Path $ProjectRoot ".venv-liveportrait"
$Repository = Join-Path $ProjectRoot "voice\models\CosyVoice"
$Model = Join-Path $Repository "pretrained_models\CosyVoice-300M-Instruct"
$Requirements = Join-Path $PSScriptRoot "cosyvoice-requirements-windows.txt"
$Commit = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
$Python = "C:\Users\31078\AppData\Local\Programs\Python\Python310\python.exe"

function Assert-NativeSuccess([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python 3.10 was not found at $Python"
}
if (-not (Test-Path -LiteralPath (Join-Path $Runtime "Scripts\python.exe"))) {
    & $Python -m venv $Runtime
    Assert-NativeSuccess "Create CosyVoice virtual environment"
}
$RuntimePython = Join-Path $Runtime "Scripts\python.exe"

if (-not (Test-Path -LiteralPath (Join-Path $Repository ".git"))) {
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $Repository
    Assert-NativeSuccess "Clone CosyVoice"
}
git -C $Repository fetch origin $Commit --depth 1
Assert-NativeSuccess "Fetch pinned CosyVoice commit"
git -C $Repository checkout --detach $Commit
Assert-NativeSuccess "Checkout pinned CosyVoice commit"
git -C $Repository submodule update --init --recursive
Assert-NativeSuccess "Update CosyVoice submodules"

& $RuntimePython -m pip install --upgrade pip "setuptools==80.9.0" wheel
Assert-NativeSuccess "Install Python packaging tools"

& $RuntimePython -c "import torch, torchaudio"
$TorchReady = $LASTEXITCODE -eq 0
if (-not $TorchReady) {
    $SeedPackages = Join-Path $SeedRuntime "Lib\site-packages"
    $SeedLibrary = Join-Path $SeedRuntime "Library"
    if ((Test-Path -LiteralPath (Join-Path $SeedPackages "torch")) -and
        (Test-Path -LiteralPath $SeedLibrary)) {
        Write-Host "Seeding CUDA PyTorch from .venv-liveportrait"
        Copy-Item -Path (Join-Path $SeedPackages "*") -Destination (Join-Path $Runtime "Lib\site-packages") -Recurse -Force
        Copy-Item -LiteralPath $SeedLibrary -Destination (Join-Path $Runtime "Library") -Recurse -Force
    } else {
        & $RuntimePython -m pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
        Assert-NativeSuccess "Install PyTorch"
    }
}

$env:PATH = "$(Join-Path $Runtime 'Library\bin');$env:PATH"
& $RuntimePython -c "import torch, torchaudio; assert torch.cuda.is_available()"
Assert-NativeSuccess "Verify CUDA PyTorch"
& $RuntimePython -m pip install openai-whisper==20231117 --no-build-isolation
Assert-NativeSuccess "Install OpenAI Whisper"
& $RuntimePython -m pip install -r $Requirements
Assert-NativeSuccess "Install CosyVoice inference dependencies"

$Aria = (Get-Command aria2c -ErrorAction SilentlyContinue).Source
if (-not $Aria) {
    throw "aria2c is required to download CosyVoice model weights"
}
New-Item -ItemType Directory -Path $Model -Force | Out-Null
$ModelFiles = @(
    @{ Name = "configuration.json"; Size = 47 },
    @{ Name = "cosyvoice.yaml"; Size = 6421 },
    @{ Name = "campplus.onnx"; Size = 28303423 },
    @{ Name = "spk2info.pt"; Size = 7772 },
    @{ Name = "llm.pt"; Size = 1242994771 },
    @{ Name = "flow.pt"; Size = 419900943 },
    @{ Name = "hift.pt"; Size = 81896716 },
    @{ Name = "speech_tokenizer_v1.onnx"; Size = 522624269 }
)
foreach ($File in $ModelFiles) {
    $Destination = Join-Path $Model $File.Name
    if ((-not (Test-Path -LiteralPath $Destination)) -or
        ((Get-Item -LiteralPath $Destination).Length -ne $File.Size)) {
        $Url = "https://modelscope.cn/api/v1/models/iic/CosyVoice-300M-Instruct/repo?Revision=master&FilePath=$($File.Name)"
        & $Aria -c -x 8 -s 8 -k 1M --file-allocation=none -d $Model -o $File.Name $Url
        Assert-NativeSuccess "Download $($File.Name)"
    }
    if ((Get-Item -LiteralPath $Destination).Length -ne $File.Size) {
        throw "Downloaded model file has the wrong size: $Destination"
    }
}

Write-Host "CosyVoice runtime ready: $RuntimePython"
Write-Host "CosyVoice model ready: $Model"
