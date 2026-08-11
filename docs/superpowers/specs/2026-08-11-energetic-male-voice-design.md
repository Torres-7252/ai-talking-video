# Energetic Male Voice Design

## Goal

Add a local, selectable Chinese energetic male voice while preserving the existing GPT-SoVITS cloned voice.

## Architecture

CosyVoice runs in a dedicated Python 3.10 virtual environment and is invoked through a subprocess runner. The main pipeline remains dependency-isolated and routes voice generation by profile provider. `CosyVoice-300M-Instruct` supplies the `Chinese Male` speaker preset and an energetic promotional instruction.

## User Experience

The create form gains a voice selector with two options:

- `My Voice`: existing GPT-SoVITS reference clone.
- `Energetic Male`: local CosyVoice Chinese male preset with bright, confident promotional delivery.

The selected profile is sent as the existing `voice` request field and stored in project metadata.

## Runtime

- Repository: official `FunAudioLLM/CosyVoice`, pinned to commit `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`.
- Environment: `.venv-cosyvoice`, Python 3.10.
- Model: `CosyVoice-300M-Instruct` under `voice/models/CosyVoice/pretrained_models`.
- GPU: CUDA when available; the RTX 4060 Laptop GPU has sufficient VRAM for the 300M model.

## Failure Handling

The provider validates runtime files before launch, captures runner output, and validates the generated WAV before returning. Missing runtime or model files produce an actionable installer command. The original GPT-SoVITS provider remains unchanged for `default`.

## Verification

- Unit tests cover profile routing, subprocess arguments, request propagation, and frontend selection.
- A real energetic-male sample is generated and checked with FFprobe and FunASR.
- The complete test suite runs before the web server is restarted.

