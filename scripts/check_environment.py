#!/usr/bin/env python3
"""Validate the local runtime, assets, and exact model files."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _command_version(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"exit code {result.returncode}")
    return (result.stdout or result.stderr).splitlines()[0]


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    checks.append(("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0]))
    for executable, args in (
        ("ffmpeg", ["ffmpeg", "-version"]),
        ("ffprobe", ["ffprobe", "-version"]),
        ("nvidia-smi", ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
    ):
        try:
            detail = _command_version(args) if shutil.which(executable) else "not found"
            checks.append((executable, shutil.which(executable) is not None, detail))
        except Exception as exc:
            checks.append((executable, False, str(exc)))

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        detail = f"torch {torch.__version__}"
        if cuda_ok:
            detail += f", {torch.cuda.get_device_name(0)}"
        checks.append(("PyTorch CUDA", cuda_ok, detail))
    except Exception as exc:
        checks.append(("PyTorch CUDA", False, str(exc)))

    for module_name in (
        "cv2",
        "diffusers",
        "fastapi",
        "funasr",
        "librosa",
        "numpy",
        "omegaconf",
        "soundfile",
        "transformers",
        "yaml",
    ):
        try:
            module = importlib.import_module(module_name)
            checks.append((module_name, True, str(getattr(module, "__version__", "installed"))))
        except Exception as exc:
            checks.append((module_name, False, str(exc)))

    from app.backend.providers.lipsync import missing_musetalk_files
    from app.backend.providers.media_utils import validate_audio
    from app.backend.providers.voice import build_tts_config

    tts_paths = [Path(value) for key, value in build_tts_config()["custom"].items() if key.endswith("_path")]
    missing_tts = [path for path in tts_paths if not path.exists()]
    checks.append(("GPT-SoVITS v3 weights", not missing_tts, f"missing {len(missing_tts)} file(s)"))

    missing_musetalk = missing_musetalk_files()
    checks.append(("MuseTalk 1.5 weights", not missing_musetalk, f"missing {len(missing_musetalk)} file(s)"))

    avatar = PROJECT_ROOT / "avatar" / "avatar.jpg"
    checks.append(("Avatar image", avatar.is_file() and avatar.stat().st_size > 0, str(avatar)))
    reference = PROJECT_ROOT / "voice" / "references" / "default.wav"
    try:
        audio_info = validate_audio(reference)
        checks.append(("Reference voice", True, f"{audio_info['duration']:.2f}s"))
    except Exception as exc:
        checks.append(("Reference voice", False, str(exc)))

    print("Local AI talking video environment")
    print("=" * 60)
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'FAIL'}] {name}: {detail}")
    failed = [name for name, passed, _ in checks if not passed]
    print("=" * 60)
    if failed:
        print("Missing or invalid: " + ", ".join(failed))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
