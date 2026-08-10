"""Ditto audio-driven portrait provider."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

from app.backend.providers.media_utils import validate_audio, validate_video


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DITTO_ROOT = Path(__file__).resolve().parent / "Ditto"
CHECKPOINT_ROOT = DITTO_ROOT / "checkpoints"
RUNTIME_PYTHON = PROJECT_ROOT / ".venv-liveportrait" / "Scripts" / "python.exe"

REQUIRED_MODEL_FILES = (
    Path("ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"),
    Path("ditto_pytorch/aux_models/2d106det.onnx"),
    Path("ditto_pytorch/aux_models/det_10g.onnx"),
    Path("ditto_pytorch/aux_models/face_landmarker.task"),
    Path("ditto_pytorch/aux_models/hubert_streaming_fix_kv.onnx"),
    Path("ditto_pytorch/aux_models/landmark203.onnx"),
    Path("ditto_pytorch/models/appearance_extractor.pth"),
    Path("ditto_pytorch/models/decoder.pth"),
    Path("ditto_pytorch/models/lmdm_v0.4_hubert.pth"),
    Path("ditto_pytorch/models/motion_extractor.pth"),
    Path("ditto_pytorch/models/stitch_network.pth"),
    Path("ditto_pytorch/models/warp_network.pth"),
)


def missing_model_files(checkpoint_root: Path = CHECKPOINT_ROOT) -> list[Path]:
    root = Path(checkpoint_root).resolve()
    return [relative for relative in REQUIRED_MODEL_FILES if not (root / relative).is_file()]


def build_inference_command(
    source_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    ditto_root: Path = DITTO_ROOT,
    python_executable: Path = RUNTIME_PYTHON,
) -> list[str]:
    root = Path(ditto_root).resolve()
    checkpoints = root / "checkpoints"
    return [
        str(Path(python_executable).resolve()),
        str(root / "inference.py"),
        "--data_root",
        str(checkpoints / "ditto_pytorch"),
        "--cfg_pkl",
        str(checkpoints / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl"),
        "--audio_path",
        str(Path(audio_path).resolve()),
        "--source_path",
        str(Path(source_path).resolve()),
        "--output_path",
        str(Path(output_path).resolve()),
    ]


def _format_missing(paths: Iterable[Path]) -> str:
    return ", ".join(str(path).replace("\\", "/") for path in paths)


def generate_avatar(source_path: str, audio_path: str, output_path: str) -> Path:
    source = Path(source_path).resolve()
    audio = Path(audio_path).resolve()
    output = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Avatar image does not exist: {source}")
    validate_audio(audio)
    if not RUNTIME_PYTHON.is_file():
        raise RuntimeError(
            "Ditto runtime is missing. Run scripts/install_ditto_runtime.ps1 first."
        )
    if not (DITTO_ROOT / "inference.py").is_file():
        raise RuntimeError(
            "Ditto source is missing. Run scripts/install_ditto_runtime.ps1 first."
        )
    missing = missing_model_files()
    if missing:
        raise RuntimeError(f"Ditto model files are incomplete: {_format_missing(missing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_inference_command(source, audio, output)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    print(f"  [avatar] Ditto: {source.name} + {audio.name} -> {output.name}")
    subprocess.run(command, cwd=DITTO_ROOT, env=environment, check=True)
    validate_video(output)
    print(f"  [avatar] Generated: {output}")
    return output

