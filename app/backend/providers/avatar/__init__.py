"""Ditto audio-driven portrait provider."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from app.backend.providers.media_utils import validate_audio, validate_video


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DITTO_ROOT = Path(__file__).resolve().parent / "Ditto"
CHECKPOINT_ROOT = DITTO_ROOT / "checkpoints"
RUNTIME_PYTHON = PROJECT_ROOT / ".venv-liveportrait" / "Scripts" / "python.exe"
DITTO_TIMEOUT_SECONDS = 15 * 60

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


def _log_tail(log_path: Path, limit: int = 4000) -> str:
    if not log_path.is_file():
        return "No Ditto log was written."
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:].strip() or "No Ditto output was written."


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Stop Ditto and its Python worker when a timed-out run is wedged."""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        process.kill()


def run_ditto(
    command: list[str],
    *,
    log_path: Path,
    timeout_seconds: int = DITTO_TIMEOUT_SECONDS,
    cwd: Path = DITTO_ROOT,
    environment: dict[str, str] | None = None,
) -> None:
    """Run Ditto with a bounded lifetime and a persistent diagnostic log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_tree(process)
            raise RuntimeError(
                f"Ditto timed out after {timeout_seconds // 60} minutes. "
                f"See {log_path.name}: {_log_tail(log_path)}"
            ) from exc

    if return_code:
        raise RuntimeError(
            f"Ditto exited with code {return_code}. See {log_path.name}: {_log_tail(log_path)}"
        )


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
    log_path = output.with_name("ditto.log")
    print(f"  [avatar] Ditto: {source.name} + {audio.name} -> {output.name}")
    run_ditto(command, log_path=log_path, environment=environment)
    validate_video(output)
    print(f"  [avatar] Generated: {output}")
    return output
