"""MuseTalk 1.5 adapter for local talking-head generation."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image

from app.backend.providers.media_utils import validate_audio, validate_video


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MUSETALK_PATH = PROJECT_ROOT / "app" / "backend" / "providers" / "lipsync" / "MuseTalk"
COMPAT_PATH = PROJECT_ROOT / "app" / "backend" / "providers" / "lipsync" / "compat"
EXPECTED_MODEL_SIZES = {
    "models/musetalkV15/unet.pth": 3_400_074_924,
    "musetalk/utils/face_detection/detection/sfd/s3fd.pth": 89_843_225,
}


def required_musetalk_files() -> tuple[Path, ...]:
    """Return the local files needed by the MuseTalk 1.5 inference path."""
    relative_paths = (
        "models/musetalkV15/musetalk.json",
        "models/musetalkV15/unet.pth",
        "models/sd-vae/config.json",
        "models/sd-vae/diffusion_pytorch_model.bin",
        "models/whisper/config.json",
        "models/whisper/preprocessor_config.json",
        "models/whisper/pytorch_model.bin",
        "models/dwpose/dw-ll_ucoco_384.pth",
        "models/face-parse-bisent/79999_iter.pth",
        "models/face-parse-bisent/resnet18-5c106cde.pth",
        "musetalk/utils/face_detection/detection/sfd/s3fd.pth",
    )
    return tuple(MUSETALK_PATH / path for path in relative_paths)


def missing_musetalk_files() -> list[Path]:
    """Return missing, empty, or size-mismatched MuseTalk model files."""
    missing = []
    for path in required_musetalk_files():
        relative_path = path.relative_to(MUSETALK_PATH).as_posix()
        expected_size = EXPECTED_MODEL_SIZES.get(relative_path)
        if (
            not path.is_file()
            or path.stat().st_size == 0
            or (expected_size is not None and path.stat().st_size != expected_size)
        ):
            missing.append(path)
    return missing


def _job_paths(output_path: Path) -> tuple[Path, Path, Path]:
    work_dir = output_path.parent / ".musetalk" / output_path.stem
    config_path = work_dir / "inference.yaml"
    result_dir = work_dir / "results"
    generated_path = result_dir / "v15" / output_path.name
    return config_path, result_dir, generated_path


def build_musetalk_job(
    avatar_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    use_fp16: bool = True,
) -> tuple[dict, list[str], Path]:
    """Build a MuseTalk YAML payload and its official v1.5 CLI command."""
    avatar = Path(avatar_path).resolve()
    audio = Path(audio_path).resolve()
    output = Path(output_path).resolve()
    config_path, result_dir, _ = _job_paths(output)

    job = {
        "task_0": {
            "video_path": str(avatar),
            "audio_path": str(audio),
            "bbox_shift": 0,
        }
    }
    command = [
        sys.executable,
        "-m",
        "scripts.inference",
        "--inference_config",
        str(config_path),
        "--result_dir",
        str(result_dir),
        "--unet_config",
        "models/musetalkV15/musetalk.json",
        "--unet_model_path",
        "models/musetalkV15/unet.pth",
        "--whisper_dir",
        "models/whisper",
        "--vae_type",
        "sd-vae",
        "--version",
        "v15",
        "--fps",
        "25",
        "--batch_size",
        "1",
        "--output_vid_name",
        output.name,
    ]
    if use_fp16:
        command.append("--use_float16")
    return job, command, MUSETALK_PATH


def _validate_talking_video(path: Path) -> dict:
    info = validate_video(path)
    if not any(stream.get("codec_type") == "audio" for stream in info["streams"]):
        raise RuntimeError(f"MuseTalk output has no audio stream: {path}")
    return info


def build_musetalk_environment() -> dict[str, str]:
    """Expose Windows/Python 3.12 compatibility modules to MuseTalk."""
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(COMPAT_PATH), existing) if value
    )
    return environment


def prepare_even_avatar(avatar_path: Path, output_path: Path) -> Path:
    """Pad odd image dimensions so MuseTalk's H.264 step can encode them."""
    avatar = Path(avatar_path).resolve()
    with Image.open(avatar) as image:
        width, height = image.size
        if width % 2 == 0 and height % 2 == 0:
            return avatar
        config_path, _, _ = _job_paths(Path(output_path).resolve())
        normalized = config_path.parent / f"{avatar.stem}_even.png"
        normalized.parent.mkdir(parents=True, exist_ok=True)
        canvas = Image.new(image.mode, (width + width % 2, height + height % 2))
        canvas.paste(image, (0, 0))
        canvas.save(normalized, format="PNG")
    return normalized


def generate_lipsync(
    avatar_path: str,
    audio_path: str,
    output_path: str,
    use_fp16: bool = True,
    avatar_cache_dir: Optional[str] = None,
) -> Path:
    """Generate a talking-head MP4 from a still image and WAV file."""
    del avatar_cache_dir  # MuseTalk's image workflow manages coordinates itself.

    avatar = Path(avatar_path).resolve()
    audio = Path(audio_path).resolve()
    output = Path(output_path).resolve()
    if not avatar.is_file():
        raise FileNotFoundError(f"Avatar image does not exist: {avatar}")
    validate_audio(audio)

    missing = missing_musetalk_files()
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "MuseTalk 1.5 model files are incomplete. Run "
            "scripts\\download_musetalk_models.ps1 first:\n" + details
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    inference_avatar = prepare_even_avatar(avatar, output)
    job, command, cwd = build_musetalk_job(
        inference_avatar, audio, output, use_fp16=use_fp16
    )
    config_path, _, generated_path = _job_paths(output)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(job, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"  [lipsync] MuseTalk 1.5: {avatar.name} + {audio.name} -> {output.name}")
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=build_musetalk_environment(),
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("MuseTalk inference timed out after 30 minutes") from exc

    log_path = output.with_suffix(".musetalk.log")
    log_path.write_text(
        f"COMMAND: {' '.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        if "CUDA out of memory" in detail:
            raise RuntimeError(
                "MuseTalk ran out of GPU memory. Close other GPU programs and retry "
                f"with FP16 enabled. Full log: {log_path}"
            )
        raise RuntimeError(f"MuseTalk failed: {detail}\nFull log: {log_path}")

    # Upstream catches some task exceptions, so a zero exit code is not sufficient.
    _validate_talking_video(generated_path)
    if generated_path != output:
        shutil.copy2(generated_path, output)
    _validate_talking_video(output)
    print(f"  [lipsync] Generated: {output}")
    return output


def preprocess_avatar(
    avatar_path: str,
    cache_dir: Optional[str] = None,
    use_fp16: bool = True,
) -> Path:
    """Validate an avatar for compatibility with the legacy pipeline call."""
    del cache_dir, use_fp16
    avatar = Path(avatar_path).resolve()
    if not avatar.is_file():
        raise FileNotFoundError(f"Avatar image does not exist: {avatar}")
    return avatar


if __name__ == "__main__":
    generate_lipsync(
        str(PROJECT_ROOT / "avatar" / "avatar.jpg"),
        str(PROJECT_ROOT / "outputs" / "acceptance" / "audio.wav"),
        str(PROJECT_ROOT / "outputs" / "acceptance" / "talking.mp4"),
    )
