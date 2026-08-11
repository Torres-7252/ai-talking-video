"""MuseTalk 1.5 adapter for local talking-head generation."""

from __future__ import annotations

import hashlib
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml
import cv2
import numpy as np
from PIL import Image

from app.backend.providers.media_utils import validate_audio, validate_video


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MUSETALK_PATH = PROJECT_ROOT / "app" / "backend" / "providers" / "lipsync" / "MuseTalk"
COMPAT_PATH = PROJECT_ROOT / "app" / "backend" / "providers" / "lipsync" / "compat"
MUSETALK_JOBS_PATH = PROJECT_ROOT / ".runtime" / "musetalk"
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
    output = Path(output_path).resolve()
    job_id = hashlib.sha256(str(output).encode("utf-8")).hexdigest()[:20]
    work_dir = MUSETALK_JOBS_PATH / job_id
    config_path = work_dir / "inference.yaml"
    result_dir = work_dir / "results"
    generated_path = result_dir / "v15" / output.name
    return config_path, result_dir, generated_path


def stage_musetalk_input(source_path: Path, work_dir: Path, name: str) -> Path:
    """Copy an input to an ASCII-only path for OpenCV on Windows."""
    source = Path(source_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MuseTalk input does not exist: {source}")
    suffix = source.suffix.lower()
    if not suffix.isascii():
        raise ValueError(f"MuseTalk input extension must be ASCII: {source.suffix}")
    staged = Path(work_dir).resolve() / "inputs" / f"{name}{suffix}"
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, staged)
    return staged


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
        "--saved_coord",
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


def _video_frame_rate(value: object) -> float:
    text = str(value or "0")
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(text)


def build_mouth_detail_mask(
    frame_shape: tuple[int, int],
    face_box: tuple[int, int, int, int],
    strength: float = 0.35,
) -> np.ndarray:
    """Build a soft mask that restores source detail around the tracked mouth."""
    height, width = frame_shape
    x1, y1, x2, y2 = (int(value) for value in face_box)
    face_width = max(1, x2 - x1)
    face_height = max(1, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = y1 + face_height * 0.72
    sigma_x = face_width * 0.45
    sigma_y = face_height * 0.19
    y_grid, x_grid = np.ogrid[:height, :width]
    distance = (
        ((x_grid - center_x) / sigma_x) ** 2
        + ((y_grid - center_y) / sigma_y) ** 2
    )
    mask = (float(strength) * np.exp(-0.5 * distance)).astype(np.float32)
    mask[mask < 0.001] = 0.0
    return mask[:, :, np.newaxis]


def blend_mouth_detail(
    generated_frame: np.ndarray,
    source_frame: np.ndarray,
    face_box: tuple[int, int, int, int],
    strength: float = 0.35,
) -> np.ndarray:
    """Restore a restrained amount of source lip texture in one output frame."""
    if generated_frame.shape != source_frame.shape:
        raise ValueError("Generated and source frames must have matching dimensions")
    frame_height, frame_width = generated_frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in face_box)
    face_width = max(1, x2 - x1)
    face_height = max(1, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = y1 + face_height * 0.72
    radius_x = face_width * 0.45 * 3.5
    radius_y = face_height * 0.19 * 3.5
    left = max(0, int(center_x - radius_x))
    right = min(frame_width, int(center_x + radius_x) + 1)
    top = max(0, int(center_y - radius_y))
    bottom = min(frame_height, int(center_y + radius_y) + 1)
    local_box = (x1 - left, y1 - top, x2 - left, y2 - top)
    mask = build_mouth_detail_mask(
        (bottom - top, right - left), local_box, strength
    )
    generated_region = generated_frame[top:bottom, left:right].astype(np.float32)
    source_region = source_frame[top:bottom, left:right].astype(np.float32)
    blended_region = generated_region * (1.0 - mask) + source_region * mask
    output = generated_frame.copy()
    output[top:bottom, left:right] = np.clip(
        np.rint(blended_region), 0, 255
    ).astype(np.uint8)
    return output


def restore_mouth_detail(
    generated_path: Path,
    source_path: Path,
    coord_path: Path,
    output_path: Path,
    strength: float = 0.35,
) -> Path:
    """Blend tracked source lip texture into a generated video and retain audio."""
    generated = Path(generated_path).resolve()
    source = Path(source_path).resolve()
    coordinates = Path(coord_path).resolve()
    output = Path(output_path).resolve()
    if not coordinates.is_file():
        raise FileNotFoundError(f"MuseTalk face coordinates do not exist: {coordinates}")
    with coordinates.open("rb") as coord_file:
        coord_list = pickle.load(coord_file)
    if not isinstance(coord_list, list) or not coord_list:
        raise RuntimeError(f"MuseTalk face coordinates are invalid: {coordinates}")

    generated_capture = cv2.VideoCapture(str(generated))
    source_capture = cv2.VideoCapture(str(source))
    if not generated_capture.isOpened() or not source_capture.isOpened():
        generated_capture.release()
        source_capture.release()
        raise RuntimeError("Could not open generated and source videos for mouth refinement")

    width = int(generated_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(generated_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(generated_capture.get(cv2.CAP_PROP_FPS) or 25.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded_output = output
    if output == generated:
        encoded_output = output.with_name(f".{output.stem}.refined.mp4")
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "pipe:0",
        "-i", str(generated),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
        "-movflags", "+faststart", str(encoded_output),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    coord_cycle = coord_list + coord_list[::-1]
    frame_index = 0
    last_source_frame = None
    try:
        while True:
            generated_ok, generated_frame = generated_capture.read()
            if not generated_ok:
                break
            source_ok, source_frame = source_capture.read()
            if source_ok:
                last_source_frame = source_frame
            elif last_source_frame is not None:
                source_frame = last_source_frame
            else:
                raise RuntimeError("Source motion video contains no readable frames")
            if generated_frame.shape != source_frame.shape:
                raise RuntimeError("Generated and source video dimensions do not match")
            face_box = coord_cycle[frame_index % len(coord_cycle)]
            if len(face_box) == 4 and face_box[2] > face_box[0] and face_box[3] > face_box[1]:
                generated_frame = blend_mouth_detail(
                    generated_frame, source_frame, face_box, strength
                )
            if process.stdin is None:
                raise RuntimeError("FFmpeg mouth refinement pipe is unavailable")
            process.stdin.write(generated_frame.tobytes())
            frame_index += 1
        if process.stdin is not None:
            process.stdin.close()
            process.stdin = None
        return_code = process.wait(timeout=300)
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if return_code != 0:
            raise RuntimeError(f"FFmpeg mouth refinement failed: {stderr.strip()}")
    except Exception:
        if process.poll() is None:
            process.kill()
        raise
    finally:
        generated_capture.release()
        source_capture.release()
        if process.stderr is not None:
            process.stderr.close()
    if encoded_output != output:
        encoded_output.replace(output)
    return output


def prepare_lipsync_input(input_path: Path, output_path: Path) -> Path:
    """Validate a motion video or normalize a still portrait for MuseTalk."""
    source = Path(input_path).resolve()
    if source.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
        return prepare_even_avatar(source, output_path)

    info = validate_video(source)
    stream = next(
        item for item in info["streams"] if item.get("codec_type") == "video"
    )
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width % 2 or height % 2:
        raise RuntimeError(
            f"MuseTalk video input must have even dimensions, got {width}x{height}"
        )
    actual_fps = _video_frame_rate(stream.get("r_frame_rate"))
    if abs(actual_fps - 25.0) > 0.01:
        raise RuntimeError(
            f"MuseTalk video input must be 25 fps, got {actual_fps:g} fps"
        )
    return source


def finalize_lipsync_output(
    generated_path: Path,
    inference_avatar: Path,
    result_dir: Path,
    output_path: Path,
    mouth_detail_strength: float = 0.35,
) -> Path:
    """Refine dynamic input or copy the official output for still portraits."""
    generated = Path(generated_path).resolve()
    avatar = Path(inference_avatar).resolve()
    output = Path(output_path).resolve()
    if avatar.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        coord_path = Path(result_dir).resolve().parent / f"{avatar.stem}.pkl"
        return restore_mouth_detail(
            generated,
            avatar,
            coord_path,
            output,
            strength=mouth_detail_strength,
        )
    if generated != output:
        shutil.copy2(generated, output)
    return output


def generate_lipsync(
    avatar_path: str,
    audio_path: str,
    output_path: str,
    use_fp16: bool = True,
    avatar_cache_dir: Optional[str] = None,
    mouth_detail_strength: float = 0.35,
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
    inference_avatar = prepare_lipsync_input(avatar, output)
    config_path, result_dir, generated_path = _job_paths(output)
    if result_dir.exists():
        shutil.rmtree(result_dir)
    staged_avatar = stage_musetalk_input(
        inference_avatar, config_path.parent, "source"
    )
    staged_audio = stage_musetalk_input(audio, config_path.parent, "audio")
    job, command, cwd = build_musetalk_job(
        staged_avatar, staged_audio, output, use_fp16=use_fp16
    )
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
    finalize_lipsync_output(
        generated_path,
        staged_avatar,
        result_dir,
        output,
        mouth_detail_strength,
    )
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
