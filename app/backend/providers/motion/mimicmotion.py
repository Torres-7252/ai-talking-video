"""Host-side adapter for the isolated MimicMotion runtime."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from app.backend.providers.media_utils import validate_video


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MIMICMOTION_ROOT = (
    PROJECT_ROOT / "app" / "backend" / "providers" / "motion" / "MimicMotion"
)
MIMICMOTION_PYTHON = PROJECT_ROOT / ".venv-mimicmotion" / "Scripts" / "python.exe"
MIMICMOTION_RUNNER = PROJECT_ROOT / "scripts" / "mimicmotion_runner.py"


def missing_mimicmotion_files(
    *,
    runtime_root: Path = MIMICMOTION_ROOT,
    python_executable: Path = MIMICMOTION_PYTHON,
    runner_path: Path = MIMICMOTION_RUNNER,
) -> list[Path]:
    runtime = Path(runtime_root).resolve()
    required = [
        Path(python_executable).resolve(),
        Path(runner_path).resolve(),
        runtime / "inference.py",
        runtime / "mimicmotion" / "utils" / "loader.py",
        runtime / "models" / "MimicMotion_1-1.pth",
        runtime / "models" / "DWPose" / "yolox_l.onnx",
        runtime / "models" / "DWPose" / "dw-ll_ucoco_384.onnx",
        runtime
        / "models"
        / "stable-video-diffusion-img2vid-xt-1-1"
        / "model_index.json",
    ]
    return [path for path in required if not path.is_file()]


def build_mimicmotion_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",
        }
    )
    return environment


def build_mimicmotion_command(
    image_path: Path,
    driver_path: Path,
    output_path: Path,
    *,
    duration: float,
    intensity: float,
    resolution: int = 576,
    runtime_root: Path = MIMICMOTION_ROOT,
    python_executable: Path = MIMICMOTION_PYTHON,
    runner_path: Path = MIMICMOTION_RUNNER,
) -> tuple[list[str], Path]:
    runtime = Path(runtime_root).resolve()
    command = [
        str(Path(python_executable).resolve()),
        str(Path(runner_path).resolve()),
        "--runtime",
        str(runtime),
        "--image",
        str(Path(image_path).resolve()),
        "--driver",
        str(Path(driver_path).resolve()),
        "--output",
        str(Path(output_path).resolve()),
        "--duration",
        str(float(duration)),
        "--intensity",
        str(float(intensity)),
        "--resolution",
        str(int(resolution)),
        "--fps",
        "15",
        "--tile-size",
        "16",
        "--tile-overlap",
        "6",
        "--steps",
        "25",
        "--decode-chunk-size",
        "1",
    ]
    return command, runtime


def validate_gesture_video(
    path: Path, *, expected_duration: float, fps: int = 15
) -> dict:
    info = validate_video(Path(path).resolve())
    video_stream = next(
        (stream for stream in info["streams"] if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise RuntimeError(f"MimicMotion output has no video stream: {path}")
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise RuntimeError("MimicMotion output must have positive even dimensions")
    rate = str(video_stream.get("r_frame_rate") or "0/1").split("/", 1)
    actual_fps = float(rate[0]) / max(float(rate[1]), 1.0)
    if abs(actual_fps - fps) > 0.1:
        raise RuntimeError(f"MimicMotion output must use {fps} fps, got {actual_fps:g}")
    if abs(float(info["duration"]) - expected_duration) > max(0.2, 1.5 / fps):
        raise RuntimeError(
            "MimicMotion output duration mismatch: "
            f"expected {expected_duration:.3f}s, got {float(info['duration']):.3f}s"
        )
    return info


def generate_gesture_motion(
    image_path: str,
    driver_video_path: str,
    output_path: str,
    duration: float,
    intensity: float = 0.25,
    *,
    timeout_seconds: int = 7200,
    runtime_root: Path = MIMICMOTION_ROOT,
    python_executable: Path = MIMICMOTION_PYTHON,
    runner_path: Path = MIMICMOTION_RUNNER,
) -> Path:
    """Generate body motion, retrying one CUDA OOM at lower resolution."""
    if duration <= 0:
        raise ValueError("Gesture duration must be positive")
    if not 0.0 <= intensity <= 1.0:
        raise ValueError("Gesture intensity must be between 0.0 and 1.0")

    image = Path(image_path).resolve()
    driver = Path(driver_video_path).resolve()
    output = Path(output_path).resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Avatar image does not exist: {image}")
    if not driver.is_file():
        raise FileNotFoundError(f"Gesture driver does not exist: {driver}")

    missing = missing_mimicmotion_files(
        runtime_root=runtime_root,
        python_executable=python_executable,
        runner_path=runner_path,
    )
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "MimicMotion runtime is incomplete. Run "
            "scripts\\install_mimicmotion_runtime.ps1 first:\n" + details
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.mimicmotion.tmp{output.suffix}")
    if temporary.exists():
        temporary.unlink()
    attempts: list[str] = []
    log_path = output.with_suffix(".mimicmotion.log")

    for attempt, resolution in enumerate((576, 448), 1):
        command, cwd = build_mimicmotion_command(
            image,
            driver,
            temporary,
            duration=duration,
            intensity=intensity,
            resolution=resolution,
            runtime_root=runtime_root,
            python_executable=python_executable,
            runner_path=runner_path,
        )
        print(
            f"  [motion] MimicMotion {resolution}p ({intensity:.2f}) "
            f"attempt {attempt} -> {output.name}"
        )
        try:
            result = subprocess.run(
                command,
                cwd=str(cwd),
                env=build_mimicmotion_environment(),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"MimicMotion inference timed out after {timeout_seconds} seconds"
            ) from exc

        attempts.append(
            f"ATTEMPT {attempt} ({resolution}p)\nCOMMAND: {' '.join(command)}"
            f"\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n"
        )
        combined = f"{result.stdout}\n{result.stderr}"
        if result.returncode == 0:
            break
        if attempt == 1 and "cuda out of memory" in combined.lower():
            if temporary.exists():
                temporary.unlink()
            continue
        log_path.write_text("\n".join(attempts), encoding="utf-8")
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"MimicMotion failed: {detail}\nFull log: {log_path}")
    else:
        raise RuntimeError("MimicMotion failed without completing an inference attempt")

    log_path.write_text("\n".join(attempts), encoding="utf-8")
    if not temporary.is_file():
        raise RuntimeError(f"MimicMotion did not create its output. Full log: {log_path}")
    validate_gesture_video(temporary, expected_duration=duration, fps=15)
    os.replace(temporary, output)
    print(f"  [motion] Generated: {output}")
    return output
