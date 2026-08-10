"""LivePortrait adapter for restrained local portrait motion."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

from app.backend.providers.media_utils import run_checked, validate_audio, validate_video


PROJECT_ROOT = Path(__file__).resolve().parents[4]
LIVEPORTRAIT_ROOT = (
    PROJECT_ROOT / "app" / "backend" / "providers" / "motion" / "LivePortrait"
)
LIVEPORTRAIT_PYTHON = (
    PROJECT_ROOT / ".venv-liveportrait" / "Scripts" / "python.exe"
)
STEADY_TEMPLATE = PROJECT_ROOT / "motion" / "templates" / "steady.pkl"

_RUNTIME_RELATIVE_FILES = (
    "inference.py",
    "pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
    "pretrained_weights/liveportrait/base_models/motion_extractor.pth",
    "pretrained_weights/liveportrait/base_models/spade_generator.pth",
    "pretrained_weights/liveportrait/base_models/warping_module.pth",
    "pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth",
    "pretrained_weights/liveportrait/landmark.onnx",
    "pretrained_weights/insightface/models/buffalo_l/2d106det.onnx",
    "pretrained_weights/insightface/models/buffalo_l/det_10g.onnx",
)


def required_liveportrait_files(
    *,
    runtime_root: Path = LIVEPORTRAIT_ROOT,
    python_executable: Path = LIVEPORTRAIT_PYTHON,
    template_path: Path = STEADY_TEMPLATE,
) -> tuple[Path, ...]:
    runtime = Path(runtime_root).resolve()
    return (
        Path(python_executable).resolve(),
        *(runtime / relative for relative in _RUNTIME_RELATIVE_FILES),
        Path(template_path).resolve(),
    )


def missing_liveportrait_files(
    *,
    runtime_root: Path = LIVEPORTRAIT_ROOT,
    python_executable: Path = LIVEPORTRAIT_PYTHON,
    template_path: Path = STEADY_TEMPLATE,
) -> list[Path]:
    return [
        path
        for path in required_liveportrait_files(
            runtime_root=runtime_root,
            python_executable=python_executable,
            template_path=template_path,
        )
        if not path.is_file() or path.stat().st_size == 0
    ]


def build_liveportrait_command(
    source_path: Path,
    driving_path: Path,
    output_dir: Path,
    *,
    intensity: float = 0.35,
    runtime_root: Path = LIVEPORTRAIT_ROOT,
    python_executable: Path = LIVEPORTRAIT_PYTHON,
) -> tuple[list[str], Path, Path]:
    source = Path(source_path).resolve()
    driving = Path(driving_path).resolve()
    output = Path(output_dir).resolve()
    runtime = Path(runtime_root).resolve()
    python = Path(python_executable).resolve()
    command = [
        str(python),
        str(runtime / "inference.py"),
        "-s",
        str(source),
        "-d",
        str(driving),
        "-o",
        str(output),
        "--driving-multiplier",
        str(intensity),
        "--source-max-dim",
        "1920",
    ]
    expected = output / f"{source.stem}--{driving.stem}.mp4"
    return command, runtime, expected


def build_liveportrait_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _number(value: float) -> str:
    return str(round(float(value), 6))


def build_motion_loop_command(
    input_path: Path,
    output_path: Path,
    *,
    clip_duration: float,
    target_duration: float,
    fps: int = 25,
    fade_duration: float = 0.2,
) -> list[str]:
    if clip_duration <= 0 or target_duration <= 0:
        raise ValueError("Motion and target durations must be positive")
    if fade_duration < 0 or fade_duration >= clip_duration:
        raise ValueError("Motion fade duration must be shorter than the clip")
    if fps <= 0:
        raise ValueError("Motion fps must be positive")

    step_duration = clip_duration - fade_duration
    copies = max(1, math.ceil((target_duration - fade_duration) / step_duration))
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    for _ in range(copies):
        command.extend(["-i", str(Path(input_path).resolve())])

    filters = [
        f"[{index}:v]settb=AVTB,setpts=PTS-STARTPTS[v{index}in]"
        for index in range(copies)
    ]
    current = "v0in"
    for index in range(1, copies):
        output_label = f"v{index}"
        offset = step_duration * index
        filters.append(
            f"[{current}][v{index}in]xfade=transition=fade:"
            f"duration={_number(fade_duration)}:offset={_number(offset)}"
            f"[{output_label}]"
        )
        current = output_label

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            f"[{current}]",
            "-an",
            "-t",
            _number(target_duration),
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(Path(output_path).resolve()),
        ]
    )
    return command


def _frame_rate(value: object) -> float:
    text = str(value or "0")
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(text)


def validate_motion_video(
    path: Path,
    *,
    fps: int = 25,
    expected_duration: float | None = None,
) -> dict:
    video = Path(path).resolve()
    info = validate_video(video)
    stream = next(
        item for item in info["streams"] if item.get("codec_type") == "video"
    )
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width % 2 or height % 2:
        raise RuntimeError(
            f"Motion video must have even dimensions, got {width}x{height}: {video}"
        )
    actual_fps = _frame_rate(stream.get("r_frame_rate"))
    if abs(actual_fps - fps) > 0.01:
        raise RuntimeError(
            f"Motion video must be {fps} fps, got {actual_fps:g} fps: {video}"
        )
    if expected_duration is not None and abs(info["duration"] - expected_duration) > 0.1:
        raise RuntimeError(
            "Motion video duration does not match audio: "
            f"{info['duration']:.3f}s vs {expected_duration:.3f}s"
        )
    return info


def generate_motion(
    avatar_path: str,
    audio_path: str,
    output_path: str,
    *,
    style: str = "steady",
    intensity: float = 0.35,
    fps: int = 25,
    timeout_seconds: int = 1800,
    runtime_root: Path = LIVEPORTRAIT_ROOT,
    python_executable: Path = LIVEPORTRAIT_PYTHON,
    template_path: Path = STEADY_TEMPLATE,
) -> Path:
    if not 0.0 <= intensity <= 1.0:
        raise ValueError("Motion intensity must be between 0.0 and 1.0")
    if style != "steady":
        raise ValueError(f"Unsupported motion style: {style}")
    if fps != 25:
        raise ValueError("LivePortrait motion output must use 25 fps")

    avatar = Path(avatar_path).resolve()
    audio = Path(audio_path).resolve()
    output = Path(output_path).resolve()
    runtime = Path(runtime_root).resolve()
    python = Path(python_executable).resolve()
    template = Path(template_path).resolve()
    if not avatar.is_file():
        raise FileNotFoundError(f"Avatar image does not exist: {avatar}")

    missing = missing_liveportrait_files(
        runtime_root=runtime,
        python_executable=python,
        template_path=template,
    )
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "LivePortrait runtime is incomplete. Run "
            "scripts\\install_liveportrait_runtime.ps1 first:\n" + details
        )

    audio_info = validate_audio(audio)
    target_duration = float(audio_info["duration"])
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output.parent / ".liveportrait" / output.stem
    raw_dir = work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    command, cwd, generated = build_liveportrait_command(
        avatar,
        template,
        raw_dir,
        intensity=intensity,
        runtime_root=runtime,
        python_executable=python,
    )

    print(
        f"  [motion] LivePortrait: {avatar.name} + {style} "
        f"({intensity:.2f}) -> {output.name}"
    )
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=build_liveportrait_environment(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"LivePortrait inference timed out after {timeout_seconds} seconds"
        ) from exc

    log_path = output.with_suffix(".liveportrait.log")
    log_path.write_text(
        f"COMMAND: {' '.join(command)}\n\nSTDOUT:\n{result.stdout}"
        f"\n\nSTDERR:\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        if "CUDA out of memory" in detail:
            raise RuntimeError(
                "LivePortrait ran out of GPU memory. Close other GPU programs and "
                f"retry. Full log: {log_path}"
            )
        raise RuntimeError(f"LivePortrait failed: {detail}\nFull log: {log_path}")

    raw_info = validate_motion_video(generated, fps=fps)
    temporary_output = output.with_name(f"{output.stem}.tmp{output.suffix}")
    loop_command = build_motion_loop_command(
        generated,
        temporary_output,
        clip_duration=float(raw_info["duration"]),
        target_duration=target_duration,
        fps=fps,
    )
    run_checked(loop_command, timeout=timeout_seconds)
    validate_motion_video(
        temporary_output,
        fps=fps,
        expected_duration=target_duration,
    )
    os.replace(temporary_output, output)
    print(f"  [motion] Generated: {output}")
    return output


from app.backend.providers.motion.mimicmotion import (  # noqa: E402
    generate_gesture_motion,
    missing_mimicmotion_files,
)
