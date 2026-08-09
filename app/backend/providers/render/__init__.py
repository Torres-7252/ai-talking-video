"""FFmpeg-only renderer for the final landscape talking video."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.backend.providers.media_utils import run_checked, validate_video
from app.backend.providers.subtitle import write_ass


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _ffmpeg_filter_path(path: Path) -> str:
    value = path.resolve().as_posix()
    return value.replace("'", r"\'").replace(":", r"\:")


def build_render_command(
    talking_video: Path,
    subtitle_ass: Path,
    output_path: Path,
    *,
    width: int = 1920,
    height: int = 1080,
    fps: int = 25,
) -> list[str]:
    """Build the deterministic FFmpeg command used for final output."""
    video_filter = ",".join(
        [
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
            f"ass=filename='{_ffmpeg_filter_path(subtitle_ass)}'",
            "format=yuv420p",
            "setparams=range=limited",
        ]
    )
    return [
        "ffmpeg",
        "-y",
        "-v",
        "warning",
        "-i",
        str(Path(talking_video).resolve()),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-x264-params",
        "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-r",
        str(fps),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(Path(output_path).resolve()),
    ]


def render_video(
    talking_video: str,
    subtitle_json: str,
    output_path: str,
    title: Optional[str] = None,
    logo_path: Optional[str] = None,
    bgm_path: Optional[str] = None,
    broll_paths: Optional[list] = None,
    image_paths: Optional[list] = None,
    width: int = 1920,
    height: int = 1080,
    fps: int = 25,
    template: str = "talking_head",
) -> Path:
    """Scale, pad, subtitle, and encode a MuseTalk output with FFmpeg."""
    del title, logo_path, bgm_path, broll_paths, image_paths, template

    talking = Path(talking_video).resolve()
    subtitle_path = Path(subtitle_json).resolve()
    output = Path(output_path).resolve()
    if not talking.is_file():
        raise FileNotFoundError(f"Talking video does not exist: {talking}")
    if not subtitle_path.is_file():
        raise FileNotFoundError(f"Subtitle JSON does not exist: {subtitle_path}")

    subtitles = json.loads(subtitle_path.read_text(encoding="utf-8"))
    if not isinstance(subtitles, list) or not subtitles:
        raise RuntimeError(f"Subtitle JSON has no segments: {subtitle_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    ass_path = output.with_suffix(".ass")
    write_ass(subtitles, ass_path, width=width, height=height)
    command = build_render_command(
        talking, ass_path, output, width=width, height=height, fps=fps
    )
    print(f"  [render] FFmpeg landscape render: {output.name}")
    run_checked(command, timeout=1800)
    info = validate_video(output, expected_size=(width, height))
    if not any(stream.get("codec_type") == "audio" for stream in info["streams"]):
        raise RuntimeError(f"Rendered video has no audio stream: {output}")
    print(f"  [render] Generated: {output}")
    return output


if __name__ == "__main__":
    render_video(
        str(PROJECT_ROOT / "outputs" / "acceptance" / "talking.mp4"),
        str(PROJECT_ROOT / "outputs" / "acceptance" / "subtitle.json"),
        str(PROJECT_ROOT / "outputs" / "acceptance" / "final.mp4"),
    )
