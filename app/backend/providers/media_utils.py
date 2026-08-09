"""Shared subprocess and media validation helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional


def run_checked(
    command: list[str],
    *,
    cwd: Optional[Path] = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"Command failed ({result.returncode}): {detail}")
    return result


def probe_media(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Media file does not exist: {path}")

    result = run_checked(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,pix_fmt,sample_rate,channels",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        timeout=30,
    )
    payload = json.loads(result.stdout)
    format_info = payload.get("format", {})
    return {
        "duration": float(format_info.get("duration") or 0),
        "size": int(format_info.get("size") or path.stat().st_size),
        "streams": payload.get("streams", []),
    }


def validate_audio(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    info = probe_media(path)
    if info["duration"] <= 0 or info["size"] <= 44:
        raise RuntimeError(f"Audio is empty or invalid: {path}")
    if not any(stream.get("codec_type") == "audio" for stream in info["streams"]):
        raise RuntimeError(f"Media has no audio stream: {path}")
    return info


def validate_video(
    path: Path,
    expected_size: Optional[tuple[int, int]] = None,
) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file does not exist: {path}")
    info = probe_media(path)
    video_streams = [s for s in info["streams"] if s.get("codec_type") == "video"]
    if info["duration"] <= 0 or info["size"] <= 0 or not video_streams:
        raise RuntimeError(f"Video is empty or invalid: {path}")
    if expected_size:
        stream = video_streams[0]
        actual_size = (int(stream.get("width") or 0), int(stream.get("height") or 0))
        if actual_size != expected_size:
            raise RuntimeError(
                f"Unexpected video size {actual_size[0]}x{actual_size[1]}; "
                f"expected {expected_size[0]}x{expected_size[1]}"
            )
    return info
