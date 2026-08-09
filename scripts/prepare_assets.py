#!/usr/bin/env python3
"""Prepare the approved avatar image and GPT-SoVITS reference audio."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROVIDERS_ROOT = PROJECT_ROOT / "app" / "backend" / "providers"
sys.path.insert(0, str(PROVIDERS_ROOT))

from media_utils import validate_audio


REFERENCE_TEXT = "大家好，我是AI足球教练。今天我们来聊一个很多球友都关心的问题。"


def prepare_assets(
    image_source: Path,
    voice_source: Path,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Path]:
    image_source = Path(image_source).resolve()
    voice_source = Path(voice_source).resolve()
    if not image_source.is_file():
        raise FileNotFoundError(f"Avatar image does not exist: {image_source}")
    if not voice_source.is_file():
        raise FileNotFoundError(f"Reference recording does not exist: {voice_source}")

    avatar_path = project_root / "avatar" / "avatar.jpg"
    reference_path = project_root / "voice" / "references" / "default.wav"
    profile_path = project_root / "voice" / "references" / "default.json"
    avatar_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_source, avatar_path)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        "1.54",
        "-to",
        "7.73",
        "-i",
        str(voice_source),
        "-ac",
        "1",
        "-ar",
        "32000",
        "-c:a",
        "pcm_s16le",
        str(reference_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg reference conversion failed")

    validate_audio(reference_path)
    profile_path.write_text(
        json.dumps(
            {"reference_text": REFERENCE_TEXT, "language": "zh"},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "avatar": avatar_path,
        "reference_audio": reference_path,
        "voice_profile": profile_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare local talking-video assets")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True)
    args = parser.parse_args()

    outputs = prepare_assets(args.image, args.voice)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
