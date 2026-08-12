#!/usr/bin/env python3
"""Generate a MuseTalk 1.5 talking-head clip."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.providers.lipsync import generate_lipsync
from scripts.output_paths import OUTPUTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a local MuseTalk 1.5 clip")
    parser.add_argument("--avatar", default=str(PROJECT_ROOT / "avatar" / "avatar.jpg"))
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default=str(OUTPUTS_ROOT / "lipsync_test" / "talking.mp4"))
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 inference")
    args = parser.parse_args()
    generate_lipsync(
        args.avatar,
        args.audio,
        args.output,
        use_fp16=not args.no_fp16,
    )


if __name__ == "__main__":
    main()
