#!/usr/bin/env python3
"""Render the final 1920x1080 subtitled video."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.providers.render import render_video
from scripts.output_paths import OUTPUTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a landscape talking video")
    parser.add_argument("--video", "--talking", dest="video", required=True)
    parser.add_argument("--subtitles", required=True, help="Timed subtitle JSON")
    parser.add_argument("--output", default=str(OUTPUTS_ROOT / "render_test" / "final.mp4"))
    args = parser.parse_args()
    render_video(args.video, args.subtitles, args.output)


if __name__ == "__main__":
    main()
