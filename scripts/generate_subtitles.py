#!/usr/bin/env python3
"""独立生成字幕"""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "backend" / "providers"))

import argparse
from pathlib import Path
from subtitle import generate_subtitles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default="./outputs/subtitle_test/")
    args = parser.parse_args()

    out_dir = Path(args.output)
    generate_subtitles(
        audio_path=args.audio,
        output_json=str(out_dir / "subtitle.json"),
        output_srt=str(out_dir / "subtitle.srt"),
    )
