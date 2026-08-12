#!/usr/bin/env python3
"""Generate local FunASR subtitles."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.providers.subtitle import generate_subtitles
from scripts.output_paths import OUTPUTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate timed subtitles with FunASR")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default=str(OUTPUTS_ROOT / "subtitle_test" / "subtitle.json"))
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--text", help="Known transcript text to use with ASR timing")
    args = parser.parse_args()

    output = Path(args.output)
    if output.suffix.lower() != ".json":
        output = output / "subtitle.json"
    generate_subtitles(
        audio_path=args.audio,
        output_json=str(output),
        output_srt=str(output.with_suffix(".srt")),
        device=args.device,
        transcript_text=args.text,
    )


if __name__ == "__main__":
    main()
