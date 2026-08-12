#!/usr/bin/env python3
"""Generate cloned speech with the local GPT-SoVITS provider."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.providers.voice import generate_voice
from scripts.output_paths import OUTPUTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local cloned speech")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default=str(OUTPUTS_ROOT / "voice_test" / "audio.wav"))
    parser.add_argument("--profile", default="default")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    generate_voice(args.text, args.output, args.profile, args.speed)


if __name__ == "__main__":
    main()
