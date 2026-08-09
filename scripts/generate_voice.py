#!/usr/bin/env python3
"""
独立脚本 - 生成声音
用法: python scripts/generate_voice.py --text "你的文案" --output ./outputs/test/audio.wav
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "backend" / "providers"))

import argparse
from voice import generate_voice

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="口播文案")
    parser.add_argument("--output", default="./outputs/voice_test/audio.wav")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    generate_voice(args.text, args.output, args.profile, args.speed)
