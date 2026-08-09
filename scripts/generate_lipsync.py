#!/usr/bin/env python3
"""独立生成嘴型同步视频"""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "backend" / "providers"))

import argparse
from pathlib import Path
from lipsync import generate_lipsync, preprocess_avatar

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--avatar", default="./avatar/avatar.mp4")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default="./outputs/lipsync_test/talking.mp4")
    parser.add_argument("--preprocess", action="store_true")
    args = parser.parse_args()

    if args.preprocess:
        preprocess_avatar(args.avatar)
    generate_lipsync(args.avatar, args.audio, args.output)
