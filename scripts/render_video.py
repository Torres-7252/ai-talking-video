#!/usr/bin/env python3
"""独立视频渲染"""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "backend" / "providers"))

import argparse
from pathlib import Path
from render import render_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--talking", required=True, help="MuseTalk 输出视频")
    parser.add_argument("--subtitles", required=True, help="FunASR JSON 字幕")
    parser.add_argument("--output", default="./outputs/render_test/packaged.mp4")
    parser.add_argument("--title", default="AI数字人口播")
    parser.add_argument("--logo", default="./assets/logo/logo.png")
    parser.add_argument("--bgm", default=None)
    parser.add_argument("--template", default="talking_head")
    args = parser.parse_args()

    render_video(
        talking_video=args.talking,
        subtitle_json=args.subtitles,
        output_path=args.output,
        title=args.title,
        logo_path=args.logo if Path(args.logo).exists() else None,
        bgm_path=args.bgm,
        template=args.template,
    )
