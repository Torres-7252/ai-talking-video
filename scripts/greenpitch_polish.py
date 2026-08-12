"""Add restrained GreenPitch training overlays to a finished vertical talking video."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


WIDTH = 1080
HEIGHT = 1920
FPS = 25


def ass_time(seconds: float) -> str:
    centiseconds = round(seconds * 100)
    hours, centiseconds = divmod(centiseconds, 360_000)
    minutes, centiseconds = divmod(centiseconds, 6_000)
    whole_seconds, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def dialogue(start: float, end: float, style: str, text: str, layer: int = 0) -> str:
    return (
        f"Dialogue: {layer},{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{text}"
    )


def write_overlay(path: Path, duration: float) -> None:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {HEIGHT}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Brand,Microsoft YaHei,34,&H00F5FFF8,&H00F5FFF8,&H00152623,&H90152623,-1,0,0,0,100,100,1,0,3,10,0,7,0,0,0,1
Style: Eyebrow,Microsoft YaHei,24,&H00BAF7C8,&H00BAF7C8,&H00152623,&H00152623,-1,0,0,0,100,100,2,0,1,0,0,7,0,0,0,1
Style: Card,Microsoft YaHei,38,&H00FFFFFF,&H00FFFFFF,&H00152623,&HBB152623,-1,0,0,0,100,100,1,0,3,12,0,7,0,0,0,1
Style: Detail,Microsoft YaHei,27,&H00D8FCE2,&H00D8FCE2,&H00152623,&HBB152623,0,0,0,0,100,100,1,0,3,10,0,7,0,0,0,1
Style: Tactic,Microsoft YaHei,34,&H005AF09A,&H005AF09A,&H00152623,&H00152623,-1,0,0,0,100,100,1,0,1,2,0,7,0,0,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    lines = [
        dialogue(0, duration, "Brand", r"{\pos(86,86)\fad(250,250)}● 绿茵进化", 2),
        dialogue(0.4, 4.2, "Eyebrow", r"{\pos(86,166)\fad(180,180)}MIDFIELD INTELLIGENCE", 2),
        dialogue(0.7, 4.2, "Card", r"{\pos(86,222)\fad(220,220)}中场核心能力", 2),
        dialogue(10.2, 20.8, "Eyebrow", r"{\pos(86,348)\fad(180,180)}绿茵进化 · 专项训练 01", 2),
        dialogue(10.5, 20.8, "Card", r"{\pos(86,402)\fad(180,180)}视野扫描", 2),
        dialogue(10.8, 20.8, "Detail", r"{\pos(86,458)\fad(180,180)}接球前完成 2 次观察", 2),
        dialogue(14.2, 18.8, "Tactic", r"{\move(126,540,760,540,0,1100)\fad(100,150)}● ───────→ 空当", 3),
        dialogue(21.5, 41.2, "Eyebrow", r"{\pos(86,348)\fad(180,180)}绿茵进化 · 专项训练 02", 2),
        dialogue(21.8, 41.2, "Card", r"{\pos(86,402)\fad(180,180)}第一脚触球", 2),
        dialogue(22.1, 41.2, "Detail", r"{\pos(86,458)\fad(180,180)}接球 · 护球 · 转身摆脱", 2),
        dialogue(29.0, 34.2, "Tactic", r"{\move(160,540,670,540,0,900)\fad(100,160)}● ─────→ 转身", 3),
        dialogue(42.5, 62.7, "Eyebrow", r"{\pos(86,348)\fad(180,180)}绿茵进化 · 战术理解 03", 2),
        dialogue(42.8, 62.7, "Card", r"{\pos(86,402)\fad(180,180)}传球与位置感", 2),
        dialogue(43.1, 62.7, "Detail", r"{\pos(86,458)\fad(180,180)}横传 | 回传 | 向前", 2),
        dialogue(50.0, 56.0, "Tactic", r"{\move(150,540,770,540,0,1200)\fad(100,160)}● ───────→ 向前线路", 3),
        dialogue(63.5, 84.5, "Eyebrow", r"{\pos(86,348)\fad(180,180)}绿茵进化 · 比赛复盘 04", 2),
        dialogue(63.8, 84.5, "Card", r"{\pos(86,402)\fad(180,180)}阅读比赛节奏", 2),
        dialogue(64.1, 84.5, "Detail", r"{\pos(86,458)\fad(180,180)}何时进攻 · 何时控制", 2),
        dialogue(72.2, 78.0, "Tactic", r"{\move(160,540,690,540,0,1000)\fad(100,160)}● ─────→ 控制节奏", 3),
        dialogue(86.0, 99.4, "Eyebrow", r"{\pos(86,348)\fad(180,260)}绿茵进化 · 中场专项训练", 2),
        dialogue(86.3, 99.4, "Card", r"{\pos(86,402)\fad(180,260)}把理解，练成能力", 2),
        dialogue(86.6, 99.4, "Detail", r"{\pos(86,458)\fad(180,260)}微信搜索：绿茵进化", 2),
    ]
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8-sig")


def video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def main(source: str, destination: str) -> None:
    source_path = Path(source).resolve()
    output_path = Path(destination).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path = output_path.with_suffix(".ass")
    write_overlay(ass_path, video_duration(source_path))
    escaped_ass = ass_path.as_posix().replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    command = [
        "ffmpeg", "-y", "-v", "warning", "-i", str(source_path),
        "-vf", f"ass=filename='{escaped_ass}',format=yuv420p",
        "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-c:a", "copy", "-movflags", "+faststart", str(output_path),
    ]
    subprocess.run(command, check=True)
    print(output_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
