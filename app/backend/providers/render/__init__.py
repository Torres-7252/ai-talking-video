#!/usr/bin/env python3
"""HyperFrames 视频包装模块 - video_renderer

负责：字幕渲染、标题、Logo、B-roll、图片插入、背景音乐、片头片尾
不负责：数字人口型（由 MuseTalk 负责）
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def render_video(
    talking_video: str,
    subtitle_json: str,
    output_path: str,
    title: Optional[str] = None,
    logo_path: Optional[str] = None,
    bgm_path: Optional[str] = None,
    broll_paths: Optional[list] = None,
    image_paths: Optional[list] = None,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    template: str = "talking_head",
) -> Path:
    """
    视频包装渲染。

    Args:
        talking_video: MuseTalk 输出的嘴型同步视频
        subtitle_json: FunASR 输出的字幕 JSON
        output_path: 打包后视频输出路径
        title: 顶部标题文本
        logo_path: Logo 图片路径
        bgm_path: 背景音乐路径
        broll_paths: B-roll 视频路径列表
        image_paths: 图片路径列表
        width, height, fps: 视频规格
        template: 模板名称

    Returns:
        包装后的视频路径
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 加载字幕
    with open(subtitle_json, "r", encoding="utf-8") as f:
        subtitles = json.load(f)

    # 构建 FFmpeg 滤镜链
    # 第一版: 使用 FFmpeg drawtext + ass 字幕实现

    # 先尝试用 moviepy 方式
    try:
        return _render_with_moviepy(
            talking_video, subtitles, output_path,
            title, logo_path, bgm_path, broll_paths, image_paths,
            width, height, fps, template
        )
    except ImportError:
        # 回退到纯 FFmpeg
        return _render_with_ffmpeg(
            talking_video, subtitle_json, output_path,
            title, logo_path, bgm_path,
            width, height, fps
        )


def _render_with_moviepy(
    talking_video: str,
    subtitles: list,
    output_path: str,
    title: Optional[str],
    logo_path: Optional[str],
    bgm_path: Optional[str],
    broll_paths: Optional[list],
    image_paths: Optional[list],
    width: int,
    height: int,
    fps: int,
    template: str,
) -> Path:
    """使用 moviepy 渲染视频"""
    from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip
    from moviepy.video.tools.subtitles import SubtitlesClip
    from moviepy.video.fx import Resize

    # 加载 talking head 视频
    talking = VideoFileClip(talking_video)

    # 调整到目标尺寸
    # 数字人放在画面下半部分
    talking_resized = talking.resized(height=height * 0.55)
    talking_pos = ("center", height * 0.45)

    clips = [talking_resized.with_position(talking_pos)]

    # 标题
    if title:
        from moviepy import TextClip
        title_clip = TextClip(
            text=title,
            font_size=56,
            color="white",
            stroke_color="black",
            stroke_width=3,
            font="Arial",
            size=(width * 0.9, None),
            method="caption",
        ).with_position(("center", 40)).with_duration(talking.duration)
        clips.append(title_clip)

    # Logo
    if logo_path and Path(logo_path).exists():
        logo = ImageClip(logo_path).resized(width=120)
        logo = logo.with_position((width - 140, 60)).with_duration(talking.duration)
        clips.append(logo)

    # 字幕 (大字幕，居中偏下)
    if subtitles:
        subtitle_clips = _build_subtitle_clips(subtitles, width, height)
        clips.append(subtitle_clips)

    # 合成
    final = CompositeVideoClip(clips, size=(width, height))
    final = final.with_duration(talking.duration)

    # 背景音乐
    if bgm_path and Path(bgm_path).exists():
        bgm = AudioFileClip(bgm_path).with_duration(talking.duration)
        bgm = bgm.with_volume_scaled(0.15)  # BGM 音量降低
        from moviepy import CompositeAudioClip
        final_audio = CompositeAudioClip([talking.audio, bgm])
        final = final.with_audio(final_audio)

    # 导出
    final.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
    )

    talking.close()
    print(f"  [render] 视频包装完成: {output_path}")
    return Path(output_path)


def _build_subtitle_clips(subtitles: list, width: int, height: int):
    """构建字幕片段"""
    from moviepy import TextClip
    clips = []

    for sub in subtitles:
        text = sub["text"]
        start = sub["start"]
        end = sub["end"]
        duration = end - start

        # 主体字幕
        txt_clip = TextClip(
            text=text,
            font_size=42,
            color="white",
            stroke_color="black",
            stroke_width=2,
            font="Arial",
            size=(width * 0.85, None),
            method="caption",
        ).with_position(("center", height - 250)).with_start(start).with_duration(duration)

        clips.append(txt_clip)

    return clips


def _render_with_ffmpeg(
    talking_video: str,
    subtitle_srt: str,
    output_path: str,
    title: Optional[str],
    logo_path: Optional[str],
    bgm_path: Optional[str],
    width: int,
    height: int,
    fps: int,
) -> Path:
    """使用纯 FFmpeg 进行视频包装"""
    output = Path(output_path)

    # 构建滤镜链
    filters = []

    # 缩放数字人视频到竖屏
    filters.append(f"scale={width}:{height}:force_original_aspect_ratio=decrease")
    filters.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")

    # 字幕 (ass/srt)
    # ...

    filter_str = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(talking_video),
        "-vf", filter_str,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-r", str(fps),
        str(output),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 渲染失败: {result.stderr}")

    print(f"  [render] FFmpeg 包装完成: {output}")
    return output


if __name__ == "__main__":
    # 独立测试
    talking = PROJECT_ROOT / "outputs" / "test_lipsync" / "talking.mp4"
    sub_json = PROJECT_ROOT / "outputs" / "test_subtitle" / "subtitle.json"

    if talking.exists() and sub_json.exists():
        render_video(
            talking_video=str(talking),
            subtitle_json=str(sub_json),
            output_path=str(PROJECT_ROOT / "outputs" / "test_render" / "packaged.mp4"),
            title="测试视频 - AI足球口播",
            template="talking_head",
        )
    else:
        print(f"请先生成 talking.mp4 和 subtitle.json")
