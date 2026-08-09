#!/usr/bin/env python3
"""FunASR 字幕生成模块 - subtitle_provider"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def generate_subtitles(
    audio_path: str,
    output_json: str,
    output_srt: Optional[str] = None,
    model_name: Optional[str] = None,
    device: str = "cuda",
) -> dict:
    """
    使用 FunASR 对音频进行语音识别，生成字幕时间轴。

    Args:
        audio_path: 音频文件路径 (wav, 16kHz)
        output_json: JSON 字幕输出路径
        output_srt: SRT 字幕输出路径 (可选)
        model_name: FunASR 模型名称
        device: cuda 或 cpu

    Returns:
        dict: 字幕数据 (text, start, end, words)
    """
    from funasr import AutoModel

    if model_name is None:
        model_name = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    if output_srt is None:
        output_srt = str(Path(output_json).with_suffix(".srt"))

    audio_file = Path(audio_path)
    json_file = Path(output_json)
    srt_file = Path(output_srt)
    json_file.parent.mkdir(parents=True, exist_ok=True)

    if not audio_file.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_file}")

    print(f"  [subtitle] FunASR 识别中...")

    model = AutoModel(
        model=model_name,
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-pytorch",
        device=device,
    )

    result = model.generate(input=str(audio_file))

    # 解析结果，生成标准字幕结构
    subtitles = []
    if result and len(result) > 0:
        for segment in result[0].get("sentence_info", []):
            words = []
            for w in segment.get("words", []):
                words.append({
                    "text": w.get("text", ""),
                    "start": round(w.get("start", 0) / 1000, 2),
                    "end": round(w.get("end", 0) / 1000, 2),
                })

            sub = {
                "text": segment.get("text", ""),
                "start": round(segment.get("start", 0) / 1000, 2),
                "end": round(segment.get("end", 0) / 1000, 2),
                "words": words,
            }
            subtitles.append(sub)

    # 保存 JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(subtitles, f, ensure_ascii=False, indent=2)
    print(f"  [subtitle] JSON 字幕: {json_file}")

    # 生成 SRT
    _write_srt(subtitles, srt_file)
    print(f"  [subtitle] SRT 字幕: {srt_file}")

    return subtitles


def _write_srt(subtitles: list, output_path: Path):
    """将字幕数据写入 SRT 文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sub in enumerate(subtitles, 1):
            start_ts = _seconds_to_srt_time(sub["start"])
            end_ts = _seconds_to_srt_time(sub["end"])
            f.write(f"{i}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{sub['text']}\n\n")


def _seconds_to_srt_time(seconds: float) -> str:
    """秒数 → SRT 时间格式 HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


if __name__ == "__main__":
    # 独立测试
    audio = PROJECT_ROOT / "outputs" / "test_voice" / "audio.wav"
    if audio.exists():
        subtitles = generate_subtitles(
            audio_path=str(audio),
            output_json=str(PROJECT_ROOT / "outputs" / "test_subtitle" / "subtitle.json"),
        )
        print(f"识别到 {len(subtitles)} 段字幕")
    else:
        print(f"音频文件不存在: {audio}")
