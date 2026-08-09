#!/usr/bin/env python3
"""GPT-SoVITS 声音生成模块 - voice_provider (直接调用版)"""

import os
import sys
import soundfile as sf
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
SOVITS_ROOT = PROJECT_ROOT / "voice" / "models" / "GPT-SoVITS"
GPT_SOVITS_DIR = SOVITS_ROOT / "GPT_SoVITS"

# Inject paths - eres2net must come first so 'import pooling_layers' works
ERES2NET_DIR = GPT_SOVITS_DIR / "eres2net"
sys.path.insert(0, str(ERES2NET_DIR))
sys.path.insert(0, str(GPT_SOVITS_DIR))
sys.path.insert(0, str(SOVITS_ROOT))


def _find_ref_audio(voice_profile: str = "default") -> str:
    """查找参考音频文件"""
    for ext in [".wav", ".mp3", ".m4a", ".flac"]:
        path = PROJECT_ROOT / "voice" / "references" / f"{voice_profile}{ext}"
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"参考音频不存在: voice/references/{voice_profile}.[wav|mp3|m4a]")


# Module-level cache for TTS instance
_tts_instance = None


def _get_tts():
    """获取或初始化 TTS 实例 (单例)"""
    global _tts_instance
    if _tts_instance is None:
        os.environ.setdefault("LANGDETECT_CACHE", str(GPT_SOVITS_DIR / "pretrained_models" / "fast_langdetect"))
        from TTS_infer_pack.TTS import TTS, TTS_Config
        cfg = TTS_Config(str(GPT_SOVITS_DIR / "configs" / "tts_infer.yaml"))
        _tts_instance = TTS(cfg)
        print("  [voice] TTS model loaded")
    return _tts_instance


def generate_voice(
    text: str,
    output_path: str,
    voice_profile: str = "default",
    speed: float = 1.0,
    api_url: Optional[str] = None,
) -> Path:
    """
    使用 GPT-SoVITS 直接生成中文 TTS 音频 (不再走 HTTP API)
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ref_audio = _find_ref_audio(voice_profile)
    tts = _get_tts()

    print(f"  [voice] Generating: {text[:30]}... -> {output.name}")
    gen = tts.run({
        "text": text,
        "text_lang": "zh",
        "ref_audio_path": ref_audio,
        "prompt_lang": "zh",
        "prompt_text": text[:60],
        "text_split_method": "cut5",
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": False,
        "speed_factor": speed,
    })

    outputs = list(gen)
    if not outputs:
        raise RuntimeError("TTS 生成失败：无输出")

    sr, audio = outputs[0]
    sf.write(str(output), audio, sr)

    size_kb = output.stat().st_size / 1024
    duration = len(audio) / sr
    print(f"  [voice] Done: {output} ({size_kb:.0f}KB, {duration:.1f}s)")
    return output


def unload_voice():
    """释放 GPU 显存"""
    global _tts_instance
    if _tts_instance is not None:
        del _tts_instance
        _tts_instance = None
        import torch
        torch.cuda.empty_cache()
        print("  [voice] Model unloaded")


if __name__ == "__main__":
    output = generate_voice(
        text="这是我的第一条AI足球口播视频。",
        output_path=str(PROJECT_ROOT / "outputs" / "test_voice_direct" / "audio.wav"),
    )
    print(f"Test complete: {output}")
