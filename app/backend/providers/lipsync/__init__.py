#!/usr/bin/env python3
"""MuseTalk 嘴型同步模块 - lipsync_provider"""

import sys
import subprocess
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# MuseTalk 路径
MUSETALK_PATH = PROJECT_ROOT / "app" / "backend" / "providers" / "lipsync" / "MuseTalk"
if MUSETALK_PATH.exists():
    sys.path.insert(0, str(MUSETALK_PATH))


def generate_lipsync(
    avatar_path: str,
    audio_path: str,
    output_path: str,
    use_fp16: bool = True,
    avatar_cache_dir: Optional[str] = None,
) -> Path:
    """
    使用 MuseTalk 驱动人物视频进行嘴型同步。

    Args:
        avatar_path: 人物视频路径
        audio_path: 音频文件路径 (wav)
        output_path: 输出视频路径 (mp4)
        use_fp16: 是否使用 FP16 加速
        avatar_cache_dir: 人物预处理缓存目录

    Returns:
        生成的嘴型同步视频路径
    """
    avatar = Path(avatar_path)
    audio = Path(audio_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not avatar.exists():
        raise FileNotFoundError(f"人物视频不存在: {avatar}")
    if not audio.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio}")

    if avatar_cache_dir is None:
        avatar_cache_dir = str(PROJECT_ROOT / "avatar" / "cache")
    cache = Path(avatar_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # 构建 MuseTalk 推理命令
    muse_script = MUSETALK_PATH / "scripts" / "inference.py"

    cmd = [
        sys.executable,
        str(muse_script),
        "--avatar", str(avatar),
        "--audio", str(audio),
        "--output", str(output),
        "--cache_dir", str(cache),
    ]

    if use_fp16:
        cmd.append("--fp16")

    print(f"  [lipsync] MuseTalk 推理: {avatar.name} + {audio.name} → {output.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
        if "CUDA out of memory" in error_msg:
            raise RuntimeError(
                f"MuseTalk failed: CUDA out of memory\n"
                f"Possible solution:\n"
                f"  1. Close other GPU programs\n"
                f"  2. Enable FP16\n"
                f"  3. Reduce batch size\n"
                f"  4. Retry from lipsync stage"
            )
        raise RuntimeError(f"MuseTalk failed: {error_msg}")

    if not output.exists():
        raise RuntimeError(f"MuseTalk 输出文件未生成: {output}")

    print(f"  [lipsync] 生成完成: {output}")
    return output


def preprocess_avatar(
    avatar_path: str,
    cache_dir: Optional[str] = None,
    use_fp16: bool = True,
) -> Path:
    """
    预处理人物视频，生成缓存数据（仅第一次需要）。

    Args:
        avatar_path: 人物视频路径
        cache_dir: 缓存目录
        use_fp16: 是否使用 FP16

    Returns:
        缓存目录路径
    """
    avatar = Path(avatar_path)
    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "avatar" / "cache"
    cache = Path(cache_dir)

    if not avatar.exists():
        raise FileNotFoundError(f"人物视频不存在: {avatar}")

    cache.mkdir(parents=True, exist_ok=True)

    # 检查是否已有缓存
    if (cache / "prepared.ready").exists():
        print(f"  [lipsync] 人物缓存已存在，跳过预处理")
        return cache

    print(f"  [lipsync] 首次预处理人物: {avatar.name}")
    muse_script = MUSETALK_PATH / "scripts" / "preprocess.py"

    cmd = [
        sys.executable,
        str(muse_script),
        "--avatar", str(avatar),
        "--cache_dir", str(cache),
    ]

    if use_fp16:
        cmd.append("--fp16")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        raise RuntimeError(f"MuseTalk 预处理失败: {result.stderr}")

    # 标记预处理完成
    (cache / "prepared.ready").touch()
    print(f"  [lipsync] 预处理完成")
    return cache


if __name__ == "__main__":
    # 独立测试
    avatar = PROJECT_ROOT / "avatar" / "avatar.mp4"
    audio = PROJECT_ROOT / "outputs" / "test_voice" / "audio.wav"
    output = PROJECT_ROOT / "outputs" / "test_lipsync" / "talking.mp4"

    if not avatar.exists():
        print(f"请先准备人物视频: {avatar}")
        sys.exit(1)

    # 首次预处理
    preprocess_avatar(str(avatar))

    # 生成嘴型同步
    if audio.exists():
        generate_lipsync(str(avatar), str(audio), str(output))
    else:
        print(f"音频文件不存在: {audio}")
