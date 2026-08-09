#!/usr/bin/env python3
"""环境检查脚本 - 检查所有必需的依赖"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def check(name, ok_msg, fail_msg, fix_hint=None):
    try:
        result = ok_msg() if callable(ok_msg) else ok_msg
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [ERROR] {name}: {fail_msg if isinstance(fail_msg, str) else fail_msg()}")
        print(f"         原因: {e}")
        if fix_hint:
            print(f"         解决: {fix_hint}")
        return False

def main():
    print("=" * 60)
    print("  AI数字人口播视频生成器 - 环境检查")
    print("=" * 60)
    results = {}

    # 1. Python
    print("\n[基础环境]")
    results["python"] = check(
        "Python", lambda: sys.version.split()[0],
        f"需要 Python >= 3.10, 当前: {sys.version_info.major}.{sys.version_info.minor}",
        "从 https://python.org 下载安装 Python 3.10+"
    )

    # 2. Node.js
    try:
        ver = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
        results["node"] = check("Node.js", ver, "Command not found", "从 https://nodejs.org 安装")
    except FileNotFoundError:
        results["node"] = check("Node.js", None, "Command not found", "从 https://nodejs.org 安装")

    # 3. NVIDIA GPU
    print("\n[GPU 环境]")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        results["nvidia_gpu"] = check("NVIDIA GPU", out, "nvidia-smi 不可用")
    except FileNotFoundError:
        results["nvidia_gpu"] = check("NVIDIA GPU", None, "nvidia-smi 不可用",
                                       "安装 NVIDIA 驱动: https://www.nvidia.com/download/")

    # 4. CUDA via PyTorch
    print("\n[CUDA / PyTorch]")
    try:
        import torch
        tor_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            results["torch_cuda"] = check("PyTorch CUDA",
                                          f"{tor_ver}, GPU: {gpu_name}",
                                          f"CUDA 不可用")
        else:
            results["torch_cuda"] = False
            print(f"  [ERROR] PyTorch CUDA: {tor_ver} 但 CUDA 不可用")
            print(f"         解决: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    except ImportError:
        results["torch_cuda"] = False
        print("  [ERROR] PyTorch 未安装")
        print("         解决: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")

    # 5. CUDA Toolkit
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True,
                           timeout=10).stdout.strip()
        ver_line = [l for l in out.split("\n") if "release" in l.lower()]
        results["cuda_toolkit"] = check("CUDA Toolkit", ver_line[0] if ver_line else out,
                                        "nvcc 不可用", "可选依赖: 从 NVIDIA 官网安装 CUDA Toolkit")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        results["cuda_toolkit"] = check("CUDA Toolkit", None, "nvcc 未找到",
                                         "可选: 从 https://developer.nvidia.com/cuda-downloads 安装")

    # 6. FFmpeg
    print("\n[多媒体工具]")
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True,
                           timeout=10).stdout.split("\n")[0]
        results["ffmpeg"] = check("FFmpeg", out, "未安装",
                                  "从 https://ffmpeg.org 下载 或 winget install ffmpeg")
    except FileNotFoundError:
        results["ffmpeg"] = check("FFmpeg", None, "Command not found",
                                  "winget install ffmpeg 或从 https://ffmpeg.org 下载")

    # 7. Python 核心库
    print("\n[Python 核心库]")
    for lib, name in [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("scipy", "SciPy"),
        ("soundfile", "SoundFile"),
        ("librosa", "Librosa"),
    ]:
        try:
            m = importlib.import_module(lib)
            ver = getattr(m, "__version__", "已安装")
            results[name.lower()] = check(name, ver, "未安装", f"pip install {lib}")
        except ImportError:
            results[name.lower()] = False
            print(f"  [ERROR] {name}: 未安装  → pip install {lib}")

    # 8. FunASR
    print("\n[FunASR]")
    try:
        import funasr
        results["funasr"] = check("FunASR", funasr.__version__, "未安装",
                                   "pip install funasr modelscope")
    except ImportError:
        results["funasr"] = False
        print("  [ERROR] FunASR: 未安装")
        print("         解决: pip install funasr modelscope")

    # 9. FastAPI
    try:
        import fastapi
        results["fastapi"] = check("FastAPI", fastapi.__version__, "未安装")
    except ImportError:
        results["fastapi"] = False
        print("  [WARN] FastAPI: 未安装 (MVP阶段可选)")

    # 10. GPT-SoVITS
    print("\n[AI 模型]")
    sovits_path = PROJECT_ROOT / "voice" / "models" / "GPT-SoVITS"
    if sovits_path.exists():
        results["gpt_sovits"] = check("GPT-SoVITS", f"路径存在: {sovits_path}",
                                    "路径不存在", "git clone GPT-SoVITS 到 voice/models/")
    else:
        results["gpt_sovits"] = False
        print(f"  [ERROR] GPT-SoVITS: 路径不存在 ({sovits_path})")
        print("         解决: git clone https://github.com/RVC-Boss/GPT-SoVITS.git voice/models/GPT-SoVITS")

    # 11. MuseTalk
    musetalk_path = PROJECT_ROOT / "app" / "backend" / "providers" / "lipsync" / "MuseTalk"
    if musetalk_path.exists():
        results["musetalk"] = check("MuseTalk", f"路径存在: {musetalk_path}",
                                    "路径不存在")
    else:
        results["musetalk"] = False
        print(f"  [ERROR] MuseTalk: 路径不存在 ({musetalk_path})")
        print("         解决: git clone https://github.com/TMElyralab/MuseTalk.git app/backend/providers/lipsync/MuseTalk")

    # 12. Avatar
    print("\n[素材]")
    avatar_path = PROJECT_ROOT / "avatar" / "avatar.mp4"
    if avatar_path.exists():
        results["avatar"] = check("Avatar视频", str(avatar_path), "文件不存在")
    else:
        results["avatar"] = False
        print(f"  [WARN] Avatar视频: 不存在 ({avatar_path})")
        print(f"         请准备人物视频放到 {avatar_path}")

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"  检查结果: {passed}/{total} 通过")
    if passed == total:
        print("  状态: 全部通过，可以开始开发！")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  缺失项: {', '.join(failed)}")
        print(f"  请先安装上述缺失依赖")
    print("=" * 60)
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
