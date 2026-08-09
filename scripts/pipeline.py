#!/usr/bin/env python3
"""
完整流水线 - AI数字人口播视频生成器

用法:
    python scripts/pipeline.py --title "为什么你的停球总是停不好" --script "为什么你的第一脚触球一直停不好？今天告诉你三个关键点。" --template football_knowledge
    python scripts/pipeline.py --project 2026-08-07_为什么你的停球总是停不好 --resume
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app" / "backend" / "providers"))


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def sanitize_filename(name: str) -> str:
    """清理文件夹名"""
    bad = '<>:"/\\|?*'
    for c in bad:
        name = name.replace(c, "_")
    return name[:50]


class Pipeline:
    def __init__(self, project_name: str, title: str, script_text: str,
                 voice_profile: str = "default", speed: float = 1.0,
                 template: str = "talking_head", resume: bool = False):
        self.project_name = project_name
        self.title = title
        self.script_text = script_text
        self.voice_profile = voice_profile
        self.speed = speed
        self.template = template
        self.resume = resume

        # 输出目录
        self.project_dir = PROJECT_ROOT / "outputs" / project_name
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # 文件路径
        self.script_file = self.project_dir / "script.txt"
        self.audio_file = self.project_dir / "audio.wav"
        self.talking_file = self.project_dir / "talking.mp4"
        self.subtitle_json = self.project_dir / "subtitle.json"
        self.subtitle_srt = self.project_dir / "subtitle.srt"
        self.packaged_file = self.project_dir / "packaged.mp4"
        self.final_file = self.project_dir / "final.mp4"
        self.metadata_file = self.project_dir / "metadata.json"

        # 元数据
        self.metadata = {
            "title": title,
            "script": script_text,
            "created_at": datetime.now().isoformat(),
            "voice_profile": voice_profile,
            "avatar": "ai_coach",
            "template": template,
            "steps": {},
        }

    def should_run(self, output_file: Path) -> bool:
        """检查是否需要运行该步骤"""
        if output_file.exists() and self.resume:
            print(f"  [skip] {output_file.name} 已存在，跳过")
            return False
        return True

    def update_metadata(self, step: str, status: str, error: Optional[str] = None):
        """更新步骤状态"""
        self.metadata["steps"][step] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "error": error,
        }
        self._save_metadata()

    def _save_metadata(self):
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def step0_setup(self):
        """STEP 0: 保存脚本文案"""
        print("\n" + "=" * 60)
        print("  STEP 0: 准备工作")
        print("=" * 60)

        with open(self.script_file, "w", encoding="utf-8") as f:
            f.write(self.script_text)
        print(f"  [setup] 文案已保存: {self.script_file}")
        self.update_metadata("setup", "done")

    def step1_voice(self):
        """STEP 1: 生成 AI 声音"""
        print("\n" + "=" * 60)
        print("  STEP 1: 生成 AI 声音 (GPT-SoVITS)")
        print("=" * 60)

        if not self.should_run(self.audio_file):
            self.update_metadata("voice", "skipped")
            return

        try:
            from voice import generate_voice
            generate_voice(
                text=self.script_text,
                output_path=str(self.audio_file),
                voice_profile=self.voice_profile,
                speed=self.speed,
            )
            self.update_metadata("voice", "done")
        except Exception as e:
            self.update_metadata("voice", "failed", str(e))
            raise

    def step2_lipsync(self):
        """STEP 2: 生成嘴型同步视频"""
        print("\n" + "=" * 60)
        print("  STEP 2: 生成嘴型同步 (MuseTalk)")
        print("=" * 60)

        if not self.should_run(self.talking_file):
            self.update_metadata("lipsync", "skipped")
            return

        if not self.audio_file.exists():
            raise FileNotFoundError(f"音频文件不存在: {self.audio_file}")

        avatar = PROJECT_ROOT / "avatar" / "avatar.mp4"
        if not avatar.exists():
            raise FileNotFoundError(f"人物视频不存在: {avatar}")

        try:
            from lipsync import generate_lipsync, preprocess_avatar

            # 第一次预处理人物
            cache = PROJECT_ROOT / "avatar" / "cache"
            if not (cache / "prepared.ready").exists():
                print("  [lipsync] 首次运行，预处理人物...")
                preprocess_avatar(str(avatar), str(cache))

            generate_lipsync(
                avatar_path=str(avatar),
                audio_path=str(self.audio_file),
                output_path=str(self.talking_file),
            )
            self.update_metadata("lipsync", "done")
        except Exception as e:
            self.update_metadata("lipsync", "failed", str(e))
            raise

    def step3_subtitle(self):
        """STEP 3: 生成字幕时间轴"""
        print("\n" + "=" * 60)
        print("  STEP 3: 生成字幕 (FunASR)")
        print("=" * 60)

        if not self.should_run(self.subtitle_json):
            self.update_metadata("subtitle", "skipped")
            return

        if not self.audio_file.exists():
            raise FileNotFoundError(f"音频文件不存在: {self.audio_file}")

        try:
            from subtitle import generate_subtitles
            generate_subtitles(
                audio_path=str(self.audio_file),
                output_json=str(self.subtitle_json),
                output_srt=str(self.subtitle_srt),
            )
            self.update_metadata("subtitle", "done")
        except Exception as e:
            self.update_metadata("subtitle", "failed", str(e))
            raise

    def step4_render(self):
        """STEP 4: 视频包装渲染"""
        print("\n" + "=" * 60)
        print("  STEP 4: 视频包装 (HyperFrames)")
        print("=" * 60)

        if not self.should_run(self.packaged_file):
            self.update_metadata("render", "skipped")
            return

        if not self.talking_file.exists():
            raise FileNotFoundError(f"嘴型视频不存在: {self.talking_file}")
        if not self.subtitle_json.exists():
            raise FileNotFoundError(f"字幕文件不存在: {self.subtitle_json}")

        try:
            from render import render_video

            logo = PROJECT_ROOT / "assets" / "logo" / "logo.png"
            bgm = PROJECT_ROOT / "assets" / "music" / "default_bgm.mp3"

            render_video(
                talking_video=str(self.talking_file),
                subtitle_json=str(self.subtitle_json),
                output_path=str(self.packaged_file),
                title=self.title,
                logo_path=str(logo) if logo.exists() else None,
                bgm_path=str(bgm) if bgm.exists() else None,
                template=self.template,
            )
            self.update_metadata("render", "done")
        except Exception as e:
            self.update_metadata("render", "failed", str(e))
            raise

    def step5_export(self):
        """STEP 5: FFmpeg 最终导出"""
        print("\n" + "=" * 60)
        print("  STEP 5: 最终导出 (FFmpeg)")
        print("=" * 60)

        if not self.should_run(self.final_file):
            self.update_metadata("export", "skipped")
            return

        input_video = self.packaged_file if self.packaged_file.exists() else self.talking_file
        if not input_video.exists():
            raise FileNotFoundError(f"输入视频不存在: {input_video}")

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_video),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "192k",
                "-vf", "scale=1080:1920,setsar=1",
                "-r", "30",
                str(self.final_file),
            ]

            print(f"  [export] FFmpeg 编码中...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg 导出失败: {result.stderr}")

            # 文件大小
            size_mb = self.final_file.stat().st_size / (1024 * 1024)
            print(f"  [export] 最终视频: {self.final_file} ({size_mb:.1f} MB)")

            # 获取视频时长
            probe_cmd = [
                "ffprobe", "-v", "quiet", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                str(self.final_file),
            ]
            probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0

            self.metadata["duration_seconds"] = round(duration, 1)
            self.metadata["file_size_mb"] = round(size_mb, 1)
            self.update_metadata("export", "done")
        except Exception as e:
            self.update_metadata("export", "failed", str(e))
            raise

    def run(self):
        """运行完整流水线"""
        start_time = time.time()
        print("\n" + "█" * 60)
        print(f"  AI数字人口播视频生成器")
        print(f"  项目: {self.project_name}")
        print(f"  模板: {self.template}")
        print("█" * 60)

        steps = [
            ("0_setup", self.step0_setup),
            ("1_voice", self.step1_voice),
            ("2_lipsync", self.step2_lipsync),
            ("3_subtitle", self.step3_subtitle),
            ("4_render", self.step4_render),
            ("5_export", self.step5_export),
        ]

        failed_step = None
        for step_name, step_fn in steps:
            try:
                step_fn()
            except Exception as e:
                failed_step = step_name
                print(f"\n  [FAILED] {step_name}: {e}")
                break

        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        if failed_step:
            print(f"  流水线在 {failed_step} 失败")
            print(f"  已完成的文件保留在: {self.project_dir}")
            print(f"  使用 --resume 从失败步骤继续")
        else:
            print(f"  流水线全部完成！")
            print(f"  最终视频: {self.final_file}")
            print(f"  总耗时: {elapsed:.0f}s")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AI数字人口播视频生成器")
    parser.add_argument("--project", type=str, help="项目名称 (用于输出目录)")
    parser.add_argument("--title", type=str, help="视频标题")
    parser.add_argument("--script", type=str, help="口播文案内容")
    parser.add_argument("--script-file", type=str, help="从文件读取口播文案")
    parser.add_argument("--voice", type=str, default="default", help="声音配置")
    parser.add_argument("--speed", type=float, default=1.0, help="语速")
    parser.add_argument("--template", type=str, default="talking_head",
                        choices=["football_knowledge", "football_training", "product_promo", "talking_head"],
                        help="视频模板")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已完成的步骤")

    args = parser.parse_args()

    # 读取脚本
    script_text = ""
    if args.script:
        script_text = args.script
    elif args.script_file:
        with open(args.script_file, "r", encoding="utf-8") as f:
            script_text = f.read().strip()

    if not script_text:
        parser.error("请提供 --script 或 --script-file")

    # 生成项目名
    project_name = args.project
    if not project_name:
        safe_title = sanitize_filename(args.title or "untitled")
        project_name = f"{get_timestamp()}_{safe_title}"

    pipeline = Pipeline(
        project_name=project_name,
        title=args.title or "AI数字人口播",
        script_text=script_text,
        voice_profile=args.voice,
        speed=args.speed,
        template=args.template,
        resume=args.resume,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
