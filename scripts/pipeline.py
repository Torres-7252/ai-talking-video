#!/usr/bin/env python3
"""Validated local AI talking-video pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = (PROJECT_ROOT / "outputs").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.providers.media_utils import probe_media, validate_audio, validate_video
from app.backend.providers.subtitle.ass_renderer import CAPTION_PRESETS


DRIVER_PROFILES = {
    "subtle_presenter": PROJECT_ROOT / "motion" / "drivers" / "subtle_presenter.mp4"
}


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def sanitize_filename(name: str) -> str:
    for character in '<>:"/\\|?*':
        name = name.replace(character, "_")
    return name.strip()[:50]


def build_motion_signature(
    avatar_path: Path,
    *,
    style: str,
    intensity: float,
    fps: int,
    audio_duration: float,
    driver_path: Optional[Path] = None,
) -> str:
    avatar = Path(avatar_path).resolve()
    digest = hashlib.sha256()
    digest.update(avatar.read_bytes())
    if driver_path is not None:
        digest.update(Path(driver_path).resolve().read_bytes())
    settings = {
        "style": style,
        "intensity": round(float(intensity), 4),
        "fps": int(fps),
        "audio_duration": round(float(audio_duration), 3),
    }
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _resolve_project_dir(project_name: str, outputs_root: Path = OUTPUTS_ROOT) -> Path:
    outputs = Path(outputs_root).resolve()
    project = (outputs / project_name).resolve()
    if project == outputs or not project.is_relative_to(outputs):
        raise ValueError(f"Invalid project name: {project_name}")
    return project


def load_resume_metadata(
    project_name: str,
    outputs_root: Path = OUTPUTS_ROOT,
) -> dict:
    metadata_path = _resolve_project_dir(project_name, outputs_root) / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Resume metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not str(metadata.get("script") or "").strip():
        raise RuntimeError(f"Resume metadata has no talking script: {metadata_path}")
    return metadata


def artifact_is_valid(step: str, path: Path) -> bool:
    """Return whether a stage artifact is complete enough to resume from."""
    path = Path(path)
    try:
        if step == "setup":
            return path.is_file() and bool(path.read_text(encoding="utf-8").strip())
        if step == "voice":
            validate_audio(path)
            return True
        if step == "motion":
            validate_video(path)
            return True
        if step in {"lipsync", "talking"}:
            info = validate_video(path)
            return any(stream.get("codec_type") == "audio" for stream in info["streams"])
        if step in {"render", "export", "final"}:
            info = validate_video(path, expected_size=(1920, 1080))
            return any(stream.get("codec_type") == "audio" for stream in info["streams"])
        if step == "subtitle":
            segments = json.loads(path.read_text(encoding="utf-8"))
            return bool(segments) and all(
                str(segment.get("text") or "").strip()
                and float(segment.get("end", 0)) > float(segment.get("start", 0))
                for segment in segments
            )
    except (OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError):
        return False
    return path.is_file() and path.stat().st_size > 0


def _artifact_summary(step: str, path: Path) -> dict:
    summary = {
        "path": str(path),
        "valid": artifact_is_valid(step, path),
        "size": path.stat().st_size if path.exists() else 0,
    }
    if summary["valid"] and step in {
        "voice",
        "motion",
        "lipsync",
        "talking",
        "render",
        "export",
        "final",
    }:
        info = probe_media(path)
        summary["duration"] = round(info["duration"], 3)
        summary["streams"] = info["streams"]
    return summary


class Pipeline:
    def __init__(
        self,
        project_name: str,
        title: str,
        script_text: str,
        voice_profile: str = "default",
        speed: float = 1.0,
        template: str = "talking_head",
        resume: bool = False,
        avatar_engine: str = "ditto",
        motion_mode: str = "natural",
        motion_style: str = "steady",
        motion_intensity: float = 0.35,
        caption_style: str = "clean",
        driver_profile: str = "subtle_presenter",
    ):
        if not script_text.strip():
            raise ValueError("The talking script cannot be empty")
        self.project_name = project_name
        self.title = title
        self.script_text = script_text.strip()
        self.voice_profile = voice_profile
        self.speed = speed
        self.template = template
        self.resume = resume
        if avatar_engine not in {"ditto", "classic"}:
            raise ValueError(f"Unsupported avatar engine: {avatar_engine}")
        if motion_mode not in {"natural", "gesture", "off"}:
            raise ValueError(f"Unsupported motion mode: {motion_mode}")
        if motion_style != "steady":
            raise ValueError(f"Unsupported motion style: {motion_style}")
        if not 0.0 <= motion_intensity <= 1.0:
            raise ValueError("Motion intensity must be between 0.0 and 1.0")
        if caption_style not in CAPTION_PRESETS:
            raise ValueError(f"Unsupported caption style: {caption_style}")
        if driver_profile not in DRIVER_PROFILES:
            raise ValueError(f"Unsupported driver profile: {driver_profile}")
        self.avatar_engine = avatar_engine
        self.motion_mode = motion_mode
        self.motion_style = motion_style
        self.motion_intensity = float(motion_intensity)
        self.caption_style = caption_style
        self.driver_profile = driver_profile

        self.project_dir = _resolve_project_dir(project_name)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        self.script_file = self.project_dir / "script.txt"
        self.audio_file = self.project_dir / "audio.wav"
        self.motion_file = self.project_dir / "motion.mp4"
        self.talking_file = self.project_dir / "talking.mp4"
        self.subtitle_json = self.project_dir / "subtitle.json"
        self.subtitle_srt = self.project_dir / "subtitle.srt"
        self.packaged_file = self.project_dir / "packaged.mp4"
        self.final_file = self.project_dir / "final.mp4"
        self.metadata_file = self.project_dir / "metadata.json"

        existing = {}
        if resume and self.metadata_file.is_file():
            try:
                existing = json.loads(self.metadata_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
        self.metadata = {
            "title": title,
            "script": self.script_text,
            "created_at": existing.get("created_at", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
            "voice_profile": voice_profile,
            "avatar": "avatar/avatar.jpg",
            "template": template,
            "avatar_engine": avatar_engine,
            "motion_mode": motion_mode,
            "motion_style": motion_style,
            "motion_intensity": self.motion_intensity,
            "caption_style": caption_style,
            "driver_profile": driver_profile,
            "output": "1920x1080, 25 fps, H.264/AAC",
            "steps": existing.get("steps", {}),
        }
        self._save_metadata()

    def should_run(self, output_file: Path, step: str) -> bool:
        if self.resume and artifact_is_valid(step, output_file):
            print(f"  [resume] Valid {step} artifact: {output_file.name}")
            return False
        if self.resume and output_file.exists():
            print(f"  [resume] Invalid {step} artifact will be regenerated: {output_file.name}")
        return True

    def _step_log(self, step: str, message: str) -> Path:
        log_path = self.project_dir / f"{step}.log"
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"[{datetime.now().isoformat()}] {message}\n")
        return log_path

    def update_metadata(
        self,
        step: str,
        status: str,
        error: Optional[str] = None,
        artifact: Optional[Path] = None,
    ) -> None:
        now = datetime.now().isoformat()
        record = self.metadata["steps"].setdefault(step, {})
        record["status"] = status
        record["error"] = error
        record["log_path"] = str(self.project_dir / f"{step}.log")
        if status == "running":
            record["started_at"] = now
            record.pop("finished_at", None)
        else:
            record["finished_at"] = now
        if artifact is not None:
            record["artifact"] = _artifact_summary(step, artifact)
        self.metadata["updated_at"] = now
        self._save_metadata()

    def _save_metadata(self) -> None:
        self.metadata_file.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _start(self, step: str, label: str) -> None:
        print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
        self._step_log(step, "started")
        self.update_metadata(step, "running")

    def _finish(self, step: str, artifact: Path, resumed: bool = False) -> None:
        status = "skipped" if resumed else "done"
        self._step_log(step, "resumed from valid artifact" if resumed else "completed")
        self.update_metadata(step, status, artifact=artifact)

    def _fail(self, step: str, exc: Exception) -> None:
        self._step_log(step, f"failed: {exc}")
        self.update_metadata(step, "failed", str(exc))

    def step0_setup(self) -> None:
        self._start("setup", "STEP 0: Save script")
        self.script_file.write_text(self.script_text, encoding="utf-8")
        self._finish("setup", self.script_file)

    def step1_voice(self) -> None:
        self._start("voice", "STEP 1: GPT-SoVITS voice clone")
        if not self.should_run(self.audio_file, "voice"):
            self._finish("voice", self.audio_file, resumed=True)
            return
        try:
            from app.backend.providers.voice import generate_voice, unload_voice

            try:
                generate_voice(
                    text=self.script_text,
                    output_path=str(self.audio_file),
                    voice_profile=self.voice_profile,
                    speed=self.speed,
                )
            finally:
                unload_voice()
            self._finish("voice", self.audio_file)
        except Exception as exc:
            self._fail("voice", exc)
            raise

    @property
    def lipsync_input(self) -> Path:
        if self.motion_mode in {"natural", "gesture"}:
            return self.motion_file
        return PROJECT_ROOT / "avatar" / "avatar.jpg"

    def step2_motion(self) -> None:
        motion_label = (
            "MimicMotion gesture motion"
            if self.motion_mode == "gesture"
            else "LivePortrait natural motion"
        )
        self._start("motion", f"STEP 2: {motion_label}")
        if self.avatar_engine == "ditto" or self.motion_mode == "off":
            reason = (
                "Ditto handles full facial motion"
                if self.avatar_engine == "ditto"
                else "motion mode is off"
            )
            self._step_log("motion", f"skipped because {reason}")
            self.update_metadata("motion", "skipped")
            return

        avatar = PROJECT_ROOT / "avatar" / "avatar.jpg"
        if not avatar.is_file():
            exc = FileNotFoundError(f"Avatar image does not exist: {avatar}")
            self._fail("motion", exc)
            raise exc
        if not artifact_is_valid("voice", self.audio_file):
            exc = RuntimeError(f"Voice artifact is invalid: {self.audio_file}")
            self._fail("motion", exc)
            raise exc

        audio_duration = probe_media(self.audio_file)["duration"]
        driver = DRIVER_PROFILES[self.driver_profile] if self.motion_mode == "gesture" else None
        if driver is not None and not driver.is_file():
            exc = FileNotFoundError(f"Gesture driver does not exist: {driver}")
            self._fail("motion", exc)
            raise exc
        signature = build_motion_signature(
            avatar,
            style=self.motion_style,
            intensity=self.motion_intensity,
            fps=15 if self.motion_mode == "gesture" else 25,
            audio_duration=audio_duration,
            driver_path=driver,
        )
        record = self.metadata["steps"].setdefault("motion", {})
        if (
            self.resume
            and artifact_is_valid("motion", self.motion_file)
            and record.get("signature") == signature
        ):
            record["signature"] = signature
            self._finish("motion", self.motion_file, resumed=True)
            return
        if self.resume and self.motion_file.exists():
            print("  [resume] Motion settings changed or artifact is invalid; regenerating")

        try:
            if self.motion_mode == "gesture":
                from app.backend.providers.motion import generate_gesture_motion

                generate_gesture_motion(
                    image_path=str(avatar),
                    driver_video_path=str(driver),
                    output_path=str(self.motion_file),
                    duration=audio_duration,
                    intensity=self.motion_intensity,
                )
            else:
                from app.backend.providers.motion import generate_motion

                generate_motion(
                    avatar_path=str(avatar),
                    audio_path=str(self.audio_file),
                    output_path=str(self.motion_file),
                    style=self.motion_style,
                    intensity=self.motion_intensity,
                    fps=25,
                )
            record["signature"] = signature
            self._finish("motion", self.motion_file)
        except Exception as exc:
            self._fail("motion", exc)
            raise

    def step2_lipsync(self) -> None:
        label = (
            "Ditto audio-driven avatar"
            if self.avatar_engine == "ditto"
            else "MuseTalk 1.5 lip sync"
        )
        self._start("lipsync", f"STEP 2: {label}")
        if not self.should_run(self.talking_file, "lipsync"):
            self._finish("lipsync", self.talking_file, resumed=True)
            return
        avatar = (
            PROJECT_ROOT / "avatar" / "avatar.jpg"
            if self.avatar_engine == "ditto"
            else self.lipsync_input
        )
        if not avatar.is_file():
            raise FileNotFoundError(f"Lip-sync input does not exist: {avatar}")
        if (
            self.avatar_engine == "classic"
            and self.motion_mode in {"natural", "gesture"}
            and not artifact_is_valid("motion", avatar)
        ):
            raise RuntimeError(f"Motion artifact is invalid: {avatar}")
        if not artifact_is_valid("voice", self.audio_file):
            raise RuntimeError(f"Voice artifact is invalid: {self.audio_file}")
        try:
            if self.avatar_engine == "ditto":
                from app.backend.providers.avatar import generate_avatar

                generate_avatar(
                    source_path=str(avatar),
                    audio_path=str(self.audio_file),
                    output_path=str(self.talking_file),
                )
                self._finish("lipsync", self.talking_file)
                return

            from app.backend.providers.lipsync import generate_lipsync

            generate_lipsync(
                avatar_path=str(avatar),
                audio_path=str(self.audio_file),
                output_path=str(self.talking_file),
                use_fp16=True,
            )
            self._finish("lipsync", self.talking_file)
        except Exception as exc:
            self._fail("lipsync", exc)
            raise

    def step3_subtitle(self) -> None:
        self._start("subtitle", "STEP 3: FunASR timed subtitles")
        if not self.should_run(self.subtitle_json, "subtitle"):
            self._finish("subtitle", self.subtitle_json, resumed=True)
            return
        try:
            from app.backend.providers.subtitle import generate_subtitles

            generate_subtitles(
                audio_path=str(self.audio_file),
                output_json=str(self.subtitle_json),
                output_srt=str(self.subtitle_srt),
                transcript_text=self.script_text,
            )
            self._finish("subtitle", self.subtitle_json)
        except Exception as exc:
            self._fail("subtitle", exc)
            raise

    def step4_render(self) -> None:
        self._start("render", "STEP 4: FFmpeg landscape render")
        if not self.should_run(self.packaged_file, "render"):
            self._finish("render", self.packaged_file, resumed=True)
            return
        try:
            from app.backend.providers.render import render_video

            render_video(
                talking_video=str(self.talking_file),
                subtitle_json=str(self.subtitle_json),
                output_path=str(self.packaged_file),
                width=1920,
                height=1080,
                fps=25,
                caption_style=self.caption_style,
            )
            self._finish("render", self.packaged_file)
        except Exception as exc:
            self._fail("render", exc)
            raise

    def step5_export(self) -> None:
        self._start("export", "STEP 5: Validate final output")
        if not self.should_run(self.final_file, "export"):
            self._finish("export", self.final_file, resumed=True)
            return
        try:
            if not artifact_is_valid("render", self.packaged_file):
                raise RuntimeError(f"Rendered artifact is invalid: {self.packaged_file}")
            shutil.copy2(self.packaged_file, self.final_file)
            info = validate_video(self.final_file, expected_size=(1920, 1080))
            self.metadata["duration_seconds"] = round(info["duration"], 3)
            self.metadata["file_size_mb"] = round(info["size"] / 1024**2, 2)
            self._finish("export", self.final_file)
        except Exception as exc:
            self._fail("export", exc)
            raise

    def run(self) -> Path:
        start_time = time.time()
        print(f"\nLocal AI talking video\nProject: {self.project_name}")
        for step in (
            self.step0_setup,
            self.step1_voice,
            self.step2_motion,
            self.step2_lipsync,
            self.step3_subtitle,
            self.step4_render,
            self.step5_export,
        ):
            step()
        print(f"\nCompleted in {time.time() - start_time:.0f}s: {self.final_file}")
        return self.final_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Local AI talking video generator")
    parser.add_argument("--project")
    parser.add_argument("--title")
    parser.add_argument("--script")
    parser.add_argument("--script-file")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--template",
        default="talking_head",
        choices=("football_knowledge", "football_training", "product_promo", "talking_head"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--avatar-engine", default="ditto", choices=("ditto", "classic")
    )
    parser.add_argument(
        "--motion-mode", default="natural", choices=("natural", "gesture", "off")
    )
    parser.add_argument("--motion-style", default="steady", choices=("steady",))
    parser.add_argument("--motion-intensity", type=float, default=0.35)
    parser.add_argument(
        "--driver-profile",
        default="subtle_presenter",
        choices=tuple(sorted(DRIVER_PROFILES)),
    )
    parser.add_argument(
        "--caption-style", default="clean", choices=tuple(sorted(CAPTION_PRESETS))
    )
    args = parser.parse_args()

    script_text = args.script or ""
    if args.script_file:
        script_text = Path(args.script_file).read_text(encoding="utf-8").strip()
    resume_metadata = {}
    if not script_text and args.resume and args.project:
        try:
            resume_metadata = load_resume_metadata(args.project)
            script_text = str(resume_metadata["script"]).strip()
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
    if not script_text:
        parser.error("Provide --script or --script-file")

    project_name = args.project or f"{get_timestamp()}_{sanitize_filename(args.title or 'untitled')}"
    Pipeline(
        project_name=project_name,
        title=args.title or resume_metadata.get("title") or "AI talking video",
        script_text=script_text,
        voice_profile=args.voice,
        speed=args.speed,
        template=args.template,
        resume=args.resume,
        avatar_engine=args.avatar_engine,
        motion_mode=args.motion_mode,
        motion_style=args.motion_style,
        motion_intensity=args.motion_intensity,
        caption_style=args.caption_style,
        driver_profile=args.driver_profile,
    ).run()


if __name__ == "__main__":
    main()
