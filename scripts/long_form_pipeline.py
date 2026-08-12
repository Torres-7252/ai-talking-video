"""Sequential long-form orchestration for the single-segment avatar pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from app.backend.providers.media_utils import validate_video
from scripts.output_paths import WORK_ROOT
from scripts.pipeline import Pipeline, resolve_final_output_path
from scripts.script_segments import concatenate_videos, split_script_into_segments


def segment_limit(speed: float) -> int:
    return max(40, min(160, int(round(4.5 * max(speed, 0.1) * 30))))


class LongFormPipeline:
    def __init__(
        self,
        *,
        project_name: str,
        title: str,
        script_text: str,
        log: Callable[[str], None],
        progress: Callable[[str, str, int, int], None] | None = None,
        **options,
    ):
        self.project_name, self.title, self.script_text, self.log, self.options = project_name, title, script_text, log, options
        self.progress = progress or (lambda stage, status, current, total: None)
        self.project_dir = WORK_ROOT / project_name
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.final_file = resolve_final_output_path(title)
        self.segments = split_script_into_segments(script_text, segment_limit(float(options.get("speed", 1.0))))

    def run(self) -> Path:
        part_files: list[Path] = []
        for index, segment in enumerate(self.segments, start=1):
            self.log(f"正在生成第 {index}/{len(self.segments)} 段（约 30 秒以内）")
            pipeline = Pipeline(project_name=f".{self.project_name}_part_{index}", title=self.title, script_text=segment, **self.options)
            part_file = self.project_dir / f"segment_{index:02d}.mp4"
            pipeline.final_file = part_file
            pipeline.metadata["final_filename"] = part_file.name
            pipeline.metadata["final_path"] = str(part_file)
            pipeline._save_metadata()
            pipeline.step0_setup()
            for stage, action in (
                ("声音生成", pipeline.step1_voice),
                ("Ditto 真实数字人", pipeline.step2_lipsync),
                ("字幕生成", pipeline.step3_subtitle),
                ("自适应比例合成", pipeline.step4_render),
                ("最终导出", pipeline.step5_export),
            ):
                self.progress(stage, "running", index, len(self.segments))
                try:
                    action()
                except Exception:
                    self.progress(stage, "failed", index, len(self.segments))
                    raise
                self.progress(stage, "done", index, len(self.segments))
            part_files.append(part_file)
        self.log(f"正在合并 {len(part_files)} 段视频")
        self.progress("最终导出", "running", len(self.segments), len(self.segments))
        concatenate_videos(part_files, self.final_file)
        info = validate_video(
            self.final_file,
            expected_size=(
                int(self.options.get("output_width", 1920)),
                int(self.options.get("output_height", 1080)),
            ),
        )
        metadata = {
            "title": self.title, "script": self.script_text, "segments": self.segments,
            "segment_count": len(self.segments), "final_filename": self.final_file.name,
            "final_path": str(self.final_file), "duration_seconds": round(info["duration"], 3),
            "file_size_mb": round(info["size"] / 1024**2, 2),
            "steps": {"long_form": {"status": "done"}},
        }
        (self.project_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self.progress("最终导出", "done", len(self.segments), len(self.segments))
        return self.final_file
