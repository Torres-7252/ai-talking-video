"""Tests for long-form pipeline progress reporting."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import long_form_pipeline


class _FakePipeline:
    instances = []

    def __init__(self, **kwargs):
        self.metadata = {}
        self.final_file = None
        self.calls = []
        type(self).instances.append(self)

    def _save_metadata(self):
        self.calls.append("save")

    def step0_setup(self): self.calls.append("setup")
    def step1_voice(self): self.calls.append("voice")
    def step2_lipsync(self): self.calls.append("ditto")
    def step3_subtitle(self): self.calls.append("subtitle")
    def step4_render(self): self.calls.append("render")
    def step5_export(self): self.calls.append("export")


class LongFormProgressTests(unittest.TestCase):
    def test_reports_each_stage_for_each_segment(self):
        _FakePipeline.instances.clear()
        events = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final = root / "final.mp4"
            final.write_bytes(b"video")
            with (
                patch.object(long_form_pipeline, "WORK_ROOT", root),
                patch.object(long_form_pipeline, "Pipeline", _FakePipeline),
                patch.object(long_form_pipeline, "resolve_final_output_path", return_value=final),
                patch.object(long_form_pipeline, "concatenate_videos"),
                patch.object(
                    long_form_pipeline,
                    "validate_video",
                    return_value={"duration": 10.0, "size": 1024},
                ),
            ):
                pipeline = long_form_pipeline.LongFormPipeline(
                    project_name="long",
                    title="title",
                    script_text=("甲" * 100) + "。" + ("乙" * 100) + "。",
                    log=lambda message: None,
                    progress=lambda stage, status, current, total: events.append(
                        (stage, status, current, total)
                    ),
                )
                pipeline.run()

        self.assertIn(("声音生成", "running", 1, 2), events)
        self.assertIn(("Ditto 真实数字人", "done", 2, 2), events)
        self.assertIn(("最终导出", "done", 2, 2), events)
        self.assertEqual(
            _FakePipeline.instances[0].calls,
            ["save", "setup", "voice", "ditto", "subtitle", "render", "export"],
        )
