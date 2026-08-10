"""Tests for validated pipeline resume behavior."""

import tempfile
import unittest
import inspect
from pathlib import Path
from unittest.mock import patch

from scripts import pipeline as pipeline_module
from scripts.pipeline import Pipeline, artifact_is_valid, load_resume_metadata


class PipelineResumeTests(unittest.TestCase):
    def test_resume_metadata_restores_script_without_cli_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = Path(temp_dir)
            project = outputs / "saved-project"
            project.mkdir()
            (project / "metadata.json").write_text(
                '{"title":"已有标题","script":"已有口播文案"}', encoding="utf-8"
            )

            metadata = load_resume_metadata("saved-project", outputs)

        self.assertEqual(metadata["title"], "已有标题")
        self.assertEqual(metadata["script"], "已有口播文案")

    def test_resume_does_not_skip_empty_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "audio.wav"
            output.touch()
            pipeline = Pipeline("case", "title", "text", resume=True)

            self.assertFalse(artifact_is_valid("voice", output))
            self.assertTrue(pipeline.should_run(output, "voice"))

    def test_resume_skips_valid_subtitle_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "subtitle.json"
            output.write_text(
                '[{"text":"测试","start":0.0,"end":1.0}]', encoding="utf-8"
            )
            pipeline = Pipeline("case", "title", "text", resume=True)

            self.assertTrue(artifact_is_valid("subtitle", output))
            self.assertFalse(pipeline.should_run(output, "subtitle"))


class PipelineMotionTests(unittest.TestCase):
    def make_pipeline(self, root: Path, **kwargs) -> Pipeline:
        project = root / "motion-case"
        with patch("scripts.pipeline._resolve_project_dir", return_value=project):
            return Pipeline("motion-case", "title", "text", **kwargs)

    def test_pipeline_defaults_to_restrained_natural_motion(self):
        parameters = inspect.signature(Pipeline).parameters
        self.assertIn("avatar_engine", parameters)
        self.assertIn("motion_mode", parameters)
        self.assertIn("motion_style", parameters)
        self.assertIn("motion_intensity", parameters)
        self.assertIn("caption_style", parameters)
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = self.make_pipeline(Path(temporary))

        self.assertEqual(pipeline.avatar_engine, "ditto")
        self.assertEqual(pipeline.motion_mode, "natural")
        self.assertEqual(pipeline.motion_style, "steady")
        self.assertEqual(pipeline.motion_intensity, 0.35)
        self.assertEqual(pipeline.caption_style, "clean")
        self.assertEqual(pipeline.metadata["motion_mode"], "natural")
        self.assertEqual(pipeline.metadata["avatar_engine"], "ditto")
        self.assertEqual(pipeline.metadata["caption_style"], "clean")

    def test_pipeline_accepts_public_caption_presets_and_rejects_unknown_style(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = self.make_pipeline(Path(temporary), caption_style="pop")
            self.assertEqual(pipeline.metadata["caption_style"], "pop")
            with self.assertRaisesRegex(ValueError, "caption style"):
                self.make_pipeline(Path(temporary), caption_style="flashy")

    def test_ditto_engine_skips_legacy_motion_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = self.make_pipeline(Path(temporary), avatar_engine="ditto")
            pipeline.step2_motion()

        self.assertEqual(pipeline.metadata["steps"]["motion"]["status"], "skipped")
        self.assertFalse(pipeline.motion_file.exists())

    def test_lipsync_input_selects_motion_video_or_original_avatar(self):
        self.assertTrue(hasattr(Pipeline, "lipsync_input"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            natural = self.make_pipeline(root, motion_mode="natural")
            fast = self.make_pipeline(root, motion_mode="off")

        self.assertEqual(natural.lipsync_input, natural.motion_file)
        self.assertEqual(
            fast.lipsync_input,
            pipeline_module.PROJECT_ROOT / "avatar" / "avatar.jpg",
        )

    def test_gesture_mode_uses_motion_video_for_musetalk(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = self.make_pipeline(
                Path(temporary),
                avatar_engine="classic",
                motion_mode="gesture",
                driver_profile="subtle_presenter",
            )

        self.assertEqual(pipeline.lipsync_input, pipeline.motion_file)
        self.assertEqual(pipeline.metadata["driver_profile"], "subtle_presenter")

    def test_motion_signature_changes_with_avatar_or_settings(self):
        self.assertTrue(hasattr(pipeline_module, "build_motion_signature"))
        with tempfile.TemporaryDirectory() as temporary:
            avatar = Path(temporary) / "avatar.jpg"
            avatar.write_bytes(b"first portrait")
            baseline = pipeline_module.build_motion_signature(
                avatar, style="steady", intensity=0.35, fps=25, audio_duration=10.0
            )
            changed_intensity = pipeline_module.build_motion_signature(
                avatar, style="steady", intensity=0.5, fps=25, audio_duration=10.0
            )
            avatar.write_bytes(b"second portrait")
            changed_avatar = pipeline_module.build_motion_signature(
                avatar, style="steady", intensity=0.35, fps=25, audio_duration=10.0
            )

        self.assertNotEqual(baseline, changed_intensity)
        self.assertNotEqual(baseline, changed_avatar)
        self.assertEqual(len(baseline), 64)

    def test_gesture_signature_changes_with_driver_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            avatar = root / "avatar.png"
            driver = root / "driver.mp4"
            avatar.write_bytes(b"avatar")
            driver.write_bytes(b"first driver")
            first = pipeline_module.build_motion_signature(
                avatar,
                style="steady",
                intensity=0.25,
                fps=15,
                audio_duration=10.0,
                driver_path=driver,
            )
            driver.write_bytes(b"changed driver")
            second = pipeline_module.build_motion_signature(
                avatar,
                style="steady",
                intensity=0.25,
                fps=15,
                audio_duration=10.0,
                driver_path=driver,
            )

        self.assertNotEqual(first, second)

    def test_render_stage_forwards_caption_style(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = self.make_pipeline(
                Path(temporary), caption_style="bar"
            )
            with patch.object(pipeline, "should_run", return_value=True), patch.object(
                pipeline, "_finish"
            ), patch(
                "app.backend.providers.render.render_video"
            ) as render_video:
                pipeline.step4_render()

        self.assertEqual(render_video.call_args.kwargs["caption_style"], "bar")

    def test_off_mode_records_skipped_motion_without_creating_artifact(self):
        self.assertTrue(hasattr(Pipeline, "step2_motion"))
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = self.make_pipeline(Path(temporary), motion_mode="off")
            pipeline.step2_motion()

        self.assertEqual(pipeline.metadata["steps"]["motion"]["status"], "skipped")
        self.assertFalse(pipeline.motion_file.exists())


if __name__ == "__main__":
    unittest.main()
