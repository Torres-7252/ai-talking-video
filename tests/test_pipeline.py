"""Tests for validated pipeline resume behavior."""

import tempfile
import unittest
from pathlib import Path

from scripts.pipeline import Pipeline, artifact_is_valid


class PipelineResumeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
