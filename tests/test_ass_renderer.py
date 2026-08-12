"""Tests for animated ASS subtitle presets."""

import tempfile
import unittest
from pathlib import Path

from app.backend.providers.subtitle import ass_renderer


class AnimatedAssTests(unittest.TestCase):
    def setUp(self):
        self.segment = {
            "text": "\u8bad\u7ec3\uff0c\u66f4\u7a33\u3002",
            "start": 0.0,
            "end": 1.2,
            "words": [
                {"text": "\u8bad", "start": 0.0, "end": 0.3},
                {"text": "\u7ec3\uff0c", "start": 0.3, "end": 0.6},
                {"text": "\u66f4", "start": 0.6, "end": 0.9},
                {"text": "\u7a33\u3002", "start": 0.9, "end": 1.2},
            ],
        }

    def test_clean_preset_emits_base_and_active_word_layers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "captions.ass"
            ass_renderer.write_animated_ass([self.segment], output, preset="clean")
            content = output.read_text(encoding="utf-8-sig")

        self.assertIn("Style: Clean", content)
        self.assertIn("Dialogue: 0", content)
        self.assertIn("Dialogue: 1", content)
        self.assertIn(r"\fad(", content)
        self.assertIn("&H0047D4FF&", content)

    def test_all_public_presets_serialize(self):
        for preset in ("clean", "pop", "bar"):
            with self.subTest(preset=preset), tempfile.TemporaryDirectory() as temp_dir:
                output = Path(temp_dir) / f"{preset}.ass"
                ass_renderer.write_animated_ass(
                    [self.segment], output, preset=preset
                )
                self.assertGreater(output.stat().st_size, 300)

    def test_portrait_caption_layers_share_the_same_anchor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "portrait.ass"
            ass_renderer.write_animated_ass(
                [self.segment], output, preset="clean", width=1080, height=1920
            )
            content = output.read_text(encoding="utf-8-sig")

        dialogue_lines = [line for line in content.splitlines() if line.startswith("Dialogue:")]
        self.assertTrue(dialogue_lines)
        self.assertTrue(all(r"\pos(540,1790)" in line for line in dialogue_lines))

    def test_unknown_preset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "caption preset"):
                ass_renderer.write_animated_ass(
                    [], Path(temp_dir) / "x.ass", preset="loud"
                )

    def test_long_caption_wraps_to_at_most_two_lines(self):
        wrapped = ass_renderer.wrap_caption(
            "\u8fd9\u662f\u4e00\u6bb5\u7528\u6765\u9a8c\u8bc1\u4e24\u884c\u5b89\u5168\u5e03\u5c40\u7684\u8f83\u957f\u4e2d\u6587\u5b57\u5e55",
            max_chars=14,
        )

        self.assertLessEqual(wrapped.count(r"\N"), 1)


if __name__ == "__main__":
    unittest.main()
