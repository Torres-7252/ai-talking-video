"""Tests for FunASR result normalization and ASS output."""

import tempfile
import unittest
from pathlib import Path

from app.backend.providers import subtitle


class SubtitleProviderTests(unittest.TestCase):
    def test_known_transcript_replaces_asr_words_but_keeps_timing(self):
        subtitles = [
            {"text": "大家好这是本地echo播测试", "start": 0.1, "end": 3.8, "words": []}
        ]

        corrected = subtitle.apply_transcript_text(
            subtitles, "大家好，这是本地AI口播测试。"
        )

        self.assertEqual(corrected[0]["start"], 0.1)
        self.assertEqual(corrected[-1]["end"], 3.8)
        self.assertIn("AI", "".join(item["text"] for item in corrected))
        self.assertNotIn("echo", "".join(item["text"] for item in corrected))

    def test_known_transcript_allocates_non_overlapping_word_timings(self):
        corrected = subtitle.apply_transcript_text(
            [{"text": "old", "start": 0.2, "end": 2.2, "words": []}],
            "\u5927\u5bb6\u597d AI \u8bad\u7ec3\u3002",
        )

        words = [word for segment in corrected for word in segment["words"]]

        self.assertTrue(words)
        self.assertEqual(words[0]["start"], 0.2)
        self.assertEqual(words[-1]["end"], 2.2)
        self.assertTrue(
            all(
                left["end"] <= right["start"]
                for left, right in zip(words, words[1:])
            )
        )

    def test_punctuation_does_not_get_an_independent_highlight(self):
        words = subtitle.allocate_word_timings(
            "\u8bad\u7ec3\uff0c\u66f4\u7a33\uff01", 0.0, 1.0
        )

        self.assertNotIn("\uff0c", [word["text"] for word in words])
        self.assertNotIn("\uff01", [word["text"] for word in words])
        self.assertEqual(
            "".join(word["text"] for word in words),
            "\u8bad\u7ec3\uff0c\u66f4\u7a33\uff01",
        )

    def test_asr_options_use_pinned_cached_models_without_optional_punctuation(self):
        options = subtitle.build_asr_options(device="cuda")

        self.assertEqual(
            options["model"],
            "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        )
        self.assertEqual(
            options["vad_model"],
            "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        )
        self.assertNotIn("punc_model", options)
        self.assertEqual(options["device"], "cuda:0")

    def test_top_level_funasr_timestamps_become_nonempty_segments(self):
        result = [
            {
                "text": "大家好我是AI教练",
                "timestamp": [[0, 300], [320, 620], [650, 900]],
            }
        ]

        subtitles = subtitle.normalize_asr_result(result, 1.0)

        self.assertTrue(subtitles)
        self.assertEqual(subtitles[0]["text"], "大家好我是AI教练")
        self.assertGreater(subtitles[0]["end"], subtitles[0]["start"])

    def test_chinese_token_spaces_are_removed(self):
        subtitles = subtitle.normalize_asr_result(
            [{"text": "大 家 好 A I 教 练", "timestamp": [[0, 1000]]}], 1.0
        )

        self.assertEqual(subtitles[0]["text"], "大家好AI教练")

    def test_empty_recognition_is_rejected(self):
        with self.assertRaises(RuntimeError):
            subtitle.normalize_asr_result([], 2.0)

    def test_write_ass_sets_landscape_canvas_and_dialogue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "subtitle.ass"
            subtitle.write_ass(
                [{"text": "本地口播测试", "start": 0.0, "end": 1.25, "words": []}],
                output,
            )
            content = output.read_text(encoding="utf-8-sig")

        self.assertIn("PlayResX: 1920", content)
        self.assertIn("PlayResY: 1080", content)
        self.assertIn("Dialogue: 0,0:00:00.00,0:00:01.25", content)


if __name__ == "__main__":
    unittest.main()
