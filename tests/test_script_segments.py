import unittest

from scripts.script_segments import split_script_into_segments


class ScriptSegmentTests(unittest.TestCase):
    def test_keeps_complete_sentences_together_when_they_fit(self):
        script = "先观察队友的位置。再判断防守人的距离。最后选择最合适的传球路线。"

        segments = split_script_into_segments(script, max_characters=18)

        self.assertEqual(segments, ["先观察队友的位置。", "再判断防守人的距离。", "最后选择最合适的传球路线。"])

    def test_splits_an_oversized_sentence_at_commas(self):
        script = "优秀的中场要先观察身后，再判断队友位置，然后选择传球路线，最后准备下一步动作。"

        segments = split_script_into_segments(script, max_characters=20)

        self.assertEqual(segments, ["优秀的中场要先观察身后，再判断队友位置，", "然后选择传球路线，最后准备下一步动作。"])

    def test_preserves_all_script_content_without_whitespace(self):
        script = "第一段内容。\n第二段内容，继续说明。第三段内容！"

        segments = split_script_into_segments(script, max_characters=10)

        self.assertEqual("".join(segments), "第一段内容。第二段内容，继续说明。第三段内容！")


if __name__ == "__main__":
    unittest.main()
