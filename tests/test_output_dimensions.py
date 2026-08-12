import unittest

from scripts.output_dimensions import dimensions_for_avatar


class OutputDimensionTests(unittest.TestCase):
    def test_portrait_avatar_uses_douyin_vertical_canvas(self):
        self.assertEqual(dimensions_for_avatar(1080, 1920), (1080, 1920))

    def test_landscape_avatar_uses_landscape_canvas(self):
        self.assertEqual(dimensions_for_avatar(1920, 1080), (1920, 1080))

    def test_four_by_three_avatar_preserves_its_ratio(self):
        self.assertEqual(dimensions_for_avatar(1600, 1200), (1440, 1080))
