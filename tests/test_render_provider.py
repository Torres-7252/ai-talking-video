"""Tests for the FFmpeg-only final renderer."""

import unittest
from pathlib import Path

from app.backend.providers import render


class RenderProviderTests(unittest.TestCase):
    def test_render_command_is_landscape_ffmpeg(self):
        command = render.build_render_command(
            Path("talking.mp4"), Path("subtitle.ass"), Path("final.mp4")
        )
        joined = " ".join(command)

        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("1920:1080", joined)
        self.assertIn("libx264", command)
        self.assertIn("aac", command)
        self.assertIn("yuv420p", command)
        self.assertIn("format=yuv420p", joined)
        self.assertIn("range=limited", joined)
        self.assertNotIn("moviepy", joined.lower())


if __name__ == "__main__":
    unittest.main()
