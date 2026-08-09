"""Tests for web project path confinement."""

import unittest

from scripts import web_server


class WebPathTests(unittest.TestCase):
    def test_project_file_rejects_parent_escape(self):
        with self.assertRaises(ValueError):
            web_server.resolve_project_file("project", "../../.env")

    def test_project_name_rejects_parent_escape(self):
        with self.assertRaises(ValueError):
            web_server.resolve_project_file("..", "final.mp4")

    def test_project_file_stays_under_outputs(self):
        path = web_server.resolve_project_file("project", "final.mp4")
        outputs = (web_server.PROJECT_ROOT / "outputs").resolve()
        self.assertTrue(path.is_relative_to(outputs))


if __name__ == "__main__":
    unittest.main()
