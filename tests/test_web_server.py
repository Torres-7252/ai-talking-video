"""Tests for web project path confinement."""

import asyncio
import inspect
import json
import unittest
from unittest.mock import patch

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


class WebMotionTests(unittest.TestCase):
    def setUp(self):
        web_server.tasks.clear()

    def test_generate_request_preserves_natural_motion_options(self):
        parameters = inspect.signature(web_server.api_generate).parameters
        self.assertIn("avatar_engine", parameters)
        self.assertIn("motion_mode", parameters)
        self.assertIn("motion_style", parameters)
        self.assertIn("motion_intensity", parameters)
        self.assertIn("caption_style", parameters)
        with patch.object(web_server.threading, "Thread"):
            response = asyncio.run(
                web_server.api_generate(
                    title="motion test",
                    script="test script",
                    voice="default",
                    speed=1.0,
                    template="talking_head",
                    resume=False,
                    avatar_engine="classic",
                    motion_mode="natural",
                    motion_style="steady",
                    motion_intensity=0.35,
                    caption_style="pop",
                )
            )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        task = web_server.tasks[payload["task_id"]]
        self.assertEqual(task["motion_mode"], "natural")
        self.assertEqual(task["avatar_engine"], "classic")
        self.assertEqual(task["motion_style"], "steady")
        self.assertEqual(task["motion_intensity"], 0.35)
        self.assertEqual(task["caption_style"], "pop")
        self.assertIn("LivePortrait 自然动作", [step["name"] for step in task["steps"]])

    def test_generate_request_rejects_invalid_motion_options(self):
        parameters = inspect.signature(web_server.api_generate).parameters
        self.assertIn("motion_mode", parameters)
        with patch.object(web_server.threading, "Thread"):
            bad_mode = asyncio.run(
                web_server.api_generate(
                    title="motion test",
                    script="test script",
                    voice="default",
                    speed=1.0,
                    template="talking_head",
                    resume=False,
                    motion_mode="random",
                    motion_style="steady",
                    motion_intensity=0.35,
                )
            )
            bad_intensity = asyncio.run(
                web_server.api_generate(
                    title="motion test",
                    script="test script",
                    voice="default",
                    speed=1.0,
                    template="talking_head",
                    resume=False,
                    motion_mode="natural",
                    motion_style="steady",
                    motion_intensity=1.5,
                )
            )

        self.assertEqual(bad_mode.status_code, 400)
        self.assertEqual(bad_intensity.status_code, 400)

    def test_generate_request_rejects_invalid_caption_style(self):
        with patch.object(web_server.threading, "Thread"):
            response = asyncio.run(
                web_server.api_generate(
                    title="caption test",
                    script="test script",
                    caption_style="flashy",
                )
            )

        self.assertEqual(response.status_code, 400)

    def test_generate_request_defaults_to_ditto_and_rejects_unknown_engine(self):
        parameters = inspect.signature(web_server.api_generate).parameters
        self.assertEqual(parameters["avatar_engine"].default.default, "ditto")
        with patch.object(web_server.threading, "Thread"):
            response = asyncio.run(
                web_server.api_generate(
                    title="ditto test",
                    script="test script",
                    avatar_engine="unknown",
                )
            )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
