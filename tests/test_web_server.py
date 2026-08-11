"""Tests for web project path confinement."""

import asyncio
import inspect
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import UploadFile

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


class WebAssetUploadTests(unittest.TestCase):
    def test_failed_voice_conversion_preserves_existing_reference(self):
        original = b"existing valid reference audio"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "voice" / "references" / "default.wav"
            target.parent.mkdir(parents=True)
            target.write_bytes(original)

            def fail_after_opening_output(command, **kwargs):
                Path(command[-1]).write_bytes(b"")
                return SimpleNamespace(returncode=1, stderr=b"decode failed")

            upload = UploadFile(filename="broken.mp3", file=BytesIO(b"x" * 2048))
            with (
                patch.object(web_server, "PROJECT_ROOT", root),
                patch.object(web_server.subprocess, "run", side_effect=fail_after_opening_output),
            ):
                response = asyncio.run(
                    web_server.api_upload_voice(upload, reference_text="reference text")
                )

            self.assertEqual(response.status_code, 400)
            self.assertEqual(target.read_bytes(), original)

    def test_voice_upload_aligns_audio_to_reference_text_before_replacing(self):
        observed = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def convert_audio(command, **kwargs):
                Path(command[-1]).write_bytes(b"converted audio")
                return SimpleNamespace(returncode=0, stderr=b"")

            def align_audio(path, reference_text):
                observed["text"] = reference_text
                path.write_bytes(b"aligned audio")
                return path

            upload = UploadFile(filename="voice.mp3", file=BytesIO(b"x" * 2048))
            with (
                patch.object(web_server, "PROJECT_ROOT", root),
                patch.object(web_server.subprocess, "run", side_effect=convert_audio),
                patch(
                    "app.backend.providers.media_utils.validate_audio",
                    return_value={"duration": 6.0, "size": 1024, "streams": []},
                ),
                patch(
                    "app.backend.providers.voice.align_reference_audio",
                    side_effect=align_audio,
                    create=True,
                ) as align_reference,
            ):
                response = asyncio.run(
                    web_server.api_upload_voice(
                        upload, reference_text="the exact spoken reference"
                    )
                )

            target = root / "voice" / "references" / "default.wav"
            saved_audio = target.read_bytes()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(observed["text"], "the exact spoken reference")
        self.assertEqual(align_reference.call_count, 1)
        self.assertEqual(saved_audio, b"aligned audio")


class WebProjectDeleteTests(unittest.TestCase):
    def setUp(self):
        web_server.tasks.clear()

    def test_running_project_cannot_be_deleted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "outputs" / "active-project"
            project.mkdir(parents=True)
            (project / "script.txt").write_text("active", encoding="utf-8")
            web_server.tasks["task-1"] = {
                "project_name": "active-project",
                "status": "running",
            }

            with patch.object(web_server, "PROJECT_ROOT", root):
                response = asyncio.run(
                    web_server.api_delete_project("active-project")
                )

            project_still_exists = project.is_dir()

        self.assertEqual(response.status_code, 409)
        self.assertTrue(project_still_exists)


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
                    driver_profile="subtle_presenter",
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

    def test_generate_request_preserves_energetic_male_voice(self):
        with patch.object(web_server.threading, "Thread"):
            response = asyncio.run(
                web_server.api_generate(
                    title="voice test",
                    script="test script",
                    voice="energetic_male",
                    avatar_engine="ditto",
                    motion_mode="natural",
                    motion_style="steady",
                    motion_intensity=0.35,
                    caption_style="clean",
                    driver_profile="subtle_presenter",
                )
            )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(web_server.tasks[payload["task_id"]]["voice"], "energetic_male")

    def test_generate_request_rejects_unknown_voice(self):
        with patch.object(web_server.threading, "Thread"):
            response = asyncio.run(
                web_server.api_generate(
                    title="voice test",
                    script="test script",
                    voice="unknown",
                    avatar_engine="ditto",
                    motion_mode="natural",
                    motion_style="steady",
                    motion_intensity=0.35,
                    caption_style="clean",
                    driver_profile="subtle_presenter",
                )
            )

        self.assertEqual(response.status_code, 400)

    def test_generate_request_keeps_chinese_title_out_of_project_path(self):
        with patch.object(web_server.threading, "Thread"):
            response = asyncio.run(
                web_server.api_generate(
                    title="绿茵进化开发者介绍",
                    script="测试文案",
                    avatar_engine="ditto",
                    motion_mode="natural",
                    motion_style="steady",
                    motion_intensity=0.35,
                    caption_style="clean",
                    driver_profile="subtle_presenter",
                )
            )

        self.assertEqual(response.status_code, 200)
        project_name = json.loads(response.body)["project_name"]
        self.assertTrue(project_name.isascii())

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

    def test_generate_request_accepts_gesture_driver_profile(self):
        with patch.object(web_server.threading, "Thread"):
            response = asyncio.run(
                web_server.api_generate(
                    title="gesture test",
                    script="test script",
                    avatar_engine="classic",
                    motion_mode="gesture",
                    motion_style="steady",
                    motion_intensity=0.25,
                    caption_style="clean",
                    driver_profile="subtle_presenter",
                )
            )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        task = web_server.tasks[payload["task_id"]]
        self.assertEqual(task["driver_profile"], "subtle_presenter")
        self.assertIn("MimicMotion 手势动作", [step["name"] for step in task["steps"]])

    def test_generate_request_rejects_unknown_driver_profile(self):
        with patch.object(web_server.threading, "Thread"):
            response = asyncio.run(
                web_server.api_generate(
                    title="gesture test",
                    script="test script",
                    motion_mode="gesture",
                    driver_profile="unknown",
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
