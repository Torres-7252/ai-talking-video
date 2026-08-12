"""Tests for web project path confinement."""

import asyncio
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
    def test_completed_task_does_not_hide_exported_project(self):
        web_server.tasks.clear()
        web_server.tasks["task-1"] = {
            "id": "task-1",
            "project_name": "completed-project",
            "title": "已完成项目",
            "status": "completed",
            "steps": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            project_dir = output_root / ".work" / "completed-project"
            project_dir.mkdir(parents=True)
            final_file = project_dir / "final.mp4"
            final_file.write_bytes(b"video")
            (project_dir / "metadata.json").write_text(
                json.dumps({"title": "已完成项目", "final_path": str(final_file)}),
                encoding="utf-8",
            )
            with patch.object(web_server, "OUTPUTS_ROOT", output_root):
                projects = web_server.get_projects()

        self.assertTrue(projects[0]["has_video"])

    def test_projects_includes_active_task_before_metadata_exists(self):
        web_server.tasks.clear()
        web_server.tasks["task-1"] = {
            "id": "task-1",
            "project_name": "active-project",
            "title": "正在生成的长视频",
            "status": "running",
            "steps": [{"name": "声音生成", "status": "running"}],
            "segment_count": 5,
            "current_segment": 2,
            "completed_segments": 1,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            (output_root / ".work" / "active-project").mkdir(parents=True)
            with patch.object(web_server, "OUTPUTS_ROOT", output_root):
                projects = web_server.get_projects()

        self.assertEqual(projects[0]["name"], "active-project")
        self.assertEqual(projects[0]["status"], "running")
        self.assertEqual(projects[0]["current_segment"], 2)

    def test_project_file_rejects_parent_escape(self):
        with self.assertRaises(ValueError):
            web_server.resolve_project_file("project", "../../.env")

    def test_project_name_rejects_parent_escape(self):
        with self.assertRaises(ValueError):
            web_server.resolve_project_file("..", "final.mp4")

    def test_project_file_stays_under_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = Path(temp_dir).resolve()
            with patch.object(web_server, "OUTPUTS_ROOT", outputs):
                path = web_server.resolve_project_file("project", "final.mp4")

        self.assertTrue(path.is_relative_to(outputs / ".work"))

    def test_default_output_directory_is_requested_drive(self):
        self.assertEqual(web_server.OUTPUTS_ROOT, Path(r"E:\ai口播输出").resolve())

    def test_open_outputs_uses_configured_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = Path(temp_dir) / "exports"
            with (
                patch.object(web_server, "OUTPUTS_ROOT", outputs),
                patch.object(web_server.subprocess, "Popen") as popen,
                patch.object(web_server.sys, "platform", "win32"),
            ):
                response = asyncio.run(web_server.api_open_outputs())

        self.assertEqual(response.status_code, 200)
        popen.assert_called_once_with(["explorer.exe", str(outputs)])


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

    def test_voice_upload_does_not_require_asr_alignment_before_replacing(self):
        observed = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def convert_audio(command, **kwargs):
                Path(command[-1]).write_bytes(b"converted audio")
                return SimpleNamespace(returncode=0, stderr=b"")

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
        align_reference.assert_not_called()
        self.assertEqual(saved_audio, b"converted audio")


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


class WebGenerationTests(unittest.TestCase):
    def setUp(self):
        web_server.tasks.clear()

    def test_generate_request_uses_ditto_only_pipeline(self):
        with patch.object(web_server.threading, "Thread"):
            response = asyncio.run(
                web_server.api_generate(
                    title="motion test",
                    script="test script",
                    voice="default",
                    speed=1.0,
                    template="talking_head",
                    resume=False,
                    caption_style="pop",
                )
            )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        task = web_server.tasks[payload["task_id"]]
        self.assertEqual(task["avatar_engine"], "ditto")
        self.assertEqual(task["caption_style"], "pop")
        self.assertEqual(
            [step["name"] for step in task["steps"]],
            ["声音生成", "Ditto 真实数字人", "字幕生成", "自适应比例合成", "最终导出"],
        )

    def test_generate_request_rejects_an_overlong_ditto_script(self):
        with patch.object(web_server.threading, "Thread"):
            response = asyncio.run(
                web_server.api_generate(
                    title="long script", script="测" * 161, avatar_engine="ditto"
                )
            )

        self.assertEqual(response.status_code, 200)

    def test_generate_request_splits_long_ditto_script_into_sentence_safe_segments(self):
        first = "第一段" + "内容" * 65 + "。"
        second = "第二段" + "内容" * 65 + "。"
        third = "第三段" + "内容" * 65 + "。"
        script = first + second + third
        with patch.object(web_server.threading, "Thread"):
            response = asyncio.run(
                web_server.api_generate(
                    title="long script", script=script, speed=1.0, avatar_engine="ditto"
                )
            )

        self.assertEqual(response.status_code, 200)
        task = web_server.tasks[json.loads(response.body)["task_id"]]
        self.assertEqual(task["segment_count"], 3)
        self.assertEqual(task["segments"], [first, second, third])

    def test_generate_request_accepts_selected_material_ids(self):
        avatar = {"id": "avatar123", "name": "Host", "file_path": Path("C:/tmp/host.jpg")}
        voice = {"id": "voice123", "name": "Narrator", "file_path": Path("C:/tmp/voice.wav"), "reference_text": "测试音频"}
        with (
            patch.object(web_server.threading, "Thread"),
            patch.object(web_server, "get_asset", side_effect=[avatar, voice]),
        ):
            response = asyncio.run(
                web_server.api_generate(
                    title="material test",
                    script="test script",
                    voice="default",
                    caption_style="clean",
                    avatar_asset_id="avatar123",
                    voice_asset_id="voice123",
                )
            )

        self.assertEqual(response.status_code, 200)
        task = web_server.tasks[json.loads(response.body)["task_id"]]
        self.assertEqual(task["avatar_asset_id"], "avatar123")
        self.assertEqual(task["voice_asset_id"], "voice123")

    def test_generate_request_preserves_energetic_male_voice(self):
        with patch.object(web_server.threading, "Thread"):
            response = asyncio.run(
                web_server.api_generate(
                    title="voice test",
                    script="test script",
                    voice="energetic_male",
                    caption_style="clean",
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
                    caption_style="clean",
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

    def test_generate_request_rejects_removed_classic_engine(self):
        with patch.object(web_server.threading, "Thread"):
            response = asyncio.run(
                web_server.api_generate(
                    title="classic test",
                    script="test script",
                    avatar_engine="classic",
                )
            )

        self.assertEqual(response.status_code, 400)

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
