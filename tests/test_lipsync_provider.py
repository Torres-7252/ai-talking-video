"""Contract tests for the MuseTalk 1.5 adapter."""

import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.backend.providers import lipsync


class MuseTalkJobTests(unittest.TestCase):
    def test_build_job_uses_official_v15_cli_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            avatar = temp / "avatar.jpg"
            audio = temp / "audio.wav"
            output = temp / "talking.mp4"

            job, command, cwd = lipsync.build_musetalk_job(
                avatar, audio, output, use_fp16=True
            )

        self.assertEqual(cwd, lipsync.MUSETALK_PATH)
        self.assertEqual(job["task_0"]["video_path"], str(avatar.resolve()))
        self.assertEqual(job["task_0"]["audio_path"], str(audio.resolve()))
        self.assertIn("-m", command)
        self.assertIn("scripts.inference", command)
        self.assertIn("--version", command)
        self.assertEqual(command[command.index("--version") + 1], "v15")
        self.assertIn("--use_float16", command)
        self.assertIn("--inference_config", command)
        self.assertNotIn("--avatar", command)
        self.assertNotIn("--audio", command)
        self.assertNotIn("--fp16", command)

    def test_required_models_include_v15_runtime_dependencies(self):
        relative_paths = {
            path.relative_to(lipsync.MUSETALK_PATH).as_posix()
            for path in lipsync.required_musetalk_files()
        }

        self.assertIn("models/musetalkV15/unet.pth", relative_paths)
        self.assertIn("models/sd-vae/diffusion_pytorch_model.bin", relative_paths)
        self.assertIn("models/whisper/pytorch_model.bin", relative_paths)
        self.assertIn("models/dwpose/dw-ll_ucoco_384.pth", relative_paths)
        self.assertIn("models/face-parse-bisent/79999_iter.pth", relative_paths)

    def test_large_downloads_have_exact_expected_sizes(self):
        self.assertEqual(
            lipsync.EXPECTED_MODEL_SIZES["models/musetalkV15/unet.pth"],
            3_400_074_924,
        )
        self.assertEqual(
            lipsync.EXPECTED_MODEL_SIZES[
                "musetalk/utils/face_detection/detection/sfd/s3fd.pth"
            ],
            89_843_225,
        )

    def test_runtime_environment_prepends_compatibility_modules(self):
        environment = lipsync.build_musetalk_environment()

        self.assertEqual(
            Path(environment["PYTHONPATH"].split(os.pathsep)[0]),
            lipsync.COMPAT_PATH,
        )

    def test_odd_avatar_is_padded_to_even_png(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            avatar = temp / "avatar.jpg"
            output = temp / "talking.mp4"
            Image.new("RGB", (7, 4), "white").save(avatar)

            normalized = lipsync.prepare_even_avatar(avatar, output)

            self.assertNotEqual(normalized, avatar)
            with Image.open(normalized) as image:
                self.assertEqual(image.size, (8, 4))

    def test_valid_motion_video_is_passed_to_musetalk_unchanged(self):
        self.assertTrue(hasattr(lipsync, "prepare_lipsync_input"))
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "motion.mp4"
            video.write_bytes(b"video")
            info = {
                "duration": 2.0,
                "size": 5,
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1920,
                        "height": 1080,
                        "r_frame_rate": "25/1",
                    }
                ],
            }
            with patch.object(lipsync, "validate_video", return_value=info):
                prepared = lipsync.prepare_lipsync_input(
                    video, Path(temp_dir) / "talking.mp4"
                )

        self.assertEqual(prepared, video.resolve())

    def test_motion_video_with_wrong_frame_rate_is_rejected(self):
        self.assertTrue(hasattr(lipsync, "prepare_lipsync_input"))
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "motion.mp4"
            video.write_bytes(b"video")
            info = {
                "duration": 2.0,
                "size": 5,
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1920,
                        "height": 1080,
                        "r_frame_rate": "30/1",
                    }
                ],
            }
            with patch.object(lipsync, "validate_video", return_value=info):
                with self.assertRaisesRegex(RuntimeError, "25 fps"):
                    lipsync.prepare_lipsync_input(
                        video, Path(temp_dir) / "talking.mp4"
                    )


if __name__ == "__main__":
    unittest.main()
