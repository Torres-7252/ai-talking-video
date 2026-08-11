"""Contract tests for the MuseTalk 1.5 adapter."""

import tempfile
import unittest
import os
import pickle
import subprocess
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from app.backend.providers import lipsync


class MuseTalkJobTests(unittest.TestCase):
    def test_job_workspace_does_not_inherit_unicode_output_path(self):
        output = Path("outputs") / "中文项目" / "talking.mp4"

        config, result_dir, generated = lipsync._job_paths(output)

        self.assertTrue(str(config).isascii())
        self.assertTrue(str(result_dir).isascii())
        self.assertTrue(str(generated).isascii())
        self.assertTrue(config.is_relative_to(lipsync.MUSETALK_JOBS_PATH))

    def test_staged_inputs_have_ascii_paths_and_preserve_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "人物动作.mp4"
            source.write_bytes(b"motion bytes")
            work_dir = Path(temp_dir) / "ascii-work"

            staged = lipsync.stage_musetalk_input(source, work_dir, "source")

            self.assertTrue(str(staged).isascii())
            self.assertEqual(staged.read_bytes(), source.read_bytes())

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
        self.assertIn("--saved_coord", command)
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

    def test_mouth_detail_mask_tracks_the_lower_face_and_fades_softly(self):
        mask = lipsync.build_mouth_detail_mask(
            frame_shape=(940, 1672),
            face_box=(767, 259, 1021, 571),
            strength=0.35,
        )

        self.assertEqual(mask.shape, (940, 1672, 1))
        self.assertAlmostEqual(float(mask.max()), 0.35, places=2)
        self.assertGreater(float(mask[484, 894, 0]), 0.34)
        self.assertLess(float(mask[259, 894, 0]), 0.01)
        self.assertEqual(float(mask[0, 0, 0]), 0.0)
        self.assertTrue(np.all(mask >= 0.0))
        self.assertTrue(np.all(mask <= 0.35))

    def test_blend_mouth_detail_restores_source_texture_at_reduced_strength(self):
        generated = np.zeros((100, 100, 3), dtype=np.uint8)
        source = np.full((100, 100, 3), 200, dtype=np.uint8)

        blended = lipsync.blend_mouth_detail(
            generated,
            source,
            face_box=(25, 10, 75, 90),
            strength=0.35,
        )

        self.assertTrue(np.allclose(blended[68, 50], 70, atol=1))
        self.assertTrue(np.array_equal(blended[0, 0], generated[0, 0]))

    def test_restore_mouth_detail_preserves_audio_and_only_softens_mouth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "source.mp4"
            generated = temp / "generated.mp4"
            coords = temp / "source.pkl"
            output = temp / "refined.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "color=c=white:s=100x100:r=25:d=0.4",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "color=c=black:s=100x100:r=25:d=0.4",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=0.4",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(generated),
                ],
                check=True,
            )
            with coords.open("wb") as coord_file:
                pickle.dump([(25, 10, 75, 90)] * 10, coord_file)

            lipsync.restore_mouth_detail(generated, source, coords, output)

            capture = cv2.VideoCapture(str(output))
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertGreater(int(frame[68, 50, 0]), 60)
            self.assertLess(int(frame[0, 0, 0]), 10)
            info = lipsync.validate_video(output)
            self.assertTrue(
                any(stream.get("codec_type") == "audio" for stream in info["streams"])
            )

    def test_finalize_video_output_uses_saved_face_tracking_for_refinement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            generated = temp / "results" / "v15" / "talking.mp4"
            motion = temp / "motion.mp4"
            output = temp / "talking.mp4"
            result_dir = temp / "results"
            with patch.object(
                lipsync, "restore_mouth_detail", return_value=output
            ) as restore:
                result = lipsync.finalize_lipsync_output(
                    generated, motion, result_dir, output
                )

        self.assertEqual(result, output)
        restore.assert_called_once_with(
            generated,
            motion,
            temp / "motion.pkl",
            output,
            strength=0.35,
        )


if __name__ == "__main__":
    unittest.main()
