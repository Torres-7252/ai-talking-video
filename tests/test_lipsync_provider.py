"""Contract tests for the MuseTalk 1.5 adapter."""

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
