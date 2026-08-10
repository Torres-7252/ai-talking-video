"""Tests for the Ditto audio-driven avatar provider."""

import tempfile
import unittest
from pathlib import Path


class DittoProviderTests(unittest.TestCase):
    def test_missing_model_files_reports_only_absent_paths(self):
        from app.backend.providers.avatar import REQUIRED_MODEL_FILES, missing_model_files

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            present = REQUIRED_MODEL_FILES[0]
            target = root / present
            target.parent.mkdir(parents=True)
            target.write_bytes(b"model")

            missing = missing_model_files(root)

        self.assertNotIn(present, missing)
        self.assertEqual(set(missing), set(REQUIRED_MODEL_FILES[1:]))

    def test_inference_command_uses_pytorch_checkpoint_and_absolute_media_paths(self):
        from app.backend.providers.avatar import build_inference_command

        root = Path("C:/ditto").resolve()
        python = Path("C:/python/python.exe").resolve()
        source = Path("C:/media/avatar.jpg").resolve()
        audio = Path("C:/media/audio.wav").resolve()
        output = Path("C:/media/talking.mp4").resolve()

        command = build_inference_command(
            source, audio, output, ditto_root=root, python_executable=python
        )

        self.assertEqual(command[0], str(python))
        self.assertEqual(command[1], str(root / "inference.py"))
        self.assertIn(str(root / "checkpoints" / "ditto_pytorch"), command)
        self.assertIn(
            str(root / "checkpoints" / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl"),
            command,
        )
        self.assertIn(str(source), command)
        self.assertIn(str(audio), command)
        self.assertIn(str(output), command)


if __name__ == "__main__":
    unittest.main()
