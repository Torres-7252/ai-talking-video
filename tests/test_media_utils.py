import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROVIDERS_ROOT = PROJECT_ROOT / "app" / "backend" / "providers"
sys.path.insert(0, str(PROVIDERS_ROOT))

import media_utils


class MediaValidationTests(unittest.TestCase):
    def test_validate_audio_rejects_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            media_utils.validate_audio(Path("missing.wav"))

    def test_probe_media_normalizes_ffprobe_output(self):
        payload = {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "audio",
                    "codec_name": "pcm_s16le",
                    "sample_rate": "32000",
                    "channels": 1,
                }
            ],
            "format": {"duration": "2.5", "size": "160044"},
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload), stderr=""
        )

        with tempfile.TemporaryDirectory() as tmp:
            media_path = Path(tmp) / "sample.wav"
            media_path.touch()
            with mock.patch("media_utils.subprocess.run", return_value=completed):
                info = media_utils.probe_media(media_path)

        self.assertEqual(info["duration"], 2.5)
        self.assertEqual(info["size"], 160044)
        self.assertEqual(info["streams"][0]["codec_type"], "audio")

    def test_validate_audio_rejects_media_without_audio_stream(self):
        info = {
            "duration": 2.0,
            "size": 1024,
            "streams": [{"codec_type": "video", "codec_name": "h264"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            media_path = Path(tmp) / "not-audio.mp4"
            media_path.touch()
            with mock.patch("media_utils.probe_media", return_value=info):
                with self.assertRaisesRegex(RuntimeError, "audio stream"):
                    media_utils.validate_audio(media_path)


if __name__ == "__main__":
    unittest.main()
