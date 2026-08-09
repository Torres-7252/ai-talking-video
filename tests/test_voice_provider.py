import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROVIDERS_ROOT = PROJECT_ROOT / "app" / "backend" / "providers"
sys.path.insert(0, str(PROVIDERS_ROOT))

import voice


class VoiceProviderTests(unittest.TestCase):
    def test_provider_can_be_imported_through_project_package(self):
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
            "from app.backend.providers.voice import build_tts_config; "
            "print(build_tts_config()['custom']['version'])"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "v3")

    def test_project_voice_profile_config_is_valid_json(self):
        config_path = PROJECT_ROOT / "config" / "profiles.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        default_voice = next(item for item in config["voices"] if item["id"] == "default")
        self.assertEqual(default_voice["reference_audio"], "./voice/references/default.wav")
        self.assertEqual(
            default_voice["reference_text"],
            "大家好，我是AI足球教练。今天我们来聊一个很多球友都关心的问题。",
        )

    def test_tts_config_uses_existing_absolute_model_paths(self):
        config = voice.build_tts_config()["custom"]

        for key in (
            "t2s_weights_path",
            "vits_weights_path",
            "bert_base_path",
            "cnhuhbert_base_path",
        ):
            model_path = Path(config[key])
            self.assertTrue(model_path.is_absolute(), key)
            self.assertTrue(model_path.exists(), f"{key}: {model_path}")

        hubert_path = Path(config["cnhuhbert_base_path"])
        self.assertTrue((hubert_path / "pytorch_model.bin").is_file())

    def test_profile_uses_reference_recording_transcript(self):
        profile = voice.load_voice_profile("default")

        self.assertEqual(profile["language"], "zh")
        self.assertEqual(
            profile["reference_text"],
            "大家好，我是AI足球教练。今天我们来聊一个很多球友都关心的问题。",
        )
        self.assertEqual(profile["reference_audio"], PROJECT_ROOT / "voice" / "references" / "default.wav")

    def test_tts_module_captures_sovits_root_as_working_directory(self):
        tts_class, _ = voice._import_tts_api()
        tts_module = sys.modules[tts_class.__module__]

        self.assertEqual(Path(tts_module.now_dir).resolve(), voice.SOVITS_ROOT.resolve())

    def test_tts_constructors_run_inside_sovits_root_and_restore_cwd(self):
        observed_cwds = []
        original_cwd = Path.cwd()

        class FakeConfig:
            def __init__(self, config):
                observed_cwds.append(Path.cwd())
                self.config = config

        class FakeTTS:
            def __init__(self, config):
                observed_cwds.append(Path.cwd())
                self.config = config

        with mock.patch("voice._import_tts_api", return_value=(FakeTTS, FakeConfig)):
            instance = voice._create_tts()

        self.assertIsInstance(instance, FakeTTS)
        self.assertEqual(observed_cwds, [voice.SOVITS_ROOT, voice.SOVITS_ROOT])
        self.assertEqual(Path.cwd(), original_cwd)

    def test_tts_generator_is_consumed_inside_sovits_root(self):
        observed_cwds = []
        original_cwd = Path.cwd()

        class FakeTTS:
            def run(self, payload):
                observed_cwds.append(Path.cwd())
                yield 32000, [0.0, 0.1]

        generated = voice._run_tts(FakeTTS(), {"text": "测试"})

        self.assertEqual(len(generated), 1)
        self.assertEqual(observed_cwds, [voice.SOVITS_ROOT])
        self.assertEqual(Path.cwd(), original_cwd)


if __name__ == "__main__":
    unittest.main()
