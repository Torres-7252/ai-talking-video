import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROVIDERS_ROOT = PROJECT_ROOT / "app" / "backend" / "providers"
sys.path.insert(0, str(PROVIDERS_ROOT))

import voice


class VoiceProviderTests(unittest.TestCase):
    def test_voice_provider_configures_runtime_dll_directory(self):
        self.assertIn("RUNTIME_LIBRARY", voice.__dict__)
        self.assertIn("add_dll_directory", (PROJECT_ROOT / "app" / "backend" / "providers" / "voice" / "__init__.py").read_text(encoding="utf-8"))

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

    def test_project_has_energetic_male_cosyvoice_profile(self):
        profile = voice.load_configured_voice_profile("energetic_male")

        self.assertEqual(profile["name"], "活力男声")
        self.assertEqual(profile["provider"], "cosyvoice")
        self.assertEqual(profile["speaker"], "中文男")
        self.assertEqual(profile["mode"], "sft")
        self.assertGreater(profile["speed_multiplier"], 1.0)
        self.assertIn("passionate", profile["instruct"])
        self.assertTrue(profile["instruct"].endswith("<|endofprompt|>"))

    def test_generate_voice_dispatches_cosyvoice_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "speech.wav"
            with mock.patch(
                "voice.cosyvoice.generate_cosyvoice",
                return_value=output,
            ) as generate_cosyvoice:
                result = voice.generate_voice(
                    "绿茵进化",
                    str(output),
                    voice_profile="energetic_male",
                    speed=1.05,
                )

        self.assertEqual(result, output)
        self.assertEqual(generate_cosyvoice.call_count, 1)
        args = generate_cosyvoice.call_args.args
        self.assertEqual(args[0], "绿茵进化。")
        self.assertEqual(args[1], str(output))
        self.assertEqual(args[2]["speaker"], "中文男")
        self.assertEqual(args[3], 1.05)

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
            "大家好，我是AI足球教练。今天我们来聊一个很多球友都关心的问题，为什么你的第一脚触球总是停不好？其实关键只有三点。",
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

    def test_reference_audio_is_trimmed_before_synthesis(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "reference.wav"
            sample_rate = 32000
            audio = np.concatenate(
                [
                    np.zeros(sample_rate),
                    np.full(sample_rate * 4, 0.2),
                    np.zeros(sample_rate),
                ]
            )
            sf.write(source, audio, sample_rate)

            prepared = voice._prepare_reference_audio(source, root / "cache")
            prepared_audio, prepared_rate = sf.read(prepared)

        self.assertEqual(prepared_rate, sample_rate)
        self.assertGreater(len(prepared_audio) / prepared_rate, 4.1)
        self.assertLess(len(prepared_audio) / prepared_rate, 4.5)

    def test_minimum_duration_scales_with_spoken_text(self):
        short = voice._minimum_generated_duration("hello", speed=1.0)
        sentence = voice._minimum_generated_duration(
            "hello this is a complete sentence", speed=1.0
        )
        faster = voice._minimum_generated_duration(
            "hello this is a complete sentence", speed=1.25
        )

        self.assertGreater(sentence, short)
        self.assertLess(faster, sentence)

    def test_generate_voice_retries_audio_that_is_too_short(self):
        sample_rate = 24000
        generated = [
            [(sample_rate, np.zeros(sample_rate))],
            [(sample_rate, np.zeros(sample_rate * 4))],
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.wav"
            reference.touch()
            output = root / "speech.wav"
            with (
                mock.patch(
                    "voice.load_voice_profile",
                    return_value={
                        "reference_audio": reference,
                        "reference_text": "reference transcript",
                        "language": "zh",
                    },
                ),
                mock.patch("voice._prepare_reference_audio", return_value=reference),
                mock.patch("voice._get_tts", return_value=object()),
                mock.patch("voice._run_tts", side_effect=generated) as run_tts,
                mock.patch(
                    "voice.validate_audio",
                    return_value={"size": 192044, "duration": 4.0},
                ),
            ):
                result = voice.generate_voice(
                    "hello this is a complete sentence", str(output)
                )

            written, written_rate = sf.read(result)

        self.assertEqual(run_tts.call_count, 2)
        self.assertEqual(run_tts.call_args_list[0].args[1]["seed"], 42)
        self.assertEqual(run_tts.call_args_list[1].args[1]["seed"], 2026)
        self.assertFalse(run_tts.call_args_list[0].args[1]["parallel_infer"])
        self.assertEqual(written_rate, sample_rate)
        self.assertEqual(len(written), sample_rate * 4)

    def test_reference_window_uses_matching_asr_timestamps(self):
        result = [
            {
                "text": "intro hello world extra",
                "timestamp": [
                    [0, 400],
                    [500, 900],
                    [900, 1300],
                    [1400, 1800],
                ],
            }
        ]

        start, end = voice._find_reference_window(result, "hello world")

        self.assertEqual(start, 0.5)
        self.assertEqual(end, 1.3)


if __name__ == "__main__":
    unittest.main()
