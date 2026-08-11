import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from app.backend.providers.voice import cosyvoice


class CosyVoiceProviderTests(unittest.TestCase):
    def test_command_uses_isolated_runtime_and_energetic_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python = root / ".venv-cosyvoice" / "Scripts" / "python.exe"
            runner = root / "scripts" / "cosyvoice_runner.py"
            model = (
                root
                / "voice"
                / "models"
                / "CosyVoice"
                / "pretrained_models"
                / "CosyVoice-300M-Instruct"
            )
            output = root / "outputs" / "speech.wav"

            command = cosyvoice.build_cosyvoice_command(
                "energetic football promotion",
                output,
                {
                    "speaker": "Chinese Male",
                    "instruct": "Bright, energetic and confident.",
                },
                speed=1.2,
                project_root=root,
            )

        self.assertEqual(Path(command[0]), python)
        self.assertEqual(Path(command[1]), runner)
        self.assertIn(str(model), command)
        self.assertEqual(command[command.index("--speaker") + 1], "Chinese Male")
        self.assertEqual(command[command.index("--mode") + 1], "instruct")
        self.assertEqual(
            command[command.index("--instruct") + 1],
            "Bright, energetic and confident.",
        )
        self.assertEqual(command[command.index("--speed") + 1], "1.2")

    def test_missing_runtime_reports_installer_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch.object(cosyvoice, "PROJECT_ROOT", root),
                patch.object(
                    cosyvoice,
                    "RUNTIME_PYTHON",
                    root / ".venv-cosyvoice" / "Scripts" / "python.exe",
                ),
                patch.object(
                    cosyvoice,
                    "RUNNER",
                    root / "scripts" / "cosyvoice_runner.py",
                ),
                patch.object(
                    cosyvoice,
                    "MODEL_DIR",
                    root / "voice" / "models" / "CosyVoice-300M-Instruct",
                ),
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError, "install_cosyvoice_runtime.ps1"
                ):
                    cosyvoice.generate_cosyvoice(
                        "test",
                        str(root / "speech.wav"),
                        {"speaker": "Chinese Male", "instruct": "energetic"},
                        1.0,
                    )

    def test_runner_uses_official_instruct_api(self):
        runner = (PROJECT_ROOT / "scripts" / "cosyvoice_runner.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("AutoModel", runner)
        self.assertIn("inference_instruct", runner)
        self.assertIn("inference_sft", runner)
        self.assertIn("stream=False", runner)
        self.assertIn("text_frontend=False", runner)
        self.assertIn("RETRY_SEEDS", runner)
        self.assertIn("manual_seed", runner)
        self.assertIn("_minimum_generated_duration", runner)
        self.assertIn("generated_duration >= minimum_duration", runner)

    def test_installer_stops_when_native_dependency_install_fails(self):
        installer = (
            PROJECT_ROOT / "scripts" / "install_cosyvoice_runtime.ps1"
        ).read_text(encoding="utf-8")

        self.assertIn("$LASTEXITCODE", installer)
        self.assertIn("openai-whisper==20231117", installer)
        self.assertIn("--no-build-isolation", installer)
        self.assertIn(".venv-liveportrait", installer)
        self.assertIn("aria2c", installer)
        self.assertIn("cosyvoice-requirements-windows.txt", installer)
        requirements = (
            PROJECT_ROOT / "scripts" / "cosyvoice-requirements-windows.txt"
        ).read_text(encoding="utf-8")
        self.assertNotIn("wetext", requirements)


if __name__ == "__main__":
    unittest.main()
