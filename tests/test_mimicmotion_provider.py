"""Tests for the isolated MimicMotion gesture adapter."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.backend.providers.motion import mimicmotion


class MimicMotionProviderTests(unittest.TestCase):
    def test_command_uses_isolated_runner_and_low_vram_defaults(self):
        command, cwd = mimicmotion.build_mimicmotion_command(
            Path("avatar.png"),
            Path("driver.mp4"),
            Path("motion.mp4"),
            duration=10.0,
            intensity=0.25,
            runtime_root=Path("runtime"),
            python_executable=Path("python.exe"),
            runner_path=Path("runner.py"),
        )

        self.assertEqual(command[0], str(Path("python.exe").resolve()))
        self.assertEqual(command[command.index("--tile-size") + 1], "16")
        self.assertEqual(command[command.index("--tile-overlap") + 1], "6")
        self.assertEqual(command[command.index("--decode-chunk-size") + 1], "1")
        self.assertEqual(command[command.index("--fps") + 1], "15")
        self.assertEqual(command[command.index("--steps") + 1], "25")
        self.assertEqual(cwd, Path("runtime").resolve())

    def test_missing_files_reports_complete_runtime_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = mimicmotion.missing_mimicmotion_files(
                runtime_root=root / "runtime",
                python_executable=root / "python.exe",
                runner_path=root / "runner.py",
            )

        names = {path.name for path in missing}
        self.assertIn("python.exe", names)
        self.assertIn("inference.py", names)
        self.assertIn("MimicMotion_1-1.pth", names)
        self.assertIn("yolox_l.onnx", names)
        self.assertIn("dw-ll_ucoco_384.onnx", names)
        self.assertIn("model_index.json", names)
        self.assertIn("runner.py", names)

    def test_invalid_duration_and_intensity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duration"):
            mimicmotion.generate_gesture_motion("a.png", "d.mp4", "o.mp4", 0)
        with self.assertRaisesRegex(ValueError, "intensity"):
            mimicmotion.generate_gesture_motion(
                "a.png", "d.mp4", "o.mp4", 1, intensity=1.1
            )

    def test_cuda_oom_retries_once_at_448_and_replaces_atomically(self):
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            python = root / "python.exe"
            runner = root / "runner.py"
            avatar = root / "avatar.png"
            driver = root / "driver.mp4"
            output = root / "motion.mp4"
            for path in (python, runner, avatar, driver):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")

            def fake_missing(**_kwargs):
                return []

            def fake_run(command, **_kwargs):
                calls.append(command)
                if len(calls) == 1:
                    return subprocess.CompletedProcess(
                        command, 1, "", "CUDA out of memory"
                    )
                generated = Path(command[command.index("--output") + 1])
                generated.write_bytes(b"valid video")
                return subprocess.CompletedProcess(command, 0, "ok", "")

            with patch.object(
                mimicmotion, "missing_mimicmotion_files", side_effect=fake_missing
            ), patch.object(mimicmotion.subprocess, "run", side_effect=fake_run), patch.object(
                mimicmotion, "validate_gesture_video", return_value={"duration": 2.0}
            ):
                result = mimicmotion.generate_gesture_motion(
                    str(avatar),
                    str(driver),
                    str(output),
                    duration=2.0,
                    runtime_root=runtime,
                    python_executable=python,
                    runner_path=runner,
                )

        self.assertEqual(result, output.resolve())
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][calls[0].index("--resolution") + 1], "576")
        self.assertEqual(calls[1][calls[1].index("--resolution") + 1], "448")

    def test_environment_enables_utf8_and_cuda_allocator(self):
        environment = mimicmotion.build_mimicmotion_environment()

        self.assertEqual(environment["PYTHONUTF8"], "1")
        self.assertEqual(environment["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(
            environment["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:256"
        )


if __name__ == "__main__":
    unittest.main()
