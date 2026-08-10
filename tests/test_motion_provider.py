import importlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_environment


MODULE_NAME = "app.backend.providers.motion"


def load_motion_module(test_case: unittest.TestCase):
    spec = importlib.util.find_spec(MODULE_NAME)
    test_case.assertIsNotNone(spec, "motion provider module must exist")
    return importlib.import_module(MODULE_NAME)


class LivePortraitMotionTests(unittest.TestCase):
    def test_motion_provider_exports_mimicmotion_entrypoint(self):
        motion = load_motion_module(self)

        self.assertTrue(callable(motion.generate_gesture_motion))

    def test_missing_mimicmotion_is_reported_as_optional(self):
        with patch(
            "app.backend.providers.motion.missing_mimicmotion_files",
            return_value=[Path("missing.pth")],
        ):
            name, ready, detail = check_environment.mimicmotion_optional_status()

        self.assertEqual(name, "MimicMotion gesture motion")
        self.assertFalse(ready)
        self.assertIn("optional", detail.lower())
        self.assertIn("install_mimicmotion_runtime.ps1", detail)

    def test_build_command_uses_official_cli_and_requested_intensity(self):
        motion = load_motion_module(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "LivePortrait"
            python = root / ".venv-liveportrait" / "Scripts" / "python.exe"
            source = root / "avatar.jpg"
            template = root / "steady.pkl"
            output_dir = root / "raw"

            command, cwd, expected = motion.build_liveportrait_command(
                source,
                template,
                output_dir,
                intensity=0.35,
                runtime_root=runtime,
                python_executable=python,
            )

            self.assertEqual(command[:2], [str(python), str(runtime / "inference.py")])
            self.assertEqual(command[command.index("-s") + 1], str(source.resolve()))
            self.assertEqual(command[command.index("-d") + 1], str(template.resolve()))
            self.assertEqual(command[command.index("-o") + 1], str(output_dir.resolve()))
            self.assertEqual(
                command[command.index("--driving-multiplier") + 1], "0.35"
            )
            self.assertEqual(
                command[command.index("--source-max-dim") + 1], "1920"
            )
            self.assertEqual(cwd, runtime.resolve())
            self.assertEqual(expected, output_dir.resolve() / "avatar--steady.mp4")

    def test_missing_files_reports_the_complete_human_runtime_contract(self):
        motion = load_motion_module(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "LivePortrait"
            python = root / ".venv-liveportrait" / "Scripts" / "python.exe"
            template = root / "motion" / "templates" / "steady.pkl"

            missing = motion.missing_liveportrait_files(
                runtime_root=runtime,
                python_executable=python,
                template_path=template,
            )

            relative = {
                path.relative_to(root).as_posix() for path in missing
            }
            self.assertEqual(
                relative,
                {
                    ".venv-liveportrait/Scripts/python.exe",
                    "LivePortrait/inference.py",
                    "LivePortrait/pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
                    "LivePortrait/pretrained_weights/liveportrait/base_models/motion_extractor.pth",
                    "LivePortrait/pretrained_weights/liveportrait/base_models/spade_generator.pth",
                    "LivePortrait/pretrained_weights/liveportrait/base_models/warping_module.pth",
                    "LivePortrait/pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth",
                    "LivePortrait/pretrained_weights/liveportrait/landmark.onnx",
                    "LivePortrait/pretrained_weights/insightface/models/buffalo_l/2d106det.onnx",
                    "LivePortrait/pretrained_weights/insightface/models/buffalo_l/det_10g.onnx",
                    "motion/templates/steady.pkl",
                },
            )

    def test_loop_command_crossfades_four_copies_for_thirty_seconds(self):
        motion = load_motion_module(self)
        command = motion.build_motion_loop_command(
            Path("raw.mp4"),
            Path("motion.mp4"),
            clip_duration=10.0,
            target_duration=30.0,
            fps=25,
            fade_duration=0.2,
        )

        self.assertEqual(command.count("-i"), 4)
        filter_graph = command[command.index("-filter_complex") + 1]
        self.assertEqual(filter_graph.count("xfade=transition=fade"), 3)
        self.assertIn("duration=0.2:offset=9.8", filter_graph)
        self.assertIn("duration=0.2:offset=19.6", filter_graph)
        self.assertIn("duration=0.2:offset=29.4", filter_graph)
        self.assertEqual(command[command.index("-map") + 1], "[v3]")
        self.assertEqual(command[command.index("-t") + 1], "30.0")
        self.assertEqual(command[command.index("-r") + 1], "25")
        self.assertIn("-an", command)

    def test_loop_command_rejects_a_fade_as_long_as_the_clip(self):
        motion = load_motion_module(self)
        with self.assertRaisesRegex(ValueError, "fade duration"):
            motion.build_motion_loop_command(
                Path("raw.mp4"),
                Path("motion.mp4"),
                clip_duration=0.2,
                target_duration=2.0,
                fade_duration=0.2,
            )

    def test_validate_motion_video_rejects_non_twenty_five_fps(self):
        motion = load_motion_module(self)
        self.assertTrue(hasattr(motion, "validate_motion_video"))
        self.assertTrue(hasattr(motion, "validate_video"))
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "motion.mp4"
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
            with patch.object(motion, "validate_video", return_value=info):
                with self.assertRaisesRegex(RuntimeError, "25 fps"):
                    motion.validate_motion_video(video, fps=25)

    def test_validate_motion_video_rejects_odd_dimensions(self):
        motion = load_motion_module(self)
        self.assertTrue(hasattr(motion, "validate_motion_video"))
        self.assertTrue(hasattr(motion, "validate_video"))
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "motion.mp4"
            video.write_bytes(b"video")
            info = {
                "duration": 2.0,
                "size": 5,
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1919,
                        "height": 1080,
                        "r_frame_rate": "25/1",
                    }
                ],
            }
            with patch.object(motion, "validate_video", return_value=info):
                with self.assertRaisesRegex(RuntimeError, "even dimensions"):
                    motion.validate_motion_video(video, fps=25)

    def test_generate_motion_rejects_out_of_range_intensity(self):
        motion = load_motion_module(self)
        self.assertTrue(hasattr(motion, "generate_motion"))
        with self.assertRaisesRegex(ValueError, "between 0.0 and 1.0"):
            motion.generate_motion(
                avatar_path="avatar.jpg",
                audio_path="audio.wav",
                output_path="motion.mp4",
                intensity=1.1,
            )

    def test_generate_motion_reports_installer_for_missing_runtime(self):
        motion = load_motion_module(self)
        self.assertTrue(hasattr(motion, "generate_motion"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            avatar = root / "avatar.jpg"
            audio = root / "audio.wav"
            avatar.write_bytes(b"image")
            audio.write_bytes(b"audio")

            with self.assertRaisesRegex(
                FileNotFoundError, "install_liveportrait_runtime.ps1"
            ):
                motion.generate_motion(
                    avatar_path=str(avatar),
                    audio_path=str(audio),
                    output_path=str(root / "motion.mp4"),
                    runtime_root=root / "LivePortrait",
                    python_executable=root / ".venv-liveportrait" / "Scripts" / "python.exe",
                    template_path=root / "steady.pkl",
                )

    def test_liveportrait_subprocess_forces_utf8_on_windows(self):
        motion = load_motion_module(self)
        self.assertTrue(hasattr(motion, "build_liveportrait_environment"))
        with patch.dict(os.environ, {"PYTHONUTF8": "0", "PYTHONIOENCODING": "gbk"}):
            environment = motion.build_liveportrait_environment()

        self.assertEqual(environment["PYTHONUTF8"], "1")
        self.assertEqual(environment["PYTHONIOENCODING"], "utf-8")


if __name__ == "__main__":
    unittest.main()
