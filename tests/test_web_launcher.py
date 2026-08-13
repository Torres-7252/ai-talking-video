"""Regression tests for the one-click web console launcher."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "启动控制台.ps1"


class WebLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = LAUNCHER.read_text(encoding="utf-8-sig")

    def test_launcher_selects_runtime_with_full_pipeline_dependencies(self):
        self.assertIn(".venv-liveportrait\\Scripts\\python.exe", self.script)
        self.assertIn("Get-Command py.exe", self.script)
        self.assertIn("pytorch_lightning", self.script)
        self.assertIn("funasr", self.script)
        self.assertIn('$ErrorActionPreference = "SilentlyContinue"', self.script)
        self.assertIn("& $WebPython scripts\\web_server.py", self.script)

    def test_launcher_prioritizes_the_project_runtime_over_system_python(self):
        project_runtime = '(Join-Path $ProjectRoot ".venv-liveportrait\\Scripts\\python.exe")'
        self.assertLess(
            self.script.index(project_runtime),
            self.script.index("Get-Command py.exe"),
        )

    def test_failed_generation_preflight_does_not_block_asset_upload_console(self):
        self.assertIn("The console will still start so assets can be managed", self.script)


if __name__ == "__main__":
    unittest.main()
