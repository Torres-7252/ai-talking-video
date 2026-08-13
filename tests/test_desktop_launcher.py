"""Regression tests for the desktop one-click launcher."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "start_web_console.ps1"


class DesktopLauncherTests(unittest.TestCase):
    def test_launcher_waits_for_the_local_console_before_opening_a_browser(self):
        script = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn(".venv-liveportrait\\Scripts\\python.exe", script)
        self.assertIn("Get-NetTCPConnection", script)
        self.assertIn("Invoke-WebRequest", script)
        self.assertLess(script.index("Invoke-WebRequest"), script.index("Start-Process $Url"))


if __name__ == "__main__":
    unittest.main()
