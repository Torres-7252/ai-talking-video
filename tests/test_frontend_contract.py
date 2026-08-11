"""Static contracts between the web API payload and the frontend."""

from pathlib import Path
import unittest


FRONTEND = Path(__file__).resolve().parents[1] / "app" / "frontend" / "index.html"


class FrontendContractTests(unittest.TestCase):
    def test_project_preview_uses_api_file_keys_with_extensions(self):
        html = FRONTEND.read_text(encoding="utf-8")

        self.assertIn("p.files?.['final.mp4']?.url", html)
        self.assertIn("p.files?.['packaged.mp4']?.url", html)
        self.assertNotIn("p.files?.final?.url", html)


    def test_generate_form_selects_and_submits_energetic_male_voice(self):
        html = FRONTEND.read_text(encoding="utf-8")

        self.assertIn('select id="voice"', html)
        self.assertIn('value="energetic_male" selected', html)
        self.assertIn(
            "fd.append('voice', document.getElementById('voice').value)",
            html,
        )


if __name__ == "__main__":
    unittest.main()
