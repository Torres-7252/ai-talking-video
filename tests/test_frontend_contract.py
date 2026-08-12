"""Static contracts between the web API payload and the frontend."""

from pathlib import Path
import unittest


FRONTEND = Path(__file__).resolve().parents[1] / "app" / "frontend" / "index.html"


class FrontendContractTests(unittest.TestCase):
    def test_project_preview_uses_api_file_keys_with_extensions(self):
        html = FRONTEND.read_text(encoding="utf-8")

        self.assertIn("p.metadata.final_filename", html)
        self.assertIn("p.files?.['packaged.mp4']?.url", html)
        self.assertNotIn("p.files?.final?.url", html)

    def test_project_preview_does_not_loop(self):
        html = FRONTEND.read_text(encoding="utf-8")

        self.assertIn("<video controls autoplay", html)
        self.assertNotIn("<video controls autoplay loop", html)

    def test_active_task_has_http_polling_fallback(self):
        html = FRONTEND.read_text(encoding="utf-8")

        self.assertIn("function pollTaskStatus()", html)
        self.assertIn("/api/tasks/${currentTask}", html)


    def test_generate_form_lists_library_reference_voices_in_the_tone_selector(self):
        html = FRONTEND.read_text(encoding="utf-8")

        self.assertIn('select id="voice"', html)
        self.assertIn("fillMaterialSelect('voice', voices, 'default-voice')", html)
        self.assertNotIn('id="voiceAsset"', html)
        self.assertIn(
            "fd.append('voice', 'default')",
            html,
        )
        self.assertIn("fd.append('voice_asset_id', document.getElementById('voice').value)", html)

    def test_material_library_can_select_upload_and_rename_assets(self):
        html = FRONTEND.read_text(encoding="utf-8")

        self.assertIn('id="avatarAsset"', html)
        self.assertIn("/api/materials/avatar", html)
        self.assertIn("/api/materials/voice", html)
        self.assertIn("renameMaterial", html)
        self.assertIn("deleteMaterial", html)
        self.assertIn("/api/materials/${kind}/${id}", html)
        self.assertIn("fd.append('avatar_asset_id'", html)
        self.assertIn("fd.append('voice_asset_id'", html)


if __name__ == "__main__":
    unittest.main()
