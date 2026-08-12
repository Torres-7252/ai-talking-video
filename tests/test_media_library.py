import tempfile
import unittest
from pathlib import Path

from scripts.media_library import add_asset, delete_asset, get_asset, list_assets, rename_asset


class MediaLibraryTests(unittest.TestCase):
    def test_library_lists_defaults_and_named_uploaded_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "avatar").mkdir()
            (root / "voice" / "references").mkdir(parents=True)
            (root / "avatar" / "avatar.jpg").write_bytes(b"avatar")
            (root / "voice" / "references" / "default.wav").write_bytes(b"voice")
            uploaded = root / "materials" / "avatars" / "custom.jpg"
            uploaded.parent.mkdir(parents=True)
            uploaded.write_bytes(b"custom")

            asset = add_asset("avatar", "Office Host", uploaded, project_root=root)
            library = list_assets(root)
            resolved = get_asset("avatar", asset["id"], root)["file_path"]

        self.assertEqual({item["name"] for item in library["avatar"]}, {"默认人物", "Office Host"})
        self.assertEqual(resolved, uploaded.resolve())

    def test_library_renames_without_changing_the_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            uploaded = root / "materials" / "voices" / "sample.wav"
            uploaded.parent.mkdir(parents=True)
            uploaded.write_bytes(b"voice")
            asset = add_asset("voice", "Original", uploaded, reference_text="hello", project_root=root)

            renamed = rename_asset("voice", asset["id"], "New Name", root)

        self.assertEqual(renamed["name"], "New Name")
        self.assertEqual(renamed["file_path"], uploaded.resolve())

    def test_library_deletes_uploaded_asset_and_its_material_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            uploaded = root / "materials" / "voices" / "sample.wav"
            uploaded.parent.mkdir(parents=True)
            uploaded.write_bytes(b"voice")
            asset = add_asset("voice", "Temporary", uploaded, project_root=root)

            deleted = delete_asset("voice", asset["id"], root)

            self.assertEqual(deleted["id"], asset["id"])
            self.assertFalse(uploaded.exists())
            self.assertEqual(list_assets(root)["voice"], [])

    def test_library_cannot_delete_system_asset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "avatar").mkdir()
            (root / "avatar" / "avatar.jpg").write_bytes(b"avatar")

            with self.assertRaises(ValueError):
                delete_asset("avatar", "default-avatar", root)
