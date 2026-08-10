"""Static contract tests for the reproducible MimicMotion installer."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install_mimicmotion_runtime.ps1"


class MimicMotionInstallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = INSTALLER.read_text(encoding="utf-8")

    def test_large_models_have_pinned_sha256_checksums(self):
        expected_hashes = {
            "b812659ea273b2758c918facf759af5d7cad9564dc35156c59ec17e93f9749a4",
            "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411",
            "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843",
            "ae616c24393dd1854372b0639e5541666f7521cbe219669255e865cb7f89466a",
            "af602cd0eb4ad6086ec94fbf1438dfb1be5ec9ac03fd0215640854e90d6463a3",
        }
        for checksum in expected_hashes:
            with self.subTest(checksum=checksum):
                self.assertIn(checksum, self.script)
        self.assertIn("Get-FileHash", self.script)

    def test_svd_download_is_limited_to_runtime_files(self):
        expected_files = {
            "model_index.json",
            "feature_extractor/preprocessor_config.json",
            "image_encoder/config.json",
            "image_encoder/model.fp16.safetensors",
            "scheduler/scheduler_config.json",
            "unet/config.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        }
        for filename in expected_files:
            with self.subTest(filename=filename):
                self.assertIn(filename, self.script)
        self.assertNotIn(
            "'stabilityai/stable-video-diffusion-img2vid-xt-1-1', '--local-dir'",
            self.script,
        )

    def test_downloader_uses_xet_high_performance_mode(self):
        self.assertIn("HF_XET_HIGH_PERFORMANCE", self.script)
        self.assertIn("huggingface_hub[hf_xet]", self.script)

    def test_svd_uses_public_byte_identical_mirror(self):
        self.assertIn(
            "'weights/stable-video-diffusion-img2vid-xt-1-1'", self.script
        )
        self.assertNotIn(
            "-Repository 'stabilityai/stable-video-diffusion-img2vid-xt-1-1'",
            self.script,
        )


if __name__ == "__main__":
    unittest.main()
