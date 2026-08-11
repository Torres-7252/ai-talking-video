"""Isolated CosyVoice provider for selectable preset voices."""

from __future__ import annotations

from pathlib import Path

from app.backend.providers.media_utils import run_checked, validate_audio


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNTIME_PYTHON = PROJECT_ROOT / ".venv-cosyvoice" / "Scripts" / "python.exe"
RUNNER = PROJECT_ROOT / "scripts" / "cosyvoice_runner.py"
MODEL_DIR = (
    PROJECT_ROOT
    / "voice"
    / "models"
    / "CosyVoice"
    / "pretrained_models"
    / "CosyVoice-300M-Instruct"
)


def build_cosyvoice_command(
    text: str,
    output_path: Path,
    profile: dict,
    speed: float,
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[str]:
    root = Path(project_root).resolve()
    return [
        str(root / ".venv-cosyvoice" / "Scripts" / "python.exe"),
        str(root / "scripts" / "cosyvoice_runner.py"),
        "--model-dir",
        str(
            root
            / "voice"
            / "models"
            / "CosyVoice"
            / "pretrained_models"
            / "CosyVoice-300M-Instruct"
        ),
        "--text",
        text,
        "--output",
        str(Path(output_path).resolve()),
        "--speaker",
        str(profile.get("speaker") or "中文男"),
        "--mode",
        str(profile.get("mode") or "instruct"),
        "--instruct",
        str(profile.get("instruct") or "Energetic and confident."),
        "--speed",
        str(float(speed) * float(profile.get("speed_multiplier") or 1.0)),
    ]


def generate_cosyvoice(
    text: str,
    output_path: str,
    profile: dict,
    speed: float,
) -> Path:
    missing = [
        path
        for path in (RUNTIME_PYTHON, RUNNER, MODEL_DIR / "cosyvoice.yaml")
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "CosyVoice runtime is missing. Run "
            "scripts\\install_cosyvoice_runtime.ps1 first: "
            + ", ".join(str(path) for path in missing)
        )

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_cosyvoice_command(text, output, profile, speed)
    print(f"  [voice] CosyVoice energetic male -> {output.name}")
    result = run_checked(command, cwd=PROJECT_ROOT, timeout=1800)
    if result.stdout.strip():
        print(result.stdout.strip())
    info = validate_audio(output)
    print(
        f"  [voice] Done: {output} "
        f"({info['size'] / 1024:.0f}KB, {info['duration']:.1f}s)"
    )
    return output
