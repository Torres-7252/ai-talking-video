"""Isolated CosyVoice provider for selectable preset voices."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.backend.providers.media_utils import validate_audio


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
COSYVOICE_TIMEOUT_SECONDS = 10 * 60


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


def _log_tail(log_path: Path, limit: int = 4000) -> str:
    if not log_path.is_file():
        return "No CosyVoice log was written."
    return log_path.read_text(encoding="utf-8", errors="replace")[-limit:].strip()


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        process.kill()


def run_cosyvoice(
    command: list[str],
    *,
    log_path: Path,
    timeout_seconds: int = COSYVOICE_TIMEOUT_SECONDS,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_tree(process)
            raise RuntimeError(
                f"CosyVoice timed out after {timeout_seconds // 60} minutes. "
                f"See {log_path.name}: {_log_tail(log_path)}"
            ) from exc
    if return_code:
        raise RuntimeError(
            f"CosyVoice exited with code {return_code}. See {log_path.name}: "
            f"{_log_tail(log_path)}"
        )


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
    log_path = output.with_name("cosyvoice.log")
    print(f"  [voice] CosyVoice energetic male -> {output.name}")
    run_cosyvoice(command, log_path=log_path)
    info = validate_audio(output)
    print(
        f"  [voice] Done: {output} "
        f"({info['size'] / 1024:.0f}KB, {info['duration']:.1f}s)"
    )
    return output
