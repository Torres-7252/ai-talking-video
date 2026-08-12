"""Sentence-safe splitting for long talking-video scripts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def visible_character_count(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _split_units(text: str, delimiters: str) -> list[str]:
    parts = re.split(f"([{re.escape(delimiters)}])", text)
    return [
        (parts[index] + (parts[index + 1] if index + 1 < len(parts) else "")).strip()
        for index in range(0, len(parts), 2)
        if parts[index].strip()
    ]


def _hard_wrap(text: str, max_characters: int) -> list[str]:
    return [text[index : index + max_characters] for index in range(0, len(text), max_characters)]


def split_script_into_segments(script: str, max_characters: int) -> list[str]:
    """Split at sentence boundaries, using commas only for long sentences."""
    normalized = re.sub(r"\s+", "", str(script))
    if not normalized:
        return []
    if max_characters < 1:
        raise ValueError("max_characters must be positive")

    segments: list[str] = []
    current = ""
    for sentence in _split_units(normalized, "。！？!?；;"):
        sentence_units = [sentence]
        if visible_character_count(sentence) > max_characters:
            sentence_units = _split_units(sentence, "，,、：:")

        for unit in sentence_units:
            if visible_character_count(unit) > max_characters:
                if current:
                    segments.append(current)
                    current = ""
                segments.extend(_hard_wrap(unit, max_characters))
                continue
            if current and visible_character_count(current + unit) > max_characters:
                segments.append(current)
                current = unit
            else:
                current += unit
    if current:
        segments.append(current)
    return segments


def concatenate_videos(parts: list[Path], output: Path) -> Path:
    if not parts:
        raise ValueError("No video segments to concatenate")
    manifest = output.with_suffix(".concat.txt")
    manifest.write_text("".join(f"file '{part.resolve().as_posix()}'\n" for part in parts), encoding="utf-8")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", str(output)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "FFmpeg concatenation failed")
    finally:
        manifest.unlink(missing_ok=True)
    return output
