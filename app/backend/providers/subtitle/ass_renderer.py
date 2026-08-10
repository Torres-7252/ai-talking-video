"""Animated ASS caption presets with deterministic active-word timing."""

from __future__ import annotations

import re
from pathlib import Path


CAPTION_PRESETS = frozenset({"clean", "pop", "bar"})
_STYLE_NAMES = {"clean": "Clean", "pop": "Pop", "bar": "Bar"}
_ACTIVE_COLOR = "&H0047D4FF&"


def _seconds_to_ass_time(seconds: float) -> str:
    centiseconds = max(0, round(float(seconds) * 100))
    hours, centiseconds = divmod(centiseconds, 360_000)
    minutes, centiseconds = divmod(centiseconds, 6_000)
    whole_seconds, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def _ass_escape(text: object) -> str:
    return (
        str(text)
        .replace("{", "(")
        .replace("}", ")")
        .replace("\r\n", r"\N")
        .replace("\r", r"\N")
        .replace("\n", r"\N")
    )


def wrap_caption(text: str, max_chars: int = 18) -> str:
    """Wrap a caption once, preferring punctuation near the midpoint."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars or max_chars < 2:
        return cleaned

    target = min(max_chars, max(1, len(cleaned) // 2))
    candidates = [
        index + 1
        for index, character in enumerate(cleaned[:-1])
        if character in "，。！？；：,.!?;: "
    ]
    nearby = [index for index in candidates if abs(index - target) <= max_chars // 2]
    split_at = min(nearby, key=lambda index: abs(index - target)) if nearby else target
    return cleaned[:split_at].rstrip() + r"\N" + cleaned[split_at:].lstrip()


def _active_phrase(text: str, words: list[dict], active_index: int) -> str:
    wrapped = wrap_caption(text)
    cursor = 0
    start = -1
    end = -1
    for index, word in enumerate(words):
        token = str(word.get("text") or "")
        if not token:
            continue
        found = wrapped.find(token, cursor)
        if found < 0:
            found = wrapped.find(token)
        if found < 0:
            continue
        cursor = found + len(token)
        if index == active_index:
            start, end = found, cursor
            break

    if start < 0:
        return _ass_escape(wrapped)
    before = _ass_escape(wrapped[:start])
    active = _ass_escape(wrapped[start:end])
    after = _ass_escape(wrapped[end:])
    return f"{before}{{\\c{_ACTIVE_COLOR}\\3c&H00383838&}}{active}{{\\r}}{after}"


def _header(width: int, height: int) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Clean,Microsoft YaHei,54,&H00FFFFFF,&H00FFFFFF,&H00141414,&H50000000,-1,0,0,0,100,100,0,0,1,3,1,2,150,150,78,1
Style: Pop,Microsoft YaHei,58,&H00FFFFFF,&H00FFFFFF,&H001C1C1C,&H50000000,-1,0,0,0,100,100,0,0,1,4,1,2,150,150,82,1
Style: Bar,Microsoft YaHei,48,&H00FFFFFF,&H00FFFFFF,&H00101010,&H98000000,-1,0,0,0,100,100,0,0,3,1,0,2,135,135,62,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""


def write_animated_ass(
    subtitles: list[dict],
    output_path: Path,
    preset: str = "clean",
    width: int = 1920,
    height: int = 1080,
) -> Path:
    """Serialize base captions and timed active-word overlays as ASS."""
    if preset not in CAPTION_PRESETS:
        raise ValueError(
            f"Unknown caption preset: {preset!r}; expected one of "
            f"{', '.join(sorted(CAPTION_PRESETS))}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    style = _STYLE_NAMES[preset]
    lines: list[str] = []

    for segment in subtitles:
        start = max(0.0, float(segment.get("start", 0.0)))
        end = max(start, float(segment.get("end", start)))
        if end <= start:
            continue
        phrase = wrap_caption(str(segment.get("text") or ""))
        if not phrase:
            continue

        base_prefix = r"{\fad(120,100)\move(960,1024,960,1008,0,160)}"
        if preset == "pop":
            base_prefix = r"{\fad(90,90)\fscx104\fscy104\t(0,150,\fscx100\fscy100)}"
        elif preset == "bar":
            base_prefix = r"{\fad(100,100)\move(930,1010,960,1010,0,180)}| "

        lines.append(
            f"Dialogue: 0,{_seconds_to_ass_time(start)},"
            f"{_seconds_to_ass_time(end)},{style},,0,0,0,,"
            f"{base_prefix}{_ass_escape(phrase)}"
        )

        words = segment.get("words") or []
        for index, word in enumerate(words):
            word_start = max(start, float(word.get("start", start)))
            word_end = min(end, float(word.get("end", end)))
            if word_end <= word_start:
                continue
            active_prefix = ""
            if preset == "pop":
                active_prefix = r"{\fscx108\fscy108\t(0,120,\fscx100\fscy100)}"
            elif preset == "bar":
                active_prefix = "| "
            lines.append(
                f"Dialogue: 1,{_seconds_to_ass_time(word_start)},"
                f"{_seconds_to_ass_time(word_end)},{style},,0,0,0,,"
                f"{active_prefix}{_active_phrase(str(segment.get('text') or ''), words, index)}"
            )

    output.write_text(_header(width, height) + "\n".join(lines) + "\n", encoding="utf-8-sig")
    return output
