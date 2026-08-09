"""Local FunASR subtitle generation and ASS serialization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from app.backend.providers.media_utils import validate_audio


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ASR_MODEL = (
    "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)
DEFAULT_VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"


def build_asr_options(
    model_name: Optional[str] = None,
    device: str = "cuda",
) -> dict:
    """Return current FunASR aliases and deterministic local options."""
    normalized_device = "cuda:0" if device == "cuda" else device
    return {
        "model": model_name or DEFAULT_ASR_MODEL,
        "vad_model": DEFAULT_VAD_MODEL,
        "device": normalized_device,
        "disable_update": True,
    }


def _seconds_from_ms(value: object) -> float:
    try:
        return max(0.0, float(value) / 1000.0)
    except (TypeError, ValueError):
        return 0.0


def _split_caption_text(text: str, max_chars: int = 18) -> list[str]:
    phrases = re.findall(r".+?[。！？!?；;，,]|.+$", text)
    chunks: list[str] = []
    for phrase in phrases:
        phrase = phrase.strip()
        while len(phrase) > max_chars:
            chunks.append(phrase[:max_chars])
            phrase = phrase[max_chars:]
        if phrase:
            chunks.append(phrase)
    return chunks


def _clean_asr_text(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    cjk = r"\u3400-\u9fff"
    text = re.sub(rf"(?<=[{cjk}])\s+(?=[{cjk}A-Za-z0-9])", "", text)
    text = re.sub(rf"(?<=[A-Za-z0-9])\s+(?=[{cjk}])", "", text)
    text = re.sub(
        r"(?<![A-Za-z])(?:[A-Za-z]\s+)+[A-Za-z](?![A-Za-z])",
        lambda match: match.group(0).replace(" ", ""),
        text,
    )
    return text


def _timed_text_segments(text: str, start: float, end: float) -> list[dict]:
    chunks = _split_caption_text(text)
    total_chars = sum(len(chunk) for chunk in chunks)
    if not chunks or total_chars == 0 or end <= start:
        return []

    segments: list[dict] = []
    offset = 0
    for chunk in chunks:
        chunk_start = start + (end - start) * offset / total_chars
        offset += len(chunk)
        chunk_end = start + (end - start) * offset / total_chars
        segments.append(
            {
                "text": chunk,
                "start": round(chunk_start, 3),
                "end": round(chunk_end, 3),
                "words": [],
            }
        )
    return segments


def normalize_asr_result(result: list, audio_duration: float) -> list[dict]:
    """Normalize FunASR sentence-level or top-level timestamps."""
    if audio_duration <= 0:
        raise RuntimeError("Audio duration must be positive")

    subtitles: list[dict] = []
    for item in result or []:
        sentence_info = item.get("sentence_info") or []
        if sentence_info:
            for sentence in sentence_info:
                text = _clean_asr_text(sentence.get("text"))
                start = min(_seconds_from_ms(sentence.get("start")), audio_duration)
                end = min(_seconds_from_ms(sentence.get("end")), audio_duration)
                if not text or end <= start:
                    continue
                words = []
                for word in sentence.get("words") or []:
                    word_start = min(_seconds_from_ms(word.get("start")), audio_duration)
                    word_end = min(_seconds_from_ms(word.get("end")), audio_duration)
                    if word_end > word_start:
                        words.append(
                            {
                                "text": str(word.get("text") or ""),
                                "start": round(word_start, 3),
                                "end": round(word_end, 3),
                            }
                        )
                subtitles.append(
                    {
                        "text": text,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "words": words,
                    }
                )
            continue

        text = _clean_asr_text(item.get("text"))
        timestamps = [
            timestamp
            for timestamp in (item.get("timestamp") or [])
            if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2
        ]
        if not text:
            continue
        if timestamps:
            start = min(_seconds_from_ms(timestamps[0][0]), audio_duration)
            end = min(_seconds_from_ms(timestamps[-1][1]), audio_duration)
        else:
            start, end = 0.0, audio_duration
        subtitles.extend(_timed_text_segments(text, start, end))

    subtitles.sort(key=lambda segment: (segment["start"], segment["end"]))
    if not subtitles:
        raise RuntimeError("FunASR returned no usable timed subtitles")
    return subtitles


def apply_transcript_text(subtitles: list[dict], transcript_text: str) -> list[dict]:
    """Keep ASR timing while using the known script as caption text."""
    text = _clean_asr_text(transcript_text)
    if not text or not subtitles:
        return subtitles
    return _timed_text_segments(text, subtitles[0]["start"], subtitles[-1]["end"])


def _seconds_to_srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def _seconds_to_ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, centiseconds = divmod(centiseconds, 360_000)
    minutes, centiseconds = divmod(centiseconds, 6_000)
    whole_seconds, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def write_srt(subtitles: list[dict], output_path: Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, segment in enumerate(subtitles, 1):
        lines.extend(
            [
                str(index),
                f"{_seconds_to_srt_time(segment['start'])} --> "
                f"{_seconds_to_srt_time(segment['end'])}",
                str(segment["text"]),
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8-sig")
    return output


def write_ass(
    subtitles: list[dict],
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    """Write subtitles using a landscape-safe ASS style."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Microsoft YaHei,52,&H00FFFFFF,&H000000FF,&H00101010,&H78000000,-1,0,0,0,100,100,0,0,1,3,1,2,120,120,72,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    dialogue = []
    for segment in subtitles:
        text = str(segment["text"]).replace("{", "（").replace("}", "）")
        text = text.replace("\r\n", r"\N").replace("\n", r"\N")
        dialogue.append(
            f"Dialogue: 0,{_seconds_to_ass_time(segment['start'])},"
            f"{_seconds_to_ass_time(segment['end'])},Default,,0,0,0,,{text}"
        )
    output.write_text(header + "\n".join(dialogue) + "\n", encoding="utf-8-sig")
    return output


def generate_subtitles(
    audio_path: str,
    output_json: str,
    output_srt: Optional[str] = None,
    model_name: Optional[str] = None,
    device: str = "cuda",
    transcript_text: Optional[str] = None,
) -> list[dict]:
    """Recognize speech locally and write JSON, SRT, and ASS subtitles."""
    from funasr import AutoModel

    audio = Path(audio_path).resolve()
    audio_info = validate_audio(audio)
    json_path = Path(output_json).resolve()
    srt_path = Path(output_srt).resolve() if output_srt else json_path.with_suffix(".srt")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    print("  [subtitle] Running local FunASR recognition...")
    model = AutoModel(**build_asr_options(model_name=model_name, device=device))
    result = model.generate(input=str(audio))
    subtitles = normalize_asr_result(result, audio_info["duration"])
    if transcript_text:
        subtitles = apply_transcript_text(subtitles, transcript_text)

    json_path.write_text(
        json.dumps(subtitles, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_srt(subtitles, srt_path)
    write_ass(subtitles, json_path.with_suffix(".ass"))
    print(f"  [subtitle] Generated {len(subtitles)} segment(s): {json_path}")
    return subtitles


if __name__ == "__main__":
    generate_subtitles(
        str(PROJECT_ROOT / "outputs" / "acceptance" / "audio.wav"),
        str(PROJECT_ROOT / "outputs" / "acceptance" / "subtitle.json"),
    )
