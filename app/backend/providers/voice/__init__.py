#!/usr/bin/env python3
"""Local GPT-SoVITS voice generation provider."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from app.backend.providers.media_utils import validate_audio
from scripts.output_paths import OUTPUTS_ROOT
from . import cosyvoice


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOVITS_ROOT = PROJECT_ROOT / "voice" / "models" / "GPT-SoVITS"
GPT_SOVITS_DIR = SOVITS_ROOT / "GPT_SoVITS"
ERES2NET_DIR = GPT_SOVITS_DIR / "eres2net"
VOICE_CACHE_DIR = OUTPUTS_ROOT / ".cache" / "voice"
TTS_RETRY_SEEDS = (42, 2026, 1234)
RUNTIME_LIBRARY = Path(sys.executable).resolve().parents[1] / "Library" / "bin"
_DLL_DIRECTORY = None

if os.name == "nt" and RUNTIME_LIBRARY.is_dir():
    _DLL_DIRECTORY = os.add_dll_directory(str(RUNTIME_LIBRARY))
    os.environ["PATH"] = f"{RUNTIME_LIBRARY}{os.pathsep}{os.environ.get('PATH', '')}"

# GPT-SoVITS still uses top-level imports in several internal modules.
for import_path in (SOVITS_ROOT, GPT_SOVITS_DIR, ERES2NET_DIR):
    value = str(import_path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def build_tts_config() -> dict:
    pretrained = GPT_SOVITS_DIR / "pretrained_models"
    return {
        "custom": {
            "device": "cuda",
            "is_half": True,
            "version": "v3",
            "t2s_weights_path": str((pretrained / "s1v3.ckpt").resolve()),
            "vits_weights_path": str((pretrained / "s2Gv3.pth").resolve()),
            "bert_base_path": str(
                (pretrained / "chinese-roberta-wwm-ext-large").resolve()
            ),
            "cnhuhbert_base_path": str(
                (pretrained / "chinese-hubert-base-hf").resolve()
            ),
        }
    }


def load_voice_profile(name: str = "default") -> dict:
    references = PROJECT_ROOT / "voice" / "references"
    profile_path = references / f"{name}.json"
    if not profile_path.is_file():
        raise FileNotFoundError(f"Voice profile does not exist: {profile_path}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    reference_text = str(profile.get("reference_text", "")).strip()
    if not reference_text:
        raise RuntimeError(f"Voice profile has no reference_text: {profile_path}")
    return {
        "reference_audio": Path(_find_ref_audio(name)).resolve(),
        "reference_text": reference_text,
        "language": str(profile.get("language") or "zh"),
    }


def load_configured_voice_profile(name: str = "default") -> dict:
    config_path = PROJECT_ROOT / "config" / "profiles.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for profile in config.get("voices", []):
        if profile.get("id") == name:
            return dict(profile)
    raise ValueError(f"Unknown voice profile: {name}")


def _find_ref_audio(voice_profile: str = "default") -> str:
    references = PROJECT_ROOT / "voice" / "references"
    for extension in (".wav", ".mp3", ".m4a", ".flac"):
        path = references / f"{voice_profile}{extension}"
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        f"Reference audio does not exist: {references / voice_profile}.[wav|mp3|m4a|flac]"
    )


_tts_instance = None


def _import_tts_api():
    previous_cwd = Path.cwd()
    try:
        os.chdir(SOVITS_ROOT)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from TTS_infer_pack.TTS import TTS, TTS_Config
    finally:
        os.chdir(previous_cwd)
    return TTS, TTS_Config


def _create_tts():
    previous_cwd = Path.cwd()
    try:
        os.chdir(SOVITS_ROOT)
        TTS, TTS_Config = _import_tts_api()
        config = TTS_Config(build_tts_config())
        return TTS(config)
    finally:
        os.chdir(previous_cwd)


def _get_tts():
    global _tts_instance
    if _tts_instance is None:
        os.environ.setdefault(
            "LANGDETECT_CACHE",
            str(GPT_SOVITS_DIR / "pretrained_models" / "fast_langdetect"),
        )
        _tts_instance = _create_tts()
        print("  [voice] GPT-SoVITS models loaded")
    return _tts_instance


def _run_tts(tts, payload: dict) -> list:
    previous_cwd = Path.cwd()
    try:
        os.chdir(SOVITS_ROOT)
        return list(tts.run(payload))
    finally:
        os.chdir(previous_cwd)


def _prepare_reference_audio(
    source_path: Path,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Remove long edge silence so GPT-SoVITS receives aligned speech."""
    source = Path(source_path).resolve()
    audio, sample_rate = sf.read(source, always_2d=True)
    mono = np.mean(audio, axis=1)
    if not len(mono) or sample_rate <= 0:
        raise RuntimeError(f"Reference audio is empty: {source}")

    frame_size = max(1, int(sample_rate * 0.02))
    frame_rms = np.array(
        [
            np.sqrt(np.mean(mono[index : index + frame_size] ** 2) + 1e-12)
            for index in range(0, len(mono), frame_size)
        ]
    )
    threshold = max(0.003, float(frame_rms.max()) * 0.05)
    active_frames = np.flatnonzero(frame_rms >= threshold)
    if not len(active_frames):
        raise RuntimeError(f"Reference audio contains no detectable speech: {source}")

    start = max(0, int(active_frames[0] * frame_size - sample_rate * 0.15))
    end = min(
        len(mono),
        int((active_frames[-1] + 1) * frame_size + sample_rate * 0.20),
    )
    prepared_audio = mono[start:end]
    duration = len(prepared_audio) / sample_rate
    if duration < 3.0 or duration > 10.0:
        raise RuntimeError(
            "Reference speech must be between 3 and 10 seconds after silence removal; "
            f"got {duration:.1f}s"
        )

    destination_dir = Path(cache_dir or VOICE_CACHE_DIR).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    destination = destination_dir / f"{source.stem}-{signature}-trimmed.wav"
    if not destination.is_file():
        sf.write(destination, prepared_audio, sample_rate, subtype="PCM_16")
    return destination


def _reference_tokens(text: str) -> list[str]:
    return re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+", str(text).casefold())


def _find_reference_window(result: list, reference_text: str) -> tuple[float, float]:
    timed_characters: list[tuple[str, float, float]] = []
    for item in result or []:
        tokens = _reference_tokens(item.get("text", ""))
        timestamps = item.get("timestamp") or []
        if len(tokens) != len(timestamps):
            continue
        for token, timestamp in zip(tokens, timestamps):
            if not isinstance(timestamp, (list, tuple)) or len(timestamp) < 2:
                continue
            start = float(timestamp[0]) / 1000.0
            end = float(timestamp[1]) / 1000.0
            timed_characters.extend((character, start, end) for character in token)

    recognized = "".join(character for character, _, _ in timed_characters)
    target = "".join(_reference_tokens(reference_text))
    match_start = recognized.find(target)
    if not target or match_start < 0:
        raise RuntimeError(
            "Reference text was not found in the uploaded recording. "
            "Please provide the exact transcript of a clear 3-10 second clip."
        )
    match_end = match_start + len(target) - 1
    return timed_characters[match_start][1], timed_characters[match_end][2]


def align_reference_audio(audio_path: Path, reference_text: str) -> Path:
    """Crop an uploaded recording to the speech matching its transcript."""
    from funasr import AutoModel

    from app.backend.providers.subtitle import build_asr_options

    path = Path(audio_path).resolve()
    model = AutoModel(**build_asr_options())
    result = model.generate(input=str(path))
    speech_start, speech_end = _find_reference_window(result, reference_text)

    audio, sample_rate = sf.read(path, always_2d=True)
    start = max(0, int((speech_start - 0.15) * sample_rate))
    end = min(len(audio), int((speech_end + 0.20) * sample_rate))
    aligned = audio[start:end]
    duration = len(aligned) / sample_rate
    if duration < 3.0 or duration > 10.0:
        raise RuntimeError(
            "Matched reference speech must be between 3 and 10 seconds; "
            f"got {duration:.1f}s"
        )
    sf.write(path, aligned, sample_rate, subtype="PCM_16")
    return path


def _minimum_generated_duration(text: str, speed: float) -> float:
    if speed <= 0:
        raise ValueError("Speech speed must be positive")
    spoken_units = sum(
        1
        for character in text
        if character.isalnum() or "\u3400" <= character <= "\u9fff"
    )
    return max(0.55, spoken_units / (8.0 * speed))


def _normalize_tts_text(text: str) -> str:
    normalized = text.strip()
    if normalized and normalized[-1] not in "。！？!?；;，,.":
        normalized += "。"
    return normalized


def _split_tts_text(text: str) -> list[tuple[str, str]]:
    """Split Chinese narration from English terms without invoking auto detection."""
    has_chinese = bool(re.search(r"[\u3400-\u9fff]", text))
    has_english = bool(re.search(r"[A-Za-z]", text))
    if not has_chinese or not has_english:
        return [(text, "en" if has_english else "zh")]

    english_term = re.compile(
        r"(?=[A-Za-z0-9._-]*[A-Za-z])[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*"
    )
    parts: list[tuple[str, str]] = []
    cursor = 0
    for match in english_term.finditer(text):
        if match.start() > cursor:
            parts.append((text[cursor : match.start()], "zh"))
        parts.append((match.group(), "en"))
        cursor = match.end()
    if cursor < len(text):
        parts.append((text[cursor:], "zh"))

    normalized: list[tuple[str, str]] = []
    for part, language in parts:
        if not part.strip():
            continue
        if language == "zh" and not re.search(r"[\u3400-\u9fff]", part) and normalized:
            previous_text, previous_language = normalized[-1]
            normalized[-1] = (previous_text + part, previous_language)
        else:
            normalized.append((part, language))
    return normalized


def _join_generated_audio(generated: list) -> tuple[int, np.ndarray]:
    if not generated:
        raise RuntimeError("GPT-SoVITS returned no audio")
    sample_rate = int(generated[0][0])
    chunks = [
        np.asarray(audio)
        for rate, audio in generated
        if int(rate) == sample_rate and len(audio)
    ]
    if not chunks:
        raise RuntimeError("GPT-SoVITS returned incompatible audio chunks")
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    return sample_rate, audio


def generate_voice(
    text: str,
    output_path: str,
    voice_profile: str = "default",
    speed: float = 1.0,
    api_url: Optional[str] = None,
    reference_audio_path: Optional[str] = None,
    reference_text: Optional[str] = None,
) -> Path:
    del api_url
    clean_text = _normalize_tts_text(text)
    if not clean_text:
        raise ValueError("Speech text cannot be empty")

    configured_profile = load_configured_voice_profile(voice_profile)
    if not reference_audio_path and configured_profile.get("provider") == "cosyvoice":
        return cosyvoice.generate_cosyvoice(
            clean_text,
            output_path,
            configured_profile,
            speed,
        )

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if reference_audio_path:
        custom_reference = Path(reference_audio_path).resolve()
        if not custom_reference.is_file():
            raise FileNotFoundError(f"Reference audio does not exist: {custom_reference}")
        prompt_text = str(reference_text or "").strip()
        if not prompt_text:
            raise ValueError("Custom reference audio requires its transcript")
        profile = {
            "reference_audio": custom_reference,
            "reference_text": prompt_text,
            "language": "zh",
        }
    else:
        profile = load_voice_profile(voice_profile)
    reference_audio = _prepare_reference_audio(profile["reference_audio"])

    print(f"  [voice] Generating: {clean_text[:30]}... -> {output.name}")
    generated_segments: list[np.ndarray] = []
    sample_rate: Optional[int] = None
    for segment_text, text_lang in _split_tts_text(clean_text):
        tts = _get_tts()
        minimum_duration = _minimum_generated_duration(segment_text, speed)
        failure_details = []
        try:
            for attempt, seed in enumerate(TTS_RETRY_SEEDS, start=1):
                payload = {
                    "text": segment_text,
                    "text_lang": text_lang,
                    "ref_audio_path": str(reference_audio),
                    "prompt_lang": profile["language"],
                    "prompt_text": profile["reference_text"],
                    "text_split_method": "cut5",
                    "batch_size": 1,
                    "media_type": "wav",
                    "streaming_mode": False,
                    "speed_factor": speed,
                    "seed": seed,
                    "parallel_infer": False,
                }
                try:
                    current_rate, audio = _join_generated_audio(_run_tts(tts, payload))
                except RuntimeError as exc:
                    failure_details.append(str(exc))
                    continue
                duration = len(audio) / current_rate
                if duration >= minimum_duration:
                    break
                failure_details.append(
                    f"attempt {attempt} produced only {duration:.1f}s "
                    f"(minimum {minimum_duration:.1f}s)"
                )
                print(f"  [voice] Retrying: {failure_details[-1]}")
            else:
                raise RuntimeError(
                    f"GPT-SoVITS could not synthesize {text_lang} segment "
                    f"{segment_text!r}: {'; '.join(failure_details)}"
                )
        finally:
            unload_voice()
        if sample_rate is None:
            sample_rate = current_rate
        elif current_rate != sample_rate:
            raise RuntimeError("GPT-SoVITS returned inconsistent sample rates")
        generated_segments.append(audio)

    if sample_rate is None or not generated_segments:
        raise RuntimeError("GPT-SoVITS returned no audio")
    pause = np.zeros(int(sample_rate * 0.08), dtype=generated_segments[0].dtype)
    audio = np.concatenate(
        [chunk for pair in zip(generated_segments, [pause] * len(generated_segments)) for chunk in pair][:-1]
    )

    sf.write(str(output), audio, sample_rate)
    info = validate_audio(output)
    print(
        f"  [voice] Done: {output} "
        f"({info['size'] / 1024:.0f}KB, {info['duration']:.1f}s)"
    )
    return output


def unload_voice() -> None:
    global _tts_instance
    if _tts_instance is None:
        return
    del _tts_instance
    _tts_instance = None
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  [voice] Models unloaded")
