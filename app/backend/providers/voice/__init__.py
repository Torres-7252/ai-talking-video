#!/usr/bin/env python3
"""Local GPT-SoVITS voice generation provider."""

from __future__ import annotations

import gc
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from app.backend.providers.media_utils import validate_audio


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOVITS_ROOT = PROJECT_ROOT / "voice" / "models" / "GPT-SoVITS"
GPT_SOVITS_DIR = SOVITS_ROOT / "GPT_SoVITS"
ERES2NET_DIR = GPT_SOVITS_DIR / "eres2net"

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


def generate_voice(
    text: str,
    output_path: str,
    voice_profile: str = "default",
    speed: float = 1.0,
    api_url: Optional[str] = None,
) -> Path:
    del api_url
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Speech text cannot be empty")

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = load_voice_profile(voice_profile)
    tts = _get_tts()

    print(f"  [voice] Generating: {clean_text[:30]}... -> {output.name}")
    generated = _run_tts(
        tts,
        {
            "text": clean_text,
            "text_lang": "zh",
            "ref_audio_path": str(profile["reference_audio"]),
            "prompt_lang": profile["language"],
            "prompt_text": profile["reference_text"],
            "text_split_method": "cut5",
            "batch_size": 1,
            "media_type": "wav",
            "streaming_mode": False,
            "speed_factor": speed,
        },
    )
    if not generated:
        raise RuntimeError("GPT-SoVITS returned no audio")

    sample_rate = int(generated[0][0])
    chunks = [np.asarray(audio) for rate, audio in generated if int(rate) == sample_rate]
    if not chunks:
        raise RuntimeError("GPT-SoVITS returned incompatible audio chunks")
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
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
