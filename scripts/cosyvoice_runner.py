#!/usr/bin/env python3
"""Run CosyVoice inference inside its isolated Python environment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COSYVOICE_ROOT = PROJECT_ROOT / "voice" / "models" / "CosyVoice"
RUNTIME_LIBRARY = Path(sys.executable).resolve().parents[1] / "Library" / "bin"
_DLL_DIRECTORY = None
if os.name == "nt" and RUNTIME_LIBRARY.is_dir():
    _DLL_DIRECTORY = os.add_dll_directory(str(RUNTIME_LIBRARY))
    os.environ["PATH"] = f"{RUNTIME_LIBRARY}{os.pathsep}{os.environ.get('PATH', '')}"

import torch
import torchaudio


sys.path.insert(0, str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS"))
sys.path.insert(0, str(COSYVOICE_ROOT))

from cosyvoice.cli.cosyvoice import AutoModel


RETRY_SEEDS = (42, 2026, 1234)


def _minimum_generated_duration(text: str, speed: float) -> float:
    spoken_units = sum(
        1
        for character in text
        if character.isalnum() or "\u3400" <= character <= "\u9fff"
    )
    return max(0.55, spoken_units / (8.0 * speed))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate speech with CosyVoice")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speaker", default="中文男")
    parser.add_argument("--mode", choices=("sft", "instruct"), default="instruct")
    parser.add_argument("--instruct", required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    print("[cosyvoice] CUDA available:", torch.cuda.is_available(), flush=True)
    print("[cosyvoice] Loading model...", flush=True)
    model = AutoModel(
        model_dir=str(args.model_dir.resolve()),
        # FP16 inference can emit an all-NaN waveform on this Windows CUDA stack.
        # The 300M model still fits comfortably on the local 8GB GPU in FP32.
        fp16=False,
    )
    print("[cosyvoice] Model loaded.", flush=True)
    if args.speaker not in model.list_available_spks():
        raise RuntimeError(
            f"CosyVoice speaker {args.speaker!r} is unavailable; "
            f"available: {model.list_available_spks()}"
        )
    chunks = []
    minimum_duration = _minimum_generated_duration(args.text, args.speed)
    for seed in RETRY_SEEDS:
        print(f"[cosyvoice] Starting inference (seed={seed})...", flush=True)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if args.mode == "sft":
            generated = model.inference_sft(
                args.text,
                args.speaker,
                stream=False,
                speed=args.speed,
                text_frontend=False,
            )
        else:
            generated = model.inference_instruct(
                args.text,
                args.speaker,
                args.instruct,
                stream=False,
                speed=args.speed,
                text_frontend=False,
            )
        chunks = [item["tts_speech"].float().cpu() for item in generated]
        if chunks and not all(torch.isfinite(chunk).all().item() for chunk in chunks):
            print(
                f"[cosyvoice] Non-finite audio from seed={seed}; retrying.",
                flush=True,
            )
            chunks = []
            continue
        generated_duration = (
            sum(chunk.shape[1] for chunk in chunks) / model.sample_rate
            if chunks
            else 0.0
        )
        print(
            f"CosyVoice attempt seed={seed}: {generated_duration:.2f}s "
            f"(minimum {minimum_duration:.2f}s)",
            flush=True,
        )
        if generated_duration >= minimum_duration:
            break
        chunks = []
    if not chunks:
        raise RuntimeError(
            "CosyVoice returned no finite complete audio after "
            f"{len(RETRY_SEEDS)} attempts (minimum {minimum_duration:.2f}s)"
        )
    audio = torch.cat(chunks, dim=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("[cosyvoice] Saving audio...", flush=True)
    torchaudio.save(str(args.output.resolve()), audio, model.sample_rate)
    print(f"CosyVoice generated {audio.shape[1] / model.sample_rate:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
